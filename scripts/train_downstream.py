import os
from datetime import datetime
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

from selbstaufsicht import models
from selbstaufsicht import datasets
from selbstaufsicht.models.self_supervised.msa.utils import get_downstream_transforms, MSACollator, get_tasks, get_downstream_metrics
from selbstaufsicht.utils import data_loader_worker_init


def main():
    parser = argparse.ArgumentParser(description='Selbstaufsicht Weakly Supervised Contact Prediction Script')
    # Pre-trained model
    parser.add_argument('--checkpoint', type=str, help="Path to pre-trained model checkpoint")
    # Training process
    parser.add_argument('--num-epochs', default=1000, type=int, help="Number of training epochs")
    parser.add_argument('--batch-size', default=1, type=int, help="Batch size (local in case of multi-gpu training)")
    parser.add_argument('--learning-rate', default=1e-4, type=float, help="Initial learning rate")
    parser.add_argument('--learning-rate-warmup', default=1000, type=int, help="Warmup parameter for inverse square root rule of learning rate scheduling")
    parser.add_argument('--precision', default=32, type=int, help="Precision used for computations")
    parser.add_argument('--disable-progress-bar', action='store_true', help="disables the training progress bar")
    parser.add_argument('--disable-shuffle', action='store_true', help="disables the dataset shuffling")
    parser.add_argument('--rng-seed', default=42, type=int, help="Random number generator seed")
    # Data parallelism
    parser.add_argument('--num-gpus', default=-1, type=int, help="Number of GPUs per node. -1 refers to using all available GPUs. 0 refers to using the CPU.")
    parser.add_argument('--num-nodes', default=1, type=int, help="Number of nodes")
    # Logging
    parser.add_argument('--log-every', default=5, type=int, help='how often to add logging rows(does not write to disk)')
    parser.add_argument('--log-dir', default='', type=str, help='Logging directory. If empty, the directory of the pre-trained model is used. Default: \"\"')
    parser.add_argument('--log-exp-name', default='', type=str, help='Logging experiment name. If empty, the experiment name of the pre-trained model is used. Default: \"\"')
    parser.add_argument('--log-run-name', default='', type=str, help='Logging run name. Supports 1989 C standard datetime codes. If empty, the run name of the pre-trained model is used, prefixed by \"downstream__\". Default: \"\"')

    args = parser.parse_args()
    
    torch.manual_seed(args.rng_seed)
    np.random.seed(args.rng_seed)
    random.seed(args.rng_seed)
    data_loader_rng = torch.Generator()
    data_loader_rng.manual_seed(args.rng_seed)
    
    num_gpus = args.num_gpus if args.num_gpus >= 0 else torch.cuda.device_count()
    if num_gpus * args.num_nodes > 1:
        dp_strategy = DDPPlugin(find_unused_parameters=True)
        # NOTE for some reason, load_from_checkpoint fails to infer the hyperparameters correctly from the checkpoint file
        checkpoint = torch.load(args.checkpoint)
    else:
        dp_strategy = None
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    h_params = checkpoint['hyper_parameters']
    
    root = os.environ['DATA_PATH']
    downstream_transform = get_downstream_transforms(subsample_depth=h_params['subsampling_depth'])
    train_metrics, val_metrics = get_downstream_metrics()
    downstream_ds = datasets.CoCoNetDataset(root, 'train', transform=downstream_transform)
    test_ds = datasets.CoCoNetDataset(root, 'val', transform=downstream_transform)
    
    dt_now = datetime.now()
    log_dir = h_params['log_dir'] if args.log_dir == "" else args.log_dir
    log_exp_name = h_params['log_exp_name'] if args.log_exp_name == "" else args.log_exp_name
    log_run_name = "downstream__" + h_params['log_run_name'] if args.log_run_name == "" else dt_now.strftime(args.log_run_name)

    if h_params['jigsaw_euclid_emb']:
        embed_size = checkpoint['state_dict']['task_heads.jigsaw.proj.weight'].size(0)
        jigsaw_euclid_emb = torch.empty((1, embed_size))
    else:
        jigsaw_euclid_emb = None

    if 'jigsaw_disable_delimiter' in h_params:
        jigsaw_delimiter = not h_params['jigsaw_disable_delimiter']
    else:
        jigsaw_delimiter = True

    tasks = []
    if h_params['task_inpainting']:
        tasks.append("inpainting")
    if h_params['task_jigsaw']:
        tasks.append("jigsaw")
    if h_params['task_contrastive']:
        tasks.append("contrastive")
    _, task_heads, task_losses, _, _ = get_tasks(tasks,
                                       h_params['feature_dim_head'] * h_params['num_heads'],
                                       subsample_depth=h_params['subsampling_depth'],
                                       subsample_mode=h_params['subsampling_mode'],
                                       crop_size=h_params['cropping_size'],
                                       crop_mode=h_params['cropping_mode'],
                                       masking=h_params['inpainting_masking_type'],
                                       p_mask=h_params['inpainting_masking_p'],
                                       jigsaw_partitions=h_params['jigsaw_partitions'],
                                       jigsaw_classes=h_params['jigsaw_permutations'],
                                       jigsaw_linear=not h_params['jigsaw_nonlinear'],
                                       jigsaw_delimiter= jigsaw_delimiter,
                                       jigsaw_euclid_emb=jigsaw_euclid_emb,
                                       simclr_temperature=h_params['contrastive_temperature'])

    model = models.self_supervised.MSAModel.load_from_checkpoint(
            checkpoint_path = args.checkpoint,
            num_blocks = h_params['num_blocks'],
            num_heads = h_params['num_heads'],
            feature_dim_head = h_params['feature_dim_head'],
            task_heads=task_heads,
            task_losses=task_losses,
            alphabet_size=len(downstream_ds.token_mapping),
            padding_token=downstream_ds.token_mapping['PADDING_TOKEN'],
            lr=args.learning_rate,
            lr_warmup=args.learning_rate_warmup,
            dropout=0.,
            emb_grad_freq_scale=not h_params['disable_emb_grad_freq_scale'],
            )
    model.tasks = ['contact']
    # TODO there probably should be a weight for contacts
    model.losses['contact'] = nn.NLLLoss(ignore_index=-1)
    model.task_heads['contact'] = models.self_supervised.msa.modules.ContactHead(h_params['num_blocks'] * h_params['num_heads'], cull_tokens=[downstream_ds.token_mapping[l] for l in ['-', '.', 'START_TOKEN', 'DELIMITER_TOKEN']])
    model.need_attn = True
    model.task_loss_weights = {'contact': 1.}
    model.train_metrics = train_metrics
    model.val_metrics = val_metrics

    train_dl = DataLoader(downstream_ds, 
                          batch_size=args.batch_size,
                          shuffle=not args.disable_shuffle,
                          num_workers=0,
                          worker_init_fn=partial(data_loader_worker_init, rng_seed=args.rng_seed),
                          generator=data_loader_rng, 
                          pin_memory=num_gpus > 0)
    test_dl = DataLoader(test_ds, 
                         batch_size=args.batch_size,
                         shuffle=False,
                         num_workers=0,
                         worker_init_fn=partial(data_loader_worker_init, rng_seed=args.rng_seed),
                         generator=data_loader_rng, 
                         pin_memory=num_gpus > 0)

    tb_logger = TensorBoardLogger(save_dir=log_dir, name=log_exp_name, version=log_run_name)
    trainer = Trainer(max_epochs=args.num_epochs, 
                      gpus=args.num_gpus, 
                      auto_select_gpus=num_gpus > 0,
                      num_nodes=args.num_nodes,
                      precision=args.precision,
                      strategy=dp_strategy,
                      enable_progress_bar=not args.disable_progress_bar, 
                      log_every_n_steps=args.log_every, 
                      logger=tb_logger)
    trainer.fit(model, train_dl, test_dl)

if __name__ == '__main__':
    main()
