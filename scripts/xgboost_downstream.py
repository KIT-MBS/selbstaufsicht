import argparse
from datetime import datetime
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from selbstaufsicht import models
from selbstaufsicht import datasets
from selbstaufsicht.models.self_supervised.msa.utils import get_downstream_transforms, get_tasks
from selbstaufsicht.utils import data_loader_worker_init


def main():
    parser = argparse.ArgumentParser(description='Selbstaufsicht Weakly Supervised Contact Prediction Script (XGBoost version)')
    # Pre-trained model
    parser.add_argument('--checkpoint', type=str, help="Path to pre-trained model checkpoint")
    # Contact prediction
    parser.add_argument('--distance-threshold', default=10., type=float, help="Minimum distance between two atoms in angström that is not considered as a contact")
    # Preprocessing
    parser.add_argument('--subsampling-mode', default='uniform', type=str, help="Subsampling mode: uniform, diversity, fixed")
    # Training process
    parser.add_argument('--batch-size', default=1, type=int, help="Batch size (attention-map computation)")
    parser.add_argument('--rng-seed', default=42, type=int, help="Random number generator seed")
    parser.add_argument('--validation-ratio', default=0.1, type=float, help="Ratio of the validation dataset w.r.t. the full training dataset, if k-fold cross validation is disabled.")
    parser.add_argument('--cv-num-folds', default=1, type=int, help="Number of folds in k-fold cross validation. If 1, then cross validation is disabled.")
    parser.add_argument('--disable-train-data-discarding', action='store_true', help="disables the size-based discarding of training data")
    # GPU
    parser.add_argument('--no-gpu', action='store_true', help="disables cuda")
    # Logging
    parser.add_argument('--log-dir', default='', type=str, help='Logging directory. If empty, the directory of the pre-trained model is used. Default: \"\"')
    parser.add_argument('--log-exp-name', default='', type=str, help='Logging experiment name. If empty, the experiment name of the pre-trained model is used. Default: \"\"')
    parser.add_argument('--log-run-name', default='', type=str, help='Logging run name. Supports 1989 C standard datetime codes. If empty, the run name of the pre-trained model is used, prefixed by \"downstream__\". Default: \"\"')

    args = parser.parse_args()

    torch.manual_seed(args.rng_seed)
    np.random.seed(args.rng_seed)
    random.seed(args.rng_seed)
    data_loader_rng = torch.Generator()
    data_loader_rng.manual_seed(args.rng_seed)
    
    if not args.no_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    checkpoint = torch.load(args.checkpoint, map_location=device)
    h_params = checkpoint['hyper_parameters']
    downstream_args = {k: v for k, v in vars(args).items()}
    
    root = os.environ['DATA_PATH']
    downstream_transform = get_downstream_transforms(subsample_depth=h_params['subsampling_depth'], subsample_mode=args.subsampling_mode, threshold=args.distance_threshold)
    train_dataset = datasets.CoCoNetDataset(root, 'train', transform=downstream_transform, discard_train_size_based=not args.disable_train_data_discarding, 
                                                  diversity_maximization=args.subsampling_mode=='diversity', max_seq_len=h_params['cropping_size'], 
                                                  min_num_seq=h_params['subsampling_depth'])
    train_dl = DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          shuffle=False,
                          num_workers=0,
                          worker_init_fn=partial(data_loader_worker_init, rng_seed=args.rng_seed),
                          generator=data_loader_rng,
                          pin_memory=False)

    dt_now = datetime.now()
    log_dir = h_params['log_dir'] if args.log_dir == "" else args.log_dir
    log_exp_name = h_params['log_exp_name'] if args.log_exp_name == "" else args.log_exp_name
    log_run_name = "downstream__" + h_params['log_run_name'] if args.log_run_name == "" else dt_now.strftime(args.log_run_name)
    if args.cv_num_folds >= 2:
        log_exp_name = log_run_name
    h_params['downstream__log_dir'] = log_dir
    h_params['downstream__log_exp_name'] = log_exp_name
    
    jigsaw_euclid_emb = None
    if 'jigsaw_euclid_emb' in h_params and h_params['jigsaw_euclid_emb']:
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
        checkpoint_path=args.checkpoint,
        num_blocks=h_params['num_blocks'],
        num_heads=h_params['num_heads'],
        feature_dim_head=h_params['feature_dim_head'],
        task_heads=task_heads,
        task_losses=task_losses,
        alphabet_size=len(train_dataset.token_mapping),
        padding_token=train_dataset.token_mapping['PADDING_TOKEN'],
        lr=h_params['learning_rate'],
        lr_warmup=h_params['learning_rate_warmup'],
        dropout=0.,
        emb_grad_freq_scale=not h_params['disable_emb_grad_freq_scale'],
        freeze_backbone=True,
        h_params=h_params)
    model.need_attn = True
    model.to(device)
    
    num_maps = h_params['num_blocks'] * h_params['num_heads']
    cull_tokens = [train_dataset.token_mapping[token] for token in ['-', '.', 'START_TOKEN', 'DELIMITER_TOKEN']]
    
    attn_maps_list = []
    targets_list = []
    
    for x, y in train_dl:
        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                x[k] = v.to(device)
        
        with torch.no_grad():
            _, attn_maps = model(x['msa'], x.get('padding_mask', None), x.get('aux_features', None))
        
        B, _, L = x['msa'].shape
        assert B == 1
        
        mask = torch.ones((L, ), device=device)
        for token in cull_tokens:
            mask -= (x['msa'][:, 0].reshape((L, )) == token).int()
        mask = mask.bool()
        degapped_L = int(mask.sum())
        mask = torch.reshape(mask, (B, 1, L))
        mask = torch.logical_and(mask.reshape((B, 1, L)).expand(B, L, L), mask.reshape((B, L, 1)).expand(B, L, L))
        mask = mask.reshape((B, 1, L, L)).expand((B, num_maps, L, L))

        attn_maps = torch.cat([m.squeeze(dim=2) for m in attn_maps], dim=1)  # [B, num_maps, L, L]
        attn_maps = attn_maps.masked_select(mask).reshape(B, num_maps, degapped_L, degapped_L)
        attn_maps = torch.permute(attn_maps, (0, 2, 3, 1))  # [B, L, L, num_maps]
        
        assert num_maps == attn_maps.shape[-1]
        
        attn_maps = attn_maps.view(-1, num_maps)  # [B*L*L, num_maps]
        attn_maps_list.append(attn_maps)
        
        targets_list.append(y['contact'].view(-1))  # [B*L*L]
    
    attn_maps = torch.cat(attn_maps_list)  # [B*L*L, num_maps]
    targets = torch.cat(targets_list)  # [B*L*L]
    
    attn_maps = attn_maps[targets != -1, :]
    targets = targets[targets != -1]
    
    attn_maps = attn_maps.cpu().numpy()
    targets = targets.cpu().numpy()
    
    print(attn_maps.shape)
        
    # for fold_idx in range(args.cv_num_folds):
    #     if args.cv_num_folds >= 2:
    #         log_run_name = 'fold_%d' % (fold_idx + 1)
    #     h_params['downstream__log_run_name'] = log_run_name
        
        
        
    #     kfold_cv_downstream.setup_fold_index(fold_idx)

    #     train_dl = kfold_cv_downstream.train_dataloader()
    #     val_dl = kfold_cv_downstream.val_dataloader()
        
    #     trainer = Trainer(max_epochs=args.num_epochs,
    #                       gpus=args.num_gpus,
    #                       auto_select_gpus=num_gpus > 0,
    #                       num_nodes=args.num_nodes,
    #                       precision=args.precision,
    #                       strategy=dp_strategy,
    #                       enable_progress_bar=not args.disable_progress_bar,
    #                       log_every_n_steps=args.log_every,
    #                       logger=tb_logger,
    #                       callbacks=[checkpoint_callback])
    #     trainer.fit(model, train_dl, val_dl)

    # if args.test:
    #     trainer = Trainer(gpus=1 if num_gpus > 0 else 0,
    #                       logger=tb_logger,
    #                       enable_progress_bar=not args.disable_progress_bar)
    #     checkpoint_path = log_dir
    #     if log_exp_name != "":
    #         checkpoint_path += '%s/' % log_exp_name
    #     checkpoint_path += '%s/checkpoints/' % log_run_name
        
    #     # seaching for the latest file is a little bit hacky, but should work
    #     checkpoint_list = glob.glob('%s*.ckpt' % checkpoint_path)
    #     latest_checkpoint = max(checkpoint_list, key=os.path.getctime)
        
    #     model.downstream_loss_device_flag = False

    #     trainer.test(model, test_dl, ckpt_path=latest_checkpoint, verbose=True)
    
if __name__ == '__main__':
    main()