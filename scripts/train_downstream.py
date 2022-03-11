import argparse
from datetime import datetime
from functools import partial
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
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
    parser.add_argument('--re-init', action='store_true', help="Re-initializes model parameters")
    parser.add_argument('--fix-backbone', action='store_true', help="Fixes backbone parameters")
    # Contact prediction
    parser.add_argument('--distance-threshold', default=10., type=float, help="Minimum distance between two atoms in angström that is not considered as a contact")
    # Preprocessing
    parser.add_argument('--subsampling-mode', default='uniform', type=str, help="Subsampling mode: uniform, diversity, fixed")
    # Training process
    parser.add_argument('--num-epochs', default=1000, type=int, help="Number of training epochs")
    parser.add_argument('--batch-size', default=1, type=int, help="Batch size (local in case of multi-gpu training)")
    parser.add_argument('--learning-rate', default=1e-4, type=float, help="Initial learning rate")
    parser.add_argument('--learning-rate-warmup', default=1000, type=int, help="Warmup parameter for inverse square root rule of learning rate scheduling")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout probability")
    parser.add_argument('--loss-contact-weight', default=0.5, type=float, help="Weight that is used to rescale loss for contacts. Weight for no-contacts equals 1 minus the set value.")
    parser.add_argument('--precision', default=32, type=int, help="Precision used for computations")
    parser.add_argument('--disable-progress-bar', action='store_true', help="disables the training progress bar")
    parser.add_argument('--disable-shuffle', action='store_true', help="disables the dataset shuffling")
    parser.add_argument('--rng-seed', default=42, type=int, help="Random number generator seed")
    parser.add_argument('--validation-ratio', default=0.1, type=float, help="Ratio of the validation dataset w.r.t. the full training dataset, if k-fold cross validation is disabled.")
    parser.add_argument('--cv-num-folds', default=1, type=int, help="Number of folds in k-fold cross validation. If 1, then cross validation is disabled.")
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

    num_gpus = args.num_gpus if args.num_gpus >= 0 else torch.cuda.device_count()
    if num_gpus * args.num_nodes > 1:
        dp_strategy = DDPPlugin(find_unused_parameters=True)
        # NOTE for some reason, load_from_checkpoint fails to infer the hyperparameters correctly from the checkpoint file
        checkpoint = torch.load(args.checkpoint)
    else:
        dp_strategy = None
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    h_params = checkpoint['hyper_parameters']

    learning_rate = 0.0001
    
    downstream_transform = get_downstream_transforms(subsample_depth=h_params['subsampling_depth'], subsample_mode=args.subsampling_mode, threshold=args.distance_threshold)
    kfold_cv_downstream = datasets.KFoldCVDownstream(downstream_transform, num_folds=args.cv_num_folds, val_ratio=args.validation_ratio, batch_size=args.batch_size, shuffle=not args.disable_shuffle, rng_seed=args.rng_seed)

    dt_now = datetime.now()
    log_dir = h_params['log_dir'] if args.log_dir == "" else args.log_dir
    log_exp_name = h_params['log_exp_name'] if args.log_exp_name == "" else args.log_exp_name
    log_run_name = "downstream__" + h_params['log_run_name'] if args.log_run_name == "" else dt_now.strftime(args.log_run_name)
    if args.cv_num_folds >= 2:
        log_exp_name = log_run_name

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
        
    for fold_idx in range(args.cv_num_folds):
        train_metrics, val_metrics = get_downstream_metrics()
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

        if args.re_init:
            model = models.self_supervised.MSAModel(
                    num_blocks = h_params['num_blocks'],
                    num_heads = h_params['num_heads'],
                    dim_head = h_params['feature_dim_head'],
                    task_heads=task_heads,
                    task_losses=task_losses,
                    alphabet_size=len(downstream_ds.token_mapping),
                    padding_token=downstream_ds.token_mapping['PADDING_TOKEN'],
                    lr=args.learning_rate,
                    lr_warmup=args.learning_rate_warmup,
                    dropout=args.dropout,
                    emb_grad_freq_scale=not h_params['disable_emb_grad_freq_scale'],
                    fix_backbone=args.fix_backbone)
        else:
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
                    dropout=args.dropout,
                    emb_grad_freq_scale=not h_params['disable_emb_grad_freq_scale'],
                    fix_backbone=args.fix_backbone)
        model.tasks = ['contact']
        model.losses['contact'] = nn.NLLLoss(weight=torch.tensor([1-args.loss_contact_weight, args.loss_contact_weight]), ignore_index=-1)
        model.task_heads['contact'] = models.self_supervised.msa.modules.ContactHead(h_params['num_blocks'] * h_params['num_heads'], cull_tokens=[downstream_ds.token_mapping[l] for l in ['-', '.', 'START_TOKEN', 'DELIMITER_TOKEN']])
        model.need_attn = True
        model.task_loss_weights = {'contact': 1.}
        model.train_metrics = train_metrics
        model.val_metrics = val_metrics
        
        kfold_cv_downstream.setup_fold_index(fold_idx)

        train_dl = kfold_cv_downstream.train_dataloader()
        val_dl = kfold_cv_downstream.val_dataloader()
        
        if args.cv_num_folds >= 2:
            log_run_name = 'fold_%d' % (fold_idx + 1)

        tb_logger = TensorBoardLogger(save_dir=log_dir, name=log_exp_name, version=log_run_name)
        checkpoint_callback = ModelCheckpoint(monitor='contact_validation_topLprec', filename="downstream-{epoch:02d}-{contact_validation_topLprec:.4f}", mode='max')
        
        trainer = Trainer(max_epochs=args.num_epochs,
                          gpus=args.num_gpus,
                          auto_select_gpus=num_gpus > 0,
                          num_nodes=args.num_nodes,
                          precision=args.precision,
                          strategy=dp_strategy,
                          enable_progress_bar=not args.disable_progress_bar,
                          log_every_n_steps=args.log_every,
                          logger=tb_logger,
                          callbacks=[checkpoint_callback])
        trainer.fit(model, train_dl, val_dl)

if __name__ == '__main__':
    main()
