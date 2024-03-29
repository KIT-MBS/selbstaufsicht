import argparse
import glob
import os
import random
from datetime import datetime

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import L1Loss, MarginRankingLoss, MSELoss
from torch.utils.data import DataLoader

from selbstaufsicht import datasets, models
from selbstaufsicht.models.self_supervised.msa.utils import (
    get_downstream_metrics,
    get_downstream_transforms,
    get_tasks,
)
from selbstaufsicht.modules import BinaryFocalLoss, DiceLoss, SigmoidCrossEntropyLoss

# from selbstaufsicht.utils import data_loader_worker_init


def main():
    parser = argparse.ArgumentParser(description='Selbstaufsicht Weakly Supervised Downstream Training Script')
    # Pre-trained model
    parser.add_argument('--checkpoint', type=str, help="Path to pre-trained model checkpoint")
    parser.add_argument('--re-init', action='store_true', help="Re-initializes model parameters")
    parser.add_argument('--freeze-backbone', action='store_true', help="Freezes backbone parameters")
    # Task
    parser.add_argument('--task', default='contact', type=str, help="Downstream task ('contact', 'thermostable')")
    parser.add_argument('--distance-threshold', default=10., type=float, help="Minimum distance between two atoms in angström that is not considered as a contact")
    # Preprocessing
    parser.add_argument('--subsampling-mode', default='uniform', type=str, help="Subsampling mode: uniform, diversity, fixed")
    # Training process
    parser.add_argument('--num-epochs', default=100, type=int, help="Number of training epochs")
    parser.add_argument('--batch-size', default=1, type=int, help="Batch size (local in case of multi-gpu training)")
    parser.add_argument('--learning-rate', default=1e-4, type=float, help="Initial learning rate")
    parser.add_argument('--learning-rate-warmup', default=0, type=int, help="Warmup parameter for inverse square root rule of learning rate scheduling")
    parser.add_argument('--dropout', default=0., type=float, help="Dropout probability")
    parser.add_argument('--loss', default='focal', type=str, help="Loss function: cross-entropy, focal, dice (for contact task); mse (for thermostable task)")
    parser.add_argument('--loss-contact-weight', default=0.95, type=float, help="Weight that is used to rescale loss for contacts. Weight for no-contacts equals 1 minus the set value.")
    parser.add_argument('--loss-focal-gamma', default=2., type=float, help="Exponential weight used for focal loss.")
    parser.add_argument('--precision', default=16, type=int, help="Precision used for computations")
    parser.add_argument('--disable-progress-bar', action='store_true', help="disables the training progress bar")
    parser.add_argument('--disable-shuffle', action='store_true', help="disables the dataset shuffling")
    parser.add_argument('--rng-seed', default=42, type=int, help="Random number generator seed")
    parser.add_argument('--secondary-window', default=-1, type=int, help="window to ignore around secondary structure contacts. -1 means secondary contac will be predicted without special consideration.")
    parser.add_argument('--validation-ratio', default=0.2, type=float, help="Ratio of the validation dataset w.r.t. the full training dataset, if k-fold cross validation is disabled.")
    parser.add_argument('--cv-num-folds', default=1, type=int, help="Number of folds in k-fold cross validation. If 1, then cross validation is disabled.")
    parser.add_argument('--disable-train-data-discarding', action='store_true', help="disables the size-based discarding of training data")
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
    secondary_window = args.secondary_window

    num_gpus = args.num_gpus if args.num_gpus >= 0 else torch.cuda.device_count()
    if num_gpus * args.num_nodes > 1:
        dp_strategy = "ddp"
       # NOTE for some reason, load_from_checkpoint fails to infer the hyperparameters correctly from the checkpoint file
        checkpoint = torch.load(args.checkpoint)
    else:
        dp_strategy = None
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    h_params = checkpoint['hyper_parameters']
    downstream_args = {'downstream__' + k: v for k, v in vars(args).items()}
    h_params.update(downstream_args)

    downstream_transform = get_downstream_transforms(task=args.task, subsample_depth=h_params['subsampling_depth'], subsample_mode=args.subsampling_mode, threshold=args.distance_threshold, secondary_window=secondary_window,crop_size=h_params['cropping_size']-1)
    kfold_cv_downstream = datasets.KFoldCVDownstream(downstream_transform,
                                                     args.task,
                                                     num_folds=args.cv_num_folds,
                                                     val_ratio=args.validation_ratio,
                                                     batch_size=args.batch_size,
                                                     shuffle=not args.disable_shuffle,
                                                     rng_seed=args.rng_seed,
                                                     discard_train_size_based=not args.disable_train_data_discarding,
                                                     diversity_maximization=args.subsampling_mode == 'diversity',
                                                     max_seq_len=h_params['cropping_size'],
                                                     min_num_seq=h_params['subsampling_depth'],
                                                     secondary_window=secondary_window)
    

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
    if h_params['task_jigsaw_boot']:
        tasks.append("jigsaw_boot")

    for fold_idx in range(args.cv_num_folds):
        if args.cv_num_folds >= 2:
            log_run_name = 'fold_%d' % (fold_idx + 1)
        h_params['downstream__log_run_name'] = log_run_name

        train_metrics, val_metrics, _ = get_downstream_metrics(task=args.task)
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
                                                     jigsaw_delimiter=jigsaw_delimiter,
                                                     jigsaw_euclid_emb=jigsaw_euclid_emb,
                                                     simclr_temperature=h_params['contrastive_temperature'],
                                                     jigsaw_boot_ratio=h_params['jigsaw_boot_ratio'],
                                                     per_token=h_params['boot_per_token'],
                                                     boot_same=h_params['boot_same'],
                                                     frozen=h_params['frozen'],
                                                     seq_dist=h_params['seq_dist'])

        if args.re_init:
            model = models.self_supervised.MSAModel(
                    num_blocks=h_params['num_blocks'],
                    num_heads=h_params['num_heads'],
                    dim_head=h_params['feature_dim_head'],
                    task_heads=task_heads,
                    task_losses=task_losses,
                    alphabet_size=len(kfold_cv_downstream.train_dataset.token_mapping),
                    padding_token=kfold_cv_downstream.train_dataset.token_mapping['PADDING_TOKEN'],
                    lr=args.learning_rate,
                    lr_warmup=args.learning_rate_warmup,
                    dropout=args.dropout,
                    emb_grad_freq_scale=not h_params['disable_emb_grad_freq_scale'],
                    freeze_backbone=args.freeze_backbone,
                    h_params=h_params)
        else:
            model = models.self_supervised.MSAModel.load_from_checkpoint(
                    checkpoint_path=args.checkpoint,
                    num_blocks=h_params['num_blocks'],
                    num_heads=h_params['num_heads'],
                    feature_dim_head=h_params['feature_dim_head'],
                    task_heads=task_heads,
                    task_losses=task_losses,
                    alphabet_size=len(kfold_cv_downstream.train_dataset.token_mapping),
                    padding_token=kfold_cv_downstream.train_dataset.token_mapping['PADDING_TOKEN'],
                    lr=args.learning_rate,
                    lr_warmup=args.learning_rate_warmup,
                    dropout=args.dropout,
                    emb_grad_freq_scale=not h_params['disable_emb_grad_freq_scale'],
                    freeze_backbone=args.freeze_backbone,
                    max_seqlen=h_params['cropping_size'],
                    h_params=h_params)
       
       
        model.tasks = [args.task]
        if args.task == 'contact':
            model.task_heads[args.task] = models.self_supervised.msa.modules.ContactHead(h_params['num_blocks'] * h_params['num_heads'], cull_tokens=[kfold_cv_downstream.train_dataset.token_mapping[token] for token in ['-', '.', 'START_TOKEN', 'DELIMITER_TOKEN']])
            model.need_attn = True
        elif args.task == 'thermostable':
            model.task_heads[args.task] = models.self_supervised.msa.modules.ThermoStableHead(12*64, 1)
            model.need_attn = False  
        model.task_loss_weights = {args.task: 1.}
       
        model.train_metrics = train_metrics
        model.val_metrics = val_metrics

        if args.loss == 'cross-entropy':
            assert args.task == 'contact'
            model.losses[args.task] = SigmoidCrossEntropyLoss(weight=torch.tensor([1-args.loss_contact_weight, args.loss_contact_weight]), ignore_index=-1)
        elif args.loss == 'focal':
            assert args.task == 'contact'
            model.losses[args.task] = BinaryFocalLoss(gamma=args.loss_focal_gamma, weight=torch.tensor([1-args.loss_contact_weight, args.loss_contact_weight]), ignore_index=-1)
        elif args.loss == 'dice':
            assert args.task == 'contact'
            model.losses[args.task] = DiceLoss(ignore_index=-1)
        elif args.loss== 'mse':
            assert args.task == 'thermostable'
            model.losses[args.task] = MSELoss()
        else:
            raise ValueError("Unknown loss: %s" % args.loss)

        kfold_cv_downstream.setup_fold_index(fold_idx)

        train_dl = kfold_cv_downstream.train_dataloader()
        val_dl = kfold_cv_downstream.val_dataloader()

        tb_logger = TensorBoardLogger(save_dir=log_dir, name=log_exp_name, version=log_run_name)

        checkpoint_callback_last = ModelCheckpoint(monitor=None, filename="downstream-{epoch:02d}-last")
        callbacks = [checkpoint_callback_last]
        if args.task == 'contact':
            checkpoint_callback_valloss = ModelCheckpoint(monitor='contact_validation_loss', filename="downstream-{epoch:02d}-{loss:.4f}", mode='min')
            checkpoint_callback_toplprec = ModelCheckpoint(monitor='contact_validation_topLprec', filename="downstream-{epoch:02d}-{contact_validation_topLprec:.4f}", mode='max')
            checkpoint_callback_toplprecpos = ModelCheckpoint(monitor='contact_validation_topLprec_coconet', filename="downstream-{epoch:02d}-{contact_validation_topLprecpos:.4f}", mode='max')
            checkpoint_callback_matthews = ModelCheckpoint(monitor='contact_validation_Global_matthews', filename="downstream-{epoch:02d}-{contact_validation_Global_matthews:.4f}", mode='max')
            checkpoint_callback_f1score = ModelCheckpoint(monitor='contact_validation_Global_F1score', filename="downstream-{epoch:02d}-{contact_validation_Global_F1score:.4f}", mode='max')
            
            callbacks.append(checkpoint_callback_valloss)
            callbacks.append(checkpoint_callback_toplprec)
            callbacks.append(checkpoint_callback_toplprecpos)
            callbacks.append(checkpoint_callback_matthews)
            callbacks.append(checkpoint_callback_f1score)
        elif args.task == 'thermostable':
            checkpoint_callback_valloss = ModelCheckpoint(monitor='thermostable_validation_loss', filename="downstream-{epoch:02d}-{loss:.4f}", mode='min') 
            checkpoint_callback_scorr = ModelCheckpoint(monitor='thermostable_validation_scorr', filename="downstream-{epoch:02d}-{scorr:.4f}", mode='max')
            checkpoint_callback_pcorr = ModelCheckpoint(monitor='thermostable_validation_pcorr', filename="downstream-{epoch:02d}-{pcorr:.4f}", mode='max')
            
            callbacks.append(checkpoint_callback_valloss)
            callbacks.append(checkpoint_callback_scorr)
            callbacks.append(checkpoint_callback_pcorr)
        
        trainer = Trainer(max_epochs=args.num_epochs,
                          gpus=args.num_gpus,
                          auto_select_gpus=num_gpus > 0,
                          num_nodes=args.num_nodes,
                          precision=args.precision,
                          strategy=dp_strategy,
                          enable_progress_bar=not args.disable_progress_bar,
                          log_every_n_steps=args.log_every,
                          logger=tb_logger,
                          callbacks=callbacks)
        trainer.fit(model, train_dl, val_dl)


if __name__ == '__main__':
    main()
