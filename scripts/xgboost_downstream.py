import argparse
from datetime import datetime
from functools import partial
import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
from torch import nn
from torch.utils.data import DataLoader
import xgboost as xgb

from selbstaufsicht import models
from selbstaufsicht import datasets
from selbstaufsicht.models.self_supervised.msa.utils import get_downstream_transforms, get_tasks
from selbstaufsicht.utils import data_loader_worker_init


def xgb_accuracy(preds: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    """
    Custom XGBoost Metric for accuracy.

    Args:
        preds (np.ndarray): Predictions [B].
        dtrain (xgb.DMatrix): Training data (x: [B, num_maps], y: [B]).

    Returns:
        Tuple[str, float]: Metric name; metric value.
    """
    
    y = dtrain.get_label()  # [B]
    acc = float(sum(y == np.round(preds))) / len(y) 
    
    return 'accuracy', acc


def main():
    parser = argparse.ArgumentParser(description='Selbstaufsicht Weakly Supervised Contact Prediction Script (XGBoost version)')
    # Pre-trained model
    parser.add_argument('--checkpoint', type=str, help="Path to pre-trained model checkpoint")
    # Contact prediction
    parser.add_argument('--distance-threshold', default=10., type=float, help="Minimum distance between two atoms in angström that is not considered as a contact")
    # Preprocessing
    parser.add_argument('--subsampling-mode', default='uniform', type=str, help="Subsampling mode: uniform, diversity, fixed")
    parser.add_argument('--diag-shift', default=4, type=int, help="Width of the area around the main diagonal of prediction maps that is ignored.")
    # Training process
    parser.add_argument('--booster', default='dart', type=str, help="Booster algorithm used by XGBoost: gbtree, dart.")
    parser.add_argument('--num-round', default=100, type=int, help="Number of rounds performed by XGBoost.")
    parser.add_argument('--num-early-stopping-round', default=100, type=int, help="Number of rounds in which the validation metric needs to improve at least once in order to continue training.")
    parser.add_argument('--batch-size', default=1, type=int, help="Batch size (attention-map computation). Currently restricted to 1.")
    parser.add_argument('--disable-progress-bar', action='store_true', help="disables the training progress bar")
    parser.add_argument('--disable-shuffle', action='store_true', help="disables the dataset shuffling")
    parser.add_argument('--rng-seed', default=42, type=int, help="Random number generator seed")
    parser.add_argument('--validation-ratio', default=0.2, type=float, help="Ratio of the validation dataset w.r.t. the full training dataset, if k-fold cross validation is disabled.")
    parser.add_argument('--cv-num-folds', default=1, type=int, help="Number of folds in k-fold cross validation. If 1, then cross validation is disabled.")
    parser.add_argument('--disable-train-data-discarding', action='store_true', help="disables the size-based discarding of training data")
    # XGBoost HParams
    parser.add_argument('--learning-rate', default=0.3, type=float, help="Learning rate used by XGBoost.")
    parser.add_argument('--gamma', default=0.3, type=float, help="Minimum loss reduction required to make a further partition on a leaf node of the tree. Increases model bias.")
    parser.add_argument('--max-depth', default=6, type=int, help="Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 indicates no limit on depth.")
    parser.add_argument('--min-child-weight', default=1., type=float, help="Minimum sum of instance weight (hessian) needed in a child. Increases model bias.")
    parser.add_argument('--xgb-subsampling-rate', default=1., type=float, help="Subsample ratio of the training instances used by XGBoost. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees, preventing overfitting.")
    parser.add_argument('--xgb-subsampling-mode', default='uniform', type=str, help="The method used to sample the training instances by XGBoost: uniform, gradient_based.")
    parser.add_argument('--scale-pos-weight', default=1., type=float, help="Controls the balance of positive and negative weights, useful for unbalanced classes.")
    parser.add_argument('--dart-dropout', default=0., type=float, help="Tree dropout rate of XGBoost DART.")
    # GPU
    parser.add_argument('--no-gpu', action='store_true', help="disables cuda")
    # Logging
    parser.add_argument('--log-dir', default='xgb_logs/', type=str, help='Logging directory. If empty, the directory of the pre-trained model is used. Default: \"xgb_logs/\"')
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
    log_exp_name = h_params['log_exp_name'] if args.log_exp_name == "" else args.log_exp_name
    log_run_name = "downstream__xgb__" + h_params['log_run_name'] if args.log_run_name == "" else dt_now.strftime(args.log_run_name)
    
    if log_exp_name == "":
        log_path = os.path.join(args.log_dir, log_run_name)
    else:
        log_path = os.path.join(args.log_dir, log_exp_name, log_run_name)
        
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
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

        attn_maps = torch.cat([m.squeeze(dim=2) for m in attn_maps], dim=1)  # [1, num_maps, L, L]
        attn_maps = attn_maps.masked_select(mask).reshape(B, num_maps, degapped_L, degapped_L)
        attn_maps = torch.permute(attn_maps, (0, 2, 3, 1))  # [1, L, L, num_maps]
        
        assert num_maps == attn_maps.shape[-1]
        
        attn_maps = attn_maps.view(-1, num_maps)  # [1*L*L, num_maps]
        target = y['contact'].view(-1)  # [1*L*L]
        
        # exclude lower triangle and unknown target points, apply diag shift
        mask = target != -1
        mask = torch.logical_and(mask, torch.triu(torch.ones_like(mask), args.diag_shift))
        
        attn_maps = attn_maps[mask, :]
        target = target[mask]
        
        attn_maps_list.append(attn_maps)
        targets_list.append(target)
    
    attn_maps = torch.cat(attn_maps_list)  # [B*L*L, num_maps]
    targets = torch.cat(targets_list)  # [B*L*L]
    
    attn_maps = attn_maps.cpu().numpy()
    targets = targets.cpu().numpy()
    
    params = {
        'booster': args.booster,
        'eta': args.learning_rate,
        'gamma': args.gamma,
        'max_depth': args.max_depth,
        'min_child_weight': args.min_child_weight,
        'subsample': args.xgb_subsampling_rate,
        'sampling_method': args.xgb_subsampling_mode,
        'scale_pos_weight': args.scale_pos_weight,
        'objective': 'binary:logistic',
        'seed': args.rng_seed
    }
    
    if args.booster == 'dart':
        params['rate_drop'] = args.dart_dropout
        
    if args.no_gpu:
        params['tree_method'] = 'hist'
    else:
        params['tree_method'] = 'gpu_hist'
    
    if args.cv_num_folds == 1:
        val_size = int(args.validation_ratio * attn_maps.shape[0])
        indices = np.random.permutation(attn_maps.shape[0])
        
        train_attn_maps, val_attn_maps = attn_maps[indices[val_size:], :], attn_maps[indices[:val_size], :]
        train_targets, val_targets = targets[indices[val_size:]], targets[indices[:val_size]]
        
        train_data = xgb.DMatrix(train_attn_maps, label=train_targets)
        val_data = xgb.DMatrix(val_attn_maps, label=val_targets)
        
        evals_result = {}
        xgb_model = xgb.train(params, train_data, evals=[(val_data, 'validation')], evals_result=evals_result, num_boost_round=args.num_round, feval=xgb_accuracy, 
                              maximize=True, early_stopping_rounds=args.num_early_stopping_round, verbose_eval=not args.disable_progress_bar)
        xgb_model.save_model(os.path.join(log_path, 'checkpoint.model'))
        
        results = {}
        for k1, v1 in evals_result:
            for k2, v2 in v1:
                results['%s_%s' % (k2, k1)] = v2
        results = pd.DataFrame.from_dict(results)
        results.to_csv(os.path.join(log_path, 'train_log.csv'))
        
    elif args.cv_num_folds > 1:
        data = xgb.DMatrix(attn_maps, label=targets)
        folds = KFold(args.cv_num_folds, shuffle=not args.disable_shuffle, random_state=args.rng_seed)
        
        results = xgb.cv(params, data, num_boost_round=args.num_round, nfold=args.cv_num_folds, folds=folds, feval=xgb_accuracy, maximize=True, 
                         early_stopping_rounds=args.num_early_stopping_round, verbose_eval=not args.disable_progress_bar)
        
        print('\n\nCV Results:\n', results)
        with open(os.path.join(log_path, 'cv_log.txt'), 'w') as f:
            f.writelines(results)
    else:
        raise ValueError("Number of CV folds must be positive!")


if __name__ == '__main__':
    main()