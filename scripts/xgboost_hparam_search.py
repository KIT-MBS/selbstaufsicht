import argparse
from functools import partial
import os
import random
from typing import Any, Dict, List, Tuple

from mpi4py import MPI
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
from torch import nn
from torch.utils.data import DataLoader
import xgboost as xgb

from propulate import Propulator
from propulate.utils import get_default_propagator

from selbstaufsicht import models
from selbstaufsicht import datasets
from selbstaufsicht.models.self_supervised.msa.utils import get_downstream_transforms, get_tasks
from selbstaufsicht.utils import data_loader_worker_init


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Applies sigmoid function.
    
    Args:
        x (np.ndarray): Input data.
        
    Returns:
        np.ndarray: Output data.
    """

    return 1 / (1 + np.exp(-x))


def xgb_topLPrec(preds: np.ndarray, dtrain: xgb.DMatrix, msa_mappings: Tuple[np.ndarray, np.ndarray], L_mapping: np.ndarray, treat_all_preds_positive: bool = False) -> Tuple[str, float]:
    """
    Custom XGBoost Metric for top-L-precision.

    Args:
        preds (np.ndarray): Predictions [B] as logits.
        dtrain (xgb.DMatrix): Training data (x: [B, num_maps], y: [B]).
        msa_mappings (Tuple[np.ndarray, np.ndarray]): Mapping: Data point -> MSA index [B] (Train, Val).
        L_mapping (np.ndarray): Mapping: MSA index -> MSA L.
        treat_all_preds_positive (bool, optional): Whether all non-ignored preds are treated as positives, analogous to the CocoNet paper. Defaults to False.

    Returns:
        Tuple[str, float]: Metric name; metric value.
    """
    
    y = dtrain.get_label()  # [B]
    
    # Dirty hack: Find out by data length, whether training or validation is active. Only works, if training and validation dataset have different lengths.
    B = len(y)
    if len(msa_mappings[0]) == B:
        msa_mapping = msa_mappings[0]
    elif len(msa_mappings[1]) == B:
        msa_mapping = msa_mappings[1]
    else:
        raise ValueError("Given data length does not match to msa_mappings: %d != (%d, %d)" % (B, len(msa_mappings[0]), len(msa_mappings[1])))
    
    msa_indices = np.unique(msa_mapping)
    tp = 0
    fp = 0
    
    # for each MSA, find top-L and compute true/false positives
    for msa_idx in msa_indices:
        mask = msa_mapping == msa_idx  # [B]
        preds_ = preds[mask]
        y_ = y[mask]
        
        L = L_mapping[msa_idx]
        L_idx = np.argpartition(preds_, -L)[-L:]  # [L]
        
        preds_ = np.round(sigmoid(preds_[L_idx]))
        y_ = y_[L_idx]
        
        if treat_all_preds_positive:
            tp += sum(y_ == 1)
            fp += sum(y_ == 0)
        else:
            tp += sum(np.logical_and(preds_ == 1, y_ == 1))
            fp += sum(np.logical_and(preds_ == 1, y_ == 0))
    
    top_l_prec = float(tp) / (tp + fp)
    
    return 'topLPrec', top_l_prec


def hparam_objective(params: Dict[str, Any], attn_maps: np.ndarray, targets: np.ndarray, msa_mapping: np.ndarray, L_mapping: np.ndarray, 
                     num_early_stopping_round: int, cv_num_folds: int, treat_all_preds_positive: bool, gpu_id: int) -> float:
    data = xgb.DMatrix(attn_maps, label=targets)
    rng_seed = random.randint(0, 2**32-1)
    splits = [split for split in KFold(cv_num_folds, shuffle=True, random_state=rng_seed).split(range(attn_maps.shape[0]))]
    max_objectives = []
    
    xgb_params = {
        'booster': 'dart',
        'objective': 'binary:logitraw',
        'tree_method': 'gpu_hist',
        'gpu_id': gpu_id,
        'eta': params['learning_rate'],
        'gamma': params['gamma'],
        'max_depth': params['max_depth'],
        'colsample_bytree': params['colsample_bytree'],
        'colsample_bylevel': params['colsample_bylevel'],
        'subsample': params['xgb_subsampling_rate'],
        'sampling_method': params['xgb_subsampling_mode'],
        'scale_pos_weight': params['scale_pos_weight'],
        'rate_drop': params['dart_dropout']
    }
    
    for idx in range(cv_num_folds):
        train_indices, val_indices = splits[idx]

        train_attn_maps, val_attn_maps = attn_maps[train_indices, :], attn_maps[val_indices, :]
        train_targets, val_targets = targets[train_indices], targets[val_indices]
        train_msa_mapping, val_msa_mapping = msa_mapping[train_indices], msa_mapping[val_indices]
        
        train_data = xgb.DMatrix(train_attn_maps, label=train_targets)
        val_data = xgb.DMatrix(val_attn_maps, label=val_targets)
        
        evals_result = {}
        metric = partial(xgb_topLPrec, msa_mappings=(train_msa_mapping, val_msa_mapping), L_mapping=L_mapping, treat_all_preds_positive=treat_all_preds_positive)
        xgb.train(xgb_params, train_data, evals=[(train_data, 'train'), (val_data, 'validation')], evals_result=evals_result, num_boost_round=params['num_round'], 
                  feval=metric, maximize=True, early_stopping_rounds=num_early_stopping_round, verbose_eval=False)
        
        results = {}
        for k1, v1 in evals_result.items():
            for k2, v2 in v1.items():
                results['%s_%s' % (k2, k1)] = v2
        results = pd.DataFrame.from_dict(results)
        max_objectives.append(results['topLPrec_validation'].max())
    
    max_objectives = np.array(max_objectives)
    return -max_objectives.mean()
    

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
    parser.add_argument('--batch-size', default=1, type=int, help="Batch size (attention-map computation). Currently restricted to 1.")
    parser.add_argument('--rng-seed', default=42, type=int, help="Random number generator seed")
    parser.add_argument('--cv-num-folds', default=5, type=int, help="Number of folds in k-fold cross validation. If 1, then cross validation is disabled.")
    parser.add_argument('--disable-train-data-discarding', action='store_true', help="disables the size-based discarding of training data")
    parser.add_argument('--treat-all-preds-positive', action='store_true', help="Whether all non-ignored preds are treated as positives, analogous to the CocoNet paper.")
    # XGBoost HParams
    parser.add_argument('--num-round-min', default=10, type=int, help="Minimum number of rounds performed by XGBoost. Also equals the number of trees.")
    parser.add_argument('--num-round-max', default=500, type=int, help="Maximum number of rounds performed by XGBoost. Also equals the number of trees.")
    parser.add_argument('--num-early-stopping-round', default=10, type=int, help="Number of rounds in which the validation metric needs to improve at least once in order to continue training.")
    parser.add_argument('--learning-rate-min', default=0.01, type=float, help="Minimum learning rate used by XGBoost.")
    parser.add_argument('--learning-rate-max', default=0.5, type=float, help="Maximum learning rate used by XGBoost.")
    parser.add_argument('--gamma-min', default=0., type=float, help="Minimum loss reduction required to make a further partition on a leaf node of the tree (min). Increases model bias.")
    parser.add_argument('--gamma-max', default=1., type=float, help="Minimum loss reduction required to make a further partition on a leaf node of the tree (max). Increases model bias.")
    parser.add_argument('--max-depth-min', default=4, type=int, help="Maximum depth of a tree (min). Increasing this value will make the model more complex and more likely to overfit.")
    parser.add_argument('--max-depth-max', default=7, type=int, help="Maximum depth of a tree (max). Increasing this value will make the model more complex and more likely to overfit.")
    parser.add_argument('--colsample-bytree-min', default=0.4, type=float, help="Minimum subsample ratio of columns when constructing each tree.")
    parser.add_argument('--colsample-bytree-max', default=1., type=float, help="Maximum subsample ratio of columns when constructing each tree.")
    parser.add_argument('--colsample-bylevel-min', default=0.4, type=float, help="Minimum subsample ratio of columns for each level.")
    parser.add_argument('--colsample-bylevel-max', default=1., type=float, help="Maximum subsample ratio of columns for each level.")
    parser.add_argument('--xgb-subsampling-rate-min', default=0.4, type=float, help="Minimum subsample ratio of the training instances used by XGBoost.")
    parser.add_argument('--xgb-subsampling-rate-max', default=1., type=float, help="Maximum subsample ratio of the training instances used by XGBoost.")
    parser.add_argument('--xgb-subsampling-modes', default='uniform,gradient_based', type=str, help="The method used to sample the training instances by XGBoost: uniform, gradient_based.")
    parser.add_argument('--scale-pos-weight-min', default=1., type=float, help="Controls the balance of positive and negative weights, useful for unbalanced classes (min).")
    parser.add_argument('--scale-pos-weight-max', default=50., type=float, help="Controls the balance of positive and negative weights, useful for unbalanced classes (max).")
    parser.add_argument('--dart-dropout-min', default=0., type=float, help="Minimum tree dropout rate of XGBoost DART.")
    parser.add_argument('--dart-dropout-max', default=0.5, type=float, help="Maximum tree dropout rate of XGBoost DART.")
    # GPU
    parser.add_argument('--num-gpu-per-node', default=4, type=int, help="Number of available GPUs per node.")
    # Propulate
    parser.add_argument('--prop-pop-size', default=8, type=int, help="Population size in the evolutionary algorithm.")
    parser.add_argument('--prop-mate-p', default=.7, type=float, help="Probability for mate uniform in the evolutionary algorithm.")
    parser.add_argument('--prop-mut-p', default=.4, type=float, help="Probability for point mutation in the evolutionary algorithm.")
    parser.add_argument('--prop-rand-p', default=.1, type=float, help="Probability for init uniform in the evolutionary algorithm.")
    parser.add_argument('--prop-num-generations', default=3, type=int, help="Number of generations in the evolutionary algorithm.")

    args = parser.parse_args()
    
    gpu_id = MPI.COMM_WORLD.Get_rank() % args.num_gpu_per_node

    torch.manual_seed(args.rng_seed)
    np.random.seed(args.rng_seed)
    random.seed(args.rng_seed)
    data_loader_rng = torch.Generator()
    data_loader_rng.manual_seed(args.rng_seed)
    
    device = torch.device('cuda:%d' % gpu_id)

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
    msa_mapping_list = []
    L_mapping_list = []
    
    for idx, (x, y) in enumerate(train_dl):
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
        msa_mapping = torch.full_like(target, idx)  # [1*L*L]
        
        # exclude lower triangle and unknown target points, apply diag shift
        mask = target != -1
        mask = torch.logical_and(mask, torch.triu(torch.ones_like(mask), args.diag_shift))
        
        attn_maps = attn_maps[mask, :]
        target = target[mask]
        msa_mapping = msa_mapping[mask]
        
        attn_maps_list.append(attn_maps)
        targets_list.append(target)
        msa_mapping_list.append(msa_mapping)
        L_mapping_list.append(L)
    
    attn_maps = torch.cat(attn_maps_list)  # [B*L*L, num_maps]
    targets = torch.cat(targets_list)  # [B*L*L]
    msa_mapping = torch.cat(msa_mapping_list)  # [B*L*L]
    
    attn_maps = attn_maps.cpu().numpy()
    targets = targets.cpu().numpy()
    msa_mapping = msa_mapping.cpu().numpy()
    L_mapping = np.array(L_mapping_list)
    
    limits = {
        'num_round': (args.num_round_min, args.num_round_max),
        'learning_rate': (args.learning_rate_min, args.learning_rate_max),
        'gamma': (args.gamma_min, args.gamma_max),
        'max_depth': (args.max_depth_min, args.max_depth_max),
        'colsample_bytree': (args.colsample_bytree_min, args.colsample_bytree_max),
        'colsample_bylevel': (args.colsample_bylevel_min, args.colsample_bylevel_max),
        'xgb_subsampling_rate': (args.xgb_subsampling_rate_min, args.xgb_subsampling_rate_max),
        'xgb_subsampling_mode': tuple(args.xgb_subsampling_modes.split(',')),
        'scale_pos_weight': (args.scale_pos_weight_min, args.scale_pos_weight_max),
        'dart_dropout': (args.dart_dropout_min, args.dart_dropout_max)
    }
    
    objective_fn = partial(hparam_objective, attn_maps=attn_maps, targets=targets, msa_mapping=msa_mapping, L_mapping=L_mapping, 
                           num_early_stopping_round=args.num_early_stopping_round, cv_num_folds=args.cv_num_folds, 
                           treat_all_preds_positive=args.treat_all_preds_positive, gpu_id=gpu_id)
    propagator = get_default_propagator(args.prop_pop_size, limits, args.prop_mate_p, args.prop_mut_p, args.prop_rand_p)
    propulator = Propulator(objective_fn, propagator, generations=args.prop_num_generations)
    propulator.propulate()
    propulator.summarize()


if __name__ == '__main__':
    main()