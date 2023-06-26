import argparse
import gc

# import os
import random
from functools import partial
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

# import mpi4py
# mpi4py.rc.initialize=False
# mpi4py.rc.finalize=False
from mpi4py import MPI
from propulate import Islands
from propulate.propagators import SelectBest, SelectWorst
from propulate.utils import get_default_propagator
from sklearn.model_selection import KFold

from selbstaufsicht.models.xgb import xgb_utils

# from propulate.propagators import SelectUniform



def xgb_topkLPrec(preds: np.ndarray, dtrain: xgb.DMatrix, msa_mappings: Tuple[np.ndarray, np.ndarray], L_mapping: np.ndarray, k: float = 1., treat_all_preds_positive: bool = False) -> Tuple[str, float]:
    """
    Custom XGBoost Metric for top-(k*L)-precision.

    Args:
        preds (np.ndarray): Predictions [B] as logits.
        dtrain (xgb.DMatrix): Training data (x: [B, num_maps], y: [B]).
        msa_mappings (Tuple[np.ndarray, np.ndarray]): Mapping: Data point -> MSA index [B] (Train, Val).
        L_mapping (np.ndarray): Mapping: MSA index -> MSA L.
        k (float, optional): Coefficient k that is used in computing the top-(k*L)-precision. Defaults to 1.
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

    top_l_prec = xgb_utils.xgb_topkLPrec(preds, dtrain, msa_mapping, L_mapping, k=k, treat_all_preds_positive=treat_all_preds_positive)

    return 'top-%sL-Prec' % str(k), top_l_prec


def hparam_objective(params: Dict[str, Any], attn_maps: np.ndarray, targets: np.ndarray, msa_mapping: np.ndarray, L_mapping: np.ndarray,
                     num_early_stopping_round: int, cv_num_folds: int, k: float, treat_all_preds_positive: bool, gpu_id: int) -> float:
    # data = xgb.DMatrix(attn_maps, label=targets)
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
        metric = partial(xgb_topkLPrec, msa_mappings=(train_msa_mapping, val_msa_mapping), L_mapping=L_mapping, k=k, treat_all_preds_positive=treat_all_preds_positive)
        booster = xgb.train(xgb_params, train_data, evals=[(train_data, 'train'), (val_data, 'validation')], evals_result=evals_result, num_boost_round=params['num_round'],
                            feval=metric, maximize=True, early_stopping_rounds=num_early_stopping_round, verbose_eval=False)
        # free GPU memory manually to avoid OOM error
        booster.__del__()
        gc.collect()

        results = {}
        for k1, v1 in evals_result.items():
            for k2, v2 in v1.items():
                results['%s_%s' % (k2, k1)] = v2
        results = pd.DataFrame.from_dict(results)
        max_objectives.append(results['top-%sL-Prec_validation' % str(k)].max())

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
    parser.add_argument('--cv-num-folds', default=5, type=int, help="Number of folds in k-fold cross validation. If 1, then cross validation is disabled.")
    parser.add_argument('--disable-train-data-discarding', action='store_true', help="disables the size-based discarding of training data")
    parser.add_argument('--top-l-prec-coeff', default=1., type=float, help="Coefficient k that is used in computing the top-(k*L)-precision.")
    parser.add_argument('--treat-all-preds-positive', action='store_true', help="Whether all non-ignored preds are treated as positives, analogous to the CocoNet paper.")
    # XGBoost HParams
    parser.add_argument('--num-round-min', default=1, type=int, help="Minimum number of rounds performed by XGBoost. Also equals the number of trees.")
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
    parser.add_argument('--prop-migr-p', default=.1, type=float, help="Migration probability in the evolutionary algorithm.")
    parser.add_argument('--prop-num-generations', default=3, type=int, help="Number of generations in the evolutionary algorithm.")

    args = parser.parse_args()

    xgb.set_config(verbosity=0)

    gpu_id = MPI.COMM_WORLD.Get_rank() % args.num_gpu_per_node
    rng_seed = MPI.COMM_WORLD.Get_rank()

    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    random.seed(rng_seed)

    device = torch.device('cuda:%d' % gpu_id)

    h_params = xgb_utils.get_checkpoint_hparams(args.checkpoint, device)
    train_dl = xgb_utils.create_dataloader('train', args.batch_size, args.subsampling_mode, args.distance_threshold, h_params, rng_seed=args.rng_seed, disable_train_data_discarding=args.disable_train_data_discarding)

    cull_tokens = xgb_utils.get_cull_tokens(train_dl.dataset)
    model = xgb_utils.load_backbone(args.checkpoint, device, train_dl.dataset, cull_tokens, h_params)
    attn_maps, targets, _, _, msa_mapping, L_mapping = xgb_utils.compute_attn_maps(model, train_dl, cull_tokens, args.diag_shift, h_params, device)

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
                           num_early_stopping_round=args.num_early_stopping_round, cv_num_folds=args.cv_num_folds, k=args.top_l_prec_coeff,
                           treat_all_preds_positive=args.treat_all_preds_positive, gpu_id=gpu_id)
    num_migrants = 1
    migration_topology = num_migrants*np.ones((4, 4), dtype=int)
    np.fill_diagonal(migration_topology, 0)
    propagator = get_default_propagator(args.prop_pop_size, limits, args.prop_mate_p, args.prop_mut_p, args.prop_rand_p)
    islands = Islands(objective_fn, propagator, generations=args.prop_num_generations,
                      num_isles=4, isle_sizes=[4, 4, 4, 4], migration_topology=migration_topology,
                      load_checkpoint="pop_cpt.p",
                      save_checkpoint="pop_cpt.p",
                      migration_probability=args.prop_migr_p,
                      emigration_propagator=SelectBest, immigration_propagator=SelectWorst,
                      pollination=False)
    islands.evolve(top_n=1, logging_interval=1)


if __name__ == '__main__':
    main()
