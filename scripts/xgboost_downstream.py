import argparse
from datetime import datetime
from functools import partial
import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
import xgboost as xgb

from selbstaufsicht.models import xgb_contact


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
    
    top_l_prec = xgb_contact.xgb_topkLPrec(preds, dtrain, msa_mapping, L_mapping, k=k, treat_all_preds_positive=treat_all_preds_positive)
    
    return 'top-%sL-Prec' % str(k), top_l_prec


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
    parser.add_argument('--disable-progress-bar', action='store_true', help="disables the training progress bar")
    parser.add_argument('--disable-shuffle', action='store_true', help="disables the dataset shuffling")
    parser.add_argument('--rng-seed', default=42, type=int, help="Random number generator seed")
    parser.add_argument('--validation-ratio', default=0.2, type=float, help="Ratio of the validation dataset w.r.t. the full training dataset, if k-fold cross validation is disabled.")
    parser.add_argument('--cv-num-folds', default=1, type=int, help="Number of folds in k-fold cross validation. If 1, then cross validation is disabled.")
    parser.add_argument('--disable-train-data-discarding', action='store_true', help="disables the size-based discarding of training data")
    parser.add_argument('--top-l-prec-coeff', default=1., type=float, help="Coefficient k that is used in computing the top-(k*L)-precision.")
    parser.add_argument('--treat-all-preds-positive', action='store_true', help="Whether all non-ignored preds are treated as positives, analogous to the CocoNet paper.")
    # XGBoost HParams
    parser.add_argument('--num-round', default=100, type=int, help="Number of rounds performed by XGBoost. Also equals the number of trees.")
    parser.add_argument('--num-early-stopping-round', default=20, type=int, help="Number of rounds in which the validation metric needs to improve at least once in order to continue training.")
    parser.add_argument('--learning-rate', default=0.3, type=float, help="Learning rate used by XGBoost.")
    parser.add_argument('--gamma', default=0.3, type=float, help="Minimum loss reduction required to make a further partition on a leaf node of the tree. Increases model bias.")
    parser.add_argument('--max-depth', default=6, type=int, help="Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 indicates no limit on depth.")
    parser.add_argument('--min-child-weight', default=5, type=int, help="Minimum sum of instance weight (hessian) needed in a child.")
    parser.add_argument('--colsample-bytree', default=0.5, type=float, help="Subsample ratio of columns when constructing each tree.")
    parser.add_argument('--colsample-bylevel', default=0.5, type=float, help="Subsample ratio of columns for each level.")
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
    
    if not args.no_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    hparams = xgb_contact.get_checkpoint_hparams(args.checkpoint)
    train_dl = xgb_contact.create_dataloader('train', args.batch_size, args.subsampling_mode, args.distance_threshold, hparams, rng_seed=args.rng_seed, disable_train_data_discarding=args.disable_train_data_discarding)

    dt_now = datetime.now()
    log_exp_name = h_params['log_exp_name'] if args.log_exp_name == "" else args.log_exp_name
    log_run_name = "downstream__xgb__k_%s__" % str(args.top_l_prec_coeff).replace('.', '_') + h_params['log_run_name'] if args.log_run_name == "" else dt_now.strftime(args.log_run_name)
    
    if log_exp_name == "":
        log_path = os.path.join(args.log_dir, log_run_name)
    else:
        log_path = os.path.join(args.log_dir, log_exp_name, log_run_name)
        
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    cull_tokens = xgb_contact.get_cull_tokens(train_dl.dataset)
    model = xgb_contact.load_backbone(args.checkpoint, device, cull_tokens, hparams)
    attn_maps, targets, _, msa_mapping, L_mapping = xgb_contact.compute_attn_maps(model, train_dl, cull_tokens, args.diag_shift, device)
    
    params = {
        'booster': args.booster,
        'eta': args.learning_rate,
        'gamma': args.gamma,
        'max_depth': args.max_depth,
        'min_child_weight': args.min_child_weight,
        'colsample_bytree': args.colsample_bytree,
        'colsample_bylevel': args.colsample_bylevel,
        'subsample': args.xgb_subsampling_rate,
        'sampling_method': args.xgb_subsampling_mode,
        'scale_pos_weight': args.scale_pos_weight,
        'objective': 'binary:logitraw',
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
        train_msa_mapping, val_msa_mapping = msa_mapping[indices[val_size:]], msa_mapping[indices[:val_size]]
        
        train_data = xgb.DMatrix(train_attn_maps, label=train_targets)
        val_data = xgb.DMatrix(val_attn_maps, label=val_targets)
        
        evals_result = {}
        metric = partial(xgb_topkLPrec, msa_mappings=(train_msa_mapping, val_msa_mapping), L_mapping=L_mapping, k=args.top_l_prec_coeff, treat_all_preds_positive=args.treat_all_preds_positive)
        xgb_model = xgb.train(params, train_data, evals=[(train_data, 'train'), (val_data, 'validation')], evals_result=evals_result, num_boost_round=args.num_round, 
                              feval=metric, maximize=True, early_stopping_rounds=args.num_early_stopping_round, verbose_eval=not args.disable_progress_bar)
        xgb_model.save_model(os.path.join(log_path, 'model_checkpoint.json'))
        
        results = {}
        for k1, v1 in evals_result.items():
            for k2, v2 in v1.items():
                results['%s_%s' % (k2, k1)] = v2
        results = pd.DataFrame.from_dict(results)
        results.to_csv(os.path.join(log_path, 'train_log.csv'))
        
    elif args.cv_num_folds > 1:
        data = xgb.DMatrix(attn_maps, label=targets)
        splits = [split for split in KFold(args.cv_num_folds, shuffle=not args.disable_shuffle, random_state=args.rng_seed).split(range(attn_maps.shape[0]))]
        
        for idx in range(args.cv_num_folds):
            train_indices, val_indices = splits[idx]

            train_attn_maps, val_attn_maps = attn_maps[train_indices, :], attn_maps[val_indices, :]
            train_targets, val_targets = targets[train_indices], targets[val_indices]
            train_msa_mapping, val_msa_mapping = msa_mapping[train_indices], msa_mapping[val_indices]
            
            train_data = xgb.DMatrix(train_attn_maps, label=train_targets)
            val_data = xgb.DMatrix(val_attn_maps, label=val_targets)
            
            evals_result = {}
            metric = partial(xgb_topkLPrec, msa_mappings=(train_msa_mapping, val_msa_mapping), L_mapping=L_mapping, k=args.top_l_prec_coeff, treat_all_preds_positive=args.treat_all_preds_positive)
            xgb.train(params, train_data, evals=[(train_data, 'train'), (val_data, 'validation')], evals_result=evals_result, num_boost_round=args.num_round, 
                    feval=metric, maximize=True, early_stopping_rounds=args.num_early_stopping_round, verbose_eval=not args.disable_progress_bar)
            
            results = {}
            for k1, v1 in evals_result.items():
                for k2, v2 in v1.items():
                    results['%s_%s' % (k2, k1)] = v2
            results = pd.DataFrame.from_dict(results)
            results.to_csv(os.path.join(log_path, 'cv_%d_log.csv' % idx))
    else:
        raise ValueError("Number of CV folds must be positive!")


if __name__ == '__main__':
    main()