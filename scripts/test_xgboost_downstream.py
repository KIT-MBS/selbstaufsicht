import argparse
from datetime import datetime
from functools import partial
import json
import random
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader
import xgboost as xgb

from selbstaufsicht import models
from selbstaufsicht import datasets
from selbstaufsicht.models.self_supervised.msa.utils import get_downstream_transforms, MSACollator, get_tasks, get_downstream_metrics


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Applies sigmoid function.
    
    Args:
        x (np.ndarray): Input data.
        
    Returns:
        np.ndarray: Output data.
    """

    return 1 / (1 + np.exp(-x))


def xgb_topkLPrec_var_k(preds: np.ndarray, dtest: xgb.DMatrix, msa_mapping: np.ndarray, L_mapping: np.ndarray, k: np.ndarray, treat_all_preds_positive: bool = True) -> Tuple[Dict[float, np.ndarray], Dict[int, np.ndarray]]:
    """
    Custom XGBoost Metric for top-L-precision with support for various k.

    Args:
        preds (np.ndarray): Predictions [B] as logits.
        dtest (xgb.DMatrix): Test data (x: [B, num_maps], y: [B]).
        msa_mapping (np.ndarray): Mapping: Data point -> MSA index [B].
        L_mapping (np.ndarray): Mapping: MSA index -> MSA L.
        k (np.ndarray): Coefficients k that are used in computing the top-(k*L)-precision [num_k, num_msa].
        treat_all_preds_positive (bool, optional): Whether all non-ignored preds are treated as positives, analogous to the CocoNet paper. Defaults to True.

    Returns:
        Tuple[Dict[float, np.ndarray], Dict[float, np.ndarray]]: Metric values per k (relative); Metric values per k*L (absolute).
    """
    
    y = dtest.get_label()  # [B]
    
    msa_indices = np.unique(msa_mapping)
    
    top_l_prec_dict_rel = dict()
    top_l_prec_dict_abs = dict()
    
    # for each MSA, find top-L and compute true/false positives
    for msa_idx in msa_indices:
        mask = msa_mapping == msa_idx  # [B]
        preds_ = preds[mask]
        y_ = y[mask]
        
        for k_idx in range(len(k)):
            k_ = k[k_idx]
            L = L_mapping[msa_idx]
            k_L = min(max(1, int(k_*L)), len(y_))
            stop_flag = k_L == len(y_)
            L_idx = np.argpartition(preds_, -k_L)[-k_L:]  # [k*L]
            
            preds__ = np.round(sigmoid(preds_[L_idx]))
            y__ = y_[L_idx]
            
            if treat_all_preds_positive:
                tp = sum(y__ == 1)
                fp = sum(y__ == 0)
            else:
                tp = sum(np.logical_and(preds__ == 1, y__ == 1))
                fp = sum(np.logical_and(preds__ == 1, y__ == 0))
    
            top_l_prec = float(tp) / (tp + fp)
            
            if k_ not in top_l_prec_dict_rel:
                top_l_prec_dict_rel[k_] = []
            if k_L not in top_l_prec_dict_abs:
                top_l_prec_dict_abs[k_L] = []
            
            top_l_prec_dict_rel[k_].append(top_l_prec)
            top_l_prec_dict_abs[k_L].append(top_l_prec)
            
            if stop_flag:
                break
    
    top_l_prec_dict_rel = {k: np.array(v) for k, v in top_l_prec_dict_rel.items()}
    top_l_prec_dict_abs = {k: np.array(v) for k, v in top_l_prec_dict_abs.items()}
    
    return top_l_prec_dict_rel, top_l_prec_dict_abs


def xgb_topkLPrec(preds: np.ndarray, dtest: xgb.DMatrix, msa_mapping: np.ndarray, L_mapping: np.ndarray, k: float = 1., treat_all_preds_positive: bool = False) -> float:
    """
    Custom XGBoost Metric for top-L-precision.

    Args:
        preds (np.ndarray): Predictions [B] as logits.
        dtest (xgb.DMatrix): Test data (x: [B, num_maps], y: [B]).
        msa_mapping (np.ndarray): Mapping: Data point -> MSA index [B].
        L_mapping (np.ndarray): Mapping: MSA index -> MSA L.
        k (float, optional): Coefficient k that is used in computing the top-(k*L)-precision. Defaults to 1.
        treat_all_preds_positive (bool, optional): Whether all non-ignored preds are treated as positives, analogous to the CocoNet paper. Defaults to False.

    Returns:
        float: Metric value.
    """
    
    y = dtest.get_label()  # [B]
    
    msa_indices = np.unique(msa_mapping)
    tp = 0
    fp = 0
    
    # for each MSA, find top-L and compute true/false positives
    for msa_idx in msa_indices:
        mask = msa_mapping == msa_idx  # [B]
        preds_ = preds[mask]
        y_ = y[mask]
        
        L = L_mapping[msa_idx]
        kL = min(max(1, int(k*L)), len(y_))
        L_idx = np.argpartition(preds_, -kL)[-kL:]  # [k*L]
        
        preds_ = np.round(sigmoid(preds_[L_idx]))
        y_ = y_[L_idx]
        
        if treat_all_preds_positive:
            tp += sum(y_ == 1)
            fp += sum(y_ == 0)
        else:
            tp += sum(np.logical_and(preds_ == 1, y_ == 1))
            fp += sum(np.logical_and(preds_ == 1, y_ == 0))
    
    top_l_prec = float(tp) / (tp + fp)
    
    return top_l_prec


def plot_contact_maps(preds: np.ndarray, dtest: xgb.DMatrix, msa_mapping: np.ndarray, msa_mask: np.ndarray, L_mapping: np.ndarray, save_dir: str) -> None:
    """
    Plots predictions and ground truth of contact maps side by side.

    Args:
        preds (np.ndarray): Predictions [B] as logits.
        dtest (xgb.DMatrix): Test data (x: [B, num_maps], y: [B]).
        msa_mapping (np.ndarray): Mapping: Data point -> MSA index [B].
        msa_mask (np.ndarray): MSA mask [B].
        L_mapping (np.ndarray): Mapping: MSA index -> MSA L.
        save_dir (str): Directory, where plots are saved.
    """
    
    y = dtest.get_label()  # [B]
    
    msa_indices = np.unique(msa_mapping)
    
    preds_ = np.full(len(msa_mask), -np.inf)
    assert sum(msa_mask) == len(preds)
    preds_[msa_mask] = preds
    
    y_ = np.zeros(len(msa_mask), dtype=int)
    assert sum(msa_mask) == len(y)
    y_[msa_mask] = y
    
    # for each MSA, plot prediction and ground-truth
    for msa_idx in msa_indices:
        mask = msa_mapping == msa_idx  # [B]
        L = L_mapping[msa_idx]
        
        preds_shaped = sigmoid(preds_[mask].reshape((L, L)))
        preds_shaped += preds_shaped.T
        preds_shaped_binary = np.round(preds_shaped).astype(bool)
        y_shaped = y_[mask].reshape((L, L)).astype(bool)
        y_shaped += y_shaped.T
        
        fig, ax = plt.subplots(1, 3)
        sns.heatmap(preds_shaped, fmt='', ax=ax[0])
        sns.heatmap(preds_shaped_binary, fmt='', ax=ax[1])
        sns.heatmap(y_shaped, fmt='', ax=ax[2])

        ax[0].set_aspect('equal')
        ax[0].set_title("Prediction")
        ax[1].set_aspect('equal')
        ax[1].set_title("Prediction (binary)")
        ax[2].set_aspect('equal')
        ax[2].set_title("Target")

        fig.set_size_inches(15, 5)
        fig.suptitle("Test Data: MSA %d" % msa_idx)
        fig.savefig(os.path.join(save_dir, '%d.pdf' % msa_idx))


def plot_top_l_prec_over_k(top_l_prec_dict_rel: Dict[float, np.ndarray], top_l_prec_dict_abs: Dict[int, np.ndarray], save_dir: str) -> None:
    """
    Creates plots for top-(k*L)-precision over k and (k*L), respectively.

    Args:
        top_l_prec_dict_rel (Dict[float, np.ndarray]): Metric values per k (relative). 
        top_l_prec_dict_abs (Dict[int, np.ndarray]): Metric values per k*L (absolute).
        save_dir (str): Directory, where plots are saved.
    """
    
    x_rel = np.array([key for key in top_l_prec_dict_rel.keys()])
    y_rel = np.array([val.mean() for val in top_l_prec_dict_rel.values()])
    std_rel = np.array([val.std(ddof=1) for val in top_l_prec_dict_rel.values()])
    
    x_abs = np.array([key for key in top_l_prec_dict_abs.keys()])
    y_abs = np.array([val.mean() for val in top_l_prec_dict_abs.values()])
    std_abs = np.array([val.std(ddof=1) for val in top_l_prec_dict_abs.values()])
    
    fig, ax = plt.subplots(1, 2)
    
    ax[0].set_title("Relative")
    ax[0].set_xlabel("k")
    ax[0].set_ylabel("Top-(k*L)-Precision")
    ax[0].plot(x_rel, y_rel, 'r-')
    ax[0].fill_between(x_rel, y_rel - std_rel, y_rel + std_rel, color='r', alpha=0.2)
    
    ax[1].set_title("Absolute")
    ax[1].set_xlabel("k*L")
    ax[1].set_ylabel("Top-(k*L)-Precision")
    ax[0].plot(x_abs, y_abs, 'b-')
    ax[0].fill_between(x_abs, y_abs - std_abs, y_abs + std_abs, color='b', alpha=0.2)
    
    fig.set_size_inches(15, 5)
    fig.suptitle("Relative and Absolute Top-(k*L)-Precision Plots")
    fig.savefig(os.path.join(save_dir, 'topLPrec.pdf'))


def main():
    parser = argparse.ArgumentParser(description='Selbstaufsicht Weakly Supervised Contact Prediction Script (XGBoost version)')
    # Trained models
    parser.add_argument('--checkpoint', type=str, help="Path to downstream model checkpoint")
    parser.add_argument('--xgboost-checkpoint', type=str, help="Path to xgboost model checkpoint")
    # Contact prediction
    parser.add_argument('--distance-threshold', default=10., type=float, help="Minimum distance between two atoms in angström that is not considered as a contact")
    # Preprocessing
    parser.add_argument('--subsampling-mode', default='uniform', type=str, help="Subsampling mode: uniform, diversity, fixed")
    parser.add_argument('--diag-shift', default=4, type=int, help="Width of the area around the main diagonal of prediction maps that is ignored.")
    # Test process
    parser.add_argument('--batch-size', default=1, type=int, help="Batch size (local in case of multi-gpu testing)")
    parser.add_argument('--min-k', default=0.01, type=float, help="Minimum coefficient k that is used in computing the top-(k*L)-precision.")
    parser.add_argument('--max-k', default=-1, type=float, help="Maximum coefficient k that is used in computing the top-(k*L)-precision. -1 refers to maximum L/2 of the longest sequence.")
    parser.add_argument('--num-k', default=500, type=int, help="Number of samples for k used in computing the top-(k*L)-precision. 1 disables top-(k*L)-precision over k plot and uses min-k as k.")
    parser.add_argument('--treat-all-preds-positive', action='store_true', help="Whether all non-ignored preds are treated as positives, analogous to the CocoNet paper.")
    # Visualization
    parser.add_argument('--vis-dir', type=str, default='', help="Directory, where plots are saved. If empty, no plots are created.")
    parser.add_argument('--vis-contact-maps', action='store_true', help="Creates contact map plots.")
    parser.add_argument('--vis-k-plot', action='store_true', help="Creates top-(k*L)-precision over k plot.")
    # GPU
    parser.add_argument('--no-gpu', action='store_true', help="disables cuda")

    args = parser.parse_args()
    
    if not args.no_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        
    checkpoint = torch.load(args.checkpoint, map_location=device)
    h_params = checkpoint['hyper_parameters']
    
    downstream_transform = get_downstream_transforms(subsample_depth=h_params['subsampling_depth'], subsample_mode=args.subsampling_mode, threshold=args.distance_threshold)
    root = os.environ['DATA_PATH']
    test_dataset = datasets.CoCoNetDataset(root, 'test', transform=downstream_transform, diversity_maximization=args.subsampling_mode=='diversity')

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
    
    num_maps = h_params['num_blocks'] * h_params['num_heads']
    cull_tokens = [test_dataset.token_mapping[token] for token in ['-', '.', 'START_TOKEN', 'DELIMITER_TOKEN']]
    if 'downstream' in args.checkpoint:
        task_heads['contact'] = models.self_supervised.msa.modules.ContactHead(num_maps, cull_tokens=cull_tokens)
        task_losses['contact'] = None

    model = models.self_supervised.MSAModel.load_from_checkpoint(
        checkpoint_path = args.checkpoint,
        num_blocks = h_params['num_blocks'],
        num_heads = h_params['num_heads'],
        feature_dim_head = h_params['feature_dim_head'],
        task_heads=task_heads,
        task_losses=task_losses,
        alphabet_size=len(test_dataset.token_mapping),
        padding_token=test_dataset.token_mapping['PADDING_TOKEN'],
        dropout=0.,
        emb_grad_freq_scale=not h_params['disable_emb_grad_freq_scale'])
    model.need_attn = True
    model.to(device)

    test_dl = DataLoader(test_dataset,
                         batch_size=args.batch_size,
                         shuffle=False,
                         num_workers=0,
                         pin_memory=False)
    
    attn_maps_list = []
    targets_list = []
    msa_mapping_list = []
    msa_mask_list = []
    msa_mapping_filtered_list = []
    L_mapping_list = []
    
    for idx, (x, y) in enumerate(test_dl):
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
        
        attn_maps_triu = attn_maps.view(-1, num_maps)  # [1*L*L, num_maps]
        attn_maps_tril = torch.permute(attn_maps, (0, 2, 1, 3)).reshape(-1, num_maps)  # [1*L*L, num_maps]
        target = y['contact'].view(-1)  # [1*L*L]
        msa_mapping = torch.full_like(target, idx)  # [1*L*L]
        
        # exclude unknown target points, apply diag shift, averge over both triangle matrices
        mask = y['contact'] != -1
        mask_triu = torch.triu(torch.ones_like(mask), args.diag_shift).view(-1)  # [1*L*L]
        
        mask = mask.view(-1)  # [1*L*L]
        mask_attn_maps = mask[mask_triu]
        mask_target = torch.logical_and(mask, mask_triu)
        
        attn_maps_triu = attn_maps_triu[mask_triu, :]
        attn_maps_tril = attn_maps_tril[mask_triu, :]
        
        attn_maps = 0.5 * (attn_maps_triu + attn_maps_tril)
        attn_maps = attn_maps[mask_attn_maps, :]
        target = target[mask_target]
        msa_mapping_filtered = msa_mapping[mask_target]
        
        attn_maps_list.append(attn_maps)
        targets_list.append(target)
        msa_mapping_list.append(msa_mapping)
        msa_mask_list.append(mask_target)
        msa_mapping_filtered_list.append(msa_mapping_filtered)
        L_mapping_list.append(degapped_L)
    
    attn_maps = torch.cat(attn_maps_list)  # [B*L*L/2, num_maps]
    targets = torch.cat(targets_list)  # [B*L*L/2]
    msa_mapping = torch.cat(msa_mapping_list)  # [B*L*L]
    msa_mask = torch.cat(msa_mask_list)  # [B*L*L]
    msa_mapping_filtered = torch.cat(msa_mapping_filtered_list)  # [B*L*L/2]
    
    attn_maps = attn_maps.cpu().numpy()
    targets = targets.cpu().numpy()
    msa_mapping = msa_mapping.cpu().numpy()
    msa_mask = msa_mask.cpu().numpy()
    msa_mapping_filtered = msa_mapping_filtered.cpu().numpy()
    L_mapping = np.array(L_mapping_list)
    
    test_data = xgb.DMatrix(attn_maps, label=targets)
    
    xgb_model = xgb.Booster(model_file=args.xgboost_checkpoint)
    
    preds = xgb_model.predict(test_data, iteration_range=(0, xgb_model.best_iteration), strict_shape=True)[:, 0]
    
    if args.num_k == 1:
        top_l_prec = xgb_topkLPrec(preds, test_data, msa_mapping_filtered, L_mapping, args.min_k, args.treat_all_preds_positive)
        print("Top-%sL-Prec:" % str(args.min_k), top_l_prec)
    else:
        min_k = args.min_k
        if args.max_k == -1:
            max_k = max(L_mapping / 2)
        else:
            max_k = args.max_k
        k_range = np.linspace(min_k, max_k, args.num_k)  # [num_k]
        
        top_l_prec_dict_rel, top_l_prec_dict_abs = xgb_topkLPrec_var_k(preds, test_data, msa_mapping_filtered, L_mapping, k_range, args.treat_all_preds_positive)
        print("Relative Top-kL_Prec:", top_l_prec_dict_rel)
        print("Absolute Top-kL_Prec:", top_l_prec_dict_abs)
        
        if args.vis_dir != '' and args.vis_k_plot:
            plot_top_l_prec_over_k(top_l_prec_dict_rel, top_l_prec_dict_abs, args.vis_dir)
        
    if args.vis_dir != '' and args.vis_contact_maps:
        plot_contact_maps(preds, test_data, msa_mapping, msa_mask, L_mapping, args.vis_dir)

if __name__ == '__main__':
    main()
