import argparse
from collections import OrderedDict
from datetime import datetime
from functools import partial
import json
import random
import os
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import xgboost as xgb

from selbstaufsicht.models import xgb_contact


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
        
        preds_shaped = xgb_contact.sigmoid(preds_[mask].reshape((L, L)))
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


def plot_top_l_prec_over_k(top_l_prec_dict_rel: Dict[float, np.ndarray], top_l_prec_list_abs: Dict[int, Dict[int, float]], save_dir: str) -> None:
    """
    Creates plots for top-(k*L)-precision over k and (k*L), respectively.

    Args:
        top_l_prec_dict_rel (Dict[float, np.ndarray]): Metric values per k (relative). 
        top_l_prec_list_abs (Dict[int, Dict[int, float]]): Metric values per k per MSA (absolute).
        save_dir (str): Directory, where plots are saved.
    """
    
    x_rel = np.array([key for key in top_l_prec_dict_rel.keys()])
    sort_indices = np.argsort(x_rel)
    x_rel = x_rel[sort_indices]
    y_rel = np.array([val.mean() for val in top_l_prec_dict_rel.values()])[sort_indices]
    std_rel = np.array([val.std(ddof=1) for val in top_l_prec_dict_rel.values()])[sort_indices]
    
    fig = plt.gcf()
    ax = plt.gca()
    fig.set_size_inches(16, 9)
    fig.suptitle("Top-(k*L)-Precision for relative k (all MSAs)")
    ax.set_xlabel("k")
    ax.set_ylabel("Top-(k*L)-Precision")
    ax.set_xscale('log')
    ax.plot(x_rel, y_rel, 'r-')
    ax.fill_between(x_rel, y_rel - std_rel, y_rel + std_rel, color='r', alpha=0.2)
    fig.savefig(os.path.join(save_dir, 'topLPrec_relative.pdf'))
    
    for idx, top_l_prec_dict in top_l_prec_list_abs.items():
        plt.clf()
        fig = plt.gcf()
        ax = plt.gca()
        fig.set_size_inches(16, 9)
        fig.suptitle("Top-k-Precision for absolute k (MSA %d)" % idx)
        ax.set_xlabel("k")
        ax.set_ylabel("Top-k-Precision")
        ax.set_xscale('log')
        ax.plot(top_l_prec_dict.keys(), top_l_prec_dict.values(), 'b-')
        fig.savefig(os.path.join(save_dir, 'topLPrec_absolute_%d.pdf' % idx))


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
        
    h_params = xgb_contact.get_checkpoint_hparams(args.checkpoint, device)
    test_dl = xgb_contact.create_dataloader('test', args.batch_size, args.subsampling_mode, args.distance_threshold, h_params)
    
    cull_tokens = xgb_contact.get_cull_tokens(test_dl.dataset)
    model = xgb_contact.load_backbone(args.checkpoint, device, test_dl.dataset, cull_tokens, h_params)
    attn_maps, targets, msa_mapping, msa_mask, msa_mapping_filtered, L_mapping = xgb_contact.compute_attn_maps(model, test_dl, cull_tokens, args.diag_shift, h_params, device)  
    
    test_data = xgb.DMatrix(attn_maps, label=targets)
    
    xgb_model = xgb.Booster(model_file=args.xgboost_checkpoint)
    
    preds = xgb_model.predict(test_data, iteration_range=(0, xgb_model.best_iteration), strict_shape=True)[:, 0]
    
    if args.num_k == 1:
        top_l_prec = xgb_contact.xgb_topkLPrec(preds, test_data, msa_mapping_filtered, L_mapping, args.min_k, args.treat_all_preds_positive)
        f1_score = xgb_contact.xgb_F1Score(preds, test_data, msa_mapping_filtered)
        print("Top-%sL-Prec:" % str(args.min_k), top_l_prec)
        print("F1-Score:", f1_score)
    else:
        min_k = args.min_k
        if args.max_k == -1:
            max_k = max(L_mapping / 2)
        else:
            max_k = args.max_k
        k_range = np.broadcast_to(np.linspace(min_k, max_k, args.num_k)[..., None], (args.num_k, len(test_dl)))  # [num_k, num_msa]
        top_l_prec_dict_rel = xgb_contact.xgb_topkLPrec_var_k(preds, test_data, msa_mapping_filtered, L_mapping, k_range, treat_all_preds_positive=args.treat_all_preds_positive)
        
        k_range = np.linspace(1, 0.5*L_mapping**2, args.num_k, dtype=int)  # [num_k, num_msa]
        top_l_prec_dict_abs = xgb_contact.xgb_topkLPrec_var_k(preds, test_data, msa_mapping_filtered, L_mapping, k_range, relative_k=False, treat_all_preds_positive=args.treat_all_preds_positive)
        
        if args.vis_dir != '' and args.vis_k_plot:
            top_l_prec_plot_dir = os.path.join(args.vis_dir, 'top_l_prec_plots')
            os.makedirs(top_l_prec_plot_dir, exist_ok=True)
            plot_top_l_prec_over_k(top_l_prec_dict_rel, top_l_prec_dict_abs, top_l_prec_plot_dir)
        
    if args.vis_dir != '' and args.vis_contact_maps:
        contact_map_plot_dir = os.path.join(args.vis_dir, 'contact_maps')
        os.makedirs(contact_map_plot_dir, exist_ok=True)
        plot_contact_maps(preds, test_data, msa_mapping, msa_mask, L_mapping, contact_map_plot_dir)

if __name__ == '__main__':
    main()
