import argparse
# from collections import OrderedDict
# from datetime import datetime
# from functools import partial
# import json
# import random
import os
from typing import Dict, List

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import xgboost as xgb

from selbstaufsicht.models.xgb import xgb_contact


def store_contact_maps_data(preds: np.ndarray, dtest: xgb.DMatrix, msa_mapping: np.ndarray, msa_mask: np.ndarray, L_mapping: np.ndarray, pdb_ids: List[str], top_l: bool, save_dir: str) -> None:
    """
    Creates and stores data for predictions and ground truth of contact maps.

    Args:
        preds (np.ndarray): Predictions [B] as logits.
        dtest (xgb.DMatrix): Test data (x: [B, num_maps], y: [B]).
        msa_mapping (np.ndarray): Mapping: Data point -> MSA index [B].
        msa_mask (np.ndarray): MSA mask [B].
        L_mapping (np.ndarray): Mapping: MSA index -> MSA L.
        pdb_ids (List[str]): List of PDB ids [B].
        top_l (bool): Only take top-L predictions.
        save_dir (str): Directory, where data is stored.
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

        preds_shaped = preds_[mask].reshape((L, L))
        y_shaped = y_[mask].reshape((L, L)).astype(bool)
        
        if top_l:
            kL = min(max(1, int(L)), len(y_shaped))
            L_idx = np.c_[np.unravel_index(np.argpartition(preds_shaped.ravel(), -kL)[-kL:], preds_shaped.shape)]
            preds_shaped = np.zeros_like(y_shaped)
            preds_shaped[L_idx[:, 0], L_idx[:, 1]] = True
        else:
            preds_shaped = xgb_contact.sigmoid(preds_shaped).astype(bool)
        
        np.save(os.path.join(save_dir, '%s_preds.npy' % pdb_ids[msa_idx]), preds_shaped)
        np.save(os.path.join(save_dir, '%s_target.npy' % pdb_ids[msa_idx]), y_shaped)


def store_top_l_prec_over_k_data(top_l_prec_dict_rel: Dict[float, np.ndarray], top_l_prec_list_abs: Dict[int, Dict[int, float]], pdb_ids: List[str], save_dir: str) -> None:
    """
    Creates and stores data for top-(k*L)-precision over k and (k*L), respectively.

    Args:
        top_l_prec_dict_rel (Dict[float, np.ndarray]): Metric values per k (relative).
        top_l_prec_list_abs (Dict[int, Dict[int, float]]): Metric values per k per MSA (absolute).
        pdb_ids (List[str]): List of PDB ids [B].
        save_dir (str): Directory, where data is stored.
    """

    x_rel = np.array([key for key in top_l_prec_dict_rel.keys()])
    sort_indices = np.argsort(x_rel)
    x_rel = x_rel[sort_indices]
    y_rel = np.array([val.mean() for val in top_l_prec_dict_rel.values()])[sort_indices]
    std_rel = np.array([val.std(ddof=1) for val in top_l_prec_dict_rel.values()])[sort_indices]
    
    np.save(os.path.join(save_dir, 'x_rel.npy'), x_rel)
    np.save(os.path.join(save_dir, 'y_rel.npy'), y_rel)
    np.save(os.path.join(save_dir, 'std_rel.npy'), std_rel)
    
    for idx, top_l_prec_dict in top_l_prec_list_abs.items():
        x_abs = np.array(list(top_l_prec_dict.keys()))
        y_abs = np.array(list(top_l_prec_dict.values()))
        
        np.save(os.path.join(save_dir, '%s_x_abs.npy' % pdb_ids[idx]), x_abs)
        np.save(os.path.join(save_dir, '%s_y_abs.npy' % pdb_ids[idx]), y_abs)


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
    parser.add_argument('--secondary-window', default=-1, type=int, help="window to ignore around secondary structure contacts. -1 means secondary contac will be predicted without special consideration.")
    # Test process
    parser.add_argument('--batch-size', default=1, type=int, help="Batch size (local in case of multi-gpu testing)")
    parser.add_argument('--min-k', default=0.01, type=float, help="Minimum coefficient k that is used in computing the top-(k*L)-precision.")
    parser.add_argument('--max-k', default=-1, type=float, help="Maximum coefficient k that is used in computing the top-(k*L)-precision. -1 refers to maximum L/2 of the longest sequence.")
    parser.add_argument('--num-k', default=500, type=int, help="Number of samples for k used in computing the top-(k*L)-precision. 1 disables top-(k*L)-precision over k plot and uses min-k as k.")
    parser.add_argument('--log-scale', action='store_true', help="Uses log-scale for coefficient k.")
    parser.add_argument('--treat-all-preds-positive', action='store_true', help="Whether all non-ignored preds are treated as positives, analogous to the CocoNet paper.")
    parser.add_argument('--individual-results', action='store_true', help="Whether to compute metric results also individually for each structure.")
    # Visualization
    parser.add_argument('--vis-dir', type=str, default='', help="Directory, where visualization data is stored. If empty, no data is stored.")
    parser.add_argument('--vis-contact-maps', action='store_true', help="Creates data for contact map plots.")
    parser.add_argument('--vis-top-l', action='store_true', help="Uses only top-L contacts for contact map data")
    parser.add_argument('--vis-k-plot', action='store_true', help="Creates data for top-(k*L)-precision over k plot.")
    # GPU
    parser.add_argument('--no-gpu', action='store_true', help="disables cuda")

    args = parser.parse_args()
    secondary_window = args.secondary_window

    if not args.no_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    h_params = xgb_contact.get_checkpoint_hparams(args.checkpoint, device)
    test_dl = xgb_contact.create_dataloader('test', args.batch_size, args.subsampling_mode, args.distance_threshold, h_params, secondary_window=secondary_window)

    cull_tokens = xgb_contact.get_cull_tokens(test_dl.dataset)
    model = xgb_contact.load_backbone(args.checkpoint, device, test_dl.dataset, cull_tokens, h_params)
    attn_maps, targets, msa_mapping, msa_mask, msa_mapping_filtered, L_mapping, pdb_ids = xgb_contact.compute_attn_maps(model, test_dl, cull_tokens, args.diag_shift, h_params, device)

    test_data = xgb.DMatrix(attn_maps, label=targets)

    xgb_model = xgb.Booster(model_file=args.xgboost_checkpoint)

    preds = xgb_model.predict(test_data, iteration_range=(0, xgb_model.best_iteration), strict_shape=True)[:, 0]

    if args.num_k == 1:
        top_l_prec_pos = xgb_contact.xgb_topkLPrec(preds, test_data, msa_mapping_filtered, L_mapping, args.min_k, treat_all_preds_positive=True)
        top_l_prec = xgb_contact.xgb_topkLPrec(preds, test_data, msa_mapping_filtered, L_mapping, args.min_k, treat_all_preds_positive=False)
        global_precision = xgb_contact.xgb_precision(preds, test_data, msa_mapping_filtered)
        global_recall = xgb_contact.xgb_recall(preds, test_data, msa_mapping_filtered)
        global_f1_score = xgb_contact.xgb_F1Score(preds, test_data, msa_mapping_filtered)
        matthews = xgb_contact.xgb_Matthews(preds, test_data, msa_mapping_filtered)

        print("Top-%sL-Positive-Precision:" % str(args.min_k), top_l_prec_pos)
        print("Top-%sL-Precision:" % str(args.min_k), top_l_prec)
        print("Global Precision:", global_precision)
        print("Global Recall:", global_recall)
        print("Global F1-Score:", global_f1_score)
        print("Global Matthews CorrCoeff:", matthews)
        
        if args.individual_results:
            top_l_prec = xgb_contact.xgb_topkLPrec(preds, test_data, msa_mapping_filtered, L_mapping, args.min_k, args.treat_all_preds_positive, reduce=False)
            global_precision = xgb_contact.xgb_precision(preds, test_data, msa_mapping_filtered, reduce=False)
            global_recall = xgb_contact.xgb_recall(preds, test_data, msa_mapping_filtered, reduce=False)
            global_f1_score = xgb_contact.xgb_F1Score(preds, test_data, msa_mapping_filtered, reduce=False)
            matthews = xgb_contact.xgb_Matthews(preds, test_data, msa_mapping_filtered, reduce=False)
            
            for idx, pdb_id in enumerate(pdb_ids):
                print("[%s] Top-%sL-Precision:" % (pdb_id, str(args.min_k)), top_l_prec[idx])
                print("[%s] Global Precision:" % pdb_id, global_precision[idx])
                print("[%s] Global Recall:" % pdb_id, global_recall[idx])
                print("[%s] Global F1-Score:" % pdb_id, global_f1_score[idx])
                print("[%s] Global Matthews CorrCoeff:" % pdb_id, matthews[idx])
                
    else:
        min_k = args.min_k
        if args.max_k == -1:
            max_k = max(L_mapping / 2)
        else:
            max_k = args.max_k
        
        if args.log_scale:
            min_k = np.log10(min_k)
            max_k = np.log10(max_k)
            k_range_rel = np.broadcast_to(np.logspace(min_k, max_k, args.num_k)[..., None], (args.num_k, len(test_dl)))  # [num_k, num_msa]
            min_k = 0
            max_k = np.log10(0.5*L_mapping**2)
            k_range_abs = np.logspace(min_k, max_k, args.num_k, dtype=int)  # [num_k, num_msa]
        else:
            k_range_rel = np.broadcast_to(np.linspace(min_k, max_k, args.num_k)[..., None], (args.num_k, len(test_dl)))  # [num_k, num_msa]
            k_range_abs = np.linspace(1, 0.5*L_mapping**2, args.num_k, dtype=int)  # [num_k, num_msa]
            
        top_l_prec_dict_rel = xgb_contact.xgb_topkLPrec_var_k(preds, test_data, msa_mapping_filtered, L_mapping, k_range_rel, treat_all_preds_positive=args.treat_all_preds_positive)
        top_l_prec_dict_abs = xgb_contact.xgb_topkLPrec_var_k(preds, test_data, msa_mapping_filtered, L_mapping, k_range_abs, relative_k=False, treat_all_preds_positive=args.treat_all_preds_positive)

        if args.vis_dir != '' and args.vis_k_plot:
            top_l_prec_plot_dir = os.path.join(args.vis_dir, 'top_l_prec_plots')
            os.makedirs(top_l_prec_plot_dir, exist_ok=True)
            store_top_l_prec_over_k_data(top_l_prec_dict_rel, top_l_prec_dict_abs, pdb_ids, top_l_prec_plot_dir)

    if args.vis_dir != '' and args.vis_contact_maps:
        contact_map_plot_dir = os.path.join(args.vis_dir, 'contact_maps')
        os.makedirs(contact_map_plot_dir, exist_ok=True)
        store_contact_maps_data(preds, test_data, msa_mapping, msa_mask, L_mapping, pdb_ids, args.vis_top_l, contact_map_plot_dir)


if __name__ == '__main__':
    main()
