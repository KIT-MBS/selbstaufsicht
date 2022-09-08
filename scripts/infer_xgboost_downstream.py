import argparse
import os
from pathlib import Path
# from typing import Dict, List
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import xgboost as xgb

from selbstaufsicht.models.xgb import xgb_contact


def save_contact_maps(preds: np.ndarray, msa_mapping: np.ndarray, msa_mask: np.ndarray, L_mapping: np.ndarray, save_dir: str, file_names: List[str]) -> None:
    """
    Saves predictions.

    Args:
        preds (np.ndarray): Predictions [B] as logits.
        msa_mapping (np.ndarray): Mapping: Data point -> MSA index [B].
        msa_mask (np.ndarray): MSA mask [B].
        L_mapping (np.ndarray): Mapping: MSA index -> MSA L.
        save_dir (str): Directory, where plots are saved.
        file_names (List[str]): File names for plots.
    """

    msa_indices = np.unique(msa_mapping)

    preds_ = np.full(len(msa_mask), -np.inf)
    assert sum(msa_mask) == len(preds)
    preds_[msa_mask] = preds

    # for each MSA, save prediction
    for msa_idx in msa_indices:
        file_name = file_names[msa_idx]
        mask = msa_mapping == msa_idx  # [B]
        L = L_mapping[msa_idx]

        preds_shaped = xgb_contact.sigmoid(preds_[mask].reshape((L, L)))
        preds_shaped += preds_shaped.T
        preds_shaped_binary = np.round(preds_shaped).astype(bool)

        np.save(os.path.join(save_dir, '%s.npy' % file_name), preds_shaped)
        np.save(os.path.join(save_dir, '%s_binary.npy' % file_name), preds_shaped_binary)


def plot_contact_maps(preds: np.ndarray, msa_mapping: np.ndarray, msa_mask: np.ndarray, L_mapping: np.ndarray, save_dir: str, file_names: List[str]) -> None:
    """
    Plots predictions.

    Args:
        preds (np.ndarray): Predictions [B] as logits.
        msa_mapping (np.ndarray): Mapping: Data point -> MSA index [B].
        msa_mask (np.ndarray): MSA mask [B].
        L_mapping (np.ndarray): Mapping: MSA index -> MSA L.
        save_dir (str): Directory, where plots are saved.
        file_names (List[str]): File names for plots.
    """

    msa_indices = np.unique(msa_mapping)

    preds_ = np.full(len(msa_mask), -np.inf)
    assert sum(msa_mask) == len(preds)
    preds_[msa_mask] = preds

    # for each MSA, plot prediction
    for msa_idx in msa_indices:
        file_name = file_names[msa_idx]
        mask = msa_mapping == msa_idx  # [B]
        L = L_mapping[msa_idx]

        preds_shaped = xgb_contact.sigmoid(preds_[mask].reshape((L, L)))
        preds_shaped += preds_shaped.T
        preds_shaped_binary = np.round(preds_shaped).astype(bool)

        fig, ax = plt.subplots(1, 2)
        sns.heatmap(preds_shaped, fmt='', ax=ax[0])
        sns.heatmap(preds_shaped_binary, fmt='', ax=ax[1])

        ax[0].set_aspect('equal')
        ax[0].set_title("Prediction")
        ax[1].set_aspect('equal')
        ax[1].set_title("Prediction (binary)")

        fig.set_size_inches(10, 5)
        fig.suptitle("Inference Data: MSA %s" % file_name)
        fig.savefig(os.path.join(save_dir, '%s.pdf' % file_name))


def main():
    parser = argparse.ArgumentParser(description='Selbstaufsicht Weakly Supervised Contact Prediction Script (XGBoost version)')
    # I/O
    parser.add_argument('-i', '--input', type=str, nargs='+', help="Input FASTA file(s) containting RNA MSA(s).")
    parser.add_argument('-o', '--output', type=str, default='', help="Output directory for predictions. If none is provided, the directory of the first input file is used.")
    # Trained models
    parser.add_argument('--checkpoint', type=str, help="Path to downstream model checkpoint")
    parser.add_argument('--xgboost-checkpoint', type=str, help="Path to xgboost model checkpoint")
    # Contact prediction
    parser.add_argument('--distance-threshold', default=10., type=float, help="Minimum distance between two atoms in angström that is not considered as a contact")
    # Preprocessing
    parser.add_argument('--subsampling-mode', default='uniform', type=str, help="Subsampling mode: uniform, diversity, fixed")
    parser.add_argument('--diag-shift', default=4, type=int, help="Width of the area around the main diagonal of prediction maps that is ignored.")
    # Inference process
    parser.add_argument('--batch-size', default=1, type=int, help="Batch size (local in case of multi-gpu testing)")
    # Visualization
    parser.add_argument('--vis-dir', type=str, default='', help="Directory, where plots are saved. If empty, no plots are created.")
    # GPU
    parser.add_argument('--no-gpu', action='store_true', help="disables cuda")

    args = parser.parse_args()

    if not args.no_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    h_params = xgb_contact.get_checkpoint_hparams(args.checkpoint, device)
    inference_dl = xgb_contact.create_dataloader('inference', args.batch_size, args.subsampling_mode, args.distance_threshold, h_params, fasta_files=args.input)

    cull_tokens = xgb_contact.get_cull_tokens(inference_dl.dataset)
    model = xgb_contact.load_backbone(args.checkpoint, device, inference_dl.dataset, cull_tokens, h_params)
    attn_maps, _, msa_mapping, msa_mask, msa_mapping_filtered, L_mapping, _ = xgb_contact.compute_attn_maps(model, inference_dl, cull_tokens, args.diag_shift, h_params, device)

    inference_data = xgb.DMatrix(attn_maps, label=None)

    xgb_model = xgb.Booster(model_file=args.xgboost_checkpoint)

    preds = xgb_model.predict(inference_data, iteration_range=(0, xgb_model.best_iteration), strict_shape=True)[:, 0]

    file_names = [Path(fasta_file).stem for fasta_file in args.input]
    if args.output != '':
        save_dir = os.path.dirname(args.input[0])
    else:
        save_dir = args.output

    save_contact_maps(preds, msa_mapping, msa_mask, L_mapping, save_dir, file_names)

    if args.vis_dir != '':
        plot_contact_maps(preds, msa_mapping, msa_mask, L_mapping, args.vis_dir, file_names)


if __name__ == '__main__':
    main()
