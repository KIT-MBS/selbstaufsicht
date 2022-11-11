import argparse
from selbstaufsicht import datasets
from torch.utils.data import DataLoader
import os
from pathlib import Path
# from typing import Dict, List
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def main():
    parser = argparse.ArgumentParser(description='Selbstaufsicht Weakly Supervised Contact Prediction Script (XGBoost version)')
    # I/O
    parser.add_argument('-i', '--input', type=str, nargs='+', help="Input FASTA file(s) containting RNA MSA(s).")
    parser.add_argument('-o', '--output', type=str, default='', help="Output directory for predictions. If none is provided, the directory of the first input file is used.")
    # Trained models
    parser.add_argument('--checkpoint', type=str, help="Path to downstream model checkpoint")
    parser.add_argument('--xgboost-checkpoint', type=str, help="Path to xgboost model checkpoint")
    # Contact prediction
    parser.add_argument('--distance-threshold', default=10., type=float, help="Minimum distance between two atoms in angström that is not considered as a contact"
)
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

    num_gpus = args.num_gpus if args.num_gpus >= 0 else torch.cuda.device_count()
    if num_gpus * args.num_nodes > 1:
        dp_strategy = DDPPlugin(find_unused_parameters=True)
        # NOTE for some reason, load_from_checkpoint fails to infer the hyperparameters correctly from the checkpoint file
        checkpoint = torch.load(args.checkpoint)
    else:
        dp_strategy = None
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    h_params = checkpoint['hyper_parameters']

    dataset = datasets.InferenceDataset(fasta_files, transform=downstream_transform)

    inference_dl=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=False)

    #jigsaw_euclid_emb = None
    #if 'jigsaw_euclid_emb' in h_params and h_params['jigsaw_euclid_emb']:
    #    embed_size = checkpoint['state_dict']['task_heads.jigsaw.proj.weight'].size(0)
    #    jigsaw_euclid_emb = torch.empty((1, embed_size))
    #else:
    #    jigsaw_euclid_emb = None

    #if 'jigsaw_disable_delimiter' in h_params:
    #    jigsaw_delimiter = not h_params['jigsaw_disable_delimiter']
    #else:
    #    jigsaw_delimiter = True

    #tasks = []
    #if h_params['task_inpainting']:
    #    tasks.append("inpainting")
    #if h_params['task_jigsaw']:
    #    tasks.append("jigsaw")
    #if h_params['task_contrastive']:
    #    tasks.append("contrastive")
    #if h_params['task_jigsaw_boot']:
    #    tasks.append("jigsaw_boot")

    #_, task_heads, task_losses, _, _ = get_tasks(tasks,
     #                                            h_params['feature_dim_head'] * h_params['num_heads'],
     #                                            subsample_depth=h_params['subsampling_depth'],
     #                                            subsample_mode=h_params['subsampling_mode'],
     #                                            crop_size=h_params['cropping_size'],
     #                                            crop_mode=h_params['cropping_mode'],
     #                                            masking=h_params['inpainting_masking_type'],
     #                                            p_mask=h_params['inpainting_masking_p'],
     #                                            jigsaw_partitions=h_params['jigsaw_partitions'],
     #                                            jigsaw_classes=h_params['jigsaw_permutations'],
     #                                            jigsaw_linear=not h_params['jigsaw_nonlinear'],
     #                                            jigsaw_delimiter=jigsaw_delimiter,
     #                                            jigsaw_euclid_emb=jigsaw_euclid_emb,
     #                                            simclr_temperature=h_params['contrastive_temperature'],
     #                                            jigsaw_boot_ratio=h_params['jigsaw_boot_ratio'],
     #                                            per_token=h_params['boot_per_token'],
     #                                            boot_same=h_params['boot_same'],
     #                                            frozen=h_params['frozen'],
     #                                            seq_dist=h_params['seq_dist'],
     #                                            )

#    num_maps = h_params['num_blocks'] * h_params['num_heads']
    #if 'downstream' in checkpoint:
    #    task_heads['jigsaw'] = models.self_supervised.msa.modules.JigsawHead(12*64,1)
    #    task_losses['jigsaw'] = None

    #model = models.self_supervised.MSAModel.load_from_checkpoint(
    #    checkpoint_path=checkpoint,
    #    num_blocks=h_params['num_blocks'],
    #    num_heads=h_params['num_heads'],
    #    feature_dim_head=h_params['feature_dim_head'],
    #    task_heads=task_heads,
    #    task_losses=task_losses,
    #    alphabet_size=len(dataset.token_mapping),
    #    padding_token=dataset.token_mapping['PADDING_TOKEN'],
    #    lr=h_params['learning_rate'],
    #    lr_warmup=h_params['learning_rate_warmup'],
    #    dropout=0.,
    #    emb_grad_freq_scale=not h_params['disable_emb_grad_freq_scale'],
    #    freeze_backbone=True,
    #    max_seqlen=h_params['cropping_size'],
    #    h_params=h_params)
    #model.need_attn = True
    #model.to(device)
