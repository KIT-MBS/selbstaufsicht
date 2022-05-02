import argparse
from datetime import datetime
import random
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from selbstaufsicht import models
from selbstaufsicht import datasets
from selbstaufsicht.models.self_supervised.msa.utils import get_downstream_transforms, MSACollator, get_tasks, get_downstream_metrics


def main():
    parser = argparse.ArgumentParser(description='Selbstaufsicht Weakly Supervised Contact Prediction Script')
    # Trained models
    parser.add_argument('--checkpoint', type=str, help="Path to downstream model checkpoint")
    # Preprocessing
    parser.add_argument('--subsampling-mode', default='uniform', type=str, help="Subsampling mode: uniform, diversity, fixed")
    # Test process
    parser.add_argument('--batch-size', default=1, type=int, help="Batch size (local in case of multi-gpu testing)")
    parser.add_argument('--disable-progress-bar', action='store_true', help="disables the training progress bar")
    # GPU
    parser.add_argument('--no-gpu', action='store_true', help="disables cuda")

    args = parser.parse_args()

    if args.no_gpu:
        # NOTE for some reason, load_from_checkpoint fails to infer the hyperparameters correctly from the checkpoint file
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(args.checkpoint)
        
    h_params = checkpoint['hyper_parameters']
    
    downstream_transform = get_downstream_transforms(subsample_depth=h_params['subsampling_depth'], subsample_mode=args.subsampling_mode, threshold=h_params['downstream__distance_threshold'])
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
        
    _, _, test_metrics = get_downstream_metrics()
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
    task_heads['contact'] = models.self_supervised.msa.modules.ContactHead(h_params['num_blocks'] * h_params['num_heads'], cull_tokens=[test_dataset.token_mapping[l] for l in ['-', '.', 'START_TOKEN', 'DELIMITER_TOKEN']])
    task_losses['contact'] = nn.NLLLoss(weight=torch.tensor([1-h_params['downstream__loss_contact_weight'], h_params['downstream__loss_contact_weight']]), ignore_index=-1)

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
    model.tasks = ['contact']
    model.need_attn = True
    model.task_loss_weights = {'contact': 1.}
    model.test_metrics = test_metrics

    test_dl = DataLoader(test_dataset,
                         batch_size=args.batch_size,
                         shuffle=False,
                         num_workers=0,
                         pin_memory=False)
    tb_logger = TensorBoardLogger(save_dir=h_params['downstream__log_dir'], name=h_params['downstream__log_exp_name'], version=h_params['downstream__log_run_name'])
    trainer = Trainer(gpus=0 if args.no_gpu else 1,
                      logger=tb_logger,
                      enable_progress_bar=not args.disable_progress_bar)
    trainer.test(model, test_dl, verbose=True)

if __name__ == '__main__':
    main()
