import os
from datetime import datetime
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from selbstaufsicht import models
from selbstaufsicht import datasets
from selbstaufsicht.models.self_supervised.msa.utils import get_downstream_transforms, MSACollator, get_tasks


def main():
    parser = argparse.ArgumentParser(description='Selbstaufsicht Weakly Supervised Contact Prediction Script')
    parser.add_argument('--checkpoint', type=str, help="Path to pre-trained model checkpoint")
    parser.add_argument('--log-run-name', default='%d_%m_%Y__%H_%M_%S', type=str,
                        help='Logging run name. Supports 1989 C standard datetime codes. Default: \"%%d_%%m_%%Y__%%H_%%M_%%S\"')

    args = parser.parse_args()

    root = os.environ['DATA_PATH']
    downstream_transform = get_downstream_transforms(subsample_depth=50)
    downstream_ds = datasets.CoCoNetDataset(root, 'train', transform=downstream_transform)
    test_ds = datasets.CoCoNetDataset(root, 'val', transform=downstream_transform)

    # NOTE for some reason, load_from_checkpoint fails to infer the hyperparameters correctly from the checkpoint file
    checkpoint = torch.load(args.checkpoint)
    h_params = checkpoint['hyper_parameters']
    learning_rate = 0.0001

    tasks = []
    if h_params['task_inpainting']:
        tasks.append("inpainting")
    if h_params['task_jigsaw']:
        tasks.append("jigsaw")
    if h_params['task_contrastive']:
        tasks.append("contrastive")
    _, task_heads, task_losses, _ = get_tasks(tasks,
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
                                    simclr_temperature=h_params['contrastive_temperature'])

    model = models.self_supervised.MSAModel.load_from_checkpoint(
            checkpoint_path = args.checkpoint,
            num_blocks = h_params['num_blocks'],
            num_heads = h_params['num_heads'],
            feature_dim_head = h_params['feature_dim_head'],
            task_heads=task_heads,
            task_losses=task_losses,
            alphabet_size=len(downstream_ds.token_mapping),
            padding_token=downstream_ds.token_mapping['PADDING_TOKEN'],
            lr=learning_rate,
            lr_warmup=1,
            dropout=0.,
            emb_grad_freq_scale=not h_params['disable_emb_grad_freq_scale'],
            )
    model.tasks = ['contact']
    model.losses['contact'] = nn.CrossEntropyLoss(ignore_index=-1)  # TODO there probably should be a weight for contacts
    model.task_heads['contact'] = models.self_supervised.msa.modules.ContactHead(h_params['num_blocks'] * h_params['num_heads'], cull_tokens=[downstream_ds.token_mapping[l] for l in ['-', '.', 'START_TOKEN', 'DELIMITER_TOKEN']])
    model.need_attn = True
    model.task_loss_weights = {'contact': 1.}
    # TODO accuracy
    model.metrics = {'contact': []}

    train_dl = DataLoader(downstream_ds, batch_size=1, shuffle=True)

    dt_now = datetime.now()
    log_run_name = 'downstream_' + dt_now.strftime(args.log_run_name)
    tb_logger = TensorBoardLogger(save_dir=h_params['log_dir'], name=h_params['log_exp_name'], version=log_run_name)
    trainer = Trainer()
    trainer.fit(model, train_dl)

if __name__ == '__main__':
    main()
