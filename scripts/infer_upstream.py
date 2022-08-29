import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from selbstaufsicht import models
from selbstaufsicht import datasets
from selbstaufsicht.models.self_supervised.msa.utils import get_downstream_transforms, get_tasks, get_downstream_metrics


def main():
    parser = argparse.ArgumentParser(description='Selbstaufsicht Weakly Supervised Contact Prediction Script')
    # Trained models
    parser.add_argument('--checkpoint', type=str, help="Path to model checkpoint")
    # Test process
    parser.add_argument('--batch-size', default=1, type=int, help="Batch size (local in case of multi-gpu testing)")
    parser.add_argument('--disable-progress-bar', action='store_true', help="disables the training progress bar")
    # GPU
    parser.add_argument('--no-gpu', action='store_true', help="disables cuda")

    args = parser.parse_args()
    secondary_window = args.secondary_window

    if args.no_gpu:
        # NOTE for some reason, load_from_checkpoint fails to infer the hyperparameters correctly from the checkpoint file
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(args.checkpoint)

    h_params = checkpoint['hyper_parameters']
    
    tasks = []
    if h_params['task_inpainting']:
        tasks.append("inpainting")
    if h_params['task_jigsaw']:
        tasks.append("jigsaw")
    if h_params['task_contrastive']:
        tasks.append("contrastive")
    if h_params['task_jigsaw_boot']:
        tasks.append("jigsaw_boot")
    
    actual_cropping_size = h_params['cropping_size'] - 1 - int(not h_params['jigsaw_disable_delimiter']) * (h_params['jigsaw_partitions'] + 1)
    transform, task_heads, task_losses, train_metrics, val_metrics = get_tasks(tasks,
                                                                               h_params['feature_dim_head'] * h_params['num_heads'],
                                                                               subsample_depth=h_params['subsampling_depth'],
                                                                               subsample_mode=h_params['subsampling_mode'],
                                                                               crop_size=actual_cropping_size,
                                                                               crop_mode=h_params['cropping_mode'],
                                                                               masking=h_params['inpainting_masking_type'],
                                                                               p_mask=h_params['inpainting_masking_p'],
                                                                               p_mask_static=h_params['inpainting_masking_p_static'],
                                                                               p_mask_nonstatic=h_params['inpainting_masking_p_nonstatic'],
                                                                               p_mask_unchanged=h_params['inpainting_masking_p_unchanged'],
                                                                               jigsaw_partitions=h_params['jigsaw_partitions'],
                                                                               jigsaw_classes=h_params['jigsaw_permutations'],
                                                                               jigsaw_linear=not h_params['jigsaw_nonlinear'],
                                                                               jigsaw_euclid_emb=None,
                                                                               jigsaw_delimiter=not h_params['jigsaw_disable_delimiter'],
                                                                               simclr_temperature=h_params['contrastive_temperature'],
                                                                               jigsaw_boot_ratio=h_params['jigsaw_boot_ratio'],
                                                                               per_token=h_params['boot_per_token'],
                                                                               boot_same=h_params['boot_same'],
                                                                               frozen=h_params['frozen'],
                                                                               seq_dist=h_params['seq_dist'])

    root = os.environ['DATA_PATH']
    dataset_name = h_params['dataset'].lower()
    
    downstream_transform = get_downstream_transforms(subsample_depth=h_params['subsampling_depth'], jigsaw_partitions=h_params['jigsaw_partitions'])
    downstream_ds = datasets.CoCoNetDataset(root, 'train', transform=downstream_transform)
    test_ds = datasets.CoCoNetDataset(root, 'val', transform=downstream_transform)
    exclude_ids = downstream_ds.fam_ids + test_ds.fam_ids

    if dataset_name == 'xfam':
        dataset_path = os.path.join(root, 'Xfam')
        ds = datasets.XfamDataset(dataset_path, download=True, transform=transform, mode=h_params['xfam_mode'], version=h_params['xfam_version'], exclude_ids=exclude_ids)
    elif dataset_name == 'zwd':
        dataset_path = os.path.join(root, 'zwd')
        ds = datasets.ZwdDataset(dataset_path, transform=transform)
    elif dataset_name == 'combined':
        xfam_path = os.path.join(root, 'Xfam')
        zwd_path = os.path.join(root, 'zwd')
        xfam_ds = datasets.XfamDataset(xfam_path, download=True, transform=transform, mode=h_params['xfam_mode'], version=h_params['xfam_version'], exclude_ids=exclude_ids)
        zwd_ds = datasets.ZwdDataset(zwd_path, transform=transform)
        ds = datasets.CombinedDataset(xfam_ds, zwd_ds)
        del xfam_ds
        del zwd_ds
    elif dataset_name == 'dummy':
        ds = datasets.DummyDataset(transform=transform)
    else:
        raise ValueError("Unknown dataset: %s" % h_params['dataset'])

    num_data_samples = h_params['num_data_samples'] if h_params['num_data_samples'] >= 0 else len(ds.samples)
    ds.num_data_samples = num_data_samples
    ds.jigsaw_force_permutations = h_params['jigsaw_force_permutations']
    
    dl = DataLoader(ds,
                    batch_size=h_params['batch_size'],
                    shuffle=False,
                    collate_fn=MSACollator(ds.token_mapping['PADDING_TOKEN'], frozen=h_params['frozen']),
                    num_workers=0,
                    pin_memory=False)

    model = models.self_supervised.MSAModel.load_from_checkpoint(
        checkpoint_path=args.checkpoint,
        num_blocks=h_params['num_blocks'],
        num_heads=h_params['num_heads'],
        feature_dim_head=h_params['feature_dim_head'],
        attention='fast-axial',
        task_heads=task_heads,
        task_losses=task_losses,
        alphabet_size=len(test_dataset.token_mapping),
        padding_token=test_dataset.token_mapping['PADDING_TOKEN'],
        dropout=0.,
        emb_grad_freq_scale=not h_params['disable_emb_grad_freq_scale'],
        max_seqlen=h_params['cropping_size'])
    model.test_metrics = val_metrics

    tb_logger = TensorBoardLogger(save_dir=h_params['log_dir'], name=h_params['log_exp_name'], version=h_params['log_run_name'] + "_inference")
    trainer = Trainer(gpus=0 if args.no_gpu else 1,
                      logger=tb_logger,
                      enable_progress_bar=not args.disable_progress_bar)
    trainer.test(model, dl, verbose=True)


if __name__ == '__main__':
    main()
