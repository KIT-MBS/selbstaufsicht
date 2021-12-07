import argparse
from datetime import datetime
from functools import partial
import numpy as np
import os
import random

import torch
from torch.utils.data import DataLoader, Subset

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

from selbstaufsicht.utils import data_loader_worker_init
from selbstaufsicht import models
from selbstaufsicht import datasets
from selbstaufsicht.models.self_supervised.msa.utils import get_tasks, MSACollator


parser = argparse.ArgumentParser(description='Selbstaufsicht Training Script')
# Architecture
parser.add_argument('--num-blocks', default=2, type=int, help="Number of consecutive Transmorpher blocks")
parser.add_argument('--feature-dim', default=768, type=int, help="Size of the feature dimension")
parser.add_argument('--num-heads', default=12, type=int, help="Number of parallel Transmorpher heads")
# Dataset
parser.add_argument('--dataset', default='xfam', type=str, help="Used dataset: xfam, dummy")
parser.add_argument('--num-data-samples', default=-1, type=int, help="Number of used samples from dataset. Non-positive numbers refer to using all data.")
parser.add_argument('--xfam-version', default='9.1', type=str, help="Xfam dataset version")
parser.add_argument('--xfam-mode', default='seed', type=str, help="Xfam dataset mode: seed, full, or enhanced")
# Training process
parser.add_argument('--num-epochs', default=20, type=int, help="Number of training epochs")
parser.add_argument('--batch-size', default=2, type=int, help="Batch size (local in case of multi-gpu training)")
parser.add_argument('--learning-rate', default=1e-4, type=float, help="Initial learning rate")
parser.add_argument('--learning-rate-warmup', default=2000, type=int, help="Warmup parameter for inverse square root rule of learning rate scheduling")
parser.add_argument('--precision', default=32, type=int, help="Precision used for computations")
parser.add_argument('--disable-progress-bar', action='store_true', help="disables the training progress bar")
parser.add_argument('--rng-seed', default=42, type=int, help="Random number generator seed")
# Data parallelism
parser.add_argument('--num-gpus', default=1, type=int, help="Number of GPUs per node")
parser.add_argument('--num-nodes', default=1, type=int, help="Number of nodes")
parser.add_argument('--num-workers', default=1, type=int, help="Number of data loader worker processes")
# Upstream tasks
parser.add_argument('--task-inpainting', action='store_true', help="Activates the inpainting task")
parser.add_argument('--task-jigsaw', action='store_true', help="Activates the jigsaw task")
parser.add_argument('--task-contrastive', action='store_true', help="Activates the contrastive task")
# Upstream task configuration
parser.add_argument('--subsampling-depth', default=4, type=int, help="Number of subsampled sequences")
parser.add_argument('--subsampling-mode', default='uniform', type=str, help="Subsampling mode: uniform, diversity")
parser.add_argument('--cropping-size', default=50, type=int, help="Maximum uncropped sequence length")
parser.add_argument('--inpainting-masking-type', default='token', type=str, help="MSA masking type in the inpainting task")
parser.add_argument('--inpainting-masking-p', default=0.15, type=float, help="MSA masking ratio in the inpainting task")
parser.add_argument('--jigsaw-partitions', default=3, type=int, help="Number of partitions in the jigsaw task")
parser.add_argument('--jigsaw-permutations', default=4, type=int, help="Number of permutations in the jigsaw task")
parser.add_argument('--contrastive-temperature', default=100., type=float, help="SimCLR temperature in the contrastive task")
# Logging
parser.add_argument('--log-every', default=50, type=int, help='how often to add logging rows(does not write to disk)')
parser.add_argument('--log-dir', default='lightning_logs/', type=str, help='Logging directory. Default: \"lightning_logs/\"')
parser.add_argument('--log-exp-name', default='', type=str, help='Logging experiment name. If empty, this structure level is omitted. Default: \"\"')
parser.add_argument('--log-run-name', default='%d_%m_%Y__%H_%M_%S', type=str, help='Logging run name. Supports 1989 C standard datetime codes. Default: \"%%d_%%m_%%Y__%%H_%%M_%%S\"')


args = parser.parse_args()
dt_now = datetime.now()
log_run_name = dt_now.strftime(args.log_run_name)

d_head = args.feature_dim // args.num_heads
assert d_head * args.num_heads == args.feature_dim

tasks = []
if args.task_inpainting:
    tasks.append("inpainting")
if args.task_jigsaw:
    tasks.append("jigsaw")
if args.task_contrastive:
    tasks.append("contrastive")

torch.manual_seed(args.rng_seed)
np.random.seed(args.rng_seed)
random.seed(args.rng_seed)
data_loader_rng = torch.Generator()
data_loader_rng.manual_seed(args.rng_seed)

if args.num_gpus * args.num_nodes > 1:
    dp_strategy = DDPPlugin(find_unused_parameters=False)
else:
    dp_strategy = None

# TODO should take token mapping?
transform, task_heads, task_losses, metrics = get_tasks(tasks,
                                                        args.feature_dim,
                                                        subsample_depth=args.subsampling_depth,
                                                        subsample_mode=args.subsampling_mode,
                                                        crop=args.cropping_size,
                                                        masking=args.inpainting_masking_type,
                                                        p_mask=args.inpainting_masking_p,
                                                        jigsaw_partitions=args.jigsaw_partitions,
                                                        jigsaw_classes=args.jigsaw_permutations,
                                                        simclr_temperature=args.contrastive_temperature)

root = os.environ['DATA_PATH']
dataset_name = args.dataset.lower()
# NOTE MSA transformer: num_layers=12, d=768, num_heads=12, batch_size=512, lr=10**-4, **-2 lr schedule, 32 V100 GPUs for 100k updates, finetune for 25k more

if dataset_name == 'xfam':
    dataset_path = os.path.join(root, 'Xfam')
    ds = datasets.Xfam(dataset_path, download=True, transform=transform, mode=args.xfam_mode, version=args.xfam_version)
elif dataset_name == 'dummy':
    ds = datasets.Dummy(transform=transform)

else:
    raise ValueError("Unknown dataset: %s" % args.dataset)

if args.num_data_samples > 0:
    tm = ds.token_mapping
    ds = Subset(ds, torch.arange(args.num_data_samples))
    ds.token_mapping = tm

dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=MSACollator(ds.token_mapping['PADDING_TOKEN']), num_workers=args.num_workers,
                worker_init_fn=partial(data_loader_worker_init, rng_seed=args.rng_seed), generator=data_loader_rng, pin_memory=True)
# TODO should pass padding token index here
model = models.self_supervised.MSAModel(
    args.num_blocks,
    args.num_heads,
    d_head,
    aux_input_dim=2,
    task_heads=task_heads,
    task_losses=task_losses,
    metrics=metrics,
    in_dict_size=len(ds.token_mapping), padding_token=ds.token_mapping['PADDING_TOKEN'],
    lr=args.learning_rate,
    lr_warmup=args.learning_rate_warmup
)
tb_logger = TensorBoardLogger(save_dir=args.log_dir, name=args.log_exp_name, version=log_run_name)
trainer = Trainer(max_epochs=args.num_epochs,
                  gpus=args.num_gpus,
                  num_nodes=args.num_nodes,
                  precision=args.precision,
                  accelerator="gpu",
                  strategy=dp_strategy,
                  enable_progress_bar=not args.disable_progress_bar,
                  log_every_n_steps=args.log_every,
                  logger=tb_logger)
trainer.fit(model, dl)
