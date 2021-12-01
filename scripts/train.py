import argparse
import os
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin

from selbstaufsicht import models
from selbstaufsicht import datasets
from selbstaufsicht.models.self_supervised.msa.utils import get_tasks, MSACollator


parser = argparse.ArgumentParser(description='Selbstaufsicht Training Script')
# Architecture
parser.add_argument('--num-blocks', default=2, type=int, help="Number of consecutive Transmorpher blocks")
parser.add_argument('--feature-dim', default=768, type=int, help="Size of the feature dimension")
parser.add_argument('--num-heads', default=12, type=int, help="Number of parallel Transmorpher heads")
# Dataset
parser.add_argument('--xfam-version', default='9.1', type=str, help="Xfam dataset version")
parser.add_argument('--num-data-samples', default=-1, type=int, help="Number of used samples from dataset. Negative numbers refer to using all data.")
# Training process
parser.add_argument('--num-epochs', default=20, type=int, help="Number of training epochs")
parser.add_argument('--batch-size', default=2, type=int, help="Batch size (local in case of multi-gpu training)")
parser.add_argument('--learning-rate', default=1e-4, type=float, help="Initial learning rate")
parser.add_argument('--learning-rate-warmup', default=16000, type=int, help="Warmup parameter for inverse square root rule of learning rate scheduling")
parser.add_argument('--precision', default=32, type=int, help="Precision used for computations")
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
parser.add_argument('--subsampling-mode', default='uniform', type=str, help="Subsampling mode (uniform, diversity)")
parser.add_argument('--cropping-size', default=50, type=int, help="Maximum uncropped sequence length")
parser.add_argument('--inpainting-masking-type', default='token', type=str, help="MSA masking type in the inpainting task")
parser.add_argument('--inpainting-masking-p', default=0.15, type=float, help="MSA masking ratio in the inpainting task")
parser.add_argument('--jigsaw-partitions', default=3, type=int, help="Number of partitions in the jigsaw task")
parser.add_argument('--jigsaw-permutations', default=4, type=int, help="Number of permutations in the jigsaw task")
parser.add_argument('--contrastive-temperature', default=100., type=float, help="SimCLR temperature in the contrastive task")

args = parser.parse_args()

d_head = args.feature_dim // args.num_heads

tasks = []
if args.task_inpainting:
    tasks.append("inpainting")
if args.task_jigsaw:
    tasks.append("jigsaw")
if args.task_contrastive:
    tasks.append("contrastive")

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

root = os.environ['DATA_PATH'] + 'Xfam'
# NOTE MSA transformer: num_layers=12, d=768, num_heads=12, batch_size=512, lr=10**-4, **-2 lr schedule, 32 V100 GPUs for 100k updates, finetune for 25k more
ds = datasets.Xfam(root, download=True, transform=transform, version=args.xfam_version, debug_size=args.num_data_samples)
dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=MSACollator(ds.token_mapping['PADDING_TOKEN']), num_workers=args.num_workers, pin_memory=True)
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
trainer = Trainer(max_epochs=args.num_epochs, gpus=args.num_gpus, num_nodes=args.num_nodes, precision=args.precision, accelerator="gpu", strategy=dp_strategy)
trainer.fit(model, dl)
