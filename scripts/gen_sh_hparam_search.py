import argparse
from collections import namedtuple
import math
import torch

# Generates bash scripts for random-search-based hyperparameter optimization.

# AUXILIARY TYPES AND FUNCTIONS
VarRange = namedtuple('VarRange', ['min', 'max', 'type'])
rescale_rand = lambda rand_val, min_val, max_val: min_val + (max_val - min_val) * rand_val 

# DEFINE PARSER
parser = argparse.ArgumentParser(description='Selbstaufsicht Random-Search Hyperparameter Optimization - Batch Generation Script')
# BASH SCRIPT GENERATION PARAMETERS
parser.add_argument('--num-scripts', default=1, type=int, help="Number of generated bash scripts")
parser.add_argument('--file-prefix', default='run_hparam_search', type=str, help="Prefix of generated bash scripts' filenames")
# BASH SCRIPT PARAMETERS
parser.add_argument('--ntasks', default=1, type=int, help="Number of tasks")
parser.add_argument('--nodes', default=1, type=int, help="Number of used nodes")
parser.add_argument('--cpus-per-task', default=16, type=int, help="Number of CPUs per task")
parser.add_argument('--mem', default=501600, type=int, help="Total CPU memory in MB")
parser.add_argument('--num-gpus', default=4, type=int, help="Number of used GPUs")
parser.add_argument('--time', default='2-00:00:00', type=str, help="Walltime (max 2d)")
parser.add_argument('--job-name', default='profile', type=str, help="Name of the job in SLURM")
parser.add_argument('--mail-type', default='BEGIN,END,FAIL', type=str, help="E-Mail triggers")
parser.add_argument('--mail-user', default='fabrice.lehr@dlr.de,oskar.taubert@kit.edu', type=str, help="E-Mail recipients")
parser.add_argument('--partition', default='accelerated', type=str, help="Used node type")
# CONSTANT TRAIN SCRIPT PARAMETERS
parser.add_argument('--dataset', default='combined', type=str, help="Used dataset: xfam, zwd, combined, dummy")
parser.add_argument('--num-data-samples', default=-1, type=int, help="Number of used samples from dataset. Non-positive numbers refer to using all data.")
parser.add_argument('--xfam-version', default='14.6', type=str, help="Xfam dataset version")
parser.add_argument('--xfam-mode', default='seed', type=str, help="Xfam dataset mode: seed, full, or enhanced")
parser.add_argument('--num-epochs', default=10000, type=int, help="Number of training epochs")
parser.add_argument('--batch-size', default=1, type=int, help="Batch size (local in case of multi-gpu training)")
parser.add_argument('--learning-rate-warmup', default=400, type=int, help="Warmup parameter for inverse square root rule of learning rate scheduling")
parser.add_argument('--precision', default=16, type=int, help="Precision used for computations")
parser.add_argument('--rng-seed', default=42, type=int, help="Random number generator seed")
parser.add_argument('--dp-strategy', default='zero-2', type=str, help="Data-parallelism strategy: ddp, zero-2, or zero-3. Note that DeepSpeed ZeRO requires precision=16.")
parser.add_argument('--dp-zero-bucket-size', default=500000000, type=int, help="Allocated bucket size for DeepSpeed ZeRO DP strategy.")
parser.add_argument('--num-workers', default=16, type=int, help="Number of data loader worker processes")
parser.add_argument('--subsampling-depth', default=100, type=int, help="Number of subsampled sequences")
parser.add_argument('--subsampling-mode', default='uniform', type=str, help="Subsampling mode: uniform, diversity, fixed")
parser.add_argument('--cropping-size', default=400, type=int, help="Maximum uncropped sequence length")
parser.add_argument('--cropping-mode', default='random-dependent', type=str, help="Cropping mode: random-dependent, random-independent, fixed")
parser.add_argument('--inpainting-masking-p', default=0.15, type=float, help="MSA masking ratio in the inpainting task")
parser.add_argument('--inpainting-loss-weight', default=5., type=float, help="Relative task loss weight. Is normalized before use.")
parser.add_argument('--jigsaw-permutations-max', default=30, type=int, help="Maximum number of permutations in the jigsaw task")
parser.add_argument('--jigsaw-loss-weight', default=1., type=float, help="Relative task loss weight. Is normalized before use.")
parser.add_argument('--contrastive-loss-weight', default=1., type=float, help="Relative task loss weight. Is normalized before use.")
parser.add_argument('--log-every', default=4000, type=int, help='how often to add logging rows(does not write to disk)')
parser.add_argument('--log-dir', default='lightning_logs/', type=str, help='Logging directory. Default: \"lightning_logs/\"')
parser.add_argument('--log-exp-name', default='hparam_random_search', type=str, help='Logging experiment name. If empty, this structure level is omitted. Default: \"hparam_random_search\"')
# VARIABLE TRAIN SCRIPT PARAMETERS
parser.add_argument('--num-blocks-min', default=4, type=int, help="Minumum number of consecutive Transmorpher blocks")
parser.add_argument('--num-blocks-max', default=12, type=int, help="Maximum number of consecutive Transmorpher blocks")
parser.add_argument('--feature-dim-head-min', default=32, type=int, help="Minimum size of the feature dimension per Transmorpher head")
parser.add_argument('--feature-dim-head-max', default=128, type=int, help="Maximum size of the feature dimension per Transmorpher head")
parser.add_argument('--num-heads-min', default=8, type=int, help="Minimum number of parallel Transmorpher heads")
parser.add_argument('--num-heads-max', default=16, type=int, help="Maximum number of parallel Transmorpher heads")
parser.add_argument('--learning-rate-min', default=1e-6, type=float, help="Minimum initial learning rate")
parser.add_argument('--learning-rate-max', default=1e-4, type=float, help="Maximum initial learning rate")
parser.add_argument('--dropout-min', default=0.1, type=float, help="Minimum dropout probability")
parser.add_argument('--dropout-max', default=0.5, type=float, help="Maximum dropout probability")
parser.add_argument('--inpainting-masking-types', default='token,column', type=str, help="MSA masking types in the inpainting task")
parser.add_argument('--jigsaw-partitions-min', default=3, type=int, help="Minimum number of partitions in the jigsaw task")
parser.add_argument('--jigsaw-partitions-max', default=5, type=int, help="Maximum number of partitions in the jigsaw task")
parser.add_argument('--contrastive-temperature-min', default=90, type=float, help="Minimum SimCLR temperature in the contrastive task")
parser.add_argument('--contrastive-temperature-max', default=110, type=float, help="Maximum SimCLR temperature in the contrastive task")
args = parser.parse_args()

# DEFINE CONSTANT PARAMETERS
constant_parameters = ['dataset', 
                       'num_data_samples', 
                       'xfam_version', 
                       'xfam_mode', 
                       'num_epochs', 
                       'batch_size',
                       'learning_rate_warmup',
                       'precision',
                       'rng_seed',
                       'dp_strategy',
                       'dp_zero_bucket_size',
                       'num_workers',
                       'subsampling_depth',
                       'subsampling_mode',
                       'cropping_size',
                       'cropping_mode',
                       'inpainting_masking_p',
                       'inpainting_loss_weight',
                       'jigsaw_loss_weight',
                       'contrastive_loss_weight',
                       'log_every',
                       'log_dir',
                       'log_exp_name']

# DEFINE VARIABLE RANGES
num_blocks_range = VarRange(args.num_blocks_min, args.num_blocks_max, int)
feature_dim_head_range = VarRange(args.feature_dim_head_min, args.feature_dim_head_max, int)
num_heads_range = VarRange(args.num_heads_min, args.num_heads_max, int)
learning_rate_range = VarRange(args.learning_rate_min, args.learning_rate_max, float)
dropout_range = VarRange(args.dropout_min, args.dropout_max, float)
task_range = ['inpainting', 'jigsaw', 'inpainting+jigsaw', 'inpainting+jigsaw+contrastive']
inpainting_masking_type_range = args.inpainting_masking_types.split(',')
jigsaw_partitions_range = VarRange(args.jigsaw_partitions_min, args.jigsaw_partitions_max, int)
contrastive_temperature_range = VarRange(args.contrastive_temperature_min, args.contrastive_temperature_max, float)
ranges = {'--num-blocks ': num_blocks_range, 
        '--feature-dim-head ': feature_dim_head_range, 
        '--num-heads ': num_heads_range, 
        '--learning-rate ': learning_rate_range, 
        '--dropout ': dropout_range, 
        '--task-': task_range, 
        '--inpainting-masking-type ': inpainting_masking_type_range, 
        '--jigsaw-partitions ': jigsaw_partitions_range, 
        '--contrastive-temperature ': contrastive_temperature_range}


def create_script():
    lines = []
    command_segments = []

    # DEFINE BASH SCRIPT LINES
    lines += [
            "#!/bin/bash\n",
            "#SBATCH --ntasks=%d\n" % args.ntasks,
            "#SBATCH --cpus-per-task=%d\n" % args.cpus_per_task,
            "#SBATCH --mem=%dmb\n" % args.mem,
            "#SBATCH --nodes=%d\n" % args.nodes,
            "#SBATCH --time=%s\n" % args.time,
            "#SBATCH --job-name=%s\n" % args.job_name,
            "#SBATCH --mail-type=%s\n" % args.mail_type,
            "#SBATCH --mail-user=%s\n" % args.mail_user,
            "#SBATCH --partition=%s\n" % args.partition,
            "#SBATCH --gres=gpu:%d\n" % args.num_gpus,
            "\n"
            "module load devel/cuda/11.1\n",
            "source ../../profile-venv/bin/activate\n",
            "echo $OMP_NUM_THREADS\n",
            "\n"]

    # DEFINE RUN COMMAND WITH CONSTANT PARAMETERS
    command_segments += ["srun python train.py --disable-emb-grad-freq-scale --disable-progress-bar "]
    for arg_key, arg_val in vars(args).items():
        if arg_key in constant_parameters:
            if type(arg_val) is float:
                arg_val = format(arg_val, '.6f')
            command_segments.append("--%s %s " % (arg_key.replace('_', '-'), str(arg_val)))

    # PROCESS VARIABLE RANGES, SAMPLE VARIABLES
    variables = dict.fromkeys(ranges)
    for arg_name, range_ in ranges.items():
        if isinstance(range_, tuple):
            min_val, max_val, type_val = range_
            if type_val is int:
                rand_val = torch.randint(min_val, max_val + 1, (1,)).item()
            elif type_val is float:
                rand_val = rescale_rand(torch.rand((1,)).item(), min_val, max_val)
            else:
                raise ValueError("Unexpected rand_val:", rand_val)
        elif isinstance(range_, list):
            rand_idx = torch.randint(0, len(range_), (1,)).item()
            rand_val = range_[rand_idx]
        else:
            raise ValueError("Unexpected type_val:", type_val)
        variables[arg_name] = rand_val
        if type(rand_val) is float:
                rand_val = format(rand_val, '.6f')
        val_split = str(rand_val).split('+')
        for val in val_split:
            command_segments.append("%s%s " % (arg_name, val))
    log_run_name = "t_%s__nb_%d__dh_%d__nh_%d__lr_%.6f__dr_%.1f__inp_%s__jig_%d__con_%.1f" % (variables['--task-'],
                                                                                            variables['--num-blocks '],
                                                                                            variables['--feature-dim-head '],
                                                                                            variables['--num-heads '],
                                                                                            variables['--learning-rate '],
                                                                                            variables['--dropout '],
                                                                                            variables['--inpainting-masking-type '],
                                                                                            variables['--jigsaw-partitions '],
                                                                                            variables['--contrastive-temperature '])
    log_run_name = log_run_name.replace('.', '-')
    command_segments.append("--log-run-name %s " % log_run_name)
    jigsaw_permutations = min(math.factorial(variables['--jigsaw-partitions ']), args.jigsaw_permutations_max)
    command_segments.append("--jigsaw-permutations %d" % jigsaw_permutations)

    # WRITE LINES TO DISK
    command = "".join(command_segments)
    lines.append(command)

    sh_filename = '%s__%s.sh' % (args.file_prefix, log_run_name)
        
    with open(sh_filename, "w") as filestream:
        filestream.writelines(lines)


if __name__ == '__main__':
    for _ in range(args.num_scripts):
        create_script()
