
# Generates bash scripts for grid-search-based hyperparameter optimization.

# BASH SCRIPT PARAMETERS
cpus_per_task = 14
mem_per_cpu_gb = 8
num_gpus = 1
num_nodes = 1
time = '12:00:00'
job_name = 'profile'
mail_type = 'BEGIN,END,FAIL'
mail_user = 'fabrice.lehr@dlr.de,oskar.taubert@kit.edu'
partition = 'haicore-gpu4'
gres = 'gpu:%d' % num_gpus

# TRAIN SCRIPT PARAMETERS
num_blocks_list = [4, 8, 12, 16]
num_heads_list = [8, 12, 16]
feature_dim_list = [512, 768, 1024]
batch_size_list = [1, 2, 4, 8]
lr_list = [1e-2]

num_workers = 1
precision = 32
num_epochs = 50
lr_warmup = 2000
xfam_version = '14.6'
xfam_mode = 'seed'

task_inpainting = True
task_jigsaw = False
task_contrastive = False

subsampling_depth = 4
subsampling_mode = 'uniform'
cropping_size = 100
inpainting_masking_type = 'token'
inpainting_masking_p = 0.15
jigsaw_partitions = 2
jigsaw_permutations = 2
contrastive_temperature = 100.

# OTHER PARAMETERS
log_dir = 'lightning_logs/'
log_run_name = '%d_%m_%Y__%H_%M_%S'
run_file_prefix = 'run_grid'
########################################

task_arg = ''
task_name = ''
if task_inpainting:
    task_arg += '--task-inpainting '
    task_name += 'inp_'
if task_jigsaw:
    task_arg += '--task-jigsaw '
    task_name += 'jig_'
if task_contrastive:
    task_arg += '--task-contrastive '
    task_name += 'con_'


def create_script(log_exp_name, num_blocks, num_heads, feature_dim, batch_size, lr):
    lines = [
        "#!/bin/bash\n",
        "#SBATCH --ntasks=1\n",
        "#SBATCH --cpus-per-task=%d\n" % cpus_per_task,
        "#SBATCH --mem-per-cpu=%dgb\n" % mem_per_cpu_gb,
        "#SBATCH --nodes=%d\n" % num_nodes,
        "#SBATCH --time=%s\n" % time,
        "#SBATCH --job-name=%s\n" % job_name,
        "#SBATCH --mail-type=%s\n" % mail_type,
        "#SBATCH --mail-user=%s\n" % mail_user,
        "#SBATCH --partition=%s\n" % partition,
        "#SBATCH --gres=%s\n" % gres,
        "\n"
        "module load devel/cuda/11.1\n",
        "\n",
        """\
srun python train.py --num-blocks %d --num-heads %d --feature-dim %d \
--xfam-version %s --xfam-mode %s --num-epochs %d --batch-size %d \
--learning-rate %.6f --learning-rate-warmup %d --precision %d \
--disable-progress-bar --num-gpus %d --num-nodes %d --num-workers %d %s\
--subsampling-depth %d --subsampling-mode %s --cropping-size %d \
--inpainting-masking-type %s --inpainting-masking-p %.2f --jigsaw-partitions %d \
--jigsaw-permutations %d --contrastive-temperature %.3f --log-dir %s \
--log-exp-name %s --log-run-name %s""" % (num_blocks, 
                                                  num_heads, 
                                                  feature_dim, 
                                                  xfam_version, 
                                                  xfam_mode,
                                                  num_epochs,
                                                  batch_size,
                                                  lr, 
                                                  lr_warmup, 
                                                  precision,
                                                  num_gpus,
                                                  num_nodes,
                                                  num_workers,
                                                  task_arg,
                                                  subsampling_depth,
                                                  subsampling_mode,
                                                  cropping_size,
                                                  inpainting_masking_type,
                                                  inpainting_masking_p,
                                                  jigsaw_partitions,
                                                  jigsaw_permutations,
                                                  contrastive_temperature,
                                                  log_dir,
                                                  log_exp_name,
                                                  log_run_name)
    ]
    
    sh_filename = '%s__%s.sh' % (run_file_prefix, log_exp_name)
    
    with open(sh_filename, "w") as filestream:
        filestream.writelines(lines)


def main():
    for num_blocks in num_blocks_list:
        for num_heads in num_heads_list:
            for feature_dim in feature_dim_list:
                for batch_size in batch_size_list:
                    eff_batch_size = batch_size * num_gpus * num_nodes
                    for lr in lr_list:
                        log_exp_name = ("%s_nb_%d__nh_%d__d_%d__bs_%d__lr_%.3f" % (task_name,
                                                                                  num_blocks, 
                                                                                  num_heads, 
                                                                                  feature_dim, 
                                                                                  eff_batch_size, 
                                                                                  lr)).replace('.', '_')
                        create_script(log_exp_name, num_blocks, num_heads, feature_dim, batch_size, lr)


if __name__ == '__main__':
    main()