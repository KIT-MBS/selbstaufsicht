import glob
import subprocess

# Starts bash scripts for random-search-based hyperparameter optimization.

run_file_prefix = 'run_hparam_search'
rel_working_dir = 'profile/selbstaufsicht/scripts/'

for filename in glob.glob("%s*" % run_file_prefix):
    command = "sbatch %s" % filename
    subprocess.run(command, shell=True)
