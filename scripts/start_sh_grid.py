import glob
import os
import subprocess

# Starts bash scripts for grid-search-based hyperparameter optimization.

run_file_prefix = 'run_grid'
rel_working_dir = 'profile/selbstaufsicht/scripts/'
home = os.environ['HOME']

abs_working_dir = os.path.join(home, rel_working_dir)

for filename in glob.glob("%s*" % run_file_prefix):
    command = "sbatch %s" % filename
    subprocess.run(command, cwd=abs_working_dir)