#!/bin/bash -l

# here you specify the settings of the slurm job
# try to revise some of them, such as job_name, reservation time

#SBATCH -A uppmax2021-2-14 # project no.
#SBATCH -p core -n 4 # resource
#SBATCH -M snowy # cluster name
#SBATCH -t 48:00:00 # time
#SBATCH -J guj_baseline # job name
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#I have removed the two lines, for "short" jobs and reservation

# some references:
# snowy user guide: https://www.uppmax.uu.se/support/user-guides/snowy-user-guide/
# slurm user guide: https://uppmax.uu.se/support/user-guides/slurm-user-guide/

# note: remember to change the settings for your job, 
# both the SBATCH settings (at the beginning) and your experimental settings (below)

# now you start to write the code  

# load modules and set the environment
module load python/3.6.8
source /proj/uppmax2021-2-14/mt21/bin/activate

python train.py

# reset the working environment
deactivate

