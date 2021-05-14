#!/bin/bash
#SBATCH --job-name=pyPier        # create a short name for your job
#SBATCH --error=./slurm_out/slurm-%A.err        # error
#SBATCH --output=./slurm_out/slurm-%A.out       # output 
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=20:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=pierb@princeton.edu

module purge
module load anaconda3/2020.11
module load cudatoolkit/10.2 cudnn/cuda-10.2/7.6.5 rh/devtoolset/8
conda activate torch-env

make
