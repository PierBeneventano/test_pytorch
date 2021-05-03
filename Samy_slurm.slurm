#!/bin/bash
#SBATCH --job-name=gan-lsun         # create a short name for your job
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=10          # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=8G                    # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1                # number of gpus per node
#SBATCH --time=01:30:00             # total run time limit (HH:MM:SS)





cd ~/dcgan/


SEED=$1
LR_G=$2
LR_D=$3
BATCH=$4
M_G=$5
M_D=$6
FA=$7
SA=$8
WD_G=${9}
WD_D=${10}
SAVE=${11}
OPT_CHOICE=${12}
ARCH=${13}
BETA2=${14}

python3 train_cifar.py --lr_G ${LR_G} \
  --lr_D ${LR_D} \
  --batch_size ${BATCH} \
  --momentum_G ${M_G}\
  --momentum_D ${M_D}\
  --weight_decay_G ${WD_G}\
  --weight_decay_D ${WD_D}\
  --time1 ${FA}\
  --time2 ${SA}\
  --save ${SAVE}\
  --seed ${SEED}\
  --optim_choice ${OPT_CHOICE}\
  --arch ${ARCH}\