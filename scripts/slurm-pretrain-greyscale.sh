#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL   # required to send email notifcations
#SBATCH --mail-user=ffj20 # required to send email notifcations

. /vol/cuda/11.7.1/setup.sh
export LD_LIBRARY_PATH=/vol/bitbucket/${USER}/individual-project-refined/venv/lib/python3.10/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH}
TERM=vt100                # or TERM=xterm
/usr/bin/nvidia-smi
uptime

cd individual-project-refined
export PATH=/vol/bitbucket/${USER}/individual-project-refined/venv/bin/:$PATH
source activate

# ==============================================================================
# slurm-pretrain-greyscale
# Pretrain octmnist, tissuemnist with novel greyscale augmentations
# time needed: ~12 hrs
# ==============================================================================

python -m pretrain.simclr.train -c octmnist -epochs 200 -aug greyscale -fout pretrain-octmnist
python -m pretrain.simclr.train -c tissuemnist -epochs 200 -aug greyscale -fout pretrain-tissuemnist
