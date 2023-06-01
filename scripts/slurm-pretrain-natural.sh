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
# slurm-pretrain-natural
# Pretrain pathmnist, dermamnist, bloodmnist with natural augmentations
# time needed: ~12 hrs
# ==============================================================================

python -m pretrain.simclr.train -c pathmnist -epochs 201 -aug natural -fout pretrain-pathmnist
python -m pretrain.simclr.train -c dermamnist -epochs 2000 -aug natural -fout pretrain-dermamnist
python -m pretrain.simclr.train -c bloodmnist -epochs 2000 -aug natural -fout pretrain-bloodmnist
