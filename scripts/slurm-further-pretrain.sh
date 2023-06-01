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
# slurm-further-pretrain
# Further pretrain models for retina and dermatology with novel augmentations
# time needed: ~6 hrs
# ==============================================================================

python -m pretrain.simclr.train -c retinamnist -epochs 5000 -aug novel -fin pretrain-pathmnist -fout pretrain-path-retinamnist
python -m pretrain.simclr.train -c dermamnist -epochs 2000 -aug novel -fin pretrain-pathmnist -fout pretrain-path-dermamnist
