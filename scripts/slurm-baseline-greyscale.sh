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
# slurm-baseline-greyscale
# Baseline models for tissuemnist, octmnist
# time needed: ~6 hrs
# ==============================================================================

for samples in 100 250 1000
do
    python -m downstream.resnet.train -c octmnist -epochs 500 -samples $samples -fout baseline-octmnist-$samples-samples
    python -m downstream.resnet.train -c tissuemnist -epochs 500 -samples $samples -fout baseline-tissuemnist-$samples-samples
done

python -m downstream.logistic_regression.train -c octmnist -epochs 500 -fout baseline-octmnist
python -m downstream.resnet.train -c tissuemnist -epochs 500 -fout baseline-tissuemnist
