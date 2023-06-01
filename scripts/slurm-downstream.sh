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
# slurm-downstream
# Finetune pathmnist, dermamnist, bloodmnist (both frozen & unfrozen encoder)
# with 100, 250 1000 and 100% labelled images
# time needed: ~6 hrs
# ==============================================================================

# Frozen encoder

for samples in 100 250 1000
do
    python -m downstream.logistic_regression.train -c pathmnist -epochs 500 -samples $samples -fin pretrain-pathmnist -fout downstream-pathmnist-$samples-samples
    python -m downstream.logistic_regression.train -c dermamnist -epochs 500 -samples $samples -fin pretrain-dermamnist -fout downstream-dermamnist-$samples-samples
    python -m downstream.logistic_regression.train -c bloodmnist -epochs 500 -samples $samples -fin pretrain-bloodmnist -fout downstream-bloodmnist-$samples-samples
done

python -m downstream.logistic_regression.train -c pathmnist -epochs 500 -fin pretrain-pathmnist -fout downstream-pathmnist
python -m downstream.logistic_regression.train -c dermamnist -epochs 500 -fin pretrain-dermamnist -fout downstream-dermamnist
python -m downstream.logistic_regression.train -c bloodmnist -epochs 500 -fin pretrain-bloodmnist -fout downstream-bloodmnist

# Unfrozen encoder

for samples in 100 250 1000
do
    python -m downstream.resnet.train -c pathmnist -epochs 500 -samples $samples -fin pretrain-pathmnist -fout downstream-pathmnist-$samples-samples
    python -m downstream.resnet.train -c dermamnist -epochs 500 -samples $samples -fin pretrain-dermamnist -fout downstream-dermamnist-$samples-samples
    python -m downstream.resnet.train -c bloodmnist -epochs 500 -samples $samples -fin pretrain-bloodmnist -fout downstream-bloodmnist-$samples-samples
done

python -m downstream.resnet.train -c pathmnist -epochs 500 -fin pretrain-pathmnist -fout downstream-pathmnist
python -m downstream.resnet.train -c dermamnist -epochs 500 -fin pretrain-dermamnist -fout downstream-dermamnist
python -m downstream.resnet.train -c bloodmnist -epochs 500 -fin pretrain-bloodmnist -fout downstream-bloodmnist
