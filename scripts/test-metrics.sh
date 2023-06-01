#!/bin/bash

# ==============================================================================
# Example snapshot of quantitative testing script
# time needed: ~30 minutes
# ==============================================================================

# python -m downstream.resnet.test -c dermamnist -fin downstream-dermamnist-10-spc-path-derma >> logs.txt
# python -m downstream.resnet.test -c dermamnist -fin downstream-dermamnist-25-spc-path-derma >> logs.txt
# python -m downstream.resnet.test -c dermamnist -fin downstream-dermamnist-100-spc-path-derma >> logs.txt

# python -m downstream.resnet.test -c dermamnist -fin downstream-dermamnist-10-spc-path >> logs.txt
# python -m downstream.resnet.test -c dermamnist -fin downstream-dermamnist-25-spc-path >> logs.txt
# python -m downstream.resnet.test -c dermamnist -fin downstream-dermamnist-100-spc-path >> logs.txt

# python -m downstream.logistic_regression.test -c dermamnist -fencoder pretrain-path-dermamnist -fin downstream-dermamnist-10-spc-path-derma >> logs.txt
# python -m downstream.logistic_regression.test -c dermamnist -fencoder pretrain-path-dermamnist -fin downstream-dermamnist-25-spc-path-derma >> logs.txt
# python -m downstream.logistic_regression.test -c dermamnist -fencoder pretrain-path-dermamnist -fin downstream-dermamnist-100-spc-path-derma >> logs.txt

# python -m downstream.logistic_regression.test -c dermamnist -fencoder pretrain-pathmnist -fin downstream-dermamnist-10-spc-path >> logs.txt
# python -m downstream.logistic_regression.test -c dermamnist -fencoder pretrain-pathmnist -fin downstream-dermamnist-25-spc-path >> logs.txt
# python -m downstream.logistic_regression.test -c dermamnist -fencoder pretrain-pathmnist -fin downstream-dermamnist-100-spc-path >> logs.txt

python -m downstream.resnet.test -c retinamnist -fin baseline-retinamnist-100-samples >> logs.txt
python -m downstream.resnet.test -c retinamnist -fin baseline-retinamnist-250-samples >> logs.txt
python -m downstream.resnet.test -c retinamnist -fin baseline-retinamnist-1000-samples >> logs.txt

python -m downstream.resnet.test -c tissuemnist -fin baseline-tissuemnist-100-samples >> logs.txt
python -m downstream.resnet.test -c tissuemnist -fin baseline-tissuemnist-250-samples >> logs.txt
python -m downstream.resnet.test -c tissuemnist -fin baseline-tissuemnist-1000-samples >> logs.txt
python -m downstream.resnet.test -c tissuemnist -fin baseline-tissuemnist >> logs.txt

python -m downstream.resnet.test -c octmnist -fin baseline-octmnist-100-samples >> logs.txt
python -m downstream.resnet.test -c octmnist -fin baseline-octmnist-250-samples >> logs.txt
python -m downstream.resnet.test -c octmnist -fin baseline-octmnist-1000-samples >> logs.txt
python -m downstream.resnet.test -c octmnist -fin baseline-octmnist >> logs.txt
