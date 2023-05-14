# Individual Project

This is the main repository for my BEng dissertation at Imperial College titled
"Evaluating Contrastive and Non-Contrastive Learning for Medical Image
Classification".

In this repository, we provide a comprehensive set of frameworks written in
PyTorch Lightning to perform and evaluate self-supervised contrastive learning
using SimCLR on medical imaging data pipelined from the MedMNIST database.

## Setup

### Virtual environment

```bash
# Create virtual environment
$ python -m venv venv
# Activate on Linux, OS X
$ source venv/bin/activate
# Activate on Windows
$ source venv/Scripts/activate
# Check Python 3.10.9 is used. Some scripts may fail on Python 3.11
$ python
Python 3.10.9
>>> exit()
# Install requirements
$ pip install -r requirements.txt
```

## Guide

To perform SimCLR pretraining, navigate to `pretrain/simclr` directory and read
`README.md` for instructions. Then, to perform downstream transfer learning,
navigate to `downstream/resnet` or `downstream/logistic_regression` and read
`README.md`.

To perform baseline supervised learning, navigate to `downstream/resnet`.

## Contribute

### Update requirements

```bash
$ pip freeze > requirements.txt
```
