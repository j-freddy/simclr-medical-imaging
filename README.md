# Individual Project

This is the main repository for my BEng dissertation at Imperial College titled
"Evaluating Contrastive and Non-Contrastive Learning for Medical Image
Classification".

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

## Pretrain

### SimCLR

**Train**

```bash
$ python -m pretrain.simclr.train
```

**TensorBoard**

```bash
$ tensorboard --logdir pretrain/simclr/models/tb_logs
```

## Downstream

### Logistic Regression

**Train**

```bash
$ python -m downstream.logistic_regression.train
```

**TensorBoard**

```bash
$ tensorboard --logdir downstream/logistic_regression/models/tb_logs
```

## Contribute

### Update requirements

```bash
$ pip freeze > requirements.txt
```
