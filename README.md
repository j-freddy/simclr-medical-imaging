# Individual Project

This is the main repository for my BEng dissertation at Imperial College titled
"Evaluating Contrastive and Non-Contrastive Learning for Medical Image
Classification".

In this repository, we provide a comprehensive set of frameworks written in
PyTorch Lightning to perform and evaluate self-supervised contrastive learning
using SimCLR on medical imaging data pipelined from the MedMNIST database.

## Background

A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)
is a contrastive learning method that aims to learn useful representations of
images through training a convolutional neural network (CNN) to recognise
similarities between a pair of augmented data points derived from the same input
image. The idea is that the network may learn to extract useful, generalised
features that can be used in downstream tasks.

We use ResNet-18 as the CNN architecture.

Original SimCLR papers:
- [A Simple Framework for Contrastive Learning of Visual Representations][1]
```bibtex
@inproceedings{chen2020simple,
  title={A simple framework for contrastive learning of visual representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  booktitle={International conference on machine learning},
  pages={1597--1607},
  year={2020},
  organization={PMLR}
}
```
- [Big Self-Supervised Models are Strong Semi-Supervised Learners][2]
```bibtex
@article{chen2020big,
  title={Big self-supervised models are strong semi-supervised learners},
  author={Chen, Ting and Kornblith, Simon and Swersky, Kevin and Norouzi, Mohammad and Hinton, Geoffrey E},
  journal={Advances in neural information processing systems},
  volume={33},
  pages={22243--22255},
  year={2020}
}
```

[1]: https://arxiv.org/pdf/2002.05709.pdf
[2]: https://arxiv.org/pdf/2006.10029.pdf

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

Regardless of the environment, all programs search for models (i.e. `.ckpt`
files) in `models/`. For example, when performing downstream learning, the
program searches for the pretrained file in `pretrain/simclr/models/`. If you
place the model in a different folder, you need to update `MODEL_DIR` in
`utils.py`.

## Contribute

### Update requirements

```bash
$ pip freeze > requirements.txt
```

## Credits

We source medical images from [MedMNIST](https://medmnist.com/).

```bibtex
@article{medmnistv2,
    title={MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification},
    author={Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Zequan and Zhao, Lin and Ke, Bilian and Pfister, Hanspeter and Ni, Bingbing},
    journal={Scientific Data},
    volume={10},
    number={1},
    pages={41},
    year={2023},
    publisher={Nature Publishing Group UK London}
}
```
