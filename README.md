# Evaluating SimCLR for Medical Image Classification

This repository contains the codebase for the experiments conducted and
published in the paper "Evaluating SimCLR for Medical Image Classification" as
part of my final year individual research project at Imperial College
London (BEng JMC 2023).

In this repository, we provide a comprehensive set of frameworks written in
PyTorch Lightning to perform and evaluate self-supervised contrastive learning
using SimCLR on medical imaging data pipelined from the MedMNIST database.

Abstract:
> Computer-aided diagnosis (CADx) plays a crucial role in assisting radiologists
  with interpreting medical images. Over recent years, there has been
  significant advancements in image classification models, such as deep neural
  networks and Vision Transformers. Training such models require lots of
  labelled data, a prerequisite often not met in medical environments as
  labelling images is time-consuming and requires expertise.<br><br>
  An alternative training paradigm is self-supervised learning, which involves
  pretraining a model with unlabelled data followed by finetuning it with
  labelled data. This paradigm has achieved strong performance on classifying
  natural images, even with limited labelled data.<br><br>
  This thesis aims to explore the potential of SimCLR, a state-of-the-art
  self-supervised learning framework, for medical image classification. We
  evaluate this framework on a wide range of medical imaging modalities,
  including colon pathology, dermatology, blood cells, retina fundus and other
  medical scans. We find significant improvement over baseline supervised
  metrics (an increase of up to 30.6% in accuracy). We simulate different data
  settings and explore tackling class imbalance, as well as transfer learning on
  different datasets. We find downsampling images to be a viable solution for
  some modalities in bringing down training times (12 hours to pretrain a model
  for classifying blood cells that achieves over 0.95 AUC after finetuning). We
  propose a novel augmentation sequence which shows consistent improvement over
  the original framework.

## Background

A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)
is a state-of-the-art contrastive learning method that aims to learn useful
representations of images through training a convolutional neural network (the
codebase uses ResNet-18) to recognise similarities between a pair of augmented
data points derived from the same input image. The idea is that the network may
learn to extract useful, generalisable features that can be used for downstream
tasks.

Original SimCLR papers:
- [A Simple Framework for Contrastive Learning of Visual Representations][simclr]
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
- [Big Self-Supervised Models are Strong Semi-Supervised Learners][simclrv2]
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

[simclr]: https://arxiv.org/pdf/2002.05709.pdf
[simclrv2]: https://arxiv.org/pdf/2006.10029.pdf

<!-- Contributions -->
<!--
## Contributions

- how well does a SimCLR setup that works well for natural images transfer to medical images?
- 4 augmentation sequences (list them out)
- lack of data
- unbalanced dataset
- evaluation metrics & representations
-->

## Usage guide

### Installation

1. Clone this repository.
```bash
git clone https://github.com/j-freddy/simclr-medical-imaging
```

2. Create virtual environment with Python 3.10.9. Some scripts may fail on
   Python 3.11.
```bash
# Go inside repo
cd simclr-medical imaging
# Create virtual environment
python -m venv venv
# Activate virtual environment
source venv/bin/activate     # For Linux, Mac OS X
source venv/Scripts/activate # For Windows
```

3. Install required packages.
```bash
pip install -r requirements.txt
```

### Usage

The codebase provides in-depth support for SimCLR pretraining, finetuning
(downstream transfer learning), testing, data preview and feature analysis via
PCA and t-SNE.

Navigate to one of the following pages below. Each environment has a
comprehensive documentation with example usage.

Pretrain:
- Go to `pretrain/simclr` directory and see `README.md`.

Finetune with frozen encoder:
- Go to `downstream/logistic_regression` and see `README.md`.

Finetune with unfrozen encoder:
- Go to `downstream/resnet` and see `README.md`.

Baseline:
- Go to `downstream/resnet` and see `README.md`.

Regardless of the experiment, all programs search for models (`.ckpt` files) in
`models/`. For example, when performing downstream learning, the program
searches for the pretrained file in `pretrain/simclr/models/`. If you place the
model in a different folder, you need to update `MODEL_DIR` in `utils.py`.

Note that the saved model is always the latest model after training with the
specified number of epochs. To replace the model with the best-performing
version in terms of validation accuracy, read instructions in
`scripts/replace-with-best-checkpoint.sh`.

### Existing models

A collection of pretrained and finetuned models can be accessed on [Zenodo][zenodo].

[zenodo]: https://zenodo.org/record/8048780

## Contribute

### Update requirements

```bash
$ pip freeze > requirements.txt
```

## Credits

The code for pretraining and downstream learning is heavily adapted from a
tutorial within the PyTorch Lightning documentation authored by Phillip Lippe
under the CC BY-SA license ([tutorial][tut]).

[tut]: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/13-contrastive-learning.html

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
