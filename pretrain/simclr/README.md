# SimCLR

Make sure you are currently in the `root` folder.

## Background

A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)
is a contrastive learning method that aims to learn useful representations of
images through training a convolutional neural network (CNN) to recognise
similarities between a pair of augmented data points derived from the same input
image. The idea is that the network may learn to extract useful, generalised
features that can be used in downstream tasks.

We use ResNet-18 as the CNN architecture.

## Train

```bash
$ python -m pretrain.simclr.train
```

## TensorBoard

```bash
$ tensorboard --logdir pretrain/simclr/models/tb_logs
```
