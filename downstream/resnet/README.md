# ResNet

Make sure you are currently in the `root` folder.

## Background

The base encoder of a pretrained model (e.g. ResNet-18) is extracted, and an
extra linear layer (with cross-entropy loss) is appended to the end of the
encoder so the output becomes a predicted label. The entire encoder gets
finetuned during transfer learning.

## Train

Train a model starting from a pretrained ResNet-18 architecture.

```bash
$ python -m downstream.resnet.train
```

## Baseline

Train a model starting from a newly initialised ResNet-18 architecture.

```bash
$ python -m downstream.resnet.baseline
```

## TensorBoard

```bash
$ tensorboard --logdir downstream/resnet/models/tb_logs
```
