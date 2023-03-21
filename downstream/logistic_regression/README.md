# Logistic Regression

Make sure you are currently in the `root` folder.

## Background

The base encoder of a pretrained model (e.g. ResNet-18) is extracted. Images are
passed into the encoder, then the encoded features are then passed into a
1-layer logistic regression model (with cross-entropy loss) to output a
predicted label. The base encoder stays fixed, and only the logistic regression
model gets finetuned during transfer learning.

## Train

```bash
$ python -m downstream.logistic_regression.train
```

## TensorBoard

```bash
$ tensorboard --logdir downstream/logistic_regression/models/tb_logs
```
