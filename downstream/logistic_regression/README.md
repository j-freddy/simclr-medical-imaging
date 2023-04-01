# Logistic Regression

Make sure you are currently in the `root` folder.

## Background

The base encoder of a pretrained model (e.g. ResNet-18) is extracted. Images are
passed into the encoder, then the encoded features are then passed into a
1-layer logistic regression model (with cross-entropy loss) to output a
predicted label. The base encoder stays fixed, and only the logistic regression
model gets finetuned during transfer learning.

## Train

## Train

```bash
$ python -m downstream.logistic_regression.train -c C -epochs EPOCHS -fin FIN [-samples SAMPLES] [-fout FOUT]
# Run for help/description
$ python -m downstream.logistic_regression.train -h
```

If training successful, the model can be found in `models/`.

`-c`
- Specifies MedMNIST2D dataset to be used: https://medmnist.com/
- Accepted arguments below
```py
pathmnist, chestmnist, dermamnist, octmnist, pneumoniamnist, retinamnist, 
breastmnist, bloodmnist, tissuemnist, organamnist, organcmnist, organsmnist
```

`epochs`
- Maximum number of epochs

`samples`
- Number of training samples
- Default: uses all training samples

`fin`
- Pretrained model filename

`fout`
- Output model filename
- Default: `pretrain-[category]`

### Example

You must have an existing pretrained model. If not, read
`pretrain/simclr/README.md`.

```bash
# Quick demo: takes 5 minutes to train
$ python -m downstream.logistic_regression.train -c breastmnist -epochs 12 -samples 20 -fin simclr-demo -fout simclr-demo
# Takes 1 hour to train on GPU
$ python -m downstream.logistic_regression.train -c dermamnist -epochs 1000 -samples 100 -fin pretrain-dermamnist
```

If training successful for the demo, the model can be found as
`models/simclr-demo.ckpt`.

## TensorBoard

```bash
$ tensorboard --logdir downstream/logistic_regression/models/tb_logs
```
