# Logistic Regression

Make sure you are currently in the `root` folder.

## Context

The base encoder of a pretrained model (ResNet-18) is extracted. Images are
passed into the encoder, then the encoded features are then passed into a
1-layer linear logistic regression head (with cross-entropy loss) to output a
predicted label. The base encoder stays fixed and only the linear head gets
finetuned during transfer learning.

Since the base encoder is fixed, only the linear head is saved
after training for minimum redundancy.

## Train

```bash
$ python -m downstream.logistic_regression.train -c C -epochs EPOCHS -fin FIN [-samples SAMPLES] [-spc SPC] [-fout FOUT]
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

`spc`
- Number of training samples per class
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

## Test

Calculate and print test metrics on an existing model.

```bash
$ python -m downstream.logistic_regression.test -c C -fencoder FENCODER -fin FIN
# Run for help/description
$ python -m downstream.logistic_regression.test -h
```

`-c`
- Specifies MedMNIST2D dataset to be used: https://medmnist.com/
- Accepted arguments below
```py
pathmnist, chestmnist, dermamnist, octmnist, pneumoniamnist, retinamnist, 
breastmnist, bloodmnist, tissuemnist, organamnist, organcmnist, organsmnist
```

`fencoder`
- Base encoder filename
- The encoder file must reside under `/pretrain/simclr/models`

`fin`
- Input logistic regression model head filename

### Example

```bash
# Note that the encoder file (fencoder flag) is:
#   /pretrain/simclr/models/simclr-demo.ckpt
# and the logistic regression head file (fin flag) is:
#   /downstream/logistic_reegression/models/simclr-demo.ckpt
$ python -m downstream.logistic_regression.test -c breastmnist -fencoder simclr-demo -fin simclr-demo
# A larger dataset can take 5-10 minutes
$ python -m downstream.logistic_regression.test -c dermamnist -fencoder pretrain-dermamnist -fin downstream-dermamnist-100-samples
```


## TensorBoard

```bash
$ tensorboard --logdir downstream/logistic_regression/models/tb_logs
```
