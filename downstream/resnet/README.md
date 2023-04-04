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
$ python -m downstream.resnet.train -c C -epochs EPOCHS [-samples SAMPLES] [-fin FIN] [-fout FOUT]
# Run for help/description
$ python -m downstream.resnet.train -h
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
- Default: creates a  newly initialised ResNet-18 model

`fout`
- Output model filename
- Default: `pretrain-[category]`

### Example

If you want to perform downstream transfer learning, you must have an existing
pretrained model. If not, read `pretrain/simclr/README.md`.

You do not need an existing model to perform supervised learning. This is for
the purpose of creating baseline metrics.

```bash
# Quick demo: takes 5 minutes to train
$ python -m downstream.resnet.train -c breastmnist -epochs 12 -samples 20 -fin simclr-demo -fout simclr-demo
# Takes 1 hour to train on GPU
$ python -m downstream.resnet.train -c dermamnist -epochs 1000 -samples 100 -fin pretrain-dermamnist
# Baseline: Supervised learning
$ python -m downstream.resnet.train -c dermamnist -epochs 1000 -samples 100
```

If training successful for the demo, the model can be found as
`models/simclr-demo.ckpt`.

## TensorBoard

```bash
$ tensorboard --logdir downstream/resnet/models/tb_logs
```
