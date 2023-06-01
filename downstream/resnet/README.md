# ResNet

Make sure you are currently in the `root` folder.

## Context

The base encoder of a pretrained model (ResNet-18) is extracted, and an
extra linear layer (with cross-entropy loss) is appended to the end of the
encoder so the output becomes a predicted label. The entire encoder gets
finetuned during transfer learning.

## Train

Train a model starting from a pretrained ResNet-18 architecture.

```bash
$ python -m downstream.resnet.train -c C -epochs EPOCHS [-samples SAMPLES] [-spc SPC] [-fin FIN] [-fout FOUT]
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

`spc`
- Number of training samples per class
- Default: uses all training samples

`fin`
- Pretrained model filename
- Default: creates a newly initialised ResNet-18 model

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

## Test

Calculate and print test metrics on an existing model.

```bash
$ python -m downstream.resnet.test -c C -fin FIN
# Run for help/description
$ python -m downstream.resnet.test -h
```

`-c`
- Specifies MedMNIST2D dataset to be used: https://medmnist.com/
- Accepted arguments below
```py
pathmnist, chestmnist, dermamnist, octmnist, pneumoniamnist, retinamnist, 
breastmnist, bloodmnist, tissuemnist, organamnist, organcmnist, organsmnist
```

`fin`
- Input downstream/baseline model filename

### Example

```bash
$ python -m downstream.resnet.test -c breastmnist -fin simclr-demo
# A larger dataset can take 5-10 minutes
$ python -m downstream.resnet.test -c dermamnist -fin downstream-dermamnist-100-samples
```

## Feature analysis

Perform feature analysis on learned representations using dimensionality
reduction techniques like PCA and t-SNE. Visualisations are saved in the root
respository under `out/`.

You must have an existing downstream/baseline model.

```bash
$ python -m downstream.resnet.feature_analysis -c C -fin FIN -tsne
# Run for help/description
$ python -m downstream.resnet.feature_analysis -h
```

`-c`
- Specifies MedMNIST2D dataset to be used: https://medmnist.com/
- Accepted arguments below
```py
pathmnist, chestmnist, dermamnist, octmnist, pneumoniamnist, retinamnist, 
breastmnist, bloodmnist, tissuemnist, organamnist, organcmnist, organsmnist
```

`fin`
- Input downstream/baseline model filename. Data is passed through base encoder
  to output features. Components are learned on train features and reduced test
  features are visualised.

### Example

```bash
$ python -m downstream.resnet.feature_analysis -c breastmnist -fin simclr-demo
$ python -m downstream.resnet.feature_analysis -c pathmnist -fin baseline-pathmnist-18 -tsne
```

## TensorBoard

```bash
$ tensorboard --logdir downstream/resnet/models/tb_logs
```
