# SimCLR

Make sure you are currently in the `root` folder.

## Train

```bash
$ python -m pretrain.simclr.train -c C -epochs EPOCHS -aug AUG [-samples SAMPLES] [-fin FIN] [-fout FOUT]
# Run for help/description
$ python -m pretrain.simclr.train -h
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

`aug`
- Specifies which augmentation sequence to use. Accepted inputs: `natural`,
  `novel`, `simple`, `greyscale`. Use [Data preview](#data-preview) environment
  to see effect of augmentations.

`samples`
- Number of training samples
- Default: uses all training samples

`fin`
- Input pretrained model filename used as a starting point for further
  pretraining
- Default: newly initialised ResNet-18

`fout`
- Output model filename
- Default: `pretrain-[category]`

### Example

```bash
# Quick demo: takes 5 minutes to train
$ python -m pretrain.simclr.train -c breastmnist -epochs 3 -aug natural -samples 20 -fout simclr-demo
# Takes 1 day to train on GPU
$ python -m pretrain.simclr.train -c dermamnist -epochs 2000 -aug natural
# Perform further pretraining on a pretrained model
$ python -m pretrain.simclr.train -c breastmnist -epochs 2000 -aug natural -fin pretrain-dermamnist
```

If training successful for the demo, the model can be found as
`models/simclr-demo.ckpt`.

## Feature analysis

Perform feature analysis on learned representations using dimensionality
reduction techniques like PCA and t-SNE. Visualisations are saved in the root
respository under `out/`.

You must have an existing pretrained model.

```bash
$ python -m pretrain.simclr.feature_analysis -c C -fin FIN -tsne
# Run for help/description
$ python -m pretrain.simclr.feature_analysis -h
```

`-c`
- Specifies MedMNIST2D dataset to be used: https://medmnist.com/
- Accepted arguments below
```py
pathmnist, chestmnist, dermamnist, octmnist, pneumoniamnist, retinamnist, 
breastmnist, bloodmnist, tissuemnist, organamnist, organcmnist, organsmnist
```

`fin`
- Input pretrained model filename. Data is passed through base encoder to output
  features. Components are learned on train features and reduced test features
  are visualised.

`tsne`
- If not included, perform PCA then t-SNE with default parameters, visualising
  the reduced test data points. If included, perform t-SNE with various
  perplexity values, visualising both reduced train and test data points.

### Example

```bash
$ python -m pretrain.simclr.feature_analysis -c breastmnist -fin simclr-demo
$ python -m pretrain.simclr.feature_analysis -c dermamnist -fin pretrain-dermamnist -tsne
```

## Data preview

Visualise original and augmented images to compare the effect of data
augmentations.

### Example

```bash
$ python -m pretrain.simclr.data_preview -c pathmnist -aug novel
```

## TensorBoard

```bash
$ tensorboard --logdir pretrain/simclr/models/tb_logs
```
