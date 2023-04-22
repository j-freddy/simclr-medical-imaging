import argparse
from enum import Enum
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision


SEED = 1969
NUM_WORKERS = os.cpu_count()
DATASET_PATH = "data/"

SIMCLR_CHECKPOINT_PATH = "pretrain/simclr/models/"
LOGISTIC_REGRESSION_CHECKPOINT_PATH = "downstream/logistic_regression/models/"
RESNET_TRANSFER_CHECKPOINT_PATH = "downstream/resnet/models/"

COLORS = [
    "#212529", # Black
    "#c92a2a", # Red
    "#862e9c", # Grape
    "#5f3dc4", # Violet
    "#4263eb", # Indigo
    "#1864ab", # Blue
    "#1098ad", # Cyan
    "#2b8a3e", # Green
    "#f08c00", # Yellow
]


class MedMNISTCategory(Enum):
    PATH = "pathmnist"
    CHEST = "chestmnist"
    DERMA = "dermamnist"
    OCT = "octmnist"
    PNEUMONIA = "pneumoniamnist"
    RETINA = "retinamnist"
    BREAST = "breastmnist"
    BLOOD = "bloodmnist"
    TISSUE = "tissuemnist"
    ORGANA = "organamnist"
    ORGANC = "organcmnist"
    ORGANS = "organsmnist"


class SplitType(Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


def parse_args(downstream=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, help="Data category", required=True)
    parser.add_argument("-epochs", type=int,
                        help="Maximum number of epochs", required=True)
    # Optional. Default is to use all samples
    parser.add_argument("-samples", type=int, help="Number of samples")

    if not downstream:
        # Optional. Default is new ResNet model.
        parser.add_argument(
            "-fin", type=str, help="Initial model (to further pretrain)")

    if downstream:
        # Optional. Default is new ResNet model.
        parser.add_argument("-fin", type=str, help="Pretrained model filename")

    # Optional. Default is "[pretrain/downstream]-[category]"
    parser.add_argument("-fout", type=str, help="Output model filename")

    args = parser.parse_args()

    if args.fin:
        args.fin += ".ckpt"
    return args.c, args.epochs, args.samples, args.fin, args.fout


def parse_args_test(logistic_regression=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, help="Data category", required=True)

    if logistic_regression:
        # Logistic regression model requires a separate base encoder
        parser.add_argument("-fencoder", type=str,
                            help="Base encoder filename", required=True)

    parser.add_argument("-fin", type=str, help="Model filename", required=True)

    args = parser.parse_args()
    args.fin += ".ckpt"

    if logistic_regression:
        return args.c, args.fencoder, args.fin
    return args.c, args.fin


def setup_device():
    # Use GPU if available
    device = torch.device("cuda") if torch.cuda.is_available()\
        else torch.device("cpu")

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        # Enforce all operations to be deterministic on GPU for reproducibility
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    return device


def get_accelerator_info():
    if torch.cuda.is_available():
        accelerator = "gpu"
        num_threads = torch.cuda.device_count()
    else:
        accelerator = "cpu"
        num_threads = torch.get_num_threads()

    # TODO Getting error when setting devices > 1
    num_threads = 1

    return accelerator, num_threads


def show_example_images(data, num_examples=12, reshape=False):
    imgs = torch.stack(
        [img for idx in range(num_examples) for img in data[idx][0]],
        dim=0
    )

    if reshape:
        imgs = imgs.reshape(-1, 3, imgs.shape[-1], imgs.shape[-1])

    img_grid = torchvision.utils.make_grid(
        imgs,
        nrow=6,
        normalize=True,
        pad_value=0.9
    )
    img_grid = img_grid.permute(1, 2, 0)

    plt.figure(figsize=(10, 5))
    plt.imshow(img_grid)
    plt.axis("off")
    plt.show()


def convert_to_rgb(img):
    return img.convert("RGB")


def get_feats(feats_data):
    data_loader = data.DataLoader(
        feats_data,
        batch_size=64,
        shuffle=False
    )

    features = []

    for batch in data_loader:
        features.append(batch[0])

    features = torch.cat(features, dim=0)

    return features.numpy()


def get_labels(dataset):
    dataloader = data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=False
    )

    return torch.cat([
        batch_labels for _, batch_labels in dataloader
    ]).flatten().numpy()


def encode_data_features(network, dataset, device, batch_size=64, sort=True):
    # Remove projection head g(.)
    network.fc = nn.Identity()
    # Set network to evaluation mode
    network.eval()
    # Move network to specified device
    network.to(device)

    data_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
    )

    feats, labels = [], []

    for batch_imgs, batch_labels in data_loader:
        # Move images to specified device
        batch_imgs = batch_imgs.to(device)
        # f(.)
        batch_feats = network(batch_imgs)
        # Detach tensor from current graph and move to CPU
        feats.append(batch_feats.detach().cpu())
        labels.append(batch_labels)

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)

    # Remove extra axis
    labels = labels.squeeze()

    # Sort images by labels
    if sort:
        labels, indexes = labels.sort()
        feats = feats[indexes]

    return data.TensorDataset(feats, labels)
