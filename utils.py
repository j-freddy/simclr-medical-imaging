from enum import Enum
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as data
from torchmetrics import AUROC
import torchvision


# ==============================================================================
#     CONSTANTS
# ==============================================================================

# Global seed for pseudorandom numbers for reproducibility
SEED = 1969
NUM_WORKERS = os.cpu_count()
# Path to data files
DATASET_PATH = "data/"
# Path to general output files (e.g. matplotlib graphs)
OUT_PATH = "out/"

# Path to ResNet models
MODEL_DIR = "models/"
SIMCLR_CHECKPOINT_PATH = f"pretrain/simclr/{MODEL_DIR}"
LOGISTIC_REGRESSION_CHECKPOINT_PATH = f"downstream/logistic_regression/{MODEL_DIR}"
RESNET_TRANSFER_CHECKPOINT_PATH = f"downstream/resnet/{MODEL_DIR}"

DIMENSIONALITY_REDUCTION_SAMPLES = 2000

# Colours for plots
COLORS = [
    "#212529",  # Black
    "#c92a2a",  # Red
    "#5f3dc4",  # Violet
    "#1864ab",  # Blue
    "#2b8a3e",  # Green
    "#862e9c",  # Grape
    "#4263eb",  # Indigo
    "#1098ad",  # Cyan
    "#f08c00",  # Yellow
]

# ==============================================================================
#     ENUMS
# ==============================================================================


class MedMNISTCategory(Enum):
    """
    MedMNIST v2 modalities - https://medmnist.com/
    """
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

# ==============================================================================
#     FUNCTIONS
# ==============================================================================


def setup_device():
    """
    Set up Torch device and set seed. Enforce all operations to be
    deterministic.

    Returns:
        torch.device: Device used for Torch scripts.
    """
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
    """
    Get accelerator type (GPU or CPU) and the number of available threads.

    Returns:
        accelerator (str): The accelerator being used.
        num_threads (int): The number of threads being used.
    """
    if torch.cuda.is_available():
        accelerator = "gpu"
        num_threads = torch.cuda.device_count()
    else:
        accelerator = "cpu"
        num_threads = torch.get_num_threads()

    # Temporarily overriding number of threads to 1
    # Multi-GPU usage is not supported for current setup
    num_threads = 1

    return accelerator, num_threads


def show_example_images(data, num_examples=12, reshape=False):
    """
    Display a grid of example images.

    Args:
        data (torch.Tensor): Images data.
        num_examples (int, optional): The number of examples to display.
            Defaults to 12.
        reshape (bool, optional): Corrects the shape of data. Defaults to
            False.
    """
    imgs = torch.stack(
        [img for idx in range(num_examples) for img in data[idx][0]],
        dim=0
    )

    if reshape:
        imgs = imgs.reshape(-1, 3, imgs.shape[-1], imgs.shape[-1])

    img_grid = torchvision.utils.make_grid(
        imgs,
        # Number of images per row
        nrow=2,
        normalize=True,
        pad_value=0.9,
    )
    img_grid = img_grid.permute(1, 2, 0)

    plt.figure(figsize=(8, 8))
    plt.imshow(img_grid)
    plt.axis("off")
    plt.show()


def show_original_and_augmented_example_images(
    data,
    augmented_data,
    num_examples=6,
    views=2,
):
    """
    Display a grid of original and augmented example images side by side.

    Args:
        data (torch.Tensor): Images data.
        augmented_data (torch.Tensor): Augmented images data.
        num_examples (int, optional): The number of examples to display.
            Defaults to 6.
        views (int, optional): The number of augmented images to display per
            example. Defaults to 2.
    """
    imgs = torch.stack(
        [img for idx in range(num_examples) for img in data[idx][0]],
        dim=0
    )

    augmented_imgs = torch.stack(
        [img for idx in range(num_examples) for img in augmented_data[idx][0]],
        dim=0
    )

    # Reshape non-augmented images
    imgs = imgs.reshape(-1, 3, imgs.shape[-1], imgs.shape[-1])

    img_grid = torchvision.utils.make_grid(
        imgs,
        nrow=1,
        normalize=True,
        pad_value=0.9,
    )

    augmented_img_grid = torchvision.utils.make_grid(
        augmented_imgs,
        # Number of images per row
        nrow=views,
        normalize=True,
        pad_value=0.9,
    )

    img_grid = img_grid.permute(1, 2, 0)
    augmented_img_grid = augmented_img_grid.permute(1, 2, 0)

    _, ax = plt.subplots(1, 2, gridspec_kw={
        "width_ratios": [1, views],
        "wspace": 0,
        "hspace": 0,
    })
    ax[0].imshow(img_grid)
    ax[0].axis("off")
    ax[1].imshow(augmented_img_grid)
    ax[1].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def convert_to_rgb(img):
    """
    Convert an image to RGB.

    Args:
        img (PIL.Image.Image): The input image.

    Returns:
        PIL.Image.Image: The same image in RGB.
    """
    return img.convert("RGB")


def convert_to_ycbcr(img):
    """
    Convert an image to YCbCr.

    Args:
        img (PIL.Image.Image): The input image.

    Returns:
        PIL.Image.Image: The same image in YCbCr.
    """
    return img.convert("YCbCr")


def get_feats(feats_data):
    """
    Given a Torch dataset feats_data consisting of features and labels, extract
    the features only as a numpy array.

    Args:
        feats_data (torch.data.TensorDataset): Dataset with features and labels.

    Returns:
        numpy.ndarray: The features extracted from feats_data.
    """

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


def get_labels_as_tensor(dataset):
    """
    Given a Torch dataset, extract the labels only as a Tensor.

    Args:
        dataset (torch.data.TensorDataset): Dataset with features and labels.

    Returns:
        torch.Tensor: The labels corresponding to dataset.
    """
    dataloader = data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=False
    )

    return torch.cat([
        batch_labels for _, batch_labels in dataloader
    ]).flatten()


def get_labels(dataset):
    """
    Given a Torch dataset, extract the labels only as a numpy array.

    Args:
        dataset (torch.data.TensorDataset): Dataset with features and labels.

    Returns:
        numpy.ndarray: The labels corresponding to dataset.
    """
    return get_labels_as_tensor(dataset).numpy()


def encode_data_features(network, dataset, device, batch_size=64, sort=True):
    """
    Given a network encoder, pass the dataset through the encoder, remove the FC
    layers and return the encoded features.

    Args:
        network (torch.nn.Module): The network used for encoding features.
        dataset (torch.utils.data.Dataset): The input dataset.
        device (torch.device): Device used for computation.
        batch_size (int, optional): The batch size. Defaults to 64.
        sort (bool, optional): Sort the features by labels. Defaults to True.

    Returns:
        torch.utils.data.TensorDataset: Dataset containing the encoded
            features and labels.
    """
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


def get_auroc_metric(model, test_loader, num_classes):
    """
    Compute the AUROC (Area Under the Receiver Operating Characteristic) metric
    for a multiclass classification task.

    Args:
        model (torch.nn.Module): -
        test_loader (torch.utils.data.DataLoader): The data loader for the test
            dataset used to compute the metric.
        num_classes (int): The number of classes in the classification task.

    Returns:
        float: The AUROC metric value.
    """
    y_true = []
    y_pred = []

    for batch in test_loader:
        x, y = batch
        y_true.extend(y)
        y_pred.extend(model(x))

    y_true = torch.stack(y_true).squeeze()
    y_pred = torch.stack(y_pred)

    auroc_metric = AUROC(task="multiclass", num_classes=num_classes)
    return auroc_metric(y_pred, y_true).item()
