from enum import Enum
import os
import matplotlib.pyplot as plt
import torch
import torchvision


SEED = 1969
NUM_WORKERS = os.cpu_count()
DATASET_PATH = "data/"

SIMCLR_CHECKPOINT_PATH = "pretrain/simclr/models/"
LOGISTIC_REGRESSION_CHECKPOINT_PATH = "downstream/logistic_regression/models/"
RESNET_TRANSFER_CHECKPOINT_PATH = "downstream/resnet/models/"


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
