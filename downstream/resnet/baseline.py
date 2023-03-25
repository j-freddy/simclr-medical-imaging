from copy import deepcopy
import sys
import pytorch_lightning as pl
import torchvision

from downloader import Downloader
from downstream.resnet.train import finetune_resnet
from downstream.resnet.utils import summarise
from pretrain.simclr.utils import get_pretrained_model
from utils import (
    NUM_WORKERS,
    SEED,
    MedMNISTCategory,
    setup_device,
    show_example_images,
    SplitType,
)


def set_args():
    DATA_FLAG = MedMNISTCategory.DERMA
    # TODO Infer this from dataset
    NUM_CLASSES = 7
    MAX_EPOCHS = 2000

    return (
        DATA_FLAG,
        NUM_CLASSES,
        MAX_EPOCHS,
    )

if __name__ == "__main__":
    DATA_FLAG, NUM_CLASSES, MAX_EPOCHS = set_args()

    # Seed
    pl.seed_everything(SEED)

    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    print(f"Number of workers: {NUM_WORKERS}")

    # Load data
    downloader = Downloader()
    train_data = downloader.load(DATA_FLAG, SplitType.TRAIN)
    val_data = downloader.load(DATA_FLAG, SplitType.VALIDATION)
    test_data = downloader.load(DATA_FLAG, SplitType.TEST)

    # # Show example images
    # show_example_images(train_data, reshape=True)
    # show_example_images(val_data, reshape=True)
    # show_example_images(test_data, reshape=True)
    # sys.exit()

    model_name = f"baseline-{DATA_FLAG.value}"

    # Initialise new ResNet-18 model
    hidden_dim = 128

    backbone = torchvision.models.resnet18(
        weights=None,
        num_classes=4 * hidden_dim
    )

    model, result = finetune_resnet(
        backbone,
        batch_size=128,
        train_data=train_data,
        test_data=test_data,
        model_name=model_name,
        max_epochs=MAX_EPOCHS,
        lr=0.001,
        momentum=0.9,
    )

    summarise()
