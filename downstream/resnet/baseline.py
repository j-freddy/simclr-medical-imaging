from copy import deepcopy
from medmnist import INFO
import sys
import pytorch_lightning as pl
import torch.nn as nn
import torchvision

from downloader import Downloader
from downstream.resnet.train import finetune_resnet
from downstream.resnet.utils import summarise
from pretrain.simclr.utils import get_pretrained_model
from utils import (
    NUM_WORKERS,
    SEED,
    parse_args,
    setup_device,
    show_example_images,
    SplitType,
)

if __name__ == "__main__":
    (
        DATA_FLAG,
        MAX_EPOCHS,
        NUM_SAMPLES,
        MODEL_NAME,
    ) = parse_args()

    # Seed
    pl.seed_everything(SEED)

    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    print(f"Number of workers: {NUM_WORKERS}")

    # Load data
    num_samples = NUM_SAMPLES or -1

    downloader = Downloader()
    train_data = downloader.load(DATA_FLAG, SplitType.TRAIN, num_samples)
    val_data = downloader.load(DATA_FLAG, SplitType.VALIDATION)
    test_data = downloader.load(DATA_FLAG, SplitType.TEST)

    # # Show example images
    # show_example_images(train_data, reshape=True)
    # show_example_images(val_data, reshape=True)
    # show_example_images(test_data, reshape=True)
    # sys.exit()

    model_name = MODEL_NAME or f"baseline-{DATA_FLAG.value}"

    # Initialise new ResNet-18 model
    hidden_dim = 128

    backbone = torchvision.models.resnet18(
        weights=None,
        num_classes=4 * hidden_dim,
    )

    backbone.fc = nn.Sequential(
        backbone.fc,
        nn.ReLU(inplace=True),
    )

    model, result = finetune_resnet(
        backbone,
        device,
        batch_size=min(128, len(train_data)),
        train_data=train_data,
        test_data=test_data,
        model_name=model_name,
        num_classes=len(INFO[DATA_FLAG]["label"]),
        max_epochs=MAX_EPOCHS,
        lr=0.001,
        momentum=0.9,
    )

    summarise()
    print(result)
