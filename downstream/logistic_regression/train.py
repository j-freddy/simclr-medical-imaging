from copy import deepcopy
import os
import pytorch_lightning as pl
import sys
import torch.nn as nn

from downstream.logistic_regression.downloader import Downloader
from downstream.logistic_regression.utils import summarise
from pretrain.simclr.utils import CHECKPOINT_PATH, get_pretrained_model
from utils import (
    NUM_WORKERS,
    SEED,
    MedMNISTCategory,
    setup_device,
    show_example_images,
    SplitType,
)

def prepare_data_features(pretrained_model, dataset):
    # Deep copy convolutional network
    network = deepcopy(pretrained_model.convnet)

    # Remove projection head g(.)
    network.fc = nn.Identity()
    # Set network to evaluation mode
    network.eval()
    # Move network to specified device
    network.to(device)

if __name__ == "__main__":
    DATA_FLAG = MedMNISTCategory.RETINA
    PRETRAINED_FILE = f"pretrain-retinamnist.ckpt"
    MAX_EPOCHS = 2

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

    # Show example images
    show_example_images(train_data, reshape=True)
    show_example_images(val_data, reshape=True)
    show_example_images(test_data, reshape=True)
    sys.exit()

    filename = f"downstream-{DATA_FLAG.value}.ckpt"

    pretrained_path = os.path.join(CHECKPOINT_PATH, PRETRAINED_FILE)
    destination_path = os.path.join(CHECKPOINT_PATH, filename)

    # Get pretrained model
    pretrained_model = get_pretrained_model(pretrained_path)

    print("Preparing data features...")

    # train_feats = prepare_data_features(pretrained_model, train_data)
    # test_feats = prepare_data_features(pretrained_model, test_data)

    print("Preparing data features: Done!")

    # Train model

    # TODO

    summarise()
