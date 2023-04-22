import os
import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch.utils.data as data
from downloader import Downloader

from pretrain.simclr.contrastive_downloader import ContrastiveDownloader
from pretrain.simclr.simclrlm import SimCLRLM
from pretrain.simclr.utils import (
    get_pretrained_model,
    summarise,
)
from utils import (
    NUM_WORKERS,
    SEED,
    SIMCLR_CHECKPOINT_PATH,
    SplitType,
    get_accelerator_info,
    parse_args_img_viewer,
    setup_device,
    show_example_images,
    show_original_and_augmented_example_images,
)


if __name__ == "__main__":
    DATA_FLAG = parse_args_img_viewer()

    # Seed
    pl.seed_everything(SEED)

    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    print(f"Number of workers: {NUM_WORKERS}")

    downloader = Downloader()
    train_data = downloader.load(
        DATA_FLAG,
        SplitType.TRAIN,
    )

    contrastive_downloader = ContrastiveDownloader()
    augmented_train_data = contrastive_downloader.load(
        DATA_FLAG,
        SplitType.TRAIN,
    )

    # Show example images
    show_original_and_augmented_example_images(train_data, augmented_train_data)
