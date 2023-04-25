import pytorch_lightning as pl
from downloader import Downloader

from pretrain.simclr.contrastive_downloader import ContrastiveDownloader
from utils import (
    NUM_WORKERS,
    SEED,
    SplitType,
    parse_args_img_viewer,
    setup_device,
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

    # 5 different augmentations of same image
    VIEWS = 5

    contrastive_downloader = ContrastiveDownloader()
    augmented_train_data = contrastive_downloader.load(
        DATA_FLAG,
        SplitType.TRAIN,
        views=VIEWS,
    )

    # Show example images
    show_original_and_augmented_example_images(
        train_data,
        augmented_train_data,
        views=VIEWS
    )
