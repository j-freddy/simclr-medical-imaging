import sys
import pytorch_lightning as pl

from const import NUM_WORKERS, SEED
from downloader import Downloader
from utils import (
    MedMNISTCategory,
    SplitType,
    setup_device,
    show_example_images,
)

if __name__ == "__main__":
    DATA_FLAG = MedMNISTCategory.RETINA
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
