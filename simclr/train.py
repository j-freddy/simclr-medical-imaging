import pytorch_lightning as pl

from const import NUM_WORKERS, SEED
from loader import Loader
from utils import setup_device, show_example_images

if __name__ == "__main__":
    # Seed
    pl.seed_everything(SEED)

    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    print(f"Number of workers: {NUM_WORKERS}")

    # Load data
    loader = Loader()
    train_data = loader.load("dermamnist", "train")

    # Show example images
    show_example_images(train_data)
