from medmnist import INFO
import numpy as np
import pytorch_lightning as pl

from downloader import Downloader
from utils import (
    SEED,
    SplitType,
    get_labels,
    parse_args_feature_analysis,
    setup_device,
)


if __name__ == "__main__":
    (
        DATA_FLAG,
        MODEL_NAME,
    ) = parse_args_feature_analysis()

    # Seed
    pl.seed_everything(SEED)
    np.random.seed(SEED)

    # Setup device
    device = setup_device()
    print(f"Device: {device}")

    # Load data
    downloader = Downloader()
    train_data = downloader.load(DATA_FLAG, SplitType.TRAIN)
    val_data = downloader.load(DATA_FLAG, SplitType.VALIDATION)
    test_data = downloader.load(DATA_FLAG, SplitType.TEST)

    train_labels = get_labels(train_data)
    val_labels = get_labels(val_data)
    test_labels = get_labels(test_data)

    _, label_counts_train = np.unique(train_labels, return_counts=True)
    _, label_counts_val = np.unique(val_labels, return_counts=True)
    _, label_counts_test = np.unique(test_labels, return_counts=True)

    labels_dict = INFO[DATA_FLAG]["label"]

    print(labels_dict)
    
    print(label_counts_train)
    print(label_counts_val)
    print(label_counts_test)