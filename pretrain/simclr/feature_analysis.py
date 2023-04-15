from matplotlib import pyplot as plt
from medmnist import INFO
import os
import pytorch_lightning as pl
from sklearn import decomposition
import sys

from dimensionality_reduction import perform_pca
from downloader import Downloader
from pretrain.simclr.utils import get_data_features_from_pretrained_model, get_pretrained_model
from utils import SEED, SIMCLR_CHECKPOINT_PATH, SplitType, get_feats, get_labels, parse_args_test, setup_device


# TODO Write a README.md for this file
# python -m pretrain.simclr.feature_analysis -c pathmnist -fin pretrain-pathmnist


if __name__ == "__main__":
    (
        DATA_FLAG,
        MODEL_NAME,
    ) = parse_args_test()

    # Use stylish plots
    plt.style.use("ggplot")

    # Seed
    pl.seed_everything(SEED)

    # Setup device
    device = setup_device()
    print(f"Device: {device}")

    # Load data
    downloader = Downloader()
    train_data = downloader.load(DATA_FLAG, SplitType.TRAIN)
    test_data = downloader.load(DATA_FLAG, SplitType.TRAIN, num_samples=1000)
    test_labels = get_labels(test_data)

    # Load SimCLR model
    encoder_path = os.path.join(SIMCLR_CHECKPOINT_PATH, MODEL_NAME)
    encoder_model = get_pretrained_model(encoder_path)
    print("SimCLR model loaded")

    print("Preparing data features...")
    train_feats_data = get_data_features_from_pretrained_model(encoder_model, train_data, device)
    test_feats_data = get_data_features_from_pretrained_model(encoder_model, test_data, device)
    print("Preparing data features: Done!")

    train_feats = get_feats(train_feats_data)
    test_feats = get_feats(test_feats_data)

    # In SimCLR pretraining we used a batch size of 128 and features = size*4
    assert train_feats.shape[1] == 512
    assert test_feats.shape[1] == 512

    num_classes = len(INFO[DATA_FLAG]["label"])

    # Perform PCA
    perform_pca(train_feats, test_feats, test_labels, num_classes)
