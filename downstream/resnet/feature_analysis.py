from copy import deepcopy
from matplotlib import pyplot as plt
from medmnist import INFO
import os
import pytorch_lightning as pl
import torch.nn as nn

from dimensionality_reduction import perform_pca
from downloader import Downloader
from downstream.resnet.resnet_transferlm import ResNetTransferLM
from downstream.resnet.train import initialise_new_network
from utils import (
    PCA_SAMPLES,
    SEED,
    RESNET_TRANSFER_CHECKPOINT_PATH,
    SplitType,
    encode_data_features,
    get_feats,
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

    # Setup device
    device = setup_device()
    print(f"Device: {device}")

    # Load data
    downloader = Downloader()
    train_data = downloader.load(DATA_FLAG, SplitType.TRAIN)
    test_data = downloader.load(DATA_FLAG, SplitType.TRAIN, num_samples=PCA_SAMPLES)
    test_labels = get_labels(test_data)

    # Load ResNet model
    encoder_path = os.path.join(RESNET_TRANSFER_CHECKPOINT_PATH, MODEL_NAME)

    # Without this, getting errors due to missing backbone parameter
    resnet_base = initialise_new_network()

    resnet_base.fc = nn.Sequential(
        resnet_base.fc,
        nn.ReLU(inplace=True),
    )

    encoder_model = ResNetTransferLM.load_from_checkpoint(
        encoder_path,
        backbone=resnet_base,
    )
    print("ResNet model loaded")

    # Deep copy convolutional network
    network = deepcopy(encoder_model.backbone)

    print("Preparing data features...")
    train_feats_data = encode_data_features(network, train_data, device, sort=False)
    test_feats_data = encode_data_features(network, test_data, device, sort=False)
    print("Preparing data features: Done!")

    train_feats = get_feats(train_feats_data)
    test_feats = get_feats(test_feats_data)

    # In SimCLR pretraining we used a batch size of 128 and features = size*4
    assert train_feats.shape[1] == 512
    assert test_feats.shape[1] == 512

    # Perform PCA
    perform_pca(train_feats, test_feats, test_labels, DATA_FLAG)
