from copy import deepcopy
import os
import pytorch_lightning as pl

from dimensionality_reduction import perform_pca
from downloader import Downloader
from pretrain.simclr.utils import get_pretrained_model
from utils import (
    SEED,
    SIMCLR_CHECKPOINT_PATH,
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
    # Take first 1000 test samples
    test_data = downloader.load(DATA_FLAG, SplitType.TRAIN, num_samples=1000)
    test_labels = get_labels(test_data)

    # Load SimCLR model
    encoder_path = os.path.join(SIMCLR_CHECKPOINT_PATH, MODEL_NAME)
    encoder_model = get_pretrained_model(encoder_path)
    print("SimCLR model loaded")

    print("Preparing data features...")
    network = deepcopy(encoder_model.convnet)
    train_feats_data = encode_data_features(network, train_data, device, sort=False)
    test_feats_data = encode_data_features(network, test_data, device, sort=False)
    print("Preparing data features: Done!")

    train_feats = get_feats(train_feats_data)
    test_feats = get_feats(test_feats_data)

    # In SimCLR pretraining we used a batch size of 128 and features = size*4
    assert train_feats.shape[1] == 512
    assert test_feats.shape[1] == 512

    # Perform PCA
    perform_pca(train_feats, test_feats, test_labels, DATA_FLAG, legend=False)
