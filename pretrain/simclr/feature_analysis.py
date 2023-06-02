from copy import deepcopy
import numpy as np
import os
import pytorch_lightning as pl


from args_parser import Arguments
from dimensionality_reduction import perform_feature_analysis
from downloader import Downloader
from pretrain.simclr.utils import get_pretrained_model
from utils import (
    DIMENSIONALITY_REDUCTION_SAMPLES,
    SEED,
    SIMCLR_CHECKPOINT_PATH,
    SplitType,
    encode_data_features,
    get_labels,
    setup_device,
)


if __name__ == "__main__":
    (
        DATA_FLAG,
        MODEL_NAME,
        EXPLORE_TSNE_ONLY,
        LEGEND,
    ) = Arguments.parse_args_feature_analysis()

    # Seed
    pl.seed_everything(SEED)
    np.random.seed(SEED)

    # Setup device
    device = setup_device()
    print(f"Device: {device}")

    # Load data
    downloader = Downloader()
    train_data = downloader.load(DATA_FLAG, SplitType.TRAIN)
    test_data = downloader.load(
        DATA_FLAG,
        SplitType.TEST,
        num_samples=DIMENSIONALITY_REDUCTION_SAMPLES,
    )
    train_labels = get_labels(train_data)
    labels = get_labels(test_data)

    # Load SimCLR model
    encoder_path = os.path.join(SIMCLR_CHECKPOINT_PATH, MODEL_NAME)
    encoder_model = get_pretrained_model(encoder_path)
    print("SimCLR model loaded")

    print("Preparing data features...")
    network = deepcopy(encoder_model.convnet)
    train_feats_data = encode_data_features(
        network,
        train_data,
        device,
        sort=False,
    )
    test_feats_data = encode_data_features(
        network,
        test_data,
        device,
        sort=False
    )
    print("Preparing data features: Done!")

    perform_feature_analysis(
        train_feats_data,
        test_feats_data,
        train_labels,
        labels,
        DATA_FLAG,
        explore_tsne_only=EXPLORE_TSNE_ONLY,
        legend=LEGEND,
    )
