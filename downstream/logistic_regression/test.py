from medmnist import INFO
import numpy as np
import os
import pytorch_lightning as pl
import torch.utils.data as data

from args_parser import Arguments
from downloader import Downloader
from downstream.logistic_regression.logistic_regressionlm import LogisticRegressionLM
from pretrain.simclr.utils import get_data_features_from_pretrained_model, get_pretrained_model
from utils import (
    LOGISTIC_REGRESSION_CHECKPOINT_PATH,
    NUM_WORKERS,
    SEED,
    SIMCLR_CHECKPOINT_PATH,
    SplitType,
    get_accelerator_info,
    get_auroc_metric,
    setup_device,
)

if __name__ == "__main__":
    (
        DATA_FLAG,
        ENCODER_NAME,
        MODEL_NAME,
    ) = Arguments.parse_args_test(logistic_regression=True)

    # Seed
    pl.seed_everything(SEED)
    np.random.seed(SEED)

    # Setup device
    device = setup_device()
    print(f"Device: {device}")

    # Load data
    downloader = Downloader()
    test_data = downloader.load(DATA_FLAG, SplitType.TEST)

    # Load base encoder
    encoder_path = os.path.join(SIMCLR_CHECKPOINT_PATH, ENCODER_NAME)
    encoder_model = get_pretrained_model(encoder_path)
    print("Base encoder loaded")

    # Load logistic regression model
    pretrained_path = os.path.join(
        LOGISTIC_REGRESSION_CHECKPOINT_PATH,
        MODEL_NAME,
    )

    model = LogisticRegressionLM.load_from_checkpoint(pretrained_path)
    print("Logistic regression loaded")

    print("Preparing data features...")
    test_feats_data = get_data_features_from_pretrained_model(
        encoder_model, test_data, device)
    print("Preparing data features: Done!")

    # Test

    test_loader = data.DataLoader(
        test_feats_data,
        batch_size=min(64, len(test_data)),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )

    # Trainer
    accelerator, num_threads = get_accelerator_info()

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=num_threads,
    )

    # Do not require optional logging
    trainer.logger._default_hp_metric = None

    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    
    result = {
        "top-1 acc": test_result[0]["test_acc"],
        "auroc": get_auroc_metric(
            model,
            test_loader,
            num_classes=len(INFO[DATA_FLAG]["label"])
        ),
    }

    print(f"Model name: {MODEL_NAME}")
    print(f"Dataset: {DATA_FLAG}")
    print(result)
