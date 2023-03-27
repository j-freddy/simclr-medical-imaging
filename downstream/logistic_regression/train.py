from copy import deepcopy
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import sys
import torch
import torch.nn as nn
import torch.utils.data as data

from downloader import Downloader
from downstream.logistic_regression.logistic_regressionlm import LogisticRegression
from downstream.logistic_regression.utils import summarise
from pretrain.simclr.utils import encode_data_features, get_pretrained_model
from utils import (
    LOGISTIC_REGRESSION_CHECKPOINT_PATH,
    NUM_WORKERS,
    SEED,
    SIMCLR_CHECKPOINT_PATH,
    MedMNISTCategory,
    get_accelerator_info,
    setup_device,
    show_example_images,
    SplitType,
)


def set_args():
    DATA_FLAG = MedMNISTCategory.DERMA
    PRETRAINED_FILE = f"pretrain-dermamnist-thousand.ckpt"
    # TODO Infer this from dataset
    NUM_CLASSES = 7
    MAX_EPOCHS = 2000

    return (
        DATA_FLAG,
        PRETRAINED_FILE,
        NUM_CLASSES,
        MAX_EPOCHS,
    )


def train_logistic_regression(
    batch_size,
    train_feats_data,
    test_feats_data,
    max_epochs=100,
    **kwargs,
):
    destination_path = os.path.join(
        LOGISTIC_REGRESSION_CHECKPOINT_PATH,
        f"{model_name}.ckpt"
    )
    tb_path = os.path.join(LOGISTIC_REGRESSION_CHECKPOINT_PATH, "tb_logs")

    model = None

    # Check if model already exists
    if os.path.isfile(destination_path):
        print(f"Model already exists at: {destination_path}")

        # Automatically load model with saved hyperparameters
        model = LogisticRegression.load_from_checkpoint(destination_path)
        print("Model loaded")
    else:
        model = LogisticRegression(**kwargs)
        print("Model created")
    
    # Tensorboard
    logger = TensorBoardLogger(save_dir=tb_path, name=model_name)

    # Trainer
    accelerator, num_threads = get_accelerator_info()

    trainer = pl.Trainer(
        default_root_dir=LOGISTIC_REGRESSION_CHECKPOINT_PATH,
        accelerator=accelerator,
        devices=num_threads,
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[
            # Save model as checkpoint periodically under checkpoints folder
            ModelCheckpoint(
                save_weights_only=False,
                mode="max",
                monitor="val_acc"
            ),
            # Auto-logs learning rate
            LearningRateMonitor("epoch"),
        ],
        check_val_every_n_epoch=10,
    )

    # Do not require optional logging
    trainer.logger._default_hp_metric = None

    train_loader = data.DataLoader(
        train_feats_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )

    test_loader = data.DataLoader(
        test_feats_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )

    pl.seed_everything(SEED)

    # Train model
    trainer.fit(model, train_loader, test_loader)

    # Load best checkpoint after training
    model = LogisticRegression.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )

    # Save model
    trainer.save_checkpoint(destination_path)

    # Test best model on train and test set
    train_result = trainer.test(model, dataloaders=train_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {
        "train": train_result[0]["test_acc"],
        "test": test_result[0]["test_acc"]
    }

    return model, result


if __name__ == "__main__":
    DATA_FLAG, PRETRAINED_FILE, NUM_CLASSES, MAX_EPOCHS = set_args()

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
    # show_example_images(train_data, reshape=True)
    # show_example_images(val_data, reshape=True)
    # show_example_images(test_data, reshape=True)
    # sys.exit()

    model_name = f"downstream-{DATA_FLAG.value}-sgd-steplr"

    # Get pretrained model
    # TODO This function should be in root/utils.py and should be able to load
    # models other than SimCLR
    pretrained_path = os.path.join(SIMCLR_CHECKPOINT_PATH, PRETRAINED_FILE)
    pretrained_model = get_pretrained_model(pretrained_path)

    print("Preparing data features...")

    train_feats = encode_data_features(pretrained_model, train_data, device)
    test_feats = encode_data_features(pretrained_model, test_data, device)

    print("Preparing data features: Done!")

    # Train model

    _, d = train_feats.tensors[0].shape

    model, results = train_logistic_regression(
        batch_size=64,
        train_feats_data=train_feats,
        test_feats_data=test_feats,
        max_epochs=MAX_EPOCHS,
        feature_dim=d,
        num_classes=NUM_CLASSES,
        lr=1e-3,
        weight_decay=1e-3,
    )

    print(results)

    summarise()
