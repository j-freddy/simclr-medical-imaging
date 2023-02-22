from copy import deepcopy
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import sys
import torch
import torch.nn as nn
import torch.utils.data as data

from downstream.logistic_regression.downloader import Downloader
from downstream.logistic_regression.logistic_regressionlm import LogisticRegression
from downstream.logistic_regression.utils import CHECKPOINT_PATH, summarise
from pretrain.simclr.utils import get_pretrained_model
from utils import (
    NUM_WORKERS,
    SEED,
    MedMNISTCategory,
    get_accelerator_info,
    setup_device,
    show_example_images,
    SplitType,
)


def prepare_data_features(pretrained_model, dataset, batch_size=64):
    # Deep copy convolutional network
    network = deepcopy(pretrained_model.convnet)

    # Remove projection head g(.)
    network.fc = nn.Identity()
    # Set network to evaluation mode
    network.eval()
    # Move network to specified device
    network.to(device)

    data_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
    )

    feats, labels = [], []

    for batch_imgs, batch_labels in data_loader:
        # TODO Understand this
        batch_imgs = batch_imgs.to(device)
        batch_feats = network(batch_imgs)
        feats.append(batch_feats.detach().cpu())
        labels.append(batch_labels)

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)

    # Sort images by labels
    labels, indexes = labels.sort()
    feats = feats[indexes]

    return data.TensorDataset(feats, labels)


def train_logistic_regression(
    batch_size,
    train_feats_data,
    test_feats_data,
    model_suffix,
    max_epochs=100,
    **kwargs,
):
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

    # Trainer
    accelerator, num_threads = get_accelerator_info()

    trainer = pl.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        accelerator=accelerator,
        devices=num_threads,
        max_epochs=max_epochs,
        callbacks=[
            # Save model as checkpoint periodically under checkpoints folder
            ModelCheckpoint(
                save_weights_only=False,
                mode="max",
                monitor="val_acc_top5"
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
        num_workers=0
    )

    test_loader = data.DataLoader(
        test_feats_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=0
    )

    pl.seed_everything(SEED)

    # Train model
    trainer.fit(model, train_loader, test_loader)

    # Load best checkpoint after training
    model = LogisticRegression.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )

    # Test best model on train and validation set
    train_result = trainer.test(model, dataloaders=train_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {
        "train": train_result[0]["test_acc"],
        "test": test_result[0]["test_acc"]
    }

    return model, result


if __name__ == "__main__":
    DATA_FLAG = MedMNISTCategory.RETINA
    PRETRAINED_FILE = f"pretrain-retinamnist.ckpt"
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
    # show_example_images(train_data, reshape=True)
    # show_example_images(val_data, reshape=True)
    # show_example_images(test_data, reshape=True)
    # sys.exit()

    filename = f"downstream-{DATA_FLAG.value}.ckpt"

    pretrained_path = os.path.join(CHECKPOINT_PATH, PRETRAINED_FILE)
    destination_path = os.path.join(CHECKPOINT_PATH, filename)

    # Get pretrained model
    # TODO This function should be in root/utils.py and should be able to load
    # models other than SimCLR
    pretrained_model = get_pretrained_model(pretrained_path)

    print("Preparing data features...")

    train_feats = prepare_data_features(pretrained_model, train_data)
    test_feats = prepare_data_features(pretrained_model, test_data)

    print("Preparing data features: Done!")

    # Train model

    # TODO

    # for num_imgs_per_label in [10, 20, 50]:
    #     print(f"Training: {num_imgs_per_label}")
        
    #     sub_train_set = get_smaller_dataset(train_feats_simclr, num_imgs_per_label)

    #     _, small_set_results = train_logreg(
    #         batch_size=64,
    #         train_feats_data=sub_train_set,
    #         test_feats_data=test_feats_simclr,
    #         model_suffix=num_imgs_per_label,
    #         feature_dim=train_feats_simclr.tensors[0].shape[1],
    #         num_classes=10,
    #         lr=1e-3,
    #         weight_decay=1e-3,
    #     )

    # results[num_imgs_per_label] = small_set_results

    summarise()
