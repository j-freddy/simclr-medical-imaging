from copy import deepcopy
import os
import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch.utils.data as data

from downloader import Downloader
from downstream.resnet.ResNetTransferLM import ResNetTransferLM
from downstream.resnet.utils import summarise
from pretrain.simclr.utils import get_pretrained_model
from utils import (
    NUM_WORKERS,
    RESNET_TRANSFER_CHECKPOINT_PATH,
    SEED,
    SIMCLR_CHECKPOINT_PATH,
    MedMNISTCategory,
    get_accelerator_info,
    setup_device,
    show_example_images,
    SplitType,
)


def set_args():
    DATA_FLAG = MedMNISTCategory.BREAST
    PRETRAINED_FILE = f"pretrain-dermamnist-thousand.ckpt"
    # TODO Infer this from dataset
    NUM_CLASSES = 2
    MAX_EPOCHS = 50

    return (
        DATA_FLAG,
        PRETRAINED_FILE,
        NUM_CLASSES,
        MAX_EPOCHS,
    )


def finetune_resnet(
        network,
        batch_size,
        train_data,
        test_data,
        max_epochs=100,
        **kwargs
    ):
    destination_path = os.path.join(
        RESNET_TRANSFER_CHECKPOINT_PATH,
        f"{model_name}.ckpt"
    )
    tb_path = os.path.join(RESNET_TRANSFER_CHECKPOINT_PATH, "tb_logs")

    model = None

    # Check if model already exists
    if os.path.isfile(destination_path):
        print(f"Model already exists at: {destination_path}")

        # Automatically load model with saved hyperparameters
        model = ResNetTransferLM.load_from_checkpoint(destination_path)
        print("Model loaded")
    else:
        # Move network to specified device
        network.to(device)

        model = ResNetTransferLM(network, NUM_CLASSES, **kwargs)
        print("Model created")

    # Tensorboard
    logger = TensorBoardLogger(save_dir=tb_path, name=model_name)

    # Trainer
    accelerator, num_threads = get_accelerator_info()

    trainer = pl.Trainer(
        default_root_dir=RESNET_TRANSFER_CHECKPOINT_PATH,
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
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )

    test_loader = data.DataLoader(
        test_data,
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
    model = ResNetTransferLM.load_from_checkpoint(
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

    model_name = f"downstream-unfixed-{DATA_FLAG.value}"

    # Get pretrained model
    # TODO This function should be in root/utils.py and should be able to load
    # models other than SimCLR
    pretrained_path = os.path.join(SIMCLR_CHECKPOINT_PATH, PRETRAINED_FILE)
    pretrained_model = get_pretrained_model(pretrained_path)

    # Deep copy convolutional network
    network = deepcopy(pretrained_model.convnet)

    model, result = finetune_resnet(
        network,
        batch_size=128,
        train_data=train_data,
        test_data=test_data,
        max_epochs=MAX_EPOCHS,
        lr=0.001,
        momentum=0.9,
    )

    summarise()