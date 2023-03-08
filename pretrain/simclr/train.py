import os
import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch.utils.data as data

from pretrain.simclr.contrastive_downloader import ContrastiveDownloader
from pretrain.simclr.simclrlm import SimCLRLM
from pretrain.simclr.utils import (
    get_pretrained_model,
    summarise,
)
from utils import (
    NUM_WORKERS,
    SEED,
    SIMCLR_CHECKPOINT_PATH,
    MedMNISTCategory,
    SplitType,
    get_accelerator_info,
    setup_device,
    show_example_images,
)


def set_args():
    DATA_FLAG = MedMNISTCategory.BREAST
    MAX_EPOCHS = 2

    return (
        DATA_FLAG,
        MAX_EPOCHS,
    )


def train_simclr(
    train_data,
    val_data,
    model_name,
    batch_size,
    max_epochs=100,
    pretrained_path=None,
    **kwargs,
):
    destination_path = os.path.join(
        SIMCLR_CHECKPOINT_PATH,
        f"{model_name}.ckpt"
    )
    tb_path = os.path.join(SIMCLR_CHECKPOINT_PATH, "tb_logs")

    # Check if model already exists
    if os.path.isfile(destination_path):
        print(f"Model already exists at: {destination_path}")

        # Automatically load model with saved hyperparameters
        model = SimCLRLM.load_from_checkpoint(destination_path)

        print("Model loaded")
        return model

    # Tensorboard
    logger = TensorBoardLogger(save_dir=tb_path, name=model_name)

    # Trainer
    accelerator, num_threads = get_accelerator_info()

    trainer = pl.Trainer(
        default_root_dir=SIMCLR_CHECKPOINT_PATH,
        accelerator=accelerator,
        devices=num_threads,
        max_epochs=max_epochs,
        logger=logger,
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
    )

    # Do not require optional logging
    trainer.logger._default_hp_metric = None

    train_loader = data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )

    pl.seed_everything(SEED)

    # Initialise model as a new model
    # If pretrained_path specified, initialise model to pretrained model
    model = None

    if pretrained_path == None:
        model = SimCLRLM(max_epochs=max_epochs, **kwargs)
        print("Model initialised as new model")
    else:
        model = get_pretrained_model(pretrained_path)
        print(f"Model initialised to pretrained model at: {pretrained_path}")

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Load best checkpoint after training
    model = SimCLRLM.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )

    # Save pretrained model
    trainer.save_checkpoint(destination_path)

    return model


if __name__ == "__main__":
    DATA_FLAG, MAX_EPOCHS = set_args()

    # Seed
    pl.seed_everything(SEED)

    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    print(f"Number of workers: {NUM_WORKERS}")

    # Load data
    downloader = ContrastiveDownloader()
    train_data = downloader.load(DATA_FLAG, SplitType.TRAIN)
    val_data = downloader.load(DATA_FLAG, SplitType.VALIDATION)

    # Show example images
    # show_example_images(train_data)
    # show_example_images(val_data)
    # sys.exit()

    model_name = f"pretrain-{DATA_FLAG.value}"

    # Train model
    model = train_simclr(
        train_data,
        val_data,
        model_name,
        max_epochs=MAX_EPOCHS,
        batch_size=256,
        hidden_dim=128,
        lr=5e-4,
        temperature=0.07,
        weight_decay=1e-4,
    )

    # Further pretraining from initial pretrained model

    # model = train_simclr(
    #     train_data,
    #     val_data,
    #     destination_path,
    #     max_epochs=MAX_EPOCHS,
    #     batch_size=256,
    #     pretrained_path=os.path.join(
    #         SIMCLR_CHECKPOINT_PATH,
    #         "initial-pretrain.ckpt"
    #     )
    # )

    summarise()
