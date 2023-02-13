import os
import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch.utils.data as data

from const import CHECKPOINT_PATH, NUM_WORKERS, SEED
from contrastive_downloader import ContrastiveDownloader
from simclrlm import SimCLRLM
from utils import (
    MedMNISTCategory,
    SplitType,
    setup_device,
    show_example_images,
    summarise
)


def train_simclr(
    train_data,
    val_data,
    destination_path,
    batch_size,
    max_epochs=100,
    pretrained_path=None,
    **kwargs
):
    # Check if model already exists
    if os.path.isfile(destination_path):
        print(f"Model already exists at: {destination_path}")

        # Automatically load model with saved hyperparameters
        model = SimCLRLM.load_from_checkpoint(destination_path)

        print("Model loaded")
        return model

    trainer = pl.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        # TODO Deprecated
        gpus=1 if str(device) == "cuda:0" else 0,
        max_epochs=max_epochs,
        # What to log
        callbacks=[
            ModelCheckpoint(
                save_weights_only=False,
                mode="max",
                monitor="val_acc_top5"
            ),
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
        # Check if pretrained model exists
        if not os.path.isfile(pretrained_path):
            raise FileNotFoundError(
                f"Pretrained model does not exist at: {pretrained_path}"
            )

        model = SimCLRLM.load_from_checkpoint(pretrained_path)
        print(f"Model initialised to pretrained model at: {pretrained_path}")
    
    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Load best checkpoint after training
    model = SimCLRLM.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )

    # Save pretrained model
    # torch.save(model.state_dict(), destination_path)
    trainer.save_checkpoint(destination_path)

    return model


if __name__ == "__main__":
    DATA_FLAG = MedMNISTCategory.RETINA
    MAX_EPOCHS = 2

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
    test_data = downloader.load(DATA_FLAG, SplitType.TEST)

    # Show example images
    # show_example_images(train_data)
    # show_example_images(val_data)
    # show_example_images(test_data)
    # sys.exit()

    # Train model
    filename =f"pretrain-{DATA_FLAG.value}.ckpt"
    destination_path = os.path.join(CHECKPOINT_PATH, filename)

    model = train_simclr(
        train_data,
        val_data,
        destination_path,
        max_epochs=MAX_EPOCHS,
        batch_size=256,
        hidden_dim=128,
        lr=5e-4,
        temperature=0.07,
        weight_decay=1e-4,
    )

    # Supports further pretraining from initial pretrained model

    # model = train_simclr(
    #     train_data,
    #     val_data,
    #     destination_path,
    #     max_epochs=MAX_EPOCHS,
    #     batch_size=256,
    #     pretrained_path=os.path.join(CHECKPOINT_PATH, "initial-pretrain.ckpt")
    # )

    summarise()
