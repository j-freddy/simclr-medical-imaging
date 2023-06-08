import os
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch.utils.data as data
from args_parser import Arguments

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
    SplitType,
    get_accelerator_info,
    setup_device,
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
    np.random.seed(SEED)

    # Initialise model as a new model
    # If pretrained_path specified, initialise model to pretrained model
    model = None

    if pretrained_path:
        model = get_pretrained_model(pretrained_path)
        print(f"Model initialised to pretrained model at: {pretrained_path}")
    else:
        model = SimCLRLM(max_epochs=max_epochs, **kwargs)
        print("Model initialised as new model")

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
    (
        DATA_FLAG,
        MAX_EPOCHS,
        AUG_TYPE,
        NUM_SAMPLES,
        INITIAL_PRETRAIN,
        MODEL_NAME,
    ) = Arguments.parse_args_train()

    # Seed
    pl.seed_everything(SEED)
    np.random.seed(SEED)

    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    print(f"Number of workers: {NUM_WORKERS}")

    # Load data
    downloader = ContrastiveDownloader(AUG_TYPE)
    train_data = downloader.load(DATA_FLAG, SplitType.TRAIN, NUM_SAMPLES)
    val_data = downloader.load(DATA_FLAG, SplitType.VALIDATION)

    model_name = MODEL_NAME or f"pretrain-{DATA_FLAG}"

    # Specifies path to model for further pretraining
    # Otherwise perform pretraining from a newly initialised resnet model
    initial_pretrain_path = os.path.join(
        SIMCLR_CHECKPOINT_PATH,
        INITIAL_PRETRAIN
    ) if INITIAL_PRETRAIN else None

    # Train model
    model = train_simclr(
        train_data,
        val_data,
        model_name,
        batch_size=min(256, len(train_data)),
        max_epochs=MAX_EPOCHS,
        pretrained_path=initial_pretrain_path,
        hidden_dim=128,
        lr=5e-4,
        temperature=0.07,
        weight_decay=1e-4,
    )

    summarise()
