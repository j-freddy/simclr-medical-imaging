import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
import torch.utils.data as data

from const import CHECKPOINT_PATH, NUM_WORKERS, SEED
from loader import Loader
from simclrlm import SimCLRLM
from utils import MedMNISTCategory, SplitType, setup_device, show_example_images


def train_simclr(train_data, val_data, filepath, batch_size, max_epochs=100,
    **kwargs):
    # Check if pretrained model exists
    if os.path.isfile(filepath):
        print(f"Found pretrained model at {filepath}, loading...")
        # Automatically load model with saved hyperparameters
        return SimCLRLM.load_from_checkpoint(filepath)

    print(f"No existing model found at: {filepath}")

    trainer = pl.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        # TODO Deprecated
        gpus=1 if str(device) == "cuda:0" else 0,
        max_epochs=max_epochs,
        # What to log
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True,
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
    model = SimCLRLM(max_epochs=max_epochs, **kwargs)
    
    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Load best checkpoint after training
    model = SimCLRLM.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )

    # Save pretrained model
    torch.save(model.state_dict(), filepath)

    return model


if __name__ == "__main__":
    data_flag = MedMNISTCategory.DERMA

    # Seed
    pl.seed_everything(SEED)

    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    print(f"Number of workers: {NUM_WORKERS}")

    # Load data
    loader = Loader()
    train_data = loader.load(data_flag, SplitType.TRAIN)
    val_data = loader.load(data_flag, SplitType.VALIDATION)
    test_data = loader.load(data_flag, SplitType.TEST)

    # Show example images
    # show_example_images(train_data)
    # show_example_images(val_data)
    # show_example_images(test_data)

    # Train model
    filename =f"pretrain-{data_flag.value}.ckpt"
    filepath = os.path.join(CHECKPOINT_PATH, filename)

    model = train_simclr(
        train_data,
        val_data,
        filepath,
        # TODO
        # max_epochs=100,
        max_epochs=1,
        batch_size=256,
        hidden_dim=128,
        lr=5e-4,
        temperature=0.07,
        weight_decay=1e-4,
    )

    print("Done :)")
