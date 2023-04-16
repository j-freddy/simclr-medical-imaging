import os
import pytorch_lightning as pl
import torch.nn as nn
import torch.utils.data as data

from downloader import Downloader
from downstream.resnet.resnet_transferlm import ResNetTransferLM
from downstream.resnet.train import initialise_new_network
from utils import NUM_WORKERS, RESNET_TRANSFER_CHECKPOINT_PATH, SEED, SplitType, get_accelerator_info, parse_args_test


if __name__ == "__main__":
    (
        DATA_FLAG,
        MODEL_NAME,
    ) = parse_args_test()

    # Seed
    pl.seed_everything(SEED)

    # Load data
    downloader = Downloader()
    test_data = downloader.load(DATA_FLAG, SplitType.TEST)

    pretrained_path = os.path.join(
        RESNET_TRANSFER_CHECKPOINT_PATH,
        MODEL_NAME,
    )

    # Load model
    # Without this, getting errors due to missing backbone parameter
    resnet_base = initialise_new_network()

    resnet_base.fc = nn.Sequential(
        resnet_base.fc,
        nn.ReLU(inplace=True),
    )

    # Automatically load model with saved hyperparameters
    model = ResNetTransferLM.load_from_checkpoint(
        pretrained_path,
        backbone=resnet_base,
    )
    print("Model loaded")

    # Test

    test_loader = data.DataLoader(
        test_data,
        batch_size=min(128, len(test_data)),
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
        "test": test_result[0]["test_acc"]
    }

    print(result)