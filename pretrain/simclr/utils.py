from copy import deepcopy
import os

from pretrain.simclr.contrastive_downloader import ContrastiveDownloader
from pretrain.simclr.novel_contrastive_downloader import NovelContrastiveDownloader
from pretrain.simclr.simclrlm import SimCLRLM
from utils import AugmentationSequenceType, encode_data_features


def get_pretrained_model(path):
    # Check if pretrained model exists
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Pretrained model does not exist at: {path}"
        )

    return SimCLRLM.load_from_checkpoint(path)


def get_data_features_from_pretrained_model(
    pretrained_model,
    dataset,
    device,
    batch_size=64,
):
    # Deep copy convolutional network
    network = deepcopy(pretrained_model.convnet)
    return encode_data_features(network, dataset, device, batch_size)


def get_contrastive_downloader(augtype):
    if augtype == AugmentationSequenceType.NATURAL.value:
        return ContrastiveDownloader()
    elif augtype == AugmentationSequenceType.NOVEL.value:
        return NovelContrastiveDownloader()
    else:
        raise ValueError("Augmentation flag is invalid")


def summarise():
    print("Done! :)")
