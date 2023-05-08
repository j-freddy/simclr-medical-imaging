from copy import deepcopy
import os

from pretrain.simclr.simclrlm import SimCLRLM
from utils import encode_data_features


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


def summarise():
    print("Done! :)")
