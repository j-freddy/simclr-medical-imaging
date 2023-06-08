from copy import deepcopy
import os

from pretrain.simclr.simclrlm import SimCLRLM
from utils import encode_data_features


def get_pretrained_model(path):
    """
    Load pretrained SimCLR model from a .ckpt checkpoint file.

    Args:
        path (str): The path to the checkpoint file.

    Returns:
        SimCLRLM: The loaded pretrained model.

    Raises:
        FileNotFoundError: If no file exists at the specified filepath.
    """
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
    """
    Given a pretrained model, extract the network encoder, remove the FC
    layers, pass the dataset through the encoder and return the encoded features.

    Args:
        pretrained_model (SimCLRLM): The pretrained SimCLR model.
        dataset (torch.utils.data.Dataset): The input dataset.
        device (torch.device): Device used for computation.
        batch_size (int, optional): The batch size. Defaults to 64.
        sort (bool, optional): Sort the features by labels. Defaults to True.

    Returns:
        torch.utils.data.TensorDataset: Dataset containing the encoded
            features and labels.
    """
    # Deep copy convolutional network
    network = deepcopy(pretrained_model.convnet)
    return encode_data_features(network, dataset, device, batch_size)


def summarise():
    print("Done! :)")
