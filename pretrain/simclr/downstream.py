from copy import deepcopy
import torch
import torch.nn as nn
import torch.utils.data as data

from utils import NUM_WORKERS


def encode_data_features(pretrained_model, dataset, device, batch_size=64):
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
        # Move images to specified device
        batch_imgs = batch_imgs.to(device)
        # f(.)
        batch_feats = network(batch_imgs)
        # Detach tensor from current graph and move to CPU
        feats.append(batch_feats.detach().cpu())
        labels.append(batch_labels)

        # TODO Temporary

        # [64, 512] (batch size = 64)
        # [64, 512] for STL-10
        # print(batch_feats.detach().cpu().shape)
        # [64, 1]
        # [64] for STL-10
        # print(batch_labels.shape)
        # assert False

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)

    # TODO Temporary

    # [1080, 512]
    # [5000, 512] for STL-10
    print(feats.shape)
    # [1080, 1]
    # [5000] for STL-10
    print(labels.shape)

    assert False

    # Sort images by labels
    labels, indexes = labels.sort()
    feats = feats[indexes]

    return data.TensorDataset(feats, labels)
