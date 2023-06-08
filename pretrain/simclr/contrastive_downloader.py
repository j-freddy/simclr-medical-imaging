import medmnist
from medmnist import INFO
import os
import torch
import torch.utils.data as data

from pretrain.simclr.aug_sequences import AugmentationSequenceType, augmentation_sequence_map
from utils import DATASET_PATH


class ContrastiveDownloader:
    """
    A pipeline to fetch images from the MedMNIST dataset and apply augmentations
    to each image twice to create a positive pair. To use in contrastive
    learning.

    Attributes:
        transforms (torchvision.transforms.Compose): Apply a fixed sequence of
            data augmentations to each image. This sequence is applied when data
            point is loaded (e.g. during a forward pass in training).

    Args:
        aug_sequence (str, optional): Specify which augmentation sequence to
            use. Defaults to natural.

    Raises:
        ValueError: If aug_sequence is invalid.
    """

    def __init__(self, aug_sequence=AugmentationSequenceType.NATURAL.value):
        # Fetch augmentation sequence
        try:
            self.transforms = augmentation_sequence_map[aug_sequence]
        except KeyError:
            raise ValueError("Augmentation flag is invalid")

    def load(self, data_flag, split_type, num_samples=-1, views=2):
        """
        Load dataset corresponding to data_flag. If dataset is not found
        locally, download dataset from MedMNIST and load it.

        Args:
            data_flag (str): Data modality.
            split_type (SplitType): Whether to load train, validation or test
                data.
            num_samples (int, optional): The number of samples to load. Defaults
                to -1.
            views (int, optional): The number of augmented images created from
                the same original image. Defaults to 2.

        Returns:
            torch.utils.data.Dataset: The loaded dataset.
        """
        DataClass = getattr(medmnist, INFO[data_flag]["python_class"])

        if not os.path.exists(DATASET_PATH):
            os.makedirs(DATASET_PATH)

        dataclass = DataClass(
            root=DATASET_PATH,
            split=split_type.value,
            transform=ContrastiveTransformations(self.transforms, views),
            download=True,
        )

        if num_samples == -1:
            # Use entire data class
            return dataclass

        indices = torch.randperm(len(dataclass))[:num_samples]
        return data.Subset(dataclass, indices)


class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]
