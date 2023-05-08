import medmnist
from medmnist import INFO
import os
import torch
import torch.utils.data as data
from torchvision import transforms

from pretrain.simclr.aug_sequences import AugmentationSequenceType, augmentation_sequence_map
from utils import DATASET_PATH, convert_to_rgb


class ContrastiveDownloader:
    def __init__(self, aug_sequence=AugmentationSequenceType.NATURAL.value):
        # Fetch augmentation sequence
        try:
            self.transforms = augmentation_sequence_map[aug_sequence]
        except KeyError:
            raise ValueError("Augmentation flag is invalid")

    def load(self, data_flag, split_type, num_samples=-1, views=2):
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
        # TODO Validate dataclass has samples from each class
        return data.Subset(dataclass, indices)


class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]
