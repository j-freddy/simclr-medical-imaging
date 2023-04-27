import medmnist
from medmnist import INFO
import os
import torch
import torch.utils.data as data
from torchvision import transforms

from utils import DATASET_PATH, convert_to_rgb


class NovelContrastiveDownloader:
    def __init__(self):
        # Define augmentation sequence
        self.transforms = transforms.Compose([
            # Normalise to 3 channels
            transforms.Lambda(convert_to_rgb),
            # Transformation 1: random horizontal flip
            transforms.RandomHorizontalFlip(),
            # Transformation 5: Gaussian blur
            transforms.GaussianBlur(kernel_size=9),

            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def load(self, data_flag, split_type, num_samples=-1, views=2):
        DataClass = getattr(medmnist, INFO[data_flag]["python_class"])

        if not os.path.exists(DATASET_PATH):
            os.makedirs(DATASET_PATH)

        dataclass = DataClass(
            root=DATASET_PATH,
            split=split_type.value,
            transform=NovelContrastiveTransformations(self.transforms, views),
            download=True,
        )

        if num_samples == -1:
            # Use entire data class
            return dataclass
    
        indices = torch.randperm(len(dataclass))[:num_samples]
        # TODO Validate dataclass has samples from each class
        return data.Subset(dataclass, indices)


class NovelContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]
