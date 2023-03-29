import medmnist
from medmnist import INFO
import os
import torch
import torch.utils.data as data
from torchvision import transforms

from utils import DATASET_PATH, convert_to_rgb


class ContrastiveDownloader:
    def __init__(self):
        # Define augmentation sequence
        self.transforms = transforms.Compose([
            # Normalise to 3 channels
            transforms.Lambda(convert_to_rgb),
            # Transformation 1: random horizontal flip
            transforms.RandomHorizontalFlip(),
            # Transformation 2: crop-and-resize
            transforms.RandomResizedCrop(size=96),
            # Transformation 3: colour distortion
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.5,
                        contrast=0.5,
                        saturation=0.5,
                        hue=0.1
                    )
                ],
                p=0.8
            ),
            # Transformation 4: random greyscale
            transforms.RandomGrayscale(p=0.2),
            # Transformation 5: Gaussian blur
            transforms.GaussianBlur(kernel_size=9),

            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def load(self, data_flag, split_type, num_samples=-1):
        DataClass = getattr(medmnist, INFO[data_flag]["python_class"])

        if not os.path.exists(DATASET_PATH):
            os.makedirs(DATASET_PATH)

        dataclass = DataClass(
            root=DATASET_PATH,
            split=split_type.value,
            transform=ContrastiveTransformations(self.transforms),
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
