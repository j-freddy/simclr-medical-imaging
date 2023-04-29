import medmnist
from medmnist import INFO
import os
import torch
import torch.utils.data as data
from torchvision import transforms

from custom_augs import RandomAdjustSharpness, RandomEqualize
from utils import DATASET_PATH, convert_to_rgb

# Medical images:
# - low contrast and noise
#   - solution: histogram equalisation enhances image contrast
#   - alternatively: apply high contrast convolution filter, then apply median
#     filter (or gaussian blur)
# - greyscale colour space
#   - solution: replace colour distortion with elastic deformation
# - large dimensions
#   - solution: if we get good performance with 28x28 MedMNIST images without
#     upscaling & artificially resizing it to something larger, then
#     self-supervised learning works fine if we downscale the original images
#     for the categories we tested

class NovelContrastiveDownloader:
    def __init__(self):
        # Define augmentation sequence
        self.transforms = transforms.Compose([
            # Normalise to 3 channels
            transforms.Lambda(convert_to_rgb),
            # Crop-and-resize
            transforms.RandomResizedCrop(size=28, scale=(0.5, 1)),
            # Histogram equalisation and sharpness to tackle low contrast
            RandomEqualize(0.5),
            RandomAdjustSharpness(factor_low=1, factor_high=10),
            # Apply smaller colour distortion
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.04,
                    )
                ],
                p=0.8,
            ),
            # Gaussian blur: kept to tackle noise
            transforms.GaussianBlur(kernel_size=3),
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
