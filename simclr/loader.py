import medmnist
from medmnist import INFO
import os
from torchvision import transforms

from const import DATASET_PATH


class Loader:
    def __init__(self):
        # Apply augmentation sequence
        self.contrast_transforms = transforms.Compose([
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

    def load(self, data_flag, split_type):
        DataClass = getattr(medmnist, INFO[data_flag.value]["python_class"])

        if not os.path.exists(DATASET_PATH):
            os.makedirs(DATASET_PATH)

        return DataClass(
            root=DATASET_PATH,
            split=split_type.value,
            transform=ContrastiveTransformations(self.contrast_transforms),
            download=True,
        )


class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]
