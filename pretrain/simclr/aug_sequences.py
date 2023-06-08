from enum import Enum
from torchvision import transforms
from pretrain.simclr.custom_augs import RandomAdjustSharpness, RandomEqualize

from utils import convert_to_rgb


class AugmentationSequenceType(Enum):
    """
    Augmentation sequence for SimCLR pretraining

    Natural:
        random horizontal flip > crop-and-resize > colour distortion > random
        greyscale > Gaussian blur
    Simple:
        crop-and-resize > colour distortion > Gaussian blur
    Novel:
        random horizontal flip > crop-and-resize > colour distortion > random
        greyscale > Gaussian blur > random histogram equalisation > random
        sharpness
    Greyscale:
        random horizontal flip > crop-and-resize > random greyscale > Gaussian
        blur > random histogram equalisation > random sharpness
    """
    NATURAL = "natural"
    SIMPLE = "simple"
    NOVEL = "novel"
    NOVEL_GREYSCALE = "greyscale"


augmentation_sequence_map = {
    AugmentationSequenceType.NATURAL.value: transforms.Compose([
        # Normalise to 3 channels
        transforms.Lambda(convert_to_rgb),
        # Transformation 1: random horizontal flip
        transforms.RandomHorizontalFlip(),
        # Transformation 2: crop-and-resize
        transforms.RandomResizedCrop(size=28),
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
            p=0.8,
        ),
        # Transformation 4: random greyscale
        transforms.RandomGrayscale(p=0.2),
        # Transformation 5: Gaussian blur
        transforms.GaussianBlur(kernel_size=9),

        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]),

    AugmentationSequenceType.SIMPLE.value: transforms.Compose([
        # Normalise to 3 channels
        transforms.Lambda(convert_to_rgb),
        # Transformation 1: crop-and-resize
        transforms.RandomResizedCrop(size=28),
        # Transformation 2: colour distortion
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.5,
                    contrast=0.5,
                    saturation=0.5,
                    hue=0.1
                )
            ],
            p=0.8,
        ),
        # Transformation 3: Gaussian blur
        transforms.GaussianBlur(kernel_size=9),

        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]),

    AugmentationSequenceType.NOVEL.value: transforms.Compose([
        # Normalise to 3 channels
        transforms.Lambda(convert_to_rgb),
        # Transformation 1: random horizontal flip
        transforms.RandomHorizontalFlip(),
        # Transformation 2: crop-and-resize
        transforms.RandomResizedCrop(size=28),
        # Transformation 3: smaller colour distortion
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
        # Transformation 4: random greyscale
        transforms.RandomGrayscale(p=0.2),
        # Transformation 5: Gaussian blur
        transforms.GaussianBlur(kernel_size=9),
        # Transformation 6: Histogram equalisation and sharpness to tackle low
        # contrast
        RandomEqualize(0.5),
        RandomAdjustSharpness(factor_low=1, factor_high=10),

        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]),

    AugmentationSequenceType.NOVEL_GREYSCALE.value: transforms.Compose([
        # Normalise to 3 channels
        transforms.Lambda(convert_to_rgb),
        # Transformation 1: random horizontal flip
        transforms.RandomHorizontalFlip(),
        # Transformation 2: crop-and-resize
        transforms.RandomResizedCrop(size=28),
        # Transformation 3: Gaussian blur
        transforms.GaussianBlur(kernel_size=9),
        # Transformation 4: Histogram equalisation and sharpness to tackle low
        # contrast
        RandomEqualize(0.5),
        RandomAdjustSharpness(factor_low=1, factor_high=10),

        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]),
}
