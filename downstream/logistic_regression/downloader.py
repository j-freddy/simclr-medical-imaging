import medmnist
from medmnist import INFO
import os
from torchvision import transforms

from utils import DATASET_PATH, convert_to_rgb


class Downloader:
    def __init__(self):
        self.transforms = transforms.Compose([
            # Normalise to 3 channels
            transforms.Lambda(convert_to_rgb),
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
            transform=self.transforms,
            download=True,
        )
