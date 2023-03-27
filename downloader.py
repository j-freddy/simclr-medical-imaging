import medmnist
from medmnist import INFO
import os
import torch
import torch.utils.data as data
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

    def load(self, data_flag, split_type, num_samples=-1):
        DataClass = getattr(medmnist, INFO[data_flag.value]["python_class"])

        if not os.path.exists(DATASET_PATH):
            os.makedirs(DATASET_PATH)

        dataclass = DataClass(
            root=DATASET_PATH,
            split=split_type.value,
            transform=self.transforms,
            download=True,
        )

        if num_samples == -1:
            # Use entire data class
            return dataclass
        
        indices = torch.randperm(len(dataclass))[:num_samples]
        # TODO Validate dataclass has samples from each class
        return data.Subset(dataclass, indices)
