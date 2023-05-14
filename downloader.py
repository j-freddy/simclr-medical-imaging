import sys
import medmnist
from medmnist import INFO
import os
import torch
import torch.utils.data as data
from torchvision import transforms

from utils import DATASET_PATH, convert_to_rgb, get_labels, get_labels_as_tensor


class Downloader:
    def __init__(self):
        self.transforms = transforms.Compose([
            # Normalise to 3 channels
            transforms.Lambda(convert_to_rgb),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def load(self, data_flag, split_type, num_samples=-1, samples_per_class=-1):
        DataClass = getattr(medmnist, INFO[data_flag]["python_class"])

        if not os.path.exists(DATASET_PATH):
            os.makedirs(DATASET_PATH)

        dataclass = DataClass(
            root=DATASET_PATH,
            split=split_type.value,
            transform=self.transforms,
            download=True,
        )

        if num_samples == -1 and samples_per_class == -1:
            # Use entire data class
            return dataclass
        
        if num_samples > 0:
            assert samples_per_class == -1
            indices = torch.randperm(len(dataclass))[:num_samples]
        
        # For each class, if dataset contains less than @samples_per_class data
        # points, it just uses all the data points for that class.

        # The aim is to balance out dataset by performing undersampling
        
        if samples_per_class > 0:
            assert num_samples == -1

            class_indices = []
            num_classes = len(INFO[data_flag]["label"])

            # TODO This is quite inefficient
            # We go through the entire dataset and fetch corresponding labels
            labels = get_labels_as_tensor(dataclass)

            for idx in range(num_classes):
                samples = torch.where(labels == idx)[0]
                samples = samples[
                    torch.randperm(len(samples))[:samples_per_class]
                ]
                class_indices.append(samples)

            indices = torch.cat(class_indices)
        
        return data.Subset(dataclass, indices)
