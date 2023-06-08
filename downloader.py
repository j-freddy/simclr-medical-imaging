import sys
import medmnist
from medmnist import INFO
import os
import torch
import torch.utils.data as data
from torchvision import transforms

from utils import DATASET_PATH, convert_to_rgb, get_labels_as_tensor


class Downloader:
    """
    A pipeline to fetch images from the MedMNIST dataset.

    Attributes:
        transforms (torchvision.transforms.Compose): A fixed sequence of
            transforms to apply to every data point: convert to RGB, store as a
            Tensor and normalise image. This sequence is applied when data point
            is loaded (e.g. during a forward pass in training).
    """

    def __init__(self):
        self.transforms = transforms.Compose([
            # Normalise to 3 channels
            transforms.Lambda(convert_to_rgb),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def load(self, data_flag, split_type, num_samples=-1, samples_per_class=-1):
        """
        Load dataset corresponding to data_flag. If dataset is not found
        locally, download dataset from MedMNIST and load it.

        Args:
            data_flag (str): Data modality.
            split_type (SplitType): Whether to load train, validation or test
                data.
            num_samples (int, optional): The number of samples to load. Defaults
                to -1.
            samples_per_class (int, optional): The number of samples per class
                to load. Takes priority over num_samples if both specified.
                Defaults to -1.

        Returns:
            torch.utils.data.Dataset: The loaded dataset.
        """
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

            # Go through entire dataset and fetch corresponding labels
            labels = get_labels_as_tensor(dataclass)

            for idx in range(num_classes):
                samples = torch.where(labels == idx)[0]
                samples = samples[
                    torch.randperm(len(samples))[:samples_per_class]
                ]
                class_indices.append(samples)

            indices = torch.cat(class_indices)
        
        return data.Subset(dataclass, indices)
