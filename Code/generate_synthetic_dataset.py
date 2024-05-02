import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class SyntheticDataset(Dataset):
    def __init__(self, num_samples=60000, num_classes=10, image_shape=(3, 32, 32), train=True, transform=None):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_shape = image_shape
        self.train = train
        self.transform = transform

        # Generate synthetic data and labels
        self.data = self._generate_data()
        self.targets = self._generate_labels()

        # Define class names
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        self.dataset_split = self
       
    def _generate_data(self):
        # Generate synthetic images (random pixels)
        data = np.random.randint(0, 256, size=(self.num_samples, *self.image_shape), dtype=np.uint8)
        return data

    def _generate_labels(self):
        # Generate synthetic labels (random integers representing class indices)
        labels = np.random.randint(0, self.num_classes, size=self.num_samples)
        return labels.tolist()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return image and its corresponding label
        image = torch.tensor(self.data[idx]).float() / 255.0  # Normalize to [0, 1]
        label = self.targets[idx]
        return image, label
        
    def split_dataset(self):
        return self.dataset_split