import os
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
from torchvision import transforms

class MultispectralDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.samples = []
        self.labels = []
        self.label_to_index = {}
        
        # Load dataset
        self._load_dataset()

    def _load_dataset(self):
        for index, label in enumerate(sorted(os.listdir(self.directory))):
            label_dir = os.path.join(self.directory, label)
            if os.path.isdir(label_dir):
                self.label_to_index[label] = index
                for file_name in os.listdir(label_dir):
                    if file_name.endswith('.tif'):
                        self.samples.append(os.path.join(label_dir, file_name))
                        self.labels.append(index)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        with rasterio.open(img_path) as src:
            img = src.read() / 65535.0  # Normalize
            img = np.transpose(img, (1, 2, 0))  # Change to HxWxC for PyTorch
        
        if self.transform:
            img = self.transform(img)

        return img, label
