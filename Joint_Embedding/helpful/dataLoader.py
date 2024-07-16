import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def returnDataLabel(root_directory, class_To_idx, size = (299, 299)):
    data = []
    labels = []

    for folder in ["variation", "high_risk", "low_risk"]:
        folder_path = os.path.join(root_directory, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                img = Image.open(file_path).convert('RGB')
                img = img.resize(size)
                img_arr = np.array(img)
                data.append(img_arr)
                labels.append(class_To_idx[folder])
    data = np.stack(data)
    labels = np.array(labels).reshape(-1, 1)
    return data, labels

class CustomImageDataset(Dataset):
    def __init__(self, data = None, labels = None, transform= None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.data)

def returnDataLoader(root_directory, classToIdx, transform, batch_size, shuffle):
    data, labels = returnDataLabel(root_directory, classToIdx)
    dataset = CustomImageDataset(data, labels, transform)
    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = 4)

class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_sample = self.data[index]
        target = self.targets[index]
        
        # Convert to PyTorch tensors
        data_sample = torch.tensor(data_sample, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.long)
        
        return data_sample, target

class FilteredDataset(Dataset):
    def __init__(self, dataset, classes):
        self.dataset = dataset
        self.classes = classes
        self.indices = [i for i, label in enumerate(dataset.targets) if label in classes]
        self.targets = [dataset.targets[i] for i in self.indices]
        self.data = [dataset.data[i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.targets[idx]
        label = self.classes.index(label)
        return self.dataset.transform(img), label