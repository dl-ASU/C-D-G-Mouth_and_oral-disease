# dataset.py
import torch
from torch.utils.data import random_split
from data_preprocessing import CustomDataset, create_dataloader
from config import DATASET_PATH, BATCH_SIZE

def split_dataset(dataset, test_split=0.2):
    dataset_size = len(dataset)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - test_size
    return random_split(dataset, [train_size, test_size])

def prepare_data_loaders(dataset):
    train_dataset, test_dataset = split_dataset(dataset)
    train_dataloader = create_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = create_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_dataloader, test_dataloader
