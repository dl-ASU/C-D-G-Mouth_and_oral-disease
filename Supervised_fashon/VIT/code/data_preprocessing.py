# data_preprocessing.py
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from config import DATASET_PATH, CATEGORIES, CATEGORY_TO_IDX, REGIONS

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.classes = CATEGORIES

        for category in CATEGORIES:
            for region in REGIONS:
                folder_path = os.path.join(root_dir, category, region)
                if os.path.isdir(folder_path):
                    for filename in os.listdir(folder_path):
                        img_path = os.path.join(folder_path, filename)
                        self.data.append((img_path, CATEGORY_TO_IDX[category]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.long)  # Convert label to tensor
        except Exception as e:
            print(f"Failed to load image {img_path}: {e}")
            return None, None

def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
