import os
from torch.utils.data import Dataset
import torch
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_map = {}

        # Create a mapping of class names to indices
        current_index = 0
        for anatomical_location in os.listdir(root_dir):
            location_path = os.path.join(root_dir, anatomical_location)
            if os.path.isdir(location_path) and not anatomical_location.startswith('.'):
                for risk_level in os.listdir(location_path):
                    risk_path = os.path.join(location_path, risk_level)
                    if os.path.isdir(risk_path) and not risk_level.startswith('.'):
                        class_name = f"{anatomical_location}_{risk_level}"
                        if class_name not in self.class_map:
                            self.class_map[class_name] = current_index
                            current_index += 1

                        for img_name in os.listdir(risk_path):
                            img_path = os.path.join(risk_path, img_name)
                            if img_path.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                                self.image_paths.append(img_path)
                                self.labels.append(self.class_map[class_name])

    def __len__(self):
        return len(self.image_paths)

    def get_location(self, labels):
        locations = torch.div(labels, 3, rounding_mode='floor')
        return locations

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        anatomical_location = self.get_location(label)

        return image, anatomical_location, label