import os
import random
import numpy as np
from PIL import Image
from collections import Counter
from torchvision import transforms

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
from config import test_size, val_size
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


class CustomDataset(Dataset):
    def __init__(self, all_data, transform=None, title = "Dist-data", oversample = True,  save_augmented=False, idx_to_class = None, idx_to_site = None, save_path="augmentedDataSet"):
        self.transform = transform
        self.title = title
        self.idx_to_class = idx_to_class
        self.idx_to_site = idx_to_site

        # Populate dataset attributes
        self.image_paths, self.labels, self.sites = zip(*all_data)
        if oversample:
            self._oversample()
        self.save_augmented = save_augmented
        self.save_path = save_path
        
        if self.save_augmented and not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.save_path = os.path.join(save_path, title)

        if self.save_augmented and not os.path.exists(self.save_path):
            os.makedirs(self.save_path)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        site = self.sites[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)
            if self.save_augmented:
                self.save_image(image, idx, self.idx_to_class[label], self.idx_to_site[site])

        return image, label, site

    def save_image(self, img, idx, label, site):
        # Convert tensor back to PIL image if using torchvision.transforms
        img_to_save = transforms.ToPILImage()(img)
        newPath = os.path.join(self.save_path, label)
        if not os.path.exists(newPath):
            os.makedirs(newPath)
        newPath = os.path.join(newPath, site)
        if not os.path.exists(newPath):
            os.makedirs(newPath)

        # Construct the image save path
        save_img_path = os.path.join(newPath, f"augmented_image_{label}_{site}_{idx}.jpg")
        
        # Save the image
        img_to_save.save(save_img_path)

    def _oversample(self):
        # Combine labels and sites into a unique key
        combined_keys = list(zip(self.labels, self.sites))

        # Count occurrences of each combination of label and site
        label_site_counts = Counter(combined_keys)

        # Find the maximum count for any combination of label and site
        max_count = max(label_site_counts.values())

        # New lists for oversampled data
        new_image_paths, new_labels, new_sites = [], [], []
        remainders = {}

        # Oversample each combination of label and site
        for img_path, label, site in zip(self.image_paths, self.labels, self.sites):
            key = (label, site)
            count = label_site_counts[key]
            num_duplicates = max_count // count

            if key not in remainders:
                remainder = max_count % count
                remainders[key] = remainder

            new_image_paths.extend([img_path] * num_duplicates)
            new_labels.extend([label] * num_duplicates)
            new_sites.extend([site] * num_duplicates)

            if remainders[key] > 0:
                new_image_paths.append(img_path)
                new_labels.append(label)
                new_sites.append(site)
                remainders[key] -= 1

        # Update the dataset with the oversampled data
        self.image_paths = new_image_paths
        self.labels = new_labels
        self.sites = new_sites
        self._shuffle()
        plt.figure(figsize=(10, 5))
        plt.bar(np.arange(len(label_site_counts.keys())), label_site_counts.values())
        plt.title(f'{self.title} - before oversampling')
        plt.show()
        new_coun = Counter(zip(self.labels, self.sites))
        plt.figure(figsize=(10, 5))
        plt.bar(np.arange(len(label_site_counts.keys())), new_coun.values())
        plt.title(f'{self.title} - after oversampling')
        plt.show()

    def _shuffle(self):
        combined = list(zip(self.image_paths, self.labels, self.sites))
        random.shuffle(combined)
        self.image_paths[:], self.labels[:], self.sites[:] = zip(*combined)

def get_data(full_data_path):
    # Collect all image paths, labels, and sites
    labels = []
    sites = []
    site_to_idx = {}
    idx_to_site = []
    idx_to_class = []

    all_data = []

    for class_idx, class_dir in enumerate(os.listdir(full_data_path)):
        if class_dir not in idx_to_class:
            idx_to_class.append(class_dir)

        class_path = os.path.join(full_data_path, class_dir)
        if os.path.isdir(class_path):
            for site in os.listdir(class_path):
                site_path = os.path.join(class_path, site)

                if os.path.isdir(site_path):
                    if site not in site_to_idx:
                        site_to_idx[site] = len(idx_to_site)
                        idx_to_site.append(site)
                    for img_name in os.listdir(site_path):
                        img_path = os.path.join(site_path, img_name)
                        if os.path.isfile(img_path):
                            all_data.append((img_path, class_idx, site_to_idx[site]))

    # Assuming all_data is a list of tuples: (image_path, label, site)
    image_paths, labels, sites = zip(*all_data)

    # Combine labels and sites into a single array for stratification
    labels_sites = np.array([f"{label}_{site}" for label, site in zip(labels, sites)])

    # Perform a stratified split
    train_indices, test_val_indices = train_test_split(
        range(len(all_data)), 
        test_size=test_size + val_size, 
        stratify=labels_sites
    )
    
    # Extract the corresponding labels and sites for test+val split
    test_val_labels_sites = np.array([labels_sites[i] for i in test_val_indices])

    test_indices, val_indices = train_test_split(
        test_val_indices,
        test_size=0.5,
        stratify=test_val_labels_sites
    )

    # Split the data based on the indices
    stra_train_data = [all_data[i] for i in train_indices]
    stra_val_data = [all_data[i] for i in val_indices]
    stra_test_data = [all_data[i] for i in test_indices]
    return stra_train_data, stra_test_data, stra_val_data, idx_to_class, idx_to_site

def load_data(data_path):
    # LoadData only
    site_to_idx = {}
    idx_to_site = []
    idx_to_class = []

    all_data = []

    for class_idx, class_dir in enumerate(os.listdir(data_path)):
        if class_dir not in idx_to_class:
            idx_to_class.append(class_dir)

        class_path = os.path.join(data_path, class_dir)
        if os.path.isdir(class_path):
            for site in os.listdir(class_path):
                site_path = os.path.join(class_path, site)

                if os.path.isdir(site_path):
                    if site not in site_to_idx:
                        site_to_idx[site] = len(idx_to_site)
                        idx_to_site.append(site)
                    for img_name in os.listdir(site_path):
                        img_path = os.path.join(site_path, img_name)
                        if os.path.isfile(img_path):
                            all_data.append((img_path, class_idx, site_to_idx[site]))

    return all_data, idx_to_class, idx_to_site