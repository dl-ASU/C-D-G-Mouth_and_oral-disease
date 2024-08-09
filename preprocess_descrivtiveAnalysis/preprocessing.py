import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy import stats
from skimage import img_as_float, exposure
import os

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.classes = categories
        
        for category in categories:
            for region in regions:
                folder_path = os.path.join(root_dir, category, region)
                if os.path.isdir(folder_path):
                    for filename in os.listdir(folder_path):
                        img_path = os.path.join(folder_path, filename)
                        self.data.append((img_path, category_to_idx[category]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Apply preprocessing and transformations
            if self.transform:
                image = self.transform(image)
            
            return image, torch.tensor(label, dtype=torch.long)  # Convert label to tensor
        except Exception as e:
            print(f"Failed to load image {img_path}: {e}")
            return None, None
    
    def analyze_dataset(self, dataloader):
        image_sizes = []
        pixel_means = []
        pixel_stds = []
        labels = []

        for images, batch_labels in dataloader:
            # Calculate image sizes
            batch_size = images.size(0)
            for i in range(batch_size):
                img = images[i].numpy().transpose(1, 2, 0)  # Convert to (H, W, C)
                image_sizes.append(img.shape[:2])  # (H, W)
            
            # Calculate pixel intensity statistics
            pixel_means.append(images.mean(dim=[0, 2, 3]))  # Mean per channel
            pixel_stds.append(images.std(dim=[0, 2, 3]))  # Std per channel
            
            labels.extend(batch_labels.numpy())  # Collect labels

        # Convert lists to numpy arrays
        image_sizes = np.array(image_sizes)
        pixel_means = np.array([mean.numpy() for mean in pixel_means])
        pixel_stds = np.array([std.numpy() for std in pixel_stds])
        labels = np.array(labels)

        # Summary statistics for pixel intensities
        mean_per_channel = pixel_means.mean(axis=0)
        std_per_channel = pixel_stds.mean(axis=0)
        variance_per_channel = pixel_stds.var(axis=0)

        print("\nPixel Intensity Statistics:")
        print(f"Mean Pixel Value per Channel: {mean_per_channel}")
        print(f"Std Dev of Pixel Value per Channel: {std_per_channel}")
        print(f"Variance of Pixel Value per Channel: {variance_per_channel}")

        # Label distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_distribution = dict(zip(categories, counts))
        
        print("\nLabel Distribution:")
        for category, count in label_distribution.items():
            print(f"{category}: {count} samples ({count / len(labels) * 100:.2f}%)")
        
        # Pearson's correlation coefficient for pixel intensity
        if pixel_means.shape[0] > 1:  # At least two samples needed for correlation
            pearson_r = np.corrcoef(pixel_means.reshape(-1), pixel_stds.reshape(-1))[0, 1]
            print(f"\nPearson's R between Mean and Std Dev of Pixel Values: {pearson_r:.2f}")

# Define the transformations directly
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image
    transforms.Lambda(lambda x: img_as_float(np.array(x))),  # Convert PIL Image to float numpy array and normalize
    transforms.Lambda(lambda x: np.stack([exposure.equalize_hist(x[..., i]) for i in range(x.shape[2])], axis=-1)),  # Apply histogram equalization to each channel
    transforms.Lambda(lambda x: (x * 255).astype(np.uint8)),  # Convert to 8-bit image
    transforms.ToPILImage(),  # Convert back to PIL Image
    transforms.ToTensor()  # Convert image to tensor
])

# Define the path to the dataset
dataset_path = '/kaggle/input/dpr-dataset/preprocessed_images'

# Create dataset and dataloader
dataset = CustomDataset(root_dir=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Run the analysis
dataset.analyze_dataset(dataloader)
