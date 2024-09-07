from torchvision import transforms
import torch
from PIL import Image
import random
from config import our_mean, our_std, imagenet_mean,   imagenet_std

# Custom Transform for Horizontal Flip with Label Update
class ConditionalHorizontalFlip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, img, site_label):
        if random.random() < self.flip_prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)  # Perform horizontal flip
            # Update the site label after flip
            if site_label == 'buccal_mucosa_left':
                site_label = 'buccal_mucosa_right'
            elif site_label == 'buccal_mucosa_right':
                site_label = 'buccal_mucosa_left'
            elif site_label == 'lateral_border_of_tongue_left':
                site_label = 'lateral_border_of_tongue_right'
            elif site_label == 'lateral_border_of_tongue_right':
                site_label = 'lateral_border_of_tongue_left'
        return img, site_label

# Custom Transform for Vertical Flip with Label Update
class ConditionalVerticalFlip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, img, site_label):
        if random.random() < self.flip_prob:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)  # Perform vertical flip
            # Update the site label after flip
            if site_label == 'upper_labial_mucosa':
                site_label = 'lower_labial_mucosa'
            elif site_label == 'lower_labial_mucosa':
                site_label = 'upper_labial_mucosa'
        return img, site_label

# Explanation of each transformation:

# 1. RandomResizedCrop(size=224, scale=(0.8, 1.0)): Simulates the zoom range by cropping and resizing the image.
# 2. RandomRotation(degrees=25): Rotates images by ±25 degrees.
# 3. RandomHorizontalFlip and RandomVerticalFlip: Randomly flip images horizontally and vertically.
# 4. RandomAffine(translate=(0.1, 0.1), shear=0.2): Width shift (0.1), height shift (0.1), and shear (0.2).
# 5. ColorJitter(brightness=(0.5, 1.0)): Randomly adjust brightness (range from 0.5 to 1.0).
# 6. RandomApply([Lambda]): Applies a channel shift (by adding random noise to the channels).
# 7. ToTensor: Converts the image to a tensor.
# 8. Normalize: Optional normalization step to bring the image tensor into the desired range.

# Define the transformations based on the description provided
transform = transforms.Compose([
    # transforms.RandomResizedCrop(size=299, scale=(0.8, 1.0)),   # Randomly zoom in/out
    # transforms.RandomRotation(degrees=25),                      # Rotate by 25 degrees

    # transforms.RandomHorizontalFlip(p=0.5),                     # Flip horizontally with a 50% chance

    # transforms.RandomVerticalFlip(p=0.5),                       # Flip vertically with a 50% chance
    # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=0.2),  # Width & height shift, and shear
    transforms.ColorJitter(brightness=(0.5, 1.0)),              # Brightness adjustment (0.5 to 1.0)
    transforms.ToTensor(),                                      # Convert image to tensor
    # transforms.RandomApply([transforms.Lambda(lambda x: x + (0.05 * torch.randn_like(x)))], p=0.5), # Channel shift
    transforms.Normalize(mean=our_mean, std=our_std) # Normalize (optional)
])