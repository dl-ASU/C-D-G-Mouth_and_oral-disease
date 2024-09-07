from torchvision import transforms
import torch
import torch.nn as nn

class CustomRandomHorizontalFlip(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.hflib = False

    def forward(self, img):
        if torch.rand(1) < self.p:
            self.hflib = True
            return transforms.functional.hflip(img)
        else:
          self.hflib = False
          return img

class CustomRandomVerticalFlip(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.vflib = False

    def forward(self, img):
        if torch.rand(1) < self.p:
            self.vflib = True
            return transforms.functional.vflip(img)
        else:
          self.vflib = False
          return img

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
    transforms.RandomResizedCrop(size=299, scale=(0.8, 1.0)),   # Randomly zoom in/out
    transforms.RandomRotation(degrees=25),                      # Rotate by 25 degrees
    CustomRandomHorizontalFlip(p=0.5),                          # Flip horizontally with a 50% chance
    CustomRandomVerticalFlip(p=0.5),                            # Flip vertically with a 50% chance
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=0.2),  # Width & height shift, and shear
    transforms.ColorJitter(brightness=(0.5, 1.0)),              # Brightness adjustment (0.5 to 1.0)
    transforms.ToTensor(),                                      # Convert image to tensor
    # transforms.RandomApply([transforms.Lambda(lambda x: x + (0.05 * torch.randn_like(x)))], p=0.5), # Channel shift
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize (optional)
])