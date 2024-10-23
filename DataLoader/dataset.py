import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from script_dyn_crop import dynamic_cropping

# def custom_collate_fn(batch):
#     images, boxes, sites = zip(*batch)
    
#     # Stack images
#     images = torch.stack(images)

#     # Pad boxes
#     boxes = [b for b in boxes if b is not None]  # Exclude None boxes
#     boxes = pad_sequence(boxes, batch_first=True, padding_value=0)  # Pad with 0

#     # Convert sites to a list
#     sites = [site for site in sites if site is not None]  # Exclude None sites

#     return images, boxes, sites

class YOLODataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, crop=True, crop_percentage=0.25):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.crop = crop  
        self.crop_percentage = crop_percentage
        
        self.image_dir = os.path.join(root_dir, 'images', mode)
        self.label_dir = os.path.join(root_dir, 'labels', mode)
        
        self.image_paths = []
        self.label_paths = []
        self.sites = set()
        
        print(f"Processing area '{os.path.basename(root_dir)}' with {len(os.listdir(self.image_dir))} directories")
        
        for site in os.listdir(self.image_dir):
            site_path = os.path.join(self.image_dir, site)
            if os.path.isdir(site_path):
                self.sites.add(site)
                image_count = 0
                label_count = 0
                
                for img_file in os.listdir(site_path):
                    self.image_paths.append(os.path.join(site_path, img_file))
                    label_file = f"{os.path.splitext(img_file)[0]}.txt"
                    label_path = os.path.join(self.label_dir, site, label_file)
                    
                    if os.path.exists(label_path):
                        self.label_paths.append(label_path)
                        image_count += 1
                        label_count += 1
                
                print(f"Mode is: {mode} - Images in site '{site}' are : {image_count}, and Labels: {label_count}")

        print("*********************************************************************")
        print(f"Total number of images loaded in {mode} mode: {len(self.image_paths)}")
        print(f"Total number of sites: {len(self.sites)}")
        print(f"Finished processing area '{os.path.basename(root_dir)}' in mode {mode}'")
        print("*********************************************************************")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # Load current image
        image = Image.open(img_path).convert('RGB')

        # Load corresponding boxes to the current label
        with open(label_path, 'r') as f:
            boxes = [list(map(float, line.strip().split())) for line in f.readlines()]

        # Convert boxes to a tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)

        # Crop image if required
        if self.crop:
            cropped_image, original_image = dynamic_cropping(img_path, label_path, crop_percentage=self.crop_percentage)
            if cropped_image is not None:
                image = Image.fromarray(cropped_image)  # Convert cropped image back to PIL format
            else:
                print(f"Skipping image {img_path} due to cropping failure.")
                return None, None, None  # Skip this image

        # Apply transformations to the image if set
        if self.transform:
            image = self.transform(image)

        return image, boxes, os.path.basename(os.path.dirname(img_path))
    
if __name__ =="__main__":
    root_dir = '/kaggle/working/processed_data'

    target_size = (224, 224)  

    transform = transforms.Compose([
        transforms.Resize(target_size), 
        transforms.ToTensor(),
    ])

    dataset = YOLODataset(root_dir=root_dir, mode='train', transform=transform, crop_percentage=0.25)
    print(f"Total length of dataset now is {len(dataset)}")
