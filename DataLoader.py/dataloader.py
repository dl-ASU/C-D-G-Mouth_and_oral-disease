import os
import torch
from torchvision import transforms
from PIL import Image
from script_dyn_crop import dynamic_cropping

class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and labels.
            mode (string): 'train' or 'test' to select the dataset.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        
        # Set up the image and label directories based on the mode
        self.image_dir = os.path.join(root_dir, 'images', mode)
        self.label_dir = os.path.join(root_dir, 'labels', mode)
        
        # List all image paths
        self.image_paths = []
        self.label_paths = []
        self.sites = set()  # To keep track of unique sites
        
        print(f"Processing area '{os.path.basename(root_dir)}' with {len(os.listdir(self.image_dir))} directories")
        
        for site in os.listdir(self.image_dir):
            site_path = os.path.join(self.image_dir, site)
            if os.path.isdir(site_path):
                self.sites.add(site)  # Add to set of unique sites
                image_count = 0
                label_count = 0
                
                for img_file in os.listdir(site_path):
                    # Append the image path regardless of its extension
                    self.image_paths.append(os.path.join(site_path, img_file))
                    label_file = f"{os.path.splitext(img_file)[0]}.txt"  # Replace any extension with .txt
                    label_path = os.path.join(self.label_dir, site, label_file)
                    
                    if os.path.exists(label_path):
                        self.label_paths.append(label_path)
                        image_count += 1
                        label_count += 1  # Increment only if label exists
                
                print(f"Mode is:{mode} - Images in site'{site}' are : {image_count}, and Labels: {label_count}")
                
        print("*********************************************************************")
        print(f"Total number of images loaded in {mode} mode: {len(self.image_paths)}")
        print(f"Total number of sites: {len(self.sites)}")
        print(f"Finished processing area '{os.path.basename(root_dir)} in mode {mode}'")
        print("*********************************************************************")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Load labels
        with open(label_path, 'r') as f:
            boxes = [list(map(float, line.strip().split())) for line in f.readlines()]

        if self.transform:
            image = self.transform(image)

        # convert boxes to a tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)

        # debugging statement for current item
        print(f"Loading image with path: {os.path.basename(img_path)}, in Site: {os.path.basename(os.path.dirname(img_path))}, Labels: {boxes.size(0)} boxes.")

        # returns image tensor,boxes tensor and the site of this image
        return image, boxes, os.path.basename(os.path.dirname(img_path))

if __name__=="__main__":

    root_dir = '/kaggle/working/processed_data'

    transform = transforms.Compose([
                                     # Resize images for YOLO
        transforms.ToTensor(),
    ])

    dataset = YOLODataset(root_dir=root_dir, mode='train', transform=transform)

    for i in range(5):
        print(f"Length of dataset is {len(dataset)}")
        print("-----------------------------")
        image, boxes,site = dataset[i]
        print(f"Image {i} is {image} \n and Boxes : {boxes} in site {site}")
        print("-----------------------------")
