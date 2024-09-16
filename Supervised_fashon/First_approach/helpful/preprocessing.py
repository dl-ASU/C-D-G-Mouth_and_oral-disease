import os
import cv2
import numpy as np
import pyheif
import rawpy
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style()

def read_image(img_path):
    ext = os.path.splitext(img_path)[-1].lower()
    if ext in ['.jpg', '.jpeg', '.png']:
        img = cv2.imread(img_path)
    elif ext == '.heic':
        heif_file = pyheif.read(img_path)
        img = Image.frombytes(
            heif_file.mode, 
            heif_file.size, 
            heif_file.data, 
            "raw",
            heif_file.mode,
            heif_file.stride,
        )
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    elif ext in ['.cr2', '.nef']:
        with rawpy.imread(img_path) as raw:
            rgb = raw.postprocess()
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        img = None
    return img, ext

# Paths
base_path = 'data_DRP/LatestDataset'
output_path = 'data_DRP/LatestDataset_processed_244'

# Create output directory if it does not exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Parameters
img_size = (224, 244)

# Initialize lists to store data
data = []
labels = []
cites = []

# Initialize counters and lists for statistical analysis
ext_counter = defaultdict(int)
dimensions = []

# Loop through directories
for site in os.listdir(base_path):
    site_path = os.path.join(base_path, site)
    print(site)

    if os.path.isdir(site_path):
        for classs in os.listdir(site_path):
            class_path = os.path.join(site_path, classs)
            i = 0
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)

                # Skip system files
                if img_name == 'desktop.ini':
                    continue

                # Read and resize image
                img, ext = read_image(img_path)

                if img is not None:
                    # Update counters and lists for statistical analysis
                    ext_counter[ext] += 1
                    dimensions.append(img.shape[:2])

                    img = cv2.resize(img, img_size)

                    # Replace spaces in image name with underscores
                    img_name_clean = img_name.replace(' ', '_')
                    new_img_name = f"{site}_{classs}_{i}"

                    # Replace spaces in image name with underscores
                    img_name_clean = img_name.replace(' ', '_')
                    new_img_name = f"{classs}_{site}_{i}.jpg"
                    save_img_path = os.path.join(output_path, classs, site)
                    
                    # Create directories if they don't exist
                    if not os.path.exists(save_img_path):
                        os.makedirs(save_img_path)

                    # Save the image as JPEG
                    cv2.imwrite(os.path.join(save_img_path, new_img_name), img)

                    i += 1

                else:
                    print(f"Failed to read: {img_path}")