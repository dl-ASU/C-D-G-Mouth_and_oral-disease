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
base_path = 'DRP'
output_path = 'preprocessed_images'

# Create output directory if it does not exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Parameters
img_size = (600, 600)

# Initialize lists to store data
data = []
labels = []
cites = []

# Initialize counters and lists for statistical analysis
ext_counter = defaultdict(int)
dimensions = []

def save_batch(data, labels, cites, batch_idx):
    np.save(os.path.join(output_path, f'images_batch_{batch_idx}.npy'), np.array(data))
    np.save(os.path.join(output_path, f'labels_batch_{batch_idx}.npy'), np.array(labels))
    np.save(os.path.join(output_path, f'cites_batch_{batch_idx}.npy'), np.array(cites))

# Loop through directories
batch_idx = 0
batch_size = 500  # Process n images at a time
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

                    # Append image data and label
                    data.append(img)
                    labels.append(classs)
                    cites.append(site)
                    i += 1

                    # Save batch if it reaches batch size
                    if len(data) >= batch_size:
                        save_batch(data, labels, cites, batch_idx)
                        batch_idx += 1
                        data = []
                        labels = []
                        cites = []
                else:
                    print(f"Failed to read: {img_path}")

# Save remaining images
if data:
    save_batch(data, labels, cites, batch_idx)

# Distribution of dimensions
dimensions = np.array(dimensions)
np.save(os.path.join(output_path, 'dimentions.npy'), dimensions)