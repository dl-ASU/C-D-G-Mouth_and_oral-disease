import os
import cv2
import numpy as np
import pyheif
import rawpy
from PIL import Image

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
    return img

# Paths
base_path = 'DRP'
output_path = 'preprocessed_images'

# Create output directory if it does not exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Parameters
img_size = (300, 300)

# Initialize lists to store data
data = []
labels = []
cites = []

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

                # Read and resize image
                img = read_image(img_path)
                if img is not None:
                    img = cv2.resize(img, img_size)

                    # Replace spaces in image name with underscores
                    img_name_clean = img_name.replace(' ', '_')
                    new_img_name = f"{site}_{classs}_{i}"\

                    # Append image data and label
                    data.append(img)
                    labels.append(classs)
                    cites.append(site)
                    i = i + 1
                else:
                    print(img_path)

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)
cites = np.array(cites)

# Save the data and labels for later use
np.save(os.path.join(output_path, 'images.npy'), data)
np.save(os.path.join(output_path, 'labels.npy'), labels)
np.save(os.path.join(output_path, 'cites.npy'), cites)