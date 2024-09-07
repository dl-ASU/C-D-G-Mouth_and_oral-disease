import os
from config import full_dataset
from Dataset import get_data
from PIL import Image

def save_images(dataset, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, (img_path, label, site) in enumerate(dataset):
        # Load image from the file path
        img = Image.open(img_path).convert('RGB')

        # Define the label and site directory paths
        label_dir = os.path.join(save_dir, idx_to_class[label])
        site_dir = os.path.join(label_dir, idx_to_site[site])
        if not os.path.exists(site_dir):
            os.makedirs(site_dir)

        # Save the image to the appropriate folder
        img_name = os.path.basename(img_path)
        img.save(os.path.join(site_dir, img_name))

# Define the base directory for saving the data
base_dir = "SplittedDataset"

# Get the stratified train, validation, and test datasets
stra_train_data, stra_val_data, stra_test_data, idx_to_class, idx_to_site = get_data(full_dataset)

# Save the images into their respective folders inside "PreDataset"
save_images(stra_train_data, os.path.join(base_dir, "train"))
save_images(stra_val_data, os.path.join(base_dir, "validation"))
save_images(stra_test_data, os.path.join(base_dir, "test"))


print("Data split and saved successfully.")