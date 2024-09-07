import os
import shutil

# Paths to original datasets and the new combined folder
original_path = '/home/waleed/Documents/Medical/data_DRP/TotalDataset'
test_path = '/home/waleed/Documents/Medical/data_DRP/latest'
totaldataset_path = '/home/waleed/Documents/Medical/data_DRP/LatestTotalDataset'

# Create the TotalDataset folder
if not os.path.exists(totaldataset_path):
    os.makedirs(totaldataset_path)

# Function to copy all images from a dataset folder to TotalDataset while maintaining structure
def copy_images_to_totaldataset(source_path, destination_path):
    for root, _, files in os.walk(source_path):
        relative_root = os.path.relpath(root, source_path)
        destination_subfolder = os.path.join(destination_path, relative_root)
        
        if not os.path.exists(destination_subfolder):
            os.makedirs(destination_subfolder)
        
        for file in files:
            source_file = os.path.join(root, file)
            destination_file = os.path.join(destination_subfolder, file)
            
            # Copy the file to the destination folder
            shutil.copy2(source_file, destination_file)

# Copy images from Original_DRP to TotalDataset
copy_images_to_totaldataset(original_path, totaldataset_path)

# Copy images from Test_Dataset to TotalDataset
copy_images_to_totaldataset(test_path, totaldataset_path)

print("All images have been copied to LatestTotalDataset.")