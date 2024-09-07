import os
import shutil

# Paths to the original and test datasets
original_path = '/home/waleed/Documents/Medical/data_DRP/Original_DRP'  # Update with the actual path to Original_DRP
test_path = '/home/waleed/Documents/Medical/data_DRP/Test_Dataset'  # Update with the actual path to Test_Dataset
redundant_path = '/home/waleed/Documents/Medical/data_DRP/redundunt_Images'  # Folder to store redundant images

# Create the redundant folder
if not os.path.exists(redundant_path):
    os.makedirs(redundant_path)

def compare_and_move_redundant_images(original_dir, test_dir, redundant_dir):
    for root, _, files in os.walk(test_dir):
        relative_root = os.path.relpath(root, test_path)
        original_subfolder = os.path.join(original_path, relative_root)
        
        if not os.path.exists(original_subfolder):
            continue

        for file in files:
            original_file = os.path.join(original_subfolder, file)
            test_file = os.path.join(root, file)
            if os.path.exists(original_file):
                # Create the same folder structure in the redundant directory
                redundant_subfolder = os.path.join(redundant_path, relative_root)
                if not os.path.exists(redundant_subfolder):
                    os.makedirs(redundant_subfolder)
                
                # Copy both original and test images to the redundant folder
                newFile = file.split('.')[0]
                ext = file.split('.')[-1]
                shutil.copy2(original_file, os.path.join(redundant_subfolder, newFile + '_original.' + ext))
                shutil.copy2(test_file, os.path.join(redundant_subfolder, newFile + '_test.' + ext))
                print(file.split('.')[0])
                print(file)
                # Delete the redundant image from the test dataset
                os.remove(test_file)

# Execute the function
compare_and_move_redundant_images(original_path, test_path, redundant_path)

print("Redundant images moved and deleted from Test_Dataset.")
