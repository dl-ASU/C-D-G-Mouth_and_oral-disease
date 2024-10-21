import os
import json
import base64
import random
import argparse
from pathlib import Path
import yaml

# extract image from base64 and save label in YOLO format
def process_json_file(json_file_path, image_dir, label_dir, area_subdir, is_train=True):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # process image data
    image_data = data.get('imageData')
    image_filename = data.get('imagePath')
    if image_data:
        image_bytes = base64.b64decode(image_data)
        subdir = 'train' if is_train else 'test'
        
        # create area subdirectory in the target directory
        target_image_area_dir = os.path.join(image_dir, subdir, area_subdir)
        Path(target_image_area_dir).mkdir(parents=True, exist_ok=True)
        
        image_path = os.path.join(target_image_area_dir, image_filename)
        with open(image_path, 'wb') as img_file:
            img_file.write(image_bytes)
    
    # process corresponding label
    label_filename = f"{Path(image_filename).stem}.txt"
    target_label_area_dir = os.path.join(label_dir, subdir, area_subdir)
    Path(target_label_area_dir).mkdir(parents=True, exist_ok=True)
    
    label_path = os.path.join(target_label_area_dir, label_filename)
    
    with open(label_path, 'w') as label_file:
        for shape in data['shapes']:
            # assign class_id based on the source directory
            class_id = 0 # if 'low' in json_file_path else 1
            points = shape['points']
            x_min, y_min = points[0]
            x_max, y_max = points[1]
            
            # convert to YOLO format: x_center, y_center, width, height
            x_center = (x_min + x_max) / 2 / data['imageWidth']
            y_center = (y_min + y_max) / 2 / data['imageHeight']
            width = (x_max - x_min) / data['imageWidth']
            height = (y_max - y_min) / data['imageHeight']
            
            label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# iterate through dataset directories and process files
def process_dataset(source_dirs, target_image_dir, target_label_dir, split_ratio):
    # ensure the directories for train and test splits exist
    for subdir in ['train', 'test']:
        Path(os.path.join(target_image_dir, subdir)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(target_label_dir, subdir)).mkdir(parents=True, exist_ok=True)
    
    # process dataset
    for source_dir in source_dirs:
        for site in os.listdir(source_dir):
            site_path = os.path.join(source_dir, site)
            if os.path.isdir(site_path):
                json_files = [f for f in os.listdir(site_path) if f.endswith('.json')]
                random.shuffle(json_files)
                train_split_index = int(len(json_files) * split_ratio)
                
                # count total JSON files
                total_json_files = len(json_files)
                print(f"Processing area '{os.path.basename(site_path)}' with {total_json_files} JSON files")
                
                # process each json file
                for i, json_file in enumerate(json_files):
                    json_file_path = os.path.join(site_path, json_file)
                    is_train = i < train_split_index
                    
                    # extract area subdirectory name
                    area_subdir = os.path.basename(site_path)
                    
                    # process the json file with the area subdir included in the target path
                    process_json_file(json_file_path, target_image_dir, target_label_dir, area_subdir, is_train)
                
                # count generated images and labels in train/test directories for the current area
                for subdir in ['train', 'test']:
                    target_image_area_dir = os.path.join(target_image_dir, subdir, area_subdir)
                    target_label_area_dir = os.path.join(target_label_dir, subdir, area_subdir)
                    
                    # count files in the target image and label directories
                    num_images = len([f for f in os.listdir(target_image_area_dir) if os.path.isfile(os.path.join(target_image_area_dir, f))])
                    num_labels = len([f for f in os.listdir(target_label_area_dir) if os.path.isfile(os.path.join(target_label_area_dir, f))])
                    
                    print(f"  {subdir} set - Images in '{area_subdir}': {num_images}, Labels: {num_labels}")
                
                print(f"Finished processing area '{area_subdir}'\n")

# Function to generate the YAML file for YOLO training
def generate_yaml_file(target_image_dir, target_label_dir, num_classes, class_names, output_yaml_path):
    yaml_data = {
        'train': os.path.join(target_image_dir, 'train'),
        'val': os.path.join(target_image_dir, 'test'),  # Assuming test set is used for validation
        'nc': num_classes,
        'names': class_names
    }

    # Save the YAML file
    with open(output_yaml_path, 'w') as yaml_file:
        yaml.dump(yaml_data, yaml_file, default_flow_style=False)
    
    print(f"YAML file saved at: {output_yaml_path}")

# setting up argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Process dataset and split into train/test.')
    parser.add_argument('--source_dirs', nargs='+', required=True, help='Full paths to source directories.')
    parser.add_argument('--target_image_dir', required=True, help='Full path to target image directory.')
    parser.add_argument('--target_label_dir', required=True, help='Full path to target label directory.')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Train/test split ratio.')
    return parser.parse_args()

if __name__ == '__main__':
    # parse command line arguments
    args = parse_args()
    process_dataset(args.source_dirs, args.target_image_dir, args.target_label_dir, args.split_ratio)

    # specify class details
    num_classes = 1
    class_names = ['disease']

    # Generate the dataset YAML file
    output_yaml_path = os.path.join(args.target_image_dir, 'dataset.yaml')
    generate_yaml_file(args.target_image_dir, args.target_label_dir, num_classes, class_names, output_yaml_path)