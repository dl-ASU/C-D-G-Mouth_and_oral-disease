import os
import json
import base64
import random
import argparse
from pathlib import Path

# extract image from base64 and save label
def process_json_file(json_file_path, image_dir, label_dir, is_train=True):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    image_data = data.get('imageData')
    image_filename = data.get('imagePath')
    if image_data:
        image_bytes = base64.b64decode(image_data)
        subdir = 'train' if is_train else 'test'
        image_path = os.path.join(image_dir, subdir, image_filename)
        with open(image_path, 'wb') as img_file:
            img_file.write(image_bytes)
    
    # corresponding label in YOLO format
    label_filename = f"{Path(image_filename).stem}.txt"
    label_path = os.path.join(label_dir, subdir, label_filename)
    
    with open(label_path, 'w') as label_file:
        for shape in data['shapes']:
            class_id = 0 if 'low' in json_file_path else 1
            points = shape['points']
            x_min, y_min = points[0]
            x_max, y_max = points[1]
            
            # YOLO format: x_center, y_center, width, height
            x_center = (x_min + x_max) / 2 / data['imageWidth']
            y_center = (y_min + y_max) / 2 / data['imageHeight']
            width = (x_max - x_min) / data['imageWidth']
            height = (y_max - y_min) / data['imageHeight']
            
            label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# iterate through dataset directories and process files
def process_dataset(source_dirs, split_ratio):
    for source_dir in source_dirs:
        for site in os.listdir(source_dir):
            site_path = os.path.join(source_dir, site)
            if os.path.isdir(site_path):
                json_files = [f for f in os.listdir(site_path) if f.endswith('.json')]
                random.shuffle(json_files)
                train_split_index = int(len(json_files) * split_ratio)
                
                # process each json file
                for i, json_file in enumerate(json_files):
                    json_file_path = os.path.join(site_path, json_file)
                    is_train = i < train_split_index
                    process_json_file(json_file_path, target_image_dir, target_label_dir, is_train)

# set up argument parsing
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
    
    # process the dataset
    process_dataset(args.source_dirs, args.target_image_dir, args.target_label_dir, args.split_ratio)
