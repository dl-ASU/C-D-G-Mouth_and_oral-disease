import os
import json
import base64
import random
from pathlib import Path

source_dirs = ['low', 'high']
target_image_dir = 'dataset/images'
target_label_dir = 'dataset/labels'

# train|test split ratio
split_ratio = 0.8

for subdir in ['train', 'test']:
    Path(os.path.join(target_image_dir, subdir)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(target_label_dir, subdir)).mkdir(parents=True, exist_ok=True)

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

def main(single_json_file=None):
    if single_json_file:
        print(f"Processing single JSON file: {single_json_file}")
        process_json_file(single_json_file, target_image_dir, target_label_dir, is_train=True)
    else:
        # iterate through the dataset directories
        for source_dir in source_dirs:
            for site in os.listdir(source_dir):
                site_path = os.path.join(source_dir, site)
                if os.path.isdir(site_path):
                    json_files = [f for f in os.listdir(site_path) if f.endswith('.json')]
                    random.shuffle(json_files)
                    train_split_index = int(len(json_files) * split_ratio)
                    
                    for i, json_file in enumerate(json_files):
                        json_file_path = os.path.join(site_path, json_file)
                        is_train = i < train_split_index
                        process_json_file(json_file_path, target_image_dir, target_label_dir, is_train)

    print("Dataset preprocessing completed.")

single_json_file = 'IMG_20190525_123527.json'
main(single_json_file)
