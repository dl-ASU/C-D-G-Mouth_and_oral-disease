import cv2
import numpy as np
import os


def dynamic_cropping(image_folder, txt_folder, output_folder, crop_percentage=0.5):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_filename in os.listdir(image_folder):
        if image_filename.endswith(".jpg") or image_filename.endswith(".png"):

            image_path = os.path.join(image_folder, image_filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Could not open {image_filename}")
                continue

            height, width, _ = image.shape
            min_dim = min(height, width)
            crop_size = int(crop_percentage * min_dim)

            # Load corresponding .txt file
            txt_filename = image_filename.replace(".jpg", ".txt").replace(".png", ".txt")
            txt_path = os.path.join(txt_folder, txt_filename)

            #check if the txt file exists or not aslan 
            if not os.path.exists(txt_path):
                print(f"Could not find corresponding txt file for {image_filename}")
                continue

            # Extract x_center, y_center from the .txt file
            with open(txt_path, 'r') as file:
                line = file.readline()
                _, x_center, y_center, _, _ = map(float, line.split())

            x_center_pixel = int(x_center * width)
            y_center_pixel = int(y_center * height)

            #will check later if the square is really square or not
            x1 = x_center_pixel - crop_size // 2
            y1 = y_center_pixel - crop_size // 2
            x2 = x_center_pixel + crop_size // 2
            y2 = y_center_pixel + crop_size // 2

            #shift right if the left side goes out of bounds
            if x1 < 0:
                x2 += abs(x1)
                x1 = 0

            #shift down if the top side goes out of bounds
            if y1 < 0:
                y2 += abs(y1)  
                y1 = 0

            #shift left if the right side goes out of bounds
            if x2 > width:
                x1 -= (x2 - width)  
                x2 = width

            #shift up if the bottom side goes out of bounds
            if y2 > height:
                y1 -= (y2 - height)
                y2 = height

            # Crop the image
            cropped_image = image[y1:y2, x1:x2]

            # Save the processed image
            output_image_path = os.path.join(output_folder, image_filename)
            cv2.imwrite(output_image_path, cropped_image)

            print(f"Processed and saved {output_image_path}")

if __name__ == "__main__":
    image_folder = 'working/runs/detect/predict/detected original img'  # Folder with images
    txt_folder = 'working/runs/detect/predict/detected one box txt'  # Folder with corresponding .txt files
    output_folder = 'working/runs/detect/predict/cropped_images'  # Folder to save cropped and processed images

    dynamic_cropping(image_folder, txt_folder, output_folder)

