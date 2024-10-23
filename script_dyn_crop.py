import cv2
import os
from helpful.read_imgs import read_image

def dynamic_cropping(image_path, txt_path, crop_percentage=0.25):

        image, _ = read_image(image_path)

        if image is None:
            print(f"Could not open {image_path.split('/')[0]}")

        height, width, _ = image.shape

        # Load corresponding .txt file

        #check if the txt file exists or not aslan 
        if not os.path.exists(txt_path):
            print(f"Could not find corresponding txt file for {image_path.split('/')[0]}")

        # Extract x_center, y_center from the .txt file
        with open(txt_path, 'r') as file:
            line = file.readline()
            _, x_center, y_center, w, h = map(float, line.split())

        x_center_pixel = int(x_center * width)
        y_center_pixel = int(y_center * height)
        w = int(w * width)
        h = int(h * height)

        min_dim = min(height, width)
        crop_size = int(crop_percentage * min_dim)
        crop_size = max(crop_size, w, h)

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

        return cropped_image, image