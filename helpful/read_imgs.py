import os
import pyheif
import rawpy
from PIL import Image
import cv2
import numpy as np

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