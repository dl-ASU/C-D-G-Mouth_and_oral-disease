import os
import torch
from config import device
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
from collections import Counter


def make_mask(batch_anatomical_location, num_classes):
  batch_size = batch_anatomical_location.shape[0]
  tensor = torch.zeros((batch_size, num_classes), dtype=torch.uint8)  # Create a tensor of False values
  tensor = tensor.to(device)

  for idx1 in range(batch_size):
    val = batch_anatomical_location[idx1]

    for idx2 in range(val*3, (val + 1)*3):
      tensor[idx1][idx2] = 1


  return tensor


def plot_distribution(labels, title):
    counter = Counter(labels)
    classes = list(counter.keys())
    counts = list(counter.values())

    plt.figure(figsize=(10, 5))
    plt.bar(classes, counts, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

def delete_corrupted_images(data_path):
    for root, dirs, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Verify if it's an image
            except (UnidentifiedImageError, OSError) as e:
                print(f"Deleting corrupted image: {file_path} - {e}")
                os.remove(file_path)


