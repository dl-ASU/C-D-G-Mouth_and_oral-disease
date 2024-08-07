# config.py
import os

# Paths
DATASET_PATH = '/kaggle/input/dpr-dataset/preprocessed_images'
MODEL_SAVE_PATH = 'vit_model.pth'
PLOTS_SAVE_PATH = '/kaggle/working/plots.png'
TSNE_PLOT_SAVE_PATH = '/kaggle/working/tsne_plot.png'

# Categories
CATEGORIES = ["high", "low", "normal"]
CATEGORY_TO_IDX = {category: idx for idx, category in enumerate(CATEGORIES)}

# Regions
REGIONS = [
    "buccal_mucosa_left", "buccal_mucosa_right", "dorsum_of_tongue",
    "floor_of_mouth", "gingive", "lateral_broder_of_tongue_left",
    "lateral_broder_of_tongue_right", "lower_labial_mucosa",
    "palate", "upper_labial_mucosa", "ventral_of_tongue"
]

# Training configuration
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
