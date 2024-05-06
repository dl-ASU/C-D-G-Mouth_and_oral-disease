import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(root_folder, transform, test_size=0.2, random_state=42):
    """
    Loads and preprocesses the image data.

    Args:
    - root_folder (str): Path to the root folder containing the dataset.
    - transform (torchvision.transforms.Compose): Transformations to apply to the data.
    - test_size (float): The proportion of the dataset to include in the test split.
    - random_state (int): Controls the randomness of the training and testing data splits.

    Returns:
    - train_dataset (torchvision.datasets.DatasetFolder): Training dataset.
    - val_dataset (torchvision.datasets.DatasetFolder): Validation dataset.
    """
    dataset = datasets.ImageFolder(root=root_folder, transform=transform)

    # Define the sizes for training and validation sets
    train_size = int((1 - test_size) * len(dataset))
    val_size = len(dataset) - train_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = train_test_split(dataset, test_size=test_size, random_state=random_state)

    return train_dataset, val_dataset
