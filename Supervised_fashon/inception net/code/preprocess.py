from torchvision import  transforms
from Dataset import CustomImageDataset
from config import full_data_dir, batch_size, validation_fraction
from torch.utils.data import  DataLoader, random_split
from Util import plot_distribution

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    # transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomRotation(degrees=15),  # Randomly rotate the image by degrees within (-10, 10)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_loaders():
    # Load the dataset without oversampling
    dataset = CustomImageDataset(root_dir=full_data_dir, transform=preprocess)

    # Plot distribution before oversampling
    plot_distribution(dataset.labels, title="Distribution of All Data Before Splitting")

    # Split the dataset into training and validation sets
    train_size = int((1 - validation_fraction) * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    # Extract labels for train and validation sets
    train_labels = [dataset.labels[i] for i in train_dataset.indices]
    validation_labels = [dataset.labels[i] for i in validation_dataset.indices]

    # Plot distribution before oversampling
    plot_distribution(train_labels, title="Training Data Distribution Before Oversampling")
    plot_distribution(validation_labels, title="Validation Data Distribution")

    # Apply oversampling to the training dataset only
    train_dataset.dataset.image_paths = [train_dataset.dataset.image_paths[i] for i in train_dataset.indices]
    train_dataset.dataset.labels = train_labels
    train_dataset.dataset._oversample()

    # Plot distribution after oversampling
    plot_distribution(train_dataset.dataset.labels, title="Training Data Distribution After Oversampling")

    # Create DataLoaders for training and validation
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader, validation_dataloader


