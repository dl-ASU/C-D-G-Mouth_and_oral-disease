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
    dataset = CustomImageDataset(root_dir=full_data_dir, transform=preprocess)
    plot_distribution(dataset.labels, title="Distribution of All Data Before Splitting")


    train_size = int((1 - validation_fraction) * len(dataset))
    validation_size = len(dataset) - train_size
    train_split, validation_split = random_split(dataset, [train_size, validation_size])


    train_labels = [dataset.labels[i] for i in train_split.indices]
    train_image_paths = [dataset.image_paths[i] for i in train_split.indices]


    validation_labels = [dataset.labels[i] for i in validation_split.indices]
    validation_image_paths = [dataset.image_paths[i] for i in validation_split.indices]


    train_dataset = CustomImageDataset(labels=train_labels, image_paths=train_image_paths, transform = preprocess)
    validation_dataset = CustomImageDataset(labels=validation_labels, image_paths=validation_image_paths, transform = preprocess)


    plot_distribution(train_dataset.labels, title="Training Data Distribution Before Oversampling")
    train_dataset._oversample()
    plot_distribution(train_dataset.labels, title="Training Data Distribution After Oversampling")

    plot_distribution(validation_dataset.labels, title="Validation Data Distribution")


    common_images = set(train_dataset.image_paths).intersection(set(validation_dataset.image_paths))
    if common_images:
        raise ValueError(f"Data leakage detected! {len(common_images)} common images found between training and validation sets.")
    else: 
        print("NO Data leakage")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader, validation_dataloader


