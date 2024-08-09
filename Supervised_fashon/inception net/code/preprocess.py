from torchvision import  transforms
from Dataset import CustomImageDataset
from config import full_data_dir, batch_size, test_fraction
from torch.utils.data import  DataLoader, random_split

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomRotation(degrees=10),  # Randomly rotate the image by degrees within (-10, 10)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_loaders():
    dataset = CustomImageDataset(root_dir=full_data_dir, transform=preprocess)


    train_size = int((1 - test_fraction) * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders for training and testation
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader, test_dataloader



