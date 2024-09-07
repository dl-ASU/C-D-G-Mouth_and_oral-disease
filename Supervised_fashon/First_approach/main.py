import torch
from torch import nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from transformations import CustomRandomHorizontalFlip, CustomRandomVerticalFlip

from Dataset import CustomDataset, load_data
from SEmodel import Model
from train import train
from base_model import device
from config import imagenet_mean, imagenet_std, batch_size, num_classes, num_epochs, num_sites, learning_rate, sche_milestones, gamma, l2, embedding_dim
from config import full_train_data_path, full_val_data_path, full_test_data_path

from helpful.vis_metrics import plots, DoAna

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Model training parameters")

    # Add arguments for each hyperparameter
    parser.add_argument('--num_classes', type=int, default=num_classes, help="Number of classes")
    parser.add_argument('--num_sites', type=int, default=num_sites, help="Number of sites")
    parser.add_argument('--embedding_dim', type=int, default=embedding_dim, help="Embedding dimension")
    parser.add_argument('--learning_rate', type=float, default=learning_rate, help="Learning rate")
    parser.add_argument('--shape', type=int, default=299, help="Learning rate")

    parser.add_argument('--num_epochs', type=int, default=num_epochs, help="Number of epochs")
    parser.add_argument('--l2', type=float, default=l2, help="L2 regularization")
    parser.add_argument('--batch_size', type=int, default=batch_size, help="Batch size")
    parser.add_argument('--gamma', type=float, default=gamma, help="Gamma")
    parser.add_argument('--full_train_data_path', type=str, default=full_train_data_path, help="Full train data path")
    parser.add_argument('--full_val_data_path', type=str, default=full_val_data_path, help="Full validation data path")
    parser.add_argument('--full_test_data_path', type=str, default=full_test_data_path, help="Full test data path")
    parser.add_argument('--base', type=str, default=None, help="Base model")

    # Boolean flags
    parser.add_argument('--ignore', action='store_true', help="Disable symmetries (default: True, use --ignore to set False)")
    parser.add_argument('--oversample', action='store_true', help="Disable oversampling")
    parser.add_argument('--save_augmented', action='store_true', help="Save augmented data")

    return parser.parse_args()

args = parse_args()
# Define the transformations based on the description provided
transform = transforms.Compose([
    transforms.RandomResizedCrop(size=args.shape, scale=(0.8, 1.0)),   # Randomly zoom in/out
    transforms.RandomRotation(degrees=25),                      # Rotate by 25 degrees
    CustomRandomHorizontalFlip(p=0.5),                          # Flip horizontally with a 50% chance
    CustomRandomVerticalFlip(p=0.5),                            # Flip vertically with a 50% chance
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=0.2),  # Width & height shift, and shear
    transforms.ColorJitter(brightness=(0.5, 1.0)),              # Brightness adjustment (0.5 to 1.0)
    transforms.ToTensor(),                                      # Convert image to tensor
    # transforms.RandomApply([transforms.Lambda(lambda x: x + (0.05 * torch.randn_like(x)))], p=0.5), # Channel shift
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize (optional)
])

stra_train_data, idx_to_class, idx_to_site = load_data(args.full_train_data_path, args.ignore)
stra_test_data, _, _ = load_data(args.full_test_data_path, args.ignore)
stra_val_data, _, _ = load_data(args.full_val_data_path, args.ignore)
print(idx_to_site)

train_set = CustomDataset(stra_train_data, transform, "train_distribution", oversample = args.oversample, idx_to_class=idx_to_class, idx_to_site=idx_to_site, save_augmented=args.save_augmented)
val_set = CustomDataset(stra_test_data, transform, "val_distribution", oversample = args.oversample, idx_to_class=idx_to_class, idx_to_site=idx_to_site, save_augmented=args.save_augmented)
test_set = CustomDataset(stra_test_data, "test_distribution", oversample=False)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers = 4, pin_memory =True)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers = 4, pin_memory =True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers = 4, pin_memory =True)

torch.cuda.empty_cache()
model = Model(num_classes=args.num_classes, num_sites=args.num_sites, base = args.base)
model = nn.DataParallel(model).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = sche_milestones, gamma = args.gamma)
criterion = nn.CrossEntropyLoss()

train_accuracy, train_precision, train_recall, train_loss, test_accuracy, test_precision, test_recall, test_loss = train(model, criterion, optimizer, scheduler, train_loader, val_loader, args.num_epochs)
plots(train_accuracy, train_precision, train_recall, train_loss, test_accuracy, test_precision, test_recall, test_loss, idx_to_class, idx_to_site, num_classes)
DoAna(model, test_loader, idx_to_class, idx_to_site)