import torch
from torch import nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from transformations import CustomRandomHorizontalFlip, CustomRandomVerticalFlip
from transformers import ViTForImageClassification
from Dataset import CustomDataset, load_data
from models.model import get_arch
from train.train import get_train
from config import imagenet_mean, imagenet_std, batch_size, num_classes, num_epochs, num_sites, learning_rate, sche_milestones, gamma, l2, embedding_dim,dropout
from config import full_train_data_path, full_val_data_path, full_test_data_path, parse_args

from helpful.vis_metrics import plots, DoAna
import warnings
warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda. is_available() else 'cpu'

args = parse_args()

if args.transform:
    # Define the transformations based on the description provided
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=args.shape, scale=(0.8, 1.0)),   # Randomly zoom in/out
        transforms.RandomRotation(degrees=35),                      # Rotate by 25 degrees
        CustomRandomHorizontalFlip(p=0.5),                          # Flip horizontally with a 50% chance
        # CustomRandomVerticalFlip(p=0.5),                            # Flip vertically with a 50% chance
        transforms.RandomAffine(degrees=10, translate=(0.15, 0.15), shear=0.2),  # Width & height shift, and shear
        transforms.ColorJitter(brightness=(0.5, 1.0)),              # Brightness adjustment (0.5 to 1.0)
        transforms.ToTensor(),                                      # Convert image to tensor
        # transforms.RandomApply([transforms.Lambda(lambda x: x + (0.05 * torch.randn_like(x)))], p=0.5), # Channel shift
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
else:
    # Define the transformations based on the description provided
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=args.shape, scale=(0.9, 1.0)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=0),
        transforms.Resize((args.shape, args.shape)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
 
test_transform = transforms.Compose([
    transforms.Resize((args.shape, args.shape)),
    transforms.ToTensor(),                                      # Convert image to tensor
    # transforms.RandomApply([transforms.Lambda(lambda x: x + (0.05 * torch.randn_like(x)))], p=0.5), # Channel shift
    transforms.Normalize(mean=imagenet_mean, std=imagenet_mean)
])
 
stra_train_data, idx_to_class, idx_to_site = load_data(args.full_train_data_path, args.ignore)
stra_test_data, _, _ = load_data(args.full_test_data_path, args.ignore)
stra_val_data, _, _ = load_data(args.full_val_data_path, args.ignore)

print(idx_to_site)
print(idx_to_class)

train_set = CustomDataset(stra_train_data, transform, "train_distribution", oversample = args.oversample, idx_to_class=idx_to_class, idx_to_site=idx_to_site, save_augmented=args.save_augmented, ignore=args.ignore)
val_set = CustomDataset(stra_test_data, transform, "val_distribution", oversample = args.oversample, idx_to_class=idx_to_class, idx_to_site=idx_to_site, save_augmented=args.save_augmented, ignore=args.ignore)
test_set = CustomDataset(stra_test_data, test_transform, title = "test_distribution", oversample=False, ignore=args.ignore)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers = 4, pin_memory =True)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers = 4, pin_memory =True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers = 4, pin_memory =True)

print(device)
torch.cuda.empty_cache()

print(args.arch, args.base)

model = get_arch(mode = args.arch, num_classes=args.num_classes, num_sites=args.num_sites, base = args.base)
model = nn.DataParallel(model).to(device)

train = get_train(args.arch, model, train_loader, test_loader, args) 
train_accuracy, train_precision, train_recall, train_loss, test_accuracy, test_precision, test_recall, test_loss = train(model, train_loader, val_loader, args)

plots(train_accuracy, train_precision, train_recall, train_loss, test_accuracy, test_precision, test_recall, test_loss, idx_to_class, idx_to_site, num_classes)
DoAna(model, test_loader, idx_to_class, idx_to_site, args.folder_name, args.csv_name)