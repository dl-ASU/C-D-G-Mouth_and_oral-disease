import torch
from torch import nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from transformations import CustomRandomHorizontalFlip, CustomRandomVerticalFlip
from basic_model import BasicModel
from Dataset import CustomDataset, load_data
from base_model import device
from SEmodel import Model
from train import train

from base_model import device
from config import imagenet_mean, imagenet_std, batch_size, num_classes, num_epochs, num_sites, learning_rate, sche_milestones, gamma, l2, embedding_dim,dropout
from config import full_train_data_path, full_val_data_path, full_test_data_path

from helpful.vis_metrics import plots, DoAna
import warnings
warnings.filterwarnings("ignore")
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Model training parameters")

    # Add arguments for each hyperparameter
    parser.add_argument('--num_classes', type=int, default=num_classes, help="Number of classes")
    parser.add_argument('--num_sites', type=int, default=num_sites, help="Number of sites")
    parser.add_argument('--embedding_dim', type=int, default=embedding_dim, help="Embedding dimension")
    parser.add_argument('--learning_rate', type=float, default=learning_rate, help="Learning rate")
    parser.add_argument('--shape', type=int, default=299, help="Shape of the image")
    parser.add_argument('--dropout', type=float, default=dropout, help="Dropout probability")
    parser.add_argument('--n_heads', type=int, default=8, help="n_heads")
    parser.add_argument('--feedforward', type=int, default=512, help="feedforward")
    parser.add_argument('--n_layers', type=int, default=8, help="n_layers")
    
    parser.add_argument('--folder_name', type=str, default='Testing Images', help="Folder name of the testing images")
    parser.add_argument('--csv_name', type=str, default='output_csv', help="Csv output of the images")

    parser.add_argument('--num_epochs', type=int, default=num_epochs, help="Number of epochs")
    parser.add_argument('--l2', type=float, default=l2, help="L2 regularization")
    parser.add_argument('--batch_size', type=int, default=batch_size, help="Batch size")
    parser.add_argument('--gamma', type=float, default=gamma, help="Gamma")
    parser.add_argument('--optim', type=str, default="AdamW", help="Optimizer")

    parser.add_argument('--full_train_data_path', type=str, default=full_train_data_path, help="Full train data path")
    parser.add_argument('--full_val_data_path', type=str, default=full_val_data_path, help="Full validation data path")
    parser.add_argument('--full_test_data_path', type=str, default=full_test_data_path, help="Full test data path")
    parser.add_argument('--base', type=str, default=None, help="Base model")

    # Boolean flags
    parser.add_argument('--use_scheduler', action='store_true', help="Use scheduler")
    parser.add_argument('--freeze_base', action='store_true', help="Freeze Base True")
    parser.add_argument('--freeze', action='store_true', help="Freeze True")
    parser.add_argument('--to_freeze', type=int, default=0, help="parameters to freeze")

    parser.add_argument('--ignore', action='store_true', help="Disable symmetries (default: True, use --ignore to set False)")
    parser.add_argument('--oversample', action='store_true', help="Disable oversampling")
    parser.add_argument('--save_augmented', action='store_true', help="Save augmented data")
    parser.add_argument('--transform', action='store_true', help="More Transformations")

    return parser.parse_args()

args = parse_args()

if args.transform:
    # Define the transformations based on the description provided
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=args.shape, scale=(0.8, 1.0)),   # Randomly zoom in/out
        transforms.RandomRotation(degrees=35),                      # Rotate by 25 degrees
        CustomRandomHorizontalFlip(p=0.5),                          # Flip horizontally with a 50% chance
        CustomRandomVerticalFlip(p=0.5),                            # Flip vertically with a 50% chance
        transforms.RandomAffine(degrees=10, translate=(0.15, 0.15), shear=0.2),  # Width & height shift, and shear
        transforms.ColorJitter(brightness=(0.5, 1.0)),              # Brightness adjustment (0.5 to 1.0)
        transforms.ToTensor(),                                      # Convert image to tensor
        # transforms.RandomApply([transforms.Lambda(lambda x: x + (0.05 * torch.randn_like(x)))], p=0.5), # Channel shift
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize (optional)
    ])
else:
    # Define the transformations based on the description provided
    transform = transforms.Compose([
        # transforms.RandomResizedCrop(size=args.shape, scale=(0.9, 1.0)),
        # transforms.RandomRotation(degrees=25),
        # CustomRandomHorizontalFlip(p=0.5),
        # CustomRandomVerticalFlip(p=0.5),
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=0),
        transforms.Resize((args.shape, args.shape)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

test_transform = transforms.Compose([
    transforms.Resize((args.shape, args.shape)),
    transforms.ToTensor(),                                      # Convert image to tensor
    # transforms.RandomApply([transforms.Lambda(lambda x: x + (0.05 * torch.randn_like(x)))], p=0.5), # Channel shift
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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

torch.cuda.empty_cache()

model = BasicModel(num_classes=args.num_classes, num_sites=args.num_sites, base = args.base , freeze_base=args.freeze_base , dropout = args.dropout)
model = nn.DataParallel(model).to(device)

if args.optim == "AdamW":
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.l2,fused=True)

elif args.optim=='RMSprop':
    optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate,weight_decay=args.l2)
    
elif args.optim=='Adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate,weight_decay=args.l2)
    
else:
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate , weight_decay=args.l2)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = sche_milestones, gamma = args.gamma)
criterion = nn.CrossEntropyLoss()

train_accuracy, train_precision, train_recall, train_loss, test_accuracy, test_precision, test_recall, test_loss = train(model, criterion, optimizer, scheduler, train_loader, val_loader, args.num_epochs, args.base, args.freeze,args.to_freeze,args.use_scheduler)

plots(train_accuracy, train_precision, train_recall, train_loss, test_accuracy, test_precision, test_recall, test_loss, idx_to_class, idx_to_site, num_classes)
DoAna(model, test_loader, idx_to_class, idx_to_site,args.folder_name,args.csv_name)