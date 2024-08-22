import torch
from torch import nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from Dataset import CustomDataset, get_data
from Wmodel import Model
from train import train
from base_model import device
from config import imagenet_mean, imagenet_std, batch_size, num_classes, num_epochs, num_sites, learning_rate, sche_milestones, gamma, l2
from vis_metrics import plots, DoAna

transformation = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.RandomRotation(degrees=15),
    transforms.CenterCrop(224),
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.20, hue=0),
    # transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.025, 0.5))], p=0.5),  # Adding noise
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])


stra_train_data, stra_test_data, idx_to_class, idx_to_site = get_data()

train_set = CustomDataset(stra_train_data, transformation, "train distribution")
test_set = CustomDataset(stra_test_data, transformation, "test distribution")

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers = 4, pin_memory =True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers = 4, pin_memory =True)

torch.cuda.empty_cache()

model = Model(num_classes=num_classes, num_sites=num_sites, base = "google").to(device)
# model = nn.DataParallel(model).to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=l2)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = sche_milestones, gamma = gamma)

criterion = nn.CrossEntropyLoss()
train_accuracy, train_precision, train_recall, train_loss, test_accuracy, test_precision, test_recall, test_loss = train(model, criterion, optimizer, scheduler, train_loader, test_loader, num_epochs)

plots(train_accuracy, train_precision, train_recall, train_loss, test_accuracy, test_precision, test_recall, test_loss, idx_to_class, idx_to_site, num_classes)
DoAna(model, test_loader, idx_to_class, idx_to_site)