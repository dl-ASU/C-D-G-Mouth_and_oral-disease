import torch
from torch import nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from Dataset import CustomDataset, load_data
from SEmodel import Model
from train import train
from base_model import device
from config import imagenet_mean, imagenet_std, batch_size, num_classes, num_epochs, num_sites, learning_rate, sche_milestones, gamma, l2
from config import full_train_data_path, full_val_data_path, full_test_data_path

from helpful.vis_metrics import plots, DoAna
from transformations import transform

stra_train_data, idx_to_class, idx_to_site = load_data(full_train_data_path)
stra_test_data, _, _ = load_data(full_test_data_path)
stra_val_data, _, _ = load_data(full_val_data_path)


train_set = CustomDataset(stra_train_data, transform, "train_distribution", idx_to_class=idx_to_class, idx_to_site=idx_to_site, save_augmented=True)
val_set = CustomDataset(stra_test_data, transform, "val_distribution", idx_to_class=idx_to_class, idx_to_site=idx_to_site, save_augmented=True)
test_set = CustomDataset(stra_test_data, "test_distribution", oversample=False)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers = 4, pin_memory =True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers = 4, pin_memory =True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers = 4, pin_memory =True)

torch.cuda.empty_cache()
model = Model(num_classes=num_classes, num_sites=num_sites, base = "inception").to(device)
# model = nn.DataParallel(model).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = sche_milestones, gamma = gamma)
criterion = nn.CrossEntropyLoss()

train_accuracy, train_precision, train_recall, train_loss, test_accuracy, test_precision, test_recall, test_loss = train(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs)
plots(train_accuracy, train_precision, train_recall, train_loss, test_accuracy, test_precision, test_recall, test_loss, idx_to_class, idx_to_site, num_classes)
DoAna(model, test_loader, idx_to_class, idx_to_site)