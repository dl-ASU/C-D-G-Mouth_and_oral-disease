import os
import argparse

import numpy as np 
import torch

from torchvision import datasets
from torchvision.transforms import v2

from helpful.sampling import *
from helpful.losses import *
from helpful.models import *
from helpful.dataLoader import *
from helpful.helpfulFunctions import *

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda') if cuda else torch.device('cpu')

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

BASE_DIR = os.getcwd()

# Define command-line arguments
parser = argparse.ArgumentParser(description="Specify the training parameters.")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size (default: 256)")
parser.add_argument("--epochs", type=int, default=15, help="Epochs (default: 15)")
parser.add_argument("--N", type=int, default=16, help="Negative samples prt instance per patch (default: 16)")

args = parser.parse_args()

trans = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = True),
    v2.RandomHorizontalFlip(),
    v2.RandomRotation(25),
    v2.RandomAffine(degrees=0, translate=(0, 0.2), scale=(1.0, 1.2), shear=20),
    v2.Normalize(mean=[0.5], std=[0.5])
])

if cuda:
    torch.cuda.empty_cache()

model = ModClassifier().to(device)
print_trainable_parameters(model)

# Setup training data
train_data = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=True, # get training data
    download=True, # download data if it doesn't exist on disk
    transform=trans, # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)

test_data = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=False, # get training data
    download=True, # download data if it doesn't exist on disk
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale = True)]) # images come as PIL format, we want to turn into Torch tensors
)

# Filter the datasets to include only the specified classes
classes = [0, 5]
train_data = FilteredDataset(train_data, classes)
test_data = FilteredDataset(test_data, classes)

print("range of values: ", train_data.data[0].min(), train_data.data[0].max())

# Create DataLoader
train_generator = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle = True)
test_generator = torch.utils.data.DataLoader(test_data, batch_size = args.batch_size, shuffle = False)

# Define the construction Loss function
loss_fn = nn.MSELoss()

# inistantiate your optimizer and the scheduler procedure
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [15, 35], gamma = 0.5)

# Define an empty lists to save the losses during the training process
train_loss_values = []
var_loss_values = []
cov_loss_values = []
con_loss_values = []
outputs = []
values_loss_contr = []
values_loss_recon = []
values_loss = []
values_loss_test = []
epoch_count = []

print(device)
print("Training Started ....")

do = nn.Dropout(0.2)
for epoch in range(args.epochs):

    # Training Mode
    model.train()

    train_loss = 0
    for train_x, train_y in train_generator:
      pos, neg = sample_contrastive_pairs_SL(train_x, train_y, args.N)
      train_x, pos, neg, train_y = train_x.to(device), pos.to(device), neg.to(device), train_y.to(device)

      # Feedforward
      emb_actu = model.encode(train_x)
      emb_pos = model.encode(pos)
      emb_neg = model.encode(neg.reshape(-1, 1, 28, 28)).reshape(-1, args.N, 2)
      y_pred = model.decode(emb_actu)

      # Calculate the loss function and the accuracy
      _std_loss = std_loss(emb_actu, emb_pos)
      _cov_loss = cov_loss(emb_actu, emb_pos)
      loss_recon = loss_fn(y_pred.view((-1, 28 * 28)), train_x.view((-1, 28 * 28)))
      loss_contra = contrastive_loss(emb_actu, emb_pos, emb_neg)

      train_loss += (loss_recon.item() + loss_contra.item() + _std_loss.item() + _cov_loss.item())
      loss = loss_recon + loss_contra + _std_loss + _cov_loss

      # At start of each Epoch
      optimizer.zero_grad()

      # Do the back probagation and update the parameters
      loss.backward()
      optimizer.step()
      values_loss_recon.append(loss_recon.item())
      values_loss_contr.append(loss_contra.item())

    train_loss /= len(train_generator)
    values_loss.append(train_loss)

    # Evaluation mode
    model.eval()

    with torch.inference_mode():
      test_loss = 0

      for test_x, test_y in test_generator:
          # pos, neg = sample_contrastive_pairs(test_x, test_y, N)

          test_x, test_y = test_x.to(device), test_y.to(device)

          # Feedforward again for the evaluation phase
          emb_actu = model.encode(test_x)
          # emb_pos = model.encode(pos)
          # emb_neg = model.encode(neg.reshape(-1, 1, 28, 28)).reshape(-1, N, 512)
          y_pred_test = model.decode(emb_actu)

          # Calculate the loss for the test dataset
          loss_recon = loss_fn(y_pred_test.view((-1, 28 * 28)), test_x.view((-1, 28 * 28)))
          # loss_contra = contrastive_loss(emb_actu, emb_pos, emb_neg)
          test_loss += loss_recon.item()

    test_loss /= len(test_generator)
    outputs.append((epoch, test_x, y_pred_test))

    # Append loss values for the training process
    values_loss_test.append(test_loss)
    epoch_count.append(epoch)
    print(f"Epoch : {epoch + 1} | training_Loss: {train_loss:.4f} | testing_Loss: {test_loss:.4f}")

# Denoising (Mask) AutoEncoder and Contrastive Learning in the Latent variable
for k in range(0, args.epochs, 4):
    plt.figure(figsize=(9, 2))
    plt.gray()
    imgs = outputs[k][1].cpu().detach().numpy()
    recon = outputs[k][2].cpu().detach().numpy()
    for i, item in enumerate(imgs):
        if i >= 9: break
        plt.subplot(2, 9, i+1)
        # item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
        # item: 1, 28, 28
        plt.imshow(item[0])

    for i, item in enumerate(recon):
        if i >= 9: break
        plt.subplot(2, 9, 9+i+1) # row_length + i + 1
        # item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
        # item: 1, 28, 28
        plt.imshow(item[0])


# Embedding images using the encoder
device = 'cuda'

# Evaluation mode
model.eval()
embeddings = []
labels = []
with torch.inference_mode():
  for test_x, test_y in test_generator:
      # Feedforward again for the evaluation phase
      embedding = model.forward(test_x.to("cuda"))
      labels.append(test_y.numpy())

      embeddings.append(embedding.cpu().detach().numpy())

# Convert lists to arrays
embeddings = np.concatenate(embeddings, axis=0)
labels = np.concatenate(labels, axis=0)

labels = {
    0: "Normal",
    1: "Low",
    2: "High",
}

# Plotting
plt.figure(figsize=(8, 8))
for class_value in np.unique(labels):
    class_mask = labels == class_value
    plt.scatter(embeddings[class_mask, 0], embeddings[class_mask, 1], label=f"{fashion_mnist_labels[class_value]}", alpha=0.5, s=3)

plt.legend()
plt.show()