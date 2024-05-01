import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device='cuda'):
    """
    Train a PyTorch model.

    Args:
    - model (torch.nn.Module): The neural network model to train.
    - train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    - valid_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    - criterion (torch.nn.Module): The loss function.
    - optimizer (torch.optim.Optimizer): The optimization algorithm.
    - num_epochs (int): The number of epochs to train the model.
    - device (str): The device to train the model on ('cuda' or 'cpu').

    Returns:
    - model (torch.nn.Module): The trained model.
    """
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for name, param in model.named_parameters():
            if (name.endswith('weight')):
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Wrap train_loader with tqdm
        for batch_inputs, batch_targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

            # Forward pass
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()

            epoch_loss += loss.item()

            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Train Loss: {epoch_loss/len(train_loader):.4f}')

        # Validation phase
        model.eval()
        with torch.no_grad():
            for batch_inputs, batch_targets in tqdm(valid_loader, desc=f'validating Epoch {epoch+1}/{num_epochs}'):
                batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

                # Forward pass
                outputs = model(batch_inputs)
                _, predicted = torch.max(outputs, 1)
                total += batch_targets.size(0)
                correct += (predicted == batch_targets).sum().item()

        # Calculate accuracy
        accuracy = 100 * correct / total

        print(f'Epoch [{epoch+1}/{num_epochs}], Average Train Loss: {epoch_loss/len(train_loader):.4f}, Validation Accuracy: {accuracy:.2f}%')
        torch.save(model.state_dict(), f'model_{epoch+1}.pth')

    return model
