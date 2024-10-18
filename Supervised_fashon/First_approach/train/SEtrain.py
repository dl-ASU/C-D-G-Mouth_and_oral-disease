import torch
import torch.nn as nn
from torch import optim
import numpy as np

from helpful.helpful import print_trainable_parameters, setTrainable, FreezeFirstN
from config import dic, epochs_sch, sche_milestones
from models.base_model import device
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score
 
def train(model, train_loader, test_loader, args):

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = sche_milestones, gamma = args.gamma)
    criterion = nn.CrossEntropyLoss()

    # Lists to store metrics
    train_accuracy = []
    train_precision = []
    train_recall = []
    train_loss = []

    test_accuracy = []
    test_precision = []
    test_recall = []
    test_loss = []

    if args.freeze:
        FreezeFirstN(model, 10000)
    print_trainable_parameters(model)

    print('Training started.')

    for epoch in range(args.num_epochs):
        model.train()
        all_labels = []
        all_preds = []
        cum_loss = 0

        # set more parameters
        if args.freeze and epoch in epochs_sch.keys():
            setTrainable(model, dic[args.base][epochs_sch[epoch]])
            print_trainable_parameters(model)

        for images, labels, sites in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}'):
            images, labels, sites = images.to(device), labels.to(device), sites.to(device)

            # Forward pass
            outputs = model(images, sites)
            loss = criterion(outputs, labels)

            # Zero the parameter gradients
            optimizer.zero_grad()
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Get predictions and true labels
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            cum_loss += loss.item()

        scheduler.step()

        # Calculate metrics
        epoch_accuracy = accuracy_score(all_labels, all_preds)
        epoch_precision = precision_score(all_labels, all_preds, average=None)
        epoch_recall = recall_score(all_labels, all_preds, average=None)
        cum_loss /= len(train_loader)

        # Append metrics to lists
        train_accuracy.append(epoch_accuracy)
        train_precision.append(epoch_precision.tolist())
        train_recall.append(epoch_recall.tolist())
        train_loss.append(cum_loss)

        model.eval()
        test_labels = []
        test_preds = []
        t_loss = 0

        with torch.no_grad():
            for images, labels, sites in tqdm(test_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}'):
                images, labels, sites = images.to(device), labels.to(device), sites.to(device)

                # Forward pass
                outputs = model(images, sites)
                t_loss += criterion(outputs, labels).item()

                # Get predictions and true labels
                _, preds = torch.max(outputs, 1)

                test_labels.extend(labels.cpu().numpy())
                test_preds.extend(preds.cpu().numpy())
    
        # Calculate metrics
        Tepoch_accuracy = accuracy_score(test_labels, test_preds)
        Tepoch_precision = precision_score(test_labels, test_preds, average=None)
        Tepoch_recall = recall_score(test_labels, test_preds, average=None)
        t_loss /= len(test_loader)

        # Append metrics to lists
        test_accuracy.append(Tepoch_accuracy)
        test_precision.append(Tepoch_precision.tolist())
        test_recall.append(Tepoch_recall.tolist())
        test_loss.append(t_loss)

        if epoch%5==0 and epoch !=  0:
            torch.save(model.state_dict(), f"model_{args.base}_epoch_{epoch}.pth")

        print(f'Epoch [{epoch + 1}/{args.num_epochs}], Loss: {cum_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Precision: {np.mean(epoch_precision):.4f}, Recall: {np.mean(epoch_recall):.4f}')
        print(f'Epoch [{epoch + 1}/{args.num_epochs}], Loss: {t_loss:.4f}, Accuracy: {Tepoch_accuracy:.4f}, Precision: {np.mean(Tepoch_precision):.4f}, Recall: {np.mean(Tepoch_recall):.4f}')
        print("----------------------------------------------------------------------------------------------")
    print('Training finished.')
    return train_accuracy, train_precision, train_recall, train_loss, test_accuracy, test_precision, test_recall, test_loss