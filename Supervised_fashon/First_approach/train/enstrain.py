import torch
import numpy as np
from torch import optim
import torch.nn as nn

from helpful.helpful import print_trainable_parameters, setTrainable, FreezeFirstN
from config import dic, epochs_sch
from models.base_model import device
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score

# def train(model, train_loader, test_loader, args):

#     # Lists to store metrics
#     train_accuracy = []
#     train_precision = []
#     train_recall = []
#     train_loss = []

#     test_accuracy = []
#     test_precision = []
#     test_recall = []
#     test_loss = []

#     optimizers=[]
#     criterion = nn.CrossEntropyLoss()

#     for m in model.models: 
#         optimizers.append(optim.Adam(m.parameters(), lr=args.learning_rate , weight_decay=args.l2))
#         print_trainable_parameters(m)

#     print('Training started.')

#     for epoch in range(args.num_epochs):
#         model.train()
#         all_labels = []
#         all_preds = []
#         cum_loss = 0

#         for images, labels, sites in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}'):
#             images, labels, sites = images.to(device), labels.to(device), sites.to(device)

#             # Forward pass
#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             # Zero the parameter gradients
#             optimizers.zero_grad()
#             # Backward pass and optimize
#             loss.backward()
#             optimizers.step()

#             # Get predictions and true labels
#             _, preds = torch.max(outputs, 1)

#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(preds.cpu().numpy())
#             cum_loss += loss.item()


#         # Calculate metrics
#         epoch_accuracy = accuracy_score(all_labels, all_preds)
#         epoch_precision = precision_score(all_labels, all_preds, average=None)
#         epoch_recall = recall_score(all_labels, all_preds, average=None)
#         cum_loss /= len(train_loader)

#         # Append metrics to lists
#         train_accuracy.append(epoch_accuracy)
#         train_precision.append(epoch_precision.tolist())
#         train_recall.append(epoch_recall.tolist())
#         train_loss.append(cum_loss)

#         model.eval()
#         test_labels = []
#         test_preds = []
#         t_loss = 0

#         with torch.no_grad():
#             for images, labels, sites in tqdm(test_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}'):
#                 images, labels, sites = images.to(device), labels.to(device), sites.to(device)

#                 # Forward pass
#                 outputs = model(images)
#                 t_loss += criterion(outputs, labels).item()

#                 # Get predictions and true labels
#                 _, preds = torch.max(outputs, 1)

#                 test_labels.extend(labels.cpu().numpy())
#                 test_preds.extend(preds.cpu().numpy())

#         # Calculate metrics
#         Tepoch_accuracy = accuracy_score(test_labels, test_preds)
#         Tepoch_precision = precision_score(test_labels, test_preds, average=None)
#         Tepoch_recall = recall_score(test_labels, test_preds, average=None)
#         t_loss /= len(test_loader)

#         # Append metrics to lists
#         test_accuracy.append(Tepoch_accuracy)
#         test_precision.append(Tepoch_precision.tolist())
#         test_recall.append(Tepoch_recall.tolist())
#         test_loss.append(t_loss)


#         print(f'Epoch [{epoch + 1}/{args.num_epochs}], Loss: {cum_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Precision: {np.mean(epoch_precision):.4f}, Recall: {np.mean(epoch_recall):.4f}')
#         print(f'Epoch [{epoch + 1}/{args.num_epochs}], Loss: {t_loss:.4f}, Accuracy: {Tepoch_accuracy:.4f}, Precision: {np.mean(Tepoch_precision):.4f}, Recall: {np.mean(Tepoch_recall):.4f}')
#         print("----------------------------------------------------------------------------------------------")
    
#         if epoch%5==0 and epoch!=0:
#             torch.save(model.state_dict(), f"model_ensembel_epoch_{epoch}.pth")
    
#     print('Training finished.')
#     return train_accuracy, train_precision, train_recall, train_loss, test_accuracy, test_precision, test_recall, test_loss



# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def train(model, train_loader, test_loader, args):

    # Lists to store metrics
    train_accuracy = []
    train_precision = []
    train_recall = []
    train_loss = []

    test_accuracy = []
    test_precision = []
    test_recall = []
    test_loss = []

    optimizers = []
    criterion = nn.CrossEntropyLoss()

    # Initialize optimizers for each model in the ensemble
    for m in model.models: 
        optimizers.append(optim.Adam(m.parameters(), lr=args.learning_rate, weight_decay=args.l2))
        print_trainable_parameters(m)

    print('Training started.')

    for epoch in range(args.num_epochs):
        model.train()
        all_labels = [[] for _ in range(len(model.models))]
        all_preds = [[] for _ in range(len(model.models))]
        cum_losses = [0 for _ in range(len(model.models))]

        # Train each model independently
        for images, labels, sites in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}'):
            images, labels, sites = images.to(device), labels.to(device), sites.to(device)

            # Forward pass through each model in the ensemble
            outputs_list = model(images)

            for idx, (output, optimizer) in enumerate(zip(outputs_list, optimizers)):
                optimizer.zero_grad()

                # Compute loss and backward pass for each model
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                # Collect predictions for this model
                _, preds = torch.max(output, 1)
                all_labels[idx].extend(labels.cpu().numpy())
                all_preds[idx].extend(preds.cpu().numpy())
                cum_losses[idx] += loss.item()

        # Calculate and store metrics for each model
        epoch_accuracy = []
        epoch_precision = []
        epoch_recall = []

        for idx in range(len(model.models)):
            model_accuracy = accuracy_score(all_labels[idx], all_preds[idx])
            model_precision = precision_score(all_labels[idx], all_preds[idx], average=None)
            model_recall = recall_score(all_labels[idx], all_preds[idx], average=None)
            cum_losses[idx] /= len(train_loader)

            epoch_accuracy.append(model_accuracy)
            epoch_precision.append(np.mean(model_precision))
            epoch_recall.append(np.mean(model_recall))

        # Aggregate and store metrics across all models
        train_accuracy.append(np.mean(epoch_accuracy))
        train_precision.append(np.mean(epoch_precision))
        train_recall.append(np.mean(epoch_recall))
        train_loss.append(np.mean(cum_losses))

        # Evaluation (testing) phase
        model.eval()
        test_labels = [[] for _ in range(len(model.models))]
        test_preds = [[] for _ in range(len(model.models))]
        t_losses = [0 for _ in range(len(model.models))]

        with torch.no_grad():
            for images, labels, sites in tqdm(test_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}'):
                images, labels, sites = images.to(device), labels.to(device), sites.to(device)

                # Forward pass through each model in the ensemble
                outputs_list = model(images)

                for idx, output in enumerate(outputs_list):
                    # Compute loss
                    loss = criterion(output, labels)
                    t_losses[idx] += loss.item()

                    # Collect predictions
                    _, preds = torch.max(output, 1)
                    test_labels[idx].extend(labels.cpu().numpy())
                    test_preds[idx].extend(preds.cpu().numpy())

        # Calculate and store test metrics for each model
        Tepoch_accuracy = []
        Tepoch_precision = []
        Tepoch_recall = []

        for idx in range(len(model.models)):
            model_accuracy = accuracy_score(test_labels[idx], test_preds[idx])
            model_precision = precision_score(test_labels[idx], test_preds[idx], average=None)
            model_recall = recall_score(test_labels[idx], test_preds[idx], average=None)
            t_losses[idx] /= len(test_loader)

            Tepoch_accuracy.append(model_accuracy)
            Tepoch_precision.append(np.mean(model_precision))
            Tepoch_recall.append(np.mean(model_recall))

        # Aggregate and store test metrics across all models
        test_accuracy.append(np.mean(Tepoch_accuracy))
        test_precision.append(np.mean(Tepoch_precision))
        test_recall.append(np.mean(Tepoch_recall))
        test_loss.append(np.mean(t_losses))

        # Print epoch metrics
        print(f'Epoch [{epoch + 1}/{args.num_epochs}], Train Loss: {np.mean(cum_losses):.4f}, '
              f'Accuracy: {np.mean(epoch_accuracy):.4f}, Precision: {np.mean(epoch_precision):.4f}, '
              f'Recall: {np.mean(epoch_recall):.4f}')
        
        print(f'Epoch [{epoch + 1}/{args.num_epochs}], Test Loss: {np.mean(t_losses):.4f}, '
              f'Accuracy: {np.mean(Tepoch_accuracy):.4f}, Precision: {np.mean(Tepoch_precision):.4f}, '
              f'Recall: {np.mean(Tepoch_recall):.4f}')
        
        print("----------------------------------------------------------------------------------------------")
    
        if epoch % 5 == 0 and epoch != 0:
            torch.save(model.state_dict(), f"model_ensemble_epoch_{epoch}.pth")
    
    print('Training finished.')
    return train_accuracy, train_precision, train_recall, train_loss, test_accuracy, test_precision, test_recall, test_loss