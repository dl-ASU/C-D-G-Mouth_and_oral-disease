import numpy as np
import torch

from helpful.Analysis import allinone
from helpful.helpful import label_site_error_analysis, label_site_all_analysis
from base_model import device

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def plots(train_accuracy, train_precision, train_recall, train_loss, test_accuracy, test_precision, test_recall, test_loss, idx_to_class, idx_to_sites, num_classes = 3):
    # Plot Accuracy
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(train_accuracy, marker='o')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # Plot Precision
    plt.subplot(1, 3, 2)
    for i in range(num_classes):
        plt.plot(np.array(train_precision)[:,i], marker='o', label=f'Class {idx_to_class[i]}')
    plt.title('Training Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)

    # Plot Recall
    plt.subplot(1, 3, 3)
    for i in range(num_classes):
        plt.plot(np.array(train_recall)[:,i], marker='o', label=f'Class {idx_to_class[i]}')
    plt.title('Training Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('metrics_training.png')

    plt.close()
    # plt.show()

    # Plot Loss
    plt.figure(figsize=(6, 4))
    plt.plot(train_loss, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


    # Plot Accuracy
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(test_accuracy, marker='o')
    plt.title('Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # Plot Precision
    plt.subplot(1, 3, 2)
    for i in range(num_classes):
        plt.plot(np.array(test_precision)[:,i], marker='o', label=f'Class {idx_to_class[i]}')
    plt.title('Testing Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)

    # Plot Recall
    plt.subplot(1, 3, 3)
    for i in range(num_classes):
        plt.plot(np.array(test_recall)[:,i], marker='o', label=f'Class {idx_to_class[i]}')
    plt.title('Testing Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Plot Loss
    plt.figure(figsize=(6, 4))
    plt.plot(train_loss, marker='o')
    plt.title('Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


    # Plot Accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(train_accuracy, marker='o', label='Training Accuracy')
    plt.plot(test_accuracy, marker='o', label='Testing Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Precision
    for i in range(num_classes):
        plt.figure(figsize=(6, 4))
        plt.plot(np.array(train_precision)[:, i], marker='o', label=f'Training Precision Class {idx_to_class[i]}')
        plt.plot(np.array(test_precision)[:, i], marker='o', linestyle='--', label=f'Testing Precision Class {idx_to_class[i]}')
        plt.title('Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Plot Recall
    for i in range(num_classes):
        plt.figure(figsize=(6, 4))
        plt.plot(np.array(train_recall)[:, i], marker='o', label=f'Training Recall Class {idx_to_class[i]}')
        plt.plot(np.array(test_recall)[:, i], marker='o', linestyle='--', label=f'Testing Recall Class {idx_to_class[i]}')
        plt.title('Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Plot Loss
    plt.figure(figsize=(6, 4))
    plt.plot(train_loss, marker='o', label='Training Loss')
    plt.plot(test_loss, marker='o', label='Testing Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def DoAna(model, test_loader, idx_to_class, idx_to_site):
    model.eval()
    test_labels = []
    test_preds = []
    t_sites = []

    with torch.no_grad():
        for images, labels, sites in test_loader:
            images, labels, sites = images.to(device), labels.to(device), sites.to(device)

            # Forward pass
            outputs = model(images, sites)

            # Get predictions and true labels
            _, preds = torch.max(outputs, 1)

            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())
            t_sites.extend(sites.cpu().numpy())

    error_analysis = label_site_error_analysis(test_labels, test_preds, t_sites)
    data_analysis = label_site_all_analysis(test_labels, test_preds, t_sites)

    allinone(data_analysis, error_analysis, idx_to_class, idx_to_site)