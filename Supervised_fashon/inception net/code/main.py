# main.py
import torch
from model_training import initialize_model, train_model
from preprocess import get_loaders
from metrics import plot_metrics, plot_tsne
from config import device, CATEGORIES, num_classes
import numpy as np
from Util import make_mask

def main():
    model, optimizer, criterion = initialize_model()
    train_dataloader, test_dataloader = get_loaders()


    train_losses, test_losses, test_accuracies, test_precisions, test_recalls = train_model(model, optimizer, criterion, train_dataloader, test_dataloader)

    plot_metrics(train_losses, test_losses, test_accuracies, test_precisions, test_recalls)


    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for images, batch_anatomical_location, batch_targets in test_dataloader:
            images = images.to(device)

            mask = make_mask(batch_anatomical_location, num_classes)

            model(images, mask)
            outputs = model.features

            features.append(outputs.cpu().numpy())
            labels.append(batch_targets.numpy())
            
    features = np.concatenate(features)
    labels = np.concatenate(labels)

    labels = labels % 3

    plot_tsne(features, labels, CATEGORIES)


if __name__ == "__main__":
    main()
