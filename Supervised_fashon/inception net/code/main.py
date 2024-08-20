# main.py
import torch
from model_training import initialize_model, train_model
from preprocess import get_loaders
from metrics import *
from config import device, num_classes
import numpy as np
from Util import make_mask



def main():
    model, optimizer, criterion = initialize_model()
    ## chat gpt over sampling all the classes (the data in the train loders is like this :batch_inputs_imgs, batch_anatomical_location, batch_targets )
    train_dataloader, validation_dataloader = get_loaders()


    train_losses, validation_losses, validation_accuracies, validation_precisions, validation_recalls = train_model(
        model, optimizer, criterion, train_dataloader, validation_dataloader
        )

    plot_metrics(train_losses, validation_losses, validation_accuracies, validation_precisions, validation_recalls)


    model.eval()
    features, y_pred, labels, sites = [], [], [], []

    with torch.no_grad():
        for images, batch_anatomical_location, batch_targets in validation_dataloader:
            images = images.to(device)

            mask = make_mask(batch_anatomical_location, num_classes)

            batch_y_pred = model(images, mask)

            outputs = model.features

            features.append(outputs.cpu().numpy())
            labels.append(batch_targets.numpy())

            sites.extend(batch_anatomical_location)

            
            y_pred.append(np.argmax(batch_y_pred.cpu().numpy(), axis=1))
            
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    y_pred = np.concatenate(y_pred)

    labels = labels % 3
    y_pred = y_pred % 3
    

    data_analysis = label_site_all_analysis(labels, y_pred, sites)

    run_error_analysis(data_analysis)



    plot_tsne(features, labels)
    plot_confusion_matrix(labels, y_pred)


if __name__ == "__main__":
    main()
