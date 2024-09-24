# main.py
import torch
from config import DATASET_PATH, CATEGORIES
from data_preprocessing import CustomDataset, get_transform
from dataset import prepare_data_loaders
from model_training import initialize_model, train_model
from metrics import plot_metrics, plot_tsne

def main():
    transform = get_transform()
    dataset = CustomDataset(root_dir=DATASET_PATH, transform=transform)
    train_dataloader, test_dataloader = prepare_data_loaders(dataset)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = initialize_model(num_labels=len(CATEGORIES))
    model.to(device)

    train_losses, val_losses, val_accuracies, val_precisions, val_recalls = train_model(
        model, train_dataloader, test_dataloader, device
    )

    plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, NUM_EPOCHS)

    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for images, lbls in test_dataloader:
            images = images.to(device)
            outputs = model(images).logits
            features.append(outputs.cpu().numpy())
            labels.append(lbls.numpy())
    features = np.concatenate(features)
    labels = np.concatenate(labels)

    plot_tsne(features, labels, CATEGORIES)

if __name__ == "__main__":
    main()
