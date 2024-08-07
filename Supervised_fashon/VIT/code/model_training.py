# model_training.py
import torch
from transformers import ViTForImageClassification
from torch.optim import AdamW
from sklearn.metrics import precision_score, recall_score, accuracy_score
from config import NUM_EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH

def initialize_model(num_labels):
    model_name = "google/vit-base-patch16-224-in21k"
    model = ViTForImageClassification.from_pretrained(model_name, num_labels=num_labels)
    return model

def train_model(model, train_dataloader, test_dataloader, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    train_losses, val_losses, val_accuracies, val_precisions, val_recalls = [], [], [], [], []

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_dataloader)
        train_losses.append(epoch_train_loss)

        model.eval()
        val_running_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for images, labels in test_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).logits
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_running_loss / len(test_dataloader)
        epoch_val_accuracy = accuracy_score(all_labels, all_preds)
        epoch_val_precision = precision_score(all_labels, all_preds, average='weighted')
        epoch_val_recall = recall_score(all_labels, all_preds, average='weighted')

        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)
        val_precisions.append(epoch_val_precision)
        val_recalls.append(epoch_val_recall)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Accuracy: {epoch_val_accuracy:.4f}, "
              f"Precision: {epoch_val_precision:.4f}, Recall: {epoch_val_recall:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    return train_losses, val_losses, val_accuracies, val_precisions, val_recalls
