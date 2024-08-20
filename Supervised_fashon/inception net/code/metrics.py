# metrics.py
import matplotlib.pyplot as plt
from config import PLOTS_SAVE_PATH, TSNE_PLOT_SAVE_PATH, MATRIX_SAVE_PATH, CATEGORIES,LOCATIONS,num_epochs
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.manifold import TSNE
import os
import sys
from collections import Counter
sys.path.append(os.path.abspath(os.path.join('..','..' , 'First_approach')))
from  ErrorAnalysis_functions.confusionmatrix import *
from  ErrorAnalysis_functions.accuracy import *
from  ErrorAnalysis_functions.analysis import *
from  ErrorAnalysis_functions.PredictionDistributions import *




def plot_metrics(train_losses, validation_losses, validation_accuracies, validation_precisions, validation_recalls):
    sns.set()
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.plot(epochs, validation_losses, 'r-o', label='validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and validation Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, validation_accuracies, 'g-o', label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('validation Accuracy')

    plt.subplot(1, 3, 3)
    plt.plot(epochs, validation_precisions, 'r-o', label='Precision')
    plt.plot(epochs, validation_recalls, 'm-o', label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('validation Precision and Recall')
    plt.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_SAVE_PATH)
    plt.close()

def plot_tsne(features, labels):
    tsne = TSNE(n_components=2, random_state=0)
    tsne_features = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=labels, cmap='tab10')
    plt.legend(handles=scatter.legend_elements()[0], labels=CATEGORIES)
    plt.title("t-SNE visualization of inception net features")
    plt.savefig(TSNE_PLOT_SAVE_PATH)
    plt.close()



def plot_confusion_matrix(y_true, y_pred):
    # Generate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    
    # Save the plot
    plt.savefig(MATRIX_SAVE_PATH)
    plt.close()



def label_site_all_analysis(y_true, y_pred, sites):
    error_pairs = [(true, pred, site) for true, pred, site in zip(y_true, y_pred, sites)]
    error_analysis = Counter(error_pairs)
    return error_analysis

def run_error_analysis(data_analysis):
    confusionAnalysis(data_analysis, CATEGORIES, LOCATIONS)
    PredictionDistribution(data_analysis, CATEGORIES, LOCATIONS)
    accuracy(data_analysis, CATEGORIES, LOCATIONS)
    ConfusionANDerrors(data_analysis, CATEGORIES, LOCATIONS)

