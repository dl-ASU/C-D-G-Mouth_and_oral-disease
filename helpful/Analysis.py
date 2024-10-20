import numpy as np

from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def PredictionDistribution(error_counter, class_names, site_names, title):
    # Initialize counters for each site and label combination
    site_label_counts = {site: {label: 0 for label in class_names} for site in site_names}

    # Populate the site_label_counts dictionary with predictions
    for (true_class, pred_class, site_index), count in error_counter.items():
        site_name = site_names[site_index]
        pred_label = class_names[pred_class]
        site_label_counts[site_name][pred_label] += count

    # Display the analysis
    print(f"Site-Specific {title}:")
    for site, label_counts in site_label_counts.items():
        print(f"\nSite: {site}")
        total = sum(label_counts.values())
        for label, count in label_counts.items():
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  Predicted {label}: {count} times ({percentage:.2f}%)")

    # Identify sites with strong biases
    print("\nSites with Strong Prediction Biases:")
    for site, label_counts in site_label_counts.items():
        total = sum(label_counts.values())
        max_label = max(label_counts, key=label_counts.get)
        max_count = label_counts[max_label]
        percentage = (max_count / total) * 100 if total > 0 else 0

        # Check if the prediction for one label dominates the others
        if percentage > 80:  # Adjust the threshold as needed
            print(f"Site: {site} has a strong bias towards predicting '{max_label}' ({percentage:.2f}%)")

    # Visualize the site-specific prediction distribution as bar plots
    plt.figure(figsize=(14, 10))
    for i, (site, label_counts) in enumerate(site_label_counts.items(), 1):
        plt.subplot(4, 3, i)
        plt.bar(label_counts.keys(), label_counts.values(), color=['blue', 'green', 'red'])
        plt.title(site)
        plt.ylabel('Count')
        plt.xlabel('Predicted Label')
        plt.ylim(0, max(max(label_counts.values()), 10))
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()


def accuracy(error_counter, class_names, site_names):
    # Initialize counters for each site and label combination
    site_label_true_pred = {site: {label: {'correct': 0, 'total': 0} for label in class_names} for site in site_names}

    # Populate the site_label_true_pred dictionary with correct and total counts
    for (true_class, pred_class, site_index), count in error_counter.items():
        site_name = site_names[site_index]
        true_label = class_names[true_class]
        pred_label = class_names[pred_class]

        # Increment the total count for the true label
        site_label_true_pred[site_name][true_label]['total'] += count

        # Increment the correct count if the prediction was accurate
        if true_class == pred_class:
            site_label_true_pred[site_name][true_label]['correct'] += count

    # Calculate and display the accuracy for each label at each site
    print("Site-Specific Accuracy for Each Label:")
    for site, label_counts in site_label_true_pred.items():
        print(f"\nSite: {site}")
        for label, counts in label_counts.items():
            correct = counts['correct']
            total = counts['total']
            accuracy = (correct / total) * 100 if total > 0 else 0
            print(f"  Label '{label}': {correct}/{total} correct ({accuracy:.2f}% accuracy)")

    # Visualize the accuracy for each label at each site as bar plots
    plt.figure(figsize=(14, 10))
    for i, (site, label_counts) in enumerate(site_label_true_pred.items(), 1):
        plt.subplot(4, 3, i)
        labels = list(label_counts.keys())
        accuracies = [(label_counts[label]['correct'] / label_counts[label]['total']) * 100 if label_counts[label]['total'] > 0 else 0 for label in labels]

        plt.bar(labels, accuracies, color=['blue', 'green', 'red'])
        plt.title(site)
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Label')
        plt.ylim(0, 100)
    plt.tight_layout()
    plt.suptitle("Accuracy for each site on each label (recall)")
    plt.show()


def ConfusionANDerrors(error_counter, class_names, site_names):
    # Extract true labels and predicted labels
    true_labels = []
    pred_labels = []

    for (true_class, pred_class, _), count in error_counter.items():
        true_labels.extend([true_class] * count)
        pred_labels.extend([pred_class] * count)

    # Convert to numpy arrays for further processing
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    # 1. Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # 2. Classification Report (Precision, Recall, F1-Score)
    print("Classification Report:")
    print(classification_report(true_labels, pred_labels, target_names=class_names))

    # 3. Site-specific Error Analysis
    site_errors = {site: 0 for site in site_names}
    for (true_class, pred_class, site_index), count in error_counter.items():
        if true_class != pred_class:
            site_errors[site_names[site_index]] += count

    # Display site-specific errors
    print("\nSite-specific Misclassification Counts:")
    for site, error_count in site_errors.items():
        print(f"{site}: {error_count}")

    # 4. Visualize Misclassifications (example)
    # Assuming you have access to the original images, you could plot some examples of misclassifications.
    # Here we just print the count for demonstration.
    print("\nMisclassified Examples:")
    for (true_class, pred_class, sample_index), count in error_counter.items():
        if true_class != pred_class:
            print(f"Sample Index {sample_index}: True Label = {class_names[true_class]}, Predicted Label = {class_names[pred_class]} (Count: {count})")


def confusionAnalysis(error_counter, class_names, site_names):
    # Initialize confusion matrix and total errors for each site
    confusion_matrix = {site: {label: {other_label: 0 for other_label in class_names} for label in class_names} for site in site_names}
    site_total_errors = {site: 0 for site in site_names}

    # Populate the confusion matrix and count total errors for each site
    for (true_class, pred_class, site_index), count in error_counter.items():
        site_name = site_names[site_index]
        true_label = class_names[true_class]
        pred_label = class_names[pred_class]

        # Increment the count for the true_label being classified as pred_label
        confusion_matrix[site_name][true_label][pred_label] += count

        # Increment the total errors for this site if there's a misclassification
        if true_class != pred_class:
            site_total_errors[site_name] += count

    # Display the confusion matrix results with contribution to overall error
    print("Confusion Matrix Analysis with Error Contribution:")
    for site, label_confusions in confusion_matrix.items():
        print(f"\nSite: {site}")
        total_errors = site_total_errors[site]
        for true_label, pred_counts in label_confusions.items():
            for pred_label, count in pred_counts.items():
                if true_label != pred_label:  # Only show misclassifications
                    contribution = (count / total_errors) * 100 if total_errors > 0 else 0
                    print(f"  True Label '{true_label}' classified as '{pred_label}': {count} times "
                          f"({contribution:.2f}% of site errors)")


def allinone(data_analysis, error_analysis, idx_to_class, idx_to_site):
    PredictionDistribution(error_analysis, idx_to_class, idx_to_site, "Model-Error Distribution")
    confusionAnalysis(data_analysis, idx_to_class, idx_to_site)
    PredictionDistribution(data_analysis, idx_to_class, idx_to_site, "Model-Predictions Distribution")
    accuracy(data_analysis, idx_to_class, idx_to_site)
    ConfusionANDerrors(data_analysis, idx_to_class, idx_to_site)