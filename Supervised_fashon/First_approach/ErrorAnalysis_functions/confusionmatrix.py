import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.metrics import classification_report, confusion_matrix
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