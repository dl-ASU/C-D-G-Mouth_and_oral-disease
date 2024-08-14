import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

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
    plt.show()