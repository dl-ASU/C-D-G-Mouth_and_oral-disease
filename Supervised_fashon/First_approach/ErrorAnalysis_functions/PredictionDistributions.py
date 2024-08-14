import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

def PredictionDistribution(error_counter, class_names, site_names):
    # Initialize counters for each site and label combination
    site_label_counts = {site: {label: 0 for label in class_names} for site in site_names}

    # Populate the site_label_counts dictionary with predictions
    for (true_class, pred_class, site_index), count in error_counter.items():
        site_name = site_names[site_index]
        pred_label = class_names[pred_class]
        site_label_counts[site_name][pred_label] += count

    # Display the analysis
    print("Site-Specific Prediction Distribution:")
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
    plt.show()