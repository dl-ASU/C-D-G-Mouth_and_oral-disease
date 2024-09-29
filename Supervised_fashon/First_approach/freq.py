import os
from collections import Counter

def count_label_site_images(data_path):
    """
    Function to count the number of images for each label-site combination.
    
    Args:
        data_path (str): The path to the dataset.

    Returns:
        label_site_counts (dict): A dictionary with (label, site) as keys and image count as values.
    """
    site_to_idx = {}
    idx_to_site = []
    idx_to_class = []

    all_data = []

    # Traverse through the dataset directory
    for class_idx, class_dir in enumerate(os.listdir(data_path)):
        if class_dir not in idx_to_class:
            idx_to_class.append(class_dir)

        class_path = os.path.join(data_path, class_dir)
        if os.path.isdir(class_path):
            for site in os.listdir(class_path):
                site_path = os.path.join(class_path, site)


                if os.path.isdir(site_path):
                    if site not in site_to_idx:
                        site_to_idx[site] = len(idx_to_site)
                        idx_to_site.append(site)
                    for img_name in os.listdir(site_path):
                        img_path = os.path.join(site_path, img_name)
                        if os.path.isfile(img_path):
                            all_data.append((class_idx, site_to_idx[site]))

    # Count the number of occurrences of each (label, site) pair
    label_site_counts = Counter(all_data)

    # Convert the counts to a more readable dictionary with class and site names
    label_site_dict = {}
    for (label_idx, site_idx), count in label_site_counts.items():
        label = idx_to_class[label_idx]
        site = idx_to_site[site_idx]
        label_site_dict[(label, site)] = count

    return label_site_dict

# Example usage
data_path = "/home/waleed/Documents/Medical/data_DRP/LatestDataset_processed_299"
label_site_image_counts = count_label_site_images(data_path)

# Print the results
for label_site, count in label_site_image_counts.items():
    print(f"Label-Site: {label_site}, Count: {count}")
