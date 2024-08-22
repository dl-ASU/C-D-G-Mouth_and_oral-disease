# Hyperparameters
num_classes = 3
num_sites = 11
embedding_dim = 128
learning_rate = 0.001
num_epochs = 30
l2 = 1e-6
batch_size = 24
gamma = 0.75

sche_milestones = [10, 20, 30, 40, 42, 43, 44, 45, 46, 47, 48, 49]


our_mean = [0.5027, 0.5017, 0.5018]
our_std = [0.2904, 0.2886, 0.2886]

imagenet_mean=[0.485, 0.456, 0.406]
imagenet_std=[0.229, 0.224, 0.225]

test_size = 0.2

full_data_path = "/home/waleed/Documents/Medical/test/preprocessed_images_256"