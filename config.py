import argparse

# Hyperparameters
num_classes = 3
num_sites = 11
embedding_dim = 128
learning_rate = 0.001
num_epochs = 40
l2 = 0.01
batch_size = 32
gamma = 0.75
dropout=0.1
sche_milestones = [10, 20, 30, 40, 42, 43, 44, 45, 46, 47, 48, 49]

our_mean = [0.5027, 0.5017, 0.5018]
our_std = [0.2904, 0.2886, 0.2886]

imagenet_mean=[0.485, 0.456, 0.406]
imagenet_std=[0.229, 0.224, 0.225]

def parse_args():
    parser = argparse.ArgumentParser(description="Model training parameters")

    # Add arguments for each hyperparameter
    parser.add_argument('--num_classes', type=int, default=num_classes, help="Number of classes")
    parser.add_argument('--num_sites', type=int, default=num_sites, help="Number of sites")
    parser.add_argument('--embedding_dim', type=int, default=embedding_dim, help="Embedding dimension")
    parser.add_argument('--shape', type=int, default=299, help="Shape of the image")

    parser.add_argument('--folder_name', type=str, default='MisClassified_Images', help="Folder name of the testing images")
    parser.add_argument('--csv_name', type=str, default='MisClassified_Images', help="Csv output of the images")

    parser.add_argument('--num_epochs', type=int, default=num_epochs, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=batch_size, help="Batch size")

    parser.add_argument('--learning_rate', type=float, default=learning_rate, help="Learning rate")
    parser.add_argument('--optim', type=str, default="AdamW", help="Optimizer")
    parser.add_argument('--l2', type=float, default=l2, help="L2 regularization")
    parser.add_argument('--gamma', type=float, default=gamma, help="Gamma")
    parser.add_argument('--dropout', type=float, default=dropout, help="Dropout probability")

    parser.add_argument('--arch', type=str, default="SE", help="Architecture")
    parser.add_argument('--base', type=str, default="resnet50", help="Base model")

    # Boolean flags
    parser.add_argument('--oversample', action='store_true', help="Disable oversampling")
    parser.add_argument('--transform', action='store_true', help="More Transformations")
    parser.add_argument('--freeze', action='store_true', help="Freeze True")
    parser.add_argument('--ignore', action='store_true', help="Disable symmetries (default: True, use --ignore to set False)")
    parser.add_argument('--save_augmented', action='store_true', help="Save augmented data")

    return parser.parse_args()