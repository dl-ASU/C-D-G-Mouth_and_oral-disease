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

pre_trained = {'ser', 'google', 'inception', 'effnet_b4', 'resnet50', 'convnext'}

#                    [5, 11, 15, 45, 90, 100] % 
dic = {"inception": [692, 687, 675, 549, 248, 0],     # output: 1536                                                                55 m
       "ViT":       [200, 196, 180, 132, 68, 0],      # 180, 164, 148, 132, 116, 100, 84, 68, 52, 36, 20, 4 # output: 768           86 m
       "google":    [416, 413, 387, 284, 205, 0],     # 387, 284, 205, 127, 75, 23, 3  # output 1792                                    25 m
       "effnet_b4": [416, 414, 387, 283, 127, 0],     # 400, 387, 374,361, 348, 335, 322, 309, 296, 283,        205,        127,          75,        23,         3 # output 1792         17 m
       "resnet50":  [159, 150, 129, 72, 33, 0],       # 7: 150 141, 129,     6: 111, 102, 93, 84,     5: 63, 54, 45      4: 24, 15, 3    # output 2048                                   20 m
       "convnext":  [340, 309, 62, 31, 4, 0],         # # output 1024 90 m
       "custom":    [0, 0, 0, 0, 0, 0, 0]}            # output 1024 # 82 m

epochs_sch = {0:0, 9:1, 14:2, 19:3, 24:4, 29:5}
test_size = 0.15
val_size = 0.15

full_dataset = "/home/waleed/Documents/Medical/data_DRP/LatestDataset_processed_299"

full_train_data_path = "/home/waleed/Documents/Medical/data_DRP/SplittedDataset/train"
full_val_data_path = "/home/waleed/Documents/Medical/data_DRP/SplittedDataset/validation"
full_test_data_path = "/home/waleed/Documents/Medical/data_DRP/SplittedDataset/test"



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

    parser.add_argument('--full_train_data_path', type=str, default=full_train_data_path, help="Full train data path")
    parser.add_argument('--full_val_data_path', type=str, default=full_val_data_path, help="Full validation data path")
    parser.add_argument('--full_test_data_path', type=str, default=full_test_data_path, help="Full test data path")

    # Boolean flags
    parser.add_argument('--oversample', action='store_true', help="Disable oversampling")
    parser.add_argument('--transform', action='store_true', help="More Transformations")
    parser.add_argument('--freeze', action='store_true', help="Freeze True")
    parser.add_argument('--ignore', action='store_true', help="Disable symmetries (default: True, use --ignore to set False)")
    parser.add_argument('--save_augmented', action='store_true', help="Save augmented data")

    return parser.parse_args()