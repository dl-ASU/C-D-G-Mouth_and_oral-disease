import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 33
num_epochs = 30
batch_size = 32
validation_fraction = 0.2



full_data_dir = '/home/mohammaddallash/Documents/GitHub/C-D-G-Mouth_and_oral-disease/Supervised_fashon/inception net/resized dataset'  


CATEGORIES = ["high", "low", "normal"]
PLOTS_SAVE_PATH = '../images/plots.png'
TSNE_PLOT_SAVE_PATH = '../images/tsne_plot.png'