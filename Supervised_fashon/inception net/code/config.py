import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 33
num_epochs = 30
batch_size = 32
validation_fraction = 0.2



full_data_dir = '/home/mohammaddallash/Documents/GitHub/C-D-G-Mouth_and_oral-disease/Supervised_fashon/inception net/resized dataset'  


CATEGORIES = ["high", "low", "normal"]
LOCATIONS = ['buccal mucosa left', 'buccal mucosa right', 'dorsum of tongue', 'floor of mouth', 'gingiva', 'lateral border of tongue left', 'lateral border of tongue right', 'lower labial mucosa', 'palate', 'upper labial mucosa', 'ventral of tongue']


PLOTS_SAVE_PATH = '../images/plots.png'
TSNE_PLOT_SAVE_PATH = '../images/tsne_plot.png'

MATRIX_SAVE_PATH = '../images/matrix_plot.png'