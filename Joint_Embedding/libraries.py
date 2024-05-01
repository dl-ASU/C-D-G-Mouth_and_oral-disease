import os
import itertools
import zipfile
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, random_split

import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data.dataset import Subset
from torchvision.utils import make_grid, save_image
import timm

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
