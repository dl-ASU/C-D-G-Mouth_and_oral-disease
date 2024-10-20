from .base_model import get_image_encoder
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, num_classes, p=0.5, base="inception"):
        super(Model, self).__init__()
        self.base = base
        self.image_encoder = get_image_encoder(base)
        self.fc1 = nn.Linear(self.image_encoder.feature_vector, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=p)
        self.bn = nn.BatchNorm1d(128)

    def forward(self, img):

        # Pass image through the image encoder
        x = self.image_encoder(img)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.bn(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x