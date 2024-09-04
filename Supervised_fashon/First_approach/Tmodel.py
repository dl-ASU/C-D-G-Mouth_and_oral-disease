from base_model import ImageEncoder
from torch import nn


class Classifier(nn.Module):
    def __init__(self, num_classes, base = "inception"):
        super(Classifier, self).__init__()
        self.base = base
        self.image_encoder = ImageEncoder(base)
        self.fc1 = nn.Linear(1536, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.dropout = nn.Dropout(p = 0.25)
        self.relu = nn.ReLU()

    def forward(self, img):
        img_encoded = self.image_encoder(img)
        output = self.dropout(self.relu(self.fc1(img_encoded)))
        output = self.relu(self.fc2(output))
        return output