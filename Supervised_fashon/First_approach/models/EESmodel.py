from base_model import ImageEncoder, device, cuda
import torch
from torch import nn

class EarlySiteEncoder(nn.Module):
    def __init__(self, num_sites, embedding_dim, reduced_channels=1):
        super(EarlySiteEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.reduced_channels = reduced_channels
        self.embedding = nn.Embedding(num_sites, embedding_dim)
        self.fc = nn.Linear(embedding_dim, self.reduced_channels * self.embedding_dim * self.embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        x = self.relu(self.fc(x))
        x = x.view(-1, self.reduced_channels, self.embedding_dim, self.embedding_dim)  # Reshape to (reduced_channels, 299, 299)
        return x


class Classifier(nn.Module):
    def __init__(self, num_classes, base=None, p=0.5):
        super(Classifier, self).__init__()
        self.base = base
        self.num_classes = num_classes
        self.softmax = nn.Softmax(dim=1)
        
        if self.base == 'inception':
            self.fc1 = nn.Linear(1536, 1024)
        elif self.base == 'ViT':
            self.fc1 = nn.Linear(768, 1024)
        elif self.base == "google":
            self.fc1 = nn.Linear(1792, 1024)
        else:
            self.fc1 = nn.Linear(512, 1024)

        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.bn(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class Model(nn.Module):
    def __init__(self, num_classes, num_sites, embedding_dim=128, base="inception", id2label=None, label2id=None):
        super(Model, self).__init__()
        self.base = base
        self.image_encoder = ImageEncoder(base, id2label=id2label, label2id=label2id)
        self.site_encoder = EarlySiteEncoder(num_sites, embedding_dim, reduced_channels=1)
        self.classifier = Classifier(num_classes, base=base)
        self.max = nn.MaxPool1d(196)

    def forward(self, img, site):
        site_encoded = self.site_encoder(site)
        
        # Concatenate along the channel dimension
        img = torch.cat([img, site_encoded], dim=1)

        img_encoded = self.image_encoder(img)

        if self.base == "ViT":
            img_encoded = self.max(img_encoded.transpose(1, 2)).squeeze(2)
        elif self.base == "google":
            img_encoded = img_encoded["features"]

        output = self.classifier(img_encoded)
        return output
