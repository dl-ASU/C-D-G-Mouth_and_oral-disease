from base_model import ImageEncoder, device, cuda
import torch
from torch import nn

class Classifier(nn.Module):
    def __init__(self, num_classes, num_sites, base=None, p=0.5):
        super(Classifier, self).__init__()
        self.base = base
        self.num_sites = num_sites
        self.num_classes = num_classes
        self.softmax = nn.Softmax(dim=1)

        if self.base == 'inception':
            self.fc1 = nn.Linear(1536, 1024)
        elif self.base == 'ViT':
            self.fc1 = nn.Linear(768, 1024)
        elif self.base == 'resnet50':
            self.fc1 = nn.Linear(2048, 1024)
        elif self.base == "google" or self.base == "effnet_b4":
            self.fc1 = nn.Linear(1792, 1024)
        else:
            self.fc1 = nn.Linear(512, 1024)

        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, num_classes * num_sites)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x, site):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.bn(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        # Reshape to (batch_size, num_sites, num_classes)
        x = x.view(-1, self.num_sites, self.num_classes)
        mask = torch.full_like(x, float('-inf'))
        for i in range(x.size(0)):
            mask[i, site[i]] = 0 
        x = x + mask

        # Reshape back to (batch_size, num_sites * num_classes)
        x = x.view(-1, self.num_sites * self.num_classes)

        return x

class Model(nn.Module):
    def __init__(self, num_classes, num_sites, base = "inception", id2label = None, label2id = None):
        super(Model, self).__init__()
        self.base = base
        self.image_encoder = ImageEncoder(base, id2label = id2label, label2id = label2id)
        self.classifier = Classifier(num_classes, num_sites, base = base)
        self.max = nn.MaxPool1d(196)

    def forward(self, img, site):
        img_encoded = self.image_encoder(img) # [:, 0, :] # take the fuken called CLS token bull fuken shit

        if self.base == "ViT":
            img_encoded = self.max(img_encoded.transpose(1, 2)).squeeze(2)
        elif self.base == "google":
            img_encoded = img_encoded["features"]

        output = self.classifier(img_encoded, site)
        return output