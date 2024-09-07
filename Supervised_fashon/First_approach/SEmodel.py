from base_model import ImageEncoder, device, cuda
import torch
from torch import nn

class SiteEncoder(nn.Module):
    def __init__(self, num_sites, embedding_dim):
        super(SiteEncoder, self).__init__()
        self.embedding = nn.Embedding(num_sites, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 512)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        x = self.relu(self.fc(x))
        return x


class Classifier(nn.Module):
    def __init__(self, num_classes, base=None, p=0.5):
        super(Classifier, self).__init__()
        self.base = base
        self.num_classes = num_classes
        self.softmax = nn.Softmax(dim=1)
        
        if self.base == 'inception':
            self.fc1 = nn.Linear(1536 + 512, 1024)
        elif self.base == 'ViT':
            self.fc1 = nn.Linear(512 + 768, 1024)
        elif self.base == "google":
            self.fc1 = nn.Linear(512 + 1792, 1024)
        else:
            self.fc1 = nn.Linear(1024, 1024)

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
    def __init__(self, num_classes, num_sites, embedding_dim = 128, base = "inception", id2label = None, label2id = None):
        super(Model, self).__init__()
        self.base = base
        self.image_encoder = ImageEncoder(base, id2label = id2label, label2id = label2id)
        self.site_encoder = SiteEncoder(num_sites, embedding_dim)
        self.classifier = Classifier(num_classes, base = base)
        self.max = nn.MaxPool1d(196)

    def forward(self, img, site):
        img_encoded = self.image_encoder(img) # [:, 0, :] # take the fuken called CLS token bull fuken shit
        if self.base == "ViT":
            img_encoded = self.max(img_encoded.transpose(1, 2)).squeeze(2)
        elif self.base == "google":
            img_encoded = img_encoded["features"]
        site_encoded = self.site_encoder(site)
        combined = torch.cat((img_encoded, site_encoded), dim=1)
        output = self.classifier(combined)
        return output