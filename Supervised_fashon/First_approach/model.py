import torch
from torch import nn
import timm

class ImageEncoder(nn.Module):
    def __init__(self, base = 'inception'):
        super(ImageEncoder, self).__init__()
        self.base = base
        if self.base == 'inception':
            self.inc_base = nn.Sequential()
            inceptionResnetV2 = timm.create_model('inception_resnet_v2', pretrained=True)
            for name, child in list(inceptionResnetV2.named_children())[:-1]:
                self.inc_base.add_module(name, child)
            # for name, child in list(model.named_children())[:-1]:
            #     self.le_base.add_module(name, child)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.fc1 = nn.Linear(256 * 25 * 25, 512)
            self.relu = nn.ReLU()

    def forward(self, x):
        if self.base == 'inception':
            return self.inc_base(x)
        else:
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.pool(self.relu(self.conv3(x)))
            x = x.view(x.size(0), -1)  # Flatten the tensor
            x = self.relu(self.fc1(x))
            return x

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
    def __init__(self, num_classes, base = 'inception'):
        super(Classifier, self).__init__()
        if base == 'inception':
            self.fc1 = nn.Linear(1536, 512)
        else:
            self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CombinedModel(nn.Module):
    def __init__(self, num_classes, num_sites, embedding_dim, base = "inception",):
        super(CombinedModel, self).__init__()
        self.image_encoder = ImageEncoder(base)
        self.site_encoder = SiteEncoder(num_sites, embedding_dim)
        self.classifier = Classifier(num_classes)

    def forward(self, img, site):
        img_encoded = self.image_encoder(img)
        site_encoded = self.site_encoder(site)
        combined = torch.cat((img_encoded, site_encoded), dim=1)
        output = self.classifier(combined)
        return output