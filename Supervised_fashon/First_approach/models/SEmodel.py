from .base_model import get_image_encoder
from helpful.helpful import print_trainable_parameters
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
    def __init__(self, feature_vector, num_classes, p=0.5):
        super(Classifier, self).__init__()
        self.num_classes = num_classes

        self.fc1 = nn.Linear(feature_vector + 512, 1024)
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
    def __init__(self, num_classes, num_sites, embedding_dim=128, base="inception"):
        super(Model, self).__init__()
        self.base = base
        self.image_encoder = get_image_encoder(base)
        print("Image-Encoder: ")
        print_trainable_parameters(self.image_encoder)
        
        self.site_encoder = SiteEncoder(num_sites, embedding_dim)
        print("Site-Encoder: ")
        print_trainable_parameters(self.site_encoder)

        self.classifier = Classifier(self.image_encoder.feature_vector, num_classes)
        print("Classifier: ")
        print_trainable_parameters(self.classifier)

        # Only used for Vision Transformers to reduce the patch embedding output
        self.max = nn.MaxPool1d(196)

    def forward(self, img, site):

        # Pass image through the image encoder
        img_encoded = self.image_encoder(img)

        if self.base == "ViT":
            # Max pooling for ViT to combine patch embeddings
            img_encoded = self.max(img_encoded.transpose(1, 2)).squeeze(2)
        elif self.base == "google":
            img_encoded = img_encoded["features"]

        # Pass site information through the site encoder
        site_encoded = self.site_encoder(site)
        # Concatenate image and site embeddings
        combined = torch.cat((img_encoded, site_encoded), dim=1)

        # Pass the combined embeddings through the classifier
        output = self.classifier(combined)

        return output