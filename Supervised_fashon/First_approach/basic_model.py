import torch
import torch.nn as nn
from base_model import ImageEncoder
import torch.nn.functional as F

class BasicModel(nn.Module):
    def __init__(self, num_classes,num_sites,base=None, freeze_base=False , dropout = 0.1):
        super().__init__()
        self.flag = True if base == 'ViT' else False
        self.base = ImageEncoder(base=base,freeze_base=freeze_base)
        
        feature_vector = self.base.feature_vector
        self.embed = nn.Embedding(num_sites,feature_vector)
        
        self.fc1 = nn.Linear(feature_vector,1024)
        self.ln1 = nn.LayerNorm(1024)
        self.do1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(1024,512)
        self.ln2 = nn.LayerNorm(512)
        self.do2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(512,256)
        self.ln3 = nn.LayerNorm(256)
        self.do3 = nn.Dropout(dropout)
        
        self.out = nn.Linear(256,num_classes)

    def forward(self, image , sites):
        
        image_features = self.base(image)
        image_features = image_features + self.embed(sites)
        if self.flag:
            image_features = image_features[:, 0, :]
        
        x = self.do1(F.relu(self.ln1(self.fc1(image_features)))) 
        x = self.do2(F.relu(self.ln2(self.fc2(x)))) 
        x = self.do3(F.relu(self.ln3(self.fc3(x)))) 
        
        return self.out(x)

