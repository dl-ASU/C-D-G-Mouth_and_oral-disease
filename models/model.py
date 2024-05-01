import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class ModifiedEfficientNet(nn.Module):
    def __init__(self, n_classes):
        super(ModifiedEfficientNet, self).__init__()
        
        # Define transposed convolutional layers for encoding
        self.trans_conv1 = nn.ConvTranspose2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)
    
        # Load pre-trained EfficientNet model
        self.base_model = EfficientNet.from_pretrained('efficientnet-b2')
        
        # Modify the fully connected layer
        num_ftrs = self.base_model._fc.in_features
        self.base_model._fc = nn.Linear(num_ftrs, n_classes)
        
    def forward(self, x):
        x = self.trans_conv1(x)  # Add unsqueeze to make the tensor 4D   
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        # Forward pass through EfficientNet base model
        x = self.base_model(x)
        return x
