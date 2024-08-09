import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, feature_extractor, num_classes):
        super(Model, self).__init__()
        self.feature_extractor = feature_extractor

        # Define the FNN to classify the extracted features
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten the output from the feature extractor
            nn.Linear(1792, 1024),  # Adjust the input size accordingly
            nn.ReLU(),  # Add ReLU activation function
            nn.Linear(1024, num_classes)  # Adjust the input size accordingly
        )

        self.features = None
        # Freeze the feature extractor parameters
        self._freeze_feature_extractor()

    def _freeze_feature_extractor(self):
        # Set the feature extractor parameters' requires_grad attribute to False
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x, mask):
        # Extract featuraes using the provided feature extractor
        features = self.feature_extractor(x)['features']

        self.features = features

        # Classify the features using the FNN
        output = self.classifier(features)


        logits = torch.mul(output, mask)


        logits = logits.masked_fill(mask == 0, float('-inf'))




        return logits