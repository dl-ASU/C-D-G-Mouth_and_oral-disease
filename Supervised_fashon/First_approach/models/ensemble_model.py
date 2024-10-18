import torch.nn as nn
from .SEmodel import Model


class ModelEnsembleInd(nn.Module):
    def __init__(self, num_classes=3, num_sites=11, base = None, model_classes = ['GhostNet', 'MnasNet', 'ShuffleNetV2', 'squeezenet1_0', 'mobilenet_v2', 'resnet18', 'efficientnet_b0'], dropout=0.1):
        super(ModelEnsembleInd, self).__init__()
        self.models = nn.ModuleList([Model(num_classes, num_sites, 128, model_class) for model_class in model_classes])

    def forward(self, x):
        output = []
        for model in self.models:
            output.append(model(x))

        return output