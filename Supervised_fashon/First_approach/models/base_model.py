import torch
from torch import nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import ViTModel
import timm
from config import pre_trained

cuda = True if torch.cuda.is_available() else False
device = 'cuda' if cuda else 'cpu'

class ImageEncoderBase(nn.Module):
    """Base class for different image encoders."""
    def __init__(self, dropout=0.1):
        super(ImageEncoderBase, self).__init__()
        self.dropout = dropout

    def set_dropout(self, model, p):
        """Recursively set dropout probability in a model."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = p

    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented in subclasses.")

# --------------------------------------------------------------------------------------------------------------------------------------------------------

class InceptionEncoder(ImageEncoderBase):
    def __init__(self, dropout=0.1):
        super().__init__(dropout)
        inception_resnet_v2 = timm.create_model('inception_resnet_v2', pretrained=True)
        self.inc_base = nn.Sequential(*list(inception_resnet_v2.children())[:-1])
        self.feature_vector = 1536
        self.set_dropout(self.inc_base, p=dropout)

    def forward(self, x):
        return self.inc_base(x)

# --------------------------------------------------------------------------------------------------------------------------------------------------------

class ViTEncoder(ImageEncoderBase):
    def __init__(self, id2label=None, label2id=None, dropout=0.1):
        super().__init__(dropout)
        self.inc_base = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k',
                                                 id2label=id2label,
                                                 label2id=label2id)
        self.feature_vector = 768
        self.set_dropout(self.inc_base, p=0.25)

    def forward(self, x):
        return self.inc_base(x).last_hidden_state

# --------------------------------------------------------------------------------------------------------------------------------------------------------
class EffNetb0Encoder(ImageEncoderBase):
    def __init__(self, dropout=0.1):
        super().__init__(dropout)
        model = models.efficientnet_b0(pretrained=True)
        self.inc_base = nn.Sequential(*list(model.children())[:-1])
        self.feature_vector = 1280
        self.set_dropout(self.inc_base, p=dropout)

    def forward(self, x):
        return self.inc_base(x)

class EffNetB4Encoder(ImageEncoderBase):
    def __init__(self, dropout=0.1):
        super().__init__(dropout)
        model = timm.create_model('tf_efficientnet_b4', pretrained=True)
        self.inc_base = nn.Sequential(*list(model.children())[:-1])
        self.feature_vector = 1792
        self.set_dropout(self.inc_base, p=dropout)

    def forward(self, x):
        return self.inc_base(x)

# --------------------------------------------------------------------------------------------------------------------------------------------------------
class Resnet50(ImageEncoderBase):
    def __init__(self, dropout=0.1):
        super().__init__(dropout)
        resnet50 = timm.create_model('resnet50', pretrained=True)
        self.inc_base = nn.Sequential(*list(resnet50.children())[:-1])
        self.feature_vector = 2048
        self.set_dropout(self.inc_base, p=dropout)

    def forward(self, x):
        return self.inc_base(x)

class Resnet18(ImageEncoderBase):
    def __init__(self, dropout=0.1):
        super().__init__(dropout)
        model = models.resnet18(pretrained=True)
        self.inc_base = nn.Sequential(*list(model.children())[:-1])
        self.feature_vector = 512
        self.set_dropout(self.inc_base, p=dropout)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        feature = self.inc_base(x)
        if len(feature.shape) > 2:
            feature = self.pooling(feature)
            feature = feature.squeeze(3).squeeze(2)
        return feature
    
class Resnet34(ImageEncoderBase):
    def __init__(self, dropout=0.1):
        super().__init__(dropout)
        resnet34 = models.resnet34(pretrained=True)
        # Remove the final classification layer
        self.inc_base = nn.Sequential(*list(resnet34.children())[:-1])
        self.feature_vector = 512  # Feature vector size for ResNet-34
        self.set_dropout(self.inc_base, p=dropout)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        feature = self.inc_base(x)
        if len(feature.shape) > 2:
            feature = self.pooling(feature)
            feature = feature.squeeze(3).squeeze(2)
        return feature

# --------------------------------------------------------------------------------------------------------------------------------------------------------

class Google(ImageEncoderBase):
    def __init__(self, dropout=0.1):
        super().__init__(dropout)
        model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)
        return_nodes = {'classifier.dropout': 'features'}
        self.inc_base = create_feature_extractor(model, return_nodes)
        self.feature_vector = 1792
        self.set_dropout(self.inc_base, p=dropout)

    def forward(self, x):
        return self.inc_base(x)

# -----------------------------------------------------------------------------------------------------------------------------------------

class Seresnex(ImageEncoderBase):
    def __init__(self, dropout=0.1):
        super().__init__(dropout)
        model = timm.create_model('seresnextaa101d_32x8d.sw_in12k_ft_in1k_288', pretrained=True)
        self.inc_base = nn.Sequential(*list(model.children())[:-1])
        self.feature_vector = 2048
        self.set_dropout(self.inc_base, p=0.25)

    def forward(self, x):
        return self.inc_base(x)

# -----------------------------------------------------------------------------------------------------------------------------------------

class Densenet(ImageEncoderBase):
    def __init__(self, dropout=0.1):
        super().__init__(dropout)
        densenet = timm.create_model('densenet201', pretrained=True)
        self.inc_base = nn.Sequential(*list(densenet.children())[:-1])
        self.feature_vector = 1920
        self.set_dropout(self.inc_base, p=dropout)

    def forward(self, x):
        return self.inc_base(x)


# -----------------------------------------------------------------------------------------------------------------------------------------


class Squeezenet(ImageEncoderBase):
    def __init__(self, dropout=0.1):
        super().__init__(dropout)
        model = models.squeezenet1_0(pretrained=True)
        self.inc_base = nn.Sequential(*list(model.children())[:-1])
        self.feature_vector = 512
        self.set_dropout(self.inc_base, p=dropout)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        feature = self.inc_base(x)
        if len(feature.shape) > 2:
            feature = self.pooling(feature)
            feature = feature.squeeze(3).squeeze(2)
        return feature
    
# -----------------------------------------------------------------------------------------------------------------------------------------

class Mobilenet_v2(ImageEncoderBase):
    def __init__(self, dropout=0.1):
        super().__init__(dropout)
        model = models.mobilenet_v2(pretrained=True)
        self.inc_base = nn.Sequential(*list(model.children())[:-1])
        self.feature_vector = 1280
        self.set_dropout(self.inc_base, p=dropout)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        feature = self.inc_base(x)
        if len(feature.shape) > 2:
            feature = self.pooling(feature)
            feature = feature.squeeze(3).squeeze(2)
        return feature
    
class MobileNetV3(ImageEncoderBase):
    def __init__(self, dropout=0.1):
        super().__init__(dropout)
        mobilenet_v3 = models.mobilenet_v3_large(pretrained=True)
        # Remove the final classification layer
        self.inc_base = nn.Sequential(*list(mobilenet_v3.children())[:-1])
        self.feature_vector = 1280  # Feature vector size for MobileNetV3
        self.set_dropout(self.inc_base, p=dropout)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        feature = self.inc_base(x)
        if len(feature.shape) > 2:
            feature = self.pooling(feature)
            feature = feature.squeeze(3).squeeze(2)
        return feature
# -----------------------------------------------------------------------------------------------------------------------------------------

class ShuffleNetV2(ImageEncoderBase):
    def __init__(self, dropout=0.1):
        super().__init__(dropout)
        shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
        # Remove the final classification layer
        self.inc_base = nn.Sequential(*list(shufflenet.children())[:-1])
        self.feature_vector = 1024  # Feature vector size for ShuffleNetV2
        self.set_dropout(self.inc_base, p=dropout)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        feature = self.inc_base(x)
        if len(feature.shape) > 2:
            feature = self.pooling(feature)
            feature = feature.squeeze(3).squeeze(2)
        return feature
# -----------------------------------------------------------------------------------------------------------------------------------------

class MnasNet(ImageEncoderBase):
    def __init__(self, dropout=0.1):
        super().__init__(dropout)
        mnasnet = models.mnasnet1_0(pretrained=True)
        # Remove the final classification layer
        self.inc_base = nn.Sequential(*list(mnasnet.children())[:-1])
        self.feature_vector = 1280  # Feature vector size for MnasNet
        self.set_dropout(self.inc_base, p=dropout)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        feature = self.inc_base(x)
        if len(feature.shape) > 2:
            feature = self.pooling(feature)
            feature = feature.squeeze(3).squeeze(2)
        return feature
# -----------------------------------------------------------------------------------------------------------------------------------------

class GhostNet(ImageEncoderBase):
    def __init__(self, dropout=0.1):
        super().__init__(dropout)
        ghostnet = timm.create_model('ghostnet_100', pretrained=True)
        # Remove the final classification layer
        self.inc_base = nn.Sequential(*list(ghostnet.children())[:-1])
        self.feature_vector = 1280  # Feature vector size for GhostNet
        self.set_dropout(self.inc_base, p=dropout)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        feature = self.inc_base(x)
        if len(feature.shape) > 2:
            feature = self.pooling(feature)
            feature = feature.squeeze(3).squeeze(2)
        return feature

# -----------------------------------------------------------------------------------------------------------------------------------------


class ModelEnsemble(nn.Module):
    def __init__(self, model_classes = ['GhostNet', 'MnasNet', 'ShuffleNetV2', 'squeezenet1_0', 'mobilenet_v2', 'resnet18', 'efficientnet_b0'], dropout=0.1):
        super(ModelEnsemble, self).__init__()

        self.models = nn.ModuleList([get_image_encoder(model_class) for model_class in model_classes])

        # You can choose a method for combining feature vectors
        # Here we concatenate the feature vectors from each model
        self.feature_vector = sum([model.feature_vector for model in self.models])
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # Pass the input through each model and collect feature vectors
        features = []

        for model in self.models:
            feature = model(x)
            if len(feature.shape) > 2:
                feature = self.pooling(feature)  # Apply global pooling to get [B, C, 1, 1]
                feature = feature.squeeze(3).squeeze(2)  # Flatten to [B, C]
            features.append(feature)

        output = torch.cat(features, dim=1)
        return output

# -----------------------------------------------------------------------------------------------------------------------------------------

# Factory pattern for mapping base names to classes
ENCODER_FACTORY = {
    'inception': InceptionEncoder,

    'ViT': ViTEncoder,

    'effnet_b4': EffNetB4Encoder,
    'efficientnet_b0': EffNetb0Encoder,

    'resnet50': Resnet50,
    'resnet34': Resnet34,
    'resnet18': Resnet18,

    'mobilenet_v2': Mobilenet_v2,
    'mobilenet_v3': MobileNetV3,

    'google': Google,

    'ser': Seresnex,

    'densenet': Densenet,

    'squeezenet1_0': Squeezenet,

    'ShuffleNetV2': ShuffleNetV2,

    'MnasNet': MnasNet,

    'GhostNet': GhostNet,

    'ensemble': ModelEnsemble,
}

def get_image_encoder(base, *args, **kwargs):
    """Fetch the appropriate ImageEncoder class based on the base model."""
    if base not in ENCODER_FACTORY:
        raise ValueError(f"Base model {base} is not available.")
    return ENCODER_FACTORY[base](*args, **kwargs)