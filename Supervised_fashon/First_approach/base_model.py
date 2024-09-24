import torch
from torch import nn
import timm
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import ViTModel

cuda = True if torch.cuda.is_available() else False
device = 'cuda' if cuda else 'cpu'

class ImageEncoder(nn.Module):
    def __init__(self, base=None, id2label=None, label2id=None,freeze_base = False, dropout = 0.1):
        super(ImageEncoder, self).__init__()
        self.base = base
        self.relu = nn.ReLU()

        if self.base == 'inception':
            inception_resnet_v2 = timm.create_model('inception_resnet_v2', pretrained=True)
            self.inc_base = nn.Sequential(*list(inception_resnet_v2.children())[:-1])
            if freeze_base:
                self.freeze_base()
            self.set_dropout(self.inc_base, p=dropout)
            self.feature_vector = 1536

        elif self.base == "ViT":
            self.inc_base = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k',
                                                     id2label=id2label,
                                                     label2id=label2id)
            self.set_dropout(self.inc_base, p=0.25)

        elif self.base == "effnet_b4":
            model = timm.create_model('tf_efficientnet_b4', pretrained=True)
            self.inc_base = nn.Sequential(*list(model.children())[:-1])
            if freeze_base:
                self.freeze_base()
            self.set_dropout(self.inc_base, p=dropout)
            self.feature_vector = 1792
            
        elif self.base == "resnet50":
            resnet50 = timm.create_model('resnet50', pretrained=True)
            self.inc_base = nn.Sequential(*list(resnet50.children())[:-1])
            if freeze_base:
                self.freeze_base()
            self.set_dropout(self.inc_base, p=dropout)
            self.feature_vector = 2048

        elif self.base == "convnext":
            self.inc_base = timm.create_model('convnext_base', pretrained=True)
            # self.inc_base = nn.Sequential(*list(convnext_base.children())[:-1])
            if freeze_base:
                self.freeze_base()
            self.set_dropout(self.inc_base, p=dropout)
            self.feature_vector = 1000

        elif self.base == "google":
            model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)
            return_nodes = {'classifier.dropout': 'features'}
            self.inc_base = create_feature_extractor(model, return_nodes)

        elif self.base == "ser":
            model = timm.create_model('seresnextaa101d_32x8d.sw_in12k_ft_in1k_288', pretrained=True)
            self.inc_base = nn.Sequential(*list(model.children())[:-1])
            self.set_dropout(self.inc_base, p=0.25)

        elif self.base =='densenet':
            densenet = timm.create_model('densenet201', pretrained=True)
            self.inc_base = nn.Sequential(*list(densenet.children())[:-1])
            if freeze_base:
                self.freeze_base()
            self.set_dropout(self.inc_base, p=dropout)
            self.feature_vector = 1920
            
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.fc1 = nn.Linear(256 * 25 * 25, 512)
        # print(self.inc_base.default_cfg['input_size'])
        
    def set_dropout(self, model, p):
        """Recursively set dropout probability in a model."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = p

    def freeze_base(self):
        """Freeze the base model."""
        for param in self.inc_base.parameters():
            param.requires_grad = False            
    
    def forward(self, x):

        if self.base == 'ViT':
            x = self.inc_base(x)
            return x.last_hidden_state

        elif self.base in {'ser', 'google', 'inception', 'effnet_b4', 'resnet50', 'convnext'}:
            x = self.inc_base(x)
            return x

        else:
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.pool(self.relu(self.conv3(x)))
            x = x.view(x.size(0), -1)  # Flatten the tensor
            x = self.relu(self.fc1(x))
            return x
