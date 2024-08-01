import torch
import torch.nn as nn
import timm

class ModClassifier(nn.Module):
    def __init__(self):
        super(ModClassifier, self).__init__()
        self.base = nn.Sequential()
        model = torch.load("/kaggle/input/model_v1.pth/pytorch/model_v1/1/model_v1.pth")
        for name0, child0 in list(model.named_children()):
            for name, child in list(child0.named_children())[:-1]:
                self.base.add_module(name, child)

        self.emb = nn.Linear(1536, 2)

    def forward(self, img):
        img = self.base(img)
        embed = self.emb(img)
        return embed

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.inc_base = nn.Sequential()
        inceptionResnetV2 = timm.create_model('inception_resnet_v2', pretrained=True)
        for name, child in list(inceptionResnetV2.named_children())[:-1]:
            self.inc_base.add_module(name, child)
        # for name, child in list(model.named_children())[:-1]:
        #     self.le_base.add_module(name, child)
        self.aux_layer = nn.Sequential(nn.Linear(1536, 7), nn.Softmax(dim = 1))

    def forward(self, img):
        out1 = self.inc_base(img)
        label = self.aux_layer(out1)
        return label

class OnlyEncoder(nn.Module):
  def __init__(self):
    super().__init__()

    # Build and define the Encoder
    self.encoder = nn.Sequential(

        nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = (3, 3), padding = "same"),
        nn.ReLU(),
        nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3), padding = "same"),
        nn.ReLU(),
        nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3, 3), padding = "same"),
        nn.ReLU()
        )
    self.InLatentSpace = nn.Sequential(
        nn.Linear(in_features = 28 * 28 * 128, out_features = 512)
    )

  def forward(self, x):
    x = self.encoder(x)
    x = x.reshape((-1, 128 * 28 * 28))
    x = self.InLatentSpace(x)
    return(x)

class AutoEncoder(nn.Module):
  def __init__(self):
    super().__init__()

    # Build and define your encoder
    self.encoder = nn.Sequential(

        nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = (3, 3), padding = "same"),
        nn.ReLU(),
        nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3), padding = "same"),
        nn.ReLU(),
        nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3, 3), padding = "same"),
        nn.ReLU()
        )

    self.InLatentSpace = nn.Sequential(
        nn.Linear(in_features = 28 * 28 * 128, out_features = 512),
        nn.ReLU(),
        nn.Linear(in_features = 512, out_features = 2)
    )

    self.OutLatentSpace = nn.Sequential(
        nn.Linear(in_features = 2, out_features = 512),
        nn.ReLU(),
        nn.Linear(in_features = 512, out_features = 28 * 28 * 128),
        nn.ReLU()
        )

    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding = 1),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding = 1),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(3, 3), padding = 1),
        nn.Tanh()
        )

  def forward(self, x):
    x = self.encoder(x)
    x = x.reshape((-1, 128 * 28 * 28))
    x = self.InLatentSpace(x)
    x = self.OutLatentSpace(x)
    x = x.reshape((-1, 128, 28, 28))
    x = self.decoder(x)
    return(x)

  def encode(self, x):
    x = self.encoder(x)
    x = x.reshape((-1, 128 * 28 * 28))
    x = self.InLatentSpace(x)
    return x

  def decode(self, x):
    x = self.OutLatentSpace(x)
    x = x.reshape((-1, 128, 28, 28))
    x = self.decoder(x)
    return(x)