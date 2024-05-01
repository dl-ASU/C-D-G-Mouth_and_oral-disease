import torch.nn as nn

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
        # self.le_base = nn.Sequential()
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
        inceptionResnetV2 = timm.create_model('inception_resnet_v2', pretrained=True)
        for name, child in list(inceptionResnetV2.named_children())[:-1]:
            self.inc_base.add_module(name, child)
        # for name, child in list(model.named_children())[:-1]:
        #     self.le_base.add_module(name, child)
        self.aux_layer = nn.Sequential(nn.Linear(1536, 7), nn.Softmax(dim = 1))

    def forward(self, img):
        out1 = self.inc_base(img)
        # out2 = self.le_base(img)
        # out = torch.cat((out1, out2), dim = 1)
        label = self.aux_layer(out1)
        return label
