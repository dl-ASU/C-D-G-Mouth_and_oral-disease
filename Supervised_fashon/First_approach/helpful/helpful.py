from collections import Counter
from torchvision.utils import  make_grid
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def setTrainable(model, n):
    counter = 0
    for param in model.parameters():
        counter += 1
        if(counter > n):
            param.requires_grad = True

def FreezeFirstN(model, n):
    counter = 0
    for param in model.parameters():
        counter += 1
        if(counter < n):
            param.requires_grad = False

def printParamSize(model):
    for param in model.parameters():
        print(param.size())

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

def show_tensor_images(image_tensor, num_images=16, size=(3, 256, 256)):
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=4)
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())

def label_site_error_analysis(y_true, y_pred, sites):
    error_pairs = [(true, pred, site) for true, pred, site in zip(y_true, y_pred, sites) if true != pred]
    error_analysis = Counter(error_pairs)
    return error_analysis

def label_site_all_analysis(y_true, y_pred, sites):
    error_pairs = [(true, pred, site) for true, pred, site in zip(y_true, y_pred, sites)]
    error_analysis = Counter(error_pairs)
    return error_analysis
