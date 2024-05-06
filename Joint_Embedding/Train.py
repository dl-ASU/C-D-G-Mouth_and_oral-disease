import subprocess
import os
import shutil
import argparse
from libraries import *
from helpfulFunctions import *
from dataLoader import *
from sampling import *
from losses import *
from net_v0 import *

BASE_DIR = os.getcwd()

# Define command-line arguments
parser = argparse.ArgumentParser(description="Specify the training parameters.")
parser.add_argument("--path", type=str, default="/kaggle/input/oral-classification-v2/datasetV2", help="Patch size (default: /kaggle/input/oral-classification-v2/datasetV2)")
parser.add_argument("--patch_size", type=int, default=24, help="Patch size (default: 24)")
parser.add_argument("--epochs", type=int, default=15, help="Epochs (default: 15)")
parser.add_argument("--N", type=int, default=3, help="Negative samples prt instance per patch (default: 3)")
parser.add_argument("--loss", type=str , default = "constructive loss" , help = "type of the loss (default: constructive loss)")

args = parser.parse_args()

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.RandomAffine(degrees=0, translate=(0, 0.2), scale=(1.0, 1.2), shear=20),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

classToIdx = {"variation" : 0, "high_risk" : 1, "low_risk" : 2}

data_loader = returnDataLoader(args.path, classToIdx, trans, args.patch_size, True)

torch.cuda.empty_cache()
model = ModClassifier().cuda()
print_trainable_parameters(model)
model = nn.DataParallel(model)

# inistantiate your optimizer and the scheduler procedure
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [15, 35], gamma = 0.5)

# Define an empty lists to save the losses during the training process
train_loss_values = []
var_loss_values = []
cov_loss_values = []
con_loss_values = []


for epoch in range(args.epochs):

    # Training Mode
    model.train()
    train_loss = 0
    var_lossv = 0
    con_lossv = 0
    cov_lossv = 0


    for train_x, train_y in data_loader:
        try:
            pos, neg = sample_contrastive_pairs(train_x, train_y, args.N)
        except:
            print("Skipped")
            continue

        train_x, pos, neg, train_y = train_x.to(device), pos.to(device), neg.to(device), train_y.to(device)

        # Feedforward
        emb_actu = model(train_x)
        emb_pos = model(pos)
        emb_neg = model(neg.reshape(-1, 3, 299, 299)).reshape(-1, args.N, 2)
        _std_loss = std_loss(emb_actu, emb_pos)
        _cov_loss = cov_loss(emb_actu, emb_pos)
        if str.lower(args.loss ) == "constructive loss":
            loss_contra = contrastive_loss(emb_actu, emb_pos, emb_neg)
        elif str.lower(args.loss ) == "triplet loss":
            loss_contra =  triplet_loss(emb_actu,emb_pos,emb_neg)
        elif str.lower(args.loss) == "npair loss":
            loss_contra = npair_loss(emb_actu, emb_pos, emb_neg)
        elif str.lower(args.loss) == "online contrastive loss":
            loss_contra = online_contrastive_loss(emb_actu, emb_pos)
        

        loss = (loss_contra + 2*_std_loss + _cov_loss) / 4
        train_loss += loss.item()
        con_lossv += loss_contra.item()
        var_lossv += _std_loss.item()
        cov_lossv += _cov_loss.item()



        # At start of each Epoch
        optimizer.zero_grad()

        # Do the back probagation and update the parameters
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    con_lossv /= len(data_loader)
    cov_lossv /= len(data_loader)
    var_lossv /= len(data_loader)

    scheduler.step()

    train_loss_values.append(train_loss)
    var_loss_values.append(var_lossv)
    cov_loss_values.append(cov_lossv)
    con_loss_values.append(con_lossv)

    print(f"Epoch: {epoch + 1} | training_loss : {train_loss:.4f} | Sim_loss : {loss_contra:.4f} | Var_loss : {_std_loss:.4f} | Cov_loss : {_cov_loss:.4f}")

# Convert lists to NumPy arrays
train_loss_values = np.array(train_loss_values)
var_loss_values = np.array(var_loss_values)
cov_loss_values = np.array(cov_loss_values)
con_loss_values = np.array(con_loss_values)


# Create the new folder
directory = os.path.join(BASE_DIR, "results")
os.makedirs(directory, exist_ok=True)

# Save NumPy arrays with specified directory
np.save(os.path.join(directory, 'train_loss_values.npy'), train_loss_values)
np.save(os.path.join(directory, 'var_loss_values.npy'), var_loss_values)
np.save(os.path.join(directory, 'cov_loss_values.npy'), cov_loss_values)
np.save(os.path.join(directory, 'con_loss_values.npy'), con_loss_values)

# Specify the file name for saving the model
model_file = os.path.join(directory, 'model.pt')

# Save the model to the specified directory
torch.save(model.state_dict(), model_file)
