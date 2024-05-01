from data_loading_proprocessing import load_and_preprocess_data
from mode import ModifiedEfficientNet
from training import train_model

# Define your transformations
transform = transforms.Compose([
#     transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
])

# Load and preprocess the data
root_folder = '/kaggle/input/dental-data/dataset-20240310T151810Z-001/dataset'
train_dataset, val_dataset = load_and_preprocess_data(root_folder, transform)

# Create model
model = ModifiedEfficientNet(4)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Define DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

# Train the model
trained_model = train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=20)
