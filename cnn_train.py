import torch
import torch.optim as optim
from torch.utils.data import random_split
import cnn_dataset
from torchvision import transforms
import cnn_architecture
import torch.nn as nn 

# Assuming your images and labels are correctly sized
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts numpy array (HxWxC) in the range [0, 1] to a torch.FloatTensor of shape (CxHxW)
])

dataset = cnn_dataset.MultispectralDataset(directory=r"C:\Users\s.angerer\Privat\Studium\veg_classification\patches\bands", transform=transform)
dataloader = cnn_dataset.DataLoader(dataset, batch_size=32, shuffle=True)


# Set up device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Instantiate the model, send it to the device
num_channels = 12  # Assuming 12 spectral bands
num_classes = len(dataset.label_to_index)  # Number of classes
model = cnn_architecture.SimpleCNN(num_channels, num_classes).to(device)

# Split dataset into training and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = cnn_dataset.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = cnn_dataset.DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

    # Validation loop (after each epoch)
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Validation Accuracy: {100 * correct / total}%")

print('Finished Training')
