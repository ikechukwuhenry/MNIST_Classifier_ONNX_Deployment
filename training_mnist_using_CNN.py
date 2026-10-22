import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import os


# import Model
from simple_cnn_model import SimpleMNIST_CNN


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.2])        
])


# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root = './data', train = True, download = True, transform = transform)

test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# GPU support
device = (torch.device("mps") if torch.backends.mps.is_available() 
          else torch.device("cuda") if torch.cuda.is_available else torch.device("cpu"))

# instantiate the model
model = SimpleMNIST_CNN().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
saved_model_path = os.path.join(BASE_DIR, 'deployment/mnist_cnn_model.pt')

def train_epoch(model, train_loader, loss_function, optimizer, device, saved_model_path):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()

        # Track progress
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Print every 100 batches
        if batch_idx % 100 == 0:
            avg_loss = running_loss / 100
            accuracy = 100.0 * correct / total
            print(f'   [{batch_idx * 64}/{60,000}]'
                  f'Loss: {avg_loss:.3f} | Accuracy: {accuracy:.1f}%')
            running_loss = 0.0

    torch.save(model.state_dict(), saved_model_path)
 
train_epoch(model, train_loader, loss_function, optimizer, device, saved_model_path=saved_model_path)

saved_model = SimpleMNIST_CNN()
saved_model.load_state_dict(torch.load(saved_model_path))
saved_model.eval()

dummy_input = torch.randn(1, 1, 28, 28)

onnx_file_path = os.path.join(BASE_DIR, 'deployment/mnist_cnn_model.onnx')

# Export os ONNX format
torch.onnx.export(
        saved_model,                    # Pytorch model
        dummy_input,                    # Input tensor
        onnx_file_path,         # Output file name
        export_params=True,             # Store the trained weights in the model file
        opset_version=11,               # ONNX version
        do_constant_folding=True,       # Optimize constant expressions
        input_names=['images'],         # Input name
        output_names=['predictions'],   # Output name
        dynamic_axes={'images': {0: 'batch_size'}, 'predictions': {0: 'batch_size'}}  # Variable batch size
    )