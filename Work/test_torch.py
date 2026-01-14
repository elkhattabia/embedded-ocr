import torch
import torch.nn as nn
import time
import numpy as np
from torchvision.models import resnet18

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Create synthetic dataset
num_samples = 5000
input_channels = 3
input_size = 224
batch_size = 32

# Generate random data
X = torch.randn(num_samples, input_channels, input_size, input_size)
y = torch.randint(0, 1000, (num_samples,))

# Create DataLoader
dataset = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load model
model = resnet18(pretrained=False)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
print("\nStarting training...")
start_time = time.time()

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 20 == 19:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {running_loss/20:.4f}')
            running_loss = 0.0

elapsed = time.time() - start_time
print(f"\nTotal training time: {elapsed:.2f} seconds")
print(f"Average time per epoch: {elapsed/num_epochs:.2f} seconds")