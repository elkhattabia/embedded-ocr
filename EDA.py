import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from collections import Counter
import torch.nn as nn
import torch.optim as optim
# Transformations (normalisation standard MNIST)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Train set
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Test set
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# DataLoaders
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

print("MNIST chargé avec PyTorch")

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),              
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)          # 
        )

    def forward(self, x):
        return self.model(x)

      
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(model, loader):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def test(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


for epoch in range(5):
    loss = train(model, train_loader)
    acc = test(model, test_loader)
    print(f"Epoch {epoch+1}: Loss={loss:.4f}, Test Acc={acc*100:.2f}%")


#%%

#exemples


fig,axes = plt.subplots(1,6,figsize=(12,2))

for i in range(6):
    img,label = train_dataset[i]
    axes[i].imshow(img.squeeze(),cmap="gray")
    axes[i].set_title(f"label : {label}")
    axes[i].axis("off")
    
plt.savefig("exemple d'images") 
plt.show()


# distribution des classes

labels =[label for _,label in train_dataset]
counter = Counter(labels)

plt.bar(counter.keys(),counter.values())
plt.title("Distribution des classes dans MNIST (train)")
plt.xlabel("Classe") 
plt.ylabel("Nombre d'images") 
plt.savefig("Distribution des classes dans MNIST (train)")
plt.show()


# Empiler toutes les images dans un seul tenseur
all_pixels = torch.stack([img for img, _ in train_dataset])  # shape: [60000, 1, 28, 28]

mean = all_pixels.mean().item()
std  = all_pixels.std().item()

print(f"Moyenne des pixels : {mean:.4f}")
print(f"Écart-type des pixels : {std:.4f}")







