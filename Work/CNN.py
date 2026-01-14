import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import os

# ---------------------------------------------------------
# 1. Chargement MNIST
# ---------------------------------------------------------
def load_mnist(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convertit en [0,1]
    ])

    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ---------------------------------------------------------
# 2. Chargement BDD personnalisée (optionnel)
# ---------------------------------------------------------
class CustomDigits(Dataset):
    def __init__(self, path="data/custom_digits/"):
        self.images = []
        self.labels = []

        for digit in range(10):
            digit_path = os.path.join(path, str(digit))
            if not os.path.isdir(digit_path):
                continue

            for fname in os.listdir(digit_path):
                if fname.endswith(".bmp"):
                    img = Image.open(os.path.join(digit_path, fname)).convert("L")
                    img = img.resize((28, 28))
                    img = np.array(img, dtype=np.float32) / 255.0
                    img = torch.tensor(img).unsqueeze(0)  # (1, 28, 28)

                    self.images.append(img)
                    self.labels.append(digit)

        self.images = torch.stack(self.images)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# ---------------------------------------------------------
# 3. Architecture CNN légère (équivalent TF)
# ---------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# ---------------------------------------------------------
# 4. Entraînement
# ---------------------------------------------------------
def train_model(model, train_loader, test_loader, epochs=5, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

        evaluate(model, test_loader)

    return model


# ---------------------------------------------------------
# 5. Évaluation
# ---------------------------------------------------------
def evaluate(model, test_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    print(f"Accuracy: {acc:.4f}")
    return acc


# ---------------------------------------------------------
# 6. Sauvegarde des poids (pour export C)
# ---------------------------------------------------------
def save_weights(model, path="weights_cnn"):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "cnn_weights.pth"))
    print("Poids sauvegardés dans", path)


# ---------------------------------------------------------
# 7. Script principal
# ---------------------------------------------------------
if __name__ == "__main__":
    train_loader, test_loader = load_mnist()

    model = CNN()
    print(model)

    train_model(model, train_loader, test_loader, epochs=5)

    save_weights(model)
