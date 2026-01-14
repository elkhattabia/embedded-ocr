import os
from PIL import Image
from torch.utils.data import Dataset

class CustomMNIST(Dataset):
    def __init__(self, root_dir,split, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.split_dir = os.path.join(self.root_dir, self.split)
        self.files = [
            f for f in os.listdir(split)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        filepath = os.path.join(self.split_dir, filename)

        # Label = first character of filename
        label = int(filename[0])

        # Load image
        img = Image.open(filepath).convert("L")   # grayscale

        if self.transform:
            img = self.transform(img)

        return img, label
