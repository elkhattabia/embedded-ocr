import os
import shutil
import random

def split_dataset(root, train_ratio=0.9):
    # List all files
    files = [f for f in os.listdir(root)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    random.shuffle(files)

    # Compute split index
    split_idx = int(len(files) * train_ratio)

    train_files = files[:split_idx]
    test_files = files[split_idx:]

    # Create subfolders
    train_dir = os.path.join(root, "train")
    test_dir = os.path.join(root, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Move files
    for f in train_files:
        shutil.move(os.path.join(root, f), os.path.join(train_dir, f))

    for f in test_files:
        shutil.move(os.path.join(root, f), os.path.join(test_dir, f))

    print(f"Done. {len(train_files)} train files, {len(test_files)} test files.")

# Usage
split_dataset("./data/custom_dataset", train_ratio=0.9)
