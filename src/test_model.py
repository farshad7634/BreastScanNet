import torch
from model import BreastCancerClassifier
from dataset import BreastCancerSonogramDataset
from torchvision import transforms
from torch.utils.data import DataLoader

# Step 1: Set up transform and dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

data_dir = r"E:\Projects\BreastScanNet\BreastScanNet VS code\data"
dataset = BreastCancerSonogramDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Step 2: Initialize the model
model = BreastCancerClassifier(num_classes=3)

# Step 3: Run a forward pass
model.eval()  # Set to eval mode
with torch.no_grad():
    imgs, labels = next(iter(dataloader))  # Get a batch
    outputs = model(imgs)  # Forward pass

    print("Input shape:", imgs.shape)        # e.g. [4, 1, 224, 224]
    print("Output shape:", outputs.shape)    # e.g. [4, 3]
    print("Model predictions (raw):", outputs)
