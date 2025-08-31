import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import os
import rasterio

# -------------------------
# 1. Load dataset from your actual files
# -------------------------
# Set data paths
base_dir = os.path.join("GLACIER HACKATHON", "train", "Train")
band_dirs = [
    os.path.join(base_dir, "Band1"),
    os.path.join(base_dir, "Band2"),
    os.path.join(base_dir, "Band3"),
    os.path.join(base_dir, "Band4"),
    os.path.join(base_dir, "Band5")
]
label_dir = os.path.join(base_dir, "label")

# Collect band and label files
band_files_list = []
for band_dir in band_dirs:
    files = [f for f in os.listdir(band_dir) if f.lower().endswith(('.tif', '.img'))]
    files.sort()
    band_files_list.append(files)

label_files = [f for f in os.listdir(label_dir) if f.lower().endswith(('.tif', '.img'))]
label_files.sort()

class GlacierDataset(Dataset):
    """Dataset to load glacier images and masks from disk"""
    def __init__(self, band_dirs, band_files_list, label_dir, label_files):
        self.band_dirs = band_dirs
        self.band_files_list = band_files_list
        self.label_dir = label_dir
        self.label_files = label_files

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, idx):
        # Load label mask
        label_file = self.label_files[idx]
        label_path = os.path.join(self.label_dir, label_file)
        with rasterio.open(label_path) as src:
            label_data = src.read(1)
        label_tensor = torch.tensor(label_data, dtype=torch.float32)

        # Load image bands for this label
        band_stack = []
        for band_idx, band_files in enumerate(self.band_files_list):
            image_file = band_files[idx]
            band_path = os.path.join(self.band_dirs[band_idx], image_file)
            with rasterio.open(band_path) as src:
                band_data = src.read(1)
            band_stack.append(band_data)
        image_tensor = torch.tensor(np.stack(band_stack, axis=0), dtype=torch.float32)

        return image_tensor, label_tensor

# Create dataset and dataloader
train_dataset = GlacierDataset(band_dirs, band_files_list, label_dir, label_files)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# -------------------------
# 2. Define UNet model
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=5, out_ch=1):
        super().__init__()
        self.dconv_down1 = DoubleConv(in_ch, 32)
        self.dconv_down2 = DoubleConv(32, 64)
        self.dconv_down3 = DoubleConv(64, 128)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.dconv_up2 = DoubleConv(128, 64)
        self.conv_last = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        x1 = self.dconv_down1(x)
        x2 = self.maxpool(x1)

        x3 = self.dconv_down2(x2)
        x4 = self.maxpool(x3)

        x5 = self.dconv_down3(x4)

        x = self.upsample(x5)
        x = torch.cat([x, x3], dim=1)
        x = self.dconv_up2(x)

        return self.conv_last(x)

# -------------------------
# 3. Setup device & model
# -------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = UNet(in_ch=5, out_ch=1).to(device)
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# -------------------------
# 4. Training loop
# -------------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

num_epochs = 10
best_loss, best_model = float('inf'), None

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, masks in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss /= len(train_dataloader.dataset)

    print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}")
    scheduler.step(train_loss)

    # save best model
    if train_loss < best_loss:
        best_loss, best_model = train_loss, model.state_dict().copy()
        print("✅ Best model updated")

# Save weights
torch.save(best_model, "best_glacier_unet.pth")