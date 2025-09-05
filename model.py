import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import os
import rasterio
from sklearn.metrics import matthews_corrcoef
import re
import random
import torch.nn.functional as F
import platform

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# -------------------------
# 1. Load dataset with proper validation split
# -------------------------
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

# -------------------------
# 2. Region-based split for validation
# -------------------------
def extract_region_id(filename):
    patterns = [
        r'(Region[A-Z]+)_glacier',
        r'([A-Za-z]+)_glacier',
        r'(region[A-Z]+)_',
        r'([A-Za-z0-9]+)_glacier'
    ]
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return match.group(1)
    return filename.split('_')[0]

region_files = {}
for label_file in label_files:
    region_id = extract_region_id(label_file)
    if region_id not in region_files:
        region_files[region_id] = []
    region_files[region_id].append(label_file)

regions = list(region_files.keys())
print(f"Available regions: {regions}")

if len(regions) < 2:
    random.shuffle(label_files)
    split_idx = int(0.8 * len(label_files))
    train_label_files = label_files[:split_idx]
    val_label_files = label_files[split_idx:]
    print("Warning: Only one region found, using random split")
else:
    val_region = regions[0]
    train_regions = regions[1:]
    train_label_files = []
    for region in train_regions:
        train_label_files.extend(region_files[region])
    val_label_files = region_files[val_region]

print(f"Training samples: {len(train_label_files)}")
print(f"Validation samples: {len(val_label_files)}")

# -------------------------
# 3. Data Augmentation Transformations
# -------------------------
class SatelliteTransform:
    def __init__(self, augment=False):
        self.augment = augment

    def __call__(self, image, mask):
        if self.augment:
            if random.random() > 0.5:
                image = torch.flip(image, [2])
                mask = torch.flip(mask, [1])
            if random.random() > 0.5:
                image = torch.flip(image, [1])
                mask = torch.flip(mask, [0])
            rot = random.choice([0, 1, 2, 3])
            image = torch.rot90(image, rot, [1, 2])
            mask = torch.rot90(mask, rot, [0, 1])
        return image, mask

# -------------------------
# 4. Dataset class with improved file matching
# -------------------------
def extract_common_id(filename):
    patterns = [
        r'(Region[A-Z]+_glacier\d+)',
        r'([A-Za-z]+_glacier\d+)',
        r'([A-Za-z0-9]+_glacier\d+)',
        r'(glacier\d+_[A-Za-z0-9]+)',
        r'([A-Za-z0-9]+_\d+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return match.group(1)
    filename = filename.replace('B2_', '').replace('B3_', '').replace('B4_', '').replace('B6_', '').replace('B10_', '')
    filename = filename.replace('Y_output_resized_', '').replace('mask_', '').replace('label_', '')
    return os.path.splitext(filename)[0]

class GlacierDataset(Dataset):
    def __init__(self, band_dirs, band_files_list, label_dir, label_files, transform=None):
        self.band_dirs = band_dirs
        self.band_files_list = band_files_list
        self.label_dir = label_dir
        self.label_files = label_files
        self.transform = transform
        self.file_mapping = self._create_file_mapping()

    def _create_file_mapping(self):
        mapping = {}
        for label_file in self.label_files:
            common_id = extract_common_id(label_file)
            band_files = []
            for band_files_in_dir in self.band_files_list:
                found_file = None
                for band_file in band_files_in_dir:
                    band_common_id = extract_common_id(band_file)
                    if band_common_id == common_id:
                        found_file = band_file
                        break
                if found_file is None:
                    label_base = os.path.splitext(label_file)[0]
                    for band_file in band_files_in_dir:
                        band_base = os.path.splitext(band_file)[0]
                        if label_base in band_base or band_base in label_base:
                            found_file = band_file
                            break
                band_files.append(found_file)
            mapping[label_file] = band_files
        return mapping

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, idx):
        label_file = self.label_files[idx]
        label_path = os.path.join(self.label_dir, label_file)
        with rasterio.open(label_path) as src:
            label_data = src.read(1)
        label_tensor = torch.tensor(label_data, dtype=torch.float32)
        band_files = self.file_mapping[label_file]
        band_stack = []
        for band_idx, band_file in enumerate(band_files):
            if band_file is None:
                band_data = np.zeros_like(label_data)
            else:
                band_path = os.path.join(self.band_dirs[band_idx], band_file)
                with rasterio.open(band_path) as src:
                    band_data = src.read(1)
            band_stack.append(band_data)
        image_tensor = torch.tensor(np.stack(band_stack, axis=0), dtype=torch.float32)
        if self.transform:
            image_tensor, label_tensor = self.transform(image_tensor, label_tensor)
        return image_tensor, label_tensor

train_transform = SatelliteTransform(augment=True)
val_transform = SatelliteTransform(augment=False)

# -------------------------
# 5. MCC Calculation Function
# -------------------------
def calculate_mcc(predictions, targets):
    pred_binary = (torch.sigmoid(predictions) > 0.5).float()
    targets = targets.unsqueeze(1) if targets.ndim == 3 else targets
    pred_flat = pred_binary.view(-1).cpu().numpy()
    target_flat = targets.view(-1).cpu().numpy()
    if np.all(pred_flat == pred_flat[0]) or np.all(target_flat == target_flat[0]):
        return 0.0
    return matthews_corrcoef(target_flat, pred_flat)

# -------------------------
# 6. Define UNet model
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=5, out_ch=1):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.final(d1)

# -------------------------
# 7. Training loop
# -------------------------
if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Pin memory flag
    pin_memory_flag = (device.type == "cuda")

    # Safer num_workers
    num_workers_flag = 0 if platform.system() == "Windows" else 4

    # Datasets
    train_dataset = GlacierDataset(band_dirs, band_files_list, label_dir, train_label_files, transform=train_transform)
    val_dataset = GlacierDataset(band_dirs, band_files_list, label_dir, val_label_files, transform=val_transform)

    # DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=num_workers_flag, pin_memory=pin_memory_flag)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=num_workers_flag, pin_memory=pin_memory_flag)

    # Model
    model = UNet(in_ch=5, out_ch=1).to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    num_epochs = 20
    best_mcc = -1.0
    best_model = None
    train_losses, val_losses, val_mccs = [], [], []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss, train_batches = 0, 0
        for images, masks in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            if masks.shape[2:] != outputs.shape[2:]:
                masks = F.interpolate(masks.float(), size=outputs.shape[2:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, masks.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
        train_loss /= train_batches
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss, val_batches = 0, 0
        val_mcc_scores = []
        with torch.no_grad():
            for images, masks in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Val"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)
                if masks.shape[2:] != outputs.shape[2:]:
                    masks = F.interpolate(masks.float(), size=outputs.shape[2:], mode='bilinear', align_corners=False)
                loss = criterion(outputs, masks.float())
                val_loss += loss.item()
                val_batches += 1
                mcc = calculate_mcc(outputs, masks)
                val_mcc_scores.append(mcc)
        val_loss /= val_batches
        avg_val_mcc = np.mean(val_mcc_scores)
        val_losses.append(val_loss)
        val_mccs.append(avg_val_mcc)

        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Val MCC {avg_val_mcc:.4f}, LR {optimizer.param_groups[0]['lr']:.2e}")

        # Save best model
        if avg_val_mcc > best_mcc:
            best_mcc = avg_val_mcc
            best_model = model.state_dict().copy()
            torch.save(best_model, "model.pth")
            print(f"✅ Best model updated with MCC: {best_mcc:.4f}")

        scheduler.step(avg_val_mcc)

        # Early stopping
        if epoch > 10 and avg_val_mcc < max(val_mccs[-5:]):
            print("Early stopping triggered")
            break

    # Final evaluation
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    final_mcc_scores = []
    with torch.no_grad():
        for images, masks in val_dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            if masks.shape[2:] != outputs.shape[2:]:
                masks = F.interpolate(masks.float(), size=outputs.shape[2:], mode='bilinear', align_corners=False)
            mcc = calculate_mcc(outputs, masks)
            final_mcc_scores.append(mcc)

    final_mcc = np.mean(final_mcc_scores)
    print(f"Final MCC on unseen region: {final_mcc:.4f}")
    print(f"Best model saved as 'model.pth' with MCC: {best_mcc:.4f}")
