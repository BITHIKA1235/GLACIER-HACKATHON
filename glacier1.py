import os
import numpy as np
import rasterio
from matplotlib import pyplot as plt

# --- Step 1: Define dataset directories ---
base_dir = os.path.join("GLACIER HACKATHON", "train", "Train")

# Band directories (now including Band4 and Band5)
band_dirs = [
    os.path.join(base_dir, "Band1"),
    os.path.join(base_dir, "Band2"),
    os.path.join(base_dir, "Band3"),
    os.path.join(base_dir, "Band4"),
    os.path.join(base_dir, "Band5")
]

# Label directory
label_dir = os.path.join(base_dir, "Label")

# --- Step 2: Collect file lists ---
band_files_list = []
for band_dir in band_dirs:
    files = [f for f in os.listdir(band_dir) if f.lower().endswith(('.tif', '.img'))]
    files.sort()
    band_files_list.append(files)

label_files = [f for f in os.listdir(label_dir) if f.lower().endswith(('.tif', '.img'))]
label_files.sort()

print("📂 Dataset summary:")
for i, band_dir in enumerate(band_dirs, 1):
    print(f"Number of files in Band{i}: {len(band_files_list[i-1])}")
print(f"Number of Label files: {len(label_files)}")

# --- Step 3: Load images and labels ---
images = []
labels = []

for idx, label_file in enumerate(label_files):
    # Load label
    label_path = os.path.join(label_dir, label_file)
    with rasterio.open(label_path) as src:
        label_data = src.read(1)   # single-band label
    labels.append(label_data)

    # Load all bands for this sample
    band_stack = []
    for band_idx, band_files in enumerate(band_files_list):
        image_file = band_files[idx]
        band_path = os.path.join(band_dirs[band_idx], image_file)
        with rasterio.open(band_path) as src:
            band_data = src.read(1)
        band_stack.append(band_data)

    # Stack into (bands, H, W)
    multi_band_image = np.stack(band_stack, axis=0)
    images.append(multi_band_image)

# --- Step 4: Summary ---
print(f"\n✅ Loaded {len(images)} multi-band images (with 5 bands each) and {len(labels)} labels.")
if images and labels:
    print("First image shape (bands, H, W):", images[0].shape)
    print("First label shape (H, W):", labels[0].shape)

    # --- Step 5: Visualization ---
    plt.figure(figsize=(15, 4))
    for b in range(5):
        plt.subplot(1, 6, b+1)
        plt.imshow(images[0][b], cmap='gray')
        plt.title(f"Band {b+1}")
        plt.axis("off")

    plt.subplot(1, 6, 6)
    plt.imshow(labels[0], cmap='gray')
    plt.title("Label")
    plt.axis("off")
    plt.show()
