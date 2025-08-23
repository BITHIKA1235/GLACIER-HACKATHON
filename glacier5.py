import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import rasterio

# Set data paths
base_path = os.path.join("GLACIER HACKATHON", "train", "Train")
band_paths = {
    'Band1': os.path.join(base_path, 'Band1'),
    'Band2': os.path.join(base_path, 'Band2'), 
    'Band3': os.path.join(base_path, 'Band3'),
    'Band4': os.path.join(base_path, 'Band4'),
    'Band5': os.path.join(base_path, 'Band5')
}
mask_path = os.path.join(base_path, 'label')

# ----------------------------
# Load image bands
# ----------------------------
def load_image_parts(scene_id, band_paths):
    parts = scene_id.split('_')
    scene_key = '_'.join(parts[-2:]).replace('.tif', '')

    band_files = {
        'Band1': f'B2_B2_masked_{scene_key}.tif',
        'Band2': f'B3_B3_masked_{scene_key}.tif',
        'Band3': f'B4_B4_masked_{scene_key}.tif',
        'Band4': f'B6_B6_masked_{scene_key}.tif',
        'Band5': f'B10_B10_masked_{scene_key}.tif'
    }

    bands = []
    for band_name, band_dir in band_paths.items():
        file_path = os.path.join(band_dir, band_files[band_name])

        if not os.path.exists(file_path):
            return None

        with rasterio.open(file_path) as src:
            bands.append(src.read(1))

    return np.stack(bands, axis=0)

# ----------------------------
# Load mask
# ----------------------------
def load_glacier_mask(scene_id, mask_dir):
    parts = scene_id.split('_')
    scene_key = '_'.join(parts[-2:]).replace('.tif', '')
    mask_file = f'Y_output_resized_{scene_key}.tif'
    mask_path = os.path.join(mask_dir, mask_file)

    if not os.path.exists(mask_path):
        return None

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        return (mask > 0).astype(np.uint8)

# ----------------------------
# Dataset class
# ----------------------------
class GlacierDataset(Dataset):
    def __init__(self, scene_ids, band_paths, mask_path):
        self.scene_ids = scene_ids
        self.band_paths = band_paths
        self.mask_path = mask_path

    def __len__(self):
        return len(self.scene_ids)

    def __getitem__(self, idx):
        scene_id = self.scene_ids[idx]
        image = load_image_parts(scene_id, self.band_paths)
        mask = load_glacier_mask(scene_id, self.mask_path)

        if image is None or mask is None:
            return torch.zeros((5, 256, 256)), torch.zeros((256, 256)).long()

        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).long()

        return image_tensor, mask_tensor

# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    scene_files = [f for f in os.listdir(band_paths['Band1']) if f.endswith('.tif')]

    # Keep only valid scenes
    valid_scene_ids = []
    for scene_id in scene_files:
        if load_image_parts(scene_id, band_paths) is not None and load_glacier_mask(scene_id, mask_path) is not None:
            valid_scene_ids.append(scene_id)

    dataset = GlacierDataset(valid_scene_ids, band_paths, mask_path)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Example sample
    if len(dataset) > 0:
        sample_img, sample_mask = dataset[0]
        print("Image shape:", sample_img.shape)  # (5, H, W)
        print("Mask shape:", sample_mask.shape)  # (H, W)
        print(f"Dataloader created with {len(dataloader)} batches")
    else:
        print("No valid data found.")
