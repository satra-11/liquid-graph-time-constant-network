#!/usr/bin/env python
"""
Extract CNN features from HDD dataset images and save them as .npy files.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm


# Define the CNN architecture (same as in controllers)
import torchvision.models as models


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.compress = nn.Conv2d(512, 128, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))

        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x: (Batch, 3, 224, 224)
        x = self.backbone(x)  # -> (Batch, 512, 7, 7) ※入力サイズにより可変
        x = self.compress(x)  # -> (Batch, 128, 7, 7)
        x = self.avgpool(x)  # -> (Batch, 128, 8, 8) 強制的に8x8にする
        return x


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)
        return img, str(path)


def extract_features(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup model
    model = FeatureExtractor().to(device)
    model.eval()

    # Setup transform
    transform = T.Compose(
        [
            T.Resize(
                (64, 64)
            ),  # Resize to match model input expectation if needed, or keep original if model handles it.
            # The original code used (224, 224) in HDDLoader but the model parameters say frame_height=64.
            # Wait, HDDLoader defaults to (224, 224).
            # But the controllers take frame_height=64.
            # Let's check HDDLoader again. It resizes to frame_size.
            # And run.py passes frame_height=64 to controllers.
            # But run.py does NOT pass frame_size to setup_dataloaders, so it uses default (224, 224).
            # This seems like a discrepancy in the original code?
            # The CNN has AdaptiveAvgPool2d((8, 8)), so input size doesn't strictly matter for output spatial dim,
            # but it matters for the receptive field and intermediate sizes.
            # If the original code trained with 224x224, I should use 224x224.
            # Let's check run.py again.
            # run.py calls setup_dataloaders(..., sequence_length=..., batch_size=...).
            # It does NOT pass frame_size. So HDDLoader uses (224, 224).
            # The controllers are initialized with frame_height=64, frame_width=64.
            # But these arguments are stored but NOT used in the CNN definition (it's hardcoded).
            # So the model receives 224x224 images.
            # I should use (224, 224) here to match.
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_dir = data_dir / "camera"
    sequences = [d.name for d in camera_dir.iterdir() if d.is_dir()]
    sequences.sort()

    print(f"Found {len(sequences)} sequences.")

    for seq in tqdm(sequences):
        seq_dir = camera_dir / seq
        image_paths = sorted(list(seq_dir.glob("*.jpg")))

        if not image_paths:
            continue

        dataset = ImageDataset(image_paths, transform=transform)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True
        )

        features_list = []

        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(device)
                feats = model(imgs)  # (B, 128, 8, 8)
                # We want to save as (B, 128, 8, 8) or flattened?
                # The original code expects (B, T, C, H, W) -> CNN -> (B*T, 128, 8, 8)
                # So we should save (N, 128, 8, 8) where N is number of frames.
                features_list.append(feats.cpu().numpy())

        if features_list:
            all_features = np.concatenate(features_list, axis=0)  # (N, 128, 8, 8)

            # Save to output_dir
            # Structure: output_dir / seq.npy
            save_path = output_dir / f"{seq}.npy"
            np.save(save_path, all_features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data/raw")
    parser.add_argument("--output-dir", type=str, default="./data/processed")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    extract_features(args)
