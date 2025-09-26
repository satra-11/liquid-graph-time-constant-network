import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class MissingDataGenerator:
    """Generate various types of missing data patterns for time series images."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        
    def random_pixel_drop(self, x: torch.Tensor, p: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random pixel dropout."""
        mask = torch.rand_like(x) > p
        return x * mask, mask
        
    def block_occlusion(self, x: torch.Tensor, p: float, block_size: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """Block occlusion (Cutout-style)."""
        B, T, H, W = x.shape
        mask = torch.ones_like(x)
        
        n_blocks = int(p * H * W / (block_size * block_size))
        
        for b in range(B):
            for t in range(T):
                for _ in range(n_blocks):
                    y = self.rng.randint(0, H - block_size + 1)
                    x_pos = self.rng.randint(0, W - block_size + 1)
                    mask[b, t, y:y+block_size, x_pos:x_pos+block_size] = 0
                    
        return x * mask, mask
    
    def stripe_removal(self, x: torch.Tensor, p: float, axis: str = 'row') -> Tuple[torch.Tensor, torch.Tensor]:
        """Remove horizontal or vertical stripes."""
        B, T, H, W = x.shape
        mask = torch.ones_like(x)
        
        if axis == 'row':
            n_stripes = int(p * H)
            for b in range(B):
                for t in range(T):
                    rows = self.rng.choice(H, n_stripes, replace=False)
                    mask[b, t, rows, :] = 0
        else:  # column
            n_stripes = int(p * W)
            for b in range(B):
                for t in range(T):
                    cols = self.rng.choice(W, n_stripes, replace=False)
                    mask[b, t, :, cols] = 0
                    
        return x * mask, mask
    
    def channel_missing(self, x: torch.Tensor, p: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Drop entire channels (e.g., RGB -> RG)."""
        B, T, C, H, W = x.shape
        mask = torch.ones_like(x)
        
        n_channels_drop = int(p * C)
        if n_channels_drop > 0:
            for b in range(B):
                for t in range(T):
                    channels = self.rng.choice(C, n_channels_drop, replace=False)
                    mask[b, t, channels, :, :] = 0
                    
        return x * mask, mask
    
    def salt_pepper_noise(self, x: torch.Tensor, p: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Salt and pepper noise as missingness."""
        noise_mask = torch.rand_like(x) < p
        corrupted = x.clone()
        salt = torch.rand_like(x) > 0.5
        corrupted[noise_mask & salt] = 1.0
        corrupted[noise_mask & ~salt] = 0.0
        
        return corrupted, ~noise_mask


class ImageToSequence(nn.Module):
    """Convert images to patch sequences for time series processing."""
    
    def __init__(self, patch_size: int = 16, overlap: int = 0):
        super().__init__()
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert (B, T, C, H, W) to (B, T, N_patches, patch_features)."""
        B, T, C, H, W = x.shape
        
        # Extract patches using unfold
        patches = x.unfold(3, self.patch_size, self.stride).unfold(4, self.patch_size, self.stride)
        # patches: (B, T, C, H_patches, W_patches, patch_size, patch_size)
        
        H_patches, W_patches = patches.shape[3], patches.shape[4]
        N_patches = H_patches * W_patches
        
        # Reshape to sequence format
        patches = patches.contiguous().view(B, T, N_patches, C * self.patch_size * self.patch_size)
        
        return patches


class TimeSeriesImageDataset:
    """Dataset wrapper for time series images with missing data."""
    
    def __init__(self, images: torch.Tensor, missing_generator: MissingDataGenerator, 
                 patch_converter: ImageToSequence, missing_rates: list = [0.0, 0.1, 0.3, 0.5, 0.7]):
        self.images = images
        self.missing_gen = missing_generator
        self.patch_converter = patch_converter
        self.missing_rates = missing_rates
        
    def get_batch(self, missing_type: str = 'random', missing_rate: float = 0.3) -> dict:
        """Get a batch with specified missing pattern."""
        
        # Apply missing pattern
        if missing_type == 'random':
            corrupted, mask = self.missing_gen.random_pixel_drop(self.images, missing_rate)
        elif missing_type == 'block':
            corrupted, mask = self.missing_gen.block_occlusion(self.images, missing_rate)
        elif missing_type == 'stripe_row':
            corrupted, mask = self.missing_gen.stripe_removal(self.images, missing_rate, 'row')
        elif missing_type == 'stripe_col':
            corrupted, mask = self.missing_gen.stripe_removal(self.images, missing_rate, 'col')
        elif missing_type == 'channel':
            corrupted, mask = self.missing_gen.channel_missing(self.images, missing_rate)
        elif missing_type == 'noise':
            corrupted, mask = self.missing_gen.salt_pepper_noise(self.images, missing_rate)
        else:
            raise ValueError(f"Unknown missing type: {missing_type}")
            
        # Convert to patch sequences
        clean_patches = self.patch_converter(self.images)
        corrupted_patches = self.patch_converter(corrupted)
        mask_patches = self.patch_converter(mask.float())
        
        return {
            'clean': clean_patches,
            'corrupted': corrupted_patches,
            'mask': mask_patches,
            'missing_rate': missing_rate,
            'missing_type': missing_type
        }