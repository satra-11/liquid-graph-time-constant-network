import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
import os
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from src.types import CorruptionConfig

    
class VideoProcessor:
    """映像データ処理と欠損生成クラス"""
    
    def __init__(self, config: CorruptionConfig):
        self.config = config
        
    def add_missing_data(self, frame: torch.Tensor) -> torch.Tensor:
        """フレームにデータ欠損を追加"""
        if self.config.missing_rate <= 0:
            return frame
            
        mask = torch.rand_like(frame) > self.config.missing_rate
        return frame * mask
    
    def add_whiteout(self, frame: torch.Tensor) -> torch.Tensor:
        """フレームに白飛びを追加"""
        if self.config.whiteout_rate <= 0:
            return frame
            
        whiteout_mask = torch.rand(frame.shape[:-1]) < self.config.whiteout_rate
        whiteout_mask = whiteout_mask.unsqueeze(-1).expand_as(frame)
        
        corrupted_frame = frame.clone()
        corrupted_frame[whiteout_mask] = 1.0  # 白飛び
        return corrupted_frame
    
    def add_noise(self, frame: torch.Tensor) -> torch.Tensor:
        """フレームにノイズを追加"""
        if self.config.noise_level <= 0:
            return frame
            
        noise = torch.randn_like(frame) * self.config.noise_level
        return torch.clamp(frame + noise, 0.0, 1.0)
    
    def add_blur(self, frame: torch.Tensor) -> torch.Tensor:
        """フレームにブラーを追加"""
        if self.config.blur_kernel <= 1:
            return frame
            
        # 簡単なガウシアンブラー
        kernel_size = self.config.blur_kernel
        sigma = kernel_size / 6.0
        
        # PyTorchのconv2dを使用したブラー実装
        channels = frame.shape[-1]
        kernel = self._gaussian_kernel(kernel_size, sigma, channels)
        
        # フレームを(B, C, H, W)形式に変換
        if frame.dim() == 3:
            frame_conv = frame.permute(2, 0, 1).unsqueeze(0)
        else:
            frame_conv = frame.permute(0, 3, 1, 2)
            
        blurred = F.conv2d(frame_conv, kernel, padding=kernel_size//2, groups=channels)
        
        # 元の形式に戻す
        if frame.dim() == 3:
            return blurred.squeeze(0).permute(1, 2, 0)
        else:
            return blurred.permute(0, 2, 3, 1)
    
    def _gaussian_kernel(self, kernel_size: int, sigma: float, channels: int) -> torch.Tensor:
        """ガウシアンカーネルを生成"""
        x = torch.arange(kernel_size, dtype=torch.float32)
        x = x - kernel_size // 2
        gauss = torch.exp(-(x**2) / (2 * sigma**2))
        kernel = gauss[:, None] * gauss[None, :]
        kernel = kernel / kernel.sum()
        
        # チャンネル数に合わせて拡張
        kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
        return kernel
    
    def corrupt_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """フレームに全ての汚損を適用"""
        corrupted = frame
        corrupted = self.add_missing_data(corrupted)
        corrupted = self.add_whiteout(corrupted)
        corrupted = self.add_noise(corrupted)
        corrupted = self.add_blur(corrupted)
        return corrupted

class DrivingDataset(Dataset):
    """HDDから自律走行データを読み込むためのデータセット"""

    def __init__(
        self,
        camera_dir: str,
        target_dir: str,
        sequence_length: int,
        frame_height: int = 64,
        frame_width: int = 64,
        corruption_config: Optional[CorruptionConfig] = None
    ):
        self.sequence_length = sequence_length
        self.video_processor = VideoProcessor(corruption_config or CorruptionConfig())

        self.transform = transforms.Compose([
            transforms.Resize((frame_height, frame_width)),
            transforms.ToTensor(),
        ])

        self.sequences = []
        target_files = sorted(glob(os.path.join(target_dir, '*.npy')))

        for target_file in target_files:
            seq_name = os.path.basename(target_file).replace('.npy', '')
            camera_seq_dir = os.path.join(camera_dir, seq_name)

            if os.path.isdir(camera_seq_dir):
                image_files = sorted(glob(os.path.join(camera_seq_dir, '*.jpg'))) # Assuming .jpg, adjust if needed
                if len(image_files) >= self.sequence_length:
                    self.sequences.append({
                        'name': seq_name,
                        'images': image_files,
                        'target': target_file
                    })

    def __len__(self):
        # 各シーケンスから取り出せるサブシーケンスの総数を返す
        total = 0
        for seq in self.sequences:
            total += len(seq['images']) - self.sequence_length + 1
        return total

    def __getitem__(self, idx):
        # idxから、どのシーケンスのどの開始フレームかを特定する
        seq_idx = 0
        frame_offset = idx
        for i, seq in enumerate(self.sequences):
            num_sub_sequences = len(seq['images']) - self.sequence_length + 1
            if frame_offset < num_sub_sequences:
                seq_idx = i
                break
            frame_offset -= num_sub_sequences

        selected_sequence = self.sequences[seq_idx]
        start_frame = frame_offset

        # 画像シーケンスを読み込む
        image_paths = selected_sequence['images'][start_frame : start_frame + self.sequence_length]

        clean_frames = []
        for img_path in image_paths:
            img = Image.open(img_path).convert('RGB')
            clean_frames.append(self.transform(img))

        clean_frames_tensor = torch.stack(clean_frames) # (T, C, H, W)
        # (T, H, W, C) に変換
        clean_frames_tensor = clean_frames_tensor.permute(0, 2, 3, 1)

        # 汚損フレームを生成
        corrupted_frames_tensor = self.video_processor.corrupt_frame(clean_frames_tensor.clone())

        # ターゲットを読み込む
        targets_full = np.load(selected_sequence['target'])
        targets = torch.from_numpy(targets_full[start_frame : start_frame + self.sequence_length]).float()

        return clean_frames_tensor, corrupted_frames_tensor, targets
