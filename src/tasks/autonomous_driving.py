import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
import os
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset


@dataclass
class CorruptionConfig:
    """映像データの欠損・白飛び設定"""
    missing_rate: float = 0.1  # 欠損率
    whiteout_rate: float = 0.05  # 白飛び率
    noise_level: float = 0.02  # ノイズレベル
    blur_kernel: int = 3  # ブラーカーネルサイズ
    
    
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


class AutonomousDrivingTask:
    """自律走行タスクのシミュレーション"""
    
    def __init__(
        self,
        frame_height: int = 64,
        frame_width: int = 64,
        sequence_length: int = 10,
        speed_range: Tuple[float, float] = (0.6, 1.0),
        corruption_config: Optional[CorruptionConfig] = None
    ):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.sequence_length = sequence_length
        self.speed_range = speed_range
        self.corruption_config = corruption_config or CorruptionConfig()
        self.video_processor = VideoProcessor(self.corruption_config)
        
        # シンプルな道路環境のパラメータ
        self.road_width = 0.3  # 道路幅（画像幅に対する比率）
        self.lane_width = 0.05  # 車線幅
        
    def generate_sequence(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """映像シーケンスとターゲット制御信号を生成"""
        clean_frames = []
        corrupted_frames = []
        control_targets = []
        
        for b in range(batch_size):
            batch_clean = []
            batch_corrupted = []
            batch_controls = []
            
            # 初期車両位置
            vehicle_pos = 0.5 + 0.1 * (torch.rand(1).item() - 0.5)
            
            for t in range(self.sequence_length):
                # フレーム生成
                frame = self.generate_synthetic_road_frame(t, vehicle_pos) # TODO: Implement this method
                
                # 制御ターゲット（車線中央維持）
                target_steering = -(vehicle_pos - 0.5) * 2.0  # 中央復帰制御
                target_speed = 0.8  # 一定速度
                
                # 車両位置更新（簡単な動力学）
                vehicle_pos += target_steering * 0.01 + 0.001 * (torch.rand(1).item() - 0.5)
                vehicle_pos = np.clip(vehicle_pos, 0.2, 0.8)
                
                # 汚損フレーム生成
                corrupted_frame = self.video_processor.corrupt_frame(frame)
                
                batch_clean.append(frame)
                batch_corrupted.append(corrupted_frame)
                batch_controls.append(torch.tensor([target_steering, target_speed], dtype=torch.float32))
            
            clean_frames.append(torch.stack(batch_clean))
            corrupted_frames.append(torch.stack(batch_corrupted))
            control_targets.append(torch.stack(batch_controls))
        
        return (
            torch.stack(clean_frames),      # (B, T, H, W, C)
            torch.stack(corrupted_frames),  # (B, T, H, W, C)
            torch.stack(control_targets)    # (B, T, 2)
        )
    
    def create_graph_adjacency(self, frames: torch.Tensor) -> torch.Tensor:
        """フレーム特徴からグラフ隣接行列を作成"""
        B, T, H, W, C = frames.shape
        
        # フレームをパッチに分割してノードとする
        patch_size = 8
        patches_h = H // patch_size
        patches_w = W // patch_size
        num_nodes = patches_h * patches_w
        
        # パッチ特徴抽出
        patches = frames.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(B, T, patches_h, patches_w, patch_size * patch_size * C)
        
        # 空間的隣接性に基づく隣接行列
        adjacency = torch.zeros(B, T, num_nodes, num_nodes)
        
        for i in range(patches_h):
            for j in range(patches_w):
                node_idx = i * patches_w + j
                
                # 4近傍接続
                neighbors = [
                    (i-1, j), (i+1, j), (i, j-1), (i, j+1)
                ]
                
                for ni, nj in neighbors:
                    if 0 <= ni < patches_h and 0 <= nj < patches_w:
                        neighbor_idx = ni * patches_w + nj
                        adjacency[:, :, node_idx, neighbor_idx] = 1.0
        
        return adjacency

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
