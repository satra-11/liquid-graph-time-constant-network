import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


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
        corruption_config: Optional[CorruptionConfig] = None
    ):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.sequence_length = sequence_length
        self.corruption_config = corruption_config or CorruptionConfig()
        self.video_processor = VideoProcessor(self.corruption_config)
        
        # シンプルな道路環境のパラメータ
        self.road_width = 0.3  # 道路幅（画像幅に対する比率）
        self.lane_width = 0.05  # 車線幅
        
    def generate_synthetic_road_frame(self, t: int, vehicle_pos: float = 0.5) -> torch.Tensor:
        """合成道路フレームを生成"""
        frame = torch.zeros(self.frame_height, self.frame_width, 3)
        
        # 道路の描画
        road_center = int(self.frame_width * 0.5)
        road_half_width = int(self.frame_width * self.road_width / 2)
        
        # 道路面（グレー）
        frame[:, road_center-road_half_width:road_center+road_half_width, :] = 0.5
        
        # 車線（白線）
        lane_pos1 = road_center - int(self.frame_width * self.lane_width / 2)
        lane_pos2 = road_center + int(self.frame_width * self.lane_width / 2)
        
        # 破線の車線を描画
        for y in range(0, self.frame_height, 20):
            if (y + t) % 40 < 20:  # 破線効果
                frame[y:min(y+10, self.frame_height), lane_pos1:lane_pos1+2, :] = 1.0
                frame[y:min(y+10, self.frame_height), lane_pos2:lane_pos2+2, :] = 1.0
        
        # 車両の描画（簡単な矩形）
        vehicle_x = int(self.frame_width * vehicle_pos)
        vehicle_y = int(self.frame_height * 0.8)
        vehicle_size = 8
        
        frame[vehicle_y-vehicle_size:vehicle_y+vehicle_size, 
              vehicle_x-vehicle_size:vehicle_x+vehicle_size, 0] = 1.0  # 赤い車両
        
        return frame
    
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
                frame = self.generate_synthetic_road_frame(t, vehicle_pos)
                
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
                batch_controls.append(torch.tensor([target_steering, target_speed]))
            
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
        patch_features = patches.mean(dim=-1)  # 各パッチの平均特徴
        
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


class DrivingController(nn.Module):
    """映像データから制御信号を生成するコントローラー"""
    
    def __init__(
        self,
        frame_height: int = 64,
        frame_width: int = 64,
        hidden_dim: int = 64,
        K: int = 2,
        use_lgtcn: bool = True
    ):
        super().__init__()
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.hidden_dim = hidden_dim
        self.K = K
        self.use_lgtcn = use_lgtcn
        
        # CNN特徴抽出器
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # パッチ特徴エンコーダー
        patch_size = 8 * 8 * 128  # 8x8の128チャンネル特徴
        self.patch_encoder = nn.Sequential(
            nn.Linear(patch_size, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )
        
        # LGTCN または LTCN
        if use_lgtcn:
            from ..layers import LGTCNLayer
            self.temporal_processor = LGTCNLayer(hidden_dim, hidden_dim, K)
        else:
            from ..layers import LTCNLayer
            # パッチ数に応じてブロック数を調整
            num_patches = 64  # 8x8パッチ
            k_per_block = hidden_dim // 4  # ブロックあたりの次元
            num_blocks = 4
            self.temporal_processor = LTCNLayer(hidden_dim, k_per_block, num_blocks)
        
        # 制御信号デコーダー
        self.control_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # steering, speed
        )
        
    def forward(
        self,
        frames: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            frames: (B, T, H, W, C) 映像フレーム
            adjacency: (B, T, N, N) グラフ隣接行列
            hidden_state: 前の隠れ状態
            
        Returns:
            controls: (B, T, 2) 制御信号
            new_hidden_state: 新しい隠れ状態
        """
        B, T, H, W, C = frames.shape
        
        # フレームをCNN処理
        frames_flat = frames.view(-1, C, H, W)
        features = self.feature_extractor(frames_flat)  # (B*T, 128, 8, 8)
        
        # パッチ特徴として整形
        features = features.view(B, T, -1)  # (B, T, 128*8*8)
        patch_features = self.patch_encoder(features)  # (B, T, hidden_dim)
        
        if self.use_lgtcn:
            # LGTCNの場合：グラフ処理
            if adjacency is None:
                # デフォルトの隣接行列（全結合）
                num_nodes = 1  # 簡単化：1ノードとして扱う
                adjacency = torch.ones(B, num_nodes, num_nodes, device=frames.device)
            
            from ..utils import compute_support_powers
            S_powers = compute_support_powers(adjacency, self.K)
            S_powers_2d = [sp.squeeze(-3) for sp in S_powers] if len(S_powers) > 0 else []
            
            # 時系列処理
            controls = []
            current_hidden = hidden_state
            
            for t in range(T):
                if current_hidden is None:
                    current_hidden = torch.zeros(B, 1, self.hidden_dim, device=frames.device)
                
                # パッチ特徴を入力として使用
                input_features = patch_features[:, t:t+1, :]  # (B, 1, hidden_dim)
                
                # LGTCN処理
                next_hidden = self.temporal_processor(
                    current_hidden.squeeze(1), 
                    input_features.squeeze(1), 
                    S_powers_2d
                )
                next_hidden = next_hidden.unsqueeze(1)
                
                # 制御信号生成
                control = self.control_decoder(next_hidden.squeeze(1))
                controls.append(control)
                current_hidden = next_hidden
                
            controls = torch.stack(controls, dim=1)
            final_hidden = current_hidden
            
        else:
            # LTCNの場合：系列処理
            controls = []
            
            if hidden_state is None:
                hidden_state = torch.zeros(B, self.temporal_processor.N, device=frames.device)
            
            current_hidden = hidden_state
            
            for t in range(T):
                # LTCN処理
                next_hidden = self.temporal_processor(
                    current_hidden,
                    patch_features[:, t, :],  # (B, hidden_dim)
                    dt=0.1,
                    n_steps=1
                )
                
                # 制御信号生成（平均プール）
                pooled_hidden = next_hidden.view(B, -1, self.temporal_processor.k).mean(dim=1)
                control = self.control_decoder(pooled_hidden)
                controls.append(control)
                current_hidden = next_hidden
                
            controls = torch.stack(controls, dim=1)
            final_hidden = current_hidden
        
        return controls, final_hidden