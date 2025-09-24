import torch
import torch.nn as nn
from typing import Optional, Tuple

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