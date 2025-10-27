import torch
import torch.nn as nn
from typing import Optional, Tuple

from src.core.layers import CfGCNLayer
from src.utils import compute_s_powers, compute_laplacian, compute_random_walk_matrix

class CfGCNController(nn.Module):
    """映像データからCfGCNを使って制御信号を生成するコントローラー"""
    
    def __init__(
        self,
        frame_height: int = 64,
        frame_width: int = 64,
        hidden_dim: int = 64,
        K: int = 2,
        output_dim: int = 1,
        matrix_type: str = "adjacency",
    ):
        super().__init__()
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.hidden_dim = hidden_dim
        self.K = K
        self.output_dim = output_dim
        self.matrix_type = matrix_type
        self.node_encoder = nn.Linear(128, hidden_dim)
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        self.temporal_processor = CfGCNLayer(hidden_dim, hidden_dim, K)
        
        self.control_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim)
        )
        
    def forward(
        self,
        frames: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C, H, W = frames.shape
        
        frames_flat = frames.view(-1, C, H, W)
        features = self.feature_extractor(frames_flat)
        
        features = features.view(B, T, 128, 8, 8).permute(0,1,3,4,2)
        node_feats = features.view(B, T, 64, 128)
        node_feats = self.node_encoder(node_feats)
        
        attentions = []
        
        if adjacency is None:
            N = node_feats.size(2)
            adjacency = torch.ones(B, T, N, N, device=frames.device)
        
        controls = []
        current_hidden = hidden_state
        
        for t in range(T):
            if current_hidden is None:
                N = node_feats.size(2)
                current_hidden = torch.zeros(B, N, self.hidden_dim, device=frames.device)
            
            xt = node_feats[:, t, :, :]
            
            A_t = adjacency[:, t, :, :]
            attentions.append(A_t)

            if self.matrix_type == "adjacency":
                S_t = A_t
            elif self.matrix_type == "laplacian":
                S_t = compute_laplacian(A_t)
            elif self.matrix_type == "random_walk":
                S_t = compute_random_walk_matrix(A_t)
            else:
                raise ValueError(f"Unknown matrix type: {self.matrix_type}")

            S_powers = compute_s_powers(S_t, self.K)
            
            next_hidden = self.temporal_processor(
                current_hidden, 
                xt, 
                S_powers,
            )
            control = self.control_decoder(next_hidden.mean(dim=1))
            controls.append(control)
            current_hidden = next_hidden
            
        controls = torch.stack(controls, dim=1)
        final_hidden = current_hidden
        attentions = torch.stack(attentions, dim=1)
        
        return controls, final_hidden
