import torch


def add_whiteout(self, frame: torch.Tensor) -> torch.Tensor:
    """フレームに白飛びを追加"""
    whiteout_rate = 0.1
            
    whiteout_mask = torch.rand(frame.shape[:-1]) < whiteout_rate
    whiteout_mask = whiteout_mask.unsqueeze(-1).expand_as(frame)
        
    corrupted_frame = frame.clone()
    corrupted_frame[whiteout_mask] = 1.0  # 白飛び
    return corrupted_frame
    
def add_noise(self, frame: torch.Tensor) -> torch.Tensor:
    """フレームにノイズを追加"""
    noise_level = 0.1   
    noise = torch.randn_like(frame) * noise_level
    return torch.clamp(frame + noise, 0.0, 1.0)