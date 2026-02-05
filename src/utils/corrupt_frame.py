import torch


def add_gaussian_noise(
    frame: torch.Tensor, mean: float = 0.0, std: float = 0.1
) -> torch.Tensor:
    """
    ガウシアンノイズを追加する関数
    Args:
        frame (torch.Tensor): [T, H, W, C] または [C, H, W] など任意の画像テンソル (値域: 0〜1)
        mean (float): ノイズの平均
        std (float): ノイズの標準偏差（強さ）
    Returns:
        torch.Tensor: ノイズを加えたフレーム（クリップ済み）
    """
    noise = torch.normal(mean=mean, std=std, size=frame.shape, device=frame.device)
    noisy_frame = frame + noise
    return torch.clamp(noisy_frame, 0.0, 1.0)


@torch.no_grad()
def add_static_bias(frame: torch.Tensor, bias: float = 0.0) -> torch.Tensor:
    """
    Level 1: 全体的な輝度シフト (Static Bias)
    入力画像全体のピクセル値に定数を足して、クリッピングする。
    Args:
        frame: [T,H,W,C] or [H,W,C] in [0,1]
        bias: 加算する定数 (例: 0.1, 0.3, 0.5)
    """
    return torch.clamp(frame + bias, 0.0, 1.0)


@torch.no_grad()
def add_overexposure(frame: torch.Tensor, factor: float = 1.0) -> torch.Tensor:
    """
    Level 2: コントラストの破壊 (Overexposure)
    ゲインを上げて情報を飛ばす。
    Args:
        frame: [T,H,W,C] or [H,W,C] in [0,1]
        factor: 乗算する係数 (例: 1.2, 1.5, 2.0)
    """
    return torch.clamp(frame * factor, 0.0, 1.0)
