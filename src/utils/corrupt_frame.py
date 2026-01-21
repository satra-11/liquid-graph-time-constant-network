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


@torch.no_grad()
def simulate_tunnel_exit(
    frames: torch.Tensor,
    exit_idx: int = 10,
    peak_intensity: float = 3.0,
    dark_factor: float = 0.3,
    transition_duration: int = 5,
) -> torch.Tensor:
    """
    Level 3: トンネル出口シミュレーション (Time-varying Exposure)
    暗い -> 急に真っ白 -> 普通 という時系列変化を与える。

    Args:
        frames: [T, H, W, C] or [B, T, H, W, C] in [0,1]
        exit_idx: 出口を出る（白飛び開始）タイミングのインデックス
        peak_intensity: 白飛びのピーク強度 (factor)
        dark_factor: トンネル内の暗さ係数 (< 1.0)
        transition_duration: 白飛びから通常に戻るまでの期間 (フレーム数)
    """
    if frames.dim() == 4:
        # [T, H, W, C]
        T, H, W, C = frames.shape
        batch_mode = False
    elif frames.dim() == 5:
        # [B, T, H, W, C]
        B, T, H, W, C = frames.shape
        batch_mode = True
    else:
        raise ValueError("frames must be [T, H, W, C] or [B, T, H, W, C]")

    out_frames = frames.clone()

    # タイムステップの次元インデックス (4Dなら0, 5Dなら1)
    t_dim = 1 if batch_mode else 0
    len_t = frames.shape[t_dim]

    # タイムステップごとに処理
    for t in range(len_t):
        if t < exit_idx:
            # トンネル内: 暗い
            factor = dark_factor
        elif t < exit_idx + transition_duration:
            # 出口直後: 白飛び -> 徐々に回復
            progress = (t - exit_idx) / transition_duration
            current_intensity = peak_intensity - (peak_intensity - 1.0) * progress
            factor = current_intensity
        else:
            # 通常走行
            factor = 1.0

        if batch_mode:
            # [B, T, H, W, C] -> target [B, H, W, C] at index t
            out_frames[:, t] = torch.clamp(frames[:, t] * factor, 0.0, 1.0)
        else:
            # [T, H, W, C] -> target [H, W, C] at index t
            out_frames[t] = torch.clamp(frames[t] * factor, 0.0, 1.0)

    return out_frames
