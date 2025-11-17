import torch
import torch.nn.functional as F


@torch.no_grad()
def add_whiteout(
    frame: torch.Tensor,
    stops: float = 1.0,  # 露出オーバー量（絞り段）→ 2**stops 倍。強い
    bloom_strength: float = 1.0,  # にじみ量（強い）
    bloom_sigma: float = 10.0,  # にじみの広がり（強い）
    wash: float = 0.2,  # 脱色（0で色保持、1で完全モノクロ寄り）
    lift: float = 0.12,  # 黒浮き（ベース持ち上げ）
    contrast: float = 0.7,  # コントラスト（<1で低下）
    center_bias: float = 0.5,  # 画面中心をさらに飛ばす
    gamma: float = 2.0,  # トーンの自然さ用
) -> torch.Tensor:
    """
    強めの「露出オーバー白飛び」生成。frame: [T,H,W,C] or [H,W,C] in [0,1]
    """
    # --- shape整形 ---
    if frame.dim() == 4:  # [T,H,W,C] -> [N,C,H,W]
        x = frame.permute(0, 3, 1, 2).contiguous()
    elif frame.dim() == 3:  # [H,W,C] -> [1,C,H,W]
        x = frame.permute(2, 0, 1).unsqueeze(0).contiguous()
    else:
        raise ValueError("frame must be [T,H,W,C] or [H,W,C].")
    N, C, H, W = x.shape
    dev, dt = x.device, x.dtype

    # --- 1) ガンマ→線形空間で“露出オーバー”を素直に掛ける ---
    x_lin = torch.clamp(x, 0, 1) ** gamma
    x_lin = x_lin * (2.0**stops)  # ここが露出過多の本体

    # --- 2) 1.0を超えたハイライトを抽出して太めのbloom ---
    overflow = torch.relu(x_lin - 1.0)
    if C > 1:
        lum = (
            0.2126 * overflow[:, 0:1]
            + 0.7152 * overflow[:, 1:2]
            + 0.0722 * overflow[:, 2:3]
        )
    else:
        lum = overflow

    # separable Gaussian（簡素実装）
    k = int(2 * round(3 * max(1.0, bloom_sigma)) + 1)
    t = torch.arange(k, device=dev, dtype=dt) - k // 2
    g = torch.exp(-0.5 * (t / bloom_sigma) ** 2)
    g = g / g.sum()
    gH = g.view(1, 1, 1, -1)
    gV = g.view(1, 1, -1, 1)
    bloom_map = F.conv2d(lum, gH, padding=(0, k // 2))
    bloom_map = F.conv2d(bloom_map, gV, padding=(k // 2, 0))

    # RGBへ反映
    if C > 1:
        bloom_rgb = bloom_map.repeat(1, C, 1, 1)
    else:
        bloom_rgb = bloom_map
    x_lin = x_lin + bloom_strength * bloom_rgb

    # --- 3) 脱色＋黒浮き＋コントラスト低下（“白く眠い”画に寄せる） ---
    if C > 1 and wash > 0:
        gray = (
            0.2126 * x_lin[:, 0:1] + 0.7152 * x_lin[:, 1:2] + 0.0722 * x_lin[:, 2:3]
        ).repeat(1, C, 1, 1)
        x_lin = x_lin * (1 - wash) + gray * wash

    # 黒を持ち上げて全体を白寄りに
    x_lin = x_lin + lift
    # コントラストを落とす
    x_lin = (x_lin - 0.5) * contrast + 0.5

    # --- 4) 中心をさらに飛ばす（露出の偏り再現） ---
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=dev, dtype=dt),
        torch.linspace(-1, 1, W, device=dev, dtype=dt),
        indexing="ij",
    )
    center = 1 - torch.clamp(torch.sqrt(xx**2 + yy**2), 0, 1)  # 中心1, 端0
    center = center[None, None]  # [1,1,H,W]
    x_lin = x_lin * (1.0 + center_bias * center)

    # --- 5) 逆ガンマ＆クリップ ---
    out = torch.clamp(x_lin, 0, 1) ** (1.0 / gamma)

    # --- 元shapeへ ---
    if frame.dim() == 4:
        return out.permute(0, 2, 3, 1)
    else:
        return out.squeeze(0).permute(1, 2, 0)


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
