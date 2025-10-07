from dataclasses import dataclass

@dataclass
class CorruptionConfig:
    """映像データの欠損・白飛び設定"""
    missing_rate: float = 0.1  # 欠損率
    whiteout_rate: float = 0.05  # 白飛び率
    noise_level: float = 0.02  # ノイズレベル
    blur_kernel: int = 3  # ブラーカーネルサイズ
