from dataclasses import dataclass

@dataclass
class StabilityMetrics:
    """安定性メトリクス"""
    # 制御精度
    control_mse: float
    control_mae: float