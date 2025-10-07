from dataclasses import dataclass

@dataclass
class StabilityMetrics:
    """安定性メトリクス"""
    # 制御精度
    control_mse: float
    control_mae: float
    
    # 内部状態の安定性
    hidden_state_variance: float
    hidden_state_drift: float
    lyapunov_exponent: float
    
    # ロバストネス
    corruption_resilience: float
    recovery_time: float
    
    # 予測の一貫性
    prediction_consistency: float
    temporal_smoothness: float