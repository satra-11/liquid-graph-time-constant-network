import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
import json


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


class StabilityAnalyzer:
    """ネットワークの安定性分析クラス"""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cpu')
        
    def analyze_hidden_state_stability(
        self,
        model: nn.Module,
        input_sequence: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """隠れ状態の安定性を分析"""
        model.eval()
        with torch.no_grad():
            # 初期状態からの発展を追跡
            hidden_states = []
            current_hidden = None
            
            for t in range(input_sequence.size(1)):
                frame = input_sequence[:, t:t+1]
                _, current_hidden = model(frame, adjacency, current_hidden)
                hidden_states.append(current_hidden.clone())
            
            hidden_states = torch.stack(hidden_states, dim=1)  # (B, T, ...)
            
            # 分散の計算
            variance = torch.var(hidden_states, dim=1).mean().item()
            
            # ドリフトの計算（線形トレンド）
            time_steps = torch.arange(hidden_states.size(1), dtype=torch.float32)
            drift = self._calculate_drift(hidden_states, time_steps)
            
            # リアプノフ指数の近似
            lyapunov = self._estimate_lyapunov_exponent(hidden_states)
            
        return {
            'variance': variance,
            'drift': drift,
            'lyapunov_exponent': lyapunov
        }
    
    def _calculate_drift(self, states: torch.Tensor, time_steps: torch.Tensor) -> float:
        """状態のドリフト（時間的傾向）を計算"""
        B, T = states.shape[:2]
        states_flat = states.view(B, T, -1).mean(dim=-1)  # (B, T)
        
        drifts = []
        for b in range(B):
            # 線形回帰で傾きを計算
            y = states_flat[b].cpu().numpy()
            x = time_steps.cpu().numpy()
            
            # y = ax + b の a を計算
            n = len(x)
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
            drifts.append(abs(slope))
        
        return np.mean(drifts)
    
    def _estimate_lyapunov_exponent(self, states: torch.Tensor) -> float:
        """リアプノフ指数を近似計算"""
        B, T = states.shape[:2]
        states_flat = states.view(B, T, -1)
        
        # 連続する状態間の距離の対数を計算
        log_distances = []
        
        for t in range(1, T):
            prev_state = states_flat[:, t-1, :]
            curr_state = states_flat[:, t, :]
            
            distance = torch.norm(curr_state - prev_state, dim=-1)
            # 数値安定性のため小さな値を追加
            log_distance = torch.log(distance + 1e-8)
            log_distances.append(log_distance.mean().item())
        
        # 平均的な発散率
        if len(log_distances) > 1:
            return np.mean(np.diff(log_distances))
        else:
            return 0.0
    
    def analyze_corruption_resilience(
        self,
        model: nn.Module,
        clean_data: torch.Tensor,
        corrupted_data: torch.Tensor,
        targets: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """汚損に対する耐性を分析"""
        model.eval()
        
        with torch.no_grad():
            # クリーンデータでの予測
            clean_pred, _ = model(clean_data, adjacency)
            clean_loss = nn.MSELoss()(clean_pred, targets)
            
            # 汚損データでの予測
            corrupted_pred, _ = model(corrupted_data, adjacency)
            corrupted_loss = nn.MSELoss()(corrupted_pred, targets)
            
            # 耐性指標（性能劣化の少なさ）
            resilience = 1.0 - (corrupted_loss - clean_loss) / (clean_loss + 1e-8)
            
            # 回復時間の計算（段階的に汚損を減らしていく）
            recovery_time = self._estimate_recovery_time(
                model, clean_data, corrupted_data, targets, adjacency
            )
            
        return {
            'resilience': resilience.item(),
            'recovery_time': recovery_time,
            'clean_loss': clean_loss.item(),
            'corrupted_loss': corrupted_loss.item()
        }
    
    def _estimate_recovery_time(
        self,
        model: nn.Module,
        clean_data: torch.Tensor,
        corrupted_data: torch.Tensor,
        targets: torch.Tensor,
        adjacency: Optional[torch.Tensor],
        steps: int = 10
    ) -> float:
        """回復時間を推定"""
        clean_loss = nn.MSELoss()(model(clean_data, adjacency)[0], targets)
        
        for step in range(steps):
            # 徐々に汚損を減らす
            alpha = 1.0 - (step + 1) / steps
            mixed_data = alpha * corrupted_data + (1 - alpha) * clean_data
            
            pred, _ = model(mixed_data, adjacency)
            loss = nn.MSELoss()(pred, targets)
            
            # クリーンデータのlossに近づいたら回復したとみなす
            if abs(loss - clean_loss) / clean_loss < 0.1:
                return (step + 1) / steps
        
        return 1.0  # 回復に時間がかかる
    
    def analyze_prediction_consistency(
        self,
        model: nn.Module,
        data: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None,
        num_runs: int = 5
    ) -> Dict[str, float]:
        """予測の一貫性を分析（異なる初期化での変動）"""
        model.eval()
        
        predictions = []
        hidden_trajectories = []
        
        for run in range(num_runs):
            with torch.no_grad():
                pred, final_hidden = model(data, adjacency, hidden_state=None)
                predictions.append(pred)
                
                # 隠れ状態の軌跡も記録
                hidden_states = []
                current_hidden = None
                for t in range(data.size(1)):
                    frame = data[:, t:t+1]
                    _, current_hidden = model(frame, adjacency, current_hidden)
                    hidden_states.append(current_hidden.clone())
                hidden_trajectories.append(torch.stack(hidden_states, dim=1))
        
        predictions = torch.stack(predictions)  # (num_runs, B, T, output_dim)
        hidden_trajectories = torch.stack(hidden_trajectories)  # (num_runs, B, T, ...)
        
        # 予測の標準偏差
        pred_std = torch.std(predictions, dim=0).mean().item()
        
        # 時間的滑らかさ
        temporal_smoothness = self._calculate_temporal_smoothness(predictions.mean(dim=0))
        
        # 隠れ状態軌跡の一貫性
        hidden_consistency = 1.0 - torch.std(hidden_trajectories, dim=0).mean().item()
        
        return {
            'prediction_std': pred_std,
            'temporal_smoothness': temporal_smoothness,
            'hidden_consistency': hidden_consistency
        }
    
    def _calculate_temporal_smoothness(self, predictions: torch.Tensor) -> float:
        """時間的滑らかさを計算"""
        # 連続フレーム間の差分の平均
        temporal_diff = torch.diff(predictions, dim=1)
        smoothness = -torch.norm(temporal_diff, dim=-1).mean().item()
        return smoothness


class NetworkComparator:
    """LGTCNとLTCNの比較分析クラス"""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cpu')
        self.analyzer = StabilityAnalyzer(device)
    
    def compare_networks(
        self,
        lgtcn_model: nn.Module,
        ltcn_model: nn.Module,
        test_data: Dict[str, torch.Tensor],
        corruption_levels: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    ) -> Dict[str, Any]:
        """LGTCNとLTCNを包括的に比較"""
        
        results = {
            'lgtcn': {},
            'ltcn': {},
            'comparison': {}
        }
        
        # 各汚損レベルでテスト
        for corruption_level in corruption_levels:
            print(f"Testing corruption level: {corruption_level}")
            
            # データ準備
            from .autonomous_driving import VideoProcessor, CorruptionConfig
            config = CorruptionConfig(
                missing_rate=corruption_level,
                whiteout_rate=corruption_level * 0.5,
                noise_level=corruption_level * 0.1
            )
            processor = VideoProcessor(config)
            
            clean_frames = test_data['clean_frames']
            targets = test_data['targets']
            adjacency = test_data.get('adjacency')
            
            # 汚損フレーム生成
            corrupted_frames = torch.stack([
                processor.corrupt_frame(frame) for frame in clean_frames
            ])
            
            # LGTCNテスト
            lgtcn_metrics = self._evaluate_model(
                lgtcn_model, clean_frames, corrupted_frames, targets, adjacency
            )
            results['lgtcn'][f'corruption_{corruption_level}'] = lgtcn_metrics
            
            # LTCNテスト
            ltcn_metrics = self._evaluate_model(
                ltcn_model, clean_frames, corrupted_frames, targets, adjacency
            )
            results['ltcn'][f'corruption_{corruption_level}'] = ltcn_metrics
        
        # 比較サマリー
        results['comparison'] = self._generate_comparison_summary(results)
        
        return results
    
    def _evaluate_model(
        self,
        model: nn.Module,
        clean_data: torch.Tensor,
        corrupted_data: torch.Tensor,
        targets: torch.Tensor,
        adjacency: Optional[torch.Tensor]
    ) -> StabilityMetrics:
        """単一モデルの評価"""
        
        # 制御精度
        model.eval()
        with torch.no_grad():
            pred_clean, _ = model(clean_data, adjacency)
            pred_corrupted, _ = model(corrupted_data, adjacency)
            
            control_mse = nn.MSELoss()(pred_corrupted, targets).item()
            control_mae = nn.L1Loss()(pred_corrupted, targets).item()
        
        # 隠れ状態安定性
        stability_metrics = self.analyzer.analyze_hidden_state_stability(
            model, corrupted_data, adjacency
        )
        
        # 汚損耐性
        resilience_metrics = self.analyzer.analyze_corruption_resilience(
            model, clean_data, corrupted_data, targets, adjacency
        )
        
        # 予測一貫性
        consistency_metrics = self.analyzer.analyze_prediction_consistency(
            model, corrupted_data, adjacency
        )
        
        return StabilityMetrics(
            control_mse=control_mse,
            control_mae=control_mae,
            hidden_state_variance=stability_metrics['variance'],
            hidden_state_drift=stability_metrics['drift'],
            lyapunov_exponent=stability_metrics['lyapunov_exponent'],
            corruption_resilience=resilience_metrics['resilience'],
            recovery_time=resilience_metrics['recovery_time'],
            prediction_consistency=consistency_metrics['hidden_consistency'],
            temporal_smoothness=consistency_metrics['temporal_smoothness']
        )
    
    def _generate_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """比較サマリーを生成"""
        summary = {
            'winner_by_metric': {},
            'robustness_comparison': {},
            'stability_comparison': {}
        }
        
        # 各メトリクスでの優劣
        metrics = ['control_mse', 'control_mae', 'corruption_resilience', 
                  'prediction_consistency', 'temporal_smoothness']
        
        for metric in metrics:
            lgtcn_values = []
            ltcn_values = []
            
            for key in results['lgtcn'].keys():
                if key.startswith('corruption_'):
                    lgtcn_val = getattr(results['lgtcn'][key], metric)
                    ltcn_val = getattr(results['ltcn'][key], metric)
                    lgtcn_values.append(lgtcn_val)
                    ltcn_values.append(ltcn_val)
            
            lgtcn_avg = np.mean(lgtcn_values)
            ltcn_avg = np.mean(ltcn_values)
            
            # 小さい方が良いメトリクス
            if metric in ['control_mse', 'control_mae']:
                winner = 'LGTCN' if lgtcn_avg < ltcn_avg else 'LTCN'
            else:
                winner = 'LGTCN' if lgtcn_avg > ltcn_avg else 'LTCN'
            
            summary['winner_by_metric'][metric] = {
                'winner': winner,
                'lgtcn_avg': lgtcn_avg,
                'ltcn_avg': ltcn_avg,
                'difference': abs(lgtcn_avg - ltcn_avg)
            }
        
        return summary
    
    def visualize_comparison(
        self,
        results: Dict[str, Any],
        save_path: Optional[Path] = None
    ):
        """比較結果を可視化"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        corruption_levels = [float(k.split('_')[1]) for k in results['lgtcn'].keys() 
                           if k.startswith('corruption_')]
        corruption_levels.sort()
        
        metrics_to_plot = [
            ('control_mse', 'Control MSE'),
            ('control_mae', 'Control MAE'),
            ('corruption_resilience', 'Corruption Resilience'),
            ('hidden_state_variance', 'Hidden State Variance'),
            ('prediction_consistency', 'Prediction Consistency'),
            ('temporal_smoothness', 'Temporal Smoothness')
        ]
        
        for i, (metric, title) in enumerate(metrics_to_plot):
            lgtcn_values = []
            ltcn_values = []
            
            for corruption_level in corruption_levels:
                key = f'corruption_{corruption_level}'
                lgtcn_val = getattr(results['lgtcn'][key], metric)
                ltcn_val = getattr(results['ltcn'][key], metric)
                lgtcn_values.append(lgtcn_val)
                ltcn_values.append(ltcn_val)
            
            axes[i].plot(corruption_levels, lgtcn_values, 'b-o', label='LGTCN')
            axes[i].plot(corruption_levels, ltcn_values, 'r-s', label='LTCN')
            axes[i].set_xlabel('Corruption Level')
            axes[i].set_ylabel(title)
            axes[i].set_title(title)
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def save_results(self, results: Dict[str, Any], save_path: Path):
        """結果をJSONファイルに保存"""
        # StabilityMetricsをdict形式に変換
        def metrics_to_dict(metrics):
            if isinstance(metrics, StabilityMetrics):
                return {
                    'control_mse': metrics.control_mse,
                    'control_mae': metrics.control_mae,
                    'hidden_state_variance': metrics.hidden_state_variance,
                    'hidden_state_drift': metrics.hidden_state_drift,
                    'lyapunov_exponent': metrics.lyapunov_exponent,
                    'corruption_resilience': metrics.corruption_resilience,
                    'recovery_time': metrics.recovery_time,
                    'prediction_consistency': metrics.prediction_consistency,
                    'temporal_smoothness': metrics.temporal_smoothness
                }
            return metrics
        
        serializable_results = {}
        for network_type, network_results in results.items():
            serializable_results[network_type] = {}
            for key, value in network_results.items():
                serializable_results[network_type][key] = metrics_to_dict(value)
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {save_path}")