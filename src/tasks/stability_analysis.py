from matplotlib.path import Path
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import json
from src.models import LGTCNController
from src.data_models import StabilityMetrics
from src.utils import add_whiteout

class NetworkComparator:
    """LGTCNとLTCNの比較分析クラス"""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cpu')
    
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
            
            clean_frames = test_data['clean_frames']
            sensors = test_data['sensors']
            adjacency = test_data.get('adjacency')
            
            # 汚損フレーム生成
            corrupted_frames = torch.stack([
                add_whiteout(frame) for frame in clean_frames
            ])
            
            # LGTCNテスト
            lgtcn_metrics = self._evaluate_model(
                lgtcn_model, clean_frames, corrupted_frames, sensors, adjacency
            )
            results['lgtcn'][f'corruption_{corruption_level}'] = lgtcn_metrics
            
            # LTCNテスト
            ltcn_metrics = self._evaluate_model(
                ltcn_model, clean_frames, corrupted_frames, sensors, adjacency
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
        sensors: torch.Tensor,
        adjacency: Optional[torch.Tensor]
    ) -> StabilityMetrics:
        """単一モデルの評価"""
        
        clean_data = clean_data.to(self.device)
        corrupted_data = corrupted_data.to(self.device)
        sensors = sensors.to(self.device)
        if adjacency is not None:
            adjacency = adjacency.to(self.device)

        # 制御精度
        model.eval()
        with torch.no_grad():
            if isinstance(model, LGTCNController):
                pred_clean, _ = model(clean_data, adjacency)
                pred_corrupted, _ = model(corrupted_data, adjacency)
            else:
                pred_clean, _ = model(clean_data)
                pred_corrupted, _ = model(corrupted_data)
            
            sensors_steering = sensors[..., 0].unsqueeze(-1)
            control_mse = nn.MSELoss()(pred_corrupted, sensors_steering).item()
            control_mae = nn.L1Loss()(pred_corrupted, sensors_steering).item()
        

        
        return StabilityMetrics(
            control_mse=control_mse,
            control_mae=control_mae,
        )
    
    def _generate_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """比較サマリーを生成"""
        summary = {
            'winner_by_metric': {},
            'robustness_comparison': {},
            'stability_comparison': {}
        }
        
        # 各メトリクスでの優劣
        metrics = ['control_mse', 'control_mae',]
        
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
                'lgtcn_avg': float(lgtcn_avg),
                'ltcn_avg': float(ltcn_avg),
                'difference': float(abs(lgtcn_avg - ltcn_avg))
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
                    'control_mse': float(metrics.control_mse),
                    'control_mae': float(metrics.control_mae),
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