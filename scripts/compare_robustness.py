#!/usr/bin/env python3
"""
Compare robustness of LGTCN vs LTCN on time series image data with missing patterns.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from tqdm import tqdm

from lgtcn.layers import LGTCNLayer, LTCNLayer
from lgtcn.utils import MissingDataGenerator, ImageToSequence, TimeSeriesImageDataset, compute_support_powers


class ImageClassifier(nn.Module):
    """Image classifier using either LGTCN or LTCN layers."""
    
    def __init__(self, layer_type: str, input_dim: int = 768, hidden_dim: int = 64, 
                 num_classes: int = 10, K: int = 2, num_nodes: int = 49):
        super().__init__()
        self.layer_type = layer_type
        self.num_nodes = num_nodes
        
        # Input embedding
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Core layer (LGTCN or LTCN)
        if layer_type == 'lgtcn':
            self.core_layer = LGTCNLayer(hidden_dim, hidden_dim, K)
            self.requires_graph = True
        elif layer_type == 'ltcn':
            self.core_layer = LTCNLayer(hidden_dim, hidden_dim)
            self.requires_graph = False
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
            
        # Output head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Create simple grid graph for patches
        if self.requires_graph:
            self.register_buffer('graph', self._create_grid_graph())
    
    def _create_grid_graph(self) -> torch.Tensor:
        """Create adjacency matrix for grid-structured patches."""
        grid_size = int(np.sqrt(self.num_nodes))
        adj = torch.zeros(self.num_nodes, self.num_nodes)
        
        for i in range(grid_size):
            for j in range(grid_size):
                node_id = i * grid_size + j
                # Connect to neighbors (4-connectivity)
                if i > 0:  # up
                    adj[node_id, (i-1) * grid_size + j] = 1
                if i < grid_size - 1:  # down
                    adj[node_id, (i+1) * grid_size + j] = 1
                if j > 0:  # left
                    adj[node_id, i * grid_size + (j-1)] = 1
                if j < grid_size - 1:  # right
                    adj[node_id, i * grid_size + (j+1)] = 1
                    
        return adj
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (B, T, N_patches, patch_dim)
        """
        B, T, N, D = x.shape
        
        # Project input
        x_emb = self.input_proj(x)  # (B, T, N, hidden_dim)
        
        # Initialize hidden state
        if x_emb.ndim == 4:  # batch mode
            h = torch.zeros(B, N, x_emb.size(-1), device=x.device)
        else:
            h = torch.zeros(N, x_emb.size(-1), device=x.device)
        
        # Process sequence
        outputs = []
        for t in range(T):
            if self.requires_graph:
                S_powers = compute_support_powers(self.graph, 2)  # K=2
                h = self.core_layer(h, x_emb[:, t], S_powers)
            else:
                h = self.core_layer(h, x_emb[:, t])
            outputs.append(h)
        
        # Pool over patches and time
        final_h = torch.stack(outputs, dim=1)  # (B, T, N, hidden_dim)
        pooled = final_h.mean(dim=(1, 2))  # (B, hidden_dim)
        
        return self.classifier(pooled)


def create_synthetic_data(batch_size: int = 32, seq_len: int = 10, image_size: int = 112, 
                         num_classes: int = 10) -> torch.Tensor:
    """Create synthetic time series image data."""
    # Generate synthetic RGB images
    images = torch.randn(batch_size, seq_len, 3, image_size, image_size)
    images = torch.tanh(images)  # Normalize to [-1, 1]
    
    # Generate synthetic labels
    labels = torch.randint(0, num_classes, (batch_size,))
    
    return images, labels


def evaluate_robustness(model: nn.Module, dataset: TimeSeriesImageDataset, 
                       missing_types: list, missing_rates: list, device: str = 'cpu') -> dict:
    """Evaluate model robustness under different missing patterns."""
    model.eval()
    results = {}
    
    with torch.no_grad():
        for missing_type in missing_types:
            results[missing_type] = {}
            for missing_rate in missing_rates:
                batch_data = dataset.get_batch(missing_type, missing_rate)
                
                # Move to device
                corrupted = batch_data['corrupted'].to(device)
                clean = batch_data['clean'].to(device)
                
                # Generate dummy labels
                labels = torch.randint(0, 10, (corrupted.size(0),)).to(device)
                
                # Forward pass
                logits_corrupted = model(corrupted)
                logits_clean = model(clean)
                
                # Compute metrics
                acc_corrupted = (logits_corrupted.argmax(-1) == labels).float().mean().item()
                acc_clean = (logits_clean.argmax(-1) == labels).float().mean().item()
                
                # KL divergence between clean and corrupted predictions
                p_clean = F.softmax(logits_clean, dim=-1)
                p_corrupted = F.softmax(logits_corrupted, dim=-1)
                kl_div = F.kl_div(p_corrupted.log(), p_clean, reduction='batchmean').item()
                
                results[missing_type][missing_rate] = {
                    'accuracy_corrupted': acc_corrupted,
                    'accuracy_clean': acc_clean,
                    'accuracy_drop': acc_clean - acc_corrupted,
                    'kl_divergence': kl_div
                }
    
    return results


def plot_robustness_curves(results_lgtcn: dict, results_ltcn: dict, output_dir: Path):
    """Plot robustness comparison curves."""
    missing_types = list(results_lgtcn.keys())
    missing_rates = sorted(list(results_lgtcn[missing_types[0]].keys()))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('LGTCN vs LTCN Robustness Comparison', fontsize=16)
    
    for i, missing_type in enumerate(missing_types[:4]):  # Plot first 4 types
        ax = axes[i//2, i%2]
        
        # Extract accuracy data
        lgtcn_acc = [results_lgtcn[missing_type][rate]['accuracy_corrupted'] for rate in missing_rates]
        ltcn_acc = [results_ltcn[missing_type][rate]['accuracy_corrupted'] for rate in missing_rates]
        
        ax.plot(missing_rates, lgtcn_acc, 'o-', label='LGTCN', linewidth=2)
        ax.plot(missing_rates, ltcn_acc, 's--', label='LTCN', linewidth=2)
        
        ax.set_xlabel('Missing Rate')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{missing_type.title()} Missing')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'robustness_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare LGTCN vs LTCN robustness')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=112)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    
    # Create synthetic data
    print("Creating synthetic time series image data...")
    images, labels = create_synthetic_data(args.batch_size, args.seq_len, args.image_size, args.num_classes)
    
    # Setup missing data processing
    missing_gen = MissingDataGenerator(seed=args.seed)
    patch_converter = ImageToSequence(patch_size=args.patch_size)
    dataset = TimeSeriesImageDataset(images, missing_gen, patch_converter)
    
    # Calculate dimensions
    sample_patches = patch_converter(images[:1])
    patch_dim = sample_patches.shape[-1]
    num_patches = sample_patches.shape[2]
    
    print(f"Patch dimensions: {patch_dim}, Number of patches: {num_patches}")
    
    # Initialize models
    print("Initializing models...")
    model_lgtcn = ImageClassifier('lgtcn', patch_dim, args.hidden_dim, args.num_classes, 
                                 num_nodes=num_patches).to(device)
    model_ltcn = ImageClassifier('ltcn', patch_dim, args.hidden_dim, args.num_classes,
                                num_nodes=num_patches).to(device)
    
    # Test configurations
    missing_types = ['random', 'block', 'stripe_row', 'noise']
    missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    print("Evaluating LGTCN robustness...")
    results_lgtcn = evaluate_robustness(model_lgtcn, dataset, missing_types, missing_rates, device)
    
    print("Evaluating LTCN robustness...")
    results_ltcn = evaluate_robustness(model_ltcn, dataset, missing_types, missing_rates, device)
    
    # Save results
    results = {
        'lgtcn': results_lgtcn,
        'ltcn': results_ltcn,
        'config': vars(args)
    }
    
    with open(output_dir / 'robustness_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot results
    print("Generating plots...")
    plot_robustness_curves(results_lgtcn, results_ltcn, output_dir)
    
    # Print summary
    print("\n=== Robustness Comparison Summary ===")
    for missing_type in missing_types:
        print(f"\n{missing_type.upper()} Missing:")
        for rate in [0.3, 0.5, 0.7]:  # Key missing rates
            if rate in missing_rates:
                lgtcn_acc = results_lgtcn[missing_type][rate]['accuracy_corrupted']
                ltcn_acc = results_ltcn[missing_type][rate]['accuracy_corrupted']
                print(f"  {rate*100:2.0f}%: LGTCN={lgtcn_acc:.3f}, LTCN={ltcn_acc:.3f}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()