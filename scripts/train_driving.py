#!/usr/bin/env python
"""
映像データによる自律走行タスクでのLGTCN/LTCN訓練スクリプト
"""
import argparse
import random
from pathlib import Path
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from src.tasks import (
    NetworkComparator,
)
from src.core.models import LGTCNController, LTCNController
from src.data import HDDLoader

def main():
    parser = argparse.ArgumentParser(description="Train driving controllers with LGTCN/LTCN")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-sequences", type=int, default=800)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--corruption-rate", type=float, default=0.2)
    parser.add_argument("--data-dir", type=str, default="./hdd")
    parser.add_argument("--save-dir", type=str, default="./driving_results")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--sensor-sequence", type=str, default=None, help="Sensor sequence name for testing (e.g., 201702271017)")
    
    args = parser.parse_args()
    
    # デバイス設定
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # シード設定
    set_seed(args.seed)
    
    # 保存ディレクトリ作成
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # データセット作成
    print("Loading dataset from HDD...")
    
    full_dataset = HDDLoader(
        camera_dir=os.path.join(args.data_dir, 'camera'),
        sensor_dir=os.path.join(args.data_dir, 'sensor'),
        sequence_length=args.sequence_length,
    )
    
    # 訓練・検証・テストに分割
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # データローダー作成
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # モデル作成
    print("Creating models...")
    lgtcn_model = LGTCNController(
        frame_height=64,
        frame_width=64,
        hidden_dim=args.hidden_dim,
        output_dim=8,
        K=args.K
    )
    
    ltcn_model = LTCNController(
        frame_height=64,
        frame_width=64,
        output_dim=8,
        hidden_dim=args.hidden_dim,
    )
    
    # LGTCN訓練
    print("Training LGTCN...")
    lgtcn_train_losses, lgtcn_val_losses = train_model(
        lgtcn_model, train_loader, val_loader,
        num_epochs=args.epochs, learning_rate=args.lr, device=device
    )
    
    # LTCN訓練
    print("Training LTCN...")
    ltcn_train_losses, ltcn_val_losses = train_model(
        ltcn_model, train_loader, val_loader,
        num_epochs=args.epochs, learning_rate=args.lr, device=device
    )
    
    # 学習曲線をプロット
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(lgtcn_train_losses, label='LGTCN Train')
    plt.plot(ltcn_train_losses, label='LTCN Train')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(lgtcn_val_losses, label='LGTCN Val')
    plt.plot(ltcn_val_losses, label='LTCN Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png")
    plt.show()
    
    # # テストデータで評価
    # print("Evaluating networks...")
    # test_data = {
    #     'clean_frames': clean_frames[test_indices],
    #     'corrupted_frames': corrupted_frames[test_indices],
    #     'sensors': sensors[test_indices]
    # }
    # 
    # results = evaluate_networks(lgtcn_model, ltcn_model, test_data, device)
    # 
    # # 結果を保存
    # comparator = NetworkComparator(device)
    # comparator.save_results(results, save_dir / "comparison_results.json")
    # comparator.visualize_comparison(results, save_dir / "comparison_plots.png")
    
    # モデル保存
    torch.save(lgtcn_model.state_dict(), save_dir / "lgtcn_model.pth")
    torch.save(ltcn_model.state_dict(), save_dir / "ltcn_model.pth")
    
    # 訓練情報保存
    training_info = {
        'args': vars(args),
        'lgtcn_train_losses': lgtcn_train_losses,
        'lgtcn_val_losses': lgtcn_val_losses,
        'ltcn_train_losses': ltcn_train_losses,
        'ltcn_val_losses': ltcn_val_losses,
    }
    
    with open(save_dir / "training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"Training completed! Results saved to {save_dir}")
    
    # # 結果サマリー表示
    # comparison = results['comparison']
    # print("\n=== Comparison Summary ===")
    # for metric, data in comparison['winner_by_metric'].items():
    #     print(f"{metric}: {data['winner']} wins (LGTCN: {data['lgtcn_avg']:.4f}, LTCN: {data['ltcn_avg']:.4f})")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}.")




def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(
    model: LTCNController | LGTCNController,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    device: torch.device = None
):
    """モデルを訓練"""
    device = device or torch.device('cpu')
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    start_time = time.time()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 訓練フェーズ
        model.train()
        epoch_train_loss = 0.0
        count = 0
        
        for batch_idx, (frames, sensors, _, _) in enumerate(train_loader):
            frames = frames.to(device)
            sensors = sensors.to(device)
            print(f"学習中{count} / {len(train_loader)}", end='\r')
            count += 1
            
            optimizer.zero_grad()
            
            predictions, _ = model(frames)
            
            loss = criterion(predictions[:, -1, :], sensors[:, -1, :])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 検証フェーズ
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for frames, sensors, _, _ in val_loader:
                frames = frames.to(device)
                sensors = sensors.to(device)
                
                predictions, _ = model(frames)
                
                # 予測の最後のタイムステップと比較
                loss = criterion(predictions[:, -1, :], sensors[:, -1, :])
                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training finished in {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}.")
    
    return train_losses, val_losses


def evaluate_networks(
    lgtcn_model: LGTCNController,
    ltcn_model: LTCNController,
    test_data: dict,
    device: torch.device
):
    """LGTCNとLTCNを比較評価"""
    comparator = NetworkComparator(device)
    
    # テストデータ準備
    test_dict = {
        'clean_frames': test_data['clean_frames'],
        'sensors': test_data['sensors'],
        'adjacency': None 
    }
    
    print("Comparing LGTCN and LTCN...")
    results = comparator.compare_networks(
        lgtcn_model, ltcn_model, test_dict,
        corruption_levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    )
    
    return results

