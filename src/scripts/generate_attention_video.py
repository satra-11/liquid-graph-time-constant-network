#!/usr/bin/env python
"""
generate_attention_video.py - ドライブ映像からAttentionマップの動画を生成するスクリプト
"""

import argparse
from pathlib import Path
import imageio
import numpy as np
import torch
import matplotlib.pyplot as plt

from lgtcn.models import DrivingController
from scripts.train_driving import create_dataset, set_seed

def generate_video(
    model: DrivingController,
    frames: torch.Tensor,
    output_path: Path,
    device: torch.device,
):
    """Attentionマップを含む動画を生成する"""
    print("Generating attention video...")
    model = model.to(device)
    model.eval()

    # (T, H, W, C) -> (1, T, H, W, C)
    frames = frames.unsqueeze(0).to(device)

    with torch.no_grad():
        # モデルからAttentionを取得
        _, _, attentions = model(frames)

    if attentions is None:
        print("This model does not produce attentions (likely not an LGTCN model).")
        return

    # (1, T, N, N) -> (T, N, N)
    attentions = attentions.squeeze(0).cpu().numpy()
    frames = frames.squeeze(0).cpu().numpy()
    
    num_patches = attentions.shape[1] # N
    grid_size = int(np.sqrt(num_patches))

    with imageio.get_writer(output_path, mode='I', fps=10) as writer:
        for t in range(frames.shape[0]):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
            # 元のフレームを表示
            ax1.imshow(frames[t])
            ax1.set_title(f"Input Frame (t={t})")
            ax1.axis("off")
            
            # Attentionマップを表示
            # (N, N)のAttention行列から、全パッチの平均Attentionを計算
            avg_attention = attentions[t].mean(axis=0) # (N,)
            attention_map = avg_attention.reshape((grid_size, grid_size))
            
            im = ax2.imshow(attention_map, cmap='viridis')
            ax2.set_title("Average Attention Map")
            ax2.axis("off")
            
            fig.colorbar(im, ax=ax2)
            
            # FigureをNumpy配列に変換して動画フレームとして追加
            fig.canvas.draw()
            frame_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            frame_image = frame_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(frame_image)
            plt.close(fig)

    print(f"Video saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate attention map videos from a trained LGTCN model.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained LGTCN model (.pth file).")
    parser.add_argument("--num-sequences", type=int, default=10, help="Number of sequences to use from the video.")
    parser.add_argument("--sequence-length", type=int, default=50, help="Sequence length for the model.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension of the model.")
    parser.add_argument("--K", type=int, default=2, help="K value for the LGTCN model.")
    parser.add_argument("--save-dir", type=str, default="./driving_results", help="Directory to save the output video.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cpu, cuda).")
    parser.add_argument("--video-path", type=str, default=None, help="Path to a real driving video file.")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # デバイス設定
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    set_seed(args.seed)

    # 保存ディレクトリ作成
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # モデルのロード
    print(f"Loading model from {args.model_path}...")
    model = DrivingController(
        hidden_dim=args.hidden_dim,
        K=args.K,
        use_lgtcn=True  # Attention生成はLGTCNのみ対応
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # データセット作成
    print("Creating dataset...")
    _, corrupted_frames, _ = create_dataset(
        num_sequences=args.num_sequences,
        sequence_length=args.sequence_length,
        video_path=args.video_path,
    )

    # 最初のシーケンスを使って動画を生成
    video_frames = corrupted_frames[0] # (T, H, W, C)
    output_path = save_dir / "attention_video.gif"

    generate_video(model, video_frames, output_path, device)

if __name__ == "__main__":
    main()
