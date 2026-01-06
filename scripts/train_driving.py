#!/usr/bin/env python
"""
映像データによる自律走行タスクでのLGTCN/LTCN訓練スクリプト
"""

import time
import argparse
from src.driving.run import run_training


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train driving controllers with LGTCN/LTCN"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lgtcn", "ltcn", "node", "ngode"],
        help="Model to train: lgtcn, ltcn, node, or ngode",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-sequences", type=int, default=800)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--num-layers-ltcn", type=int, default=4)
    parser.add_argument("--num-layers-cfgcn", type=int, default=1)
    parser.add_argument("--corruption-rate", type=float, default=0.2)
    parser.add_argument("--data-dir", type=str, default="./data/raw")
    parser.add_argument("--processed-dir", type=str, default="./data/processed")
    parser.add_argument("--save-dir", type=str, default="./driving_results")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--sensor-sequence",
        type=str,
        default=None,
        help="Sensor sequence name for testing (e.g., 201702271017)",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to the directory containing checkpoint files to resume training from.",
    )
    args = parser.parse_args()

    start_time = time.time()
    run_training(args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        f"Total execution time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}."
    )
