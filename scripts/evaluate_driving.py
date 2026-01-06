#!/usr/bin/env python
"""
映像データによる自律走行タスクでの評価スクリプト
コマンドライン引数でモデルを指定して一つずつ評価する
"""

import time
import argparse
from src.driving.evaluate import run_single_model_evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a driving controller")
    # Data params
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to data directory"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./driving_results",
        help="Directory to save results",
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lgtcn", "ltcn", "node", "ngode"],
        help="Model type to evaluate (lgtcn, ltcn, node, ngode)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    # Model params (defaults matched to train_driving.py)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--num-layers-ltcn", type=int, default=4)
    parser.add_argument("--num-layers-cfgcn", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)

    # System params
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    start_time = time.time()
    run_single_model_evaluation(args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        f"Total execution time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}."
    )
