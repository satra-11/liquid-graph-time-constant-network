# マルチエージェントFlockingシミュレーションタスク実装計画

LTCNおよびLGTCNモデルを用いたマルチエージェントFlockingシミュレーションタスクを実装する。

**参照論文**: [arXiv:2404.13982 - Liquid-Graph Time-Constant Network for Multi-Agent Systems Control](https://arxiv.org/abs/2404.13982)

---

## 論文の実験条件（再現対象）

### シミュレーション設定
| パラメータ | 値 |
|------------|-----|
| エージェント数 N | [4, 6, 10, 12, 15] (ランダム選択) |
| 軌道数 | 60本 |
| 軌道長 | 2.5秒 |
| サンプリング時間 T | 0.05秒 |
| 初期位置間距離 | 0.6〜1.0m |
| 初期速度 | [-2, 2] m/s |
| 入力飽和 | ±5 m/s² |
| 通信範囲 R | 4m |
| 衝突回避距離 R_CA | 1m |
| ターゲット範囲 | 20m四方 |

### タスク概要
- **Leader-Follower Flocking**: 1体のリーダーがターゲットへ誘導
- **入力**: 位置r(t) ∈ R^(N×2), 速度v(t) ∈ R^(N×2)
- **出力**: 加速度u(t) ∈ R^(N×2)
- **模倣学習**: Expert Controller (Olfati-Saber式) を教師信号

### ニューラルネットワーク設定
| パラメータ | 値 |
|------------|-----|
| 入力特徴量 | 10次元 (w_i) |
| 隠れ状態次元 F | 50 |
| フィルタ長 K | 2 |
| FC層 | 128ノード×2層 (入力側) |
| 出力層 | 128ノード×2層 → 2次元 |
| サポート行列 | 正規化ラプラシアン |

### 訓練設定
| パラメータ | 値 |
|------------|-----|
| エポック数 | 120 |
| DAGGERインターバル | 20エポック |
| オプティマイザ | Adam |
| 学習率 | 1e-3 |
| β1, β2 | 0.9, 0.999 |
| 損失関数 | MSE |
| 分割比率 | Train:Val:Test = 70:10:20 |

### 評価設定
- **スケーラビリティ評価**: N = [4, 10, 25, 50]
- **通信範囲評価**: R = [2, 4, 6, 8] m
- **指標**:
  - Leader Error: ef/es (最終/初期距離比)
  - Flocking Error: 速度整列誤差

---

## Proposed Changes

### Component 1: Flocking Environment

#### [NEW] [environment.py](file:///Users/satron/Desktop/lgtcn/src/flocking/environment.py)

論文と同じFlocking環境を実装:
```python
class FlockingEnvironment:
    def __init__(
        self,
        num_agents: int = 10,
        dt: float = 0.05,
        comm_range: float = 4.0,
        collision_range: float = 1.0,
        max_accel: float = 5.0,
    ):
        ...

    def reset(self) -> tuple[torch.Tensor, torch.Tensor]:
        # エージェントを0.6〜1.0m間隔で配置
        # 速度を[-2, 2]m/sで初期化
        # リーダーをランダム選択、ターゲットを設定
        ...

    def step(self, actions: torch.Tensor) -> tuple[...]:
        # Double integrator dynamics
        # v(t+1) = v(t) + u(t) * dt
        # r(t+1) = r(t) + v(t) * dt
        ...

    def compute_expert_action(self) -> torch.Tensor:
        # Olfati-Saber式のExpert Controller
        ...

    def compute_adjacency(self) -> torch.Tensor:
        # 通信範囲内のエージェント間でエッジを生成
        ...
```

---

### Component 2: Data Generation

#### [NEW] [data.py](file:///Users/satron/Desktop/lgtcn/src/flocking/data.py)

```python
class FlockingDataset(Dataset):
    def __init__(
        self,
        num_trajectories: int = 60,
        trajectory_length: float = 2.5,
        dt: float = 0.05,
        agent_counts: list[int] = [4, 6, 10, 12, 15],
    ):
        ...

    def _collect_trajectory(self, env: FlockingEnvironment) -> dict:
        # Expert Controllerで軌道を収集
        ...

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # (observations, adjacency, expert_actions)
        ...
```

---

### Component 3: Models

#### [NEW] [models.py](file:///Users/satron/Desktop/lgtcn/src/flocking/models.py)

```python
class FlockingLGTCN(nn.Module):
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 50,
        output_dim: int = 2,
        K: int = 2,
        fc_dim: int = 128,
    ):
        # FC(128) → FC(128) → CfGCNLayer(50, K=2) → FC(128) → FC(128) → out(2)
        ...

class FlockingLTCN(nn.Module):
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 50,
        output_dim: int = 2,
        fc_dim: int = 128,
    ):
        # FC(128) → FC(128) → LTCNLayer(50) → FC(128) → FC(128) → out(2)
        ...
```

---

### Component 4: Training Pipeline

#### [NEW] [run.py](file:///Users/satron/Desktop/lgtcn/src/flocking/run.py)
#### [NEW] [engine.py](file:///Users/satron/Desktop/lgtcn/src/flocking/engine.py)

- **DAGGERアルゴリズム実装**: 20エポックごとにExpert Controllerで再収集
- **安定性正則化**: Softplus(β=10)による収縮率制約
- MLflow連携

---

### Component 5: Package Init

#### [NEW] [\_\_init\_\_.py](file:///Users/satron/Desktop/lgtcn/src/flocking/__init__.py)

---

## Verification Plan

### Automated Tests
```bash
python -m pytest test/test_flocking.py -v
```

**テスト内容:**
1. `FlockingEnvironment`の初期化・ステップ
2. Expert Controllerの衝突回避動作
3. 隣接行列の正しい計算
4. モデルの順伝播・勾配フロー

### Manual Verification
```bash
# 短時間訓練
python -m src.flocking.run --epochs 10 --num-trajectories 10

# スケーラビリティ評価
python -m src.flocking.evaluate --agent-counts 4,10,25,50
```

---

## 確認事項

> [!NOTE]
> 論文の条件に合わせて実装を進めてよいですか？
