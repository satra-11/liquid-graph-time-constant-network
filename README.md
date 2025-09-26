# Liquid-Graph Time-Constant Network (LGTC network)
This repository provides a reference implementation of the Liquid-Graph Time-Constant (LGTC) networks introduced in the paper “[Liquid-Graph Time-Constant Network for Multi-Agent Systems Control](https://arxiv.org/pdf/2404.13982)” .

# Getting started
Set up the project with **either** of the following methods, depending on the tools you prefer.
### 1. pip + venv
```bash
git clone https://github.com/satra-11/liquid-graph-time-constant-network .
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
```
### 2. Rye
```bash
git clone https://github.com/satra-11/liquid-graph-time-constant-network .
rye sync
```
### Commands
#### 1. Training
Start training by executing a command below. The results are generated at /driving_results folder.
```bash
python3 ./src/scripts/train_driving.py
```

# Model Structures

The `scripts/train_driving.py` script follows the workflow below to train and evaluate the LGTCN and LTCN models.
### LTCN Model
```mermaid
flowchart TD
  A["frames <br> (B×T×H×W×C)"]
    --> B["CNN<br>→(B·T)×128×8×8"]
    --> C["reshape/permute <br>→ B×T×64×128"]
    --> D["encoder <br>(128→H) → B×T×64×H"]
    --> F["LTCNLayer:<br>  (h_t, x_t)→ h_{t+1}"]
    --> G["decoder<br>(h_{t+1}) → B×2"]

  G --> H["time-stacked controls: B×T×2"]
  F --> I["final_hidden: B×N"]

  %% === Input を紫枠 ===
  subgraph IN["Inputs"]
    direction LR
    A
  end

  %% === Outputs をサブグラフで紫枠 ===
  subgraph OUT["Outputs"]
    direction LR
    H
    I
  end

  %% サブグラフの枠線を紫に
  style IN stroke:#8b5cf6,stroke-width:3px;
  style OUT stroke:#8b5cf6,stroke-width:3px;
```

