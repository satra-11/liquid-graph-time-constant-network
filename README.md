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
# Commands
### 1. Training
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
```mermaid
flowchart TD
  %% ===== Inputs =====
  subgraph IN["Inputs"]
    direction LR
    A["frames (B×T×H×W×C)"]
    Adj["adjacency (B×T×N×N) / None"]
    H0["hidden_state (B×N×H) / None"]
  end

  %% ===== Pipeline =====
  A --> B["CNN → (B·T)×128×8×8"]
  B --> C["reshape/permute → B×T×64×128"]
  C --> D["node_encoder (128→H) → B×T×64×H"]
  Adj --> H["LGTCN(x_t, u_t, S_powers) → x_{t+1}"]
  H0 --"x_t"--> H
  D --"u_t"--> H

  %% ===== Decode =====
  H --> P["mean over nodes → (B×H)"]
  P --> Q["control_decoder → (B×2)"]

  %% ===== Outputs =====
  subgraph OUT["Outputs"]
    direction LR
    O1["controls (B×T×2)"]
    O2["final_hidden (B×N×H)"]
  end

  Q --> O1
  H --> O2

  %% ===== Styling =====
  style IN stroke:#8b5cf6,stroke-width:3px,fill:#ffffff,color:#111;
  style OUT stroke:#8b5cf6,stroke-width:3px,fill:#ffffff,color:#111;

```

