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
```mermaid
flowchart TD
  %% ===== 左：Title =====
  subgraph L["LTCN"]
    direction TB
    LA["frames <br> (B×T×H×W×C)"]
      --> LB["CNN<br>→(B·T)×128×8×8"]
      --> LC["reshape/permute <br>→ B×T×64×128"]
      --> LD["encoder <br>(128→H) → B×T×64×H"]
      --> LF["LTCNLayer:<br>(h_t, x_t)→ h_{t+1}"]
      --> LG["decoder<br>(h_{t+1}) → B×2"]
    LG --> LH["time-stacked controls: B×T×2"]
    LF --> LI["final_hidden: B×N"]
    subgraph LIN["Inputs"]
      direction LR
      LA
    end
    subgraph LOUT["Outputs"]
      direction LR
      LH
      LI
    end
  end

  %% ===== 右：Title 2 =====
  subgraph R["LGTCN"]
    direction TB
    RIN["Inputs"]:::io
    RA["frames (B×T×H×W×C)"]:::io
    RADJ["adjacency (B×T×N×N)/None"]:::io
    RH0["hidden_state (B×N×H)/None"]:::io

    RA --> RB["CNN → (B·T)×128×8×8"]
    RB --> RC["reshape/permute → B×T×64×128"]
    RC --> RD["node_encoder (128→H) → B×T×64×H"]
    RH0 -- "x_t" --> RH["LGTCN(x_t, u_t, S_powers) → x_{t+1}"]
    RD -- "u_t" --> RH
    RADJ -. "S_powers" .-> RH
    RH --> RQ["control_decoder → (B×2)"]
    subgraph RIN["Inputs"]
      direction LR
      RA
      RADJ
      RH0
    end

    subgraph ROUT["Outputs"]
      direction LR
      RO1["controls (B×T×2)"]
      RO2["final_hidden (B×N×H)"]
    end
    RQ --> RO1
    RH --> RO2
  end

  %% スタイル
  classDef io stroke:#8b5cf6,stroke-width:3px,color:#fff;
  class LIN,LOUT,RIN,ROUT io;
```

