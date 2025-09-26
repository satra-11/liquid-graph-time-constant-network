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

## Training Workflow

The `scripts/train_driving.py` script follows the workflow below to train and evaluate the LGTCN and LTCN models.

```mermaid
graph TD
    A[Start] --> C["Create Dataset<br>create_dataset()"];
    C --> D["Split Data & Create DataLoaders<br>Train/Val/Test Sets"];
    D --> E{Create Models};
    E --> F_LGTCN[LGTCN Model];
    E --> F_LTCN[LTCN Model];

    subgraph Training Phase
        F_LGTCN --> G_LGTCN["Train LGTCN<br>train_model()"];
        F_LTCN --> G_LTCN["Train LTCN<br>train_model()"];
    end

    G_LGTCN & G_LTCN --> H["Plot Training Curves<br>Save to training_curves.png"];

    subgraph Evaluation Phase
        H --> I["Evaluate Models on Test Set<br>evaluate_networks()"];
    end
    
    I --> J[Save All Results];
    J --> K["Models (.pth)"];
    J --> L["Training Info (.json)"];
    J --> M["Comparison Results (.json)"];
    J --> N["Comparison Plots (.png)"];

    J --> O[Print Summary to Console];
    O --> P[End];

    %% Style Definitions (Purple Theme)
    classDef io fill:#F5F5F5,color:#36454F,stroke:#B0B0B0,stroke-width:2px;
    classDef models fill:#E6E6FA,color:#36454F,stroke:#6A0DAD,stroke-width:2px;
    classDef phase fill:#6A0DAD,color:#FFFFFF,stroke:#483D8B,stroke-width:2px,font-weight:bold;
    classDef startend fill:#483D8B,color:#FFFFFF,stroke:#36454F,stroke-width:2px;

    class A,P startend;
    class B,C,D,J,K,L,M,N,O io;
    class E,F_LGTCN,F_LTCN models;
    class G_LGTCN,H,I phase;
```

#### 2. Testing
Unit tests are available by following command
```bash
python -m pytest ./test
```
```

