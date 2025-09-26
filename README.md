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
python3 ./scripts/train_driving.py
```

## Training Workflow

The `scripts/train_driving.py` script follows the workflow below to train and evaluate the LGTCN and LTCN models.

```mermaid
graph TD
    A[Start] --> B[1. Setup<br>Parse Args, Set Seed, Create Dirs];
    B --> C[2. Create Dataset<br>create_dataset()];
    C --> D[3. Split Data & Create DataLoaders<br>Train/Val/Test Sets];
    D --> E{4. Create Models};
    E --> F_LGTCN[LGTCN Model];
    E --> F_LTCN[LTCN Model];

    subgraph Training Phase
        F_LGTCN --> G_LGTCN[5a. Train LGTCN<br>train_model()];
        F_LTCN --> G_LTCN[5b. Train LTCN<br>train_model()];
    end

    G_LGTCN & G_LTCN --> H[6. Plot Training Curves<br>Save to training_curves.png];

    subgraph Evaluation Phase
        H --> I[7. Evaluate Models on Test Set<br>evaluate_networks()];
    end
    
    I --> J[8. Save All Results];
    J --> K[Models (.pth)];
    J --> L[Training Info (.json)];
    J --> M[Comparison Results (.json)];
    J --> N[Comparison Plots (.png)];

    J --> O[9. Print Summary to Console];
    O --> P[End];
```

#### 2. Testing
Unit tests are available by following command
```bash
python -m pytest ./test
```
