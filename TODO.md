# TODO

## 1) Paper Reading

* [x] Read section I (INTRODUCTION) — note new concepts in Obsidian
* [x] Read section II (PRELIMINARIES) — note new concepts in Obsidian
* [x] Read section III (LIQUID-GRAPH TIME CONSTANT NETWORK) — note new concepts in Obsidian
* [x] Read section IV (CLOSED-FORM APPROXIMATION) — note new concepts in Obsidian
* [x] Read section V (VALIDATION EXAMPLE) — note new concepts in Obsidian
* [x] Read section VI (CONCLUSION) — note new concepts in Obsidian

## 2) Code Review / Refactor

* [x] Review `src/lgtcn/layers` and list refactor opportunities
* [x] Review `src/lgtcn/models` and list refactor opportunities
* [ ] Review `src/lgtcn/utils` and list refactor opportunities

  * [ ] Centralize random seed/repro utilities
  * [ ] Unify logging/timing/metrics helpers
  * [ ] Add type hints and docstrings
* [ ] Review & refactor `scripts/train.py`

  * [ ] Clean up `argparse` (required/optional/defaults)
  * [ ] Config file support (YAML/JSON)
  * [ ] Early stopping & checkpointing
* [ ] Review & refactor `scripts/rollout.py` (consistent I/O, batching)
* [ ] Review & refactor `scripts/eval_grid.py` (parallelism, CSV outputs)

## 3) Data & Preprocessing

* [ ] Choose dataset (CIFAR-10/100 or Tiny-ImageNet)
* [ ] Implement image → patch sequence (e.g., 16×16)
* [ ] Implement missingness mask generator (fixed seeds, saved masks)

  * [ ] Random pixel drop
  * [ ] Block occlusion (Cutout; multiple sizes)
  * [ ] Stripe / row / column removal
  * [ ] Channel missingness (drop one RGB channel)
  * [ ] Noise-as-missingness (salt & pepper)
* [ ] Batch pipelines for p = 0, 10, …, 70%

## 4) Model Implementations

* [ ] Baselines

  * [ ] ResNet-18 (lightweight)
  * [ ] ViT-Tiny (parameter-matched if possible)
* [ ] LTCN (LTC)

  * [ ] Integrate an existing/official implementation & run a minimal train
  * [ ] Patch-seq → embedding → LTC → classification head
* [ ] LGTCN

  * [ ] Local TCN branch + Global branch (dilated TCN / self-attn)
  * [ ] Switchable fusion (sum / concat / attention)
* [ ] Shared training loop (AdamW, cosine LR, progress bar, logging)

## 5) Evaluation Design (“Stability” Metrics)

* [ ] Accuracy\@p (accuracy per missingness rate)
* [ ] AUC-R (area under accuracy–p curve)
* [ ] Output consistency: KL divergence / agreement between clean vs. masked
* [ ] Calibration: ECE
* [ ] (Optional) Input sensitivity / gradient norms

## 6) Experimental Plan

* [ ] Train #1: train on clean data → test under missingness
* [ ] Train #2: mix missingness during training → test under missingness
* [ ] Sweep over missingness types and rates
* [ ] Repro with 3–5 random seeds
* [ ] Auto-save robustness curves (PNG/CSV)

## 7) Ablations

* [ ] LGTCN: local-only / global-only / fusion variants
* [ ] LTC: hidden size / gating / integration step width
* [ ] Patch size & patch order (raster / zigzag, etc.)
* [ ] With vs. without missingness during training

## 8) Visualizations

* [ ] Accuracy vs. missingness curves (one per model)
* [ ] Grad-CAM / attention maps to compare focus shift under occlusion
* [ ] Side-by-side examples with predictions & confidences

## 9) Test Infrastructure

* [ ] Set up pytest for PyTorch
* [ ] **Write unit tests for all layers** (IO shapes, grads, numerical stability)
* [ ] **Write unit tests for all models** (forward pass, save/load, determinism)

## 10) Additional Model Comparisons (Optional)

* [ ] **Experiment with GGNN** (same preprocessing/metrics)
* [ ] **Experiment with GraphODE** (same setup)

## 11) Reproducibility Pack

* [ ] `environment.yml` / `requirements.txt`
* [ ] Save seeds, masks, and config files (YAML)
* [ ] Save experiment logs (CSV/W\&B) and trained weights
* [ ] `README.md` with full reproduction steps

## 12) Writing

* [ ] Draft chapters (Intro / Related Work / Method / Experiments / Discussion / Conclusion)
* [ ] Create figures (curves, maps, tables)
* [ ] **Send draft to professor and address feedback**
* [ ] Build references (BibTeX)
* [ ] Final compile & formatting check

## 13) Environment & Ops

* [ ] **Ask senpai if I can use the school’s server** (GPU, job scheduling)
* [ ] Directory conventions for data/outputs (`data/`, `outputs/`)
* [ ] Simple experiment queue (sheet or task runner)

## 14) Script Checklist (Exit Criteria)

* [ ] `train.py`: CLI, configs, checkpoint/resume, logging
* [ ] `rollout.py`: load model → inference → CSV/image dumps
* [ ] `eval_grid.py`: grid search → CSV aggregation → best-config export

### Obsidian (Research Notes)
* [ ] Research & summarize: multi-agent systems (MAS)
* [ ] Research & summarize: contraction analysis
* [x] Research & summarize: ISS (Input-to-State Stability)
* [x] Research & summarize: incremental ISS (δISS)
* [x] Research & summarize: GNN
