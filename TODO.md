# TODO

- [x] **(30 min)** Read the abstract and introduction, then summarize the core problem and contributions in Obsidian.
- [ ] **(30 min)** Skim all figures and tables, documenting key takeaways from the visual summaries in Obsidian.
- [ ] **(45 min)** Read the methodology section, taking detailed notes on the LGTCN model's technical details in Obsidian.
- [ ] **(30 min)** Read the results and conclusion, summarizing the experimental setup and outcomes in Obsidian.
- [ ] **(30 min)** Review the project structure (`src`, `scripts`, `notebooks`), documenting the role of each part in Obsidian.
- [ ] **(45 min)** Analyze `scripts/train.py`, annotating and documenting the main training loop and data flow in Obsidian.
- [ ] **(60 min)** Trace the model construction via `src/lgtcn/models/lgtcn_controller.py` and its layers, creating a summary or diagram in Obsidian.
- [ ] **(45 min)** Examine `src/lgtcn/utils/graph.py`, documenting the graph data loading and processing steps in Obsidian.
- [ ] **(30 min)** Review `scripts/eval_grid.py` and `notebooks/plot_erros.py`, noting the evaluation and visualization logic in Obsidian.
- [ ] **(30 min)** Research and select a suitable baseline model for comparison (e.g., GCN, ChebNet, or a standard TCN).
- [ ] **(15 min)** Create a new model file (e.g., `src/lgtcn/models/baseline_model.py`).
- [ ] **(45 min)** Implement the data loading and preprocessing required for the baseline model, if different from the existing setup.
- [ ] **(60 min)** Implement the baseline model architecture within the new file.
- [ ] **(45 min)** Integrate the new model into `scripts/train.py` and `scripts/eval_grid.py` so it can be trained and evaluated.
- [ ] **(30 min)** Run a training and evaluation cycle for the new baseline model to ensure it works.
