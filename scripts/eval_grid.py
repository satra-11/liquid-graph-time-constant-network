# scripts/eval_grid.py
import itertools
import json
from pathlib import Path
from rollout import run_one_episode

Ns     = [5, 10, 25, 50]
ranges = [2, 4, 8, 12]
seeds  = range(5)

grid = list(itertools.product(Ns, ranges, seeds))
models = {
    "Expert": expert_policy,
    "GGNN"  : load_model("checkpoints/ggnn.pt"),
    "LGTC"  : load_model("checkpoints/lgtc.pt"),
    "CfGC"  : load_model("checkpoints/cfgc.pt"),
    "GraphODE": load_model("checkpoints/graphode.pt"),
}

records = []
for n, rng, seed in grid:
    for name, policy in models.items():
        le, fe = run_one_episode("flocking", n, rng, policy, seed)
        records.append(dict(model=name, N=n, range=rng,
                            leader_err=le.item(), flock_err=fe.item()))
Path("results").mkdir(exist_ok=True)
json.dump(records, open("results/metrics.json", "w"))
