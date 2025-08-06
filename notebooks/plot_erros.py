# notebooks/plot_errors.py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid")

df = pd.read_json("results/metrics.json")

fig, axes = plt.subplots(1, 4, figsize=(14,3))

# ① Leader-Error vs N
sns.lineplot(ax=axes[0], data=df, x="N", y="leader_err",
             hue="model", errorbar="se", style="model", marker="o")
axes[0].set(yscale="log", title="Leader Error")

# ② Flocking-Error vs N
sns.lineplot(ax=axes[1], data=df, x="N", y="flock_err",
             hue="model", errorbar="se", style="model", marker="o")
axes[1].set(yscale="log", title="Flocking Error")

# ③ Leader-Error vs range
sns.lineplot(ax=axes[2], data=df, x="range", y="leader_err",
             hue="model", errorbar="se", style="model", marker="o")
axes[2].set(yscale="log", title="Leader Error")

# ④ Flocking-Error vs range
sns.lineplot(ax=axes[3], data=df, x="range", y="flock_err",
             hue="model", errorbar="se", style="model", marker="o")
axes[3].set(yscale="log", title="Flocking Error")

plt.tight_layout()
plt.savefig("results/error_grid.png", dpi=200)
