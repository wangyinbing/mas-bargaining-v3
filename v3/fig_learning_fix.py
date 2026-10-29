"""Regenerate fig2 (learning curves) with EXECUTED efficiency.

Free dialogue executes nothing in any episode, so its curve is flat at zero by
construction; the parser-implied allocation it produces is not an executed
outcome and must not be plotted as if it were.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"C:\Users\13121\Desktop\SPB项目"
OUT = BASE + r"\output_v3\main\analysis\fig2_learning.png"

ORDER = ["free_text", "alternating_offers", "sealed_bid", "sealed_compromise", "sealed_compromise_verify"]
LABELS = {"free_text": "free dialogue", "alternating_offers": "alternating offers", "sealed_bid": "sealed bid",
          "sealed_compromise": "SPB", "sealed_compromise_verify": "SPB+"}
COLORS = {"free_text": "#888888", "alternating_offers": "#1f77b4", "sealed_bid": "#ff7f0e",
          "sealed_compromise": "#2ca02c", "sealed_compromise_verify": "#d62728"}

d = pd.read_csv(BASE + r"\output_v3\main\episodes.csv").drop_duplicates(subset=["model", "protocol", "trial", "episode"])
d["eff_exec"] = np.where(d.protocol == "free_text", 0.0, d.welfare / 10.0)

models = ["flash", "pro"]
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
for ax, model in zip(axes, models):
    for p in ORDER:
        s = d[(d.model == model) & (d.protocol == p)]
        if len(s) == 0:
            continue
        m = s.groupby("episode").eff_exec.mean()
        ax.plot(m.index, m.values, marker="o", ms=4, label=LABELS[p], color=COLORS[p],
                ls="--" if p == "free_text" else "-")
    ax.set_title(model, fontsize=10)
    ax.set_xlabel("episode within trial")
    ax.set_ylabel("executed efficiency")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT, dpi=300)
print("wrote", OUT)
