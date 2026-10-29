"""Summarise the 30-trial cross-vendor replication (xvendor30)."""
import sys, itertools
import pandas as pd, numpy as np
from scipy import stats

PATH = r"C:\Users\13121\Desktop\SPB项目\output_v3\xvendor30\episodes.csv"
d = pd.read_csv(PATH).drop_duplicates(subset=["model", "protocol", "trial", "episode"])
for c in ["deal", "rounds", "welfare", "tokens_in", "tokens_out", "demand_a"]:
    if c in d:
        d[c] = pd.to_numeric(d[c], errors="coerce")
d["eff"] = d["welfare"] / 10.0
d["tok"] = d["tokens_in"].fillna(0) + d["tokens_out"].fillna(0)

print(f"episodes: {len(d)}  errors: {d.get('error', pd.Series()).notna().sum() if 'error' in d else 0}")
rows = []
for (m, p), g in d.groupby(["model", "protocol"]):
    rows.append(dict(model=m, protocol=p, n=len(g), deal=g.deal.mean(), eff=g.eff.mean(),
                     rounds=g[g.deal == 1].rounds.mean(), tok=g.tok.mean()))
s = pd.DataFrame(rows).sort_values(["model", "protocol"])
print(s.round(3).to_string(index=False))

print("\npairwise (Mann-Whitney, Holm within model):")
for m, g in d.groupby("model"):
    ps = []
    for a, b in itertools.combinations(sorted(g.protocol.unique()), 2):
        u, p = stats.mannwhitneyu(g[g.protocol == a].eff, g[g.protocol == b].eff, alternative="two-sided")
        ps.append((f"{a} vs {b}", p))
    ps.sort(key=lambda x: x[1])
    k = len(ps)
    for i, (nm, p) in enumerate(ps):
        print(f"  {m:9s} {nm:52s} p={p:.2e}  holm={min(1, p*(k-i)):.2e}")

print("\nlatex rows:")
for _, r in s.iterrows():
    print(f"{r.model} & {r.protocol} & {int(r.n)} & {r.deal:.3f} & {r.eff:.3f} & {r.rounds:.1f} & {int(r.tok):,} \\\\")
