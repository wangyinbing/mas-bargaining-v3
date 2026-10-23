"""Cross-vendor replication summary: Qwen3-32B and GLM-4.5-Air.

Usage: python v3/summarize_xvendor.py --dir output_v3/xvendor
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

R = 10
LAB = {
    "sealed_bid": "Sealed bid",
    "sealed_compromise": "SPB (ours)",
    "sealed_compromise_verify": "SPB+ (ours)",
}
ORDER = ["sealed_bid", "sealed_compromise", "sealed_compromise_verify"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="output_v3/xvendor")
    a = ap.parse_args()
    d = Path(a.dir)
    df = pd.read_csv(d / "episodes.csv")
    df = df[df.status == "ok"].copy()
    for c in ["deal", "welfare", "rounds", "tokens_in", "tokens_out",
              "fallback_offered", "confirm_failed", "demand_a_mean"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df.protocol.isin(ORDER)]
    df["eff"] = df.welfare / R

    rows = []
    for model in sorted(df.model.unique()):
        for p in ORDER:
            s = df[(df.model == model) & (df.protocol == p)]
            if len(s) == 0:
                continue
            rows.append(dict(model=model, protocol=p, n=len(s),
                             deal=s.deal.mean(), eff=s.eff.mean(),
                             eff_sd=s.eff.std(), rounds=s.rounds.mean(),
                             tokens=(s.tokens_in + s.tokens_out).mean(),
                             fallback=(s.fallback_offered > 0).mean(),
                             confirm=(s.confirm_failed > 0).mean(),
                             demand=s.demand_a_mean.mean()))
    tab = pd.DataFrame(rows)
    print(tab.round(3).to_string(index=False))

    print("\nTests")
    for model in sorted(df.model.unique()):
        s = df[df.model == model]
        groups = [s[s.protocol == p].eff.values for p in ORDER if (s.protocol == p).any()]
        if len(groups) >= 3:
            H, p = stats.kruskal(*groups)
            print(f"{model}: Kruskal-Wallis H={H:.2f}, p={p:.2e}")
        A = s[s.protocol == "sealed_compromise"].eff.values
        B = s[s.protocol == "sealed_bid"].eff.values
        if len(A) > 1 and len(B) > 1:
            U, p = stats.mannwhitneyu(A, B, alternative="two-sided")
            print(f"{model}: SPB vs sealed  {A.mean():.3f} vs {B.mean():.3f}, U={U:.0f}, p={p:.3f}")
        A = s[s.protocol == "sealed_compromise_verify"].eff.values
        B = s[s.protocol == "sealed_compromise"].eff.values
        if len(A) > 1 and len(B) > 1:
            U, p = stats.mannwhitneyu(A, B, alternative="two-sided")
            print(f"{model}: SPB+ vs SPB   {A.mean():.3f} vs {B.mean():.3f}, U={U:.0f}, p={p:.3f}")

    tex = ["\\begin{table}[t]\\centering",
           "\\caption{Cross-vendor replication (Qwen3-32B, GLM-4.5-Air; $n$ episodes per cell).}",
           "\\label{tab:replication}",
           "\\begin{tabular}{llccccc}\\toprule",
           "Model & Protocol & $n$ & Deal rate & Efficiency & Rounds & Tokens \\\\ \\midrule"]
    last = None
    for _, r in tab.iterrows():
        if last is not None and r.model != last:
            tex.append("\\midrule")
        last = r.model
        tex.append(f"{r.model} & {LAB[r.protocol]} & {int(r.n)} & {r.deal:.2f} & "
                   f"{r.eff:.3f} & {r.rounds:.1f} & {r.tokens:,.0f} \\\\")
    tex.append("\\bottomrule\\end{tabular}\\end{table}")
    (d / "replication.tex").write_text("\n".join(tex), encoding="utf-8")
    print(f"\n[written] {d/'replication.tex'}")


if __name__ == "__main__":
    main()
