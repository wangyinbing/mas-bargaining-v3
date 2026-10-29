"""Analyse v3 experiment: protocol comparison + SPB ablation.

Usage: python v3/analysis_v3.py --dir output_v3/main
Outputs: <dir>/analysis/summary.md, *.png, tables.tex
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R_DEFAULT = 10
LABELS = {
    "free_text": "Free dialogue",
    "alternating_offers": "Alternating offers",
    "sealed_bid": "Sealed bid",
    "mediated": "Mediation",
    "sealed_compromise": "SPB (ours)",
    "sealed_compromise_verify": "SPB+ (ours)",
}
ORDER = ["free_text", "alternating_offers", "sealed_bid",
         "sealed_compromise", "sealed_compromise_verify"]
COLORS = {
    "free_text": "#9aa0a6",
    "alternating_offers": "#8ab4f8",
    "sealed_bid": "#f28b82",
    "mediated": "#fdd663",
    "sealed_compromise": "#81c995",
    "sealed_compromise_verify": "#34a853",
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def load(d):
    df = pd.read_csv(Path(d) / "episodes.csv")
    df = df[df["status"] == "ok"].copy()
    num = ["R", "T", "K", "trial", "episode", "deal", "deal_strict", "strict_round",
           "agree_elicited", "agree_native", "agree_legacy", "legacy_round",
           "perceived_mismatch", "split_a", "split_b", "allocated", "welfare",
           "pie_final", "rounds", "one_sided_claim", "tokens_in", "tokens_out",
           "calls", "wall_s", "bid_sum_first", "bid_sum_mean", "lowball_any",
           "lowball_rounds", "demand_a_mean", "demand_b_mean",
           "fallback_offered", "fallback_accepted", "confirm_failed"]
    for c in num:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Adjudicated free-dialogue outcomes (validated on 30 blind hand labels).
    jp = Path(d) / "audit" / "judge.csv"
    if jp.exists():
        j = pd.read_csv(jp)[["model", "trial", "episode", "deal", "round", "split_a", "split_b"]]
        j = j.rename(columns={"deal": "j_deal", "round": "j_round",
                              "split_a": "j_a", "split_b": "j_b"})
        df = df.merge(j, on=["model", "trial", "episode"], how="left")
        ft = df.protocol == "free_text"
        df.loc[ft, "deal"] = df.loc[ft, "j_deal"].fillna(0)
        df.loc[ft, "rounds"] = df.loc[ft, "j_round"].fillna(0)
        df.loc[ft, "split_a"] = df.loc[ft, "j_a"]
        df.loc[ft, "split_b"] = df.loc[ft, "j_b"]
        df["pie_final"] = df["rounds"].apply(lambda r: max(1, int(round(R_DEFAULT * 0.9 ** (r - 1)))) if r and r > 0 else 1)
        df["welfare"] = np.where(df["deal"] == 1, df["pie_final"], 0)
        df["legacy_eff"] = np.where(
            (df.protocol == "free_text") & (df.agree_legacy == 1),
            df["legacy_round"].apply(lambda r: max(1, int(round(R_DEFAULT * 0.9 ** (r - 1)))) if r and r > 0 else 0) / R_DEFAULT,
            0.0)

    # Executed welfare: free dialogue has no execution rule, so its executed
    # efficiency is zero by construction (agreement is reported separately).
    df.loc[df.protocol == "free_text", "welfare"] = 0.0
    df["eff"] = df["welfare"] / R_DEFAULT
    df["tokens"] = df["tokens_in"] + df["tokens_out"]
    df["proto"] = pd.Categorical(df["protocol"], ORDER, ordered=True)
    ft_mask = df.protocol == "free_text"
    df["rounds_eff"] = df["rounds"]
    df.loc[ft_mask, "rounds_eff"] = np.where(df.loc[ft_mask, "deal"] == 1,
                                             df.loc[ft_mask, "rounds"], 10)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="output_v3/main")
    args = ap.parse_args()
    d = Path(args.dir)
    out = d / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    df = load(d)
    lines = []
    add = lines.append

    add(f"# v3 analysis — {len(df)} episodes\n")
    add(f"models: {sorted(df.model.unique())}; R={R_DEFAULT}; T={df['T'].max()}; K={df['K'].max()}\n")

    # ── main table ─────────────────────────────────────────────
    rows = []
    for model in sorted(df.model.unique()):
        for p in ORDER:
            s = df[(df.model == model) & (df.protocol == p)]
            if len(s) == 0:
                continue
            k, n = int(s.deal.sum()), len(s)
            lo, hi = wilson(k, n)
            rows.append(dict(
                model=model, protocol=p, label=LABELS[p], n=n,
                deal_rate=k / n, deal_lo=lo, deal_hi=hi,
                efficiency=s.eff.mean(), eff_sd=s.eff.std(),
                rounds=s.rounds_eff.mean(),
                tokens=s.tokens.mean(),
                fallback=(s.fallback_offered > 0).mean() if "fallback_offered" in s else np.nan,
                confirm_fail=(s.confirm_failed > 0).mean() if "confirm_failed" in s else np.nan,
                demand_a=s.demand_a_mean.mean() if "demand_a_mean" in s else np.nan,
            ))
    tab = pd.DataFrame(rows)
    add("## Main results\n")
    add(tab.to_markdown(index=False, floatfmt=".3f"))
    add("")

    # ── tests ──────────────────────────────────────────────────
    add("## Tests\n")
    for model in sorted(df.model.unique()):
        s = df[df.model == model]
        protos = [p for p in ORDER if (s.protocol == p).any()]
        ct = pd.crosstab(s.protocol, s.deal)
        if ct.shape[1] == 2 and len(protos) >= 3:
            chi2, pv, dof, _ = stats.chi2_contingency(ct)
            add(f"- **{model}** deal rate across protocols: chi2({dof})={chi2:.2f}, p={pv:.2e}")
        groups = [s[s.protocol == p].eff.values for p in protos]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) >= 3:
            H, pv = stats.kruskal(*groups)
            add(f"- **{model}** efficiency across protocols: Kruskal-Wallis H={H:.2f}, p={pv:.2e}")
        # ours vs sealed (pairwise)
        for a_p, b_p in [("sealed_compromise", "sealed_bid"),
                         ("sealed_compromise_verify", "sealed_bid"),
                         ("sealed_compromise_verify", "sealed_compromise")]:
            A = s[s.protocol == a_p].eff.values
            B = s[s.protocol == b_p].eff.values
            if len(A) > 1 and len(B) > 1:
                U, pv = stats.mannwhitneyu(A, B, alternative="two-sided")
                add(f"- **{model}** {a_p} vs {b_p}: Mann-Whitney U={U:.0f}, p={pv:.2e}, "
                    f"mean {A.mean():.3f} vs {B.mean():.3f}")
    add("")

    # ── measurement comparison ─────────────────────────────────
    add("## Measurement comparison\n")
    ft = df[df.protocol == "free_text"]
    if len(ft):
        add(f"- free dialogue episodes: {len(ft)}")
        add(f"- adjudicated (validated): deal **{ft.deal.mean():.3f}**, "
            f"mean round {ft.loc[ft.deal == 1, 'rounds'].mean():.2f}, "
            f"efficiency **{ft.eff.mean():.3f}**")
        add(f"- post-hoc self-report: deal {ft.agree_elicited.mean():.3f}")
        add(f"- strict symmetric detector: deal {ft.deal_strict.mean():.3f}, "
            f"mean round {ft.loc[ft.deal_strict == 1, 'strict_round'].mean():.2f}")
        add(f"- legacy parser: deal {ft.agree_legacy.mean():.3f}, "
            f"mean round {ft.loc[ft.agree_legacy == 1, 'legacy_round'].mean():.2f}, "
            f"legacy-implied efficiency **{ft.legacy_eff.mean():.3f}**")
    st = df[df.protocol != "free_text"]
    if len(st):
        add(f"- structured protocols: mechanism deal {st.agree_native.mean():.3f}, "
            f"elicitation {st.agree_elicited.mean():.3f}, "
            f"mismatch {st.perceived_mismatch.mean():.3f}")
    add("")

    # ── learning ───────────────────────────────────────────────
    add("## Learning over repeated episodes (efficiency slope per episode)\n")
    learn_rows = []
    for model in sorted(df.model.unique()):
        for p in ORDER:
            s = df[(df.model == model) & (df.protocol == p)]
            if len(s) < 10:
                continue
            x = s.episode.values.astype(float)
            y = s.eff.values
            sl, ic, r, pv, se = stats.linregress(x, y)
            learn_rows.append(dict(model=model, protocol=p, slope=sl, p=pv, r2=r**2))
    lt = pd.DataFrame(learn_rows)
    add(lt.to_markdown(index=False, floatfmt=".4f"))
    add("")

    # ── figures ────────────────────────────────────────────────
    models = sorted(df.model.unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    width = 0.38
    xs = np.arange(len(ORDER))
    for i, model in enumerate(models):
        s = df[df.model == model]
        dr = [s[s.protocol == p].deal.mean() if (s.protocol == p).any() else np.nan for p in ORDER]
        ef = [s[s.protocol == p].eff.mean() if (s.protocol == p).any() else np.nan for p in ORDER]
        tk = [s[s.protocol == p].tokens.mean() if (s.protocol == p).any() else np.nan for p in ORDER]
        off = (i - (len(models) - 1) / 2) * width
        axes[0].bar(xs + off, dr, width, label=model)
        axes[1].bar(xs + off, ef, width, label=model)
        axes[2].bar(xs + off, tk, width, label=model)
    for ax, title in zip(axes, ["Deal rate", "Allocative efficiency (allocated / R)", "Tokens per episode"]):
        ax.set_xticks(xs)
        ax.set_xticklabels([LABELS[p].replace(" (ours)", "") for p in ORDER], rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig1_main.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 3.8), squeeze=False)
    for i, model in enumerate(models):
        ax = axes[0][i]
        for p in ORDER:
            s = df[(df.model == model) & (df.protocol == p)]
            if len(s) == 0:
                continue
            m = s.groupby("episode").eff.mean()
            ax.plot(m.index, m.values, marker="o", ms=3, label=LABELS[p], color=COLORS[p])
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("episode")
        ax.set_ylabel("efficiency")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig2_learning.png", dpi=300)
    plt.close(fig)

    # demand by protocol and model (bar with bootstrap CI)
    sealed = df[df.protocol.str.startswith("sealed") & df.demand_a_mean.notna()]
    if len(sealed):
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        protos = ["sealed_bid", "sealed_compromise", "sealed_compromise_verify"]
        xs = np.arange(len(protos))
        width = 0.36
        rng = np.random.default_rng(0)
        for i, model in enumerate(models):
            means, los, his = [], [], []
            for p in protos:
                s = sealed[(sealed.model == model) & (sealed.protocol == p)].demand_a_mean.values
                means.append(s.mean())
                bs = [rng.choice(s, len(s), replace=True).mean() for _ in range(1500)]
                los.append(np.percentile(bs, 2.5)); his.append(np.percentile(bs, 97.5))
            pos = xs + (i - (len(models) - 1) / 2) * width
            ax.bar(pos, means, width, label=model)
            ax.errorbar(pos, means, yerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
                        fmt="none", ecolor="k", capsize=3, lw=0.9)
            for x, m in zip(pos, means):
                ax.text(x, m + 0.12, f"{m:.2f}", ha="center", fontsize=7)
        ax.axhline(R_DEFAULT / 2, ls="--", c="k", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([LABELS[p].replace(" (ours)", "") for p in protos], fontsize=8)
        ax.set_ylabel("mean demand of Agent A")
        ax.set_title("Demand inflation (dashed = fair share R/2)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "fig3_demands.png", dpi=300)
        plt.close(fig)

    # ── measurement figure (free dialogue only) ────────────────
    if len(ft):
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6))
        measures = ["adjudicated\n(validated)", "legacy\nparser", "strict\nsymmetric", "self-\nreport"]
        xs = np.arange(len(measures))
        for i, model in enumerate(models):
            s = ft[ft.model == model]
            vals = [s.deal.mean(), s.agree_legacy.mean(), s.deal_strict.mean(),
                    s.agree_elicited.mean()]
            pos = xs + (i - (len(models) - 1) / 2) * 0.36
            bars = axes[0].bar(pos, vals, 0.36, label=model)
            for b, v in zip(bars, vals):
                axes[0].text(b.get_x() + b.get_width() / 2, min(v + 0.03, 1.02), f"{v:.2f}",
                             ha="center", va="bottom", fontsize=7)
        axes[0].set_xticks(xs)
        axes[0].set_xticklabels(measures, fontsize=8)
        axes[0].set_ylim(0, 1.12)
        axes[0].set_ylabel("share of episodes with a deal")
        axes[0].set_title("(a) Deal rate depends on the instrument", fontsize=10)
        axes[0].legend(fontsize=8)
        axes[0].grid(axis="y", alpha=0.3)

        xs2 = np.arange(2)
        for i, model in enumerate(models):
            s = ft[ft.model == model]
            vals = [s.eff.mean(), s.legacy_eff.mean()]
            pos = xs2 + (i - (len(models) - 1) / 2) * 0.36
            bars = axes[1].bar(pos, vals, 0.36, label=model)
            for b, v in zip(bars, vals):
                axes[1].text(b.get_x() + b.get_width() / 2, min(v + 0.02, 1.02), f"{v:.2f}",
                             ha="center", va="bottom", fontsize=7)
        axes[1].set_xticks(xs2)
        axes[1].set_xticklabels(["adjudicated\n(correct round)", "legacy parser\n(first match)"], fontsize=8)
        axes[1].set_ylim(0, 1.12)
        axes[1].set_ylabel("implied allocative efficiency")
        axes[1].set_title("(b) ... and so does welfare", fontsize=10)
        axes[1].grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "fig4_measurement.png", dpi=300)
        plt.close(fig)

    # ── latex table ────────────────────────────────────────────
    tex = []
    tex.append("\\begin{table}[t]\\centering")
    tex.append("\\caption{Main results (mean over episodes; 95\\% Wilson CI for deal rate).}")
    tex.append("\\label{tab:v3main}")
    tex.append("\\begin{tabular}{llcccccc}\\toprule")
    tex.append("Model & Protocol & $n$ & Deal rate & Efficiency & Rounds & Tokens & Fallback \\\\ \\midrule")
    for _, r in tab.iterrows():
        tex.append(f"{r.model} & {LABELS[r.protocol]} & {int(r.n)} & "
                   f"{r.deal_rate:.2f} [{r.deal_lo:.2f},{r.deal_hi:.2f}] & "
                   f"{r.efficiency:.3f} & {r.rounds:.1f} & {r.tokens:,.0f} & "
                   f"{(r.fallback if not pd.isna(r.fallback) else 0):.2f} \\\\")
    tex.append("\\bottomrule\\end{tabular}\\end{table}")
    (out / "table_main.tex").write_text("\n".join(tex), encoding="utf-8")

    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:60]))
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
