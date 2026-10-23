"""Rubinstein benchmark check on the alternating-offer episodes.

With a per-round discount delta, the subgame-perfect equilibrium of the
alternating-offer game gives the proposer the share 1/(1+delta). We compare the
proposer's share in accepted deals against that benchmark.

Output: output_v3/main/analysis/benchmark.md
"""

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
TDIR = ROOT / "output_v3" / "main" / "transcripts"
OUT = ROOT / "output_v3" / "main" / "analysis" / "benchmark.md"

DELTA = 0.9
SPE = 1 / (1 + DELTA)


def main():
    rows = []
    for f in glob.glob(str(TDIR / "*alternating_offers*.json")):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        r = d["row"]
        if r.get("status") != "ok" or int(r.get("deal", 0)) != 1:
            continue
        props = d.get("proposals") or []
        if not props:
            continue
        pname, mine, _ = props[-1]          # the accepted proposal
        rnd = int(r["rounds"])
        pie = max(1, int(round(10 * DELTA ** (rnd - 1))))
        rows.append(dict(model=r["model"], proposer=pname, share=mine / pie,
                         pie=pie, round=rnd))
    df = pd.DataFrame(rows)

    lines = [f"# Rubinstein benchmark check\n",
             f"- delta = {DELTA}; SPE proposer share 1/(1+delta) = **{SPE:.3f}**",
             f"- accepted alternating-offer deals: {len(df)} episodes\n",
             "| model | n | proposer share | sd | mean round | t vs SPE | p |",
             "|---|---|---|---|---|---|---|"]
    for model, g in df.groupby("model"):
        t, p = stats.ttest_1samp(g.share, SPE)
        lines.append(f"| {model} | {len(g)} | {g.share.mean():.3f} | {g.share.std():.3f} | "
                     f"{g['round'].mean():.2f} | {t:.2f} | {p:.2e} |")
    lines.append("")
    lines.append("## Proposer identity")
    lines.append("")
    lines.append("| model | A proposes (share) | n | B proposes (share) | n |")
    lines.append("|---|---|---|---|---|")
    for model, g in df.groupby("model"):
        a = g[g.proposer == "A"].share
        b = g[g.proposer == "B"].share
        lines.append(f"| {model} | {a.mean():.3f} | {len(a)} | {b.mean():.3f} | {len(b)} |")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
