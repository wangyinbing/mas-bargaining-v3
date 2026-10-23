"""Blind audit pack for the adjudicated NON-deals (the population, not a sample).

The main audit found only 14 adjudicated non-deals among 300 free-dialogue
episodes; two of them were already in the first 30-episode blind pack. This
script exports the remaining 12, again without any verdict, so that specificity
can be checked on the whole negative population.
"""

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TDIR = ROOT / "output_v3" / "main" / "transcripts"
OUT = ROOT / "output_v3" / "main" / "audit"
ACC = re.compile(r'\b(accept|accepted|agree|agreed|deal|yes|ok|okay)\b')

# already hand-audited in the first pack
SEEN = {"flash_free_text_010_e5.json", "flash_free_text_002_e4.json"}


def evidence(t):
    rounds = {e.get("round"): e for e in t if e.get("round")}
    out = [(r, rounds[r]) for r in sorted(rounds)[-2:]]
    extra = [r for r in sorted(rounds)
             if ACC.search((rounds[r].get("a") or "").lower())
             and ACC.search((rounds[r].get("b") or "").lower())]
    for r in extra[-2:]:
        if r not in [x[0] for x in out]:
            out.append((r, rounds[r]))
    out.sort(key=lambda x: x[0])
    return out


def main():
    j = pd.read_csv(OUT / "judge.csv")
    neg = j[(j.deal == 0) & (~j.file.isin(SEEN))].reset_index(drop=True)
    lines, key = [], {}
    for i, r in neg.iterrows():
        lid = f"N{i+1:02d}"
        key[lid] = dict(file=r.file, model=r.model, trial=int(r.trial), episode=int(r.episode))
        d = json.loads((TDIR / r.file).read_text(encoding="utf-8"))
        lines.append("=" * 78)
        lines.append(f"EPISODE {lid}   (model={r.model}, trial={r.trial}, episode={r.episode})")
        lines.append("=" * 78)
        for rnd, e in evidence(d["transcript"]):
            lines.append(f"-- round {rnd} --")
            lines.append(f"A: {(e.get('a') or '').replace(chr(10), ' ')[:520]}")
            lines.append(f"B: {(e.get('b') or '').replace(chr(10), ' ')[:520]}")
        lines.append("")
    (OUT / "audit_pack_neg.txt").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "audit_key_neg.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(neg)} non-deals -> {OUT/'audit_pack_neg.txt'}")


if __name__ == "__main__":
    main()
