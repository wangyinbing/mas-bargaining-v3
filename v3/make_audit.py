"""Build a blind audit pack for the free-dialogue strict detector.

Sampling: 30 episodes, stratified by model and by detector verdict
(15 detector-positive / 15 detector-negative, balanced across models).
The pack deliberately omits the detector verdict so that the adjudicator
can judge the transcript independently; a second script joins the two.
"""

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_v3 import strict_free_text_deal  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TDIR = ROOT / "output_v3" / "main" / "transcripts"
OUT = ROOT / "output_v3" / "main" / "audit"
ACC = re.compile(r'\b(accept|accepted|agree|agreed|deal|yes|ok|okay)\b')


def load_free_text():
    rows = []
    for f in sorted(TDIR.glob("*free_text*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        sp, rnd = strict_free_text_deal(d["transcript"], 10)
        rows.append(dict(file=f.name, model=d["row"]["model"], trial=d["row"]["trial"],
                         episode=d["row"]["episode"], transcript=d["transcript"],
                         det=sp is not None, det_round=rnd, det_split=sp))
    return rows


def evidence(t, det_round):
    """Return the messages a human would need, without any verdict."""
    out = []
    rounds = {e.get("round"): e for e in t if e.get("round")}
    # last two rounds (final state)
    for r in sorted(rounds)[-2:]:
        out.append((r, rounds[r]))
    # rounds where both messages carry an acceptance token, up to 2
    extra = []
    for r in sorted(rounds):
        e = rounds[r]
        a, b = e.get("a") or "", e.get("b") or ""
        if ACC.search(a.lower()) and ACC.search(b.lower()):
            extra.append(r)
    for r in extra[-2:]:
        if r not in [x[0] for x in out]:
            out.append((r, rounds[r]))
    out.sort(key=lambda x: x[0])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_free_text()
    rng = random.Random(20261023)

    # balanced strata: model x detector verdict
    strata = {}
    for r in rows:
        strata.setdefault((r["model"], r["det"]), []).append(r)

    picked = []
    # aim for 15 pos / 15 neg, and as balanced by model as the strata allow
    plan = [("flash", True), ("pro", True), ("flash", False), ("pro", False)]
    quotas = {("flash", True): 8, ("pro", True): 7, ("flash", False): 7, ("pro", False): 8}
    for key in plan:
        pool = strata.get(key, [])[:]
        rng.shuffle(pool)
        picked.extend(pool[:quotas[key]])

    # blind labels
    rng.shuffle(picked)
    label = {}
    for i, r in enumerate(picked, 1):
        label[r["file"]] = f"E{i:02d}"

    lines = []
    for r in picked:
        lid = label[r["file"]]
        lines.append("=" * 78)
        lines.append(f"EPISODE {lid}   (model={r['model']}, trial={r['trial']}, episode={r['episode']})")
        lines.append("=" * 78)
        for rnd, e in evidence(r["transcript"], r["det_round"]):
            a = (e.get("a") or "").replace("\n", " ")
            b = (e.get("b") or "").replace("\n", " ")
            lines.append(f"-- round {rnd} --")
            lines.append(f"A: {a[:520]}")
            lines.append(f"B: {b[:520]}")
        lines.append("")

    (OUT / "audit_pack.txt").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "audit_key.json").write_text(
        json.dumps({label[r["file"]]: dict(file=r["file"], model=r["model"],
                                           det=r["det"], det_round=r["det_round"],
                                           det_split=r["det_split"]) for r in picked},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(picked)} episodes -> {OUT/'audit_pack.txt'}")
    print("positives:", sum(1 for r in picked if r["det"]), "negatives:",
          sum(1 for r in picked if not r["det"]))


if __name__ == "__main__":
    main()
