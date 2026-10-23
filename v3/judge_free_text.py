"""Adjudicate free-dialogue outcomes from the transcript with an independent model.

Motivation: the deterministic strict detector misses common phrasings
("A gets 2, B gets 2") and compares the split against the initial pie rather
than the round's pie. The audit of 30 hand-judged episodes showed recall 0.50.
This script replaces it with a transcript adjudicator (DeepSeek-V4-Pro, reasoning
disabled) and validates the adjudicator against the 30 hand labels.

Usage:
  python v3/judge_free_text.py --labels output_v3/main/audit/labels.csv   # validation only
  python v3/judge_free_text.py --all                                     # all free-text episodes
"""

import argparse
import csv
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_v3 import load_api_key, pie  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TDIR = ROOT / "output_v3" / "main" / "transcripts"

PROMPT = """You are auditing a two-agent bargaining transcript. Agents A and B negotiate over a pie
that shrinks every round. In each round each agent sends one message.

Decide whether the two agents reached a binding agreement on a split.
An agreement requires BOTH agents to explicitly accept the SAME split, i.e. the same number of
points for A and the same number for B. A proposal, a counter-proposal, or a one-sided acceptance
does NOT count. If they agreed and later renegotiated, report the FINAL agreement, i.e. the last
round in which both agents accepted the same split.

Reply with exactly these four lines and nothing else:
DEAL: YES|NO
ROUND: <round number, or 0 if no deal>
A: <points for A, or 0>
B: <points for B, or 0>"""


def clean(t):
    return (t or "").replace("\n", " ").strip()[:700]


def render(transcript, upto=4):
    rounds = [e for e in transcript if e.get("round")]
    rounds = rounds[-upto:]
    out = []
    for e in rounds:
        out.append(f"Round {e['round']} (pie {pie(10, e['round'])} points)")
        out.append(f"A: {clean(e.get('a'))}")
        out.append(f"B: {clean(e.get('b'))}")
    return "\n".join(out)


def judge(client, transcript):
    msg = PROMPT + "\n\nTRANSCRIPT\n" + render(transcript)
    r = client.chat.completions.create(
        model="deepseek-v4-pro", messages=[{"role": "user", "content": msg}],
        max_tokens=200, temperature=0.0, reasoning_effort="none")
    txt = (r.choices[0].message.content or "").strip()
    m_d = re.search(r"DEAL\s*:\s*(YES|NO)", txt, re.I)
    m_r = re.search(r"ROUND\s*:\s*(\d+)", txt)
    m_a = re.search(r"\bA\s*:\s*(\d+)", txt)
    m_b = re.search(r"\bB\s*:\s*(\d+)", txt)
    if not m_d:
        return None, txt
    if m_d.group(1).upper() == "NO":
        return dict(deal=0, round=0, split_a=None, split_b=None), txt
    return dict(deal=1, round=int(m_r.group(1)) if m_r else 0,
                split_a=int(m_a.group(1)) if m_a else None,
                split_b=int(m_b.group(1)) if m_b else None), txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", help="CSV with hand labels (id,deal,split_a,split_b)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "output_v3" / "main" / "audit" / "judge.csv"))
    args = ap.parse_args()

    key = load_api_key()
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=120)

    files = sorted(TDIR.glob("*free_text*.json"))
    if args.limit:
        files = files[:args.limit]

    jl = ROOT / "output_v3" / "main" / "audit" / "judge.jsonl"
    done = {}
    if jl.exists():
        for line in jl.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
                done[d["file"]] = d
            except Exception:  # noqa: BLE001
                pass

    lock = threading.Lock()

    def work(f):
        if f.name in done:
            return done[f.name]
        d = json.loads(f.read_text(encoding="utf-8"))
        j, raw = judge(client, d["transcript"])
        row = dict(file=f.name, model=d["row"]["model"], trial=d["row"]["trial"],
                   episode=d["row"]["episode"], raw=raw)
        row.update(j or dict(deal=None, round=None, split_a=None, split_b=None))
        with lock:
            with jl.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(work, f): f for f in files}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rows.append(fut.result())
            except Exception as e:  # noqa: BLE001
                print("fail", futs[fut].name, type(e).__name__, e)
            if i % 50 == 0:
                print(f"  {i}/{len(files)}", flush=True)

    out = Path(args.out)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[judged] {len(rows)} -> {out}")

    if args.labels:
        lab = {r["id"]: r for r in csv.DictReader(open(args.labels, encoding="utf-8"))}
        # labels are keyed by blind id; rebuild the map from audit_key.json
        keymap = json.loads((ROOT / "output_v3/main/audit/audit_key.json").read_text(encoding="utf-8"))
        f2id = {v["file"]: k for k, v in keymap.items()}
        tp = fp = fn = tn = 0
        split_ok = 0
        for r in rows:
            lid = f2id.get(r["file"])
            if lid not in lab:
                continue
            h = lab[lid]
            hd = h["deal"] == "1"
            jd = bool(r["deal"])
            if jd and hd:
                tp += 1
                if r["split_a"] is not None and int(h["split_a"]) == r["split_a"] \
                   and int(h["split_b"]) == r["split_b"]:
                    split_ok += 1
            elif jd and not hd:
                fp += 1
            elif (not jd) and hd:
                fn += 1
            else:
                tn += 1
        n = tp + fp + fn + tn
        print(f"validation n={n}: TP={tp} FP={fp} FN={fn} TN={tn} "
              f"acc={(tp+tn)/n:.3f} precision={tp/(tp+fp) if tp+fp else float('nan'):.3f} "
              f"recall={tp/(tp+fn) if tp+fn else float('nan'):.3f} split_match={split_ok}/{tp}")


if __name__ == "__main__":
    main()
