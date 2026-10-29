"""Adjudicate the FIRST round of mutual agreement (full transcript), and compare
with the late-window reading used before.

Usage: python v3/judge_first_round.py --dir output_v3/main --workers 12
"""
import argparse, json, re, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_v3 import load_api_key  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

PROMPT = """You are auditing a two-agent bargaining transcript. Agents A and B negotiate over a pie that shrinks every round. There are 10 rounds.

Read the WHOLE transcript and answer: in which round did the two agents FIRST state acceptance of the SAME split?

Rules:
- A round counts only if both agents, in that round or in response to each other, accept the same split of the same pie.
- Report the FIRST such round, not the last one. Agents often repeat a settlement for many rounds; that does not move the round.
- If they never agree, report round 0.
- If the agreed split is stated, report it; if a point is left unallocated, report the split as stated.

Reply exactly:
FIRST_ROUND: <int>
SPLIT: A=<int>, B=<int>
"""


def render(tr):
    out = []
    for m in tr:
        out.append(f"Round {m.get('round')}:\nAgent A: {m.get('a') or ''}\nAgent B: {m.get('b') or ''}")
    return "\n\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="output_v3/main")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ddir = Path(args.dir)
    if not ddir.is_absolute():
        ddir = ROOT / ddir
    tdir = ddir / "transcripts"
    out_path = ddir / "audit" / "judge_first.csv"
    client = OpenAI(api_key=load_api_key(), base_url="https://api.deepseek.com", timeout=180)

    files = sorted(tdir.glob("*free_text*.json"))
    if args.limit:
        files = files[: args.limit]

    lock = threading.Lock()
    rows = []

    def work(f):
        d = json.loads(f.read_text(encoding="utf-8"))
        msg = PROMPT + "\n\nTRANSCRIPT\n" + render(d["transcript"])
        r = client.chat.completions.create(
            model="deepseek-chat", messages=[{"role": "user", "content": msg}],
            temperature=0, max_tokens=80, extra_body={"reasoning_effort": "none"})
        txt = (r.choices[0].message.content or "").strip()
        mr = re.search(r"FIRST_ROUND\s*:\s*(\d+)", txt, re.I)
        ms = re.search(r"SPLIT\s*:\s*A\s*=\s*(\d+)\s*,\s*B\s*=\s*(\d+)", txt, re.I)
        row = dict(file=f.name, model=d["row"]["model"], trial=d["row"]["trial"],
                   episode=d["row"]["episode"], raw=txt[:200],
                   first_round=int(mr.group(1)) if mr else None,
                   split_a=int(ms.group(1)) if ms else None,
                   split_b=int(ms.group(2)) if ms else None)
        with lock:
            rows.append(row)
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(work, f) for f in files]
        for i, _ in enumerate(as_completed(futs), 1):
            if i % 50 == 0:
                print(f"{i}/{len(files)}", flush=True)

    import pandas as pd
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("written", out_path, len(rows))


if __name__ == "__main__":
    main()
