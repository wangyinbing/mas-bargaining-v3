"""Generate neutral Chinese summaries of each audited episode's evidence.

The summary states what each agent said in the final rounds (proposals,
acceptances, rejections) without judging whether a deal was reached. The
English transcript is kept verbatim in the review page; this is a reading aid.
"""

import csv
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_v3 import load_api_key  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
AUD = ROOT / "output_v3" / "main" / "audit"
TDIR = ROOT / "output_v3" / "main" / "transcripts"
OUT = AUD / "summaries_zh.json"

PROMPT = """下面是一段两个 AI 代理（A 和 B）谈判的最后几轮对话。饼每轮缩水。

请用中文客观复述这几轮发生了什么，要求：
1. 按轮次说清楚：谁提出了什么分割（A 拿几、B 拿几）、对方是接受、拒绝，还是还价。
2. 只复述，不判断是否最终成交，不评价谁对谁错。
3. 150 字以内，不要用列表，用一段话。

对话：
{body}"""


def evidence_text(transcript, focus=None, n_last=3, limit=500):
    rounds = {e.get("round"): e for e in transcript if e.get("round")}
    keys = sorted(rounds)
    sel = keys[-n_last:]
    if focus and focus in rounds and focus not in sel:
        sel = [focus] + sel
    lines = []
    for r in sel:
        e = rounds[r]
        pie = max(1, int(round(10 * 0.9 ** (r - 1))))
        lines.append(f"第{r}轮（饼{pie}分）：")
        lines.append(f"A: {(e.get('a') or '').replace(chr(10), ' ')[:limit]}")
        lines.append(f"B: {(e.get('b') or '').replace(chr(10), ' ')[:limit]}")
    return "\n".join(lines)


def main():
    key = load_api_key("DEEPSEEK_API_KEY")
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=90)

    key_main = json.loads((AUD / "audit_key.json").read_text(encoding="utf-8"))
    key_neg = json.loads((AUD / "audit_key_neg.json").read_text(encoding="utf-8"))
    labels = {r["id"]: r for r in csv.DictReader(open(AUD / "labels.csv", encoding="utf-8"))}
    labels_neg = {r["id"]: r for r in csv.DictReader(open(AUD / "labels_neg.csv", encoding="utf-8"))}

    done = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    lock = threading.Lock()

    tasks = []
    for kf, labs in [(key_main, labels), (key_neg, labels_neg)]:
        for eid, meta in sorted(kf.items()):
            if eid not in labs or eid in done:
                continue
            t = json.loads((TDIR / meta["file"]).read_text(encoding="utf-8"))
            body = evidence_text(t["transcript"], focus=meta.get("det_round"))
            tasks.append((eid, body))

    def work(item):
        eid, body = item
        r = client.chat.completions.create(
            model="deepseek-flash", messages=[{"role": "user", "content": PROMPT.format(body=body)}],
            temperature=0.0, max_tokens=400, extra_body={"reasoning_effort": "none"})
        txt = (r.choices[0].message.content or "").strip()
        with lock:
            done[eid] = txt
            OUT.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
        return eid

    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(work, t) for t in tasks]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                print("fail", type(e).__name__, e)
    print(f"summaries: {len(done)} -> {OUT}")


if __name__ == "__main__":
    main()
