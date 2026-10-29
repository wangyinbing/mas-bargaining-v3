"""Assemble a standalone, Zenodo-ready repository for the v3 experiment.

Copies code, data (CSV + transcript tarballs, not the raw per-episode JSON),
analysis outputs and the paper into ../mas-bargaining-v3, writes the metadata
files, and initialises a local git repository (no remote, no push).
"""

import shutil
from pathlib import Path

SRC = Path(r"C:\Users\13121\Desktop\SPB项目")
DST = Path(r"C:\Users\13121\Desktop\SPB项目\归档包\mas-bargaining-v3")

if DST.exists():
    raise SystemExit(f"{DST} already exists")


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    # code
    for f in (SRC / "v3").glob("*.py"):
        copy(f, DST / "v3" / f.name)

    # main data
    md = SRC / "output_v3" / "main"
    copy(md / "episodes.csv", DST / "output_v3/main/episodes.csv")
    copy(md / "transcripts.tar.gz", DST / "output_v3/main/transcripts.tar.gz")
    for f in (md / "analysis").iterdir():
        if f.is_file():
            copy(f, DST / "output_v3/main/analysis" / f.name)
    for name in ["labels.csv", "labels_neg.csv", "judge.csv", "audit_pack.txt",
                 "audit_pack_neg.txt", "audit_key.json", "audit_key_neg.json",
                 "review.html"]:
        p = md / "audit" / name
        if p.exists():
            copy(p, DST / "output_v3/main/audit" / name)

    # cross-vendor data
    xd = SRC / "output_v3" / "xvendor"
    copy(xd / "episodes.csv", DST / "output_v3/xvendor/episodes.csv")
    copy(xd / "transcripts.tar.gz", DST / "output_v3/xvendor/transcripts.tar.gz")
    if (xd / "replication.tex").exists():
        copy(xd / "replication.tex", DST / "output_v3/xvendor/replication.tex")

    # paper
    copy(SRC / "paper_v3.tex", DST / "paper_v3.tex")
    copy(SRC / "paper_v3.pdf", DST / "paper_v3.pdf")

    (DST / "requirements.txt").write_text(
        "openai>=1.40\npandas>=2.2\nnumpy>=1.26\nscipy>=1.11\nmatplotlib>=3.8\n"
        "statsmodels>=0.14\npymupdf>=1.24\n", encoding="utf-8")

    (DST / ".gitignore").write_text(
        ".env\n__pycache__/\n*.pyc\noutput_v3/*/transcripts/\noutput_v3/*/*.log\n"
        "*.aux\n*.log\n*.out\n*.synctex.gz\n", encoding="utf-8")

    (DST / "LICENSE").write_text("""MIT License

Copyright (c) 2026 Yinbing Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""", encoding="utf-8")

    (DST / "CITATION.cff").write_text("""cff-version: 1.2.0
title: "SPB: A Surplus-Preserving Bargaining Protocol for LLM Multi-Agent Negotiation - code and data"
message: "If you use this code or these data, please cite the archived release (DOI) and the paper."
authors:
  - family-names: Wang
    given-names: Yinbing
    affiliation: "Beijing Information Science and Technology University"
type: software
version: 1.0.0
date-released: 2026-10-23
license: MIT
""", encoding="utf-8")

    (DST / ".zenodo.json").write_text("""{
  "title": "SPB: A Surplus-Preserving Bargaining Protocol for LLM Multi-Agent Negotiation - code and data",
  "description": "<p>Code, episode-level data, complete transcripts and analysis scripts for a controlled study of communication protocols in LLM bargaining. The package contains the five-protocol engine (free dialogue, alternating offers, sealed bid, SPB, SPB+), the 1,500-episode DeepSeek experiment, a 315-episode cross-vendor replication (Qwen3-32B, GLM-4.5-Air), the 42-episode hand-audited outcome labels, and the transcript adjudicator used to measure free-dialogue outcomes without regex parsing.</p><p>Code is MIT-licensed; the data files (episodes, transcripts, audit labels) are released under CC BY 4.0.</p>",
  "creators": [
    {"name": "Wang, Yinbing", "affiliation": "Beijing Information Science and Technology University"}
  ],
  "keywords": ["multi-agent systems", "large language models", "bargaining", "mechanism design", "communication protocols", "reproducibility"],
  "license": "MIT",
  "upload_type": "software",
  "access_right": "open",
  "version": "1.0.0",
  "language": "eng"
}
""", encoding="utf-8")

    (DST / "README.md").write_text("""# SPB: code and data

Code and data for the manuscript **"SPB: A Surplus-Preserving Bargaining Protocol for
LLM Multi-Agent Negotiation"** (target: *Applied Intelligence*).

The study asks a simple question that agreement rates cannot answer: when two LLM agents
negotiate over a resource that decays over time, how much of that resource is still there
when the deal closes? We call this **allocative efficiency**, propose a protocol that
preserves it (SPB), and show that the measurement instrument itself can decide the
headline number.

## What is in the package

```
v3/                     experiment engine and analysis
  run_v3.py             five protocols, repeated games, API usage accounting, resume
  legacy_parser.py      the regex parser from the earlier pilot (used as a measurement foil)
  judge_free_text.py    transcript adjudicator for free dialogue
  analysis_v3.py        statistics, figures, tables
  make_audit*.py        blind audit packs
  summarize_xvendor.py  cross-vendor replication table
output_v3/main/         1,500 episodes on two DeepSeek backbones
  episodes.csv          one row per episode
  transcripts.tar.gz    complete dialogues (JSON, one file per episode)
  analysis/             figures, summary.md, table_main.tex
  audit/                the 42 hand-audited outcomes and the adjudicator output
output_v3/xvendor/      315 episodes on Qwen3-32B and GLM-4.5-Air
paper_v3.tex / .pdf     the manuscript
```

## Reproduce

```bash
pip install -r requirements.txt

# 1) main experiment (1,500 episodes; ~25 min with 20 workers)
python v3/run_v3.py --models flash pro \\
  --protocols free_text alternating_offers sealed_bid sealed_compromise sealed_compromise_verify \\
  --trials 30 --episodes 5 --workers 20 --rounds 10 --out output_v3/main

# 2) adjudicate free-dialogue outcomes, then analyse
python v3/judge_free_text.py --labels output_v3/main/audit/labels.csv --workers 10
python v3/analysis_v3.py --dir output_v3/main

# 3) cross-vendor replication
python v3/run_v3.py --models qwen32b glm45air \\
  --protocols sealed_bid sealed_compromise sealed_compromise_verify \\
  --trials 12 --episodes 5 --workers 8 --rounds 10 --out output_v3/xvendor
python v3/summarize_xvendor.py --dir output_v3/xvendor
```

An API key is required. Put it in `.env` next to `v3/`:

```
DEEPSEEK_API_KEY=...
SILICONFLOW_API_KEY=...      # only for the cross-vendor arm
```

Model configuration matters: DeepSeek endpoints are pinned to `reasoning_effort="none"`
and Qwen/GLM to `enable_thinking=False`. Without these flags both families spend the
output budget on hidden reasoning and return empty content for short decision turns.

## Data dictionary (`episodes.csv`)

| column | meaning |
|---|---|
| `model`, `protocol`, `trial`, `episode` | identifiers; a trial is 5 repeated episodes between the same pair |
| `deal` | primary outcome: mechanism deal (structured) or adjudicated meeting of minds (free dialogue) |
| `agree_elicited` | post-hoc self-report (both agents say YES with the same split) |
| `deal_strict`, `agree_legacy` | deterministic symmetric detector; legacy regex parser |
| `split_a`, `split_b` | agreed shares |
| `allocated`, `welfare`, `pie_final` | points allocated; same as welfare; pie of the round in which the deal closed |
| `rounds` | round of agreement (adjudicated for free dialogue; `T` if none) |
| `fallback_offered`, `fallback_accepted`, `confirm_failed` | SPB/SPB+ mechanism events |
| `demand_a_mean`, `demand_b_mean` | mean sealed demand |
| `tokens_in`, `tokens_out` | API-reported usage |

Efficiency in the paper is `welfare / 10`.

## The 42 audited outcomes

Free dialogue has no execution event, so its outcome has to be read from the transcript.
`audit/labels.csv` and `audit/labels_neg.csv` contain the hand labels; `audit/judge.csv`
contains the adjudicator used for all 300 free-dialogue episodes, which agrees with the
hand labels on 42/42. `audit/review.html` is a click-through viewer for the audit.

## License

Code: MIT. Data (episodes, transcripts, audit labels): CC BY 4.0.

## Citation

If you use this package, please cite the archived release (DOI to be inserted after
archiving) and the paper.
""", encoding="utf-8")

    # local git repo
    import subprocess
    subprocess.run(["git", "init", "-b", "main"], cwd=DST, check=True,
                   stdout=subprocess.DEVNULL)
    subprocess.run(["git", "add", "."], cwd=DST, check=True)
    subprocess.run(["git", "-c", "user.name=Yinbing Wang",
                    "-c", "user.email=1934482121@qq.com",
                    "commit", "-m", "SPB code and data v1.0.0"], cwd=DST, check=True,
                   stdout=subprocess.DEVNULL)
    total = sum(f.stat().st_size for f in DST.rglob("*") if f.is_file()) / 1e6
    print(f"built {DST} ({total:.1f} MB)")
    for p in sorted(DST.rglob("*")):
        if p.is_file() and p.suffix in {".csv", ".gz", ".pdf", ".tex", ".md"}:
            print(f"  {p.relative_to(DST)}  {p.stat().st_size/1e3:.0f} KB")


if __name__ == "__main__":
    main()
