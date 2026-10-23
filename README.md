# SPB: code and data

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
python v3/run_v3.py --models flash pro \
  --protocols free_text alternating_offers sealed_bid sealed_compromise sealed_compromise_verify \
  --trials 30 --episodes 5 --workers 20 --rounds 10 --out output_v3/main

# 2) adjudicate free-dialogue outcomes, then analyse
python v3/judge_free_text.py --labels output_v3/main/audit/labels.csv --workers 10
python v3/analysis_v3.py --dir output_v3/main

# 3) cross-vendor replication
python v3/run_v3.py --models qwen32b glm45air \
  --protocols sealed_bid sealed_compromise sealed_compromise_verify \
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


## Benchmark check

`v3/benchmark_check.py` compares accepted alternating-offer outcomes with the
Rubinstein (1982) subgame-perfect benchmark for delta = 0.9 (proposer share
1/(1+delta) = 0.526). Results: `output_v3/main/analysis/benchmark.md`.

## Audit record

`output_v3/main/audit/signoff.md` records the author's review of the 42 blind
labels (14 non-deals + 28 deals), all confirmed.


## DOI

Archived release v1.0.0: [10.5281/zenodo.23599377](https://doi.org/10.5281/zenodo.23599377)  
Concept DOI: [10.5281/zenodo.23599376](https://doi.org/10.5281/zenodo.23599376)
