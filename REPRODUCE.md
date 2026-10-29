# Reproducing the numbers in the paper

Everything below runs from the repository root. Python 3.11 with `pandas`, `numpy`,
`scipy`, `matplotlib`, `openai`.

## One command check

```bash
python v3/verify_paper.py
```

This recomputes every number quoted in the paper — the abstract figures, Table 1,
Table 2 (the four readings), the strict and forgiving hook arms, the cross-vendor
table, the Kruskal–Wallis statistics, the Holm-corrected pairwise tests, the δ
sensitivity, the learning slopes, the Rubinstein benchmark and the token counts —
and compares each against the value printed in the manuscript. It exits non-zero
if any check fails. Last run: **ALL CHECKS PASSED**.

## Table → script → source data

| Artifact | Script | Data |
|---|---|---|
| Table 1 (main results) | `v3/analysis_v3.py` | `output_v3/main/episodes.csv` |
| Table 2 (four readings) | `verify_paper.py`, section *free dialogue conventions* | `output_v3/main/episodes.csv` + `audit/judge_first.csv` + `audit/judge.csv` |
| Strict hook arm | `verify_paper.py`, section *hook arms* | `output_v3/free_exec/episodes.csv` |
| Forgiving hook arm | `verify_paper.py`, section *hook arms* | `output_v3/free_loose/episodes.csv` |
| Table 3 (cross-vendor) | `verify_paper.py`, section *cross-vendor* | `output_v3/xvendor30/episodes.csv` (rows with `status == "ok"`) |
| δ sensitivity | `verify_paper.py`, section *delta sensitivity* | same episode table, recomputed at δ = 0.80/0.90/0.95 |
| Learning slopes | `verify_paper.py`, section *learning slopes* | `episodes.csv`, regressed on `episode` |
| Rubinstein benchmark | `v3/benchmark_check.py` | `output_v3/main/transcripts/*alternating_offers*.json` (`proposals[-1]`) |
| Adjudicator validation | `v3/make_audit.py`, `audit/val20_keys.csv` | the 20 hand-coded episodes |

## Re-running the experiments (requires API keys)

```bash
# main experiment: 1500 episodes
python v3/run_v3.py --models flash pro --protocols free_text alternating_offers sealed_bid \
  sealed_compromise sealed_compromise_verify --trials 30 --episodes 5 --workers 16 \
  --rounds 10 --out output_v3/main

# strict execution hook (300 episodes)
python v3/run_v3.py --models flash pro --protocols free_text_exec --trials 30 --episodes 5 \
  --workers 16 --rounds 10 --out output_v3/free_exec

# forgiving execution hook (300 episodes)
python v3/run_v3.py --models flash pro --protocols free_text_loose --trials 30 --episodes 5 \
  --workers 16 --rounds 10 --out output_v3/free_loose

# cross-vendor replication (Qwen3-32B, GLM-4.5-Air)
python v3/run_v3.py --models qwen32b glm45air --protocols sealed_bid sealed_compromise \
  sealed_compromise_verify --trials 30 --episodes 5 --workers 6 --rounds 10 \
  --out output_v3/xvendor30

# adjudication of free dialogue
python v3/judge_free_text.py --dir output_v3/main/transcripts --workers 12
python v3/judge_first_round.py      # first-acceptance convention
```

Every run is resumable: `run_v3.py` skips episodes already present with
`status == "ok"` in the target `episodes.csv` and reads any transcript already on
disk, so a killed run continues where it stopped.

## Known issues a reader should not rediscover

- **Do not trust the CSV when a run was killed.** Transcripts are written before the
  CSV row; a kill can leave completed episodes missing from the CSV. The shipped
  `xvendor30/episodes.csv` was rebuilt from its transcripts for exactly this reason
  (14 rows were recovered, 5 conflicting duplicate keys resolved by transcript mtime).
- **Three cross-vendor episodes failed on rate limits** (`status == "error"`) and are
  excluded from every number; filter on `status == "ok"`.
- **`deal` in `episodes.csv` is not the paper's agreement rate for free dialogue.**
  See `DATA.md`.
