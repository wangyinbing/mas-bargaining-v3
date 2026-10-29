# Data dictionary

## `output_v3/main/episodes.csv` — 1,500 episodes

Two DeepSeek backbones × five protocols × 30 trials × 5 repeated episodes.

| Column | Meaning |
|---|---|
| `model`, `protocol`, `trial`, `episode` | identifiers; `episode` is the position inside a repeated game (1–5) |
| `deal` | **for structured protocols**: the protocol's rule executed a split. **For `free_text` this is the strict symmetric detector's output, not an executed deal** — free dialogue has no execution rule at all. Use the adjudicated columns below. |
| `allocated`, `welfare`, `pie_final` | points allocated, and the pie still on the table at the closing round |
| `rounds` | closing round (for `free_text`: the round the episode stopped, at the limit) |
| `split_a`, `split_b` | the split that was executed (structured) or detected (free dialogue) |
| `tokens_in`, `tokens_out`, `calls` | API-reported usage |
| `demand_a_mean`, `demand_b_mean` | mean first-stage demands (sealed protocols) |
| `fallback_offered`, `fallback_accepted` | whether the compromise branch fired and was accepted |
| `confirm_failed` | number of failed commitment confirmations |
| `status` | `ok` or `error`; **always filter on `ok`** |

### Where the paper's free-dialogue numbers come from

| Paper quantity | Source |
|---|---|
| 99.7% agreement, mean round 2.1, efficiency 0.905 | `audit/judge_first.csv` (`first_round`) |
| 95.3% agreement, mean round 9.8, efficiency 0.388 | `audit/judge.csv` (`deal`, `round`) |
| 13 deals missed by a late-window reading | `judge_first.first_round > 0` AND `judge.deal == 0` |
| executed efficiency 0.000 | by construction — no rule pays a split out |

`judge.csv` reads the **last four rounds**; `judge_first.csv` reads the **whole
transcript** and reports the first mutual acceptance. Both were produced by
`v3/judge_free_text.py` and `v3/judge_first_round.py` (DeepSeek-V4-Pro,
temperature 0). The 20-episode hand check is `audit/val20_keys.csv`.

## `output_v3/free_exec/` and `output_v3/free_loose/` — 300 episodes each

Free dialogue with a binding execution hook.

- `free_exec`: strict — `PROPOSE: A=<int>, B=<int>` / `ACCEPT: A=<int>, B=<int>`
  must match exactly.
- `free_loose`: forgiving — ordinary language ("I accept 5-5") executes the
  pending proposal.

`deal` here **is** an executed deal (the hook fired).

## `output_v3/equal/episodes.csv` — 300 episodes

`sealed_compromise_equal`: identical to SPB except that the fallback ignores the
demands and splits the pie equally, so a higher demand cannot raise the fallback
share. Used in Section 6.5 to test whether demand inflation is caused by the rule
or by the models. Demands still average 5.90 (flash) and 5.09 (pro), against 6.23
and 5.20 under the proportional rule.

## `output_v3/xvendor30/episodes.csv` — 704 rows, 701 valid

Qwen3-32B and GLM-4.5-Air, three sealed protocols, 19–26 trials × 5 episodes.
Rebuilt from `transcripts/` after a killed run; three rows carry
`status == "error"` (rate limits) and must be excluded.

## `output_v3/main/transcripts/` — one JSON per episode

```json
{"row": {...}, "transcript": [{"round": 1, "a": "...", "b": "..."}],
 "bids": [...], "proposals": [["A", 6, 4]], "elicit": {...}, "legacy": {...}}
```

`proposals[-1]` is the accepted proposal in alternating offers (used by the
Rubinstein check). `row` duplicates the CSV row for that episode, which is what
makes post-hoc reconstruction possible.
