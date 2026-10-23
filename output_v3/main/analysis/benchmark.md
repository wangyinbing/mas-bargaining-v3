# Rubinstein benchmark check

- delta = 0.9; SPE proposer share 1/(1+delta) = **0.526**
- accepted alternating-offer deals: 281 episodes

| model | n | proposer share | sd | mean round | t vs SPE | p |
|---|---|---|---|---|---|---|
| flash | 131 | 0.624 | 0.104 | 1.85 | 10.79 | 9.18e-20 |
| pro | 150 | 0.509 | 0.053 | 2.93 | -3.93 | 1.27e-04 |

## Proposer identity

| model | A proposes (share) | n | B proposes (share) | n |
|---|---|---|---|---|
| flash | 0.558 | 56 | 0.673 | 75 |
| pro | 0.507 | 115 | 0.518 | 35 |