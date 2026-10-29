# v3 analysis — 1500 episodes

models: ['flash', 'pro']; R=10; T=10; K=5

## Main results

| model   | protocol                 | label              |   n |   deal_rate |   deal_lo |   deal_hi |   efficiency |   eff_sd |   rounds |    tokens |   fallback |   confirm_fail |   demand_a |
|:--------|:-------------------------|:-------------------|----:|------------:|----------:|----------:|-------------:|---------:|---------:|----------:|-----------:|---------------:|-----------:|
| flash   | free_text                | Free dialogue      | 150 |       0.953 |     0.907 |     0.977 |        0.000 |    0.000 |    9.740 | 52559.133 |      0.000 |          0.000 |    nan     |
| flash   | alternating_offers       | Alternating offers | 150 |       0.873 |     0.811 |     0.917 |        0.805 |    0.321 |    2.880 |  7805.487 |      0.000 |          0.000 |    nan     |
| flash   | sealed_bid               | Sealed bid         | 150 |       1.000 |     0.975 |     1.000 |        0.937 |    0.085 |    1.633 |  4205.693 |      0.000 |          0.000 |      5.030 |
| flash   | sealed_compromise        | SPB (ours)         | 150 |       1.000 |     0.975 |     1.000 |        1.000 |    0.000 |    1.000 |  4558.420 |      0.833 |          0.000 |      6.233 |
| flash   | sealed_compromise_verify | SPB+ (ours)        | 150 |       0.933 |     0.882 |     0.963 |        0.871 |    0.278 |    2.380 | 14487.487 |      0.967 |          0.027 |      7.571 |
| pro     | free_text                | Free dialogue      | 150 |       0.953 |     0.907 |     0.977 |        0.000 |    0.000 |    9.847 | 19310.653 |      0.000 |          0.000 |    nan     |
| pro     | alternating_offers       | Alternating offers | 150 |       1.000 |     0.975 |     1.000 |        0.865 |    0.237 |    2.927 |  5892.327 |      0.000 |          0.000 |    nan     |
| pro     | sealed_bid               | Sealed bid         | 150 |       0.993 |     0.963 |     0.999 |        0.972 |    0.102 |    1.293 |  3588.493 |      0.000 |          0.000 |      4.969 |
| pro     | sealed_compromise        | SPB (ours)         | 150 |       1.000 |     0.975 |     1.000 |        0.998 |    0.018 |    1.020 |  3906.473 |      0.207 |          0.000 |      5.203 |
| pro     | sealed_compromise_verify | SPB+ (ours)        | 150 |       1.000 |     0.975 |     1.000 |        0.960 |    0.082 |    1.407 |  5043.680 |      0.227 |          0.000 |      5.058 |

## Tests

- **flash** deal rate across protocols: chi2(4)=36.59, p=2.19e-07
- **flash** efficiency across protocols: Kruskal-Wallis H=470.99, p=1.26e-100
- **flash** sealed_compromise vs sealed_bid: Mann-Whitney U=15675, p=1.80e-17, mean 1.000 vs 0.937
- **flash** sealed_compromise_verify vs sealed_bid: Mann-Whitney U=12126, p=1.56e-01, mean 0.871 vs 0.937
- **flash** sealed_compromise_verify vs sealed_compromise: Mann-Whitney U=8625, p=3.66e-10, mean 0.871 vs 1.000
- **pro** deal rate across protocols: chi2(4)=23.50, p=1.01e-04
- **pro** efficiency across protocols: Kruskal-Wallis H=542.48, p=4.34e-116
- **pro** sealed_compromise vs sealed_bid: Mann-Whitney U=12456, p=2.06e-04, mean 0.998 vs 0.972
- **pro** sealed_compromise_verify vs sealed_bid: Mann-Whitney U=10280, p=4.46e-02, mean 0.960 vs 0.972
- **pro** sealed_compromise_verify vs sealed_compromise: Mann-Whitney U=9063, p=8.09e-08, mean 0.960 vs 0.998

## Measurement comparison

- free dialogue episodes: 300
- adjudicated (validated): deal **0.953**, mean round 9.78, efficiency **0.000**
- post-hoc self-report: deal 0.667
- strict symmetric detector: deal 0.690, mean round 7.55
- legacy parser: deal 0.917, mean round 1.92, legacy-implied efficiency **0.845**
- structured protocols: mechanism deal 0.975, elicitation 0.976, mismatch 0.003

## Learning over repeated episodes (efficiency slope per episode)

| model   | protocol                 |   slope |        p |       r2 |
|:--------|:-------------------------|--------:|---------:|---------:|
| flash   | free_text                |  0.0000 | nan      | nan      |
| flash   | alternating_offers       |  0.0313 |   0.0914 |   0.0191 |
| flash   | sealed_bid               |  0.0370 |   0.0000 |   0.3844 |
| flash   | sealed_compromise        |  0.0000 | nan      | nan      |
| flash   | sealed_compromise_verify |  0.0820 |   0.0000 |   0.1747 |
| pro     | free_text                |  0.0000 | nan      | nan      |
| pro     | alternating_offers       |  0.1227 |   0.0000 |   0.5374 |
| pro     | sealed_bid               |  0.0280 |   0.0000 |   0.1505 |
| pro     | sealed_compromise        |  0.0020 |   0.0568 |   0.0243 |
| pro     | sealed_compromise_verify |  0.0387 |   0.0000 |   0.4485 |
