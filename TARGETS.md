# Reference numbers

All values ×100. **Protocol: time-aware filtered** — only facts that are
true at the query timestamp are removed from the rank list. This is the
protocol `train_kairos.py` reports as `time_aware_filtered`, and the one
RE-GCN, TiRGN, CENET, DaeMon, TiPNN and DiMNet report.

Numbers below are taken from the DiMNet comparison table
(https://arxiv.org/html/2505.14020), which reports all methods under one
consistent protocol. Cross-paper numbers under *different* protocols are not
comparable and must not be mixed into a single table.

## ICEWS18

| Method | MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|
| CyGNet | 24.93 | 15.90 | 28.28 | 42.61 |
| xERTE | 29.31 | 21.03 | 33.40 | 45.60 |
| TITer | 29.98 | 22.05 | 33.46 | 44.83 |
| RE-GCN | 30.58 | 21.01 | 34.34 | 48.75 |
| CEN | 30.84 | 21.23 | 34.58 | 49.67 |
| DaeMon | 31.85 | 22.67 | 35.92 | 49.80 |
| TiPNN | 32.17 | 22.74 | 36.24 | 50.72 |
| **DiMNet (SOTA)** | **34.13** | **23.29** | **38.42** | **55.80** |

## GDELT

| Method | MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|
| TITer | 15.46 | 10.98 | 15.61 | 24.31 |
| xERTE | 18.09 | 12.30 | 20.06 | 30.34 |
| CyGNet | 18.48 | 11.52 | 19.57 | 31.98 |
| RE-GCN | 19.64 | 12.42 | 20.90 | 33.69 |
| CEN | 20.18 | 12.84 | 21.51 | 34.10 |
| DaeMon | 20.73 | 13.65 | 22.53 | 34.23 |
| TiPNN | 21.17 | 14.03 | 22.98 | 34.76 |
| **DiMNet (SOTA)** | **21.93** | **14.03** | **23.57** | **37.49** |

## ICEWS14

| Method | MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|
| DaeMon | 40.68 | 31.53 | 45.58 | 56.73 |
| RE-GCN | 41.78 | 31.58 | 46.65 | 61.51 |
| TiPNN | 41.79 | 32.76 | 46.92 | 58.80 |
| CEN | 42.17 | 32.10 | 47.59 | 61.43 |
| **DiMNet (SOTA)** | **45.72** | **34.41** | **51.37** | **67.99** |

## WIKI (from the DaeMon paper table)

| Method | MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|
| CyGNet | 33.89 | 29.06 | 36.10 | 41.86 |
| TANGO-DistMult | 51.15 | 49.66 | 52.16 | 53.35 |
| xERTE | 71.14 | 68.05 | 76.11 | 79.01 |
| TITer | 75.50 | 72.96 | 77.49 | 79.02 |
| RE-GCN | 77.55 | 73.75 | 80.38 | 83.68 |
| **DaeMon** | **82.38** | **78.26** | **86.03** | **88.01** |

## YAGO (from the DaeMon paper table)

| Method | MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|
| CyGNet | 52.07 | 45.36 | 56.12 | 63.77 |
| RE-NET | 58.02 | 53.06 | 61.08 | 66.29 |
| RE-GCN | 84.12 | 80.76 | 86.30 | 89.98 |
| xERTE | 84.19 | 80.09 | 88.02 | 89.78 |
| TITer | 87.47 | 84.89 | 89.96 | 90.27 |
| **DaeMon** | **91.59** | **90.03** | **93.00** | **93.34** |

Note the ceiling: DaeMon's YAGO H@10 is 93.34, not ~100. Any run of ours
reporting H@10 near 100 on YAGO is measuring something wrong, not winning.

## Different protocol — do not mix

GAttNHP (https://arxiv.org/html/2607.14733v1) reports **raw**:
ICEWS18 MRR 38.63 / H@1 28.25, GDELT 26.43 / 17.15, WIKI 51.28 / 44.45,
YAGO 51.30 / 40.17. These are higher than the filtered numbers above on
ICEWS18/GDELT and lower on WIKI/YAGO; they cannot be compared to either.
`train_kairos.py` prints a `raw` row so a comparison against GAttNHP is
possible on its own terms.

## Our own results so far

| Run | Dataset | Protocol | MRR | H@1 | Note |
|---|---|---|---|---|---|
| TREA | ICEWS18 | time-aware filtered | 31.5 | 21.8 | below DaeMon and DiMNet |
| TREA | GDELT | time-aware filtered | 19.48 | 12.21 | below RE-GCN |
| AURORA-v3 (copy-only) | YAGO | time-aware filtered | 88.82 | 83.13 | **invalid** — optimistic ties |
| NHC (prototype) | YAGO | time-aware filtered | 92.05 | 91.11 | **invalid** — optimistic ties |

Neither YAGO number is usable. Both were produced with a rank of
`1 + #(strictly better)`, which places the target ahead of every entity that
ties with it. A copy-style model gives thousands of entities the same score,
so the target is credited with rank 1 for free. The signature is
AURORA-v3's YAGO H@10 = 99.93 against DaeMon's 93.34.

`ranks_of()` in train_kairos.py now resolves ties to their average position,
`1 + #(better) + (#(tied) - 1)/2`. Every number produced before that fix has
to be regenerated before it means anything.

## What has to be beaten

| Dataset | DaeMon MRR / H@1 | Current SOTA | Source |
|---|---|---|---|
| ICEWS18 | 31.85 / 22.67 | DiMNet 34.13 / 23.29 | DiMNet table |
| GDELT | 20.73 / 13.65 | DiMNet 21.93 / 14.03 | DiMNet table |
| WIKI | 82.38 / 78.26 | DaeMon (best listed) | DaeMon table |
| YAGO | 91.59 / 90.03 | DaeMon (best listed) | DaeMon table |
