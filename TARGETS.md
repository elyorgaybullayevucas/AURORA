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

## WIKI / YAGO

**Not yet verified.** The DiMNet table does not cover them. Numbers quoted
earlier in this project (CENET ~79 H@1 on YAGO) have not been checked
against a source, and WIKI/YAGO are exactly where protocol confusion is
worst — published values differ by 30+ points depending on the setting.
Do not put a WIKI/YAGO comparison in a paper until the source and protocol
of each baseline number are confirmed.

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
| AURORA-v3 (copy-only) | YAGO | time-aware filtered | 88.82 | 83.13 | 41 parameters; baseline unverified |
| NHC (prototype) | YAGO | time-aware filtered | 92.05 | 91.11 | degenerate support, avg\|S\|=1.3 |

The two YAGO numbers are not yet meaningful as claims: the baseline they
were compared against is unverified, and the NHC run had a broken candidate
set. They are recorded here so they are not mistaken for results.
