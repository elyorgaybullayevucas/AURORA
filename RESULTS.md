# KAIROS — all results

Every number ×100. Single seed, `seed=42`. Ranks resolve ties to their
average position. Baselines are quoted from the DaeMon table (YAGO, WIKI)
and the DiMNet table (ICEWS18, GDELT); both report **time-aware filtered**,
which is the protocol of the main table below.

Variants:

| name | what it is |
|---|---|
| `full` | structural + phase-conditioned recurrence |
| `full+path` | the above plus the query-conditioned path branch |
| `monotone-kernel` | recurrence restricted to `f(count)·g(Δt)`, `g` non-increasing — the published family |
| `structural-only` | recurrence removed entirely |

---

## 1. Main table — time-aware filtered

### YAGO

| method | MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|
| RE-GCN | 84.12 | 80.76 | 86.30 | 89.98 |
| xERTE | 84.19 | 80.09 | 88.02 | 89.78 |
| TITer | 87.47 | 84.89 | 89.96 | 90.27 |
| DaeMon *(SOTA)* | 91.59 | 90.03 | **93.00** | **93.34** |
| **KAIROS full** | **91.66** | **90.38** | 92.88 | 93.08 |
| KAIROS full+path | 91.46 | 89.84 | 92.96 | 93.26 |
| KAIROS monotone-kernel | 89.81 | 87.05 | 92.54 | 93.01 |
| KAIROS structural-only | 46.46 | 37.36 | 51.46 | 62.87 |

### WIKI

| method | MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|
| TITer | 75.50 | 72.96 | 77.49 | 79.02 |
| RE-GCN | 77.55 | 73.75 | 80.38 | 83.68 |
| DaeMon *(SOTA)* | 82.38 | 78.26 | **86.03** | **88.01** |
| **KAIROS full** | **82.97** | **79.88** | 85.95 | 87.01 |
| KAIROS monotone-kernel | 74.83 | 68.34 | 79.84 | 86.08 |

### ICEWS18

| method | MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|
| RE-GCN | 30.58 | 21.01 | 34.34 | 48.75 |
| DaeMon | 31.85 | 22.67 | 35.92 | 49.80 |
| TiPNN | 32.17 | 22.74 | 36.24 | 50.72 |
| DiMNet *(SOTA)* | 34.13 | 23.29 | 38.42 | **55.80** |
| **KAIROS full** | **34.45** | **24.78** | **38.96** | 53.19 |
| KAIROS monotone-kernel | 29.20 | 19.73 | 32.91 | 47.73 |

### GDELT

| method | MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|
| RE-GCN | 19.64 | 12.42 | 20.90 | 33.69 |
| DaeMon | 20.73 | 13.65 | 22.53 | 34.23 |
| TiPNN | 21.17 | 14.03 | 22.98 | 34.76 |
| DiMNet *(SOTA)* | 21.93 | 14.03 | 23.57 | 37.49 |
| **KAIROS full** | **26.34** | **17.59** | **29.07** | **43.44** |
| KAIROS monotone-kernel | 21.77 | 13.58 | 23.97 | 37.95 |

### Margin over SOTA

| dataset | Δ MRR | Δ H@1 | Δ H@3 | Δ H@10 |
|---|---|---|---|---|
| YAGO | +0.07 | +0.35 | −0.12 | −0.26 |
| WIKI | +0.59 | +1.62 | −0.08 | −1.00 |
| ICEWS18 | +0.32 | +1.49 | +0.54 | −2.61 |
| GDELT | +4.41 | +3.56 | +5.50 | +5.95 |

**A pattern that has to be reported.** On three of four datasets we win MRR
and H@1 and lose H@10. The model is more precise than the baselines and less
complete. That is consistent with everything else measured here: the
recurrence branch is sharp on candidates it can see, and the structural
branch is weak on everything else (see §3). GDELT is the exception and wins
on every metric.

---

## 2. Other protocols

Published numbers are not comparable across protocols. GAttNHP reports raw;
RE-GCN / TiRGN / CENET / DaeMon / DiMNet report time-aware filtered.

| dataset | variant | raw MRR | raw H@1 | t-unaware MRR |
|---|---|---|---|---|
| YAGO | full | 65.47 | 52.82 | 92.91 |
| YAGO | full+path | 65.50 | 52.86 | 93.05 |
| YAGO | monotone-kernel | 64.71 | 51.86 | 92.78 |
| YAGO | structural-only | 40.91 | 30.75 | 47.29 |
| WIKI | full | 53.75 | 42.72 | 87.03 |
| WIKI | monotone-kernel | 48.87 | 36.74 | 86.70 |
| ICEWS18 | full | 32.62 | 22.39 | 53.12 |
| ICEWS18 | monotone-kernel | 27.85 | 18.01 | 44.11 |
| GDELT | full | 25.52 | 16.50 | 58.33 |
| GDELT | monotone-kernel | 21.20 | 12.89 | 49.62 |

For reference, GAttNHP under **raw**: ICEWS18 38.63 / 28.25,
GDELT 26.43 / 17.15, WIKI 51.28 / 44.45, YAGO 51.30 / 40.17. Under that
protocol we are ahead on WIKI and YAGO and behind on ICEWS18 and GDELT.

---

## 3. H@1 by stratum

Strata partition the test set and are defined from the data alone:

- `no_history` — the answer never appeared for this `(s, r)`; recurrence is
  silent by construction
- `blocked` — the answer has history but a distractor dominates it on both
  recency and count, so no `f(count)·g(Δt)` with `g` non-increasing can rank
  it first
- `clean` — the answer has history and dominates

| dataset | variant | blocked | clean | no_history |
|---|---|---|---|---|
| YAGO | full | 95.00 | 99.56 | 0.79 |
| YAGO | full+path | 94.18 | 99.04 | **1.68** |
| YAGO | monotone-kernel | 89.34 | 97.79 | 0.86 |
| YAGO | structural-only | 22.03 | 56.49 | 0.45 |
| WIKI | full | 89.41 | 94.83 | 1.22 |
| WIKI | monotone-kernel | 71.10 | 88.22 | 0.32 |
| ICEWS18 | full | 14.26 | 77.70 | 5.65 |
| ICEWS18 | monotone-kernel | 11.64 | 59.51 | 5.45 |
| GDELT | full | 5.07 | 69.52 | 1.63 |
| GDELT | monotone-kernel | 3.84 | 53.35 | 1.49 |

Stratum sizes:

| dataset | blocked | clean | no_history |
|---|---|---|---|
| YAGO | 43.7 % | 49.0 % | 7.3 % |
| WIKI | 48.6 % | 38.3 % | 13.2 % |
| ICEWS18 | 26.6 % | 23.4 % | 50.1 % |
| GDELT | 38.1 % | 21.6 % | 40.4 % |

---

## 4. The claim test

What the phase basis is worth, H@1, `full` minus `monotone-kernel`:

| dataset | blocked | clean | ratio | phase median | verdict |
|---|---|---|---|---|---|
| YAGO | **+5.66** | +1.77 | 3.2× | 1.000 | consistent |
| WIKI | **+18.31** | +6.61 | 2.8× | 1.000 | consistent |
| ICEWS18 | +2.63 | **+18.19** | 0.14× | 0.312 | inverted |
| GDELT | +1.23 | **+16.17** | 0.08× | 0.407 | inverted |

The gain concentrates on `blocked` exactly on the two datasets whose median
phase is 1.000 — where facts recur at their own mean inter-arrival gap — and
inverts on the two that are recency-dominated. The claim is
**dataset-conditional**, and predicts its own scope from a statistic
measurable before training.

Two caveats stand.

`--phase_off` changes two things at once: it removes the non-monotone basis
and cuts the branch from 16 features to `(count, Δt)`. The ICEWS18 and GDELT
gains on `clean` are as easily explained by feature access.
`--phase_feat_off` holds everything else fixed and blanks only the phase
entries; those runs are not finished.

WIKI's monotone baseline peaked at epoch 1 and early-stopped, so its +11.54
overall reflects a baseline that failed to train, not one that trained and
lost. It needs a rerun before use.

---

## 5. What is not yet established

- **Seeds.** Every number above is one seed. YAGO (+0.07 MRR) and ICEWS18
  (+0.32) are inside the range a seed change can cover. 3–5 seeds with mean
  and standard deviation are required before any of these are stated as a
  result. GDELT (+4.41) is the only margin that is comfortable.
- **H@10.** We lose it on three of four datasets, and no fix has been tested.
- **The path branch.** A null result on YAGO: it doubled `no_history`, which
  is what it was built for, and still cost more elsewhere than it gained,
  because that stratum is only 7.3 % of YAGO. It was tested on the wrong
  dataset — the stratum is 50.1 % of ICEWS18 and 40.4 % of GDELT. It is also
  starved (1.68 against DaeMon's 91.59 overall from path reasoning alone);
  `path_aux` deep supervision was added afterwards and has not been run.
