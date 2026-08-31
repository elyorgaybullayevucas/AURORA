# TKG forecasting: 15 papers, their numbers, and where we actually stand

Compiled 2026-08-31. Every entry is a real paper with a working link. Numbers
are transcribed from the paper named in the "source" column — **not**
renormalised, and **not** rerun by us.

Read the protocol warning at the bottom before comparing anything across rows.

---

## The papers

| # | Paper | Venue / year | Link | What is new in it |
|---|---|---|---|---|
| 1 | RE-GCN — Temporal KG Reasoning Based on Evolutional Representation Learning | SIGIR 2021 | [arXiv:2104.10353](https://arxiv.org/abs/2104.10353) | R-GCN per snapshot + GRU across snapshots. The backbone everything else builds on. Its `utils.py` is the de-facto evaluation reference. |
| 2 | CyGNet — Learning from History | AAAI 2021 | [arXiv:2012.08492](https://arxiv.org/abs/2012.08492) | Copy-generation: an explicit copy vector of past `(s,r,·)` frequencies. The first pure recurrence mechanism. |
| 3 | xERTE — Explainable Subgraph Reasoning | ICLR 2021 | [arXiv:2012.15537](https://arxiv.org/abs/2012.15537) | Expands a query-centred subgraph with attention; interpretable by construction. |
| 4 | TITer — Time-aware Path-based Reinforcement Learning | EMNLP 2021 | [arXiv:2109.04101](https://arxiv.org/abs/2109.04101) | RL agent walks a temporal path to the answer. |
| 5 | CEN — Complex Evolutional Pattern Learning | ACL 2022 | [aclanthology 2022.acl-short.32](https://aclanthology.org/2022.acl-short.32/) | Length-aware curriculum over evolutional patterns of different spans. |
| 6 | TiRGN — Time-guided Recurrent Graph Network | IJCAI 2022 | [ijcai.org/proceedings/2022/299](https://www.ijcai.org/proceedings/2022/299) | Local recurrent encoder + global history vector, explicitly combined. |
| 7 | CENET — Contrastive Historical / Non-historical | AAAI 2023 | [arXiv:2211.10904](https://arxiv.org/abs/2211.10904) | Learns an oracle deciding whether the answer is in the history at all, trained contrastively. Directly targets our `no_history` stratum. |
| 8 | DaeMon — Adaptive Path-Memory Network | IJCAI 2023 | [arXiv:2304.12604](https://arxiv.org/abs/2304.12604) | Query-conditioned path memory, **no entity embeddings**, so unseen candidates are still scored. Current best published on YAGO/WIKI. |
| 9 | Gastinger et al. — Comparing Apples and Oranges? | ECML PKDD 2023 | [Springer](https://link.springer.com/chapter/10.1007/978-3-031-43418-1_32) | Shows the evaluation setting alone reverses rankings between methods. Not a model — a warning. |
| 10 | LogCL — Local-Global History-aware Contrastive Learning | **ICDE 2024** | [arXiv:2312.01601](https://arxiv.org/abs/2312.01601) | Entity-aware attention over local *and* global history + four query-contrast patterns. **Highest ICEWS18 number we found from a refereed venue.** |
| 11 | TiPNN — Temporal Inductive Path Neural Network | Artif. Intell. 2024 | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0004370224000213) | Query-aware "history temporal graph"; inductive over entities. |
| 12 | Gastinger et al. — History repeats Itself (recurrency baseline) | **IJCAI 2024** | [arXiv:2404.16726](https://arxiv.org/abs/2404.16726) | A *non-learned* baseline: strict + relaxed recurrency, one blend parameter α. Beats most published models on GDELT and YAGO. The premise of our paper. |
| 13 | DiMNet — Disentangled Multi-span Evolutionary Network | 2025 | [arXiv:2505.14020](https://arxiv.org/abs/2505.14020) | Cross-time message passing at equal hop depth + active/stable disentanglement. **Our structural backbone is taken from here.** |
| 14 | CountTRuCoLa — Rule Confidence Learning | 2025 | [arXiv:2509.09474](https://arxiv.org/abs/2509.09474) | Learns per-rule confidence functions combining recency **additively** with frequency. Fully interpretable. Closest published work to our diagnosis. |
| 15 | CognTKE — Cognitive Temporal Knowledge Extrapolation | **AAAI 2025** | [arXiv:2412.16557](https://arxiv.org/abs/2412.16557) | Dual-process: System-1 one-hop global + System-2 multi-hop local, over a temporal cognitive relation digraph. |

### Three more that matter, all 2026 preprints (unrefereed)

| # | Paper | Link | Why it matters to us |
|---|---|---|---|
| 16 | Towards Better Evolution Modeling for TKGs | [arXiv:2602.08353](https://arxiv.org/abs/2602.08353) | Measures a **shortcut**: pure co-occurrence counting, no temporal information at all, lands ~23% behind SOTA on YAGO/WIKI and only **15% behind on GDELT**. Argues the standard benchmarks are partly broken and ships four bias-corrected replacements. |
| 17 | AdaTKG — Adaptive Memory | [arXiv:2605.07121](https://arxiv.org/abs/2605.07121) | Per-entity EMA memory with **one shared learnable scalar**, updated online at inference. +23.8% MRR on ICEWS18 *for emerging entities* — our worst stratum. |
| 18 | CID-TKG / CHE-TKG | [arXiv:2604.09600](https://arxiv.org/abs/2604.09600), [arXiv:2605.04652](https://arxiv.org/abs/2605.04652) | Separate "historical invariance" and "evolutionary dynamics" graphs with contrastive alignment. Report ICEWS18 38.9 / GDELT 27.4 — above everything else, ours included. |

Also seen and judged not worth pursuing: EST ([arXiv:2602.12389](https://arxiv.org/abs/2602.12389)) reports ICEWS18 MRR 46.6 and GDELT 41.2, which is more than ten points above every other number in the literature *including its own baselines* — not credible as a comparable figure. HALO ([arXiv:2505.07509](https://arxiv.org/abs/2505.07509)) filters outdated facts by a half-life decay, which is monotone and therefore sits inside our Proposition 1. DynaGen ([arXiv:2512.12669](https://arxiv.org/abs/2512.12669)) applies conditional diffusion; heavy, unclear payoff.

---

## Results, grouped by source table

**Do not compare across groups.** Each group is internally consistent because
every row was produced by one paper's own harness.

### Group A — the DiMNet table (arXiv:2505.14020), time-aware filtered

| Method | ICEWS18 MRR | H@1 | GDELT MRR | H@1 |
|---|---|---|---|---|
| CyGNet | 24.93 | 15.90 | 18.48 | 11.52 |
| RE-GCN | 30.58 | 21.01 | 19.64 | 12.42 |
| xERTE | 29.31 | 21.03 | 18.09 | 12.30 |
| TITer | 29.98 | 22.05 | 15.46 | 10.98 |
| CEN | 30.84 | 21.23 | 20.18 | 12.84 |
| DaeMon | 31.85 | 22.67 | 20.73 | 13.65 |
| TiPNN | 32.17 | 22.74 | 21.17 | 14.03 |
| **DiMNet** | **34.13** | **23.29** | **21.93** | **14.03** |
| **CADENCE (ours)** | **34.59** | **24.78** | **26.34**† | **17.59**† |

† single seed, still running.

This is the table our paper currently uses. **It omits LogCL, TiRGN, CENET and
CountTRuCoLa.** DiMNet's own "state of the art" claim is therefore made against
an incomplete field, and we inherited that gap.

### Group B — the DaeMon table, time-aware filtered

| Method | WIKI MRR | H@1 | YAGO MRR | H@1 |
|---|---|---|---|---|
| CyGNet | 33.89 | 29.06 | — | — |
| xERTE | 71.14 | 68.05 | 84.19 | 80.09 |
| TITer | 75.50 | 72.96 | 87.47 | 84.89 |
| RE-GCN | 77.55 | 73.75 | 82.30 | 78.83 |
| **DaeMon** | **82.38** | **78.26** | **91.59** | **90.03** |
| **CADENCE (ours)** | **82.94** | **79.84** | **91.64** | **90.32** |

### Group C — each paper's own reported number (mixed harnesses)

| Method | ICEWS18 MRR | GDELT MRR | WIKI MRR | YAGO MRR | Source |
|---|---|---|---|---|---|
| Recurrency baseline | 28.7 | **24.5** | 81.5 | 90.9 | IJCAI 2024 |
| LogCL | **35.67** | 23.75 | — | — | ICDE 2024 |
| CountTRuCoLa | 32.8 | 23.8 | 82.7 | 90.9 | arXiv 2025 |
| CognTKE | 35.24 | — | 83.21 | — | AAAI 2025 |
| CHE-TKG | 38.77 | 27.38 | — | — | arXiv 2026 |
| CID-TKG | 38.88 | 27.41 | — | — | arXiv 2026 |
| **CADENCE (ours)** | 34.59 | 26.34† | 82.94 | 91.64 | ours |

---

## What this means for us — read this part

**1. Our "state of the art on all four benchmarks" claim does not survive.**
LogCL (ICDE 2024, refereed) reports ICEWS18 MRR **35.67 / H@1 24.53** against
our **34.59 / 24.78**. We are behind on MRR there. CognTKE (AAAI 2025) reports
WIKI 83.21 against our 82.94. The 2026 preprints report ICEWS18 ~38.9 and
GDELT ~27.4, above us on both. The claim has to be narrowed or the comparison
has to be made properly.

**2. The gap is probably protocol, and that is not a defence — it is the
problem.** DiMNet claims SOTA at ICEWS18 34.13 while LogCL had already
published 35.67 the year before. Both say "time-aware filtered". They cannot
both be measuring the same thing. This is exactly what Gastinger et al. (#9)
documented, and a reviewer at a top venue will not accept "their number is
probably from a different setup" as an answer. The only real answers are to
rerun one or two of them in our harness, or to stop competing on the pooled
metric.

**3. Our headline margins were never the contribution anyway.** YAGO 91.64 vs
DaeMon 91.59 is **+0.05**, against a benchmark where paper #16 shows a
temporal-information-free co-occurrence counter gets within ~23%, and where
paper #12's non-learned baseline already scores 90.9. That margin means
nothing, and claiming it invites the reviewer to check the rest.

**4. But the field has moved toward exactly what we measured.** Papers #9, #12
and #16 are all saying the same thing: pooled metrics on these benchmarks are
dominated by a repetition shortcut, and progress claims built on them are
unreliable. Our monotone-blocked stratum is, by construction, the subset where
that shortcut cannot work. We have the diagnostic, the stratification, and a
controlled ablation (+1.83 to +8.11 MRR from the functional form alone with
everything else held fixed) that no leaderboard row can give.

**5. The one mechanism worth importing.** Papers #17 and #16-adjacent converge
independently on a **persistent per-entity memory** that survives past the
history window — AdaTKG does it with a single learnable EMA scalar, and reports
its gain specifically on *emerging entities*. Our largest failure pool is the
`no_history` stratum: 50.1% of ICEWS18 at 5.65 H@1, 40.4% of GDELT at 1.63. Our
history window is 10 snapshots and everything older is discarded. This is cheap
to add and aimed exactly at our weakness. CENET (#7) targets the same stratum
by a different route and should be read before implementing.

Not worth importing: CognTKE's multi-hop path reasoning (we already tested a
path branch — negative result, see `FINDINGS.md`), HALO's half-life filter
(monotone, inside Proposition 1), DynaGen's diffusion (cost).
