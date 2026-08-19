# Measured properties of the benchmarks

Produced by `diagnose.py` on the full datasets (20,000 sampled queries per
split). These are properties of the DATA. No model is involved.

## Summary (test split)

| dataset | answer in candidate set | recurrent | monotone-blocked (of recurrent) | blocked / all queries | phase median |
|---|---|---|---|---|---|
| YAGO | 94.5 % | 92.8 % | 44.8 % | **41.6 %** | **1.000** |
| WIKI | 89.3 % | 87.0 % | 33.7 % | **29.3 %** | **1.000** |
| ICEWS18 | 52.7 % | 47.7 % | 55.7 % | **26.6 %** | 0.312 |
| GDELT | 26.2 % | 24.9 % | 63.4 % | **15.8 %** | 0.407 |

## Phase distribution of the true answer

phase = Δt / mean_gap, i.e. how far through its own typical waiting time the
fact is when it recurs.

| dataset | phase < 0.5 | 0.5 ≤ phase ≤ 1.5 | phase > 2 |
|---|---|---|---|
| YAGO | 6.0 % | **94.0 %** | 0.0 % |
| WIKI | 12.1 % | **84.8 %** | 1.3 % |
| ICEWS18 | 56.9 % | 19.6 % | 19.1 % |
| GDELT | 52.8 % | 20.0 % | 22.1 % |

## What this says

**1. The premise holds, and the margin is not small.**
Between 15.8 % and 41.6 % of all test queries are monotone-blocked: there
exists a distractor with `Δt ≤ Δt*` and `count ≥ count*`, so no scorer of
the form `f(count)·g(Δt)` with `g` non-increasing can rank the true answer
strictly first. That family is every published recurrence mechanism —
CyGNet, CENET, TiRGN, RE-GCN, DaeMon, and GAttNHP's learned-γ exponential.

**2. On YAGO and WIKI the median phase is exactly 1.000.**
This is the strongest single number here. Facts recur at almost exactly
their own average inter-arrival gap: 94.0 % of YAGO answers and 84.8 % of
WIKI answers fall in phase ∈ [0.5, 1.5]. A monotone decay is maximal at
phase → 0 and therefore systematically prefers the wrong candidate on these
datasets. The signal a phase kernel is built to read is not a hypothesis
here; it is the dominant mode of the data.

**3. ICEWS18 and GDELT are a different regime.**
Median phase 0.31 / 0.41, with mass both below 0.5 (57 % / 53 %) and above 2
(19 % / 22 %). Recency dominates the bulk, so a monotone kernel is a
reasonable first approximation — which is consistent with decay-based models
performing respectably there. But the distribution is spread, not
concentrated, and the blocked rate among recurrent queries is the highest of
all four datasets (55.7 % / 63.4 %).

**4. Recurrence coverage sets where the work must be done.**
Recurrent rate: YAGO 92.8 %, WIKI 87.0 %, ICEWS18 47.7 %, GDELT 24.9 %.
On GDELT three quarters of queries have no usable history for the answer at
all, so the structural branch has to carry that dataset and the recurrence
kernel can only move ~16 % of queries. Expect the contribution to show up
largest on YAGO and WIKI and smallest on GDELT.

Note also that support recall (94.5 / 89.3 / 52.7 / 26.2) tracks the
recurrent rate closely. The candidate set is not the bottleneck — those
queries genuinely have no history. Enlarging `max_support` or `rel_topk`
would not help.

## Caveat that must not be dropped

The blocked bound applies to a recurrence scorer **in isolation**. Published
models combine such a term with a structural/graph term, and the structural
term is not subject to the proposition — it can rescue a blocked query. So
these numbers are **not** an upper bound on RE-GCN, DaeMon or DiMNet as
whole systems.

What they do establish: on 15.8–41.6 % of queries the recurrence component
is structurally unable to produce the right ordering and hands the problem
to the structural branch. The claim under test is that handling those
queries in the recurrence branch directly — where the temporal evidence
actually lives — is better than delegating them. The stratified evaluation
in `train_kairos.py` (`blocked` vs `not_blocked` rows) is what tests it.
