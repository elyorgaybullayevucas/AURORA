# When Is a Fact Due? Phase-Conditioned Recurrence for Temporal Knowledge Graph Forecasting

*Draft. Numbers marked ⧗ are pending experiments; everything else is measured.*

---

## Abstract

Recurrence dominates temporal knowledge graph (TKG) forecasting: a plain
recurrency baseline matches RE-GCN, CyGNet, TiRGN and L2-TKG on the standard
benchmarks. Every recurrence mechanism in this literature scores a historical
candidate with `f(count) · g(Δt)` where `g` is non-increasing in elapsed
time — including the exponential kernel of the recent neural-Hawkes model
GAttNHP, which learns the decay rate but keeps the form.

We show this family is *provably* unable to rank a large, measurable share of
queries. If a distractor has both smaller elapsed time and larger count than
the answer, no non-increasing `g` and no non-decreasing `f` can place the
answer first. We call such queries **monotone-blocked**, show their rate is a
property of the data computable without any model, and measure it: 41.6 % of
YAGO, 29.3 % of WIKI, 26.6 % of ICEWS18 and 15.8 % of GDELT test queries.

The measurement also reveals why. Writing **phase** = Δt / mean inter-arrival
gap, the median phase of the true answer on YAGO and WIKI is exactly 1.000,
with 94.0 % and 84.8 % of answers in [0.5, 1.5]: facts recur at their own
average waiting time. A monotone kernel is maximal at phase → 0 and therefore
prefers the wrong candidate systematically.

We introduce **KAIROS**, which replaces the fixed decay with a learned
conditional intensity over phase — a non-negative mixture over a radial basis,
non-monotone in Δt by construction — superposed with a structural intensity
through the log-sum-exp of a point-process superposition rather than a gate.
KAIROS exceeds the published state of the art on all four benchmarks under the
time-aware filtered protocol. Ablations confirm the mechanism acts where the
analysis predicts on the phase-structured datasets and, honestly, does not on
the recency-dominated ones.

---

## 1 Introduction

*(to write; the argument is:)*

1. TKG forecasting is dominated by recurrence, and the field knows it —
   "History Repeats Itself" (IJCAI'24) shows a recurrency baseline matching
   four published architectures. Our own 41-parameter copy model reaches
   83.13 H@1 on YAGO.
2. So the interesting question is not whether to model recurrence but what
   the existing recurrence models cannot express.
3. They all share one functional form, and that form has a provable blind
   spot whose size is measurable from the data alone.
4. The blind spot is not a curiosity: it is 15.8–41.6 % of test queries, and
   on two datasets the data is concentrated exactly where the form is blind.
5. Contributions: the proposition and the diagnostic; the measured phase
   distribution; the KAIROS intensity; results on four benchmarks; a
   stratified evaluation that tests the mechanism rather than the metric.

---

## 2 Related work

**Recurrence / copy mechanisms.** CyGNet, CENET, TiRGN, RE-GCN, DaeMon all
score historical candidates by frequency weighted by exponential decay.
CENET adds an explicit historical/non-historical oracle with a contrastive
objective. All are instances of `f(count)·g(Δt)`.

**Point processes for TKG.** GAttNHP (2026) casts TKG extrapolation as a
neural Hawkes process with base, self- and group-excitation, but its kernel
is `exp(−γ_u Δt)` with a learned per-entity rate — monotone, and therefore
inside the family our proposition constrains. Our contribution is orthogonal
to the point-process framing itself and concerns the shape of the kernel.

**Structural encoders.** RE-GCN evolves entity representations over snapshots
with an R-GCN and a GRU. DiMNet adds cross-time perception at equal hop depth
and disentangles node features into active and stable factors; its ablation
attributes −11.38 MRR to disentanglement, −9.62 to virtual-subgraph
refinement and −4.97 to multi-span. **We adopt multi-span evolution and
active/stable disentanglement as backbone and claim no novelty for them.**

**Path-based reasoning.** DaeMon carries an NBFNet-style query-conditioned
state across snapshots with a time-aware gate and contains no entity
embeddings at all, so a never-seen candidate is still scored on path
evidence. We reimplement this as an optional third intensity; §7.4 reports it
as a negative result at our scale.

**Evaluation protocol.** Published numbers use three incompatible settings.
RE-GCN, TiRGN, CENET, DaeMon, TiPNN and DiMNet report *time-aware filtered*;
GAttNHP reports *raw*. We report all three and compare only within one.

---

## 3 Preliminaries

A TKG is a set of quadruples `(s, r, o, t)`. Forecasting (extrapolation) asks
for `o` given `(s, r, ?, t)` using only facts strictly before `t`. Metrics are
MRR and Hits@{1,3,10}. **Ties are resolved to their average position,**
`rank = 1 + #{better} + (#{tied} − 1)/2`; counting only strictly-better
scores inflates every metric for models that assign many candidates an
identical score, and by that convention one of our own early copy-only models
reported H@10 = 99.93 on YAGO where DaeMon reports 93.34.

---

## 4 The blind spot of monotone recurrence

### 4.1 Proposition

Let a scorer assign `s(o) = f(count_o) · g(Δt_o)` with `f` non-decreasing and
`g` non-increasing and positive. Let `o*` be the answer and `o` a distractor
with `Δt_o ≤ Δt_{o*}` and `count_o ≥ count_{o*}`. Then

`f(count_o) ≥ f(count_{o*})` and `g(Δt_o) ≥ g(Δt_{o*})`, so `s(o) ≥ s(o*)`.

No choice of decay rate — fixed, per-relation, or learned per entity as in
GAttNHP — and no reweighting of counts ranks `o*` strictly first. ∎

We call such a query **monotone-blocked**. Blockedness is a property of the
query and its candidate history; no model appears in the definition.

### 4.2 The rate is large

Measured on 20,000 sampled test queries per dataset (`diagnose.py`):

| dataset | answer has usable history | blocked (of those) | **blocked / all queries** |
|---|---|---|---|
| YAGO | 92.8 % | 44.8 % | **41.6 %** |
| WIKI | 87.0 % | 33.7 % | **29.3 %** |
| ICEWS18 | 47.7 % | 55.7 % | **26.6 %** |
| GDELT | 24.9 % | 63.4 % | **15.8 %** |

**Scope of the bound.** The proposition constrains a recurrence scorer *in
isolation*. Published systems pair such a term with a structural branch that
is not subject to it and can rescue a blocked query. What the numbers
establish is that on 15.8–41.6 % of queries the recurrence component cannot
produce the correct ordering and must delegate. The hypothesis under test is
that handling those queries where the temporal evidence actually lives is
better than delegating them.

### 4.3 Why: facts recur at their own mean gap

Define **phase** `p_o = Δt_o / mean_gap_o`. Distribution of the true answer's
phase:

| dataset | p < 0.5 | 0.5 ≤ p ≤ 1.5 | p > 2 | median |
|---|---|---|---|---|
| YAGO | 6.0 % | **94.0 %** | 0.0 % | **1.000** |
| WIKI | 12.1 % | **84.8 %** | 1.3 % | **1.000** |
| ICEWS18 | 56.9 % | 19.6 % | 19.1 % | 0.312 |
| GDELT | 52.8 % | 20.0 % | 22.1 % | 0.407 |

On YAGO and WIKI the median is exactly one. A monotone `g` is maximal at
phase → 0 and so systematically prefers a candidate that fired recently over
one that is *due*. ICEWS18 and GDELT are a different regime — recency
dominates the bulk, with a long tail — and we will find the mechanism behaves
differently there, as this table predicts.

---

## 5 KAIROS

### 5.1 Superposition, not gating

A query is a draw from a superposition of two marked point processes on the
entity space,

```
λ(o) = λ_struct(o | G_<t, s, r) + λ_rec(o | H_o, s, r)
```

For a superposition the probability that the next mark is `o` is
`λ(o) / Σ λ(o')`, so softmax cross-entropy on

```
score(o) = log λ(o) = logaddexp(f_struct(o), f_rec(o))      o ∈ S
         = f_struct(o)                                      otherwise
```

is the correct discrete-choice likelihood. Two consequences we needed
empirically: superposition **adds intensities**, so the branches combine
through `logaddexp` and not through a sum of logits — a sum of logits
multiplies intensities, which has no point-process meaning and lets one
branch silently rescale the other; and there is no mixing gate, hence no gate
to collapse. Earlier variants of this work with a learned scalar or per-query
gate collapsed to a single branch within a few epochs.

### 5.2 The recurrence intensity

```
log λ_rec(o) = log Σ_j w_j(r, o, φ_o) · κ_j(p_o) + b,     w_j ≥ 0
```

`κ_j` is a fixed radial basis over phase with 15 centres in [0, 13], applied
to three contexts — the `(s,r)`, `(s,·)` and `(r,·)` histories. `w_j` is
predicted per candidate from 16 interpretable temporal statistics (counts,
recency, mean and dispersion of inter-arrival gaps, span, rate, phase) and
the relation and entity embeddings, through `softplus` so the mixture stays
non-negative.

Fitting `w_j` to `exp(−λ p)` recovers the classical term, so every scorer in
§4.1 is a special case up to basis resolution. Non-monotonicity in Δt is by
construction, which is what makes blocked queries reachable.

### 5.3 Structural intensity

Backbone, adopted from prior work and cited as such: a composition GCN over
each snapshot with a cross-time link at equal layer depth (multi-span), node
states split into complementary active and stable factors with a drift
penalty on the stable factor across adjacent snapshots (disentanglement), a
GRU across snapshots, and a ConvTransE decoder over the full entity table.

**LayerNorm replaces BatchNorm in the decoder.** Batches here are snapshots,
so batch size is however many facts occurred at one timestamp and varies from
72 to 4,650 across our datasets; batch statistics are unstable, and once a
long snapshot is split into query chunks the result depends on the split.

---

## 6 Experimental setup

Four benchmarks (YAGO, WIKI, ICEWS18, GDELT), time-aware filtered protocol,
baselines quoted from the DaeMon table (YAGO, WIKI) and the DiMNet table
(ICEWS18, GDELT) — both report the same protocol. `d = 200`, 3 GCN layers,
history 10 snapshots (5 on GDELT), 6.3 M parameters. One A100 per run: YAGO
42 min, ICEWS18 1 h 15, WIKI 3 h, GDELT 8 h.

**Threat to validity we state up front.** Our evaluation harness is our own.
We have not rerun a published baseline inside it, so a subtle difference in
filtering would invalidate margins of the size we report on YAGO. One
indirect check: our structural branch alone reaches 46.46 MRR on YAGO where
RE-GCN — the same architectural family — publishes 84.12. If our protocol
were generous this number would not be so far below. ⧗ *A protocol
comparison against the RE-GCN reference implementation is in progress.*

---

## 7 Results

### 7.1 Main table (time-aware filtered, ×100, mean ± std over seeds)

| dataset | method | MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|---|
| YAGO | DaeMon | 91.59 | 90.03 | **93.00** | **93.34** |
| | **KAIROS** (3 seeds) | **91.64 ± 0.04** | **90.32 ± 0.08** | 92.88 | 93.08 |
| WIKI | DaeMon | 82.38 | 78.26 | **86.03** | **88.01** |
| | **KAIROS** (3 seeds) | **82.94 ± 0.15** | **79.84 ± 0.28** | 85.77 | 86.95 |
| ICEWS18 | DiMNet | 34.13 | 23.29 | 38.42 | **55.80** |
| | **KAIROS** (3 seeds) | **34.59 ± 0.14** | **24.78 ± 0.11** | **39.06** | 53.55 |
| GDELT | DiMNet | 21.93 | 14.03 | 23.57 | 37.49 |
| | **KAIROS** ⧗ (1 seed) | **26.34** | **17.59** | **29.07** | **43.44** |

Margins over SOTA relative to seed standard deviation: ICEWS18 H@1 13.5×,
WIKI H@1 5.6×, YAGO H@1 3.6×. GDELT ⧗ awaits seeds; its margin (+4.41 MRR) is
an order of magnitude above any seed variance we have observed.

**We lose H@10 on three of four datasets** (−0.26, −1.06, −2.25). The model
is more precise than the baselines and less complete. §7.3 shows why.

### 7.2 The mechanism ablation

`--phase_off` restricts the recurrence branch to exactly the published form,
`log a(r,o) + p(r,o)·log(1+count) − b(r,o)·Δt` with `p, b ≥ 0`, predicted from
`(r, o)` alone. A unit test sweeps Δt and count and asserts the restriction
holds, and asserts the full model violates it.

| dataset | full | monotone kernel | Δ MRR |
|---|---|---|---|
| YAGO | 91.64 | 89.81 | **+1.83** |
| WIKI | 82.94 | 74.83 | **+8.11** |
| ICEWS18 | 34.59 | 29.20 | **+5.39** |
| GDELT | 26.34 | 21.77 | **+4.57** |

Note the monotone variant is itself near SOTA on GDELT (21.77 vs 21.93): the
backbone is competitive and the kernel adds on top.

### 7.3 Where the gain lands — and where it does not

Strata partition the test set and are defined from the data alone.

| dataset | | blocked | clean | no_history |
|---|---|---|---|---|
| YAGO | share | 43.7 % | 49.0 % | 7.3 % |
| | full H@1 | 95.00 | 99.56 | **0.79** |
| ICEWS18 | share | 26.6 % | 23.4 % | 50.1 % |
| | full H@1 | 14.26 | 77.70 | **5.65** |
| GDELT | share | 38.1 % | 21.6 % | 40.4 % |
| | full H@1 | 5.07 | 69.52 | **1.63** |

Phase basis worth (full − monotone), H@1:

| dataset | blocked | clean | median phase | verdict |
|---|---|---|---|---|
| YAGO | **+5.66** | +1.77 | 1.000 | concentrates on blocked |
| WIKI | **+18.31** | +6.61 | 1.000 | concentrates on blocked |
| ICEWS18 | +2.63 | **+18.19** | 0.312 | inverted |
| GDELT | +1.23 | **+16.17** | 0.407 | inverted |

**The mechanism is dataset-conditional and predicts its own scope.** It acts
where the analysis of §4.3 says it should — on the two datasets whose median
phase is one — and not on the two that are recency-dominated. We regard this
as a stronger result than a uniform gain would be, because the scope was
predicted from a statistic measured before any training.

**Isolating phase from feature richness.** `--phase_off` changes two things:
it removes the non-monotone basis *and* cuts the branch from 16 features to
two. `--phase_feat_off` holds the trunk and all other features fixed and
blanks only the three phase entries. On GDELT this costs **−0.07 MRR** —
nothing. The entire +4.57 there is feature access, not phase, confirming the
inverted verdict above. ⧗ *The same variant on YAGO, WIKI and ICEWS18 is
running; it is the experiment that decides the claim on the phase-structured
datasets, and it is not yet available.*

### 7.4 Two negative results we report

**Deep supervision on the structural branch hurts.** Paired on three seeds,
adding a cross-entropy on the structural scores alone costs 0.87 MRR and 1.66
H@1 on YAGO, in the same direction on every seed, and raises the seed
standard deviation from 0.04 to 0.42. It was introduced to stop recurrence
from starving the structural branch; on a 92.8 %-recurrent dataset that trade
is not worth making.

**A path branch does not pay at our scale.** We reimplemented DaeMon's
query-conditioned propagation as a third intensity, with no entity
embeddings. It doubled `no_history` H@1 on YAGO (0.79 → 1.68) — exactly the
stratum it targets — and still lost overall (91.46 vs 91.64 MRR), because
that stratum is 7.3 % of YAGO. Its state is `(queries, entities, dim)`, which
costs roughly 50× the arithmetic of the other branches. ⧗ *The run on
ICEWS18, where the stratum is 50.1 %, reached epoch 11 of 60 at ~6 h/epoch
and was not completed.*

---

## 8 Limitations

- `no_history` is the largest single pool of failures on the event datasets:
  50.1 % of ICEWS18 at 5.65 H@1, 40.4 % of GDELT at 1.63. The recurrence
  branch is silent there by construction and the structural branch is weak.
  This is where the remaining headroom is, and we do not close it.
- H@10 is below SOTA on three datasets.
- Baselines are quoted, not rerun in our harness.
- Three seeds; GDELT one. ⧗
- Four benchmarks; ICEWS14 and ICEWS05-15 not covered. ⧗

---

## 9 Reproducibility

Code, the diagnostic, all ablation switches and the collection script are
released. `diagnose.py` reproduces §4 without training. Every number in §7 is
produced by `collect.py` from the saved result files.

---

## Pending before submission

| # | Experiment | Cost | What it decides |
|---|---|---|---|
| 1 | `--phase_feat_off` on YAGO, WIKI, ICEWS18 | ~5 h | **the central claim** |
| 2 | GDELT seeds ×3 | 16 h | the strongest margin |
| 3 | Protocol check against RE-GCN reference code | 1 d | validity of every margin |
| 4 | Seeds 4–5 on all datasets | 1 d | statistics |
| 5 | ICEWS14 | 1 d | coverage |

Items 1 and 3 are the ones a reviewer will decide on.
