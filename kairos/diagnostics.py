"""
Monotone-reachability diagnostics.

This is the measurement that has to come BEFORE any training, because it
decides whether the contribution exists at all.

BACKGROUND
Every recurrence / copy mechanism in the TKG forecasting literature scores a
historical candidate with something of the form

        s(o) = f(count_o) * g(dt_o),      g non-increasing

CyGNet, CENET, TiRGN, RE-GCN and DaeMon use g(dt) = exp(-lambda*dt) with a
fixed lambda. GAttNHP (2026), the most recent point-process model, uses
exp(-gamma_u * dt) with a learned per-entity gamma. All of them are
non-increasing in dt.

DEFINITION (monotone-blocked query)
A query (s, r, ?, t) with answer o* and candidate set S is monotone-blocked
if there exists a distractor o in S with

        dt_o <= dt_{o*}   and   count_o >= count_{o*}
        and (dt_o, count_o) != (dt_{o*}, count_{o*})

For any f non-decreasing in count and any non-increasing g, such a distractor
scores at least as high as the true answer. No choice of lambda, no learned
per-entity decay rate, and no reweighting of counts can rank o* strictly
first. The query is unreachable for the entire family.

WHAT THE NUMBER MEANS
The monotone-blocked rate is a property of the DATA, not of any model. It is
an upper bound on the Hits@1 that the whole family of monotone recurrence
scorers can achieve on the recurrent subset. If it is small, the proposed
mechanism cannot matter much and we should say so. If it is large, it
quantifies exactly how much headroom a non-monotone kernel opens up.

The same routine also reports the phase distribution of true answers, which
says whether recurrence in each dataset is actually phase-structured.
"""
import numpy as np
from collections import Counter


def analyse(index, quads, n_samples=20000, seed=0, max_support=256,
            verbose=True):
    """
    quads: (M,4) array of evaluation quadruples (s, r, o, t)
    index: TKGIndex built over the full dataset

    Returns a dict of measured statistics.
    """
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(quads))
    sel = rng.choice(len(quads), n, replace=False)

    n_total = 0
    n_in_support = 0
    n_has_rel_hist = 0
    n_blocked = 0
    n_strictly_blocked = 0
    n_recurrent = 0
    phases = []
    dts = []
    blocked_examples = []

    for i in sel:
        s, r, o, t = (int(quads[i, 0]), int(quads[i, 1]),
                      int(quads[i, 2]), int(quads[i, 3]))
        ids, F = index.candidates(s, r, t, max_support)
        n_total += 1
        if len(ids) == 0:
            continue

        pos = np.searchsorted(ids, o)
        if pos >= len(ids) or ids[pos] != o:
            continue                      # answer not in the candidate set
        n_in_support += 1

        # feature layout (see TKGIndex.candidates):
        #   0 log1p count(s,r,o)   1 log1p dt(s,r,o)   5 phase(s,r,o)
        #   8 log1p count(s,.,o)   9 log1p dt(s,.,o)  11 phase(s,.,o)
        cnt = np.expm1(F[:, 0])
        dt = np.expm1(F[:, 1])
        has = F[:, 7] > 0

        if not has[pos]:
            # no (s,r) history for the answer: fall back to entity history
            cnt = np.expm1(F[:, 8])
            dt = np.expm1(F[:, 9])
            has = F[:, 8] > 0
            ph = F[:, 11]
        else:
            n_has_rel_hist += 1
            ph = F[:, 5]

        if not has[pos]:
            continue
        n_recurrent += 1
        phases.append(float(ph[pos]))
        dts.append(float(dt[pos]))

        # a distractor that dominates the answer on (dt, count)
        others = has.copy()
        others[pos] = False
        if not others.any():
            continue
        dom = (dt[others] <= dt[pos]) & (cnt[others] >= cnt[pos])
        tie = (dt[others] == dt[pos]) & (cnt[others] == cnt[pos])
        strict = dom & ~tie
        if dom.any():
            n_blocked += 1
            if len(blocked_examples) < 5:
                j = np.flatnonzero(others)[np.flatnonzero(dom)[0]]
                blocked_examples.append(
                    dict(s=s, r=r, o=o, t=t,
                         answer=dict(dt=float(dt[pos]), count=float(cnt[pos]),
                                     phase=float(ph[pos])),
                         distractor=dict(entity=int(ids[j]),
                                         dt=float(dt[j]),
                                         count=float(cnt[j]),
                                         phase=float(ph[j]))))
        if strict.any():
            n_strictly_blocked += 1

    ph = np.array(phases) if phases else np.zeros(0)
    res = dict(
        sampled=n_total,
        support_recall=n_in_support / max(n_total, 1),
        recurrent_rate=n_recurrent / max(n_total, 1),
        rel_history_rate=n_has_rel_hist / max(n_total, 1),
        monotone_blocked_of_recurrent=n_blocked / max(n_recurrent, 1),
        strictly_blocked_of_recurrent=n_strictly_blocked / max(n_recurrent, 1),
        monotone_blocked_of_all=n_blocked / max(n_total, 1),
        phase_mean=float(ph.mean()) if len(ph) else 0.0,
        phase_median=float(np.median(ph)) if len(ph) else 0.0,
        phase_frac_below_half=float((ph < 0.5).mean()) if len(ph) else 0.0,
        phase_frac_near_one=float(((ph >= 0.5) & (ph <= 1.5)).mean())
        if len(ph) else 0.0,
        phase_frac_above_two=float((ph > 2.0).mean()) if len(ph) else 0.0,
        examples=blocked_examples,
    )

    if verbose:
        print(f"    sampled queries              {res['sampled']:,}")
        print(f"    answer in candidate set      {res['support_recall']*100:6.2f} %")
        print(f"    answer has usable history    {res['recurrent_rate']*100:6.2f} %  "
              f"(these are the recurrent queries)")
        print(f"    MONOTONE-BLOCKED             "
              f"{res['monotone_blocked_of_recurrent']*100:6.2f} % of recurrent")
        print(f"      strictly blocked           "
              f"{res['strictly_blocked_of_recurrent']*100:6.2f} % of recurrent")
        print(f"    phase of true answer:  <0.5 "
              f"{res['phase_frac_below_half']*100:5.1f} %   "
              f"0.5-1.5 {res['phase_frac_near_one']*100:5.1f} %   "
              f">2 {res['phase_frac_above_two']*100:5.1f} %")
        print(f"    phase median                 {res['phase_median']:6.3f}")
    return res
