#!/usr/bin/env python
"""
Measure before building: where are the answers the model cannot reach?

    python diag_twohop.py --dataset ICEWS18 --n 4000

The no-history stratum (answer outside the candidate support S) is where every
variant fails -- ~50% of ICEWS18 test queries at ~6 H@1 -- and LogCL leads us
by 3.65 H@10 on exactly that kind of query. One cheap repair would extend S
with entities two hops from the subject in the observed history
(s -> x -> o, all strictly before t), scored by the same unconstrained
recurrence intensity with path statistics. That only helps if the missing
answers ARE two hops away, and if the two-hop set is small enough to rank.

For a sample of test queries whose answer is NOT in S this reports:
  - how often the answer is in the two-hop set at all;
  - how often it is in the top-K two-hop entities ranked by path count, for
    the K a support extension could afford;
  - how large the two-hop set is.

No model, no GPU. Uses the same index, so S is exactly the training support.
"""
import argparse

import numpy as np

from kairos.config import KairosConfig
from kairos.data import KairosData


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ICEWS18")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    # the per-dataset overrides training uses (max_support, rel_topk, ...),
    # so S here is exactly the support the model was trained with
    from kairos.config import DATASETS
    fields = KairosConfig.__dataclass_fields__
    over = {k: v for k, v in DATASETS[a.dataset].items() if k in fields}
    cfg = KairosConfig(**{**over, "dataset": a.dataset})
    data = KairosData(cfg)
    idx = data.index
    q = data.test_set.q            # (Q, 4): test quadruples plus inverses
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(q))

    Ks = (64, 128, 256, 512)
    n_nohist = in2 = 0
    inK = {k: 0 for k in Ks}
    sizes = []
    for j in order:
        s, r, o, t = (int(x) for x in q[j, :4])
        S, _ = idx.candidates(s, r, t, cfg.max_support)
        if o in set(S.tolist()):
            continue
        n_nohist += 1

        # one hop: every object of s, any relation, before t
        x_ids, x_cnt, *_ = idx.g_s.stats(s, t)
        paths = {}
        for x, cx in zip(x_ids.tolist(), x_cnt.tolist()):
            y_ids, y_cnt, *_ = idx.g_s.stats(x, t)
            for y, cy in zip(y_ids.tolist(), y_cnt.tolist()):
                if y != s:
                    paths[y] = paths.get(y, 0.0) + cx * cy
        sizes.append(len(paths))
        if o in paths:
            in2 += 1
            ranked = sorted(paths, key=paths.get, reverse=True)
            pos = ranked.index(o)
            for k in Ks:
                if pos < k:
                    inK[k] += 1
        if n_nohist >= a.n:
            break

    sz = np.array(sizes)
    print(f"\n{a.dataset}: {n_nohist} sampled test queries with the answer "
          f"OUTSIDE the support S")
    print(f"  answer within two hops of s     {100*in2/n_nohist:6.1f} %")
    for k in Ks:
        print(f"  ... and in top-{k:<4} by path count {100*inK[k]/n_nohist:6.1f} %")
    print(f"  two-hop set size: median {np.median(sz):.0f}, "
          f"90th pct {np.percentile(sz, 90):.0f}, "
          f"entities in graph {data.num_entities}")
    print("\n  Reading: if top-256 covers a large share, extending S with "
          "two-hop\n  candidates is worth building; if it does not, the "
          "missing answers\n  need a different signal.")


if __name__ == "__main__":
    main()
