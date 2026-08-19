#!/usr/bin/env python
"""
Measure whether the premise of this work holds, before spending GPU time.

    python diagnose.py                    # all four datasets
    python diagnose.py --dataset ICEWS18

Reports, per dataset and per split:
  * how often the answer is inside the recurrence candidate set
  * how often it has usable history at all
  * the MONOTONE-BLOCKED rate -- the fraction of recurrent queries that NO
    scorer of the form f(count)*g(dt), g non-increasing, can rank first.
    That family contains CyGNet, CENET, TiRGN, RE-GCN, DaeMon and the
    exponential Hawkes kernel of GAttNHP.
  * the phase distribution of true answers

If the monotone-blocked rate is small everywhere, the proposed non-monotone
kernel cannot buy much and that should change the plan.
"""
import argparse
import json
import os
import numpy as np

from aurora_cf.data import load_quadruples, get_step
from aurora_cf.tkg_index import TKGIndex
from kairos.diagnostics import analyse

DATASETS = ["YAGO", "WIKI", "ICEWS18", "GDELT"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, choices=DATASETS)
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--n_samples", type=int, default=20000)
    ap.add_argument("--max_support", type=int, default=256)
    ap.add_argument("--rel_topk", type=int, default=64)
    ap.add_argument("--out", default="checkpoints/diagnostics.json")
    a = ap.parse_args()

    names = [a.dataset] if a.dataset else DATASETS
    out = {}

    for name in names:
        base = os.path.join(a.data_dir, name)
        if not os.path.isdir(base):
            print(f"[skip] {base} not found")
            continue

        tr = load_quadruples(os.path.join(base, "train.txt"))
        va = load_quadruples(os.path.join(base, "valid.txt"))
        te = load_quadruples(os.path.join(base, "test.txt"))
        allq = np.concatenate([tr, va, te], 0)
        NE = int(allq[:, [0, 2]].max()) + 1
        NR = int(allq[:, 1].max()) + 1
        step = get_step(name)

        print(f"\n{'='*72}")
        print(f"  {name}   entities={NE:,}  relations={NR}  "
              f"train={len(tr):,}  valid={len(va):,}  test={len(te):,}  "
              f"step={step}")
        print(f"{'='*72}")

        index = TKGIndex(allq, NE, NR, step=step, use_inverse=True,
                         rel_topk=a.rel_topk)
        out[name] = {}
        for split, q in (("valid", va), ("test", te)):
            print(f"  [{split}]")
            out[name][split] = analyse(index, q, a.n_samples,
                                       max_support=a.max_support)
        del index

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved → {a.out}")

    if out:
        print(f"\n{'='*72}")
        print("  SUMMARY (test split)")
        print(f"{'='*72}")
        print(f"  {'dataset':<10} {'recurrent':>10} {'blocked':>10} "
              f"{'blocked/all':>12} {'phase med':>10}")
        for k, v in out.items():
            t = v.get("test", {})
            print(f"  {k:<10} {t.get('recurrent_rate',0)*100:9.1f}% "
                  f"{t.get('monotone_blocked_of_recurrent',0)*100:9.1f}% "
                  f"{t.get('monotone_blocked_of_all',0)*100:11.1f}% "
                  f"{t.get('phase_median',0):10.3f}")


if __name__ == "__main__":
    main()
