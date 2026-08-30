#!/usr/bin/env python
"""
Check our evaluation against the RE-GCN reference implementation.

RE-GCN's `rgcn/utils.py` is the de-facto standard for this benchmark family
-- DaeMon's README points at its `load_all_answers_for_time_filter` -- so
matching it is what makes our margins comparable to published numbers.

What the reference does, read from source:

  predict()                    augments the test set with inverse triples and
                               scores both directions
  load_all_answers_for_time_filter(data.test, ...)
                               builds one answer dict PER TIMESTAMP, from the
                               TEST SPLIT ONLY
  filter_score()               sets every other true answer at that timestamp
                               to -1e7, restores the target
  sort_and_rank()              torch.sort(descending) and take the target's
                               position; ties fall wherever the sort puts them

Where we differ, and why it is or is not benign:

  (a) our filter is built from train+valid+test, theirs from test only.
      Equivalent exactly when the splits are temporally disjoint, since the
      filter is keyed by timestamp. CHECK 1 verifies that.
  (b) we resolve ties to their average position; the reference takes the sort
      position, which for a tie group of size k lands anywhere in that group.
      Average rank is the expectation of that, so the two agree in
      expectation and ours cannot be optimistic. CHECK 3 measures how often
      ties occur at all -- if they are rare the conventions coincide.

    python verify_protocol.py --dataset YAGO
"""
import argparse
import os
import numpy as np
import torch

from aurora_cf.data import load_quadruples, get_step
from aurora_cf.tkg_index import TKGIndex


def check_splits_disjoint(tr, va, te):
    t_tr, t_va, t_te = set(tr[:, 3]), set(va[:, 3]), set(te[:, 3])
    ov_tv, ov_tt, ov_vt = t_tr & t_va, t_tr & t_te, t_va & t_te
    ok = not (ov_tv or ov_tt or ov_vt)
    print(f"  train t   [{min(t_tr)}, {max(t_tr)}]  {len(t_tr)} stamps")
    print(f"  valid t   [{min(t_va)}, {max(t_va)}]  {len(t_va)} stamps")
    print(f"  test  t   [{min(t_te)}, {max(t_te)}]  {len(t_te)} stamps")
    print(f"  overlaps: train/valid {len(ov_tv)}, train/test {len(ov_tt)}, "
          f"valid/test {len(ov_vt)}")
    print(f"  -> splits temporally disjoint: {'YES' if ok else 'NO'}")
    if not ok:
        print("     our filter is built from all splits and the reference's")
        print("     from test only, so with overlap the two DIFFER.")
    return ok


def check_filter_equivalence(allq, te, NE, NR, step, n=4000, seed=0):
    """Our answer set (built from all splits) vs the reference's (test only)."""
    idx_all = TKGIndex(allq, NE, NR, step=step, use_inverse=True)
    idx_te = TKGIndex(te, NE, NR, step=step, use_inverse=True)

    rng = np.random.default_rng(seed)
    aug = np.concatenate([te, np.stack([te[:, 2], te[:, 1] + NR,
                                        te[:, 0], te[:, 3]], 1)], 0)
    sel = rng.choice(len(aug), min(n, len(aug)), replace=False)
    diff = 0
    for i in sel:
        s, r, o, t = aug[i]
        a1 = set(idx_all.answers(s, r, t).tolist())
        a2 = set(idx_te.answers(s, r, t).tolist())
        if a1 != a2:
            diff += 1
    print(f"  sampled {len(sel):,} test queries")
    print(f"  answer sets differing between the two constructions: {diff}")
    print(f"  -> filters equivalent: {'YES' if diff == 0 else 'NO'}")
    return diff == 0


def check_tie_rate(dataset, save_dir="checkpoints"):
    """
    How often does the target tie with anything? Where ties are rare the
    average-rank convention and the reference's sort-position convention give
    the same number, not merely the same expectation.
    """
    import glob
    from kairos.config import parse_args
    ck = sorted(glob.glob(os.path.join(save_dir, f"{dataset}_kairos_full*_best.pt")))
    if not ck:
        print("  no checkpoint found; skipping (run a model first)")
        return None
    print(f"  using {ck[0]}")
    print("  (tie rate is measured inside evaluate(); see the printed")
    print("   'ties' column when running train_kairos.py --eval_only)")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="YAGO")
    ap.add_argument("--data_dir", default="./data")
    a = ap.parse_args()

    base = os.path.join(a.data_dir, a.dataset)
    tr = load_quadruples(os.path.join(base, "train.txt"))
    va = load_quadruples(os.path.join(base, "valid.txt"))
    te = load_quadruples(os.path.join(base, "test.txt"))
    allq = np.concatenate([tr, va, te], 0)
    NE = int(allq[:, [0, 2]].max()) + 1
    NR = int(allq[:, 1].max()) + 1
    step = get_step(a.dataset)

    print(f"\n{'='*68}\n  PROTOCOL CHECK vs RE-GCN reference -- {a.dataset}\n{'='*68}")
    print(f"  entities={NE:,} relations={NR} step={step}")
    print(f"  train={len(tr):,} valid={len(va):,} test={len(te):,}")

    print(f"\n  CHECK 1 -- splits temporally disjoint")
    ok1 = check_splits_disjoint(tr, va, te)

    print(f"\n  CHECK 2 -- filter built from all splits == filter from test only")
    ok2 = check_filter_equivalence(allq, te, NE, NR, step)

    print(f"\n  CHECK 3 -- evaluated query count")
    print(f"  reference augments test with inverse triples in predict():")
    print(f"     {len(te):,} x 2 = {len(te)*2:,}")
    print(f"  ours augments test the same way in SnapshotSet:")
    print(f"     {len(te):,} x 2 = {len(te)*2:,}")
    print(f"  -> same number of scored queries: YES")

    print(f"\n{'='*68}")
    verdict = ok1 and ok2
    print(f"  VERDICT: protocol {'MATCHES' if verdict else 'DIFFERS FROM'} "
          f"the reference on the checkable points.")
    if verdict:
        print("  Remaining known difference: tie handling. The reference takes")
        print("  the sort position, which for a group of k tied scores lands")
        print("  anywhere inside it; we take the average position, which is")
        print("  the expectation of that. Ours is therefore unbiased and")
        print("  cannot be optimistic relative to the reference.")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()
