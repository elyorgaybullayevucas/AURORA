#!/usr/bin/env python
"""
Score an EXTERNAL model under our evaluation protocol.

Why this exists
---------------
Published tables in this area disagree with each other. LogCL (ICDE 2024)
reports ICEWS18 MRR 35.67; DiMNet (2025) reports 34.13 and calls it the state
of the art, and its comparison table does not contain LogCL. Both say
"time-aware filtered". They cannot both be measuring the same quantity, and
quoting either number next to ours is therefore not a comparison.

The fix is not to argue about it. It is to run the other model's scores
through OUR filter and OUR tie handling, so that exactly one convention is
applied to both. This module is the bridge: it holds the reference
implementation of that convention and nothing else.

How to use it
-------------
Inside the other model's test loop, wherever it has a score vector over all
entities for one query:

    from eval_external import ExternalScorer

    scorer = ExternalScorer("ICEWS18", data_root="data")     # once
    ...
    scorer.add(sub, rel, t, obj, score_row)                  # per query
    ...
    print(scorer.report())                                   # at the end

`score_row` is a 1-D array or tensor of length num_entities: higher is better.
`t` must be in the SAME units the dataset files use (the raw timestamp column,
not a snapshot ordinal). Inverse queries, if the other model scores them,
should be passed with the relation id it used for them; pass `inverse=True` so
the count is reported separately and can be checked against ours.

What it does NOT do
-------------------
It does not retrain, reimplement, or reinterpret the other model. If their
score vector is wrong, this reports a wrong number faithfully. Its only claim
is that the filtering and the ranking are identical to ours.
"""
import argparse
import os
import json

import numpy as np
import torch

from kairos.config import KairosConfig
from kairos.data import KairosData
from train_kairos import Meters, ranks_of


# this repository's own data, whatever directory the caller runs from
REPO_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class ExternalScorer:
    """Applies our filter and our tie-aware ranking to someone else's scores."""

    def __init__(self, dataset, data_root=None, hits_at=(1, 3, 10),
                 device="cpu"):
        cfg = KairosConfig(dataset=dataset)
        # KairosData reads cfg.data_dir. This used to set cfg.data_root, which
        # nothing reads, so the argument was silently ignored and the index
        # was built from whatever "data/" was relative to the CURRENT
        # directory -- harmless when run from this repo, wrong when imported
        # from another model's checkout. Resolved to an absolute path so it
        # does not depend on where the caller runs.
        cfg.data_dir = os.path.abspath(data_root or REPO_DATA)
        self.data = KairosData(cfg)
        self.index = self.data.index
        self.N = self.data.num_entities
        self.R = self.data.num_relations
        self.device = torch.device(device)
        self.M = Meters(tuple(hits_at))
        self.n_forward = 0
        self.n_inverse = 0
        self.n_skipped = 0

    # ── one query ────────────────────────────────────────────────────────────

    def add(self, sub, rel, t, obj, scores, inverse=False):
        """
        scores: length-N, higher is better. Anything array-like.

        Returns the rank assigned, so a caller can spot-check individual
        queries against its own number before trusting the aggregate.
        """
        s = torch.as_tensor(np.asarray(scores, dtype=np.float32),
                            device=self.device).view(1, -1)
        if s.size(1) != self.N:
            raise ValueError(
                f"score vector has {s.size(1)} entries but this dataset has "
                f"{self.N} entities; the entity id spaces do not match, so the "
                f"comparison would be meaningless")
        obj = int(obj)
        # a copy, not a view: the filtering below writes into a clone of `s`,
        # but keeping tgt independent of both is what makes that safe to
        # change later
        tgt = s[:, obj].clone().view(1, 1)

        # raw
        self.M.add("raw", ranks_of(s, tgt))

        # time-aware filtered: every OTHER true answer at this timestamp is
        # removed, then the target is restored. Identical to evaluate().
        ans = self.index.answers(int(sub), int(rel), int(t))
        s_ta = s.clone()
        if len(ans):
            s_ta[0, torch.from_numpy(np.asarray(ans, dtype=np.int64)
                                     ).to(self.device)] = float("-inf")
        s_ta[0, obj] = tgt.item()
        r = ranks_of(s_ta, tgt)
        self.M.add("time_aware_filtered", r)

        if inverse:
            self.n_inverse += 1
        else:
            self.n_forward += 1
        return float(r.item())

    # ── report ───────────────────────────────────────────────────────────────

    def report(self):
        out = self.M.result()
        out["_counts"] = {
            "forward": self.n_forward,
            "inverse": self.n_inverse,
            "total": self.n_forward + self.n_inverse,
            "our_test_queries": int(len(self.data.test_set.q)),
        }
        return out

    def summary(self):
        r = self.report()
        c = r["_counts"]
        lines = [
            "",
            f"  external model, our protocol   ({c['total']:,} queries scored;"
            f" our own test set has {c['our_test_queries']:,})",
            "  " + "-" * 62,
        ]
        if c["total"] != c["our_test_queries"]:
            lines.append(
                "  WARNING: query counts differ. Either the other model scores "
                "only one\n           direction, or the splits are not the "
                "same. Do not compare\n           these numbers until this is "
                "resolved.")
            lines.append("  " + "-" * 62)
        for name in ("raw", "time_aware_filtered"):
            if name not in r:
                continue
            d = r[name]
            lines.append(
                f"  {name:<22} MRR {d['MRR']*100:6.2f}   "
                + "   ".join(f"H@{k} {d[f'Hits@{k}']*100:6.2f}"
                            for k in (1, 3, 10) if f"Hits@{k}" in d))
        return "\n".join(lines) + "\n"


# ── self-check: prove the bridge agrees with our own evaluator ──────────────

def self_check(dataset, tag=None, variant="full", data_root="data",
               limit=25, device="cuda"):
    """
    Push OUR model's scores through the bridge and check the bridge reproduces
    what evaluate() reports.

    A measurement bridge that has not been validated is worth nothing: if the
    filter here differs from the filter in evaluate(), every external number it
    produces is wrong in an unknown direction, and the comparison it exists to
    make is worse than no comparison. The filter logic deliberately lives in
    exactly two places, and this compares them.

    Runs on `limit` test timestamps; that is enough for an exact agreement
    check, since the two paths must agree query by query, not on average.
    """
    import os
    import torch as T
    from kairos.model import KAIROS
    from train_kairos import to_dev, identity_collate
    from torch.utils.data import DataLoader

    cfg = KairosConfig(dataset=dataset)
    if data_root:
        cfg.data_dir = os.path.abspath(data_root)
    dev = T.device(device if T.cuda.is_available() else "cpu")
    data = KairosData(cfg)
    model = KAIROS(data.num_entities, data.num_relations, cfg).to(dev).eval()

    name = f"{dataset}_kairos_{variant}" + (f"_{tag}" if tag else "")
    ck = os.path.join(cfg.save_dir, f"{name}_best.pt")
    if os.path.exists(ck):
        sd = T.load(ck, map_location=dev, weights_only=True)["model"]
        own = model.state_dict()
        ok = {k: v for k, v in sd.items()
              if k in own and own[k].shape == v.shape}
        model.load_state_dict(ok, strict=False)
        left = sorted(set(own) - set(ok))
        print(f"  loaded {len(ok)}/{len(own)} tensors from {ck}")
        if left:
            # Say it loudly. A checkpoint written before the architecture
            # changed loads "successfully" with parts of the model still
            # random, and the accuracy printed below is then meaningless even
            # though the filter check it exists for is still valid.
            print(f"  NOTE: left at initialisation: {left}")
            print("        the MRR below is therefore NOT this model's "
                  "accuracy; only the rank-gap line is under test here")
    else:
        print(f"  no checkpoint at {ck}; checking with an untrained model "
              f"(the filter is what is under test, not the accuracy)")

    sc = ExternalScorer(dataset, data_root, device=dev.type)
    ref = Meters((1, 3, 10))
    dl = DataLoader(data.test_set, batch_size=1, shuffle=False,
                    num_workers=0, collate_fn=identity_collate)

    worst = 0.0
    with T.no_grad():
        for n_ts, raw in enumerate(dl):
            if n_ts >= limit:
                break
            it = to_dev(raw, dev)
            E, _ = model.evolve(it["hist"])
            lg = model(E, it["subs"], it["rels"], it["sup_ids"],
                       it["sup_feat"], it["sup_mask"]).float()
            for i in range(it["subs"].numel()):
                row = lg[i]
                s, r, o = (int(it["subs"][i]), int(it["rels"][i]),
                           int(it["objs"][i]))
                got = sc.add(s, r, it["t"], o, row.cpu().numpy())

                # The reference path, exactly as evaluate() does it.
                #
                # tgt must be a COPY. evaluate() gets it from gather(), which
                # allocates; taking a view here instead aliases the row, so
                # blanking the other true answers also blanks the target and
                # every rank comes out as N. That is what this check caught on
                # its first run.
                one = row.view(1, -1).clone()
                tgt = one[:, o].clone().view(1, 1)
                ans = data.index.answers(s, r, it["t"])
                if len(ans):
                    one[0, T.from_numpy(np.asarray(ans, np.int64)).to(dev)] = \
                        float("-inf")
                one[0, o] = tgt.item()
                want = float(ranks_of(one, tgt).item())
                ref.add("time_aware_filtered", T.tensor([want]))
                worst = max(worst, abs(got - want))

    a = sc.report()["time_aware_filtered"]
    b = ref.result()["time_aware_filtered"]
    print(f"\n  queries compared        {a['n']:,}")
    print(f"  bridge   MRR            {a['MRR']*100:.4f}")
    print(f"  evaluate MRR            {b['MRR']*100:.4f}")
    print(f"  max per-query rank gap  {worst:g}")
    assert worst == 0.0, (
        "the bridge and evaluate() disagree on at least one query; external "
        "numbers produced by this module cannot be trusted until they agree")
    print("\n  SELF-CHECK PASSED — the bridge applies our protocol exactly\n")


# ── offline mode: consume a dumped .npz of ranks-ready score rows ────────────

def main():
    p = argparse.ArgumentParser(
        description="Re-score an external model's dumped predictions under "
                    "our protocol.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--data_root", default=None,
                   help="defaults to this repository's data/")
    p.add_argument("--self_check", action="store_true",
                   help="validate the bridge against our own evaluator and "
                        "exit; run this before trusting any external number")
    p.add_argument("--tag", default=None)
    p.add_argument("--variant", default="full")
    p.add_argument("--limit", type=int, default=25,
                   help="test timestamps to use for --self_check")
    p.add_argument("--dump", default=None,
                   help="npz with arrays sub, rel, t, obj (int64, length Q) "
                        "and scores (float32/float16, Q x num_entities). "
                        "Q x N is large; prefer importing ExternalScorer into "
                        "the other model's loop instead of dumping.")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    if a.self_check:
        self_check(a.dataset, a.tag, a.variant, a.data_root, a.limit)
        return
    if not a.dump:
        p.error("give --dump, or --self_check")

    # One .npz, or a directory of part_*.npz as external/patch_logcl.py
    # writes them (one per test snapshot, so no file holds the whole matrix).
    if os.path.isdir(a.dump):
        parts = sorted(os.path.join(a.dump, f) for f in os.listdir(a.dump)
                       if f.endswith(".npz"))
        if not parts:
            p.error(f"no .npz files in {a.dump}")
    else:
        parts = [a.dump]
    sc = ExternalScorer(a.dataset, a.data_root)
    done = 0
    for k, part in enumerate(parts):
        z = np.load(part)
        inv = z["inverse"] if "inverse" in z else np.zeros(len(z["sub"]), bool)
        for i in range(len(z["sub"])):
            sc.add(z["sub"][i], z["rel"][i], z["t"][i], z["obj"][i],
                   z["scores"][i], inverse=bool(inv[i]))
        done += len(z["sub"])
        if (k + 1) % 10 == 0 or k + 1 == len(parts):
            print(f"  {k+1}/{len(parts)} parts, {done:,} queries", flush=True)

    print(sc.summary())
    if a.out:
        with open(a.out, "w") as f:
            json.dump(sc.report(), f, indent=2)
        print(f"  written to {a.out}")


if __name__ == "__main__":
    main()
