"""
Data pipeline for NHC (Neural Hazard Copy).

Instead of collapsing each candidate's history into ONE scalar copy score
(as CyGNet / CENET / TiRGN / AURORA-CF all do), we keep an interpretable
temporal feature vector per candidate, so the model can LEARN the hazard
function instead of assuming exp(-lambda * dt).

Key feature: phase = dt / mean_gap.
    phase ~ 1  -> the fact is "due" to recur          (high hazard)
    phase << 1 -> just happened, too soon             (low hazard)
    phase >> 1 -> the process has died out            (low hazard)

An exponential decay is MONOTONE in dt and therefore provably cannot
represent this non-monotone "due-ness" signal. This is the core novelty.
"""
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from typing import Dict, List, Tuple

from aurora_cf.config import AURORACFConfig
from aurora_cf.data import load_quadruples, get_step

# number of temporal features per candidate
N_FEAT = 12


class HazardIndex:
    """Temporal index exposing full timestamp histories (not just last-seen)."""

    def __init__(self, quads_all: np.ndarray, step: int = 1):
        self.step = max(int(step), 1)

        # filtered-setting answer sets
        self.all_answers: Dict[Tuple, set] = defaultdict(set)
        for s, r, o, t in quads_all:
            self.all_answers[(int(s), int(r), int(t))].add(int(o))

        self._by_time_sub: Dict[Tuple, List[Tuple]] = defaultdict(list)
        for s, r, o, t in quads_all:
            self._by_time_sub[(int(t), int(s))].append((int(r), int(o)))

        # (s,r) -> {o: sorted timestamps}
        _raw: Dict[Tuple, List[int]] = defaultdict(list)
        for s, r, o, t in quads_all:
            _raw[(int(s), int(r), int(o))].append(int(t))
        self._sr_objs: Dict[Tuple, Dict[int, np.ndarray]] = defaultdict(dict)
        for (s, r, o), ts in _raw.items():
            self._sr_objs[(s, r)][o] = np.array(sorted(ts), dtype=np.int64)

        # s -> {o: sorted timestamps}   (relation-agnostic)
        _raw_e: Dict[Tuple, List[int]] = defaultdict(list)
        for s, r, o, t in quads_all:
            _raw_e[(int(s), int(o))].append(int(t))
        self._s_objs: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
        for (s, o), ts in _raw_e.items():
            self._s_objs[s][o] = np.array(sorted(ts), dtype=np.int64)

    # ── temporal statistics for one candidate ────────────────────────────────

    @staticmethod
    def _stats(ts: np.ndarray, t_q: int, step: int):
        """
        ts: sorted timestamps STRICTLY before t_q (may be empty)
        returns (count, dt, mean_gap, std_gap, span, phase)
        """
        m = len(ts)
        if m == 0:
            return 0.0, -1.0, 0.0, 0.0, 0.0, 0.0
        dt = (t_q - ts[-1]) / step
        span = (ts[-1] - ts[0]) / step
        if m >= 2:
            gaps = np.diff(ts) / step
            mean_gap = float(gaps.mean())
            std_gap = float(gaps.std())
        else:
            mean_gap = 0.0
            std_gap = 0.0
        phase = dt / mean_gap if mean_gap > 1e-6 else 0.0
        return float(m), float(dt), mean_gap, std_gap, float(span), float(phase)

    def candidate_features(self, sub: int, rel: int, t_q: int,
                           num_entities: int, max_support: int):
        """
        Build the candidate support set and its temporal feature matrix.

        Returns:
            ids   : (S,) int32   candidate entity ids
            feats : (S, N_FEAT) float32
        """
        step = self.step

        rel_hist = self._sr_objs.get((sub, rel), {})
        ent_hist = self._s_objs.get(sub, {})

        cands = set()
        for o, ts in rel_hist.items():
            if o < num_entities and ts[0] < t_q:
                cands.add(o)
        for o, ts in ent_hist.items():
            if o < num_entities and ts[0] < t_q:
                cands.add(o)

        if not cands:
            return (np.zeros(0, dtype=np.int32),
                    np.zeros((0, N_FEAT), dtype=np.float32))

        rows = []
        ids = []
        for o in cands:
            tr = rel_hist.get(o)
            tr = tr[tr < t_q] if tr is not None else np.zeros(0, dtype=np.int64)
            te = ent_hist.get(o)
            te = te[te < t_q] if te is not None else np.zeros(0, dtype=np.int64)
            if len(tr) == 0 and len(te) == 0:
                continue

            cr, dtr, mgr, sgr, spr, phr = self._stats(tr, t_q, step)
            ce, dte, mge, sge, spe, phe = self._stats(te, t_q, step)

            rows.append([
                np.log1p(cr),                       # 0  rel-history count
                np.log1p(ce),                       # 1  ent-history count
                np.log1p(max(dtr, 0.0)),            # 2  rel recency
                np.log1p(max(dte, 0.0)),            # 3  ent recency
                np.log1p(mgr),                      # 4  rel mean inter-arrival
                np.log1p(sgr),                      # 5  rel gap dispersion
                np.log1p(spr),                      # 6  rel observed span
                min(phr, 20.0),                     # 7  rel PHASE  (due-ness)
                min(phe, 20.0),                     # 8  ent PHASE
                1.0 if cr > 0 else 0.0,             # 9  has rel history
                cr / (spr + 1.0),                   # 10 rel event rate
                np.log1p(mge),                      # 11 ent mean inter-arrival
            ])
            ids.append(o)

        feats = np.asarray(rows, dtype=np.float32)
        ids = np.asarray(ids, dtype=np.int32)

        # keep the most recent `max_support` candidates (col 2 = log1p rel dt)
        if len(ids) > max_support:
            recency = np.minimum(feats[:, 2], feats[:, 3])
            keep = np.argsort(recency)[:max_support]
            ids, feats = ids[keep], feats[keep]

        return ids, feats


class HazardDataset(Dataset):

    def __init__(self, quads: np.ndarray, index: HazardIndex,
                 num_entities: int, cfg: AURORACFConfig,
                 use_inverse: bool = False, num_relations: int = None,
                 desc: str = ""):
        step = get_step(cfg.dataset)

        if use_inverse and num_relations is not None:
            inv = np.stack([quads[:, 2], quads[:, 1] + num_relations,
                            quads[:, 0], quads[:, 3]], axis=1).astype(np.int32)
            data = np.concatenate([quads, inv], axis=0)
        else:
            data = quads.copy()
        self.data = data
        N = len(data)
        S = cfg.max_support

        print(f"  [{desc}] Pre-computing hazard features "
              f"({N:,}, S_max={S}, F={N_FEAT})…", flush=True)

        self._ids = [None] * N
        self._feats = [None] * N
        hit = 0
        for i in range(N):
            s, r, o, t = data[i]
            ids, feats = index.candidate_features(
                int(s), int(r), int(t), num_entities, S)
            self._ids[i] = ids
            self._feats[i] = feats.astype(np.float16)
            if o in ids:
                hit += 1
            if (i + 1) % 200_000 == 0:
                print(f"    {i+1:,}/{N:,}", flush=True)

        self.support_recall = hit / max(N, 1)
        avg_s = np.mean([len(x) for x in self._ids])
        print(f"  [{desc}] Done.  support_recall={self.support_recall:.4f}  "
              f"avg|S|={avg_s:.1f}", flush=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s, r, o, t = self.data[idx]
        return (int(s), int(r), int(o), int(t),
                self._ids[idx], self._feats[idx])


def hazard_collate(batch):
    subs, rels, objs, times, ids_l, feats_l = zip(*batch)
    B = len(subs)
    S = max(1, max(len(x) for x in ids_l))

    sup_ids = torch.zeros(B, S, dtype=torch.long)
    sup_feat = torch.zeros(B, S, N_FEAT, dtype=torch.float32)
    sup_mask = torch.zeros(B, S, dtype=torch.bool)

    for i in range(B):
        n = len(ids_l[i])
        if n:
            sup_ids[i, :n] = torch.from_numpy(ids_l[i].astype(np.int64))
            sup_feat[i, :n] = torch.from_numpy(
                feats_l[i].astype(np.float32))
            sup_mask[i, :n] = True

    return (torch.tensor(subs, dtype=torch.long),
            torch.tensor(rels, dtype=torch.long),
            torch.tensor(objs, dtype=torch.long),
            torch.tensor(times, dtype=torch.long),
            sup_ids, sup_feat, sup_mask)


class HazardDataLoader:
    def __init__(self, cfg: AURORACFConfig):
        self.cfg = cfg
        base = os.path.join(cfg.data_dir, cfg.dataset)
        self.step = get_step(cfg.dataset)

        train_q = load_quadruples(os.path.join(base, "train.txt"))
        valid_q = load_quadruples(os.path.join(base, "valid.txt"))
        test_q = load_quadruples(os.path.join(base, "test.txt"))

        all_q = np.concatenate([train_q, valid_q, test_q], axis=0)
        self.num_entities = int(all_q[:, [0, 2]].max()) + 1
        self.num_relations = int(all_q[:, 1].max()) + 1

        print(f"[{cfg.dataset}] entities={self.num_entities:,}  "
              f"relations={self.num_relations}  train={len(train_q):,}  "
              f"valid={len(valid_q):,}  test={len(test_q):,}  step={self.step}")

        self.index = HazardIndex(all_q, step=self.step)

        self.train_set = HazardDataset(train_q, self.index, self.num_entities,
                                       cfg, use_inverse=cfg.use_inverse,
                                       num_relations=self.num_relations,
                                       desc="train")
        self.valid_set = HazardDataset(valid_q, self.index, self.num_entities,
                                       cfg, desc="valid")
        self.test_set = HazardDataset(test_q, self.index, self.num_entities,
                                      cfg, desc="test")
        print("Datasets ready.")
