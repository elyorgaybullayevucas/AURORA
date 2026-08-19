"""
Data pipeline for NHC-Full.

Fixes two structural defects of the first NHC prototype:

 1. INVERSE EDGES. The training set is augmented with (o, r+R, s, t) but the
    index was not, so every inverse sample had an empty relation history.
    Measured: train support_recall 0.4976 (~half the data carried no signal).

 2. DEGENERATE SUPPORT. Restricting candidates to the subject's own history
    gave avg|S| = 1.3 on WIKI -- there was nothing to rank, and H@10 was
    pinned to support recall. The candidate set is now the union of

        (a) objects seen for (s, r)          -- subject+relation history
        (b) objects seen for (s, ·)          -- subject history
        (c) globally frequent objects of r   -- relation prior

    (c) guarantees a non-trivial candidate set for every query and lifts the
    H@10 ceiling. Candidates from (c) carry relation-global temporal features
    so the hazard net still has something to condition on.

Per-candidate features (F = 15):
    0-11  subject-level statistics (counts, recency, gaps, PHASE)
    12-14 relation-global statistics (count, recency, phase)
"""
import os
import bisect
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

from aurora_cf.config import AURORACFConfig
from aurora_cf.data import load_quadruples, get_step

N_FEAT = 15


def _stats(ts, t_q, step):
    """ts: sorted list/array of timestamps < t_q. -> (count, dt, mean_gap, std, span, phase)"""
    m = len(ts)
    if m == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = (t_q - ts[-1]) / step
    span = (ts[-1] - ts[0]) / step
    if m >= 2:
        gaps = np.diff(np.asarray(ts, dtype=np.float64)) / step
        mean_gap = float(gaps.mean())
        std_gap = float(gaps.std())
    else:
        mean_gap = std_gap = 0.0
    phase = dt / mean_gap if mean_gap > 1e-6 else 0.0
    return float(m), float(dt), mean_gap, std_gap, float(span), float(phase)


class FullIndex:

    def __init__(self, quads_all: np.ndarray, num_relations: int,
                 step: int = 1, use_inverse: bool = True,
                 rel_topk: int = 32):
        self.step = max(int(step), 1)
        self.rel_topk = rel_topk

        if use_inverse:
            inv = np.stack([quads_all[:, 2], quads_all[:, 1] + num_relations,
                            quads_all[:, 0], quads_all[:, 3]], axis=1)
            q = np.concatenate([quads_all, inv.astype(quads_all.dtype)], axis=0)
        else:
            q = quads_all
        self.quads = q

        # filtered-setting answers (both directions)
        self.all_answers: Dict[Tuple, set] = defaultdict(set)
        for s, r, o, t in q:
            self.all_answers[(int(s), int(r), int(t))].add(int(o))

        # (t, s) -> [(r, o)]  for structural neighbourhoods
        self._by_time_sub: Dict[Tuple, List[Tuple]] = defaultdict(list)
        for s, r, o, t in q:
            self._by_time_sub[(int(t), int(s))].append((int(r), int(o)))

        # (s, r) -> {o: sorted times}
        raw = defaultdict(list)
        for s, r, o, t in q:
            raw[(int(s), int(r), int(o))].append(int(t))
        self._sr_objs: Dict[Tuple, Dict[int, List[int]]] = defaultdict(dict)
        for (s, r, o), ts in raw.items():
            self._sr_objs[(s, r)][o] = sorted(ts)

        # s -> {o: sorted times}
        raw_e = defaultdict(list)
        for s, r, o, t in q:
            raw_e[(int(s), int(o))].append(int(t))
        self._s_objs: Dict[int, Dict[int, List[int]]] = defaultdict(dict)
        for (s, o), ts in raw_e.items():
            self._s_objs[s][o] = sorted(ts)

        # r -> {o: sorted times}   (relation-global prior)
        raw_r = defaultdict(list)
        for s, r, o, t in q:
            raw_r[(int(r), int(o))].append(int(t))
        self._r_objs: Dict[int, Dict[int, List[int]]] = defaultdict(dict)
        for (r, o), ts in raw_r.items():
            self._r_objs[r][o] = sorted(ts)

        # r -> top-K globally frequent objects (static candidate expansion)
        self._r_top: Dict[int, List[int]] = {}
        for r, d in self._r_objs.items():
            cnt = Counter({o: len(ts) for o, ts in d.items()})
            self._r_top[r] = [o for o, _ in cnt.most_common(rel_topk)]

    # ── candidate set + features ─────────────────────────────────────────────

    def candidate_features(self, sub, rel, t_q, num_entities, max_support):
        step = self.step
        rel_hist = self._sr_objs.get((sub, rel), {})
        ent_hist = self._s_objs.get(sub, {})
        rel_glob = self._r_objs.get(rel, {})

        cands = set()
        for o in rel_hist:
            if o < num_entities:
                cands.add(o)
        for o in ent_hist:
            if o < num_entities:
                cands.add(o)
        for o in self._r_top.get(rel, []):
            if o < num_entities:
                cands.add(o)

        if not cands:
            return (np.zeros(0, dtype=np.int32),
                    np.zeros((0, N_FEAT), dtype=np.float32))

        ids, rows = [], []
        for o in cands:
            tr = rel_hist.get(o)
            tr = tr[:bisect.bisect_left(tr, t_q)] if tr else []
            te = ent_hist.get(o)
            te = te[:bisect.bisect_left(te, t_q)] if te else []
            tg = rel_glob.get(o)
            tg = tg[:bisect.bisect_left(tg, t_q)] if tg else []

            if not tr and not te and not tg:
                continue  # nothing observable before t_q

            cr, dtr, mgr, sgr, spr, phr = _stats(tr, t_q, step)
            ce, dte, mge, _, _, phe = _stats(te, t_q, step)
            cg, dtg, mgg, _, _, phg = _stats(tg, t_q, step)

            rows.append([
                np.log1p(cr),                 # 0  (s,r,o) count
                np.log1p(ce),                 # 1  (s,·,o) count
                np.log1p(dtr) if cr else 0.,  # 2  (s,r,o) recency
                np.log1p(dte) if ce else 0.,  # 3  (s,·,o) recency
                np.log1p(mgr),                # 4  mean inter-arrival
                np.log1p(sgr),                # 5  gap dispersion
                np.log1p(spr),                # 6  observed span
                min(phr, 20.0),               # 7  PHASE  (s,r,o)
                min(phe, 20.0),               # 8  PHASE  (s,·,o)
                1.0 if cr else 0.0,           # 9  has (s,r) history
                cr / (spr + 1.0),             # 10 event rate
                np.log1p(mge),                # 11 ent mean inter-arrival
                np.log1p(cg),                 # 12 (r,o) global count
                np.log1p(dtg) if cg else 0.,  # 13 (r,o) global recency
                min(phg, 20.0),               # 14 PHASE  (r,o) global
            ])
            ids.append(o)

        if not ids:
            return (np.zeros(0, dtype=np.int32),
                    np.zeros((0, N_FEAT), dtype=np.float32))

        feats = np.asarray(rows, dtype=np.float32)
        ids = np.asarray(ids, dtype=np.int32)

        if len(ids) > max_support:
            # prefer subject-specific candidates, then recent ones
            prio = -(feats[:, 0] + feats[:, 1]) * 10.0 + feats[:, 2]
            keep = np.argsort(prio)[:max_support]
            ids, feats = ids[keep], feats[keep]

        return ids, feats

    # ── structural neighbourhood ─────────────────────────────────────────────

    def get_neighbors(self, sub, t_q, H, K, rng):
        ne = np.zeros((H, K), dtype=np.int32)
        nr = np.zeros((H, K), dtype=np.int32)
        nm = np.zeros((H, K), dtype=bool)
        t = t_q - self.step
        for h in range(H):
            if t < 0:
                break
            facts = self._by_time_sub.get((t, sub))
            if facts:
                chosen = facts if len(facts) <= K else rng.sample(facts, K)
                for k, (r, o) in enumerate(chosen):
                    ne[h, k] = o; nr[h, k] = r; nm[h, k] = True
            t -= self.step
        return ne, nr, nm


class FullDataset(Dataset):

    def __init__(self, quads, index: FullIndex, num_entities, cfg,
                 use_inverse=False, num_relations=None, desc=""):
        import random as _random
        if use_inverse and num_relations is not None:
            inv = np.stack([quads[:, 2], quads[:, 1] + num_relations,
                            quads[:, 0], quads[:, 3]], axis=1).astype(np.int32)
            data = np.concatenate([quads, inv], axis=0)
        else:
            data = quads.copy()
        self.data = data
        N = len(data)
        H, K, S = cfg.hist_len, cfg.k_neighbors, cfg.max_support
        rng = _random.Random(cfg.seed)

        print(f"  [{desc}] Pre-computing ({N:,})  H={H} K={K} S={S} F={N_FEAT}…",
              flush=True)

        self._ne = np.zeros((N, H, K), dtype=np.int16)
        self._nr = np.zeros((N, H, K), dtype=np.int16)
        self._nm = np.zeros((N, H, K), dtype=bool)
        self._ids = [None] * N
        self._feats = [None] * N

        hit = 0
        for i in range(N):
            s, r, o, t = data[i]
            a, b, c = index.get_neighbors(int(s), int(t), H, K, rng)
            self._ne[i] = a.clip(-32768, 32767)
            self._nr[i] = b.clip(-32768, 32767)
            self._nm[i] = c

            ids, f = index.candidate_features(int(s), int(r), int(t),
                                              num_entities, S)
            self._ids[i] = ids
            self._feats[i] = f.astype(np.float16)
            if o in ids:
                hit += 1
            if (i + 1) % 200_000 == 0:
                print(f"    {i+1:,}/{N:,}", flush=True)

        self.support_recall = hit / max(N, 1)
        self.avg_support = float(np.mean([len(x) for x in self._ids]))
        print(f"  [{desc}] Done.  support_recall={self.support_recall:.4f}  "
              f"avg|S|={self.avg_support:.1f}", flush=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        s, r, o, t = self.data[i]
        return (int(s), int(r), int(o), int(t),
                torch.from_numpy(self._ne[i].astype(np.int32)).long(),
                torch.from_numpy(self._nr[i].astype(np.int32)).long(),
                torch.from_numpy(self._nm[i]),
                self._ids[i], self._feats[i])


def full_collate(batch):
    subs, rels, objs, times, ne_l, nr_l, nm_l, ids_l, f_l = zip(*batch)
    B = len(subs)
    S = max(1, max(len(x) for x in ids_l))

    sup_ids = torch.zeros(B, S, dtype=torch.long)
    sup_feat = torch.zeros(B, S, N_FEAT, dtype=torch.float32)
    sup_mask = torch.zeros(B, S, dtype=torch.bool)
    for i in range(B):
        n = len(ids_l[i])
        if n:
            sup_ids[i, :n] = torch.from_numpy(ids_l[i].astype(np.int64))
            sup_feat[i, :n] = torch.from_numpy(f_l[i].astype(np.float32))
            sup_mask[i, :n] = True

    return (torch.tensor(subs, dtype=torch.long),
            torch.tensor(rels, dtype=torch.long),
            torch.tensor(objs, dtype=torch.long),
            torch.tensor(times, dtype=torch.long),
            torch.stack(ne_l), torch.stack(nr_l), torch.stack(nm_l),
            sup_ids, sup_feat, sup_mask)


class FullDataLoader:
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

        self.index = FullIndex(all_q, self.num_relations, step=self.step,
                               use_inverse=cfg.use_inverse,
                               rel_topk=cfg.rel_topk)

        self.train_set = FullDataset(train_q, self.index, self.num_entities,
                                     cfg, use_inverse=cfg.use_inverse,
                                     num_relations=self.num_relations,
                                     desc="train")
        self.valid_set = FullDataset(valid_q, self.index, self.num_entities,
                                     cfg, desc="valid")
        self.test_set = FullDataset(test_q, self.index, self.num_entities,
                                    cfg, desc="test")
        print("Datasets ready.")
