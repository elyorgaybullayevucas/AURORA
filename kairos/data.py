"""
Snapshot-batched data pipeline for KAIROS.

Training iterates over TIMESTAMPS, not over shuffled quadruples. For a
timestamp t the model first evolves the entity representations through the
H preceding snapshots, then answers every query at t against those evolved
representations. This is the RE-GCN / TiRGN training regime and it is what
makes the structural branch strong: entity states are updated from the whole
graph at each step, not from the query subject's local neighbourhood.

Recurrence features are static per query, so they are computed once per
timestamp inside DataLoader workers (batch_size=1 over timestamps, which
gives free prefetch overlap with the GPU).
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from aurora_cf.data import load_quadruples, get_step
from aurora_cf.tkg_index import TKGIndex

N_FEAT = TKGIndex.N_FEAT


def _augment(q, R):
    inv = np.stack([q[:, 2], q[:, 1] + R, q[:, 0], q[:, 3]], 1)
    return np.concatenate([q, inv.astype(q.dtype)], 0)


class SnapshotSet(Dataset):
    """One item = one timestamp: its queries plus the preceding snapshots."""

    def __init__(self, quads, index: TKGIndex, all_edges_by_t, times_sorted,
                 cfg, num_relations):
        self.index = index
        self.H = cfg.hist_len
        self.S = cfg.max_support
        self.step = index.step
        self.edges = all_edges_by_t          # {t: (src, rel, dst)}
        self.times_sorted = times_sorted

        q = _augment(quads.astype(np.int64), num_relations)
        order = np.argsort(q[:, 3], kind="stable")
        q = q[order]
        self.times = np.unique(q[:, 3])
        bounds = np.searchsorted(q[:, 3], self.times)
        self.q = q
        self.starts = bounds
        self.ends = np.append(bounds[1:], len(q))

    def __len__(self):
        return len(self.times)

    def history(self, t):
        """Edge lists of the H snapshots preceding t (most recent first)."""
        out = []
        tt = t - self.step
        for _ in range(self.H):
            if tt < 0:
                break
            e = self.edges.get(int(tt))
            if e is not None:
                out.append(e)
            tt -= self.step
        return out[::-1]                      # oldest first, for the GRU

    def __getitem__(self, i):
        t = int(self.times[i])
        a, b = self.starts[i], self.ends[i]
        blk = self.q[a:b]
        subs, rels, objs = blk[:, 0], blk[:, 1], blk[:, 2]

        n = len(subs)
        ids_l, f_l = [], []
        for j in range(n):
            ids, F = self.index.candidates(int(subs[j]), int(rels[j]), t,
                                           self.S)
            ids_l.append(ids)
            f_l.append(F)
        S = max(1, max(len(x) for x in ids_l))
        sup_ids = np.zeros((n, S), np.int64)
        sup_feat = np.zeros((n, S, N_FEAT), np.float32)
        sup_mask = np.zeros((n, S), bool)
        for j in range(n):
            k = len(ids_l[j])
            if k:
                sup_ids[j, :k] = ids_l[j]
                sup_feat[j, :k] = f_l[j]
                sup_mask[j, :k] = True

        hist = self.history(t)
        return dict(
            t=t,
            subs=torch.from_numpy(subs),
            rels=torch.from_numpy(rels),
            objs=torch.from_numpy(objs),
            sup_ids=torch.from_numpy(sup_ids),
            sup_feat=torch.from_numpy(sup_feat),
            sup_mask=torch.from_numpy(sup_mask),
            hist=[(torch.from_numpy(s), torch.from_numpy(r),
                   torch.from_numpy(o)) for s, r, o in hist],
        )


def identity_collate(batch):
    return batch[0]


class KairosData:
    def __init__(self, cfg):
        base = os.path.join(cfg.data_dir, cfg.dataset)
        self.step = get_step(cfg.dataset)

        tr = load_quadruples(os.path.join(base, "train.txt"))
        va = load_quadruples(os.path.join(base, "valid.txt"))
        te = load_quadruples(os.path.join(base, "test.txt"))
        allq = np.concatenate([tr, va, te], 0)

        self.num_entities = int(allq[:, [0, 2]].max()) + 1
        self.num_relations = int(allq[:, 1].max()) + 1
        R = self.num_relations

        print(f"[{cfg.dataset}] entities={self.num_entities:,}  "
              f"relations={R}  train={len(tr):,} valid={len(va):,} "
              f"test={len(te):,}  step={self.step}")

        self.index = TKGIndex(allq, self.num_entities, R, step=self.step,
                              use_inverse=True, rel_topk=cfg.rel_topk)

        # snapshot edge lists (with inverse edges) over the FULL timeline;
        # only edges strictly before a query timestamp are ever consumed
        aug = _augment(allq.astype(np.int64), R)
        order = np.argsort(aug[:, 3], kind="stable")
        aug = aug[order]
        ts = np.unique(aug[:, 3])
        bnd = np.searchsorted(aug[:, 3], ts)
        ends = np.append(bnd[1:], len(aug))
        self.edges_by_t = {
            int(ts[i]): (aug[bnd[i]:ends[i], 0].astype(np.int64),
                         aug[bnd[i]:ends[i], 1].astype(np.int64),
                         aug[bnd[i]:ends[i], 2].astype(np.int64))
            for i in range(len(ts))}
        self.timeline = ts

        mk = lambda q: SnapshotSet(q, self.index, self.edges_by_t, ts, cfg, R)
        self.train_set = mk(tr)
        self.valid_set = mk(va)
        self.test_set = mk(te)

        e_per_snap = np.mean([len(v[0]) for v in self.edges_by_t.values()])
        print(f"[snapshots] {len(ts):,} timestamps   "
              f"avg {e_per_snap:,.0f} edges/snapshot   "
              f"train timestamps={len(self.train_set):,}")
