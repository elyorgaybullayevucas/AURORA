"""
Data loading for AURORA-CF.
Pre-computes neighborhoods and copy scores once at startup.
"""
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from typing import Dict, List, Tuple

from aurora_cf.config import AURORACFConfig


def load_quadruples(path: str) -> np.ndarray:
    quads = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                quads.append([int(parts[0]), int(parts[1]),
                               int(parts[2]), int(parts[3])])
    return np.array(quads, dtype=np.int32)


def get_step(dataset: str) -> int:
    return 24 if dataset in ("ICEWS18", "ICEWS14") else 1


class GraphIndex:
    def __init__(self, quads_all: np.ndarray, step: int = 1):
        self.step = step

        self.all_answers: Dict[Tuple, set] = defaultdict(set)
        for s, r, o, t in quads_all:
            self.all_answers[(int(s), int(r), int(t))].add(int(o))

        self._by_time_sub: Dict[Tuple, List[Tuple]] = defaultdict(list)
        for s, r, o, t in quads_all:
            self._by_time_sub[(int(t), int(s))].append((int(r), int(o)))

        # (s, r) → {o: sorted timestamps}
        _sr_raw: Dict[Tuple, List[int]] = defaultdict(list)
        for s, r, o, t in quads_all:
            _sr_raw[(int(s), int(r), int(o))].append(int(t))
        self._sr_objs: Dict[Tuple, Dict[int, List[int]]] = defaultdict(dict)
        for (s, r, o), times in _sr_raw.items():
            self._sr_objs[(s, r)][o] = sorted(times)

        self._by_sub: Dict[int, List[Tuple]] = defaultdict(list)
        for s, r, o, t in quads_all:
            self._by_sub[int(s)].append((int(o), int(t), int(r)))

    def get_rel_copy_scores(self, sub, rel, query_time, num_entities,
                            copy_lambda, recency_steps, recency_boost):
        scores = np.zeros(num_entities, dtype=np.float32)
        step = max(self.step, 1)
        thr  = recency_steps * step
        for o, times in self._sr_objs.get((sub, rel), {}).items():
            ts = [tt for tt in times if tt < query_time]
            if not ts:
                continue
            last_t = max(ts)
            decay  = np.exp(-copy_lambda * (query_time - last_t) / step)
            score  = np.log1p(len(ts)) * decay
            if (query_time - last_t) <= thr:
                score *= recency_boost
            if o < num_entities:
                scores[o] = score
        return scores

    def get_ent_copy_scores(self, sub, query_time, num_entities,
                            copy_lambda, recency_steps, recency_boost):
        scores = np.zeros(num_entities, dtype=np.float32)
        step   = max(self.step, 1)
        thr    = recency_steps * step
        by_obj: Dict[int, List[int]] = defaultdict(list)
        for o, t, _ in self._by_sub.get(sub, []):
            if t < query_time:
                by_obj[o].append(t)
        for o, ts in by_obj.items():
            if o >= num_entities:
                continue
            last_t = max(ts)
            decay  = np.exp(-copy_lambda * (query_time - last_t) / step)
            score  = np.log1p(len(ts)) * decay
            if (query_time - last_t) <= thr:
                score *= recency_boost
            scores[o] = score
        return scores

    def get_neighbors(self, sub, query_time, hist_len, k_neighbors, rng):
        H, K = hist_len, k_neighbors
        ne = np.zeros((H, K), dtype=np.int32)
        nr = np.zeros((H, K), dtype=np.int32)
        nm = np.zeros((H, K), dtype=bool)
        t  = query_time - self.step
        for h in range(H):
            if t < 0:
                break
            facts = self._by_time_sub.get((t, sub), [])
            if facts:
                chosen = rng.sample(facts, min(len(facts), K))
                for k, (r, o) in enumerate(chosen):
                    ne[h, k] = o
                    nr[h, k] = r
                    nm[h, k] = True
            t -= self.step
        return ne, nr, nm


class CFDataset(Dataset):
    """Pre-computed dataset: O(1) __getitem__."""

    def __init__(self, quads: np.ndarray, index: GraphIndex,
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

        H = cfg.hist_len
        K = cfg.k_neighbors
        rng = random.Random(cfg.seed)

        print(f"  [{desc}] Pre-computing neighborhoods ({N:,}, H={H}, K={K})…",
              flush=True)
        ne = np.zeros((N, H, K), dtype=np.int16)
        nr = np.zeros((N, H, K), dtype=np.int16)
        nm = np.zeros((N, H, K), dtype=bool)
        for i in range(N):
            s, _, _, t = data[i]
            ne_i, nr_i, nm_i = index.get_neighbors(int(s), int(t), H, K, rng)
            ne[i] = ne_i.clip(-32768, 32767).astype(np.int16)
            nr[i] = nr_i.clip(-32768, 32767).astype(np.int16)
            nm[i] = nm_i
            if (i + 1) % 200_000 == 0:
                print(f"    {i+1:,}/{N:,}", flush=True)

        self._ne = ne
        self._nr = nr
        self._nm = nm

        print(f"  [{desc}] Pre-computing copy scores…", flush=True)
        rc_idx, rc_val = [None] * N, [None] * N
        ec_idx, ec_val = [None] * N, [None] * N
        for i in range(N):
            s, r, _, t = data[i]
            rc = index.get_rel_copy_scores(int(s), int(r), int(t),
                                           num_entities, cfg.copy_lambda,
                                           cfg.recency_steps, cfg.recency_boost)
            nz = rc.nonzero()[0]
            rc_idx[i] = nz.astype(np.int32)
            rc_val[i] = rc[nz]

            if cfg.use_entity_copy:
                ec = index.get_ent_copy_scores(int(s), int(t),
                                               num_entities, cfg.copy_lambda,
                                               cfg.recency_steps, cfg.recency_boost)
                nze = ec.nonzero()[0]
                ec_idx[i] = nze.astype(np.int32)
                ec_val[i] = ec[nze]
            else:
                ec_idx[i] = np.zeros(0, dtype=np.int32)
                ec_val[i] = np.zeros(0, dtype=np.float32)

            if (i + 1) % 200_000 == 0:
                print(f"    {i+1:,}/{N:,}", flush=True)

        self._rc_idx = rc_idx
        self._rc_val = rc_val
        self._ec_idx = ec_idx
        self._ec_val = ec_val
        self._num_ent = num_entities
        print(f"  [{desc}] Done.", flush=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s, r, o, t = self.data[idx]
        ne = torch.from_numpy(self._ne[idx].astype(np.int32)).long()
        nr = torch.from_numpy(self._nr[idx].astype(np.int32)).long()
        nm = torch.from_numpy(self._nm[idx])
        return (int(s), int(r), int(o), int(t),
                ne, nr, nm,
                self._rc_idx[idx], self._rc_val[idx],
                self._ec_idx[idx], self._ec_val[idx])


def cf_collate(batch):
    (subs, rels, objs, times,
     ne_l, nr_l, nm_l,
     rc_idx_l, rc_val_l,
     ec_idx_l, ec_val_l) = zip(*batch)

    subs  = torch.tensor(subs,  dtype=torch.long)
    rels  = torch.tensor(rels,  dtype=torch.long)
    objs  = torch.tensor(objs,  dtype=torch.long)
    times = torch.tensor(times, dtype=torch.long)
    ne    = torch.stack(ne_l)
    nr    = torch.stack(nr_l)
    nm    = torch.stack(nm_l)

    all_idx = np.concatenate(
        [x for x in rc_idx_l if len(x) > 0] +
        [x for x in ec_idx_l if len(x) > 0]
    )
    N = int(all_idx.max()) + 1 if len(all_idx) > 0 else 1

    B = len(subs)
    rc = torch.zeros(B, N, dtype=torch.float32)
    ec = torch.zeros(B, N, dtype=torch.float32)
    for i in range(B):
        if len(rc_idx_l[i]) > 0:
            rc[i, rc_idx_l[i]] = torch.from_numpy(rc_val_l[i])
        if len(ec_idx_l[i]) > 0:
            ec[i, ec_idx_l[i]] = torch.from_numpy(ec_val_l[i])

    return subs, rels, objs, times, ne, nr, nm, rc, ec


class CFDataLoader:
    def __init__(self, cfg: AURORACFConfig):
        self.cfg  = cfg
        base      = os.path.join(cfg.data_dir, cfg.dataset)
        self.step = get_step(cfg.dataset)

        train_q = load_quadruples(os.path.join(base, "train.txt"))
        valid_q = load_quadruples(os.path.join(base, "valid.txt"))
        test_q  = load_quadruples(os.path.join(base, "test.txt"))

        all_q = np.concatenate([train_q, valid_q, test_q], axis=0)
        self.num_entities  = int(all_q[:, [0, 2]].max()) + 1
        self.num_relations = int(all_q[:, 1].max()) + 1

        print(f"[{cfg.dataset}] entities={self.num_entities:,}  "
              f"relations={self.num_relations}  "
              f"train={len(train_q):,}  valid={len(valid_q):,}  "
              f"test={len(test_q):,}  step={self.step}")

        self.index = GraphIndex(all_q, step=self.step)

        self.train_set = CFDataset(train_q, self.index, self.num_entities, cfg,
                                   use_inverse=cfg.use_inverse,
                                   num_relations=self.num_relations,
                                   desc="train")
        self.valid_set = CFDataset(valid_q, self.index, self.num_entities, cfg,
                                   desc="valid")
        self.test_set  = CFDataset(test_q,  self.index, self.num_entities, cfg,
                                   desc="test")
        print("Datasets ready.")
