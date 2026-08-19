"""
On-the-fly dataset for HAWK. Nothing is materialised per training sample:
features are produced inside DataLoader workers from the flat numpy index,
so host RAM stays flat regardless of dataset size (this is what makes GDELT
tractable) and startup is instant.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from aurora_cf.data import load_quadruples, get_step
from aurora_cf.tkg_index import TKGIndex

N_FEAT = TKGIndex.N_FEAT


class TPPDataset(Dataset):

    def __init__(self, quads, index: TKGIndex, cfg, use_inverse=False,
                 num_relations=None):
        if use_inverse and num_relations is not None:
            inv = np.stack([quads[:, 2], quads[:, 1] + num_relations,
                            quads[:, 0], quads[:, 3]], axis=1)
            data = np.concatenate([quads, inv.astype(quads.dtype)], axis=0)
        else:
            data = quads.copy()
        self.data = data.astype(np.int64)
        self.index = index
        self.H = cfg.hist_len
        self.K = cfg.k_neighbors
        self.S = cfg.max_support
        self._rng = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            self._rng = np.random.default_rng(
                1234 + (info.id if info else 0))
        s, r, o, t = self.data[i]
        ne, nr, ndt, nm = self.index.neighbors(s, t, self.H, self.K, self._rng)
        ids, F = self.index.candidates(s, r, t, self.S)
        return (int(s), int(r), int(o), int(t), ne, nr, ndt, nm, ids, F)


def tpp_collate(batch):
    subs, rels, objs, times, ne_l, nr_l, ndt_l, nm_l, ids_l, f_l = zip(*batch)
    B = len(subs)
    S = max(1, max(len(x) for x in ids_l))

    sup_ids = np.zeros((B, S), dtype=np.int64)
    sup_feat = np.zeros((B, S, N_FEAT), dtype=np.float32)
    sup_mask = np.zeros((B, S), dtype=bool)
    for i in range(B):
        n = len(ids_l[i])
        if n:
            sup_ids[i, :n] = ids_l[i]
            sup_feat[i, :n] = f_l[i]
            sup_mask[i, :n] = True

    return (torch.tensor(subs, dtype=torch.long),
            torch.tensor(rels, dtype=torch.long),
            torch.tensor(objs, dtype=torch.long),
            torch.tensor(times, dtype=torch.long),
            torch.from_numpy(np.stack(ne_l).astype(np.int64)),
            torch.from_numpy(np.stack(nr_l).astype(np.int64)),
            torch.from_numpy(np.stack(ndt_l)),
            torch.from_numpy(np.stack(nm_l)),
            torch.from_numpy(sup_ids),
            torch.from_numpy(sup_feat),
            torch.from_numpy(sup_mask))


class TPPDataLoader:
    def __init__(self, cfg):
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

        self.index = TKGIndex(all_q, self.num_entities, self.num_relations,
                              step=self.step, use_inverse=cfg.use_inverse,
                              rel_topk=cfg.rel_topk)
        print(f"[index] edges={self.index.n_edges:,}  "
              f"(inverse included)  rel_topk={cfg.rel_topk}")

        self.train_set = TPPDataset(train_q, self.index, cfg,
                                    use_inverse=cfg.use_inverse,
                                    num_relations=self.num_relations)
        self.valid_set = TPPDataset(valid_q, self.index, cfg)
        self.test_set = TPPDataset(test_q, self.index, cfg)

    def support_diagnostics(self, split="train", n=4000, seed=0):
        """Measured support recall and |S| — reported, not assumed."""
        ds = {"train": self.train_set, "valid": self.valid_set,
              "test": self.test_set}[split]
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ds), min(n, len(ds)), replace=False)
        hit = 0; tot = 0; sizes = []
        for i in idx:
            s, r, o, t = ds.data[i]
            ids, _ = self.index.candidates(s, r, t, ds.S)
            sizes.append(len(ids))
            hit += int(o in ids); tot += 1
        return hit / tot, float(np.mean(sizes))
