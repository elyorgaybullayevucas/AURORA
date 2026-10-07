"""
The candidate cache must change speed and nothing else.

Checks, on synthetic data, for every timestamp of every split and for both
serving modes (flat arrays on the device, or in host memory):

  - every tensor the cached path yields is EQUAL to the DataLoader path's
  - the history snapshots are the same edges in the same order
  - evaluate() gives identical metrics with and without the cache
"""
import os
import shutil
import tempfile

import numpy as np
import torch

import kairos.config as kc
from kairos.config import KairosConfig
from kairos.data import KairosData
from kairos import cache as C
from kairos.prism import PRISM
from train_kairos import to_dev, evaluate

torch.manual_seed(0)
rng = np.random.default_rng(0)
NE, NR, T = 60, 5, 60
q = np.array([(rng.integers(NE), rng.integers(NR), rng.integers(NE),
               rng.integers(T)) for _ in range(3000)], dtype=np.int64)
d = tempfile.mkdtemp()
os.makedirs(os.path.join(d, "SYN"), exist_ok=True)
n = len(q)
for nm, sl in (("train", q[:int(n * .8)]), ("valid", q[int(n * .8):int(n * .9)]),
               ("test", q[int(n * .9):])):
    np.savetxt(os.path.join(d, "SYN", nm + ".txt"), sl, fmt="%d", delimiter="\t")

kc.DATASETS["SYN"] = dict(kc.DATASETS["YAGO"])
cdir = tempfile.mkdtemp()
cfg = KairosConfig(dataset="SYN", data_dir=d, embed_dim=32, hazard_dim=16,
                   gcn_layers=2, conv_channels=8, hist_len=4, max_support=64,
                   rel_topk=8, num_workers=0, dropout=0.0, prism=True,
                   cache=True, cache_dir=cdir, cache_workers=0)
data = KairosData(cfg)
dev = torch.device("cpu")

path = C.ensure(cfg, data, workers=0)
edges_dev = {t: tuple(torch.from_numpy(x) for x in e)
             for t, e in data.edges_by_t.items()}
sets = {"train": data.train_set, "valid": data.valid_set, "test": data.test_set}

KEYS = ("subs", "rels", "objs", "sup_ids", "sup_feat", "sup_mask")
checked = 0
for on_gpu in (True, False):
    for split, ds in sets.items():
        cs = C.CachedSplit(os.path.join(path, split), ds, edges_dev, dev, on_gpu)
        assert len(cs) == len(ds)
        for i in range(len(ds)):
            a = to_dev(ds[i], dev)
            b = cs.get(i)
            assert a["t"] == b["t"]
            for k in KEYS:
                assert a[k].dtype == b[k].dtype, (split, i, k, a[k].dtype, b[k].dtype)
                assert a[k].shape == b[k].shape, (split, i, k, a[k].shape, b[k].shape)
                assert torch.equal(a[k], b[k]), (split, i, k)
            assert len(a["hist"]) == len(b["hist"])
            for ea, eb in zip(a["hist"], b["hist"]):
                for xa, xb in zip(ea, eb):
                    assert torch.equal(xa, xb)
            checked += 1
print(f"cached == uncached on all {checked} (timestamp, mode) items, "
      f"every tensor, both serving modes")

model = PRISM(data.num_entities, data.num_relations, cfg)
r0 = evaluate(model, data, "test", dev, cfg, verbose=False, stratify=True)
data.cache = C.load(cfg, data, dev)
r1 = evaluate(model, data, "test", dev, cfg, verbose=False, stratify=True)
for k in r0:
    assert r0[k] == r1[k], (k, r0[k], r1[k])
print(f"evaluate(): identical metrics with and without the cache "
      f"(MRR {r0['time_aware_filtered']['MRR']:.6f})")

# Windows refuses to delete a file that is still memory-mapped; Linux does
# not care. The checks above are what matters, the cleanup is best-effort.
shutil.rmtree(cdir, ignore_errors=True)
print("\nALL CACHE SMOKE TESTS PASSED")
