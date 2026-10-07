"""
Candidate cache: compute the support sets and their statistics ONCE, then
serve every epoch from GPU memory.

================================ WHY ========================================
For a query (s, r, t) the candidate support and its 16 inter-arrival
statistics depend only on the query and on facts strictly before t. They are
identical in epoch 1 and epoch 60. The DataLoader path recomputed them on the
CPU every epoch, per query, in Python -- which is why the GPUs sat idle
waiting for batches while the model itself needs only milliseconds.

Here they are computed once, by the very same code (SnapshotSet.__getitem__),
in parallel over as many workers as the machine has, and written to disk in
a padding-free CSR layout. At training time the whole cache is moved to the
GPU when it fits; when it does not (GDELT), it stays in host memory and each
timestamp is copied over with a single transfer. Either way no statistic is
recomputed, and the training loop needs no DataLoader workers at all.

Because the cache is filled by the existing code rather than a
reimplementation, the values are bit-identical to the uncached path.
smoke_cache.py asserts that, tensor by tensor.

================================ LAYOUT =====================================
Per split, in <cache_dir>/<dataset>_<key>/<split>.*

  meta.npz    times (T,), q_start / q_end (T,) query range per timestamp,
              subs / rels / objs (Q,), off (Q+1,) CSR offsets into the flat
              arrays, n_feat
  ids.bin     int64  (nnz,)          candidate entity ids, valid ones only
  feat.bin    float32(nnz, n_feat)   their statistics

The key hashes everything that can change the values: the dataset, the
support size, rel_topk, and the source of the index and data modules. A code
change there invalidates the cache instead of silently serving stale values.
"""
import hashlib
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from kairos.data import N_FEAT, identity_collate


def cache_key(cfg):
    h = hashlib.sha1()
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    for f in (os.path.join(root, "aurora_cf", "tkg_index.py"),
              os.path.join(here, "data.py")):
        with open(f, "rb") as fh:
            h.update(fh.read())
    h.update(f"{cfg.dataset}|{cfg.max_support}|{cfg.rel_topk}|{N_FEAT}"
             .encode())
    return h.hexdigest()[:12]


def _dir(cfg):
    return os.path.join(cfg.cache_dir, f"{cfg.dataset}_{cache_key(cfg)}")


# ── build ────────────────────────────────────────────────────────────────────

def build_split(ds, path, workers):
    """
    Stream one split into CSR files. Memory stays flat: each timestamp is
    appended to disk as soon as it arrives, so GDELT's tens of gigabytes are
    never held twice.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=workers,
                    collate_fn=identity_collate,
                    prefetch_factor=4 if workers > 0 else None)
    times, qs, qe = [], [], []
    subs, rels, objs, counts = [], [], [], []
    nq = 0
    t0 = time.time()
    with open(path + ".ids.bin", "wb") as fi, open(path + ".feat.bin", "wb") as ff:
        for k, it in enumerate(dl):
            m = it["sup_mask"].numpy()
            n = m.shape[0]
            times.append(it["t"]); qs.append(nq); qe.append(nq + n)
            nq += n
            subs.append(it["subs"].numpy()); rels.append(it["rels"].numpy())
            objs.append(it["objs"].numpy())
            counts.append(m.sum(1))
            # row-major over (query, slot) keeps each query's candidates
            # contiguous and in their original slot order
            fi.write(np.ascontiguousarray(it["sup_ids"].numpy()[m],
                                          dtype=np.int64).tobytes())
            ff.write(np.ascontiguousarray(it["sup_feat"].numpy()[m],
                                          dtype=np.float32).tobytes())
            if (k + 1) % 50 == 0:
                print(f"    {k+1}/{len(ds)} timestamps  "
                      f"{time.time()-t0:.0f}s", flush=True)
    counts = np.concatenate(counts).astype(np.int64)
    off = np.zeros(len(counts) + 1, np.int64)
    np.cumsum(counts, out=off[1:])
    np.savez(path + ".meta.npz", times=np.array(times, np.int64),
             q_start=np.array(qs, np.int64), q_end=np.array(qe, np.int64),
             subs=np.concatenate(subs), rels=np.concatenate(rels),
             objs=np.concatenate(objs), off=off, n_feat=N_FEAT)
    # written last: its presence is what marks the split as complete
    open(path + ".done", "w").close()
    return int(off[-1])


def ensure(cfg, data, workers=None):
    """Build any missing split, then return the cache directory."""
    d = _dir(cfg)
    workers = workers if workers is not None else min(48, os.cpu_count() or 8)
    for split in ("train", "valid", "test"):
        p = os.path.join(d, split)
        if os.path.exists(p + ".done"):
            continue
        ds = {"train": data.train_set, "valid": data.valid_set,
              "test": data.test_set}[split]
        print(f"[cache] building {cfg.dataset}/{split} with {workers} workers "
              f"-> {d}")
        nnz = build_split(ds, p, workers)
        print(f"[cache] {split}: {nnz:,} (query, candidate) entries, "
              f"{nnz * (8 + 4 * N_FEAT) / 2**30:.2f} GB")
    return d


# ── serve ────────────────────────────────────────────────────────────────────

class CachedSplit:
    """
    Serves the same dict to_dev() produced, from device-resident tensors.

    `on_gpu` decides where the flat arrays live. If they fit, they are moved
    to the device once and every timestamp is a pure GPU gather. If not, they
    stay pinned in host memory and a timestamp costs one host-to-device copy
    of its own slice -- still no computation on the CPU.
    """

    def __init__(self, path, snapset, edges_dev, device, on_gpu):
        m = np.load(path + ".meta.npz")
        self.times = m["times"]
        self.qs, self.qe = m["q_start"], m["q_end"]
        self.off = m["off"]
        self.nf = int(m["n_feat"])
        self.dev = device
        self.set = snapset
        self.edges = edges_dev
        nnz = int(self.off[-1])
        # Memory-mapped, read-only. The host path used to read the whole file
        # into a private, pinned copy per process: three GDELT seeds started
        # together each spent 30+ minutes pulling 55 GB off the same disk and
        # held 166 GB of pinned RAM between them. A memory map is backed by the
        # OS page cache, so the file is read once and SHARED by every process
        # that maps it, and the run starts immediately; each timestamp then
        # copies only its own slice to the device.
        ids = np.memmap(path + ".ids.bin", dtype=np.int64, mode="r",
                        shape=(nnz,))
        feat = np.memmap(path + ".feat.bin", dtype=np.float32, mode="r",
                         shape=(nnz, self.nf))
        if on_gpu:
            self.ids = torch.from_numpy(np.array(ids)).to(device)
            self.feat = torch.from_numpy(np.array(feat)).to(device)
        else:
            self.ids, self.feat = ids, feat
        self.subs = torch.from_numpy(m["subs"]).to(device)
        self.rels = torch.from_numpy(m["rels"]).to(device)
        self.objs = torch.from_numpy(m["objs"]).to(device)
        self.off_t = torch.from_numpy(self.off).to(device)
        self.on_gpu = on_gpu

    def __len__(self):
        return len(self.times)

    def get(self, i):
        t = int(self.times[i])
        a, b = int(self.qs[i]), int(self.qe[i])
        lo, hi = int(self.off[a]), int(self.off[b])
        if self.on_gpu:
            ids = self.ids[lo:hi]
            feat = self.feat[lo:hi]
        else:
            # np.array copies the slice out of the (read-only) map
            ids = torch.from_numpy(np.array(self.ids[lo:hi])).to(self.dev)
            feat = torch.from_numpy(np.array(self.feat[lo:hi])).to(self.dev)

        # re-pad on the device: slot j of query q is flat entry off[q] + j
        cnt = self.off_t[a + 1:b + 1] - self.off_t[a:b]           # (n,)
        n = b - a
        S = max(1, int(cnt.max().item()) if n else 1)
        slot = torch.arange(S, device=self.dev)
        mask = slot.unsqueeze(0) < cnt.unsqueeze(1)                # (n, S)
        flat = (self.off_t[a:b] - lo).unsqueeze(1) + slot           # (n, S)
        flat = torch.where(mask, flat, torch.zeros_like(flat))
        if hi > lo:
            sup_ids = torch.where(mask, ids[flat], torch.zeros_like(flat))
            sup_feat = feat[flat] * mask.unsqueeze(-1)
        else:
            sup_ids = torch.zeros(n, S, dtype=torch.long, device=self.dev)
            sup_feat = torch.zeros(n, S, self.nf, device=self.dev)
        return dict(
            t=t,
            subs=self.subs[a:b], rels=self.rels[a:b], objs=self.objs[a:b],
            sup_ids=sup_ids, sup_feat=sup_feat, sup_mask=mask,
            hist=[self.edges[tt] for tt in self.set.history_times(t)],
        )


def load(cfg, data, device, gpu_budget_frac=0.45):
    """
    Build if needed, then serve all three splits.

    The flat arrays go to the GPU only if all three splits together take less
    than `gpu_budget_frac` of the memory currently free, leaving the rest for
    the model and its activations.
    """
    d = ensure(cfg, data, cfg.cache_workers or None)
    edges_dev = {t: tuple(torch.from_numpy(x).to(device) for x in e)
                 for t, e in data.edges_by_t.items()}
    need = 0
    for split in ("train", "valid", "test"):
        off = np.load(os.path.join(d, split) + ".meta.npz")["off"]
        need += int(off[-1]) * (8 + 4 * N_FEAT)
    # On a shared machine the default is host memory. Parking gigabytes of
    # cache on a GPU other people also use is what turns someone else's
    # launch into an OOM for both jobs; the host path costs one copy per
    # timestamp and no computation. --cache_on_gpu opts in when the GPU is
    # known to be ours alone.
    on_gpu = False
    if device.type == "cuda" and cfg.cache_on_gpu:
        free, _ = torch.cuda.mem_get_info(device)
        on_gpu = need < gpu_budget_frac * free
    print(f"[cache] {need/2**30:.2f} GB of candidates -> "
          f"{'GPU memory' if on_gpu else 'memory-mapped host memory (shared)'}")
    sets = {"train": data.train_set, "valid": data.valid_set,
            "test": data.test_set}
    return {s: CachedSplit(os.path.join(d, s), sets[s], edges_dev, device,
                           on_gpu) for s in sets}
