#!/usr/bin/env python
"""
Report HAWK's memory footprint and speed.

Without --gpu it reports parameter counts and an analytic activation
estimate on CPU (no data needed). With --gpu N it additionally runs real
forward/backward steps on synthetic tensors of the true shapes and reports
measured peak GPU memory and step time, then projects epoch and run time
from each dataset's actual sample count.

    python bench_hawk.py                 # static analysis, all datasets
    python bench_hawk.py --gpu 0         # measured, all datasets
    python bench_hawk.py --gpu 0 --dataset GDELT
"""
import argparse
import time
import torch

from aurora_cf.hawk_config import HawkConfig, DATASETS
from aurora_cf.model_hawk import HAWK, N_FEAT

# entities, relations, train quadruples (before inverse augmentation)
DIMS = {
    "YAGO":    (10_623,  10,   161_540),
    "WIKI":    (12_554,  24,   539_286),
    "ICEWS18": (23_033,  256,  373_018),
    "GDELT":   (7_691,   240, 1_734_399),
}
GB = 1024 ** 3
MB = 1024 ** 2


def analytic_activations(cfg, N):
    """Dominant saved-for-backward tensors, in bytes (bf16 = 2)."""
    B, H, K, d = cfg.batch_size, cfg.hist_len, cfg.k_neighbors, cfg.embed_dim
    S, dh, dt = cfg.max_support, cfg.hazard_dim, cfg.dt_dim
    bhk = B * H * K

    shared = bhk * (2 * d + dt) * 2                    # h_ne, h_nr, h_dt
    per_layer = (bhk * (2 * d + dt)      # cat(...)
                 + bhk * 2 * d           # kv output
                 + bhk * cfg.n_heads * 4  # attention maps + copies
                 ) * 2
    attn = shared + cfg.n_layers * per_layer

    rec = (B * S * (N_FEAT + 2 * dh)           # trunk input
           + 2 * B * S * 2 * dh                # trunk hidden
           + 2 * B * S * 3 * 13) * 2           # basis + weights

    dec = (B * cfg.conv_channels * d * 2       # conv features
           + B * N * 2                         # logits bf16
           + B * N * 4 * 2)                    # float logits + softmax grad
    return attn, rec, dec


def report(name, gpu=None):
    N, R, n_train = DIMS[name]
    cfg = HawkConfig(**{k: v for k, v in DATASETS[name].items()
                        if k in HawkConfig.__dataclass_fields__},
                     dataset=name)
    model = HAWK(N, R, cfg)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # AdamW: fp32 weights + grads + 2 moments
    optim_bytes = n_par * 16
    attn, rec, dec = analytic_activations(cfg, N)
    act = attn + rec + dec

    n_samples = n_train * 2                    # inverse augmentation
    steps = n_samples // cfg.batch_size

    print(f"\n{'='*66}")
    print(f"  {name}   entities={N:,}  relations={R}  "
          f"train={n_train:,} → {n_samples:,} with inverse")
    print(f"  d={cfg.embed_dim} layers={cfg.n_layers} H={cfg.hist_len} "
          f"K={cfg.k_neighbors} S={cfg.max_support} B={cfg.batch_size}")
    print(f"{'='*66}")
    print(f"  parameters            {n_par/1e6:8.2f} M")
    print(f"  weights+grads+Adam    {optim_bytes/GB:8.2f} GB")
    print(f"  activations (est.)    {act/GB:8.2f} GB"
          f"   [attn {attn/GB:.2f} | rec {rec/GB:.2f} | dec {dec/GB:.2f}]")
    print(f"  total (est.)          {(optim_bytes+act)/GB:8.2f} GB")
    print(f"  steps / epoch         {steps:8,}")

    if gpu is None:
        return

    dev = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    B, H, K, S = cfg.batch_size, cfg.hist_len, cfg.k_neighbors, cfg.max_support
    g = torch.Generator(device="cpu").manual_seed(0)
    subs = torch.randint(0, N, (B,), generator=g).to(dev)
    rels = torch.randint(0, 2 * R, (B,), generator=g).to(dev)
    objs = torch.randint(0, N, (B,), generator=g).to(dev)
    ne = torch.randint(0, N, (B, H, K), generator=g).to(dev)
    nr = torch.randint(0, 2 * R, (B, H, K), generator=g).to(dev)
    ndt = torch.rand(B, H, K, generator=g).to(dev) * 10
    nm = (torch.rand(B, H, K, generator=g) > 0.3).to(dev)
    sup_ids = torch.randint(0, N, (B, S), generator=g).to(dev)
    sup_feat = torch.rand(B, S, N_FEAT, generator=g).to(dev) * 3
    sup_mask = (torch.rand(B, S, generator=g) > 0.2).to(dev)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(subs, rels, ne, nr, ndt, nm, sup_ids, sup_feat, sup_mask)
            loss = torch.nn.functional.cross_entropy(lg.float(), objs)
        loss.backward()
        opt.step()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    n_it = 10
    for _ in range(n_it):
        step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n_it
    peak = torch.cuda.max_memory_allocated(dev)

    ep_gpu = dt * steps
    print(f"  ── measured on GPU {gpu} ──")
    print(f"  peak memory           {peak/GB:8.2f} GB")
    print(f"  step time             {dt*1000:8.1f} ms")
    print(f"  epoch (GPU-bound)     {ep_gpu/60:8.1f} min")
    print(f"  {cfg.epochs} epochs           {ep_gpu*cfg.epochs/3600:8.1f} h"
          f"   (excludes data loading and eval)")

    del model, opt
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--dataset", default=None, choices=list(DIMS))
    a = ap.parse_args()
    names = [a.dataset] if a.dataset else list(DIMS)
    for nm in names:
        report(nm, a.gpu)
    print()
