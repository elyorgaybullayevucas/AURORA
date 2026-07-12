#!/usr/bin/env python
"""
AURORA-v3: Two-mode TKG Forecasting

YAGO/WIKI  → copy_only=True  → pure calibrated copy, no neural
ICEWS18/GDELT → copy_only=False → fixed_gate copy + neural

Usage:
    python train_v3.py --dataset YAGO    --gpu 7
    python train_v3.py --dataset WIKI    --gpu 1
    python train_v3.py --dataset ICEWS18 --gpu 2
    python train_v3.py --dataset GDELT   --gpu 4
"""
import os, time, json, random, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from tqdm import tqdm

from aurora_cf.config import parse_args
from aurora_cf.model_v3 import AURORAv3Model
from aurora_cf.data import CFDataLoader, cf_collate
from aurora_cf.evaluate import evaluate

BANNER = """
╔══════════════════════════════════════════════════════════╗
║        AURORA-v3  (2025)                                ║
║  Two-Mode TKG Forecasting                               ║
╠══════════════════════════════════════════════════════════╣
║  YAGO/WIKI   → copy_only  (no neural inner product)     ║
║  ICEWS18/GDELT → fixed gate (no oscillation)            ║
╚══════════════════════════════════════════════════════════╝
"""

HEADER = (
    f"{'Ep':>4} │ {'Time':>6} │ "
    f"{'Loss':>8} {'CE':>8} │ "
    f"{'MRR':>7} {'H@1':>7} {'H@3':>7} {'H@10':>7} │ "
    f"{'LR':>9}"
)
SEP = "─" * len(HEADER)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _row(ep, t, loss_ce, m, lr, best=False):
    star = "★" if best else " "
    return (
        f"{star}{ep:>3} │ {t:>5.1f}s │ "
        f"{loss_ce[0]:>8.4f} {loss_ce[1]:>8.4f} │ "
        f"{m.get('MRR',0):>7.4f} {m.get('Hits@1',0):>7.4f} "
        f"{m.get('Hits@3',0):>7.4f} {m.get('Hits@10',0):>7.4f} │ "
        f"{lr:>9.2e}"
    )


def label_smooth_ce(logits, targets, smoothing=0.05):
    N = logits.size(-1)
    log_prob = F.log_softmax(logits, dim=-1)
    nll = -log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)
    smooth = -log_prob.mean(dim=-1)
    return ((1 - smoothing) * nll + smoothing * smooth).mean()


def main():
    cfg = parse_args()

    print(BANNER)
    mode = "copy-only" if cfg.copy_only else f"fixed-gate={cfg.fixed_gate}"
    print(f"  Dataset : {cfg.dataset}  mode={mode}")
    print(f"  d={cfg.embed_dim}  H={cfg.hist_len}  K={cfg.k_neighbors}")
    print(f"  epochs={cfg.epochs}  lr={cfg.lr}  device=GPU{cfg.gpu}")
    print()

    set_seed(cfg.seed)

    use_cuda = cfg.device == "cuda" and torch.cuda.is_available()
    if use_cuda:
        device = torch.device(f"cuda:{cfg.gpu}")
        free_mem, _ = torch.cuda.mem_get_info(cfg.gpu)
        reserve_gb  = min(12, int(free_mem * 0.85) // (1024**3))
        reserve_n   = reserve_gb * (1024**3) // 4
        print(f"[GPU] {torch.cuda.get_device_name(cfg.gpu)}  "
              f"{free_mem/1024**3:.1f}GB free → {reserve_gb}GB reserved …")
        _placeholder = torch.zeros(reserve_n, device=device)
        print(f"[GPU] Reserved ✓")
    else:
        device = torch.device("cpu")
        _placeholder = None

    loader = CFDataLoader(cfg)

    if _placeholder is not None:
        del _placeholder; torch.cuda.empty_cache()

    model = AURORAv3Model(
        num_entities=loader.num_entities,
        num_relations=loader.num_relations,
        cfg=cfg,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] AURORA-v3  params={n_params:,}  mode={mode}")

    use_amp = (device.type == "cuda")
    scaler  = GradScaler("cuda", enabled=use_amp)

    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup_ep = max(1, int(cfg.epochs * cfg.warmup_ratio))
    warmup    = LinearLR(optim, start_factor=0.1, end_factor=1.0,
                         total_iters=warmup_ep)
    cosine    = CosineAnnealingLR(optim, T_max=cfg.epochs - warmup_ep,
                                  eta_min=cfg.lr * 0.01)
    sched     = SequentialLR(optim, [warmup, cosine], milestones=[warmup_ep])

    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir,  exist_ok=True)
    log_path = os.path.join(cfg.log_dir, f"{cfg.dataset}_v3_log.jsonl")
    best_mrr, best_ep = 0.0, 0

    print(f"\n{'═'*len(HEADER)}")
    print(f"  AURORA-v3 │ {cfg.dataset} │ {mode}")
    print(f"{'═'*len(HEADER)}\n")
    print(HEADER); print(SEP)

    for ep in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()

        dl = DataLoader(
            loader.train_set, batch_size=cfg.batch_size,
            shuffle=True, num_workers=4,
            pin_memory=(device.type == "cuda"),
            drop_last=True, collate_fn=cf_collate,
            persistent_workers=True,
        )

        tot_loss = tot_ce = 0.0
        n = 0
        N = loader.num_entities

        for batch in tqdm(dl, desc=f" ep{ep}", leave=False, dynamic_ncols=True):
            subs, rels, objs, times, ne, nr, nm, rel_copy, ent_copy = batch

            subs = subs.to(device); rels = rels.to(device); objs = objs.to(device)
            ne   = ne.to(device);   nm   = nm.to(device)
            if rel_copy.shape[1] < N:
                rel_copy = torch.cat([rel_copy,
                    torch.zeros(rel_copy.shape[0], N-rel_copy.shape[1])], 1)
            if ent_copy.shape[1] < N:
                ent_copy = torch.cat([ent_copy,
                    torch.zeros(ent_copy.shape[0], N-ent_copy.shape[1])], 1)
            rel_copy = rel_copy.to(device); ent_copy = ent_copy.to(device)

            optim.zero_grad()
            with autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits, query = model(subs, rels, ne, nr, nm, rel_copy, ent_copy)
                ce = label_smooth_ce(logits, objs, cfg.label_smoothing)

                if query is not None and cfg.alpha_infonce > 0:
                    # InfoNCE only for neural mode
                    from aurora_cf.loss import AURORACFLoss
                    loss_d = AURORACFLoss(cfg.label_smoothing,
                                          cfg.alpha_infonce,
                                          cfg.infonce_temp)(
                        logits, query, objs, model.ent_emb.weight)
                    total = loss_d["_total"]
                    ce_val = loss_d["ce"]
                else:
                    total = ce
                    ce_val = ce.item()

            scaler.scale(total).backward()
            scaler.unscale_(optim)
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim); scaler.update()

            tot_loss += total.item(); tot_ce += ce_val if isinstance(ce_val, float) else ce_val.item()
            n += 1

        sched.step()
        elapsed = time.time() - t0
        lr_now  = sched.get_last_lr()[0]

        metrics = {}
        if ep % cfg.eval_every == 0:
            metrics = evaluate(model, loader, split="valid", device=device,
                               batch_size=cfg.batch_size,
                               hits_at=list(cfg.hits_at), verbose=False)

        is_best = metrics.get("MRR", 0) > best_mrr
        if is_best:
            best_mrr = metrics["MRR"]; best_ep = ep
            ck_path = os.path.join(cfg.save_dir, f"{cfg.dataset}_v3_best.pt")
            torch.save({"model": model.state_dict(), "optim": optim.state_dict()},
                       ck_path)
        if ep % 5 == 0:
            torch.save({"model": model.state_dict(), "optim": optim.state_dict()},
                       os.path.join(cfg.save_dir, f"{cfg.dataset}_v3_ep{ep:03d}.pt"))

        print(_row(ep, elapsed, (tot_loss/n, tot_ce/n), metrics, lr_now, is_best))
        with open(log_path, "a") as f:
            f.write(json.dumps({"epoch": ep, "loss": tot_loss/n,
                                "metrics": metrics, "lr": lr_now}) + "\n")

    print(SEP)
    print(f"\n★  Best valid MRR = {best_mrr:.4f}  (epoch {best_ep})")
    print("\nLoading best checkpoint → TEST evaluation …")
    ck_path = os.path.join(cfg.save_dir, f"{cfg.dataset}_v3_best.pt")
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])

    test_m = evaluate(model, loader, split="test", device=device,
                      batch_size=cfg.batch_size,
                      hits_at=list(cfg.hits_at), verbose=True)

    print(f"\n{'═'*50}")
    print(f"  TEST RESULTS — {cfg.dataset}  [{mode}]")
    print(f"{'═'*50}")
    for k, v in test_m.items():
        print(f"  {k:<12}: {v:.4f}")
    print(f"{'═'*50}")

    out = os.path.join(cfg.save_dir, f"{cfg.dataset}_v3_results.json")
    with open(out, "w") as f:
        json.dump({"dataset": cfg.dataset, "mode": mode,
                   "best_epoch": best_ep, "valid_mrr": best_mrr,
                   "test": test_m, "config": cfg.__dict__}, f,
                  indent=2, default=str)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
