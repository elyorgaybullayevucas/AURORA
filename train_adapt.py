#!/usr/bin/env python
"""
AURORA-ADAPT: Adaptive Copy-Neural TKG Forecasting

Gate = f(copy_confidence) — automatically routes:
  High recurrence (YAGO/WIKI) → copy dominant
  Low recurrence (ICEWS18/GDELT) → neural dominant

Usage:
    python train_adapt.py --dataset ICEWS18 --gpu 1
    python train_adapt.py --dataset WIKI    --gpu 2
    python train_adapt.py --dataset YAGO    --gpu 3
    python train_adapt.py --dataset GDELT   --gpu 4
"""
import os, time, json, random, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from aurora_cf.config import parse_args
from aurora_cf.model_adapt import AURORAAdaptModel
from aurora_cf.loss import AURORACFLoss
from aurora_cf.data import CFDataLoader, cf_collate
from aurora_cf.evaluate import evaluate

BANNER = """
╔══════════════════════════════════════════════════════════╗
║        AURORA-ADAPT  (2025)                             ║
║  Adaptive Copy-Neural TKG Forecasting                   ║
╠══════════════════════════════════════════════════════════╣
║  Key innovations:                                       ║
║   1. Adaptive gate: f(copy_confidence)                  ║
║   2. High recurrence → copy dominates automatically     ║
║   3. Low recurrence  → neural dominates automatically   ║
║   4. Gate stable during training (based on fixed data)  ║
║   5. One model, all datasets — no manual tuning         ║
╚══════════════════════════════════════════════════════════╝
"""

HEADER = (
    f"{'Ep':>4} │ {'Time':>6} │ "
    f"{'Loss':>8} {'CE':>8} {'NCE':>8} │ "
    f"{'MRR':>7} {'H@1':>7} {'H@3':>7} {'H@10':>7} │ "
    f"{'Gate':>6} │ {'LR':>9}"
)
SEP = "─" * len(HEADER)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _row(ep, t, ld, m, gate, lr, best=False):
    star = "★" if best else " "
    return (
        f"{star}{ep:>3} │ {t:>5.1f}s │ "
        f"{ld['total']:>8.4f} {ld['ce']:>8.4f} {ld['infonce']:>8.4f} │ "
        f"{m.get('MRR',0):>7.4f} {m.get('Hits@1',0):>7.4f} "
        f"{m.get('Hits@3',0):>7.4f} {m.get('Hits@10',0):>7.4f} │ "
        f"{gate:>6.3f} │ {lr:>9.2e}"
    )


def main():
    cfg = parse_args()

    print(BANNER)
    print(f"  Dataset    : {cfg.dataset}")
    print(f"  embed_dim  : {cfg.embed_dim}")
    print(f"  k_neighbors: {cfg.k_neighbors}")
    print(f"  hist_len   : {cfg.hist_len}")
    print(f"  epochs     : {cfg.epochs}")
    print(f"  lr         : {cfg.lr}")
    print(f"  device     : {cfg.device} (GPU {cfg.gpu})")
    print()

    set_seed(cfg.seed)

    # ── GPU reservation ───────────────────────────────────────────────────────
    use_cuda = cfg.device == "cuda" and torch.cuda.is_available()
    if use_cuda:
        device = torch.device(f"cuda:{cfg.gpu}")
        free_mem, _ = torch.cuda.mem_get_info(cfg.gpu)
        reserve_gb  = min(12, int(free_mem * 0.85) // (1024**3))
        reserve_n   = reserve_gb * (1024**3) // 4
        print(f"[Device] GPU {cfg.gpu} — {torch.cuda.get_device_name(cfg.gpu)}")
        print(f"[GPU]    {free_mem/1024**3:.1f} GB free → {reserve_gb} GB reserved …")
        _placeholder = torch.zeros(reserve_n, device=device)
        print(f"[GPU]    Reserved ✓")
    else:
        device = torch.device("cpu")
        _placeholder = None

    # ── Data ──────────────────────────────────────────────────────────────────
    loader = CFDataLoader(cfg)

    if _placeholder is not None:
        del _placeholder; torch.cuda.empty_cache()

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AURORAAdaptModel(
        num_entities=loader.num_entities,
        num_relations=loader.num_relations,
        cfg=cfg,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] AURORA-ADAPT  params={n_params:,}")

    use_amp = (device.type == "cuda")
    scaler  = GradScaler("cuda", enabled=use_amp)
    loss_fn = AURORACFLoss(cfg.label_smoothing, cfg.alpha_infonce, cfg.infonce_temp)

    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup_ep = max(1, int(cfg.epochs * cfg.warmup_ratio))
    warmup    = LinearLR(optim, start_factor=0.1, end_factor=1.0,
                         total_iters=warmup_ep)
    cosine    = CosineAnnealingLR(optim, T_max=cfg.epochs - warmup_ep,
                                  eta_min=cfg.lr * 0.01)
    sched     = SequentialLR(optim, [warmup, cosine], milestones=[warmup_ep])

    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir,  exist_ok=True)
    log_path = os.path.join(cfg.log_dir, f"{cfg.dataset}_adapt_log.jsonl")
    best_mrr, best_ep = 0.0, 0

    print(f"\n{'═'*len(HEADER)}")
    print(f"  AURORA-ADAPT │ {cfg.dataset} │ epochs={cfg.epochs} │ "
          f"d={cfg.embed_dim} │ H={cfg.hist_len} │ K={cfg.k_neighbors}")
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

        tot_total = tot_ce = tot_nce = gate_sum = 0.0
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
                loss_d = loss_fn(logits, query, objs, model.ent_emb.weight)

            scaler.scale(loss_d["_total"]).backward()
            scaler.unscale_(optim)
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim); scaler.update()

            tot_total += loss_d["total"]; tot_ce += loss_d["ce"]
            tot_nce   += loss_d["infonce"]; n += 1

            # track mean gate value (diagnostic)
            with torch.no_grad():
                c_max  = rel_copy.max(dim=-1, keepdim=True)[0]
                c_sum  = rel_copy.sum(dim=-1, keepdim=True).clamp(1e-8)
                c_conf = c_max / c_sum
                gi     = torch.cat([torch.log1p(c_max), c_conf], -1)
                gate_sum += torch.sigmoid(model.gate_net(gi)).mean().item()

        sched.step()
        elapsed = time.time() - t0
        lr_now  = sched.get_last_lr()[0]
        ld = {"total": tot_total/n, "ce": tot_ce/n, "infonce": tot_nce/n}
        avg_gate = gate_sum / n

        metrics = {}
        if ep % cfg.eval_every == 0:
            metrics = evaluate(model, loader, split="valid", device=device,
                               batch_size=cfg.batch_size,
                               hits_at=list(cfg.hits_at), verbose=False)

        is_best = metrics.get("MRR", 0) > best_mrr
        if is_best:
            best_mrr = metrics["MRR"]; best_ep = ep
            ck_path = os.path.join(cfg.save_dir, f"{cfg.dataset}_adapt_best.pt")
            torch.save({"model": model.state_dict(), "optim": optim.state_dict()},
                       ck_path)
        if ep % 5 == 0:
            torch.save({"model": model.state_dict(), "optim": optim.state_dict()},
                       os.path.join(cfg.save_dir, f"{cfg.dataset}_adapt_ep{ep:03d}.pt"))

        print(_row(ep, elapsed, ld, metrics, avg_gate, lr_now, is_best))
        with open(log_path, "a") as f:
            f.write(json.dumps({"epoch": ep, "loss": ld, "metrics": metrics,
                                "gate": avg_gate, "lr": lr_now}) + "\n")

    # ── Final test ────────────────────────────────────────────────────────────
    print(SEP)
    print(f"\n★  Best valid MRR = {best_mrr:.4f}  (epoch {best_ep})")
    print("\nLoading best checkpoint → TEST evaluation …")
    ck_path = os.path.join(cfg.save_dir, f"{cfg.dataset}_adapt_best.pt")
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])

    test_m = evaluate(model, loader, split="test", device=device,
                      batch_size=cfg.batch_size,
                      hits_at=list(cfg.hits_at), verbose=True)

    print(f"\n{'═'*50}")
    print(f"  TEST RESULTS — {cfg.dataset}")
    print(f"{'═'*50}")
    for k, v in test_m.items():
        print(f"  {k:<12}: {v:.4f}")
    print(f"{'═'*50}")

    out = os.path.join(cfg.save_dir, f"{cfg.dataset}_adapt_results.json")
    with open(out, "w") as f:
        json.dump({"dataset": cfg.dataset, "best_epoch": best_ep,
                   "valid_mrr": best_mrr, "test": test_m,
                   "config": cfg.__dict__}, f, indent=2, default=str)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
