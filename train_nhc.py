#!/usr/bin/env python
"""
NHC — Neural Hazard Copy.  Training entry point.

    python train_nhc.py --dataset YAGO    --gpu 7
    python train_nhc.py --dataset WIKI    --gpu 1
    python train_nhc.py --dataset ICEWS18 --gpu 2
    python train_nhc.py --dataset GDELT   --gpu 4
"""
import os, time, json, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from aurora_cf.config import parse_args
from aurora_cf.model_hazard import NHCModel
from aurora_cf.data_hazard import HazardDataLoader, hazard_collate
from aurora_cf.evaluate_hazard import evaluate

BANNER = """
╔════════════════════════════════════════════════════════════════╗
║   NHC — Neural Hazard Copy                                    ║
║   Temporal Knowledge Graph Forecasting                        ║
╠════════════════════════════════════════════════════════════════╣
║  Prior work scores recurrence with  f(count)·exp(−λ·Δt)       ║
║    → monotone in Δt → cannot express periodic recurrence      ║
║                                                                ║
║  NHC learns the conditional intensity                          ║
║    log λ(o | H_o, s, r) = g_θ(φ(H_o), e_r, e_o)               ║
║  with an explicit PHASE feature  Δt / mean_gap  and an RBF     ║
║  basis over it → non-monotone hazard becomes representable.    ║
╚════════════════════════════════════════════════════════════════╝
"""

HEADER = (
    f"{'Ep':>4} │ {'Time':>6} │ {'Loss':>8} │ "
    f"{'MRR':>7} {'H@1':>7} {'H@3':>7} {'H@10':>7} │ "
    f"{'SupR':>6} {'SemW':>6} │ {'LR':>9}"
)
SEP = "─" * len(HEADER)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    cfg = parse_args()
    print(BANNER)
    print(f"  Dataset={cfg.dataset}  d={cfg.embed_dim}  dh={cfg.hazard_dim}  "
          f"S_max={cfg.max_support}")
    print(f"  epochs={cfg.epochs}  lr={cfg.lr}  bs={cfg.batch_size}  "
          f"GPU={cfg.gpu}\n")

    set_seed(cfg.seed)

    use_cuda = cfg.device == "cuda" and torch.cuda.is_available()
    if use_cuda:
        device = torch.device(f"cuda:{cfg.gpu}")
        free_mem, _ = torch.cuda.mem_get_info(cfg.gpu)
        reserve_gb = min(12, int(free_mem * 0.85) // (1024 ** 3))
        print(f"[GPU] {torch.cuda.get_device_name(cfg.gpu)}  "
              f"{free_mem/1024**3:.1f}GB free → {reserve_gb}GB reserved …")
        _ph = torch.zeros(reserve_gb * (1024 ** 3) // 4, device=device)
        print("[GPU] Reserved ✓")
    else:
        device = torch.device("cpu")
        _ph = None

    loader = HazardDataLoader(cfg)
    if _ph is not None:
        del _ph; torch.cuda.empty_cache()

    model = NHCModel(loader.num_entities, loader.num_relations, cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] NHC  params={n_par:,}")

    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warm = max(1, int(cfg.epochs * cfg.warmup_ratio))
    sched = SequentialLR(
        optim,
        [LinearLR(optim, 0.1, 1.0, total_iters=warm),
         CosineAnnealingLR(optim, T_max=cfg.epochs - warm,
                           eta_min=cfg.lr * 0.01)],
        milestones=[warm])

    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    log_path = os.path.join(cfg.log_dir, f"{cfg.dataset}_nhc_log.jsonl")
    best_mrr, best_ep, patience = 0.0, 0, 0

    print(f"\n{'═'*len(HEADER)}")
    print(f"  NHC │ {cfg.dataset} │ train support recall = "
          f"{loader.train_set.support_recall:.4f}")
    print(f"{'═'*len(HEADER)}\n")
    print(HEADER); print(SEP)

    for ep in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        dl = DataLoader(loader.train_set, batch_size=cfg.batch_size,
                        shuffle=True, num_workers=4,
                        pin_memory=use_cuda, drop_last=True,
                        collate_fn=hazard_collate, persistent_workers=True)

        tot, n = 0.0, 0
        for batch in tqdm(dl, desc=f" ep{ep}", leave=False, dynamic_ncols=True):
            subs, rels, objs, times, sup_ids, sup_feat, sup_mask = batch
            subs = subs.to(device, non_blocking=True)
            rels = rels.to(device, non_blocking=True)
            objs = objs.to(device, non_blocking=True)
            sup_ids = sup_ids.to(device, non_blocking=True)
            sup_feat = sup_feat.to(device, non_blocking=True)
            sup_mask = sup_mask.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits, _ = model(subs, rels, sup_ids, sup_feat, sup_mask)
                loss = F.cross_entropy(logits.float(), objs,
                                       label_smoothing=cfg.label_smoothing)

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim); scaler.update()

            tot += loss.item(); n += 1

        sched.step()
        el = time.time() - t0
        lr_now = sched.get_last_lr()[0]
        sem_w = (torch.sigmoid(model.log_sem_w) * model.sem_cap).item()

        m = {}
        if ep % cfg.eval_every == 0:
            m = evaluate(model, loader, "valid", device,
                         cfg.batch_size, tuple(cfg.hits_at), verbose=False)

        is_best = m.get("MRR", 0) > best_mrr
        if is_best:
            best_mrr, best_ep, patience = m["MRR"], ep, 0
            torch.save({"model": model.state_dict()},
                       os.path.join(cfg.save_dir, f"{cfg.dataset}_nhc_best.pt"))
        else:
            patience += 1

        star = "★" if is_best else " "
        print(f"{star}{ep:>3} │ {el:>5.1f}s │ {tot/n:>8.4f} │ "
              f"{m.get('MRR',0):>7.4f} {m.get('Hits@1',0):>7.4f} "
              f"{m.get('Hits@3',0):>7.4f} {m.get('Hits@10',0):>7.4f} │ "
              f"{m.get('SupportRecall',0):>6.3f} {sem_w:>6.3f} │ "
              f"{lr_now:>9.2e}")

        with open(log_path, "a") as f:
            f.write(json.dumps({"epoch": ep, "loss": tot / n,
                                "metrics": m, "sem_w": sem_w,
                                "lr": lr_now}) + "\n")

        if patience >= cfg.patience:
            print(f"\n[early-stop] no improvement for {cfg.patience} epochs.")
            break

    print(SEP)
    print(f"\n★  Best valid MRR = {best_mrr:.4f}  (epoch {best_ep})")
    ck = os.path.join(cfg.save_dir, f"{cfg.dataset}_nhc_best.pt")
    if os.path.exists(ck):
        model.load_state_dict(
            torch.load(ck, map_location=device, weights_only=True)["model"])

    test_m = evaluate(model, loader, "test", device,
                      cfg.batch_size, tuple(cfg.hits_at), verbose=True)

    print(f"\n{'═'*52}")
    print(f"  TEST RESULTS — {cfg.dataset}  [NHC]")
    print(f"{'═'*52}")
    for k, v in test_m.items():
        print(f"  {k:<14}: {v:.4f}")
    print(f"{'═'*52}")

    # ── learned hazard shape (the paper figure) ──────────────────────────────
    try:
        phases = np.linspace(0.05, 6.0, 60)
        curves = {}
        for r in range(min(4, loader.num_relations)):
            c = model.hazard_curve(r, 0, phases, device).float().cpu().numpy()
            curves[f"rel_{r}"] = c.tolist()
        curve_out = os.path.join(cfg.save_dir,
                                 f"{cfg.dataset}_hazard_curves.json")
        with open(curve_out, "w") as f:
            json.dump({"phases": phases.tolist(), "curves": curves}, f, indent=2)
        print(f"Hazard curves → {curve_out}")
    except Exception as e:
        print(f"[warn] hazard curve export failed: {e}")

    out = os.path.join(cfg.save_dir, f"{cfg.dataset}_nhc_results.json")
    with open(out, "w") as f:
        json.dump({"dataset": cfg.dataset, "best_epoch": best_ep,
                   "valid_mrr": best_mrr, "test": test_m,
                   "train_support_recall": loader.train_set.support_recall,
                   "config": cfg.__dict__}, f, indent=2, default=str)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
