#!/usr/bin/env python
"""
HAWK — training / evaluation.

    python train_hawk.py --dataset ICEWS18 --gpu 2
    python train_hawk.py --dataset WIKI    --gpu 3
    python train_hawk.py --dataset YAGO    --gpu 7
    python train_hawk.py --dataset GDELT   --gpu 4

Ablations for the paper table:
    --hazard_off   structural intensity only
    --struct_off   recurrence intensity only
    --phase_off    recurrence without the phase basis (monotone-only)
"""
import os, sys, time, json, random, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from aurora_cf.hawk_config import parse_args
from aurora_cf.model_hawk import HAWK
from aurora_cf.data_tpp import TPPDataLoader, tpp_collate

BANNER = r"""
╔═══════════════════════════════════════════════════════════════════════╗
║  HAWK — Hazard-Aware Knowledge forecasting                            ║
╠═══════════════════════════════════════════════════════════════════════╣
║  λ(o|t) = λ_nov(o | G_<t, s, r)  +  λ_rec(o | H_o, s, r)              ║
║                                                                       ║
║  superposition of two point processes ⇒ score = logaddexp(f_nov,f_rec)║
║  no mixing gate exists, so no gate can collapse                       ║
║                                                                       ║
║  λ_rec = Σ_j w_j(r,o,φ)·κ_j(Δt / mean_gap)   ,  w_j ≥ 0               ║
║  prior work uses f(count)·exp(−λΔt): monotone in Δt, so periodic and  ║
║  refractory recurrence are unreachable. The RBF phase basis makes     ║
║  them reachable.                                                      ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

HDR = (f"{'Ep':>4} │ {'Time':>7} │ {'Loss':>8} │ "
       f"{'MRR':>7} {'H@1':>7} {'H@3':>7} {'H@10':>7} │ "
       f"{'recB':>7} │ {'LR':>9}")
SEP = "─" * len(HDR)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, loader, split, device, cfg, verbose=True):
    model.eval()
    ds = {"valid": loader.valid_set, "test": loader.test_set}[split]
    dl = DataLoader(ds, batch_size=cfg.eval_batch_size, shuffle=False,
                    num_workers=cfg.num_workers, collate_fn=tpp_collate,
                    pin_memory=True)
    hits_at = tuple(cfg.hits_at)
    mrr = 0.0; hits = {k: 0.0 for k in hits_at}; total = 0

    for b in tqdm(dl, desc=f"eval[{split}]", disable=not verbose,
                  dynamic_ncols=True, leave=False):
        (subs, rels, objs, times, ne, nr, ndt, nm,
         sup_ids, sup_feat, sup_mask) = [x.to(device, non_blocking=True)
                                         for x in b]
        with autocast("cuda", dtype=torch.bfloat16,
                      enabled=device.type == "cuda"):
            logits = model(subs, rels, ne, nr, ndt, nm,
                           sup_ids, sup_feat, sup_mask)
        logits = logits.float()
        B = logits.size(0)

        # ── vectorised filtered ranking ──────────────────────────────────────
        # Build every (row, other-true-answer) pair once, mask them all in a
        # single scatter, restore the target, then rank without a Python loop.
        sub_c, rel_c, tim_c = (subs.cpu().numpy(), rels.cpu().numpy(),
                               times.cpu().numpy())
        rows, cols = [], []
        for i in range(B):
            ans = loader.index.answers(sub_c[i], rel_c[i], tim_c[i])
            if len(ans):
                cols.append(ans)
                rows.append(np.full(len(ans), i, dtype=np.int64))
        tgt = logits.gather(1, objs.view(-1, 1))
        if rows:
            ri = torch.from_numpy(np.concatenate(rows)).to(device)
            ci = torch.from_numpy(np.concatenate(cols).astype(np.int64)).to(device)
            logits.index_put_((ri, ci),
                              torch.full((len(ri),), float("-inf"),
                                         device=device))
        logits.scatter_(1, objs.view(-1, 1), tgt)

        rank = (logits > tgt).sum(1) + 1
        mrr += (1.0 / rank.float()).sum().item()
        for k in hits_at:
            hits[k] += (rank <= k).sum().item()
        total += B

    res = {"MRR": mrr / total}
    res.update({f"Hits@{k}": hits[k] / total for k in hits_at})
    return res


def main():
    cfg = parse_args()
    variant = ("structural-only" if cfg.hazard_off else
               "recurrence-only" if cfg.struct_off else
               "no-phase" if cfg.phase_off else "full")

    print(BANNER)
    print(f"  dataset={cfg.dataset}   variant={variant}")
    print(f"  d={cfg.embed_dim} layers={cfg.n_layers} heads={cfg.n_heads} "
          f"H={cfg.hist_len} K={cfg.k_neighbors} S={cfg.max_support}")
    print(f"  bs={cfg.batch_size} lr={cfg.lr} epochs={cfg.epochs} "
          f"workers={cfg.num_workers} GPU={cfg.gpu}\n")
    set_seed(cfg.seed)

    use_cuda = cfg.device == "cuda" and torch.cuda.is_available()
    if use_cuda:
        device = torch.device(f"cuda:{cfg.gpu}")
        torch.cuda.set_device(device)
        free, tot = torch.cuda.mem_get_info(cfg.gpu)
        gb = max(1, min(cfg.reserve_gb, int(free * 0.9) // (1024 ** 3)))
        print(f"[GPU] {torch.cuda.get_device_name(cfg.gpu)}  "
              f"{free/1024**3:.1f}/{tot/1024**3:.1f} GB free → "
              f"reserving {gb} GB")
        _ph = torch.empty(gb * (1024 ** 3) // 2, dtype=torch.float16,
                          device=device)
        print("[GPU] reserved ✓")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device, _ph = torch.device("cpu"), None

    loader = TPPDataLoader(cfg)
    if _ph is not None:
        del _ph; torch.cuda.empty_cache()

    sr_tr, s_tr = loader.support_diagnostics("train")
    sr_te, s_te = loader.support_diagnostics("test")
    print(f"[support] train recall={sr_tr:.4f} avg|S|={s_tr:.1f}   "
          f"test recall={sr_te:.4f} avg|S|={s_te:.1f}")

    model = HAWK(loader.num_entities, loader.num_relations, cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] HAWK  params={n_par:,}")

    scaler = GradScaler("cuda", enabled=use_cuda)
    optim = AdamW(model.parameters(), lr=cfg.lr,
                  weight_decay=cfg.weight_decay, betas=(0.9, 0.98))
    warm = max(1, int(cfg.epochs * cfg.warmup_ratio))
    sched = SequentialLR(optim,
                         [LinearLR(optim, 0.05, 1.0, total_iters=warm),
                          CosineAnnealingLR(optim, T_max=max(1, cfg.epochs - warm),
                                            eta_min=cfg.lr * 0.02)],
                         milestones=[warm])

    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    name = f"{cfg.dataset}_hawk_{variant}"
    log_path = os.path.join(cfg.log_dir, f"{name}.jsonl")
    best, best_ep, bad = 0.0, 0, 0

    print(f"\n{'═'*len(HDR)}")
    print(f"  HAWK │ {cfg.dataset} │ {variant}")
    print(f"{'═'*len(HDR)}\n")
    print(HDR); print(SEP)

    for ep in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        dl = DataLoader(loader.train_set, batch_size=cfg.batch_size,
                        shuffle=True, num_workers=cfg.num_workers,
                        pin_memory=use_cuda, drop_last=True,
                        collate_fn=tpp_collate,
                        persistent_workers=cfg.num_workers > 0,
                        prefetch_factor=4 if cfg.num_workers > 0 else None)
        tot_l, nb = 0.0, 0
        for b in tqdm(dl, desc=f" ep{ep}", leave=False, dynamic_ncols=True):
            (subs, rels, objs, times, ne, nr, ndt, nm,
             sup_ids, sup_feat, sup_mask) = [x.to(device, non_blocking=True)
                                             for x in b]
            optim.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
                logits = model(subs, rels, ne, nr, ndt, nm,
                               sup_ids, sup_feat, sup_mask)
                loss = F.cross_entropy(logits.float(), objs,
                                       label_smoothing=cfg.label_smoothing)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim); scaler.update()
            tot_l += loss.item(); nb += 1

        sched.step()
        el = time.time() - t0
        m = evaluate(model, loader, "valid", device, cfg,
                     verbose=False) if ep % cfg.eval_every == 0 else {}

        is_best = m.get("MRR", 0) > best
        if is_best:
            best, best_ep, bad = m["MRR"], ep, 0
            torch.save({"model": model.state_dict(), "cfg": vars(cfg)},
                       os.path.join(cfg.save_dir, f"{name}_best.pt"))
        else:
            bad += 1

        rb = model.rec_bias.item()
        print(f"{'★' if is_best else ' '}{ep:>3} │ {el:>6.1f}s │ "
              f"{tot_l/nb:>8.4f} │ {m.get('MRR',0):>7.4f} "
              f"{m.get('Hits@1',0):>7.4f} {m.get('Hits@3',0):>7.4f} "
              f"{m.get('Hits@10',0):>7.4f} │ {rb:>7.3f} │ "
              f"{sched.get_last_lr()[0]:>9.2e}", flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps({"epoch": ep, "loss": tot_l / nb,
                                "metrics": m, "rec_bias": rb}) + "\n")
        if bad >= cfg.patience:
            print(f"\n[early-stop] {cfg.patience} epochs without improvement.")
            break

    print(SEP)
    print(f"\n★ best valid MRR = {best:.4f} (epoch {best_ep})")
    ck = os.path.join(cfg.save_dir, f"{name}_best.pt")
    if os.path.exists(ck):
        model.load_state_dict(torch.load(ck, map_location=device,
                                         weights_only=False)["model"])
    test_m = evaluate(model, loader, "test", device, cfg, verbose=True)

    print(f"\n{'═'*54}")
    print(f"  TEST — {cfg.dataset}   [{variant}]")
    print(f"{'═'*54}")
    for k, v in test_m.items():
        print(f"  {k:<12}: {v*100:.2f}")
    print(f"{'═'*54}")

    if not cfg.hazard_off:
        try:
            ph = np.linspace(0.0, 8.0, 80)
            curves = {f"rel_{r}": model.hazard_curve(r, 0, ph, device)
                      .float().cpu().numpy().tolist()
                      for r in range(min(8, loader.num_relations))}
            with open(os.path.join(cfg.save_dir,
                                   f"{name}_hazard.json"), "w") as f:
                json.dump({"phase": ph.tolist(), "curves": curves}, f, indent=2)
            print(f"hazard curves → {cfg.save_dir}/{name}_hazard.json")
        except Exception as e:
            print(f"[warn] hazard export failed: {e}")

    with open(os.path.join(cfg.save_dir, f"{name}_results.json"), "w") as f:
        json.dump({"dataset": cfg.dataset, "variant": variant,
                   "best_epoch": best_ep, "valid_mrr": best, "test": test_m,
                   "support": {"train_recall": sr_tr, "train_avg": s_tr,
                               "test_recall": sr_te, "test_avg": s_te},
                   "params": n_par, "config": vars(cfg)},
                  f, indent=2, default=str)
    print(f"saved → {cfg.save_dir}/{name}_results.json")


if __name__ == "__main__":
    main()
