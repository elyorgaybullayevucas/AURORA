#!/usr/bin/env python
"""
NHC-Full training.

    python train_full.py --dataset YAGO    --gpu 7
    python train_full.py --dataset WIKI    --gpu 3
    python train_full.py --dataset ICEWS18 --gpu 4
    python train_full.py --dataset GDELT   --gpu 5

Ablations (for the paper table):
    --hazard_off     structural backbone only
    --phase_off      hazard without the phase basis (monotone-only features)
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
from aurora_cf.model_nhc_full import NHCFullModel
from aurora_cf.data_full import FullDataLoader, full_collate

BANNER = """
╔════════════════════════════════════════════════════════════════════╗
║   NHC-Full — Neural Hazard Copy over a temporal graph encoder      ║
╠════════════════════════════════════════════════════════════════════╣
║  structural : relation-aware snapshot attention → GRU → ConvTransE ║
║               ranks ALL entities  (recall)                         ║
║  hazard     : log λ(o|H_o,s,r) = g_θ(φ(H_o), e_r, e_o)             ║
║               φ includes PHASE = Δt/mean_gap + RBF basis           ║
║               ranks the candidate set  (precision)                 ║
║  score(o)   = structural(o) + 1[o∈S]·hazard(o)   — additive, no gate║
╚════════════════════════════════════════════════════════════════════╝
"""

HEADER = (f"{'Ep':>4} │ {'Time':>7} │ {'Loss':>8} │ "
          f"{'MRR':>7} {'H@1':>7} {'H@3':>7} {'H@10':>7} │ "
          f"{'HzScl':>6} │ {'LR':>9}")
SEP = "─" * len(HEADER)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, loader, split, device, bs, hits_at=(1, 3, 10),
             verbose=True):
    model.eval()
    ds = loader.valid_set if split == "valid" else loader.test_set
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0,
                    collate_fn=full_collate)
    mrr, hits, total = 0.0, {k: 0.0 for k in hits_at}, 0

    for b in tqdm(dl, desc=f"Eval [{split}]", disable=not verbose,
                  dynamic_ncols=True):
        (subs, rels, objs, times, ne, nr, nm,
         sup_ids, sup_feat, sup_mask) = [x.to(device) for x in b]
        logits, _ = model(subs, rels, ne, nr, nm, sup_ids, sup_feat, sup_mask)
        for i in range(subs.size(0)):
            s, r, t = subs[i].item(), rels[i].item(), times[i].item()
            o = objs[i].item()
            sc = logits[i].clone()
            for of in loader.index.all_answers.get((s, r, t), ()):
                if of != o:
                    sc[of] = float("-inf")
            rank = (sc > sc[o]).sum().item() + 1
            mrr += 1.0 / rank
            for k in hits_at:
                hits[k] += float(rank <= k)
            total += 1

    res = {"MRR": mrr / total}
    res.update({f"Hits@{k}": hits[k] / total for k in hits_at})
    return res


def main():
    cfg = parse_args()
    print(BANNER)
    tag = "full"
    if cfg.hazard_off:
        tag = "structural-only"
    elif cfg.phase_off:
        tag = "hazard-no-phase"
    print(f"  Dataset={cfg.dataset}  variant={tag}")
    print(f"  d={cfg.embed_dim} dh={cfg.hazard_dim} H={cfg.hist_len} "
          f"K={cfg.k_neighbors} S={cfg.max_support} relTopK={cfg.rel_topk}")
    print(f"  epochs={cfg.epochs} lr={cfg.lr} bs={cfg.batch_size} "
          f"GPU={cfg.gpu}\n")
    set_seed(cfg.seed)

    use_cuda = cfg.device == "cuda" and torch.cuda.is_available()
    if use_cuda:
        device = torch.device(f"cuda:{cfg.gpu}")
        free, _ = torch.cuda.mem_get_info(cfg.gpu)
        gb = min(12, int(free * 0.85) // (1024 ** 3))
        print(f"[GPU] {torch.cuda.get_device_name(cfg.gpu)}  "
              f"{free/1024**3:.1f}GB free → {gb}GB reserved …")
        _ph = torch.zeros(gb * (1024 ** 3) // 4, device=device)
        print("[GPU] Reserved ✓")
    else:
        device, _ph = torch.device("cpu"), None

    loader = FullDataLoader(cfg)
    if _ph is not None:
        del _ph; torch.cuda.empty_cache()

    model = NHCFullModel(loader.num_entities, loader.num_relations,
                         cfg).to(device)
    print(f"[Model] NHC-Full  params="
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    scaler = GradScaler("cuda", enabled=use_cuda)
    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warm = max(1, int(cfg.epochs * cfg.warmup_ratio))
    sched = SequentialLR(optim,
                         [LinearLR(optim, 0.1, 1.0, total_iters=warm),
                          CosineAnnealingLR(optim, T_max=cfg.epochs - warm,
                                            eta_min=cfg.lr * 0.01)],
                         milestones=[warm])

    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    name = f"{cfg.dataset}_{tag}"
    log_path = os.path.join(cfg.log_dir, f"{name}.jsonl")
    best, best_ep, patience = 0.0, 0, 0

    print(f"\n{'═'*len(HEADER)}")
    print(f"  NHC-Full │ {cfg.dataset} │ {tag} │ "
          f"train support_recall={loader.train_set.support_recall:.4f}  "
          f"avg|S|={loader.train_set.avg_support:.1f}")
    print(f"{'═'*len(HEADER)}\n")
    print(HEADER); print(SEP)

    for ep in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        dl = DataLoader(loader.train_set, batch_size=cfg.batch_size,
                        shuffle=True, num_workers=cfg.num_workers,
                        pin_memory=use_cuda, drop_last=True,
                        collate_fn=full_collate,
                        persistent_workers=cfg.num_workers > 0)
        tot, n = 0.0, 0
        for b in tqdm(dl, desc=f" ep{ep}", leave=False, dynamic_ncols=True):
            (subs, rels, objs, times, ne, nr, nm,
             sup_ids, sup_feat, sup_mask) = [x.to(device, non_blocking=True)
                                             for x in b]
            optim.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
                logits, _ = model(subs, rels, ne, nr, nm,
                                  sup_ids, sup_feat, sup_mask)
                loss = F.cross_entropy(logits.float(), objs,
                                       label_smoothing=cfg.label_smoothing)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim); scaler.update()
            tot += loss.item(); n += 1

        sched.step()
        el = time.time() - t0
        m = {}
        if ep % cfg.eval_every == 0:
            m = evaluate(model, loader, "valid", device, cfg.batch_size,
                         tuple(cfg.hits_at), verbose=False)
        is_best = m.get("MRR", 0) > best
        if is_best:
            best, best_ep, patience = m["MRR"], ep, 0
            torch.save({"model": model.state_dict()},
                       os.path.join(cfg.save_dir, f"{name}_best.pt"))
        else:
            patience += 1

        hs = model.hazard_scale.item()
        print(f"{'★' if is_best else ' '}{ep:>3} │ {el:>6.1f}s │ "
              f"{tot/n:>8.4f} │ {m.get('MRR',0):>7.4f} "
              f"{m.get('Hits@1',0):>7.4f} {m.get('Hits@3',0):>7.4f} "
              f"{m.get('Hits@10',0):>7.4f} │ {hs:>6.3f} │ "
              f"{sched.get_last_lr()[0]:>9.2e}")
        with open(log_path, "a") as f:
            f.write(json.dumps({"epoch": ep, "loss": tot / n, "metrics": m,
                                "hazard_scale": hs}) + "\n")
        if patience >= cfg.patience:
            print(f"\n[early-stop] no improvement for {cfg.patience} epochs.")
            break

    print(SEP)
    print(f"\n★  Best valid MRR = {best:.4f}  (epoch {best_ep})")
    ck = os.path.join(cfg.save_dir, f"{name}_best.pt")
    if os.path.exists(ck):
        model.load_state_dict(torch.load(ck, map_location=device,
                                         weights_only=True)["model"])
    test_m = evaluate(model, loader, "test", device, cfg.batch_size,
                      tuple(cfg.hits_at), verbose=True)

    print(f"\n{'═'*52}")
    print(f"  TEST — {cfg.dataset}  [{tag}]")
    print(f"{'═'*52}")
    for k, v in test_m.items():
        print(f"  {k:<12}: {v:.4f}")
    print(f"{'═'*52}")

    if not cfg.hazard_off:
        try:
            ph = np.linspace(0.05, 6.0, 60)
            curves = {f"rel_{r}": model.hazard_curve(r, 0, ph, device)
                      .float().cpu().numpy().tolist()
                      for r in range(min(6, loader.num_relations))}
            with open(os.path.join(cfg.save_dir,
                                   f"{name}_hazard_curves.json"), "w") as f:
                json.dump({"phases": ph.tolist(), "curves": curves}, f, indent=2)
        except Exception as e:
            print(f"[warn] hazard curve export failed: {e}")

    with open(os.path.join(cfg.save_dir, f"{name}_results.json"), "w") as f:
        json.dump({"dataset": cfg.dataset, "variant": tag,
                   "best_epoch": best_ep, "valid_mrr": best, "test": test_m,
                   "train_support_recall": loader.train_set.support_recall,
                   "train_avg_support": loader.train_set.avg_support,
                   "config": cfg.__dict__}, f, indent=2, default=str)
    print(f"Saved → {cfg.save_dir}/{name}_results.json")


if __name__ == "__main__":
    main()
