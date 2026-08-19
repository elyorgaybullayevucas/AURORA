#!/usr/bin/env python
"""
KAIROS — training and evaluation.

    python train_kairos.py --dataset ICEWS18 --gpu 2
    python train_kairos.py --dataset GDELT   --gpu 4

Ablations:
    --rec_off      structural intensity only
    --struct_off   recurrence intensity only
    --phase_off    recurrence without the phase basis (monotone-only kernel)

Every run reports THREE evaluation protocols, because published numbers are
not comparable across them:
    raw                  no filtering
    time-aware filtered  filter other true answers at the query timestamp
    time-unaware filter  filter every object ever seen for (s, r)
RE-GCN / TiRGN / CENET report time-aware filtered; GAttNHP reports raw.
It also reports metrics split by whether a query is MONOTONE-BLOCKED, which
is where the contribution predicts the gain must appear.
"""
import os, time, json, random, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from kairos.config import parse_args
from kairos.data import KairosData, identity_collate
from kairos.model import KAIROS

BANNER = r"""
╔═══════════════════════════════════════════════════════════════════════════╗
║  KAIROS — phase-conditioned recurrence for TKG forecasting                ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  λ(o) = λ_struct(o | G_<t, s, r)  +  λ_rec(o | H_o, s, r)                 ║
║  superposition ⇒ logaddexp, not a sum of logits; no gate to collapse      ║
║                                                                           ║
║  prior recurrence scores are f(count)·g(Δt) with g non-increasing         ║
║  (CyGNet, CENET, TiRGN, RE-GCN, DaeMon; GAttNHP learns γ but keeps exp)   ║
║  ⇒ a distractor with smaller Δt and larger count can never be outranked   ║
║                                                                           ║
║  λ_rec = Σ_j w_j(r,o,φ)·κ_j(Δt / mean_gap),  w_j ≥ 0                      ║
║  non-monotone in Δt ⇒ those queries become reachable                      ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

HDR = (f"{'Ep':>4} │ {'Time':>7} │ {'Loss':>8} │ "
       f"{'MRR':>7} {'H@1':>7} {'H@3':>7} {'H@10':>7} │ "
       f"{'recB':>7} │ {'LR':>9}")
SEP = "─" * len(HDR)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def to_dev(item, dev):
    return dict(
        t=item["t"],
        subs=item["subs"].to(dev, non_blocking=True),
        rels=item["rels"].to(dev, non_blocking=True),
        objs=item["objs"].to(dev, non_blocking=True),
        sup_ids=item["sup_ids"].to(dev, non_blocking=True),
        sup_feat=item["sup_feat"].to(dev, non_blocking=True),
        sup_mask=item["sup_mask"].to(dev, non_blocking=True),
        hist=[(s.to(dev, non_blocking=True), r.to(dev, non_blocking=True),
               o.to(dev, non_blocking=True)) for s, r, o in item["hist"]],
    )


class Meters:
    """Accumulates MRR/Hits under several protocols and strata."""

    def __init__(self, hits_at):
        self.k = hits_at
        self.d = {}

    def add(self, name, ranks):
        if name not in self.d:
            self.d[name] = [0.0, {k: 0.0 for k in self.k}, 0]
        s, h, n = self.d[name]
        s += (1.0 / ranks.float()).sum().item()
        for k in self.k:
            h[k] += (ranks <= k).sum().item()
        self.d[name] = [s, h, n + ranks.numel()]

    def result(self):
        out = {}
        for name, (s, h, n) in self.d.items():
            if n == 0:
                continue
            out[name] = {"MRR": s / n, "n": n}
            out[name].update({f"Hits@{k}": h[k] / n for k in self.k})
        return out


@torch.no_grad()
def evaluate(model, data, split, device, cfg, verbose=True, stratify=False):
    model.eval()
    ds = {"valid": data.valid_set, "test": data.test_set}[split]
    dl = DataLoader(ds, batch_size=1, shuffle=False,
                    num_workers=cfg.num_workers, collate_fn=identity_collate)
    M = Meters(tuple(cfg.hits_at))
    index = data.index

    for raw in tqdm(dl, desc=f"eval[{split}]", disable=not verbose,
                    dynamic_ncols=True, leave=False):
        it = to_dev(raw, device)
        with autocast("cuda", dtype=torch.bfloat16,
                      enabled=device.type == "cuda"):
            E = model.evolve(it["hist"])
        n = it["subs"].numel()
        for a in range(0, n, cfg.query_chunk):
            b = min(a + cfg.query_chunk, n)
            with autocast("cuda", dtype=torch.bfloat16,
                          enabled=device.type == "cuda"):
                lg = model(E, it["subs"][a:b], it["rels"][a:b],
                           it["sup_ids"][a:b], it["sup_feat"][a:b],
                           it["sup_mask"][a:b])
            lg = lg.float()
            objs = it["objs"][a:b]
            tgt = lg.gather(1, objs.view(-1, 1))

            # ── raw ──────────────────────────────────────────────────────────
            M.add("raw", (lg > tgt).sum(1) + 1)

            subs_c = it["subs"][a:b].cpu().numpy()
            rels_c = it["rels"][a:b].cpu().numpy()
            t = it["t"]

            # ── time-aware filtered ──────────────────────────────────────────
            rows, cols = [], []
            for i in range(b - a):
                ans = index.answers(subs_c[i], rels_c[i], t)
                if len(ans):
                    cols.append(ans)
                    rows.append(np.full(len(ans), i, np.int64))
            lg_ta = lg.clone()
            if rows:
                lg_ta.index_put_(
                    (torch.from_numpy(np.concatenate(rows)).to(device),
                     torch.from_numpy(np.concatenate(cols).astype(np.int64))
                     .to(device)),
                    torch.tensor(float("-inf"), device=device))
            lg_ta.scatter_(1, objs.view(-1, 1), tgt)
            M.add("time_aware_filtered", (lg_ta > tgt).sum(1) + 1)

            # ── time-unaware filtered ────────────────────────────────────────
            rows, cols = [], []
            for i in range(b - a):
                ob, _ = index.g_sr.block(int(subs_c[i]) * index.R2
                                         + int(rels_c[i]))
                if ob is not None and len(ob):
                    u = np.unique(ob)
                    cols.append(u)
                    rows.append(np.full(len(u), i, np.int64))
            lg_tu = lg.clone()
            if rows:
                lg_tu.index_put_(
                    (torch.from_numpy(np.concatenate(rows)).to(device),
                     torch.from_numpy(np.concatenate(cols).astype(np.int64))
                     .to(device)),
                    torch.tensor(float("-inf"), device=device))
            lg_tu.scatter_(1, objs.view(-1, 1), tgt)
            M.add("time_unaware_filtered", (lg_tu > tgt).sum(1) + 1)

            # ── stratified by monotone-blockedness ───────────────────────────
            if stratify:
                sf = it["sup_feat"][a:b]
                sm = it["sup_mask"][a:b]
                sid = it["sup_ids"][a:b]
                ranks = (lg_ta > tgt).sum(1) + 1
                blocked = _blocked_mask(sf, sm, sid, objs)
                if blocked.any():
                    M.add("blocked", ranks[blocked])
                if (~blocked).any():
                    M.add("not_blocked", ranks[~blocked])

    return M.result()


def _blocked_mask(sup_feat, sup_mask, sup_ids, objs):
    """
    True where a distractor dominates the answer on (dt, count), i.e. the
    query cannot be ranked first by any f(count)*g(dt) with g non-increasing.
    """
    cnt = torch.expm1(sup_feat[..., 0])
    dt = torch.expm1(sup_feat[..., 1])
    has = (sup_feat[..., 7] > 0) & sup_mask
    is_ans = sup_ids == objs.view(-1, 1)
    ans_ok = (is_ans & has).any(1)

    big = torch.finfo(dt.dtype).max
    a_dt = torch.where(is_ans & has, dt, torch.full_like(dt, big)).min(1).values
    a_ct = torch.where(is_ans & has, cnt, torch.zeros_like(cnt)).max(1).values

    dom = has & ~is_ans & (dt <= a_dt.unsqueeze(1)) & (cnt >= a_ct.unsqueeze(1))
    return dom.any(1) & ans_ok


def main():
    cfg = parse_args()
    variant = ("structural-only" if cfg.rec_off else
               "recurrence-only" if cfg.struct_off else
               "monotone-kernel" if cfg.phase_off else "full")
    print(BANNER)
    print(f"  dataset={cfg.dataset}  variant={variant}  d={cfg.embed_dim}  "
          f"gcn_layers={cfg.gcn_layers}  H={cfg.hist_len}  S={cfg.max_support}")
    print(f"  epochs={cfg.epochs}  lr={cfg.lr}  workers={cfg.num_workers}  "
          f"GPU={cfg.gpu}\n")
    set_seed(cfg.seed)

    use_cuda = cfg.device == "cuda" and torch.cuda.is_available()
    device = torch.device(f"cuda:{cfg.gpu}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        free, tot = torch.cuda.mem_get_info(cfg.gpu)
        print(f"[GPU] {torch.cuda.get_device_name(cfg.gpu)}  "
              f"{free/1024**3:.1f}/{tot/1024**3:.1f} GB free")

    data = KairosData(cfg)
    model = KAIROS(data.num_entities, data.num_relations, cfg).to(device)
    print(f"[model] params="
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optim = AdamW(model.parameters(), lr=cfg.lr,
                  weight_decay=cfg.weight_decay)
    warm = max(1, int(cfg.epochs * cfg.warmup_ratio))
    sched = SequentialLR(optim,
                         [LinearLR(optim, 0.1, 1.0, total_iters=warm),
                          CosineAnnealingLR(optim, T_max=max(1, cfg.epochs - warm),
                                            eta_min=cfg.lr * 0.02)],
                         milestones=[warm])

    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    name = f"{cfg.dataset}_kairos_{variant}"
    ck = os.path.join(cfg.save_dir, f"{name}_best.pt")
    log_path = os.path.join(cfg.log_dir, f"{name}.jsonl")
    best, best_ep, bad = 0.0, 0, 0

    print(f"\n{'═'*len(HDR)}\n  KAIROS │ {cfg.dataset} │ {variant}"
          f"\n{'═'*len(HDR)}\n")
    print(HDR); print(SEP)

    for ep in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        dl = DataLoader(data.train_set, batch_size=1, shuffle=True,
                        num_workers=cfg.num_workers,
                        collate_fn=identity_collate,
                        persistent_workers=cfg.num_workers > 0,
                        prefetch_factor=4 if cfg.num_workers > 0 else None)
        tot_l, nb = 0.0, 0
        for raw in tqdm(dl, desc=f" ep{ep}", leave=False, dynamic_ncols=True):
            it = to_dev(raw, device)
            optim.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
                E = model.evolve(it["hist"])
                n = it["subs"].numel()
                loss = 0.0
                nchunk = 0
                for a in range(0, n, cfg.query_chunk):
                    b = min(a + cfg.query_chunk, n)
                    lg = model(E, it["subs"][a:b], it["rels"][a:b],
                               it["sup_ids"][a:b], it["sup_feat"][a:b],
                               it["sup_mask"][a:b])
                    loss = loss + F.cross_entropy(
                        lg.float(), it["objs"][a:b],
                        label_smoothing=cfg.label_smoothing)
                    nchunk += 1
                loss = loss / max(nchunk, 1)
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            tot_l += loss.item(); nb += 1

        sched.step()
        el = time.time() - t0
        m = {}
        if ep % cfg.eval_every == 0:
            r = evaluate(model, data, "valid", device, cfg, verbose=False)
            m = r.get("time_aware_filtered", {})

        is_best = m.get("MRR", 0) > best
        if is_best:
            best, best_ep, bad = m["MRR"], ep, 0
            torch.save({"model": model.state_dict()}, ck)
        else:
            bad += 1

        print(f"{'★' if is_best else ' '}{ep:>3} │ {el:>6.1f}s │ "
              f"{tot_l/max(nb,1):>8.4f} │ {m.get('MRR',0):>7.4f} "
              f"{m.get('Hits@1',0):>7.4f} {m.get('Hits@3',0):>7.4f} "
              f"{m.get('Hits@10',0):>7.4f} │ {model.rec_bias.item():>7.3f} │ "
              f"{sched.get_last_lr()[0]:>9.2e}", flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps({"epoch": ep, "loss": tot_l / max(nb, 1),
                                "valid": m}) + "\n")
        if bad >= cfg.patience:
            print(f"\n[early-stop] {cfg.patience} epochs without improvement.")
            break

    print(SEP)
    print(f"\n★ best valid MRR (time-aware filtered) = {best:.4f} "
          f"(epoch {best_ep})")
    if os.path.exists(ck):
        model.load_state_dict(torch.load(ck, map_location=device,
                                         weights_only=True)["model"])

    res = evaluate(model, data, "test", device, cfg, verbose=True,
                   stratify=True)

    print(f"\n{'═'*74}")
    print(f"  TEST — {cfg.dataset}   [{variant}]")
    print(f"{'═'*74}")
    print(f"  {'protocol':<24} {'n':>9} {'MRR':>8} {'H@1':>8} "
          f"{'H@3':>8} {'H@10':>8}")
    order = ["raw", "time_aware_filtered", "time_unaware_filtered",
             "blocked", "not_blocked"]
    for k in order:
        if k not in res:
            continue
        v = res[k]
        print(f"  {k:<24} {v['n']:>9,} {v['MRR']*100:>8.2f} "
              f"{v['Hits@1']*100:>8.2f} {v['Hits@3']*100:>8.2f} "
              f"{v['Hits@10']*100:>8.2f}")
    print(f"{'═'*74}")
    print("  'blocked' = queries no f(count)·g(Δt) with g non-increasing can")
    print("  rank first. The contribution predicts the gain concentrates here.")

    if not cfg.rec_off:
        try:
            ph = np.linspace(0.0, 10.0, 100)
            curves = {f"rel_{r}": model.kernel(r, 0, ph, device)
                      .float().cpu().numpy().tolist()
                      for r in range(min(8, data.num_relations))}
            with open(os.path.join(cfg.save_dir, f"{name}_kernel.json"),
                      "w") as f:
                json.dump({"phase": ph.tolist(), "curves": curves}, f, indent=2)
            print(f"  learned kernels → {cfg.save_dir}/{name}_kernel.json")
        except Exception as e:
            print(f"  [warn] kernel export failed: {e}")

    with open(os.path.join(cfg.save_dir, f"{name}_results.json"), "w") as f:
        json.dump({"dataset": cfg.dataset, "variant": variant,
                   "best_epoch": best_ep, "valid_mrr": best,
                   "test": res, "config": vars(cfg)}, f, indent=2, default=str)
    print(f"  saved → {cfg.save_dir}/{name}_results.json")


if __name__ == "__main__":
    main()
