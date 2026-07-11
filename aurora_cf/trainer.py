"""AURORA-CF Trainer."""
import os, time, json, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from aurora_cf.config import AURORACFConfig
from aurora_cf.model import AURORACFModel
from aurora_cf.loss import AURORACFLoss
from aurora_cf.data import CFDataLoader, cf_collate
from aurora_cf.evaluate import evaluate


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


HEADER = (
    f"{'Ep':>4} │ {'Time':>6} │ "
    f"{'Loss':>8} {'CE':>8} {'NCE':>8} │ "
    f"{'MRR':>7} {'H@1':>7} {'H@3':>7} {'H@10':>7} │ "
    f"{'Scale':>7} │ {'LR':>9}"
)
SEP = "─" * len(HEADER)


def _row(ep, t, ld, m, scale, lr, best=False):
    star = "★" if best else " "
    return (
        f"{star}{ep:>3} │ {t:>5.1f}s │ "
        f"{ld['total']:>8.4f} {ld['ce']:>8.4f} {ld['infonce']:>8.4f} │ "
        f"{m.get('MRR',0):>7.4f} {m.get('Hits@1',0):>7.4f} "
        f"{m.get('Hits@3',0):>7.4f} {m.get('Hits@10',0):>7.4f} │ "
        f"{scale:>7.4f} │ {lr:>9.2e}"
    )


class AURORACFTrainer:

    def __init__(self, cfg: AURORACFConfig):
        self.cfg = cfg
        set_seed(cfg.seed)

        # ── GPU reservation ───────────────────────────────────────────────────
        use_cuda = cfg.device == "cuda" and torch.cuda.is_available()
        if use_cuda:
            self.device = torch.device(f"cuda:{cfg.gpu}")
            free_mem, _ = torch.cuda.mem_get_info(cfg.gpu)
            reserve_gb  = min(12, int(free_mem * 0.85) // (1024**3))
            reserve_n   = reserve_gb * (1024**3) // 4
            print(f"[Device] GPU {cfg.gpu} — "
                  f"{torch.cuda.get_device_name(cfg.gpu)}")
            print(f"[GPU]    {free_mem/1024**3:.1f} GB free → "
                  f"{reserve_gb} GB reserved …")
            _placeholder = torch.zeros(reserve_n, device=self.device)
            print(f"[GPU]    Reserved ✓")
        else:
            self.device = torch.device("cpu")
            _placeholder = None
            print("[Device] CPU")

        # ── Data (CPU pre-computation) ─────────────────────────────────────────
        self.loader = CFDataLoader(cfg)

        # ── Model ─────────────────────────────────────────────────────────────
        if _placeholder is not None:
            del _placeholder
            torch.cuda.empty_cache()

        self.model = AURORACFModel(
            num_entities=self.loader.num_entities,
            num_relations=self.loader.num_relations,
            cfg=cfg,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters()
                       if p.requires_grad)
        print(f"[Model] AURORA-CF  params={n_params:,}")
        print(f"[Model] neural_scale init = "
              f"{torch.sigmoid(self.model.log_neural_scale).item():.4f}")

        # ── AMP ───────────────────────────────────────────────────────────────
        self.use_amp = (self.device.type == "cuda")
        self.scaler  = GradScaler("cuda", enabled=self.use_amp)

        # ── Loss ──────────────────────────────────────────────────────────────
        self.loss_fn = AURORACFLoss(
            smoothing=cfg.label_smoothing,
            alpha=cfg.alpha_infonce,
            temperature=cfg.infonce_temp,
        )

        # ── Optimizer + scheduler ─────────────────────────────────────────────
        self.optim = AdamW(self.model.parameters(),
                           lr=cfg.lr, weight_decay=cfg.weight_decay)
        warmup_ep = max(1, int(cfg.epochs * cfg.warmup_ratio))
        warmup    = LinearLR(self.optim, start_factor=0.1, end_factor=1.0,
                             total_iters=warmup_ep)
        cosine    = CosineAnnealingLR(self.optim,
                                      T_max=cfg.epochs - warmup_ep,
                                      eta_min=cfg.lr * 0.01)
        self.sched = SequentialLR(self.optim, [warmup, cosine],
                                  milestones=[warmup_ep])

        os.makedirs(cfg.save_dir, exist_ok=True)
        os.makedirs(cfg.log_dir,  exist_ok=True)
        self.log_path = os.path.join(cfg.log_dir, f"{cfg.dataset}_cf_log.jsonl")
        self.best_mrr = 0.0
        self.best_ep  = 0

    # ── Training epoch ────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        cfg = self.cfg

        dl = DataLoader(
            self.loader.train_set, batch_size=cfg.batch_size,
            shuffle=True, num_workers=4,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True, collate_fn=cf_collate,
            persistent_workers=True,
        )

        tot_total = tot_ce = tot_nce = 0.0
        n = 0

        for batch in tqdm(dl, desc=f" ep{epoch}", leave=False,
                          dynamic_ncols=True):
            subs, rels, objs, times, ne, nr, nm, rel_copy, ent_copy = batch

            subs     = subs.to(self.device)
            rels     = rels.to(self.device)
            objs     = objs.to(self.device)
            ne       = ne.to(self.device)
            nm       = nm.to(self.device)
            N        = self.loader.num_entities
            if rel_copy.shape[1] < N:
                rel_copy = torch.cat([rel_copy,
                    torch.zeros(rel_copy.shape[0], N - rel_copy.shape[1])], dim=1)
            if ent_copy.shape[1] < N:
                ent_copy = torch.cat([ent_copy,
                    torch.zeros(ent_copy.shape[0], N - ent_copy.shape[1])], dim=1)
            rel_copy = rel_copy.to(self.device)
            ent_copy = ent_copy.to(self.device)

            self.optim.zero_grad()
            with autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                logits, query = self.model(subs, rels, ne, nr, nm,
                                           rel_copy, ent_copy)
                loss_d = self.loss_fn(logits, query, objs,
                                      self.model.ent_emb.weight)

            self.scaler.scale(loss_d["_total"]).backward()
            self.scaler.unscale_(self.optim)
            clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.scaler.step(self.optim)
            self.scaler.update()

            tot_total += loss_d["total"]
            tot_ce    += loss_d["ce"]
            tot_nce   += loss_d["infonce"]
            n         += 1

        self.sched.step()
        return {"total": tot_total / n, "ce": tot_ce / n, "infonce": tot_nce / n}

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self):
        cfg = self.cfg
        print(f"\n{'═'*len(HEADER)}")
        print(f"  AURORA-CF │ {cfg.dataset} │ epochs={cfg.epochs} │ "
              f"d={cfg.embed_dim} │ H={cfg.hist_len} │ K={cfg.k_neighbors}")
        print(f"{'═'*len(HEADER)}\n")
        print(HEADER)
        print(SEP)

        for ep in range(1, cfg.epochs + 1):
            t0     = time.time()
            loss_d = self._train_epoch(ep)
            elapsed = time.time() - t0
            lr_now  = self.sched.get_last_lr()[0]
            scale   = torch.sigmoid(
                self.model.log_neural_scale).item() * self.model.max_neural_scale

            metrics = {}
            if ep % cfg.eval_every == 0:
                metrics = evaluate(
                    self.model, self.loader, split="valid",
                    device=self.device, batch_size=cfg.batch_size,
                    hits_at=list(cfg.hits_at), verbose=False,
                )

            is_best = metrics.get("MRR", 0) > self.best_mrr
            if is_best:
                self.best_mrr = metrics["MRR"]
                self.best_ep  = ep
                self._save("best")
            if ep % 5 == 0:
                self._save(f"ep{ep:03d}")

            print(_row(ep, elapsed, loss_d, metrics, scale, lr_now, is_best))

            with open(self.log_path, "a") as f:
                f.write(json.dumps({
                    "epoch": ep, "loss": loss_d, "metrics": metrics,
                    "lr": lr_now, "neural_scale": scale,
                }) + "\n")

        # ── Final test ────────────────────────────────────────────────────────
        print(SEP)
        print(f"\n★  Best valid MRR = {self.best_mrr:.4f}  (epoch {self.best_ep})")
        print("\nLoading best checkpoint → TEST evaluation …")
        self._load("best")

        test_m = evaluate(
            self.model, self.loader, split="test",
            device=self.device, batch_size=cfg.batch_size,
            hits_at=list(cfg.hits_at), verbose=True,
        )

        print(f"\n{'═'*50}")
        print(f"  TEST RESULTS — {cfg.dataset}")
        print(f"{'═'*50}")
        for k, v in test_m.items():
            print(f"  {k:<12}: {v:.4f}")
        print(f"{'═'*50}")

        out = os.path.join(cfg.save_dir, f"{cfg.dataset}_cf_results.json")
        with open(out, "w") as f:
            json.dump({"dataset": cfg.dataset, "best_epoch": self.best_ep,
                       "valid_mrr": self.best_mrr, "test": test_m,
                       "config": cfg.__dict__}, f, indent=2, default=str)
        print(f"Saved → {out}")
        return test_m

    def _save(self, tag: str):
        p = os.path.join(self.cfg.save_dir, f"{self.cfg.dataset}_cf_{tag}.pt")
        torch.save({"model": self.model.state_dict(),
                    "optim": self.optim.state_dict()}, p)

    def _load(self, tag: str):
        p = os.path.join(self.cfg.save_dir, f"{self.cfg.dataset}_cf_{tag}.pt")
        if os.path.exists(p):
            ck = torch.load(p, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ck["model"])
            print(f"  Loaded: {p}")
