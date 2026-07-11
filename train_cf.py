#!/usr/bin/env python
"""
AURORA-CF: Copy-First Temporal Knowledge Graph Forecasting

Usage:
    python train_cf.py --dataset ICEWS18 --gpu 1
    python train_cf.py --dataset WIKI    --gpu 2
    python train_cf.py --dataset YAGO    --gpu 3
    python train_cf.py --dataset GDELT   --gpu 4
"""
import sys

banner = """
╔══════════════════════════════════════════════════════════╗
║          AURORA-CF  (2025)                              ║
║  Copy-First Temporal Knowledge Graph Forecasting        ║
╠══════════════════════════════════════════════════════════╣
║  Key innovations:                                       ║
║   1. Copy-First: historical copy as primary signal      ║
║   2. Neural residual correction (additive, not gated)   ║
║   3. Per-relation copy blend (learned per relation)     ║
║   4. GRU history encoder over temporal snapshots        ║
║   5. Adaptive neural scale (starts small, grows slowly) ║
╚══════════════════════════════════════════════════════════╝
"""

from aurora_cf.config import parse_args, DATASET_CONFIGS

cfg = parse_args()

print(banner)
print(f"  Dataset    : {cfg.dataset}")
print(f"  embed_dim  : {cfg.embed_dim}")
print(f"  k_neighbors: {cfg.k_neighbors}")
print(f"  hist_len   : {cfg.hist_len}")
print(f"  gru_layers : {cfg.gru_layers}")
print(f"  epochs     : {cfg.epochs}")
print(f"  lr         : {cfg.lr}")
print(f"  α_infonce  : {cfg.alpha_infonce}")
print(f"  device     : {cfg.device} (GPU {cfg.gpu})")
print()

from aurora_cf.trainer import AURORACFTrainer

trainer = AURORACFTrainer(cfg)
results = trainer.train()
sys.exit(0)
