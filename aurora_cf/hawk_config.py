"""
HAWK configuration.

Sized for a single 40 GB A100 per run. The four datasets differ by two
orders of magnitude in edge count and by ~35 points in recurrence rate, so
capacity and history depth are set per dataset, but the MODEL and the
objective are identical everywhere — no per-dataset gate, decay constant or
mixing weight is tuned. That is a claim the paper has to be able to make.

    dataset   entities  rels  train    step   recurrence
    YAGO       10,623     10   161k      1      ~90%
    WIKI       12,554     24   539k      1      ~86%
    ICEWS18    23,033    256   373k     24      ~55%
    GDELT       7,691    240  1.73M     15      ~50%
"""
import argparse
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any

DATA_DIR = "./data"

_COMMON = dict(
    embed_dim=512, hazard_dim=128, dt_dim=32,
    n_heads=8, n_layers=2, gru_layers=1,
    conv_channels=64, dropout=0.2,
    warmup_ratio=0.08, weight_decay=1e-5, grad_clip=1.0,
    label_smoothing=0.1, rec_bias_init=-4.0,
    eval_every=1, patience=8, reserve_gb=14,
)

DATASETS: Dict[str, Dict[str, Any]] = {
    # high recurrence, small graphs -> deep history, large support
    "YAGO": dict(_COMMON, hist_len=10, k_neighbors=30, max_support=256,
                 rel_topk=48, batch_size=512, eval_batch_size=512,
                 lr=3e-4, epochs=40, num_workers=8, dropout=0.15),
    "WIKI": dict(_COMMON, hist_len=10, k_neighbors=30, max_support=256,
                 rel_topk=48, batch_size=512, eval_batch_size=512,
                 lr=3e-4, epochs=40, num_workers=8, dropout=0.15),
    # low recurrence, many relations -> structural branch carries the load
    "ICEWS18": dict(_COMMON, hist_len=12, k_neighbors=30, max_support=256,
                    rel_topk=96, batch_size=512, eval_batch_size=512,
                    lr=3e-4, epochs=50, num_workers=8, dropout=0.25),
    # largest, densest; smaller entity set so bigger batches fit
    "GDELT": dict(_COMMON, hist_len=8, k_neighbors=30, max_support=192,
                  rel_topk=96, batch_size=1024, eval_batch_size=1024,
                  lr=4e-4, epochs=30, num_workers=10, dropout=0.25,
                  patience=6),
}


@dataclass
class HawkConfig:
    dataset: str = "ICEWS18"
    data_dir: str = DATA_DIR
    use_inverse: bool = True
    # model
    embed_dim: int = 512
    hazard_dim: int = 128
    dt_dim: int = 32
    n_heads: int = 8
    n_layers: int = 2
    gru_layers: int = 1
    conv_channels: int = 64
    dropout: float = 0.2
    rec_bias_init: float = -4.0
    # data
    hist_len: int = 12
    k_neighbors: int = 30
    max_support: int = 256
    rel_topk: int = 96
    # optim
    epochs: int = 50
    batch_size: int = 512
    eval_batch_size: int = 512
    lr: float = 3e-4
    warmup_ratio: float = 0.08
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    patience: int = 8
    eval_every: int = 1
    # ablations
    hazard_off: bool = False
    struct_off: bool = False
    phase_off: bool = False
    # runtime
    num_workers: int = 8
    reserve_gb: int = 14
    hits_at: Tuple = (1, 3, 10)
    seed: int = 42
    device: str = "cuda"
    gpu: int = 0
    save_dir: str = "checkpoints"
    log_dir: str = "logs"


def parse_args(argv=None) -> HawkConfig:
    p = argparse.ArgumentParser("HAWK")
    p.add_argument("--dataset", default="ICEWS18", choices=list(DATASETS))
    p.add_argument("--data_dir", default=DATA_DIR)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--log_dir", default="logs")
    for k in ("embed_dim", "hazard_dim", "n_heads", "n_layers", "gru_layers",
              "conv_channels", "hist_len", "k_neighbors", "max_support",
              "rel_topk", "epochs", "batch_size", "eval_batch_size",
              "num_workers", "patience", "eval_every", "reserve_gb"):
        p.add_argument(f"--{k}", type=int, default=None)
    for k in ("lr", "dropout", "weight_decay", "label_smoothing",
              "warmup_ratio", "grad_clip", "rec_bias_init"):
        p.add_argument(f"--{k}", type=float, default=None)
    p.add_argument("--hazard_off", action="store_true")
    p.add_argument("--struct_off", action="store_true")
    p.add_argument("--phase_off", action="store_true")

    a = p.parse_args(argv)
    base = dict(DATASETS[a.dataset])
    for k, v in vars(a).items():
        if v is not None and not (isinstance(v, bool) and v is False):
            base[k] = v
    base["dataset"] = a.dataset
    base["data_dir"] = a.data_dir
    valid = HawkConfig.__dataclass_fields__
    return HawkConfig(**{k: v for k, v in base.items() if k in valid})
