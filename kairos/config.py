"""KAIROS configuration."""
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, Any

DATA_DIR = "./data"

# Capacity follows the published settings for this benchmark family rather
# than the size of the GPU: RE-GCN uses d=200, DiMNet d=128 with omega=3
# layers and history 10 (5 on GDELT). These datasets are small enough that
# a much larger d overfits -- an earlier d=512 run is not evidence otherwise,
# it was never compared.
_C = dict(
    embed_dim=200, hazard_dim=128, gcn_layers=3, conv_channels=64,
        # Both intensities must start at a comparable scale. The gradient that
    # reaches the recurrence branch through logaddexp is sigmoid(f_rec -
    # f_struct); at rec_bias=-4 that factor is ~0.007 and the branch is
    # starved. The first YAGO epoch showed exactly this -- rec_bias moved
    # from -4.000 to -3.991 in a whole epoch. -2 puts log lambda_rec near 0
    # at initialisation, which is where the structural logits also start.
    dropout=0.2, rec_bias_init=-2.0, aux_weight=0.1, struct_aux=0.3,
    lr=1e-3, weight_decay=1e-5, grad_clip=1.0, label_smoothing=0.1,
    warmup_ratio=0.05, eval_every=1, patience=10,
    # The recurrence trunk holds ~7 intermediates of shape
    # (chunk, max_support, 2*hazard_dim). At chunk=8192, S=256, dh=128 that
    # is ~7.5 GB and ICEWS18 died on it. 1024 keeps it under 1 GB.
    query_chunk=1024, num_workers=6, reserve_gb=0,
)

DATASETS: Dict[str, Dict[str, Any]] = {
    "YAGO":    dict(_C, hist_len=10, max_support=256, rel_topk=48,
                    epochs=60, dropout=0.15),
    "WIKI":    dict(_C, hist_len=10, max_support=256, rel_topk=48,
                    epochs=60, dropout=0.15),
    "ICEWS18": dict(_C, hist_len=10, max_support=256, rel_topk=96,
                    epochs=60, dropout=0.25),
    "GDELT":   dict(_C, hist_len=5, max_support=192, rel_topk=96,
                    epochs=40, dropout=0.25, patience=8),
}


@dataclass
class KairosConfig:
    dataset: str = "ICEWS18"
    data_dir: str = DATA_DIR
    embed_dim: int = 200
    hazard_dim: int = 128
    gcn_layers: int = 3
    conv_channels: int = 64
    dropout: float = 0.2
    rec_bias_init: float = -2.0
    aux_weight: float = 0.1
    struct_aux: float = 0.3
    hist_len: int = 12
    max_support: int = 256
    rel_topk: int = 96
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    warmup_ratio: float = 0.05
    patience: int = 10
    eval_every: int = 1
    query_chunk: int = 1024
    eval_only: bool = False
    rec_off: bool = False
    struct_off: bool = False
    phase_off: bool = False
    phase_feat_off: bool = False
    num_workers: int = 6
    reserve_gb: int = 0
    hits_at: Tuple = (1, 3, 10)
    seed: int = 42
    device: str = "cuda"
    gpu: int = 0
    tag: str = ""
    save_dir: str = "checkpoints"
    log_dir: str = "logs"


def parse_args(argv=None) -> KairosConfig:
    p = argparse.ArgumentParser("KAIROS")
    p.add_argument("--dataset", default="ICEWS18", choices=list(DATASETS))
    p.add_argument("--data_dir", default=DATA_DIR)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--tag", default="")
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--log_dir", default="logs")
    for k in ("embed_dim", "hazard_dim", "gcn_layers", "conv_channels",
              "hist_len", "max_support", "rel_topk", "epochs", "patience",
              "eval_every", "query_chunk", "num_workers", "reserve_gb"):
        p.add_argument(f"--{k}", type=int, default=None)
    for k in ("lr", "dropout", "weight_decay", "label_smoothing",
              "warmup_ratio", "grad_clip", "rec_bias_init", "aux_weight",
              "struct_aux"):
        p.add_argument(f"--{k}", type=float, default=None)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--rec_off", action="store_true")
    p.add_argument("--struct_off", action="store_true")
    p.add_argument("--phase_off", action="store_true")
    p.add_argument("--phase_feat_off", action="store_true")

    a = p.parse_args(argv)
    base = dict(DATASETS[a.dataset])
    for k, v in vars(a).items():
        if v is not None and v != "" and not (isinstance(v, bool) and v is False):
            base[k] = v
    base["dataset"], base["data_dir"] = a.dataset, a.data_dir
    fields = KairosConfig.__dataclass_fields__
    return KairosConfig(**{k: v for k, v in base.items() if k in fields})
