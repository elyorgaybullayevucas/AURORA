"""
AURORA-CF (Copy-First) configuration.
Key idea: copy scores are the primary signal; neural learns residual corrections.
"""
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, Any

DATA_DIR = "./data"

DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ICEWS18": {
        "embed_dim":         256,
        "k_neighbors":       32,
        "hist_len":          20,
        "gru_layers":        2,
        "dropout":           0.3,
        "copy_lambda":       0.5,
        "recency_steps":     2,
        "recency_boost":     4.0,
        "use_entity_copy":   True,
        "copy_only":         False,
        "fixed_gate":        0.25,
        "epochs":            60,
        "batch_size":        1024,
        "lr":                2e-4,
        "warmup_ratio":      0.1,
        "weight_decay":      1e-4,
        "grad_clip":         1.0,
        "alpha_infonce":     0.4,
        "infonce_temp":      0.07,
        "label_smoothing":   0.1,
        "neural_init_scale": -3.0,
        "neural_scale_cap":  1.0,
    },
    "WIKI": {
        "embed_dim":         256,
        "k_neighbors":       64,
        "hist_len":          20,
        "gru_layers":        2,
        "dropout":           0.2,
        "copy_lambda":       0.05,
        "recency_steps":     5,
        "recency_boost":     10.0,
        "use_entity_copy":   True,
        "copy_only":         True,
        "fixed_gate":        1.0,
        "epochs":            50,
        "batch_size":        512,
        "lr":                1e-3,
        "warmup_ratio":      0.1,
        "weight_decay":      1e-4,
        "grad_clip":         1.0,
        "alpha_infonce":     0.0,
        "infonce_temp":      0.07,
        "label_smoothing":   0.05,
        "neural_init_scale": -3.0,
        "neural_scale_cap":  0.15,
    },
    "YAGO": {
        "embed_dim":         256,
        "k_neighbors":       64,
        "hist_len":          20,
        "gru_layers":        2,
        "dropout":           0.2,
        "copy_lambda":       0.02,
        "recency_steps":     7,
        "recency_boost":     15.0,
        "use_entity_copy":   True,
        "copy_only":         True,
        "fixed_gate":        1.0,
        "epochs":            50,
        "batch_size":        512,
        "lr":                1e-3,
        "warmup_ratio":      0.1,
        "weight_decay":      1e-4,
        "grad_clip":         1.0,
        "alpha_infonce":     0.0,
        "infonce_temp":      0.07,
        "label_smoothing":   0.05,
        "neural_init_scale": -3.0,
        "neural_scale_cap":  0.12,
    },
    "GDELT": {
        "embed_dim":         256,
        "k_neighbors":       32,
        "hist_len":          15,
        "gru_layers":        2,
        "dropout":           0.3,
        "copy_lambda":       0.5,
        "recency_steps":     2,
        "recency_boost":     4.0,
        "use_entity_copy":   True,
        "copy_only":         False,
        "fixed_gate":        0.20,
        "epochs":            40,
        "batch_size":        512,
        "lr":                2e-4,
        "warmup_ratio":      0.1,
        "weight_decay":      1e-4,
        "grad_clip":         1.0,
        "alpha_infonce":     0.4,
        "infonce_temp":      0.07,
        "label_smoothing":   0.1,
        "neural_init_scale": -3.0,
        "neural_scale_cap":  1.0,
    },
}


@dataclass
class AURORACFConfig:
    dataset:           str   = "ICEWS18"
    data_dir:          str   = DATA_DIR
    embed_dim:         int   = 256
    k_neighbors:       int   = 32
    hist_len:          int   = 20
    gru_layers:        int   = 2
    dropout:           float = 0.3
    use_inverse:       bool  = True
    copy_lambda:       float = 0.5
    recency_steps:     int   = 2
    recency_boost:     float = 4.0
    use_entity_copy:   bool  = True
    copy_only:         bool  = False
    fixed_gate:        float = 0.3
    neural_init_scale: float = -3.0
    neural_scale_cap:  float = 1.0
    epochs:            int   = 60
    batch_size:        int   = 1024
    lr:                float = 2e-4
    warmup_ratio:      float = 0.1
    weight_decay:      float = 1e-4
    grad_clip:         float = 1.0
    alpha_infonce:     float = 0.4
    infonce_temp:      float = 0.07
    label_smoothing:   float = 0.1
    eval_every:        int   = 1
    hits_at:           Tuple = (1, 3, 10)
    seed:              int   = 42
    device:            str   = "cuda"
    gpu:               int   = 1
    save_dir:          str   = "checkpoints"
    log_dir:           str   = "logs"


def parse_args() -> AURORACFConfig:
    p = argparse.ArgumentParser(description="AURORA-CF")
    p.add_argument("--dataset",    type=str,   default="ICEWS18",
                   choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--data_dir",   type=str,   default=DATA_DIR)
    p.add_argument("--embed_dim",  type=int,   default=None)
    p.add_argument("--k_neighbors",type=int,   default=None)
    p.add_argument("--hist_len",   type=int,   default=None)
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--dropout",    type=float, default=None)
    p.add_argument("--copy_lambda",type=float, default=None)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--device",     type=str,   default="cuda")
    p.add_argument("--gpu",        type=int,   default=1)
    p.add_argument("--save_dir",   type=str,   default="checkpoints")
    p.add_argument("--log_dir",    type=str,   default="logs")
    p.add_argument("--eval_every", type=int,   default=1)

    args = p.parse_args()
    ds = DATASET_CONFIGS[args.dataset].copy()
    for k, v in vars(args).items():
        if v is not None and k in ds:
            ds[k] = v

    kwargs = {k: v for k, v in vars(args).items()
              if k in AURORACFConfig.__dataclass_fields__ and v is not None}
    kwargs.update(ds)
    kwargs["dataset"]  = args.dataset
    kwargs["data_dir"] = args.data_dir
    return AURORACFConfig(**{k: v for k, v in kwargs.items()
                             if k in AURORACFConfig.__dataclass_fields__})
