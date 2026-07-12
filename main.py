#!/usr/bin/env python
"""
AURORA-CF interactive launcher.
Run without arguments for interactive mode, or pass CLI args directly.

  python main.py                          # interactive
  python main.py --dataset WIKI --gpu 2  # direct
"""
import sys
import subprocess


DATASETS = ["ICEWS18", "WIKI", "YAGO", "GDELT"]

BANNER = """
╔══════════════════════════════════════════════════════════╗
║          AURORA-CF  (2025)                              ║
║  Copy-First Temporal Knowledge Graph Forecasting        ║
╚══════════════════════════════════════════════════════════╝
"""


def ask(prompt, choices=None, default=None):
    while True:
        suffix = f" [{default}]" if default is not None else ""
        if choices:
            suffix += f" ({'/'.join(str(c) for c in choices)})"
        val = input(f"  {prompt}{suffix}: ").strip()
        if val == "" and default is not None:
            return default
        if choices is None or val in [str(c) for c in choices]:
            return val
        print(f"    → Please choose from: {choices}")


def interactive():
    print(BANNER)
    print("  Interactive mode — press Enter to use defaults.\n")

    dataset = ask("Dataset", choices=DATASETS, default="ICEWS18")
    gpu     = ask("GPU index", default="1")
    epochs  = ask("Epochs (Enter = dataset default)", default="")
    bs      = ask("Batch size (Enter = dataset default)", default="")

    model_type = ask("Model", choices=["adapt", "cf"], default="adapt")
    script = "train_adapt.py" if model_type == "adapt" else "train_cf.py"

    cmd = [sys.executable, script,
           "--dataset", dataset,
           "--gpu", gpu]
    if epochs:
        cmd += ["--epochs", epochs]
    if bs:
        cmd += ["--batch_size", bs]

    print(f"\n  Running: {' '.join(cmd)}\n")
    subprocess.run(cmd)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Pass all args directly to train_cf.py
        cmd = [sys.executable, "train_cf.py"] + sys.argv[1:]
        subprocess.run(cmd)
    else:
        interactive()
