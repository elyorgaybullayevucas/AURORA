#!/usr/bin/env python
"""
One command for "where is everything".

    python status.py

Shows what is running, how far each run has got, and what has finished.
"""
import glob
import json
import os
import re
import subprocess
import time

LOGS = "logs"
CK = "checkpoints"
EPOCH = re.compile(r"^[ ★]\s*(\d+)\s*│")


def tmux_sessions():
    try:
        out = subprocess.run(["tmux", "ls"], capture_output=True, text=True,
                             timeout=10).stdout
    except Exception:
        return {}
    ses = {}
    for line in out.splitlines():
        if ":" in line:
            name = line.split(":", 1)[0]
            ses[name] = line.split("(created", 1)[-1].strip(" )")
    return ses


def last_epoch(path):
    """Last epoch line and whether the run reached its TEST section."""
    try:
        with open(path, errors="ignore") as f:
            txt = f.read()
    except OSError:
        return None, False, 0
    lines = [l for l in txt.splitlines() if EPOCH.match(l)]
    done = "TEST —" in txt or "TEST -" in txt
    best = None
    for l in lines:
        if l.startswith("★"):
            best = l
    return (lines[-1] if lines else None), done, len(lines)


def main():
    ses = tmux_sessions()
    print(f"\n{'='*78}\n  RUNNING\n{'='*78}")
    if ses:
        for n, when in sorted(ses.items()):
            print(f"  {n:<24} started {when}")
    else:
        print("  (no tmux sessions)")

    print(f"\n{'='*78}\n  PROGRESS  (last epoch line per log)\n{'='*78}")
    logs = sorted(glob.glob(os.path.join(LOGS, "*.out")),
                  key=os.path.getmtime, reverse=True)[:14]
    if not logs:
        print("  (no logs)")
    for p in logs:
        line, done, n = last_epoch(p)
        age = (time.time() - os.path.getmtime(p)) / 60
        name = os.path.basename(p)
        mark = "done " if done else ("live " if age < 10 else "stale")
        if line:
            print(f"  [{mark}] {name:<26} ep{n:<3} {line.strip()[:74]}")
        else:
            print(f"  [{mark}] {name:<26} (no epoch line yet)")

    print(f"\n{'='*78}\n  FINISHED  (time-aware filtered, x100)\n{'='*78}")
    files = sorted(glob.glob(os.path.join(CK, "*_kairos_*_results.json")),
                   key=os.path.getmtime, reverse=True)
    if not files:
        print("  (no results)")
    print(f"  {'file':<46} {'MRR':>7} {'H@1':>7} {'H@3':>7} {'H@10':>7}")
    for f in files[:18]:
        try:
            d = json.load(open(f))
            t = d["test"]["time_aware_filtered"]
        except Exception:
            continue
        print(f"  {os.path.basename(f)[:-13]:<46} "
              f"{t['MRR']*100:>7.2f} {t['Hits@1']*100:>7.2f} "
              f"{t['Hits@3']*100:>7.2f} {t['Hits@10']*100:>7.2f}")
    print()
    print("  full table with baselines and ablations:  python collect.py")
    print()


if __name__ == "__main__":
    main()
