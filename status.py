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


# Parameter counts of the CURRENT code, per (dataset, variant). PRISM adds a
# router whose size does not depend on the vocabulary (+124,930 everywhere,
# including the scalar used by the --router_const ablation).
CUR_PARAMS = {("YAGO", "full"): 6017518, ("ICEWS18", "full"): 8610328,
              ("YAGO", "prism"): 6142448, ("ICEWS18", "prism"): 8735258}


def live_trainings():
    """
    One row per training RUN, from the process table.

    tmux session names are not evidence: a session outlives its command and
    its name says nothing reliable about what runs in it. But the process
    table needs care too. One run shows up as several processes -- the shell
    tmux starts, the python process, and one DataLoader worker per
    num_workers -- all carrying the same command line. Listing processes
    showed one run six times. Rows are collapsed on (dataset, tag, variant),
    and the shell wrapper (whose command line does not START with python) is
    dropped.
    """
    try:
        out = subprocess.run(["pgrep", "-fa", "train_kairos.py"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    runs = {}
    for line in out.splitlines():
        pid, _, cmd = line.partition(" ")
        if "pgrep" in cmd or not re.match(r"(\S*/)?python\S*\s", cmd):
            continue
        ds = re.search(r"--dataset\s+(\S+)", cmd)
        tg = re.search(r"--tag\s+(\S+)", cmd)
        key = (ds.group(1) if ds else "?", tg.group(1) if tg else "-",
               "prism" if "--prism" in cmd else "full")
        runs.setdefault(key, []).append(pid)
    return sorted(runs.items())


def log_for(ds, tag):
    """
    The newest log whose header names THIS dataset and THIS tag.

    Matching on the dataset alone attached one YAGO log to every YAGO run,
    which labelled an old seed run as PRISM. The header carries tag= since the
    run that introduced this check; older logs do not and return None.
    """
    best = None
    for p in glob.glob(os.path.join(LOGS, "*.out")):
        try:
            with open(p, errors="ignore") as f:
                head = f.read(4000)
        except OSError:
            continue
        if f"dataset={ds} " not in head or f"tag={tag} " not in head:
            continue
        if best is None or os.path.getmtime(p) > os.path.getmtime(best[0]):
            m = re.search(r"params=([\d,]+)", head)
            best = (p, int(m.group(1).replace(",", "")) if m else None)
    return best


def main():
    ses = tmux_sessions()
    print(f"\n{'='*78}\n  RUNNING  (one row per run, from the process table)\n"
          f"{'='*78}")
    live = live_trainings()
    if not live:
        print("  no train_kairos.py process is running")
    for (ds, tag, variant), pids in live:
        found = log_for(ds, tag)
        exp = CUR_PARAMS.get((ds, variant))
        if found is None:
            note = ("code=?  (no log header yet: output still buffered, or the run "
                    "predates tag= in the header)")
        elif found[1] and exp:
            note = ("code=CURRENT" if found[1] == exp
                    else f"code=OLD (params {found[1]:,} != {exp:,})")
        else:
            note = os.path.basename(found[0])
        print(f"  {ds:<8} --tag {tag:<9} {variant:<6} "
              f"{len(pids)} procs   {note}")

    print(f"\n  tmux sessions: "
          + (", ".join(sorted(ses)) if ses else "(none)"))

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
