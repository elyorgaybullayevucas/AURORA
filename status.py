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


CUR_PARAMS = {"YAGO": 6017518, "ICEWS18": 8610328}


def live_trainings():
    """
    Every train_kairos.py actually running, from the process table.

    tmux session names are not evidence. A session can outlive its command, a
    command can run under a name that says nothing about it, and the session
    that was training ICEWS18 was called qcond_I18 while the launcher looked
    for qcond_ICEWS18. The process table is the only thing that knows.
    """
    try:
        out = subprocess.run(["pgrep", "-fa", "train_kairos.py"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        if "pgrep" in line:
            continue
        pid, _, cmd = line.partition(" ")
        ds = re.search(r"--dataset\s+(\S+)", cmd)
        tg = re.search(r"--tag\s+(\S+)", cmd)
        rows.append((pid, ds.group(1) if ds else "?",
                     tg.group(1) if tg else "(untagged)", cmd))
    return rows


def log_header(path):
    """params= and the variant banner from the top of a run's log."""
    try:
        with open(path, errors="ignore") as f:
            head = f.read(4000)
    except OSError:
        return None, None
    p = re.search(r"params=([\d,]+)", head)
    v = re.search(r"CADENCE\s*│\s*(\S+)\s*│\s*(\S+)", head)
    return (int(p.group(1).replace(",", "")) if p else None,
            v.group(2) if v else None)


def main():
    ses = tmux_sessions()
    print(f"\n{'='*78}\n  RUNNING  (from the process table, not session names)\n"
          f"{'='*78}")
    live = live_trainings()
    if not live:
        print("  no train_kairos.py process is running")
    for pid, ds, tag, _ in live:
        # find this run's log by (dataset, tag) rather than by session name
        cand = [p for p in glob.glob(os.path.join(LOGS, "*.out"))
                if log_header(p)[0] is not None]
        best_p, params, variant = None, None, None
        for p in cand:
            pr, vr = log_header(p)
            if os.path.getmtime(p) > time.time() - 900:
                if best_p is None or os.path.getmtime(p) > os.path.getmtime(best_p):
                    if tag in os.path.basename(p) or ds[:4] in os.path.basename(p):
                        best_p, params, variant = p, pr, vr
        note = ""
        exp = CUR_PARAMS.get(ds)
        if params and exp:
            note = ("  code=CURRENT" if params == exp
                    else f"  code=OLD (params {params:,} != {exp:,})")
        elif params:
            note = f"  params={params:,}"
        print(f"  pid {pid:<8} {ds:<8} --tag {tag:<10}"
              f"{('variant='+variant) if variant else '':<18}{note}")

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
