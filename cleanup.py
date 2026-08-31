#!/usr/bin/env python
"""
Show what is stale on this machine, and delete it only when told to.

    python cleanup.py              # show only, change nothing
    python cleanup.py --yes        # actually delete / kill

Nothing here is clever. It exists because the working directory has
accumulated three superseded model families, per-epoch checkpoints from runs
whose results were superseded months ago, and tmux sessions from June and July
that nobody has looked at since.

WHAT IS NEVER TOUCHED
  - any *_results.json                 these are the evidence in the paper;
                                       all of them together are under 100 KB
  - any *_kairos_* checkpoint          the current model family
  - any tmux session running something
  - anything under data/ or paper/

Everything else is listed, with its size, and you decide.
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import time

# Model families that predate CADENCE. Their results json files are kept; only
# the multi-hundred-megabyte weight files go.
SUPERSEDED = ("cf", "v3", "adapt", "hawk", "nhc", "full", "x")

KEEP_SESSION = re.compile(r"^(qcond|v2|logcl)", re.I)


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:,.0f}{u}" if u == "B" else f"{n/1:,.1f}{u}"
        n /= 1024.0


def size(paths):
    return sum(os.path.getsize(p) for p in paths if os.path.exists(p))


# ── tmux ─────────────────────────────────────────────────────────────────────

def tmux_sessions():
    try:
        out = subprocess.run(
            ["tmux", "list-sessions", "-F",
             "#{session_name}\t#{session_created}\t#{session_attached}"],
            capture_output=True, text=True, timeout=10)
    except Exception:
        return []
    rows = []
    for line in out.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            rows.append((parts[0], int(parts[1])))
    return rows


def session_busy(name):
    """True if the session's pane still has a live child process."""
    try:
        out = subprocess.run(
            ["tmux", "list-panes", "-t", name, "-F", "#{pane_pid}"],
            capture_output=True, text=True, timeout=10)
        pid = out.stdout.strip().split("\n")[0]
        if not pid:
            return True                      # unknown: assume busy, keep it
        kids = subprocess.run(["pgrep", "-P", pid], capture_output=True,
                              text=True, timeout=10)
        return bool(kids.stdout.strip())
    except Exception:
        return True                          # on any doubt, keep it


# ── plan ─────────────────────────────────────────────────────────────────────

def plan(save_dir="checkpoints", log_dir="logs", age_days=14):
    now = time.time()
    cutoff = now - age_days * 86400

    # per-epoch snapshots of superseded families: <DS>_<family>_ep###.pt
    ep_snaps = [p for p in glob.glob(os.path.join(save_dir, "*_ep*.pt"))
                if "kairos" not in os.path.basename(p)]

    # best checkpoints of superseded families
    old_best = []
    for p in glob.glob(os.path.join(save_dir, "*_best.pt")):
        b = os.path.basename(p)
        if "kairos" in b:
            continue
        fam = b.rsplit("_best.pt", 1)[0].split("_", 1)[-1]
        if fam in SUPERSEDED:
            old_best.append(p)

    stale_sessions = []
    for name, created in tmux_sessions():
        if KEEP_SESSION.match(name):
            continue
        if created > cutoff:
            continue
        if session_busy(name):
            continue
        stale_sessions.append((name, created))

    return ep_snaps, old_best, stale_sessions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true",
                    help="actually delete and kill; without it nothing changes")
    ap.add_argument("--save_dir", default="checkpoints")
    ap.add_argument("--log_dir", default="logs")
    ap.add_argument("--age_days", type=int, default=14,
                    help="a tmux session younger than this is never touched")
    a = ap.parse_args()

    ep_snaps, old_best, stale = plan(a.save_dir, a.log_dir, a.age_days)

    total, used, free = shutil.disk_usage(".")
    print(f"\ndisk: {human(free)} free of {human(total)}")

    print(f"\n{'='*70}\n  KEPT, ALWAYS\n{'='*70}")
    res = glob.glob(os.path.join(a.save_dir, "*_results.json"))
    kai = glob.glob(os.path.join(a.save_dir, "*_kairos_*.pt"))
    print(f"  {len(res):>4} results json      {human(size(res))}"
          f"   <- every number in the paper")
    print(f"  {len(kai):>4} kairos checkpoints {human(size(kai))}")
    live = [n for n, _ in tmux_sessions()
            if KEEP_SESSION.match(n) or session_busy(n)]
    print(f"  {len(live):>4} live sessions      {', '.join(live) or '-'}")

    print(f"\n{'='*70}\n  CANDIDATES FOR DELETION\n{'='*70}")
    print(f"  per-epoch snapshots of superseded families "
          f"{len(ep_snaps):>4}  {human(size(ep_snaps))}")
    print(f"  best checkpoints of superseded families    "
          f"{len(old_best):>4}  {human(size(old_best))}")
    print(f"  idle tmux sessions older than {a.age_days}d          "
          f"{len(stale):>4}")
    for n, c in stale:
        print(f"      {n:<22} created {time.strftime('%b %d', time.localtime(c))}")

    if not a.yes:
        print(f"\n  nothing changed. re-run with --yes to apply.\n")
        return

    freed = size(ep_snaps) + size(old_best)
    for p in ep_snaps + old_best:
        try:
            os.remove(p)
        except OSError as e:
            print(f"  [skip] {p}: {e}")
    for n, _ in stale:
        subprocess.run(["tmux", "kill-session", "-t", n],
                       capture_output=True)
    print(f"\n  removed {len(ep_snaps)+len(old_best)} files ({human(freed)}), "
          f"killed {len(stale)} sessions\n")


if __name__ == "__main__":
    main()
