#!/usr/bin/env python
"""
Build the candidate cache for a dataset without training anything.

    python build_cache.py --dataset GDELT --cache_workers 48

Uses no GPU. Safe to run next to trainings: they read the cache only when
started with --cache, and a split is marked complete only after its last file
is written, so a half-built cache is never served.
"""
import sys

from kairos.cache import ensure
from kairos.config import parse_args
from kairos.data import KairosData

if __name__ == "__main__":
    if "--cache" not in sys.argv:
        sys.argv.append("--cache")
    cfg = parse_args()
    data = KairosData(cfg)
    d = ensure(cfg, data, cfg.cache_workers or None)
    print(f"[cache] ready: {d}")
