#!/usr/bin/env bash
#
# Full PRISM sweep on all 8 GPUs: three seeds per dataset, candidate cache on.
#
#   ./run_prism.sh              # build caches, then launch
#   ./run_prism.sh --no-build   # caches already built
#
# Each GPU runs a QUEUE, not a single job: the short runs are chained behind
# each other so no GPU idles while GDELT is still training.
#
#   GPU 0-2 : ICEWS18 seed 1/2/3, then YAGO seed 1/2/3
#   GPU 3-5 : WIKI seed 1/2/3
#   GPU 6-7 : GDELT seed 1/2
#
# Tags prism / prisms2 / prisms3 are three seeds of ONE configuration, which is
# how collect.py groups them for mean +- std.
#
# Caches are built FIRST and to completion. Three runs of one dataset starting
# together would otherwise all find the cache missing and write the same files
# at once.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs checkpoints

EXTRA="--prism --cache"          # label smoothing left at its default: on YAGO
                                 # it beat the pure-factorisation variant

if [[ "${1:-}" != "--no-build" ]]; then
  echo "building caches (CPU only) ..."
  pids=()
  for ds in YAGO ICEWS18 WIKI GDELT; do
    python build_cache.py --dataset "$ds" --cache_workers 48 \
      > "logs/cache_$ds.out" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  tail -n 1 logs/cache_*.out
fi

run () {            # run <dataset> <tag> <seed> <gpu>
  echo "python train_kairos.py --dataset $1 --tag $2 --seed $3 --gpu $4 $EXTRA \
2>&1 | tee logs/prism_$1_$2.out"
}

queue () {          # queue <session> <cmd1> [cmd2 ...]  -- run in order
  local ses="$1"; shift
  local chain; chain=$(printf '%s; ' "$@")
  tmux new -d -s "$ses" "cd $(pwd) && $chain"
  echo "  [start] $ses"
}

queue gpu0 "$(run ICEWS18 prism 1 0)"   "$(run YAGO prism 1 0)"
queue gpu1 "$(run ICEWS18 prisms2 2 1)" "$(run YAGO prisms2 2 1)"
queue gpu2 "$(run ICEWS18 prisms3 3 2)" "$(run YAGO prisms3 3 2)"
queue gpu3 "$(run WIKI prism 1 3)"
queue gpu4 "$(run WIKI prisms2 2 4)"
queue gpu5 "$(run WIKI prisms3 3 5)"
queue gpu6 "$(run GDELT prism 1 6)"
queue gpu7 "$(run GDELT prisms2 2 7)"

echo
tmux ls
echo "watch:   python status.py"
echo "results: python collect.py"
