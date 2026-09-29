#!/usr/bin/env bash
#
# PRISM sweep, three seeds per dataset, on a SHARED machine.
#
#   ./run_prism.sh                     # up to 4 currently-idle GPUs
#   MAX_GPUS=2 ./run_prism.sh          # take at most 2
#   GPUS="0 3" ./run_prism.sh          # exactly these GPUs
#   ./run_prism.sh --no-build          # caches already built
#
# Other people use these GPUs. So:
#   - only GPUs that are idle right now (under 1 GB in use) are taken, and at
#     most MAX_GPUS of them (default 4), never all eight;
#   - the candidate cache stays in host memory (the default), so our GPU
#     footprint is the model alone and a colleague's launch on the same card
#     is less likely to OOM either job;
#   - every job for a GPU runs in one queue, one at a time, so we never hold
#     more cards than we were given.
#
# The 11 runs are spread over the chosen GPUs longest-first onto whichever
# queue is least loaded, so the sweep ends as early as the GPU count allows.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs checkpoints

MAX_GPUS="${MAX_GPUS:-4}"
if [[ -z "${GPUS:-}" ]]; then
  GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', ' '$2 < 1024 {print $1}' | head -n "$MAX_GPUS" | tr '\n' ' ')
fi
if [[ -z "${GPUS// /}" ]]; then
  echo "no idle GPU right now (all have >1 GB in use). Try later, or set GPUS=."
  exit 1
fi
echo "using GPUs: $GPUS"

if [[ "${1:-}" != "--no-build" ]]; then
  echo "building caches (CPU only) ..."
  pids=()
  for ds in YAGO ICEWS18 WIKI GDELT; do
    python build_cache.py --dataset "$ds" --cache_workers 32 \
      > "logs/cache_$ds.out" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  tail -n 1 logs/cache_*.out
fi

# plan: longest-processing-time-first over the chosen GPUs
python - "$GPUS" <<'PY' > logs/.prism_plan
import sys
gpus = sys.argv[1].split()
# (dataset, tag, seed, rough minutes on a free A100, uncached)
jobs = [("GDELT", "prism", 1, 420), ("GDELT", "prisms2", 2, 420),
        ("WIKI", "prism", 1, 150), ("WIKI", "prisms2", 2, 150),
        ("WIKI", "prisms3", 3, 150),
        ("ICEWS18", "prism", 1, 75), ("ICEWS18", "prisms2", 2, 75),
        ("ICEWS18", "prisms3", 3, 75),
        ("YAGO", "prism", 1, 45), ("YAGO", "prisms2", 2, 45),
        ("YAGO", "prisms3", 3, 45)]
load = {g: 0 for g in gpus}
queue = {g: [] for g in gpus}
for ds, tag, seed, mins in sorted(jobs, key=lambda j: -j[3]):
    g = min(load, key=load.get)
    load[g] += mins
    queue[g].append(f"{ds} {tag} {seed}")
for g in gpus:
    print(g + "|" + ";".join(queue[g]) + f"|{load[g]}")
PY

while IFS='|' read -r g jobs mins; do
  chain=""
  IFS=';' read -ra js <<< "$jobs"
  for j in "${js[@]}"; do
    read -r ds tag seed <<< "$j"
    chain+="python -u train_kairos.py --dataset $ds --tag $tag --seed $seed \
--gpu $g --prism --cache 2>&1 | tee logs/prism_${ds}_${tag}.out; "
  done
  tmux new -d -s "prism_gpu$g" "cd $(pwd) && $chain"
  echo "  GPU $g  ~$((mins / 60))h$((mins % 60))m  : ${jobs//;/ -> }"
done < logs/.prism_plan

echo
echo "watch:   python status.py"
echo "results: python collect.py"
