#!/usr/bin/env bash
#
# PRISM ablations: where does the gain come from?
#
#   GPUS="0 1 6 7" ./run_ablate.sh
#
# PRISM beat the shared-softmax model on WIKI (+0.39 MRR) and ICEWS18 (+0.35),
# beyond seed noise, but the stratified table put the gain on queries WITH a
# history rather than the no-history stratum it was designed for. Two switches
# separate the explanations:
#
#   --router_const   pi is one scalar for every query: does routing need
#                    to see the query?
#   --no_partition   p_N covers all entities, overlapping S: is it the
#                    DISJOINT normalisation that matters?
#
# Three seeds of each, on the two datasets where PRISM's gain is significant.
# Caches must already exist (they do after run_prism.sh); --cache reuses them.
set -euo pipefail
# Cap CPU threads per process. PyTorch otherwise sizes its pools to all 255
# cores of this shared machine; several runs then oversubscribe the CPU and
# spin -- a CPU smoke test sat at 6800% CPU for 22 minutes doing one
# minute of work. The GPU does the heavy lifting; 8 threads is plenty.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
cd "$(dirname "$0")"
mkdir -p logs checkpoints

MAX_GPUS="${MAX_GPUS:-4}"
if [[ -z "${GPUS:-}" ]]; then
  GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', ' '$2 < 1024 {print $1}' | head -n "$MAX_GPUS" | tr '\n' ' ')
fi
read -ra G <<< "$GPUS"
[[ ${#G[@]} -gt 0 ]] || { echo "no idle GPU"; exit 1; }
echo "using GPUs: ${G[*]}"

jobs=()
for ds in WIKI ICEWS18; do
  for ab in const:--router_const nopart:--no_partition; do
    name="${ab%%:*}"; flag="${ab#*:}"
    for s in 1 2 3; do
      tag="a${name}"; [[ $s -gt 1 ]] && tag="a${name}s$s"
      jobs+=("python -u train_kairos.py --dataset $ds --tag $tag --seed $s \
--prism $flag --cache 2>&1 | tee logs/ablate_${ds}_${tag}.out")
    done
  done
done

# round-robin onto one queue per GPU
declare -A Q
for i in "${!jobs[@]}"; do
  g=${G[$((i % ${#G[@]}))]}
  Q[$g]+="${jobs[$i]/--cache/--cache --gpu $g}; "
done
for g in "${G[@]}"; do
  tmux new -d -s "ablate_gpu$g" "cd $(pwd) && ${Q[$g]}"
  echo "  GPU $g: $(grep -o -- '--dataset [A-Z0-9]* --tag [a-z0-9]*' <<< "${Q[$g]}" \
       | sed 's/--dataset //; s/ --tag / /' | paste -sd',' -)"
done
echo
echo "results: python collect.py 2>/dev/null | grep -E 'prism|dataset'"
