#!/usr/bin/env bash
#
# Is PRISM's gain the new likelihood, or two settings that came with it?
#
#   GPUS="0 1 2 3" ./run_confound.sh
#
# Both PRISM ablations (constant router, overlapping supports) matched or beat
# PRISM itself on WIKI and ICEWS18, so neither component carries the gain over
# the shared-softmax model v2. PRISM differs from v2 in two further ways that
# were not part of its design:
#
#   1. struct_aux is skipped: PRISM returns no separate structural scores, so
#      train_kairos.py drops the 0.3-weighted auxiliary loss v2 trains with.
#   2. query conditioning is on: PRISM inherits the subject term from CADENCE;
#      v2 predates it.
#
# This runs plain CADENCE (no PRISM) with struct_aux = 0, with and without
# query conditioning, three seeds each. If either matches PRISM, the honest
# account is that the gain comes from those settings, not from the likelihood.
set -euo pipefail
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
  # s0q: struct_aux 0, query conditioning on   (variant "full")
  # s0n: struct_aux 0, query conditioning off  (variant "no-query-conditioning")
  for cfg in "s0q:" "s0n:--query_off"; do
    name="${cfg%%:*}"; flag="${cfg#*:}"
    for s in 1 2 3; do
      tag="$name"; [[ $s -gt 1 ]] && tag="${name}s$s"
      jobs+=("python -u train_kairos.py --dataset $ds --tag $tag --seed $s \
--struct_aux 0 $flag --cache 2>&1 | tee logs/confound_${ds}_${tag}.out")
    done
  done
done

declare -A Q
for i in "${!jobs[@]}"; do
  g=${G[$((i % ${#G[@]}))]}
  Q[$g]+="${jobs[$i]/--cache/--cache --gpu $g}; "
done
for g in "${G[@]}"; do
  tmux new -d -s "confound_gpu$g" "cd $(pwd) && ${Q[$g]}"
  echo "  GPU $g: $(grep -o -- '--dataset [A-Z0-9]* --tag [a-z0-9]*' <<< "${Q[$g]}" \
       | sed 's/--dataset //; s/ --tag / /' | paste -sd',' -)"
done
echo
echo "results: python collect.py 2>/dev/null | grep -E '^  (WIKI|ICEWS18)   \\(|v2\\) \\[|s0q|s0n|prism'"
