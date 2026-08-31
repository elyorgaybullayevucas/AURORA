#!/usr/bin/env bash
#
# Launch the CURRENT model on every dataset, one tmux session each.
#
#   ./run_current.sh              # the three not already running
#   ./run_current.sh seeds        # seeds 2 and 3 of the same configuration
#
# The current configuration is the one the paper reports: phase removed,
# recurrence intensity conditioned on the subject. Its tag is "qcond", so
# collect.py groups qcond / qconds2 / qconds3 as three seeds of one
# configuration and will not merge them with the older v2 or ds0 runs.
#
# Everything is tagged. An untagged run overwrites the checkpoint and the
# results json of whatever ran before it under the same variant name, which is
# how a result that beat the state of the art was nearly lost once already.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs checkpoints

launch () {           # launch <session> <dataset> <tag> [extra args...]
  local ses="$1" ds="$2" tag="$3"; shift 3
  if tmux has-session -t "$ses" 2>/dev/null; then
    echo "  [skip] $ses already exists"
    return
  fi
  echo "  [start] $ses   $ds  --tag $tag $*"
  tmux new -d -s "$ses" \
    "python train_kairos.py --dataset $ds --tag $tag $* 2>&1 | tee logs/$ses.out"
}

if [[ "${1:-main}" == "seeds" ]]; then
  echo "seeds 2 and 3 of the current configuration:"
  for ds in YAGO WIKI ICEWS18; do
    launch "qcond_${ds}_s2" "$ds" qconds2 --seed 2
    launch "qcond_${ds}_s3" "$ds" qconds3 --seed 3
  done
else
  echo "current model, one run per dataset:"
  launch qcond_YAGO    YAGO    qcond      # ~42 min
  launch qcond_WIKI    WIKI    qcond      # ~3 h
  launch qcond_ICEWS18 ICEWS18 qcond      # ~1 h 15
  launch qcond_GDELT   GDELT   qcond      # ~8 h
fi

echo
tmux ls
echo
echo "watch:    python status.py"
echo "results:  python collect.py"
