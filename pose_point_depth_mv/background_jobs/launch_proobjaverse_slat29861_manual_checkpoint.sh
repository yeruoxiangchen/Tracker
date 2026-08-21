#!/usr/bin/env bash
set -euo pipefail

STEP=${1:?usage: $0 10000|30000|60000}
case "${STEP}" in
  10000|30000|60000) ;;
  *) echo "ERROR: STEP must be 10000, 30000, or 60000" >&2; exit 90 ;;
esac

PROJECT=${PROJECT:-/home/zjr/Tracker}
ROOT=${ROOT:-/data/zjr/proobjaverse_official_slat_train29861_20260817_v1}
EVAL_GPUS=${EVAL_GPUS:-4,5,6,7}
SESSION=slat30k_step${STEP}_manual
LOG=${ROOT}/logs/manual_step${STEP}_targetlocked_v2.log
SUMMARY=${ROOT}/eval_legacy_protocol2128_step10k30k60k_seed424344_4gpu_targetlocked_v2/step_$(printf '%06d' "${STEP}")/manual_legacy_dev48_predicted_training_overlap_summary.json

mkdir -p "$(dirname "${LOG}")"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "ERROR: ${SESSION} is already running" >&2
  exit 91
fi

tmux new-session -d -s "${SESSION}" \
  "bash -lc 'cd ${PROJECT} && STEP=${STEP} EVAL_GPUS=${EVAL_GPUS} exec bash pose_point_depth_mv/background_jobs/run_proobjaverse_slat29861_manual_one_checkpoint_targetlocked.sh' >> \"${LOG}\" 2>&1"

sleep 3
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "RUNNING: session=${SESSION} step=${STEP} log=${LOG}"
elif [[ -s "${SUMMARY}" ]]; then
  echo "COMPLETE: step=${STEP} summary=${SUMMARY}"
else
  echo "ERROR: ${SESSION} exited; inspect ${LOG}" >&2
  exit 92
fi
