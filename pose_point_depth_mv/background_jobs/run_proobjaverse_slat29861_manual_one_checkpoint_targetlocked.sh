#!/usr/bin/env bash
set -euo pipefail

# One invocation evaluates exactly one 29,861-object SLat checkpoint.  It never
# advances to another checkpoint.  The default scope is the only endpoint
# comparison needed here: legacy Dev48 predicted support plus strict R.
# Train64/legacy-Dev64 are optional training-overlap diagnostics selected via
# EVAL_GROUPS; neither is a held-out 30K generalization test.

PROJECT=${PROJECT:-/home/zjr/Tracker}
ROOT=${ROOT:-/data/zjr/proobjaverse_official_slat_train29861_20260817_v1}
STEP=${STEP:?set exactly one STEP: 10000, 30000, or 60000}
EVAL_GPUS=${EVAL_GPUS:-4,5,6,7}
EVAL_GROUPS=${EVAL_GROUPS:-legacy_dev48_predicted_training_overlap}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/eval_legacy_protocol2128_step10k30k60k_seed424344_4gpu_targetlocked_v2}
MASTER_LOG=${MASTER_LOG:-/dev/null}

case "${STEP}" in
  10000|30000|60000) ;;
  *) echo "ERROR: STEP must be exactly one of 10000, 30000, 60000" >&2; exit 90 ;;
esac
if [[ "${STEP}" == *,* ]]; then
  echo "ERROR: this manual entry accepts one checkpoint only" >&2
  exit 90
fi

V1_ROOT=${V1_ROOT:-${ROOT}/eval_legacy_protocol2128_step10k30k60k70k_seed424344_4gpu_v1}
FLOOR_ROOT=${OUTPUT_ROOT}/step_010000/legacy_dev48_predicted_training_overlap
if [[ ",${EVAL_GROUPS}," != *,legacy_dev48_predicted_training_overlap,* ]]; then
  SS_COORD_REUSE_ROOT=${SS_COORD_REUSE_ROOT:-}
  SS_COORD_REUSE_SOURCE_STEP=${SS_COORD_REUSE_SOURCE_STEP:-10000}
  FROZEN_STOCK_FLOOR_ROOT=${FROZEN_STOCK_FLOOR_ROOT:-}
elif (( STEP == 60000 )); then
  # The requested manual order is 30K -> 60K -> 10K.  Step30K establishes
  # the first complete target-locked A/B floor; step60K reuses it and computes
  # only its checkpoint-dependent C branch.
  FLOOR_ROOT=${OUTPUT_ROOT}/step_030000/legacy_dev48_predicted_training_overlap
  for shard in shard0_16_28 shard1_28_40 shard2_40_52 shard3_52_64; do
    test -s "${FLOOR_ROOT}/${shard}/report.json" || {
      echo "ERROR: complete target-locked 30K floor is required: ${FLOOR_ROOT}/${shard}/report.json" >&2
      exit 91
    }
  done
  SS_COORD_REUSE_ROOT=${SS_COORD_REUSE_ROOT:-${FLOOR_ROOT}}
  SS_COORD_REUSE_SOURCE_STEP=${SS_COORD_REUSE_SOURCE_STEP:-30000}
  FROZEN_STOCK_FLOOR_ROOT=${FROZEN_STOCK_FLOOR_ROOT:-${FLOOR_ROOT}}
else
  # Step30K is the first full target-locked run in the requested order.  The
  # final step10K run resumes its existing partial output without changing its
  # run identity.  Both reuse only the old verified SS coordinates.
  SS_COORD_REUSE_ROOT=${SS_COORD_REUSE_ROOT:-${V1_ROOT}/step_010000/legacy_dev48_predicted_training_overlap}
  SS_COORD_REUSE_SOURCE_STEP=${SS_COORD_REUSE_SOURCE_STEP:-10000}
  FROZEN_STOCK_FLOOR_ROOT=${FROZEN_STOCK_FLOOR_ROOT:-}
fi

GROUP_TAG=${EVAL_GROUPS//,/_}
SUMMARY_PATH=${SUMMARY_PATH:-${OUTPUT_ROOT}/step_$(printf '%06d' "${STEP}")/manual_${GROUP_TAG}_summary.json}
EVAL_GPU_RESERVATION_ROOT=${EVAL_GPU_RESERVATION_ROOT:-${ROOT}/logs/gpu_reservations_slat30k_manual_step$(printf '%06d' "${STEP}")}

# A final manual invocation may only need the two CPU aggregates because all
# four GPU worker reports were completed before the old automatic scheduler was
# disarmed.  Avoid waiting for or reserving GPUs in that case.
SKIP_GPU_SETUP=${SKIP_GPU_SETUP:-0}
if [[ "${EVAL_GROUPS}" == "legacy_dev48_predicted_training_overlap" ]]; then
  predicted_root=${OUTPUT_ROOT}/step_$(printf '%06d' "${STEP}")/legacy_dev48_predicted_training_overlap
  complete=1
  for spec in shard0_16_28 shard1_28_40 shard2_40_52 shard3_52_64; do
    [[ -s "${predicted_root}/${spec}/report.json" ]] || complete=0
  done
  (( complete == 0 )) || SKIP_GPU_SETUP=1
fi
if (( SKIP_GPU_SETUP == 0 )); then
  nvidia-smi -L >/dev/null || {
    echo "ERROR: NVIDIA driver is unavailable; do not launch/resume evaluation" >&2
    exit 89
  }
fi

exec env \
  PROJECT="${PROJECT}" \
  EVAL_GPUS="${EVAL_GPUS}" \
  EVAL_TAG="manual_step$(printf '%06d' "${STEP}")" \
  STEPS="${STEP}" \
  EVAL_GROUPS="${EVAL_GROUPS}" \
  PRESERVE_INTERRUPTED_OUTPUTS=0 \
  SS_COORD_REUSE_ROOT="${SS_COORD_REUSE_ROOT}" \
  SS_COORD_REUSE_SOURCE_STEP="${SS_COORD_REUSE_SOURCE_STEP}" \
  FROZEN_STOCK_FLOOR_ROOT="${FROZEN_STOCK_FLOOR_ROOT}" \
  SKIP_GPU_SETUP="${SKIP_GPU_SETUP}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  MASTER_LOG="${MASTER_LOG}" \
  SUMMARY_PATH="${SUMMARY_PATH}" \
  EVAL_GPU_RESERVATION_ROOT="${EVAL_GPU_RESERVATION_ROOT}" \
  bash "${PROJECT}/pose_point_depth_mv/background_jobs/run_proobjaverse_slat29861_legacy_eval_4gpu.sh"
