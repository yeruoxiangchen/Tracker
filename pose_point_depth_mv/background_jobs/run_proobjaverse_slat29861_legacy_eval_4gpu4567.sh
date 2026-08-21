#!/usr/bin/env bash
set -euo pipefail

# Versioned four-GPU runtime wrapper for the parallel source-server route.
# The 70K checkpoint is intentionally excluded; 10K/30K/60K are the selected
# trajectory anchors.

PROJECT=${PROJECT:-/home/zjr/Tracker}
ROOT=${ROOT:-/data/zjr/proobjaverse_official_slat_train29861_20260817_v1}
EVAL_GPUS=${EVAL_GPUS:-4,5,6,7}
EVAL_TAG=${EVAL_TAG:-4gpu4567_hold}

exec env \
  PROJECT="${PROJECT}" \
  EVAL_GPUS="${EVAL_GPUS}" \
  EVAL_TAG="${EVAL_TAG}" \
  STEPS=10000,30000,60000 \
  OUTPUT_ROOT="${ROOT}/eval_legacy_protocol2128_step10k30k60k_seed424344_${EVAL_TAG}_v1" \
  MASTER_LOG="${ROOT}/logs/eval_legacy_protocol2128_step10k30k60k_${EVAL_TAG}_v1.log" \
  EVAL_GPU_RESERVATION_ROOT="${ROOT}/logs/gpu_reservations_slat30k_${EVAL_TAG}_v1" \
  bash "${PROJECT}/pose_point_depth_mv/background_jobs/run_proobjaverse_slat29861_legacy_eval_4gpu.sh"
