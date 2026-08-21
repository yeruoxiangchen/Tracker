#!/usr/bin/env bash
set -euo pipefail

# Versioned two-GPU runtime wrapper.  Scientific inputs and per-object
# evaluation are delegated unchanged to the registered trajectory runner.

PROJECT=${PROJECT:-/home/zjr/Tracker}
ROOT=${ROOT:-/data/zjr/proobjaverse_official_slat_train29861_20260817_v1}
EVAL_GPUS=${EVAL_GPUS:-0,3}
EVAL_TAG=${EVAL_TAG:-2gpu03_hold}

exec env \
  PROJECT="${PROJECT}" \
  EVAL_GPUS="${EVAL_GPUS}" \
  EVAL_TAG="${EVAL_TAG}" \
  OUTPUT_ROOT="${ROOT}/eval_legacy_protocol2128_step10k30k60k70k_seed424344_${EVAL_TAG}_v1" \
  MASTER_LOG="${ROOT}/logs/eval_legacy_protocol2128_step10k30k60k70k_${EVAL_TAG}_v1.log" \
  EVAL_GPU_RESERVATION_ROOT="${ROOT}/logs/gpu_reservations_slat30k_${EVAL_TAG}_v1" \
  bash "${PROJECT}/pose_point_depth_mv/background_jobs/run_proobjaverse_slat29861_legacy_eval_4gpu.sh"
