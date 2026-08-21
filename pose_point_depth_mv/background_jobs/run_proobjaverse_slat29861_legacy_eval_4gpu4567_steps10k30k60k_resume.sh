#!/usr/bin/env bash
set -euo pipefail

# Resume the already-started four-GPU source-server evaluation without
# recomputing finalized shards.  The existing 4gpu_v1 tree is retained as the
# immutable runtime identity; only the registered 10K/30K/60K anchors run.

PROJECT=${PROJECT:-/home/zjr/Tracker}
ROOT=${ROOT:-/data/zjr/proobjaverse_official_slat_train29861_20260817_v1}
EVAL_GPUS=${EVAL_GPUS:-4,5,6,7}

exec env \
  PROJECT="${PROJECT}" \
  EVAL_GPUS="${EVAL_GPUS}" \
  EVAL_TAG=4gpu \
  STEPS=10000,30000,60000 \
  PRESERVE_INTERRUPTED_OUTPUTS=1 \
  OUTPUT_ROOT="${ROOT}/eval_legacy_protocol2128_step10k30k60k70k_seed424344_4gpu_v1" \
  MASTER_LOG="${ROOT}/logs/eval_legacy_protocol2128_step10k30k60k70k_4gpu_v1.launcher.log" \
  EVAL_GPU_RESERVATION_ROOT="${ROOT}/logs/gpu_reservations_slat30k_4gpu_steps10k30k60k_resume_v1" \
  bash "${PROJECT}/pose_point_depth_mv/background_jobs/run_proobjaverse_slat29861_legacy_eval_4gpu.sh"
