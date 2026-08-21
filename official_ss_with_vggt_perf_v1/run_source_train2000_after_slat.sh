#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
RUN_ROOT=${RUN_ROOT:-/data/zjr/proobjaverse_official_native_ss_train2000_with_vggt_20260817_v1}
TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3,4,5,6,7}
CACHE_MANIFEST=${CACHE_MANIFEST:-${RUN_ROOT}/cache_train2000_official_ss_with_vggt_sidecar_historical_v2_fix2_v1/with_vggt_ss_manifest.json}
OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/VSS_with_vggt_train2000_step2000_seed42_8gpu_v1}
LOG=${LOG:-${RUN_ROOT}/logs/VSS_with_vggt_train2000_step2000_seed42_8gpu_v1.log}

mkdir -p "${RUN_ROOT}/logs"
test -s "${CACHE_MANIFEST}"

set +e
CACHE_MANIFEST="${CACHE_MANIFEST}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
TRAIN_GPUS="${TRAIN_GPUS}" \
PROJECT_ROOT="${PROJECT_ROOT}" \
  bash "${PROJECT_ROOT}/official_ss_with_vggt_perf_v1/run_train_8gpu.sh" \
  2>&1 | tee "${LOG}"
STATUS=${PIPESTATUS[0]}
set -e
printf '%s\n' "${STATUS}" >"${LOG}.exit_code"
exit "${STATUS}"
