#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
SLAT_ROOT=${SLAT_ROOT:-/data/zjr/proobjaverse_official_slat_train2000_20260813_v1}
BASE_SS_ROOT=${BASE_SS_ROOT:-/data/zjr/proobjaverse_official_native_ss_train2000_20260815_v1}
RUN_ROOT=${RUN_ROOT:-/data/zjr/proobjaverse_official_native_ss_train2000_with_vggt_20260817_v1}
BUILD_GPUS=${BUILD_GPUS:-0,1,2,3,4,5,6,7}

mkdir -p "${RUN_ROOT}/logs"

SPLIT_MANIFEST="${SLAT_ROOT}/protocol2128_train2000_v1/train.json" \
BASE_MANIFEST="${BASE_SS_ROOT}/cache_train2000_official_ss_v1/lifting_manifest.json" \
OUTPUT_DIR="${RUN_ROOT}/cache_train2000_official_ss_with_vggt_sidecar_historical_v2_fix2_v1" \
BUILD_GPUS="${BUILD_GPUS}" \
PROJECT_ROOT="${PROJECT_ROOT}" \
  bash "${PROJECT_ROOT}/official_ss_with_vggt_perf_v1/run_build_cache_8gpu.sh"

SPLIT_MANIFEST="${SLAT_ROOT}/protocol2128_train2000_v1/dev.json" \
BASE_MANIFEST="${BASE_SS_ROOT}/cache_dev64_official_ss_v1/lifting_manifest.json" \
OUTPUT_DIR="${RUN_ROOT}/cache_dev64_official_ss_with_vggt_sidecar_historical_v2_fix2_v1" \
BUILD_GPUS="${BUILD_GPUS}" \
PROJECT_ROOT="${PROJECT_ROOT}" \
  bash "${PROJECT_ROOT}/official_ss_with_vggt_perf_v1/run_build_cache_8gpu.sh"

echo "WITH-VGGT OFFICIAL SS TRAIN2000 + DEV64 CACHES COMPLETE"
