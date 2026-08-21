#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

GPU=${GPU:-2}
OUT_ROOT=/data/zjr/proobjaverse_official_dev64_native_ss_stock_slat_20260814_v1/dev64_seed424344_2gpu_v1
OUT=${OUT_ROOT}/shard1_32_64
LOG=${OUT_ROOT}/shard1_resume_safe_v1.log
UNIT=tracker-official-dev64-native-ss-slat-shard1-resume-safe-v1
WORKER_LOOP=/home/zjr/Tracker/pose_point_depth_mv/background_jobs/run_official_dev64_native_ss_stock_slat_shard1_worker_loop.sh

mkdir -p "${OUT_ROOT}"
test -s "${OUT}/run_identity.json"

if [ -s "${OUT}/report.json" ]; then
  echo "reuse completed shard1 report: ${OUT}/report.json"
elif systemctl --user is-active --quiet "${UNIT}.service"; then
  echo "already running: ${UNIT}.service"
else
  systemd-run --user \
    --unit="${UNIT}" \
    --collect \
    --property=WorkingDirectory=/home/zjr/Tracker \
    --property=StandardOutput=append:${LOG} \
    --property=StandardError=append:${LOG} \
    /usr/bin/env \
      CUDA_VISIBLE_DEVICES="${GPU}" \
      HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 \
      ATTN_BACKEND=flash_attn \
      SPCONV_ALGO=native \
      MPLCONFIGDIR=/tmp/matplotlib \
      NUMBA_CACHE_DIR=/tmp/numba_cache \
      TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /bin/bash "${WORKER_LOOP}"
fi

echo "log: ${LOG}"
echo "report: ${OUT}/report.json"
echo "current terminal remains open"
