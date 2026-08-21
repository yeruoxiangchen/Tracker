#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
GPU=${GPU:-0}
ROOT=/data/zjr/proobjaverse_official_slat_train2000_20260813_v1
CACHE=${ROOT}/cache_dev64_protocol2128_views8_v1
SS_REPORT=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1/ss_eval_synthetic_dev32_fixedcfg3_count125_v3/report.json
STOCK_FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json
OUT_ROOT=/data/zjr/proobjaverse_official_dev64_native_ss_stock_slat_20260814_v1/dev64_seed424344_2gpu_v1
OUT=${OUT_ROOT}/shard0_0_32
LOG=${OUT_ROOT}/shard0_resume_safe_v5.log
UNIT=tracker-official-dev64-native-ss-slat-shard0-resume-safe-v5
WORKER_LOOP=/home/zjr/Tracker/pose_point_depth_mv/background_jobs/run_official_dev64_native_ss_stock_slat_shard0_worker_loop.sh

mkdir -p "${OUT_ROOT}"
test -s "${OUT}/run_identity.json"

if [ -s "${OUT}/report.json" ]; then
  echo "reuse completed shard0 report: ${OUT}/report.json"
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
