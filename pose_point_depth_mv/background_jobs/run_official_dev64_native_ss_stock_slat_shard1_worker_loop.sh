#!/usr/bin/env bash
set -uo pipefail

cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
ROOT=/data/zjr/proobjaverse_official_slat_train2000_20260813_v1
CACHE=${ROOT}/cache_dev64_protocol2128_views8_v1
SS_REPORT=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1/ss_eval_synthetic_dev32_fixedcfg3_count125_v3/report.json
STOCK_FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json
OUT=/data/zjr/proobjaverse_official_dev64_native_ss_stock_slat_20260814_v1/dev64_seed424344_2gpu_v1/shard1_32_64

while true; do
  "${PY}" -u -m pose_point_depth_mv.evaluate_proobjaverse_official_native_ss_stock_slat \
    worker \
    --cache_manifest "${CACHE}/slat_manifest.json" \
    --lifting_cache_manifest "${CACHE}/lifting_manifest.json" \
    --native_ss_report "${SS_REPORT}" \
    --stock_slat_freeze "${STOCK_FREEZE}" \
    --output_dir "${OUT}" \
    --joint_seeds 42,43,44 \
    --object_start 32 \
    --object_end 64 \
    --surface_samples 20000 \
    --amp_dtype bf16 \
    --resume \
    --restart_after_recorded_failure
  rc=$?
  if [ "${rc}" -eq 75 ]; then
    echo "[official_ss:supervisor] restarting after recorded topology failure"
    continue
  fi
  exit "${rc}"
done
