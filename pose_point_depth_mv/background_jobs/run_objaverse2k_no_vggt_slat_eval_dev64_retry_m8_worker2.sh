#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
GPU=${OBJAVERSE2K_SLAT_RETRY_GPU:-2}
STEP=${OBJAVERSE2K_SLAT_EVAL_STEP:-2000}
RUN=${OBJAVERSE2K_SLAT_RUN:-/data/zjr/objaverse2k_no_vggt_slat_20260811_v1}
STEP_PAD=$(printf '%06d' "${STEP}")
DEV_CACHE=${RUN}/slat_cache_dev64_seed424344_merged_v1/manifest.json
DEV_LIFT=${RUN}/split_dev64_v1/dev/lifting_manifest.json
SS_RUN=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1
SS_REPORT=${SS_RUN}/ss_eval_synthetic_dev32_fixedcfg3_count125_v3/report.json
STOCK_FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json
M8=${SS_RUN}/slat_mixed_step2000_seed42_2gpu_v1/checkpoints/step_002000.pt
OUT=${RUN}/eval_dev64_step${STEP_PAD}_stock_m8_objaverse2k_8gpu_v1
WORKER_OUT=${OUT}/m8_worker_2
LOG=${RUN}/logs/eval_dev64_step${STEP_PAD}_m8_worker_2_retry.log
LOCK=${RUN}/logs/eval_dev64_step${STEP_PAD}_m8_worker_2_retry.lock

for REQUIRED in \
  "${DEV_CACHE}" "${DEV_LIFT}" "${SS_REPORT}" "${STOCK_FREEZE}" "${M8}"; do
  test -s "${REQUIRED}"
done
if [ -s "${WORKER_OUT}/report.json" ]; then
  echo "m8_worker_2 already complete: ${WORKER_OUT}/report.json"
  exit 0
fi
mkdir -p "${RUN}/logs"
exec 9>"${LOCK}"
if ! flock -n 9; then echo "m8_worker_2 retry is already running" >&2; exit 99; fi

RESUME=()
if [ -e "${WORKER_OUT}" ]; then RESUME=(--resume); fi
CUDA_VISIBLE_DEVICES="${GPU}" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${PY}" -u -m pose_point_depth_mv.evaluate_objaverse2k_no_vggt_slat worker \
  --cache_manifest "${DEV_CACHE}" \
  --lifting_cache_manifest "${DEV_LIFT}" \
  --checkpoint "${M8}" \
  --model_label m8 \
  --native_ss_report "${SS_REPORT}" \
  --stock_slat_freeze "${STOCK_FREEZE}" \
  --output_dir "${WORKER_OUT}" \
  --weights ema --joint_seeds 42,43,44 --noise_seed 20260811 \
  --worker_index 2 --num_workers 4 --expected_objects 64 \
  --surface_samples 20000 --amp_dtype bf16 "${RESUME[@]}" \
  >"${LOG}" 2>&1

test -s "${WORKER_OUT}/report.json"
