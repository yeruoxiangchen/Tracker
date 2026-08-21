#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=${PY:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
GPU=${GPU:-0}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/zjr/Tracker/pose_point_depth_mv/outputs/可视AR}
LOG_ROOT=${OUTPUT_ROOT}/logs
RUN_LOG=${LOG_ROOT}/real_official_slat_step8000_two_cases_20260816.log

mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${RUN_LOG}") 2>&1

SOURCE=/data/zjr/proobjaverse_official_slat_train2000_20260813_v1
SS_RUN=/data/zjr/proobjaverse_official_native_ss_train2000_20260815_v1

SS_REPORT=${SS_RUN}/dev64_step2000_eval16_64_seed424344_6gpu_v1/aggregate_v1/report.json
SLAT_CKPT=${SOURCE}/B_condition_lora_train2000_step8000_seed42_4gpu_v1/checkpoints/step_008000.pt
EXPECTED_SLAT_SHA=49edb3bbdbd86b10c5eea14e9c80a9996076b6fd65a459db12b130b6560bda4d
BRIDGE=${SS_RUN}/dev48_newss2000_stock_and_slat8000_mesh_seed424344_5gpu_v1/aggregate_v1/report.json
STOCK_FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json

COARSE_DATA=/home/zjr/Tracker/CoarseModel/datasets/reconviagen_20260520_021556
PHONE_DATA=${OUTPUT_ROOT}/datasets/20260812_171117_303

COARSE_ID=real_official_slat_step8000_reconviagen_20260520_021556_min8_diagnostic_seed42_v1
PHONE_ID=real_official_slat_step8000_ar_20260812_171117_303_seed42_v1

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ATTN_BACKEND=flash_attn
export SPCONV_ALGO=native
export MPLCONFIGDIR=/tmp/matplotlib
export NUMBA_CACHE_DIR=/tmp/numba_cache
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions

for REQUIRED in "${PY}" "${SS_REPORT}" "${SLAT_CKPT}" "${BRIDGE}" "${STOCK_FREEZE}"; do
  test -s "${REQUIRED}"
done
test -d "${COARSE_DATA}"
test -d "${PHONE_DATA}"
ACTUAL_SLAT_SHA=$(sha256sum "${SLAT_CKPT}" | awk '{print $1}')
test "${ACTUAL_SLAT_SHA}" = "${EXPECTED_SLAT_SHA}"

echo "===== real-data step8000 deployment ====="
echo "started_utc=$(date -u -Is)"
echo "gpu=${GPU}"
echo "slat_checkpoint=${SLAT_CKPT}"
echo "slat_checkpoint_sha256=${ACTUAL_SLAT_SHA}"
echo "bridge_report=${BRIDGE}"
echo "run_log=${RUN_LOG}"

echo "===== case 1/2: CoarseModel real capture, original point+mask selector ====="
echo "NOTE: this historical capture has only 8 mask-supported sparse points;"
echo "      min_object_points=8 is an explicit qualitative diagnostic exception."
"${PY}" -u -m pose_point_depth_mv.reconstruct_real_proobjaverse_official_ss_slat \
  --dataset_dir "${COARSE_DATA}" \
  --session_id "${COARSE_ID}" \
  --output_root "${OUTPUT_ROOT}" \
  --gpu "${GPU}" \
  --geometry_mode point_mask \
  --view_selection_policy lexical_even_valid_mask_fallback \
  --selected_view_count 8 \
  --min_object_points 8 \
  --min_mask_observations 2 \
  --min_mask_support_ratio 0.50 \
  --native_ss_report "${SS_REPORT}" \
  --native_slat_checkpoint "${SLAT_CKPT}" \
  --expected_slat_step 8000 \
  --cross_deployment_bridge_report "${BRIDGE}" \
  --stock_slat_freeze "${STOCK_FREEZE}" \
  --seed 42 \
  --amp_dtype bf16

echo "===== case 2/2: phone AR capture, original pose+mask azimuth selector ====="
"${PY}" -u -m pose_point_depth_mv.reconstruct_real_proobjaverse_official_ss_slat \
  --dataset_dir "${PHONE_DATA}" \
  --session_id "${PHONE_ID}" \
  --output_root "${OUTPUT_ROOT}" \
  --gpu "${GPU}" \
  --geometry_mode pose_mask \
  --view_selection_policy object_azimuth_balanced_valid_mask \
  --gravity_up_w 0 1 0 \
  --selected_view_count 8 \
  --native_ss_report "${SS_REPORT}" \
  --native_slat_checkpoint "${SLAT_CKPT}" \
  --expected_slat_step 8000 \
  --cross_deployment_bridge_report "${BRIDGE}" \
  --stock_slat_freeze "${STOCK_FREEZE}" \
  --seed 42 \
  --amp_dtype bf16

echo "REAL OFFICIAL SLAT STEP8000 TWO-CASE RECONSTRUCTION COMPLETE"
