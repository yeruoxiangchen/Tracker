#!/usr/bin/env bash
set -euo pipefail

# Reconstruct an existing phone RGB/mask/pose capture with:
# mixed no-VGGT Native SS step2000 EMA
# -> frozen released Stock SLat
# -> frozen Stock Mesh decoder.
#
# Required:
#   SOURCE_SESSION_ID=<existing phone session>
# Optional:
#   TEST_SESSION_ID=<new immutable output id>
#   GPU=<physical GPU index>
#   OUTPUT_ROOT=<phone output root>
#   NATIVE_SS_CFG=<Native SS CFG, default 3.0>
#   BYPASS_QC=1  (diagnostic only; the report remains non-formal)

PY=${PY:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
GPU=${GPU:-4}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/zjr/Tracker/pose_point_depth_mv/outputs/可视AR}
SOURCE_SESSION_ID=${SOURCE_SESSION_ID:?set SOURCE_SESSION_ID to an existing phone session}
TEST_SESSION_ID=${TEST_SESSION_ID:-${SOURCE_SESSION_ID}_native_ss_stock_slat_v1}
NATIVE_SS_CFG=${NATIVE_SS_CFG:-3.0}
BYPASS_QC=${BYPASS_QC:-0}

ARGS=(
  -u -m pose_point_depth_mv.reconstruct_existing_ar_session
  --source_session_id "${SOURCE_SESSION_ID}"
  --session_id "${TEST_SESSION_ID}"
  --output_root "${OUTPUT_ROOT}"
  --gpu "${GPU}"
  --slat_backend stock
  --native_ss_cfg_strength "${NATIVE_SS_CFG}"
)

if [[ "${BYPASS_QC}" == "1" ]]; then
  ARGS+=(--diagnostic_bypass_pose_mask_qc)
fi

CUDA_VISIBLE_DEVICES="${GPU}" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
"${PY}" "${ARGS[@]}"

REPORT="${OUTPUT_ROOT}/reconstructions/${TEST_SESSION_ID}/reconstruction_report.json"
echo "Stock SLat phone reconstruction report: ${REPORT}"
