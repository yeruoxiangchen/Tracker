#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zjr/Tracker}"
PY="${PY:-/home/zjr/anaconda3/envs/reconviagen/bin/python}"
GPU="${GPU:-4}"
RUN_NAME="${RUN_NAME:-ar_like_streaming_slam_internal7_smoke}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/trellis_point_prior_mv/outputs/ar_like_streaming_slam}"

DATASETS_DEFAULT=(
  "${ROOT}/CoarseModel/datasets/GOOD_MESH_TEST"
  "${ROOT}/CoarseModel/datasets/reconviagen_20260520_021556"
  "${ROOT}/CoarseModel/datasets/reconviagen_20260617_073549"
  "${ROOT}/CoarseModel/datasets/reconviagen_20260617_075506"
  "${ROOT}/CoarseModel/datasets/heimei"
  "${ROOT}/CoarseModel/datasets/maliao"
  "${ROOT}/CoarseModel/datasets/snoopy"
)

DATASETS_RAW="${DATASETS:-}"
if [[ -n "${DATASETS_RAW}" ]]; then
  IFS=":" read -r -a DATASET_ARGS <<< "${DATASETS_RAW}"
else
  DATASET_ARGS=("${DATASETS_DEFAULT[@]}")
fi

RUN_TRIANGULATE="${RUN_TRIANGULATE:-1}"
RUN_BUILD="${RUN_BUILD:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
PRIOR_SOURCE="${PRIOR_SOURCE:-colmap_points}"
NORMALIZATION_SOURCE="${NORMALIZATION_SOURCE:-prior_bbox}"
ALLOW_MODEL_FALLBACK="${ALLOW_MODEL_FALLBACK:-0}"
MAX_FRAMES="${MAX_FRAMES:-32}"
FRAME_SELECT="${FRAME_SELECT:-uniform}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
POINT_COUNT="${POINT_COUNT:-1500}"
MIN_PRIOR_POINTS="${MIN_PRIOR_POINTS:-40}"
TOPK_SPECS="${TOPK_SPECS:-6000,12000,16000}"
MODES="${MODES:-stock_sparse,stage2_correct}"
MESH_EVAL_SAMPLES="${MESH_EVAL_SAMPLES:-2000}"
SPARSE_SUBDIR="${SPARSE_SUBDIR:-sparse_arproxy_streaming/0}"
TRI_INPUT_SPARSE_SUBDIR="${TRI_INPUT_SPARSE_SUBDIR:-sparse/0}"

TRI_MATCHER="${TRI_MATCHER:-sequential}"
TRI_MAX_PAIR_GAP="${TRI_MAX_PAIR_GAP:-4}"
TRI_MAX_FEATURES="${TRI_MAX_FEATURES:-2048}"
TRI_FEATURE_MASK_MODE="${TRI_FEATURE_MASK_MODE:-none}"
TRI_ALLOW_PAIR_OUTSIDE_MASK="${TRI_ALLOW_PAIR_OUTSIDE_MASK:-1}"
TRI_MIN_PAIR_MATCHES="${TRI_MIN_PAIR_MATCHES:-8}"
TRI_MIN_OUTPUT_POINTS="${TRI_MIN_OUTPUT_POINTS:-${MIN_PRIOR_POINTS}}"
TRI_MIN_SUPPORT_VIEWS="${TRI_MIN_SUPPORT_VIEWS:-2}"
TRI_MIN_SUPPORT_RATIO="${TRI_MIN_SUPPORT_RATIO:-0.10}"

export ROOT PY GPU RUN_NAME OUT_ROOT
export RUN_TRIANGULATE RUN_BUILD RUN_EVAL
export PRIOR_SOURCE NORMALIZATION_SOURCE ALLOW_MODEL_FALLBACK
export MAX_FRAMES FRAME_SELECT FRAME_STRIDE POINT_COUNT MIN_PRIOR_POINTS
export TOPK_SPECS MODES MESH_EVAL_SAMPLES SPARSE_SUBDIR TRI_INPUT_SPARSE_SUBDIR
export TRI_MATCHER TRI_MAX_PAIR_GAP TRI_MAX_FEATURES TRI_FEATURE_MASK_MODE
export TRI_ALLOW_PAIR_OUTSIDE_MASK TRI_MIN_PAIR_MATCHES TRI_MIN_OUTPUT_POINTS
export TRI_MIN_SUPPORT_VIEWS TRI_MIN_SUPPORT_RATIO

DATASETS_JOINED=""
for d in "${DATASET_ARGS[@]}"; do
  if [[ -z "${DATASETS_JOINED}" ]]; then
    DATASETS_JOINED="${d}"
  else
    DATASETS_JOINED="${DATASETS_JOINED}:${d}"
  fi
done
export DATASETS="${DATASETS_JOINED}"

echo "[ar_like_streaming_slam] run=${RUN_NAME}"
echo "[ar_like_streaming_slam] datasets=${DATASETS}"
echo "[ar_like_streaming_slam] frame_select=${FRAME_SELECT} max_frames=${MAX_FRAMES} matcher=${TRI_MATCHER} gap=${TRI_MAX_PAIR_GAP}"
echo "[ar_like_streaming_slam] output=${OUT_ROOT}/${RUN_NAME}"

bash "${ROOT}/trellis_point_prior_mv/scripts/run_real_slam_prior_eval.sh"

echo "[ar_like_streaming_slam] ARpose proxy JSON files:"
for d in "${DATASET_ARGS[@]}"; do
  p="${d}/${SPARSE_SUBDIR}/arpose_tracker_slam_proxy.json"
  if [[ -f "${p}" ]]; then
    echo "  ${p}"
  fi
done
