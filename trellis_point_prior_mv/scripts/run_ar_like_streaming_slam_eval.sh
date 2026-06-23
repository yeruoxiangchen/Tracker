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

RUN_TRIANGULATE="${RUN_TRIANGULATE:-}"
RUN_BUILD="${RUN_BUILD:-1}"
RUN_EVAL="${RUN_EVAL:-0}"
RUN_RECONSTRUCT_POSE="${RUN_RECONSTRUCT_POSE:-0}"
PRIOR_BRANCH="${PRIOR_BRANCH:-streaming}"
PRIOR_SOURCE="${PRIOR_SOURCE:-colmap_points}"
NORMALIZATION_SOURCE="${NORMALIZATION_SOURCE:-prior_bbox}"
ALLOW_MODEL_FALLBACK="${ALLOW_MODEL_FALLBACK:-0}"
MAX_FRAMES="${MAX_FRAMES:-32}"
FRAME_SELECT="${FRAME_SELECT:-pose_random_farthest}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
FRAME_SELECT_SEED="${FRAME_SELECT_SEED:-42}"
POINT_COUNT="${POINT_COUNT:-1500}"
MIN_PRIOR_POINTS="${MIN_PRIOR_POINTS:-40}"
TOPK_SPECS="${TOPK_SPECS:-8192,12000,16000}"
MODES="${MODES:-stock_sparse,stage2_correct}"
MESH_EVAL_SAMPLES="${MESH_EVAL_SAMPLES:-2000}"
POSE_SPARSE_SUBDIR="${POSE_SPARSE_SUBDIR:-sparse_colmap_arproxy/0}"
case "${PRIOR_BRANCH}" in
  streaming)
    SPARSE_SUBDIR="${SPARSE_SUBDIR:-sparse_arproxy_streaming/0}"
    RUN_TRIANGULATE="${RUN_TRIANGULATE:-1}"
    ;;
  colmap_direct)
    SPARSE_SUBDIR="${SPARSE_SUBDIR:-${POSE_SPARSE_SUBDIR}}"
    RUN_TRIANGULATE="${RUN_TRIANGULATE:-0}"
    ;;
  *)
    echo "[ar_like_streaming_slam][ERROR] unsupported PRIOR_BRANCH=${PRIOR_BRANCH}; use streaming or colmap_direct" >&2
    exit 2
    ;;
esac
TRI_INPUT_SPARSE_SUBDIR="${TRI_INPUT_SPARSE_SUBDIR:-}"
if [[ -z "${TRI_INPUT_SPARSE_SUBDIR}" ]]; then
  if [[ "${RUN_RECONSTRUCT_POSE}" == "1" ]]; then
    TRI_INPUT_SPARSE_SUBDIR="${POSE_SPARSE_SUBDIR}"
  else
    TRI_INPUT_SPARSE_SUBDIR="sparse/0"
  fi
fi

COLMAP_MAX_FRAMES="${COLMAP_MAX_FRAMES:-${MAX_FRAMES}}"
COLMAP_FRAME_SELECT="${COLMAP_FRAME_SELECT:-random_uniform}"
COLMAP_FRAME_STRIDE="${COLMAP_FRAME_STRIDE:-1}"
COLMAP_FRAME_SELECT_SEED="${COLMAP_FRAME_SELECT_SEED:-${FRAME_SELECT_SEED}}"
COLMAP_MATCHER="${COLMAP_MATCHER:-sequential}"
COLMAP_SEQUENTIAL_OVERLAP="${COLMAP_SEQUENTIAL_OVERLAP:-8}"
COLMAP_MAX_FEATURES="${COLMAP_MAX_FEATURES:-4096}"
COLMAP_USE_MASKS="${COLMAP_USE_MASKS:-0}"
COLMAP_USE_GPU="${COLMAP_USE_GPU:-0}"
COLMAP_WORK_SUBDIR="${COLMAP_WORK_SUBDIR:-colmap_arproxy_work}"
COLMAP_BIN="${COLMAP_BIN:-colmap}"
COLMAP_INTRINSICS_SOURCE="${COLMAP_INTRINSICS_SOURCE:-auto}"
COLMAP_INTRINSICS_SPARSE_SUBDIR="${COLMAP_INTRINSICS_SPARSE_SUBDIR:-sparse/0}"
COLMAP_CAMERA_MODEL="${COLMAP_CAMERA_MODEL:-SIMPLE_RADIAL}"
COLMAP_CAMERA_PARAMS="${COLMAP_CAMERA_PARAMS:-}"
COLMAP_FIX_INTRINSICS="${COLMAP_FIX_INTRINSICS:-1}"

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
export PRIOR_BRANCH PRIOR_SOURCE NORMALIZATION_SOURCE ALLOW_MODEL_FALLBACK
export MAX_FRAMES FRAME_SELECT FRAME_STRIDE FRAME_SELECT_SEED POINT_COUNT MIN_PRIOR_POINTS
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
echo "[ar_like_streaming_slam] prior_branch=${PRIOR_BRANCH} sparse_subdir=${SPARSE_SUBDIR}"
echo "[ar_like_streaming_slam] frame_select=${FRAME_SELECT} max_frames=${MAX_FRAMES} matcher=${TRI_MATCHER} gap=${TRI_MAX_PAIR_GAP}"
echo "[ar_like_streaming_slam] run_reconstruct_pose=${RUN_RECONSTRUCT_POSE} pose_sparse=${TRI_INPUT_SPARSE_SUBDIR}"
echo "[ar_like_streaming_slam] output=${OUT_ROOT}/${RUN_NAME}"

if [[ "${RUN_RECONSTRUCT_POSE}" == "1" ]]; then
  COLMAP_EXTRA_ARGS=()
  if [[ "${COLMAP_USE_MASKS}" == "1" ]]; then
    COLMAP_EXTRA_ARGS+=(--use_masks)
  fi
  if [[ "${COLMAP_USE_GPU}" == "1" ]]; then
    COLMAP_EXTRA_ARGS+=(--use_gpu)
  fi
  if [[ "${COLMAP_FIX_INTRINSICS}" == "1" ]]; then
    COLMAP_EXTRA_ARGS+=(--fix_intrinsics)
  fi
  if [[ -n "${COLMAP_CAMERA_PARAMS}" ]]; then
    COLMAP_EXTRA_ARGS+=(--camera_params "${COLMAP_CAMERA_PARAMS}")
  fi
  echo "[ar_like_streaming_slam] reconstruct COLMAP pose+points -> ${POSE_SPARSE_SUBDIR}"
  "${PY}" -u "${ROOT}/trellis_point_prior_mv/reconstruct_colmap_slam_proxy.py" \
    --datasets "${DATASET_ARGS[@]}" \
    --output_sparse_subdir "${POSE_SPARSE_SUBDIR}" \
    --work_subdir "${COLMAP_WORK_SUBDIR}" \
    --max_frames "${COLMAP_MAX_FRAMES}" \
    --frame_select "${COLMAP_FRAME_SELECT}" \
    --frame_stride "${COLMAP_FRAME_STRIDE}" \
    --seed "${COLMAP_FRAME_SELECT_SEED}" \
    --matcher "${COLMAP_MATCHER}" \
    --sequential_overlap "${COLMAP_SEQUENTIAL_OVERLAP}" \
    --max_features "${COLMAP_MAX_FEATURES}" \
    --intrinsics_source "${COLMAP_INTRINSICS_SOURCE}" \
    --intrinsics_sparse_subdir "${COLMAP_INTRINSICS_SPARSE_SUBDIR}" \
    --camera_model "${COLMAP_CAMERA_MODEL}" \
    --colmap_bin "${COLMAP_BIN}" \
    --overwrite \
    --output_report "${OUT_ROOT}/${RUN_NAME}/colmap_reconstruction_report.json" \
    "${COLMAP_EXTRA_ARGS[@]}"
fi

bash "${ROOT}/trellis_point_prior_mv/scripts/run_real_slam_prior_eval.sh"

echo "[ar_like_streaming_slam] ARpose proxy JSON files:"
for d in "${DATASET_ARGS[@]}"; do
  p="${d}/${SPARSE_SUBDIR}/arpose_tracker_slam_proxy.json"
  if [[ -f "${p}" ]]; then
    echo "  ${p}"
  fi
done
