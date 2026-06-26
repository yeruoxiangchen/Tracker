#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zjr/Tracker}"
PY="${PY:-/home/zjr/anaconda3/envs/reconviagen/bin/python}"
GPU="${GPU:-4}"
RUN_NAME="${RUN_NAME:-ar_session_smoke_$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/trellis_point_prior_mv/outputs/ar_session_smoke}"

SESSION_ID="${SESSION_ID:-}"
SESSION_DATA_DIR="${SESSION_DATA_DIR:-}"
SESSION_MASK_DIR="${SESSION_MASK_DIR:-}"
SLAM_POINTS_JSONL="${SLAM_POINTS_JSONL:-}"
DATASET_NAME="${DATASET_NAME:-}"
SELECTED_INDICES="${SELECTED_INDICES:-}"

RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_TRIANGULATE="${RUN_TRIANGULATE:-}"
RUN_BUILD="${RUN_BUILD:-1}"
RUN_EVAL="${RUN_EVAL:-0}"
PRIOR_BRANCH="${PRIOR_BRANCH:-pose_streaming}"

POSE_SPARSE_SUBDIR="${POSE_SPARSE_SUBDIR:-sparse/0}"
DIRECT_SPARSE_SUBDIR="${DIRECT_SPARSE_SUBDIR:-sparse_ar_direct/0}"
STREAMING_SPARSE_SUBDIR="${STREAMING_SPARSE_SUBDIR:-sparse_ar_streaming/0}"
NORMALIZATION_SOURCE="${NORMALIZATION_SOURCE:-prior_bbox}"
MAX_FRAMES="${MAX_FRAMES:-32}"
FRAME_SELECT="${FRAME_SELECT:-pose_random_farthest}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
FRAME_SELECT_SEED="${FRAME_SELECT_SEED:-42}"
POINT_COUNT="${POINT_COUNT:-1500}"
MIN_PRIOR_POINTS="${MIN_PRIOR_POINTS:-20}"
ALLOW_MODEL_FALLBACK="${ALLOW_MODEL_FALLBACK:-0}"

TRI_MATCHER="${TRI_MATCHER:-sequential}"
TRI_MAX_PAIR_GAP="${TRI_MAX_PAIR_GAP:-4}"
TRI_MAX_FEATURES="${TRI_MAX_FEATURES:-2048}"
TRI_FEATURE_MASK_MODE="${TRI_FEATURE_MASK_MODE:-none}"
TRI_ALLOW_PAIR_OUTSIDE_MASK="${TRI_ALLOW_PAIR_OUTSIDE_MASK:-1}"
TRI_MIN_PAIR_MATCHES="${TRI_MIN_PAIR_MATCHES:-8}"
TRI_MIN_OUTPUT_POINTS="${TRI_MIN_OUTPUT_POINTS:-${MIN_PRIOR_POINTS}}"
TRI_MIN_SUPPORT_VIEWS="${TRI_MIN_SUPPORT_VIEWS:-2}"
TRI_MIN_SUPPORT_RATIO="${TRI_MIN_SUPPORT_RATIO:-0.10}"
TRIANGULATE_OVERWRITE="${TRIANGULATE_OVERWRITE:-1}"

TOPK_SPECS="${TOPK_SPECS:-8192,12000,16000}"
MODES="${MODES:-stock_sparse,stage2_correct}"
INDICES="${INDICES:-all}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-${ROOT}/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_rank_w0005_ws05_s200_seed42/checkpoints/last.ckpt}"
WEIGHTS="${WEIGHTS:-microsoft/TRELLIS-image-large}"
SS_STEPS="${SS_STEPS:-12}"
SLAT_STEPS="${SLAT_STEPS:-12}"
MESH_EVAL_SAMPLES="${MESH_EVAL_SAMPLES:-2000}"
STAGE2_BASE_GUIDANCE="${STAGE2_BASE_GUIDANCE:-none}"
STAGE2_BASE_RADIUS="${STAGE2_BASE_RADIUS:-3.0}"
STAGE2_BASE_MIN_CANDIDATES="${STAGE2_BASE_MIN_CANDIDATES:-512}"
STAGE2_UNION_STOCK="${STAGE2_UNION_STOCK:-0}"
STAGE2_SPARSE_FILTER="${STAGE2_SPARSE_FILTER:-none}"
FILTER_PRIOR_RADIUS="${FILTER_PRIOR_RADIUS:-4.0}"
FILTER_MIN_COMPONENT_SIZE="${FILTER_MIN_COMPONENT_SIZE:-64}"
FILTER_MIN_SUPPORT_VIEWS="${FILTER_MIN_SUPPORT_VIEWS:-1}"
FILTER_MIN_SUPPORT_RATIO="${FILTER_MIN_SUPPORT_RATIO:-0.0}"
VISUAL_HULL_MIN_VISIBLE_VIEWS="${VISUAL_HULL_MIN_VISIBLE_VIEWS:-1}"
VISUAL_HULL_MIN_SUPPORT_RATIO="${VISUAL_HULL_MIN_SUPPORT_RATIO:-0.0}"
FILTER_MIN_COORDS="${FILTER_MIN_COORDS:-128}"
FILTER_FALLBACK_UNFILTERED="${FILTER_FALLBACK_UNFILTERED:-0}"

RUN_DIR="${OUT_ROOT}/${RUN_NAME}"
PREPARE_DIR="${RUN_DIR}/prepare"
DATASET_ROOT="${PREPARE_DIR}/dataset"
DATASET_DIR="${DATASET_DIR:-}"

cd "${ROOT}"
mkdir -p "${RUN_DIR}"

prepare_args=()
if [[ -n "${SESSION_ID}" ]]; then
  prepare_args+=(--session_id "${SESSION_ID}")
fi
if [[ -n "${SESSION_DATA_DIR}" ]]; then
  prepare_args+=(--session_data_dir "${SESSION_DATA_DIR}")
fi
if [[ -n "${SESSION_MASK_DIR}" ]]; then
  prepare_args+=(--session_mask_dir "${SESSION_MASK_DIR}")
fi
if [[ -n "${SLAM_POINTS_JSONL}" ]]; then
  prepare_args+=(--slam_points_jsonl "${SLAM_POINTS_JSONL}")
fi
if [[ -n "${DATASET_NAME}" ]]; then
  prepare_args+=(--dataset_name "${DATASET_NAME}")
fi
if [[ -n "${SELECTED_INDICES}" ]]; then
  prepare_args+=(--selected_indices "${SELECTED_INDICES}")
fi

if [[ "${RUN_PREPARE}" == "1" ]]; then
  echo "[ar_session_smoke] prepare session -> ${PREPARE_DIR}"
  "${PY}" -u trellis_point_prior_mv/build_ar_session_smoke_dataset.py \
    --output_dir "${PREPARE_DIR}" \
    --pose_sparse_subdir "${POSE_SPARSE_SUBDIR}" \
    --direct_sparse_subdir "${DIRECT_SPARSE_SUBDIR}" \
    --overwrite \
    "${prepare_args[@]}"
fi

if [[ -z "${DATASET_DIR}" ]]; then
  mapfile -t dataset_candidates < <(find "${DATASET_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort)
  if [[ "${#dataset_candidates[@]}" -ne 1 ]]; then
    echo "[ar_session_smoke][ERROR] expected one prepared dataset under ${DATASET_ROOT}, got ${#dataset_candidates[@]}" >&2
    exit 2
  fi
  DATASET_DIR="${dataset_candidates[0]}"
fi

case "${PRIOR_BRANCH}" in
  ar_direct)
    SPARSE_SUBDIR="${SPARSE_SUBDIR:-${DIRECT_SPARSE_SUBDIR}}"
    RUN_TRIANGULATE="${RUN_TRIANGULATE:-0}"
    ;;
  pose_streaming)
    SPARSE_SUBDIR="${SPARSE_SUBDIR:-${STREAMING_SPARSE_SUBDIR}}"
    RUN_TRIANGULATE="${RUN_TRIANGULATE:-1}"
    ;;
  pose_only)
    SPARSE_SUBDIR="${SPARSE_SUBDIR:-${POSE_SPARSE_SUBDIR}}"
    RUN_TRIANGULATE="${RUN_TRIANGULATE:-0}"
    ;;
  *)
    echo "[ar_session_smoke][ERROR] unsupported PRIOR_BRANCH=${PRIOR_BRANCH}; use ar_direct, pose_streaming, or pose_only" >&2
    exit 2
    ;;
esac

if [[ "${RUN_TRIANGULATE}" == "1" && "${SPARSE_SUBDIR}" == "${POSE_SPARSE_SUBDIR}" ]]; then
  echo "[ar_session_smoke][ERROR] refusing to triangulate into input sparse dir ${SPARSE_SUBDIR}" >&2
  exit 2
fi

echo "[ar_session_smoke] run=${RUN_NAME}"
echo "[ar_session_smoke] dataset=${DATASET_DIR}"
echo "[ar_session_smoke] branch=${PRIOR_BRANCH} sparse=${SPARSE_SUBDIR}"
echo "[ar_session_smoke] eval modes=${MODES} union_stock=${STAGE2_UNION_STOCK}"
echo "[ar_session_smoke] output=${RUN_DIR}"

if [[ "${RUN_TRIANGULATE}" == "1" ]]; then
  tri_extra=()
  if [[ "${TRIANGULATE_OVERWRITE}" == "1" ]]; then
    tri_extra+=(--overwrite)
  fi
  if [[ "${TRI_ALLOW_PAIR_OUTSIDE_MASK}" == "1" ]]; then
    tri_extra+=(--allow_pair_outside_mask)
  fi
  echo "[ar_session_smoke] triangulate pose-streaming points -> ${SPARSE_SUBDIR}"
  "${PY}" -u trellis_point_prior_mv/triangulate_slam_like_points.py \
    --datasets "${DATASET_DIR}" \
    --input_sparse_subdir "${POSE_SPARSE_SUBDIR}" \
    --output_sparse_subdir "${SPARSE_SUBDIR}" \
    --max_frames "${MAX_FRAMES}" \
    --frame_select "${FRAME_SELECT}" \
    --frame_stride "${FRAME_STRIDE}" \
    --frame_select_seed "${FRAME_SELECT_SEED}" \
    --max_features "${TRI_MAX_FEATURES}" \
    --feature_mask_mode "${TRI_FEATURE_MASK_MODE}" \
    --matcher "${TRI_MATCHER}" \
    --max_pair_gap "${TRI_MAX_PAIR_GAP}" \
    --ratio_test "${TRI_RATIO_TEST:-0.75}" \
    --min_pair_matches "${TRI_MIN_PAIR_MATCHES}" \
    --max_reproj_error "${TRI_MAX_REPROJ_ERROR:-4.0}" \
    --min_triangulation_angle_deg "${TRI_MIN_ANGLE_DEG:-1.0}" \
    --min_support_views "${TRI_MIN_SUPPORT_VIEWS}" \
    --min_support_ratio "${TRI_MIN_SUPPORT_RATIO}" \
    --merge_voxel_size "${TRI_MERGE_VOXEL_SIZE:-0.002}" \
    --min_output_points "${TRI_MIN_OUTPUT_POINTS}" \
    --max_output_points "${TRI_MAX_OUTPUT_POINTS:-50000}" \
    --output_report "${RUN_DIR}/slam_like_points_report.json" \
    "${tri_extra[@]}"
fi

MANIFEST="${MANIFEST:-${RUN_DIR}/manifest/manifest.json}"
BUILD_DIR="$(dirname "${MANIFEST}")"
if [[ "${RUN_BUILD}" == "1" ]]; then
  build_extra=()
  if [[ "${ALLOW_MODEL_FALLBACK}" == "1" ]]; then
    build_extra+=(--allow_model_fallback)
  fi
  echo "[ar_session_smoke] build prior manifest -> ${BUILD_DIR}"
  "${PY}" -u trellis_point_prior_mv/build_real_slam_prior_manifest.py \
    --datasets "${DATASET_DIR}" \
    --output_dir "${BUILD_DIR}" \
    --prior_source colmap_points \
    --sparse_subdir "${SPARSE_SUBDIR}" \
    --normalization_source "${NORMALIZATION_SOURCE}" \
    --max_frames "${MAX_FRAMES}" \
    --frame_select "${FRAME_SELECT}" \
    --frame_stride "${FRAME_STRIDE}" \
    --frame_select_seed "${FRAME_SELECT_SEED}" \
    --point_count "${POINT_COUNT}" \
    --min_prior_points "${MIN_PRIOR_POINTS}" \
    --seed 42 \
    "${build_extra[@]}"
fi

EVAL_DIR="${OUTPUT_DIR:-${RUN_DIR}/mesh_eval}"
if [[ "${RUN_EVAL}" == "1" ]]; then
  filter_extra=()
  if [[ "${FILTER_FALLBACK_UNFILTERED}" == "1" ]]; then
    filter_extra+=(--filter_fallback_unfiltered)
  fi
  if [[ "${STAGE2_UNION_STOCK}" == "1" ]]; then
    filter_extra+=(--stage2_union_stock)
  fi
  echo "[ar_session_smoke] mesh eval -> ${EVAL_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn \
  SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib \
  NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PY}" -u trellis_point_prior_mv/run_real_slam_prior_mesh.py \
    --manifest "${MANIFEST}" \
    --output_dir "${EVAL_DIR}" \
    --weights "${WEIGHTS}" \
    --indices "${INDICES}" \
    --modes "${MODES}" \
    --stage2_checkpoint "${STAGE2_CHECKPOINT}" \
    --stage2_topk_specs "${TOPK_SPECS}" \
    --max_frames "${MAX_FRAMES}" \
    --cond_mode multi_stochastic \
    --ss_steps "${SS_STEPS}" \
    --slat_steps "${SLAT_STEPS}" \
    --ss_guidance_strength 7.5 \
    --slat_guidance_strength 7.5 \
    --slat_guidance_rescale 0.5 \
    --slat_rescale_t 3.0 \
    --steps 12 \
    --guidance_strength 1.0 \
    --known_latent_clamp_strength 1.0 \
    --known_clamp_start_t 0.5 \
    --known_logit_boost 0.0 \
    --known_conf_power 1.0 \
    --stage2_base_guidance "${STAGE2_BASE_GUIDANCE}" \
    --stage2_base_radius "${STAGE2_BASE_RADIUS}" \
    --stage2_base_min_candidates "${STAGE2_BASE_MIN_CANDIDATES}" \
    --mesh_eval_samples "${MESH_EVAL_SAMPLES}" \
    --stage2_sparse_filter "${STAGE2_SPARSE_FILTER}" \
    --filter_prior_radius "${FILTER_PRIOR_RADIUS}" \
    --filter_min_component_size "${FILTER_MIN_COMPONENT_SIZE}" \
    --filter_min_support_views "${FILTER_MIN_SUPPORT_VIEWS}" \
    --filter_min_support_ratio "${FILTER_MIN_SUPPORT_RATIO}" \
    --visual_hull_min_visible_views "${VISUAL_HULL_MIN_VISIBLE_VIEWS}" \
    --visual_hull_min_support_ratio "${VISUAL_HULL_MIN_SUPPORT_RATIO}" \
    --filter_min_coords "${FILTER_MIN_COORDS}" \
    "${filter_extra[@]}"
fi

echo "[ar_session_smoke] prepare_report=${PREPARE_DIR}/prepare_report.json"
echo "[ar_session_smoke] manifest=${MANIFEST}"
echo "[ar_session_smoke] build_report=${BUILD_DIR}/build_report.json"
echo "[ar_session_smoke] mesh_report=${EVAL_DIR}/report.json"
