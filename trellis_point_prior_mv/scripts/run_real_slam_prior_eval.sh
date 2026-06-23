#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zjr/Tracker}"
PY="${PY:-/home/zjr/anaconda3/envs/reconviagen/bin/python}"
GPU="${GPU:-4}"
RUN_NAME="${RUN_NAME:-real_slam_prior_goodmesh_recon2026_smoke}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/trellis_point_prior_mv/outputs/real_slam_prior}"

DATASETS_DEFAULT=(
  "${ROOT}/CoarseModel/datasets/GOOD_MESH_TEST"
  "${ROOT}/CoarseModel/datasets/reconviagen_20260520_021556"
  "${ROOT}/CoarseModel/datasets/reconviagen_20260617_073549"
  "${ROOT}/CoarseModel/datasets/reconviagen_20260617_075506"
)

DATASETS_RAW="${DATASETS:-}"
if [[ -n "${DATASETS_RAW}" ]]; then
  IFS=":" read -r -a DATASET_ARGS <<< "${DATASETS_RAW}"
else
  DATASET_ARGS=("${DATASETS_DEFAULT[@]}")
fi

PRIOR_SOURCE="${PRIOR_SOURCE:-model_surface}"
ALLOW_MODEL_FALLBACK="${ALLOW_MODEL_FALLBACK:-0}"
NORMALIZATION_SOURCE="${NORMALIZATION_SOURCE:-auto}"
SPARSE_SUBDIR="${SPARSE_SUBDIR:-sparse/0}"
RUN_TRIANGULATE="${RUN_TRIANGULATE:-0}"
TRI_INPUT_SPARSE_SUBDIR="${TRI_INPUT_SPARSE_SUBDIR:-sparse/0}"
TRIANGULATE_OVERWRITE="${TRIANGULATE_OVERWRITE:-1}"
MAX_FRAMES="${MAX_FRAMES:-18}"
POINT_COUNT="${POINT_COUNT:-1500}"
MIN_PRIOR_POINTS="${MIN_PRIOR_POINTS:-200}"
TOPK_SPECS="${TOPK_SPECS:-12000}"
MODES="${MODES:-stage2_correct}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-${ROOT}/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_rank_w0005_ws05_s200_seed42/checkpoints/last.ckpt}"
WEIGHTS="${WEIGHTS:-microsoft/TRELLIS-image-large}"
SS_STEPS="${SS_STEPS:-12}"
SLAT_STEPS="${SLAT_STEPS:-12}"
MESH_EVAL_SAMPLES="${MESH_EVAL_SAMPLES:-6000}"
RUN_BUILD="${RUN_BUILD:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

RUN_DIR="${OUT_ROOT}/${RUN_NAME}"
MANIFEST="${MANIFEST:-${RUN_DIR}/manifest/manifest.json}"
BUILD_DIR="$(dirname "${MANIFEST}")"
EVAL_DIR="${OUTPUT_DIR:-${RUN_DIR}/mesh_eval}"

cd "${ROOT}"
mkdir -p "${RUN_DIR}"

if [[ "${RUN_TRIANGULATE}" == "1" ]]; then
  if [[ "${SPARSE_SUBDIR}" == "${TRI_INPUT_SPARSE_SUBDIR}" ]]; then
    echo "[real_slam_prior_eval][ERROR] RUN_TRIANGULATE would overwrite input sparse dir: ${SPARSE_SUBDIR}" >&2
    echo "Set SPARSE_SUBDIR to something like sparse_slam_eval/0" >&2
    exit 2
  fi
  TRI_EXTRA_ARGS=()
  if [[ "${TRIANGULATE_OVERWRITE}" == "1" ]]; then
    TRI_EXTRA_ARGS+=(--overwrite)
  fi
  if [[ "${TRI_ALLOW_PAIR_OUTSIDE_MASK:-0}" == "1" ]]; then
    TRI_EXTRA_ARGS+=(--allow_pair_outside_mask)
  fi
  echo "[real_slam_prior_eval] triangulate SLAM-like points -> ${SPARSE_SUBDIR}"
  "${PY}" -u trellis_point_prior_mv/triangulate_slam_like_points.py \
    --datasets "${DATASET_ARGS[@]}" \
    --input_sparse_subdir "${TRI_INPUT_SPARSE_SUBDIR}" \
    --output_sparse_subdir "${SPARSE_SUBDIR}" \
    --max_frames "${MAX_FRAMES}" \
    --max_features "${TRI_MAX_FEATURES:-4096}" \
    --feature_mask_mode "${TRI_FEATURE_MASK_MODE:-none}" \
    --matcher "${TRI_MATCHER:-exhaustive}" \
    --max_pair_gap "${TRI_MAX_PAIR_GAP:-0}" \
    --ratio_test "${TRI_RATIO_TEST:-0.75}" \
    --min_pair_matches "${TRI_MIN_PAIR_MATCHES:-12}" \
    --max_reproj_error "${TRI_MAX_REPROJ_ERROR:-4.0}" \
    --min_triangulation_angle_deg "${TRI_MIN_ANGLE_DEG:-1.0}" \
    --min_support_views "${TRI_MIN_SUPPORT_VIEWS:-2.0}" \
    --min_support_ratio "${TRI_MIN_SUPPORT_RATIO:-0.10}" \
    --merge_voxel_size "${TRI_MERGE_VOXEL_SIZE:-0.002}" \
    --min_output_points "${TRI_MIN_OUTPUT_POINTS:-50}" \
    --max_output_points "${TRI_MAX_OUTPUT_POINTS:-50000}" \
    --output_report "${RUN_DIR}/slam_like_points_report.json" \
    "${TRI_EXTRA_ARGS[@]}"
fi

if [[ "${RUN_BUILD}" == "1" ]]; then
  echo "[real_slam_prior_eval] build manifest -> ${BUILD_DIR}"
  BUILD_EXTRA_ARGS=()
  if [[ "${ALLOW_MODEL_FALLBACK}" == "1" ]]; then
    BUILD_EXTRA_ARGS+=(--allow_model_fallback)
  fi
  "${PY}" -u trellis_point_prior_mv/build_real_slam_prior_manifest.py \
    --datasets "${DATASET_ARGS[@]}" \
    --output_dir "${BUILD_DIR}" \
    --prior_source "${PRIOR_SOURCE}" \
    --sparse_subdir "${SPARSE_SUBDIR}" \
    --normalization_source "${NORMALIZATION_SOURCE}" \
    --max_frames "${MAX_FRAMES}" \
    --point_count "${POINT_COUNT}" \
    --min_prior_points "${MIN_PRIOR_POINTS}" \
    --seed 42 \
    "${BUILD_EXTRA_ARGS[@]}"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  echo "[real_slam_prior_eval] mesh eval -> ${EVAL_DIR}"
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
    --indices all \
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
    --mesh_eval_samples "${MESH_EVAL_SAMPLES}"
fi

echo "[real_slam_prior_eval] manifest=${MANIFEST}"
echo "[real_slam_prior_eval] report=${EVAL_DIR}/report.json"
