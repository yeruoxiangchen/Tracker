#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

GPU="${GPU:-4}"
MODE="${MODE:-smoke}"
PY="${PY:-/home/zjr/anaconda3/envs/reconviagen/bin/python}"

POINT_RUN_ROOT="${POINT_RUN_ROOT:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_rank_w0005_ws05_s200_seed42}"
WEIGHTS="${WEIGHTS:-microsoft/TRELLIS-image-large}"

case "${MODE}" in
  smoke)
    MANIFEST="${MANIFEST:-${POINT_RUN_ROOT}/data/val/manifest.json}"
    INDICES="${INDICES:-0}"
    RUN_NAME="${RUN_NAME:-latent_inpaint_smoke}"
    ;;
  val8)
    MANIFEST="${MANIFEST:-${POINT_RUN_ROOT}/data/val/manifest.json}"
    INDICES="${INDICES:-0-7}"
    RUN_NAME="${RUN_NAME:-latent_inpaint_val8}"
    ;;
  val32)
    MANIFEST="${MANIFEST:-${POINT_RUN_ROOT}/data/val/manifest.json}"
    INDICES="${INDICES:-0-31}"
    RUN_NAME="${RUN_NAME:-latent_inpaint_val32}"
    ;;
  val64)
    MANIFEST="${MANIFEST:-${POINT_RUN_ROOT}/data/val/manifest.json}"
    INDICES="${INDICES:-0-63}"
    RUN_NAME="${RUN_NAME:-latent_inpaint_val64}"
    ;;
  val128)
    if [[ -z "${MANIFEST:-}" ]]; then
      if [[ -f "${POINT_RUN_ROOT}/data/val128/manifest.json" ]]; then
        MANIFEST="${POINT_RUN_ROOT}/data/val128/manifest.json"
      else
        MANIFEST="${POINT_RUN_ROOT}/data/val/manifest.json"
      fi
    fi
    INDICES="${INDICES:-all}"
    RUN_NAME="${RUN_NAME:-latent_inpaint_val128}"
    ;;
  train64)
    MANIFEST="${MANIFEST:-${POINT_RUN_ROOT}/data/train/manifest.json}"
    INDICES="${INDICES:-0-63}"
    RUN_NAME="${RUN_NAME:-latent_inpaint_train64}"
    ;;
  train512)
    MANIFEST="${MANIFEST:-${POINT_RUN_ROOT}/data/train/manifest.json}"
    INDICES="${INDICES:-0-511}"
    RUN_NAME="${RUN_NAME:-latent_inpaint_train512}"
    ;;
  train1488)
    MANIFEST="${MANIFEST:-${POINT_RUN_ROOT}/data/train/manifest.json}"
    INDICES="${INDICES:-all}"
    RUN_NAME="${RUN_NAME:-latent_inpaint_train1488}"
    ;;
  *)
    echo "Unsupported MODE=${MODE}. Use smoke, val8, val32, val64, val128, train64, train512, or train1488." >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint/${RUN_NAME}}"

echo "[build_latent_inpaint] mode=${MODE} indices=${INDICES}"
echo "[build_latent_inpaint] manifest=${MANIFEST}"
echo "[build_latent_inpaint] output=${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${PY}" -u trellis_point_prior_mv/build_latent_inpaint_dataset.py \
  --manifest "${MANIFEST}" \
  --output_dir "${OUTPUT_DIR}" \
  --weights "${WEIGHTS}" \
  --indices "${INDICES}"

echo "[build_latent_inpaint] manifest=${OUTPUT_DIR}/manifest.json"
