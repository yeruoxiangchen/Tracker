#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

GPU="${GPU:-4}"
MODE="${MODE:-smoke}"
PY="${PY:-/home/zjr/anaconda3/envs/reconviagen/bin/python}"

LATENT_RUN_ROOT="${LATENT_RUN_ROOT:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint/latent_inpaint_smoke}"
MANIFEST="${MANIFEST:-${LATENT_RUN_ROOT}/manifest.json}"
WEIGHTS="${WEIGHTS:-microsoft/TRELLIS-image-large}"
MASK_DILATE64="${MASK_DILATE64:-0,1,2}"
MASK_DILATE16="${MASK_DILATE16:-0,1}"
TOPK_SPECS="${TOPK_SPECS:-4096,8192,target_unique}"
THRESHOLD="${THRESHOLD:-0.0}"

case "${MODE}" in
  smoke)
    INDICES="${INDICES:-0}"
    RUN_NAME="${RUN_NAME:-latent_splice_sanity_smoke}"
    ;;
  val8)
    INDICES="${INDICES:-0-7}"
    RUN_NAME="${RUN_NAME:-latent_splice_sanity_val8}"
    ;;
  val32)
    INDICES="${INDICES:-0-31}"
    RUN_NAME="${RUN_NAME:-latent_splice_sanity_val32}"
    ;;
  *)
    echo "Unsupported MODE=${MODE}. Use smoke, val8, or val32." >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_splice_sanity/${RUN_NAME}}"

echo "[latent_splice_sanity] mode=${MODE} indices=${INDICES}"
echo "[latent_splice_sanity] manifest=${MANIFEST}"
echo "[latent_splice_sanity] mask_dilate64=${MASK_DILATE64} mask_dilate16=${MASK_DILATE16}"
echo "[latent_splice_sanity] topk=${TOPK_SPECS} threshold=${THRESHOLD}"
echo "[latent_splice_sanity] output=${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${PY}" -u trellis_point_prior_mv/eval_latent_splice_sanity.py \
  --manifest "${MANIFEST}" \
  --output_dir "${OUTPUT_DIR}" \
  --weights "${WEIGHTS}" \
  --indices "${INDICES}" \
  --mask_dilate64 "${MASK_DILATE64}" \
  --mask_dilate16 "${MASK_DILATE16}" \
  --topk "${TOPK_SPECS}" \
  --threshold "${THRESHOLD}"

echo "[latent_splice_sanity] report=${OUTPUT_DIR}/report.json"
