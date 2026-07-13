#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

GPU="${GPU:-4}"
MODE="${MODE:-smoke}"
PY="${PY:-/home/zjr/anaconda3/envs/reconviagen/bin/python}"

POINT_RUN_ROOT="${POINT_RUN_ROOT:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_rank_w0005_ws05_s200_seed42}"
MANIFEST="${MANIFEST:-${POINT_RUN_ROOT}/data/val/manifest.json}"
WEIGHTS="${WEIGHTS:-microsoft/TRELLIS-image-large}"
TOPK_SPECS="${TOPK_SPECS:-4096,8192,target_unique}"
THRESHOLD="${THRESHOLD:-0.0}"
PRIOR_RADIUS="${PRIOR_RADIUS:-4.0}"

case "${MODE}" in
  smoke)
    INDICES="${INDICES:-0}"
    RUN_NAME="${RUN_NAME:-sparse_vae_sanity_smoke}"
    ;;
  val8)
    INDICES="${INDICES:-0-7}"
    RUN_NAME="${RUN_NAME:-sparse_vae_sanity_val8}"
    ;;
  val32)
    INDICES="${INDICES:-0-31}"
    RUN_NAME="${RUN_NAME:-sparse_vae_sanity_val32}"
    ;;
  *)
    echo "Unsupported MODE=${MODE}. Use smoke, val8, or val32." >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/sparse_vae_sanity/${RUN_NAME}}"

echo "[sparse_vae_sanity] mode=${MODE} indices=${INDICES}"
echo "[sparse_vae_sanity] manifest=${MANIFEST}"
echo "[sparse_vae_sanity] topk=${TOPK_SPECS} threshold=${THRESHOLD}"
echo "[sparse_vae_sanity] output=${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${PY}" -u trellis_point_prior_mv/eval_sparse_vae_sanity.py \
  --manifest "${MANIFEST}" \
  --output_dir "${OUTPUT_DIR}" \
  --weights "${WEIGHTS}" \
  --indices "${INDICES}" \
  --topk "${TOPK_SPECS}" \
  --threshold "${THRESHOLD}" \
  --prior_radius "${PRIOR_RADIUS}"

echo "[sparse_vae_sanity] report=${OUTPUT_DIR}/report.json"
