#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

GPU="${GPU:-1}"
MODE="${MODE:-smoke}"
PY="${PY:-/home/zjr/anaconda3/envs/reconviagen/bin/python}"

POINT_RUN_ROOT="${POINT_RUN_ROOT:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_strictmask_s200}"
MANIFEST="${MANIFEST:-${POINT_RUN_ROOT}/data/val/manifest.json}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-${POINT_RUN_ROOT}/checkpoints/last.ckpt}"
WEIGHTS="${WEIGHTS:-microsoft/TRELLIS-image-large}"
TOPK_SPECS="${TOPK_SPECS:-r0.35_cap4096,r0.50_cap8192,r0.75_cap12000,r1.00_cap12000,target_unique}"
MODES="${MODES:-stock_sparse,target_sparse,stage2_correct}"

MAX_FRAMES="${MAX_FRAMES:-8}"
SS_STEPS="${SS_STEPS:-12}"
SLAT_STEPS="${SLAT_STEPS:-12}"
MESH_EVAL_SAMPLES="${MESH_EVAL_SAMPLES:-4000}"
KNOWN_CLAMP_START_T="${KNOWN_CLAMP_START_T:-1.0}"
STAGE2_BASE_GUIDANCE="${STAGE2_BASE_GUIDANCE:-none}"
STAGE2_BASE_RADIUS="${STAGE2_BASE_RADIUS:-3.0}"
STAGE2_BASE_MIN_CANDIDATES="${STAGE2_BASE_MIN_CANDIDATES:-512}"
STAGE2_UNION_STOCK="${STAGE2_UNION_STOCK:-0}"
STAGE2_SPARSE_FILTER="${STAGE2_SPARSE_FILTER:-none}"
FILTER_MIN_COMPONENT_SIZE="${FILTER_MIN_COMPONENT_SIZE:-64}"
FILTER_PRIOR_RADIUS="${FILTER_PRIOR_RADIUS:-4.0}"
FILTER_MIN_COORDS="${FILTER_MIN_COORDS:-128}"
FILTER_FALLBACK_UNFILTERED="${FILTER_FALLBACK_UNFILTERED:-0}"

case "${MODE}" in
  smoke)
    INDICES="${INDICES:-0}"
    RUN_NAME="${RUN_NAME:-strict_val0_stage2_relcap_sweep}"
    ;;
  val8)
    INDICES="${INDICES:-0-7}"
    RUN_NAME="${RUN_NAME:-strict_val0_7_stage2_relcap_sweep}"
    ;;
  *)
    echo "Unsupported MODE=${MODE}. Use MODE=smoke or MODE=val8." >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/${RUN_NAME}}"

echo "[mesh_topk_sweep] mode=${MODE} indices=${INDICES}"
echo "[mesh_topk_sweep] modes=${MODES}"
echo "[mesh_topk_sweep] topk_specs=${TOPK_SPECS}"
echo "[mesh_topk_sweep] base_guidance=${STAGE2_BASE_GUIDANCE} radius=${STAGE2_BASE_RADIUS}"
echo "[mesh_topk_sweep] union_stock=${STAGE2_UNION_STOCK}"
echo "[mesh_topk_sweep] sparse_filter=${STAGE2_SPARSE_FILTER} min_component=${FILTER_MIN_COMPONENT_SIZE}"
echo "[mesh_topk_sweep] output=${OUTPUT_DIR}"

filter_extra=()
if [[ "${FILTER_FALLBACK_UNFILTERED}" == "1" ]]; then
  filter_extra+=(--filter_fallback_unfiltered)
fi
if [[ "${STAGE2_UNION_STOCK}" == "1" ]]; then
  filter_extra+=(--stage2_union_stock)
fi

CUDA_VISIBLE_DEVICES="${GPU}" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${PY}" -u trellis_point_prior_mv/eval_mesh_frozen_downstream.py \
  --manifest "${MANIFEST}" \
  --output_dir "${OUTPUT_DIR}" \
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
  --known_clamp_start_t "${KNOWN_CLAMP_START_T}" \
  --known_logit_boost 0.0 \
  --known_conf_power 1.0 \
  --stage2_base_guidance "${STAGE2_BASE_GUIDANCE}" \
  --stage2_base_radius "${STAGE2_BASE_RADIUS}" \
  --stage2_base_min_candidates "${STAGE2_BASE_MIN_CANDIDATES}" \
  --stage2_sparse_filter "${STAGE2_SPARSE_FILTER}" \
  --filter_min_component_size "${FILTER_MIN_COMPONENT_SIZE}" \
  --filter_prior_radius "${FILTER_PRIOR_RADIUS}" \
  --filter_min_coords "${FILTER_MIN_COORDS}" \
  --mesh_eval_samples "${MESH_EVAL_SAMPLES}" \
  "${filter_extra[@]}"

echo "[mesh_topk_sweep] report=${OUTPUT_DIR}/report.json"
