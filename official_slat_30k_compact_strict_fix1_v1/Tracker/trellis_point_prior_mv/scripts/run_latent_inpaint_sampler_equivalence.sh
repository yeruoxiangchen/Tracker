#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

GPU="${GPU:-4}"
PY="${PY:-/home/zjr/anaconda3/envs/reconviagen/bin/python}"
DIAG="${DIAG:-stock_imageonly_native}"
WEIGHTS="${WEIGHTS:-microsoft/TRELLIS-image-large}"

LATENT_RUN_ROOT="${LATENT_RUN_ROOT:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint/latent_inpaint_smoke}"
MANIFEST="${MANIFEST:-${LATENT_RUN_ROOT}/manifest.json}"
INDICES="${INDICES:-0}"
MASK_DILATE64="${MASK_DILATE64:-0}"
MASK_DILATE16="${MASK_DILATE16:-0}"
TOPK_SPECS="${TOPK_SPECS:-4096,8192,target_unique}"
THRESHOLD="${THRESHOLD:-0.0}"
IMAGE_MAX_VIEWS="${IMAGE_MAX_VIEWS:-4}"
IMAGE_FRAME_SELECT="${IMAGE_FRAME_SELECT:-uniform}"
IMAGE_COND_AGGREGATION="${IMAGE_COND_AGGREGATION:-mean}"
IMAGE_USE_SOURCE_MASK="${IMAGE_USE_SOURCE_MASK:-1}"
IMAGE_MASK_CROP_RESOLUTION="${IMAGE_MASK_CROP_RESOLUTION:-518}"
COND_FUSION="${COND_FUSION:-image_only}"
EVAL_STEPS="${EVAL_STEPS:-0}"
USE_NATIVE_SS_CFG="${USE_NATIVE_SS_CFG:-1}"
ADD_NATIVE_SAMPLER_PRED="${ADD_NATIVE_SAMPLER_PRED:-1}"
USER_KNOWN_LATENT_CLAMP_STRENGTH="${KNOWN_LATENT_CLAMP_STRENGTH:-}"
USER_CLAMP_INITIAL_NOISE="${CLAMP_INITIAL_NOISE:-}"
KNOWN_LATENT_CLAMP_STRENGTH="${KNOWN_LATENT_CLAMP_STRENGTH:-0}"
KNOWN_CLAMP_START_T="${KNOWN_CLAMP_START_T:-1.0}"
CLAMP_INITIAL_NOISE="${CLAMP_INITIAL_NOISE:-0}"
GUIDANCE_STRENGTH="${GUIDANCE_STRENGTH:-}"
GUIDANCE_RESCALE="${GUIDANCE_RESCALE:-}"
CHECKPOINT="${CHECKPOINT:-}"
NO_LORA="${NO_LORA:-1}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
SEED="${SEED:-123}"

case "${DIAG}" in
  stock_imageonly_native)
    RUN_NAME="${RUN_NAME:-latent_inpaint_sampler_stock_imageonly_native}"
    NO_LORA=1
    CHECKPOINT=""
    COND_FUSION=image_only
    USE_NATIVE_SS_CFG=1
    KNOWN_LATENT_CLAMP_STRENGTH=0
    CLAMP_INITIAL_NOISE=0
    ;;
  stock_imageonly_cfg)
    RUN_NAME="${RUN_NAME:-latent_inpaint_sampler_stock_imageonly_cfg${GUIDANCE_STRENGTH:-manual}}"
    NO_LORA=1
    CHECKPOINT=""
    COND_FUSION=image_only
    USE_NATIVE_SS_CFG=0
    KNOWN_LATENT_CLAMP_STRENGTH=0
    CLAMP_INITIAL_NOISE=0
    ;;
  lora_imageonly_native_noclamp)
    RUN_NAME="${RUN_NAME:-latent_inpaint_sampler_lora_imageonly_native_noclamp}"
    CHECKPOINT="${CHECKPOINT:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint_flow/latent_inpaint_flow_img4_d64_0_imageonly_overfit1_s500/checkpoints/last.ckpt}"
    NO_LORA=0
    COND_FUSION=image_only
    USE_NATIVE_SS_CFG=1
    KNOWN_LATENT_CLAMP_STRENGTH=0
    CLAMP_INITIAL_NOISE=0
    ;;
  lora_imageonly_native_clamp)
    RUN_NAME="${RUN_NAME:-latent_inpaint_sampler_lora_imageonly_native_clamp}"
    CHECKPOINT="${CHECKPOINT:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint_flow/latent_inpaint_flow_img4_d64_0_imageonly_overfit1_s500/checkpoints/last.ckpt}"
    NO_LORA=0
    COND_FUSION=image_only
    USE_NATIVE_SS_CFG=1
    KNOWN_LATENT_CLAMP_STRENGTH="${USER_KNOWN_LATENT_CLAMP_STRENGTH:-1.0}"
    CLAMP_INITIAL_NOISE="${USER_CLAMP_INITIAL_NOISE:-1}"
    ;;
  lora_concat_native_noclamp)
    RUN_NAME="${RUN_NAME:-latent_inpaint_sampler_lora_concat_native_noclamp}"
    CHECKPOINT="${CHECKPOINT:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint_flow/latent_inpaint_flow_img4_d64_0_overfit1_s500/checkpoints/last.ckpt}"
    NO_LORA=0
    COND_FUSION=concat
    USE_NATIVE_SS_CFG=1
    KNOWN_LATENT_CLAMP_STRENGTH=0
    CLAMP_INITIAL_NOISE=0
    ;;
  *)
    echo "Unsupported DIAG=${DIAG}" >&2
    echo "Use: stock_imageonly_native | stock_imageonly_cfg | lora_imageonly_native_noclamp | lora_imageonly_native_clamp | lora_concat_native_noclamp" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint_flow/${RUN_NAME}}"
mkdir -p "${OUTPUT_DIR}"

ARGS=(
  -u trellis_point_prior_mv/eval_latent_inpaint_flow.py
  --manifest "${MANIFEST}"
  --output_dir "${OUTPUT_DIR}"
  --weights "${WEIGHTS}"
  --indices "${INDICES}"
  --mask_dilate64 "${MASK_DILATE64}"
  --mask_dilate16 "${MASK_DILATE16}"
  --topk "${TOPK_SPECS}"
  --threshold "${THRESHOLD}"
  --steps "${EVAL_STEPS}"
  --known_latent_clamp_strength "${KNOWN_LATENT_CLAMP_STRENGTH}"
  --known_clamp_start_t "${KNOWN_CLAMP_START_T}"
  --use_image_cond
  --image_max_views "${IMAGE_MAX_VIEWS}"
  --image_frame_select "${IMAGE_FRAME_SELECT}"
  --image_cond_aggregation "${IMAGE_COND_AGGREGATION}"
  --image_mask_crop_resolution "${IMAGE_MASK_CROP_RESOLUTION}"
  --cond_fusion "${COND_FUSION}"
  --seed "${SEED}"
  --lora_rank "${LORA_RANK}"
  --lora_alpha "${LORA_ALPHA}"
)

if [[ -n "${CHECKPOINT}" ]]; then
  ARGS+=(--checkpoint "${CHECKPOINT}")
fi
if [[ "${NO_LORA}" == "1" ]]; then
  ARGS+=(--no_lora)
fi
if [[ "${USE_NATIVE_SS_CFG}" == "1" ]]; then
  ARGS+=(--use_native_ss_cfg)
fi
if [[ "${ADD_NATIVE_SAMPLER_PRED}" == "1" ]]; then
  ARGS+=(--add_native_sampler_pred)
fi
if [[ "${IMAGE_USE_SOURCE_MASK}" == "1" ]]; then
  ARGS+=(--image_use_source_mask)
else
  ARGS+=(--no-image_use_source_mask)
fi
if [[ "${CLAMP_INITIAL_NOISE}" == "1" ]]; then
  ARGS+=(--clamp_initial_noise)
else
  ARGS+=(--no-clamp_initial_noise)
fi
if [[ -n "${GUIDANCE_STRENGTH}" ]]; then
  ARGS+=(--guidance_strength "${GUIDANCE_STRENGTH}")
fi
if [[ -n "${GUIDANCE_RESCALE}" ]]; then
  ARGS+=(--guidance_rescale "${GUIDANCE_RESCALE}")
fi

echo "[latent_sampler_equiv] diag=${DIAG}"
echo "[latent_sampler_equiv] output=${OUTPUT_DIR}"
echo "[latent_sampler_equiv] no_lora=${NO_LORA} checkpoint=${CHECKPOINT:-none} fusion=${COND_FUSION} native_cfg=${USE_NATIVE_SS_CFG} steps=${EVAL_STEPS} use_mask=${IMAGE_USE_SOURCE_MASK}"

CUDA_VISIBLE_DEVICES="${GPU}" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${PY}" "${ARGS[@]}"

echo "[latent_sampler_equiv] report=${OUTPUT_DIR}/report.json"
