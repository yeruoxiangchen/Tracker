#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

GPU="${GPU:-4}"
MODE="${MODE:-smoke}"
PY="${PY:-/home/zjr/anaconda3/envs/reconviagen/bin/python}"

RUN_BUILD="${RUN_BUILD:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

POINT_RUN_ROOT="${POINT_RUN_ROOT:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_rank_w0005_ws05_s200_seed42}"
WEIGHTS="${WEIGHTS:-microsoft/TRELLIS-image-large}"

MASK_DILATE64="${MASK_DILATE64:-0}"
MASK_DILATE16="${MASK_DILATE16:-0}"
TOPK_SPECS="${TOPK_SPECS:-4096,8192,target_unique}"
THRESHOLD="${THRESHOLD:-0.0}"

UNKNOWN_FLOW_LOSS_WEIGHT="${UNKNOWN_FLOW_LOSS_WEIGHT:-1.0}"
KNOWN_FLOW_LOSS_WEIGHT="${KNOWN_FLOW_LOSS_WEIGHT:-0.25}"
UNKNOWN_X0_LOSS_WEIGHT="${UNKNOWN_X0_LOSS_WEIGHT:-0.25}"
KNOWN_X0_LOSS_WEIGHT="${KNOWN_X0_LOSS_WEIGHT:-0.10}"
CFG_DROP_PROB="${CFG_DROP_PROB:-0.05}"
LR="${LR:-1e-4}"
TRAIN_SEED="${TRAIN_SEED:-42}"
EVAL_SEED="${EVAL_SEED:-123}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
ACCUM_BATCHES="${ACCUM_BATCHES:-1}"
SAVE_EVERY="${SAVE_EVERY:-100}"
EVAL_STEPS="${EVAL_STEPS:-12}"
GUIDANCE_STRENGTH="${GUIDANCE_STRENGTH:-1.0}"
KNOWN_LATENT_CLAMP_STRENGTH="${KNOWN_LATENT_CLAMP_STRENGTH:-1.0}"
KNOWN_CLAMP_START_T="${KNOWN_CLAMP_START_T:-0.5}"
CLAMP_INITIAL_NOISE="${CLAMP_INITIAL_NOISE:-0}"
COND_USE_FULL_Q_VIS="${COND_USE_FULL_Q_VIS:-0}"
USE_IMAGE_COND="${USE_IMAGE_COND:-0}"
IMAGE_MAX_VIEWS="${IMAGE_MAX_VIEWS:-4}"
IMAGE_FRAME_SELECT="${IMAGE_FRAME_SELECT:-uniform}"
IMAGE_SELECT_SEED="${IMAGE_SELECT_SEED:-0}"
IMAGE_COND_AGGREGATION="${IMAGE_COND_AGGREGATION:-mean}"
IMAGE_PREPROCESS="${IMAGE_PREPROCESS:-0}"
IMAGE_USE_SOURCE_MASK="${IMAGE_USE_SOURCE_MASK:-1}"
IMAGE_MASK_CROP_RESOLUTION="${IMAGE_MASK_CROP_RESOLUTION:-518}"
COND_FUSION="${COND_FUSION:-concat}"

case "${MODE}" in
  smoke)
    RUN_NAME="${RUN_NAME:-latent_inpaint_flow_smoke}"
    TRAIN_LATENT_MODE="${TRAIN_LATENT_MODE:-smoke}"
    VAL_LATENT_MODE="${VAL_LATENT_MODE:-smoke}"
    TRAIN_INDICES="${TRAIN_INDICES:-all}"
    EVAL_INDICES="${EVAL_INDICES:-0}"
    MAX_STEPS="${MAX_STEPS:-20}"
    ;;
  s200)
    RUN_NAME="${RUN_NAME:-latent_inpaint_flow_s200}"
    TRAIN_LATENT_MODE="${TRAIN_LATENT_MODE:-train64}"
    VAL_LATENT_MODE="${VAL_LATENT_MODE:-val32}"
    TRAIN_INDICES="${TRAIN_INDICES:-all}"
    EVAL_INDICES="${EVAL_INDICES:-0-31}"
    MAX_STEPS="${MAX_STEPS:-200}"
    ;;
  *)
    echo "Unsupported MODE=${MODE}. Use smoke or s200." >&2
    exit 2
    ;;
esac

TRAIN_LATENT_ROOT="${TRAIN_LATENT_ROOT:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint/latent_inpaint_${TRAIN_LATENT_MODE}}"
VAL_LATENT_ROOT="${VAL_LATENT_ROOT:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint/latent_inpaint_${VAL_LATENT_MODE}}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${TRAIN_LATENT_ROOT}/manifest.json}"
VAL_MANIFEST="${VAL_MANIFEST:-${VAL_LATENT_ROOT}/manifest.json}"
RUN_ROOT="${RUN_ROOT:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint_flow/${RUN_NAME}}"
CKPT_DIR="${CKPT_DIR:-${RUN_ROOT}/checkpoints}"
EVAL_DIR="${EVAL_DIR:-${RUN_ROOT}/eval}"
CHECKPOINT="${CHECKPOINT:-${CKPT_DIR}/last.ckpt}"

echo "[latent_inpaint_flow] mode=${MODE} run=${RUN_NAME}"
echo "[latent_inpaint_flow] train_manifest=${TRAIN_MANIFEST}"
echo "[latent_inpaint_flow] val_manifest=${VAL_MANIFEST}"
echo "[latent_inpaint_flow] mask_dilate64=${MASK_DILATE64} mask_dilate16=${MASK_DILATE16}"
echo "[latent_inpaint_flow] image_cond=${USE_IMAGE_COND} views=${IMAGE_MAX_VIEWS} aggregation=${IMAGE_COND_AGGREGATION} fusion=${COND_FUSION} use_mask=${IMAGE_USE_SOURCE_MASK}"
echo "[latent_inpaint_flow] run_build=${RUN_BUILD} run_train=${RUN_TRAIN} run_eval=${RUN_EVAL}"

if [[ "${RUN_BUILD}" == "1" ]]; then
  GPU="${GPU}" \
  MODE="${TRAIN_LATENT_MODE}" \
  RUN_NAME="latent_inpaint_${TRAIN_LATENT_MODE}" \
  POINT_RUN_ROOT="${POINT_RUN_ROOT}" \
  WEIGHTS="${WEIGHTS}" \
  bash trellis_point_prior_mv/scripts/run_build_latent_inpaint_dataset.sh

  GPU="${GPU}" \
  MODE="${VAL_LATENT_MODE}" \
  RUN_NAME="latent_inpaint_${VAL_LATENT_MODE}" \
  POINT_RUN_ROOT="${POINT_RUN_ROOT}" \
  WEIGHTS="${WEIGHTS}" \
  bash trellis_point_prior_mv/scripts/run_build_latent_inpaint_dataset.sh
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  mkdir -p "${CKPT_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn \
  SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib \
  NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PY}" -u trellis_point_prior_mv/train_latent_inpaint_flow.py \
    --manifest "${TRAIN_MANIFEST}" \
    --weights "${WEIGHTS}" \
    --save_dir "${CKPT_DIR}" \
    --indices "${TRAIN_INDICES}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --max_steps "${MAX_STEPS}" \
    --accum_batches "${ACCUM_BATCHES}" \
    --ckpt_every_n_steps "${SAVE_EVERY}" \
    --lr "${LR}" \
    --seed "${TRAIN_SEED}" \
    --cfg_drop_prob "${CFG_DROP_PROB}" \
    --mask_dilate64 "${MASK_DILATE64}" \
    --mask_dilate16 "${MASK_DILATE16}" \
    --unknown_flow_loss_weight "${UNKNOWN_FLOW_LOSS_WEIGHT}" \
    --known_flow_loss_weight "${KNOWN_FLOW_LOSS_WEIGHT}" \
    --unknown_x0_loss_weight "${UNKNOWN_X0_LOSS_WEIGHT}" \
    --known_x0_loss_weight "${KNOWN_X0_LOSS_WEIGHT}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --image_max_views "${IMAGE_MAX_VIEWS}" \
    --image_frame_select "${IMAGE_FRAME_SELECT}" \
    --image_select_seed "${IMAGE_SELECT_SEED}" \
    --image_cond_aggregation "${IMAGE_COND_AGGREGATION}" \
    --image_mask_crop_resolution "${IMAGE_MASK_CROP_RESOLUTION}" \
    --cond_fusion "${COND_FUSION}" \
    $(if [[ "${COND_USE_FULL_Q_VIS}" == "1" ]]; then echo "--cond_use_full_q_vis"; fi) \
    $(if [[ "${USE_IMAGE_COND}" == "1" ]]; then echo "--use_image_cond"; fi) \
    $(if [[ "${IMAGE_USE_SOURCE_MASK}" == "1" ]]; then echo "--image_use_source_mask"; else echo "--no-image_use_source_mask"; fi) \
    $(if [[ "${IMAGE_PREPROCESS}" == "1" ]]; then echo "--image_preprocess"; fi)
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  mkdir -p "${EVAL_DIR}"
  CLAMP_FLAG="--no-clamp_initial_noise"
  if [[ "${CLAMP_INITIAL_NOISE}" == "1" ]]; then
    CLAMP_FLAG="--clamp_initial_noise"
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
  "${PY}" -u trellis_point_prior_mv/eval_latent_inpaint_flow.py \
    --manifest "${VAL_MANIFEST}" \
    --checkpoint "${CHECKPOINT}" \
    --output_dir "${EVAL_DIR}" \
    --weights "${WEIGHTS}" \
    --indices "${EVAL_INDICES}" \
    --mask_dilate64 "${MASK_DILATE64}" \
    --mask_dilate16 "${MASK_DILATE16}" \
    --topk "${TOPK_SPECS}" \
    --threshold "${THRESHOLD}" \
    --steps "${EVAL_STEPS}" \
    --guidance_strength "${GUIDANCE_STRENGTH}" \
    --known_latent_clamp_strength "${KNOWN_LATENT_CLAMP_STRENGTH}" \
    --known_clamp_start_t "${KNOWN_CLAMP_START_T}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --image_max_views "${IMAGE_MAX_VIEWS}" \
    --image_frame_select "${IMAGE_FRAME_SELECT}" \
    --image_select_seed "${IMAGE_SELECT_SEED}" \
    --image_cond_aggregation "${IMAGE_COND_AGGREGATION}" \
    --image_mask_crop_resolution "${IMAGE_MASK_CROP_RESOLUTION}" \
    --cond_fusion "${COND_FUSION}" \
    ${CLAMP_FLAG} \
    $(if [[ "${COND_USE_FULL_Q_VIS}" == "1" ]]; then echo "--cond_use_full_q_vis"; fi) \
    $(if [[ "${USE_IMAGE_COND}" == "1" ]]; then echo "--use_image_cond"; fi) \
    $(if [[ "${IMAGE_USE_SOURCE_MASK}" == "1" ]]; then echo "--image_use_source_mask"; else echo "--no-image_use_source_mask"; fi) \
    $(if [[ "${IMAGE_PREPROCESS}" == "1" ]]; then echo "--image_preprocess"; fi)
  echo "[latent_inpaint_flow] report=${EVAL_DIR}/report.json"
fi
