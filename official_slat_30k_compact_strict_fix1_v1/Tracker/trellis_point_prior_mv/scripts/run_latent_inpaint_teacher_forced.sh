#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

GPU="${GPU:-4}"
PY="${PY:-/home/zjr/anaconda3/envs/reconviagen/bin/python}"

RUN_NAME="${RUN_NAME:-latent_inpaint_flow_d64_0_s200_teacher_forced}"
LATENT_RUN_ROOT="${LATENT_RUN_ROOT:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint/latent_inpaint_val32}"
MANIFEST="${MANIFEST:-${LATENT_RUN_ROOT}/manifest.json}"
POINT_RUN_ROOT="${POINT_RUN_ROOT:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint_flow/latent_inpaint_flow_d64_0_s200}"
CHECKPOINT="${CHECKPOINT:-${POINT_RUN_ROOT}/checkpoints/last.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint_flow/${RUN_NAME}}"
WEIGHTS="${WEIGHTS:-microsoft/TRELLIS-image-large}"

INDICES="${INDICES:-0-31}"
MASK_DILATE64="${MASK_DILATE64:-0}"
MASK_DILATE16="${MASK_DILATE16:-0}"
T_VALUES="${T_VALUES:-0.25,0.5,0.75}"
TOPK_SPECS="${TOPK_SPECS:-4096,8192,target_unique}"
THRESHOLD="${THRESHOLD:-0.0}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
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
SEED="${SEED:-123}"

echo "[latent_teacher_forced] run=${RUN_NAME}"
echo "[latent_teacher_forced] manifest=${MANIFEST}"
echo "[latent_teacher_forced] checkpoint=${CHECKPOINT}"
echo "[latent_teacher_forced] output=${OUTPUT_DIR}"
echo "[latent_teacher_forced] t_values=${T_VALUES} mask_dilate64=${MASK_DILATE64} mask_dilate16=${MASK_DILATE16}"
echo "[latent_teacher_forced] image_cond=${USE_IMAGE_COND} views=${IMAGE_MAX_VIEWS} aggregation=${IMAGE_COND_AGGREGATION} fusion=${COND_FUSION} use_mask=${IMAGE_USE_SOURCE_MASK}"

mkdir -p "${OUTPUT_DIR}"
CUDA_VISIBLE_DEVICES="${GPU}" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${PY}" -u trellis_point_prior_mv/eval_latent_inpaint_teacher_forced.py \
  --manifest "${MANIFEST}" \
  --checkpoint "${CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --weights "${WEIGHTS}" \
  --indices "${INDICES}" \
  --mask_dilate64 "${MASK_DILATE64}" \
  --mask_dilate16 "${MASK_DILATE16}" \
  --t_values "${T_VALUES}" \
  --topk "${TOPK_SPECS}" \
  --threshold "${THRESHOLD}" \
  --seed "${SEED}" \
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

echo "[latent_teacher_forced] report=${OUTPUT_DIR}/report.json"
