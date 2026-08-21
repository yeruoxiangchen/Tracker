#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

DIAG="${DIAG:-image_only_overfit}"
GPU="${GPU:-4}"

SMOKE_LATENT_ROOT="${SMOKE_LATENT_ROOT:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint/latent_inpaint_smoke}"
TRAIN_LATENT_ROOT="${TRAIN_LATENT_ROOT:-${SMOKE_LATENT_ROOT}}"
VAL_LATENT_ROOT="${VAL_LATENT_ROOT:-${SMOKE_LATENT_ROOT}}"
LATENT_RUN_ROOT="${LATENT_RUN_ROOT:-${VAL_LATENT_ROOT}}"

MASK_DILATE64="${MASK_DILATE64:-0}"
MASK_DILATE16="${MASK_DILATE16:-0}"
IMAGE_MAX_VIEWS="${IMAGE_MAX_VIEWS:-4}"
IMAGE_FRAME_SELECT="${IMAGE_FRAME_SELECT:-uniform}"
IMAGE_USE_SOURCE_MASK="${IMAGE_USE_SOURCE_MASK:-1}"
IMAGE_MASK_CROP_RESOLUTION="${IMAGE_MASK_CROP_RESOLUTION:-518}"
MAX_STEPS="${MAX_STEPS:-500}"
SAVE_EVERY="${SAVE_EVERY:-100}"
EVAL_STEPS="${EVAL_STEPS:-25}"
T_VALUES="${T_VALUES:-0.25,0.5,0.75}"
KNOWN_CLAMP_START_T="${KNOWN_CLAMP_START_T:-1.0}"
CLAMP_INITIAL_NOISE="${CLAMP_INITIAL_NOISE:-1}"
CFG_DROP_PROB="${CFG_DROP_PROB:-0}"

case "${DIAG}" in
  image_only_overfit)
    RUN_NAME="${RUN_NAME:-latent_inpaint_flow_img4_d64_0_imageonly_overfit1_s500}"
    GPU="${GPU}" \
    MODE=smoke \
    RUN_NAME="${RUN_NAME}" \
    RUN_BUILD=0 \
    RUN_TRAIN=1 \
    RUN_EVAL=1 \
    TRAIN_LATENT_ROOT="${TRAIN_LATENT_ROOT}" \
    VAL_LATENT_ROOT="${VAL_LATENT_ROOT}" \
    USE_IMAGE_COND=1 \
    IMAGE_MAX_VIEWS="${IMAGE_MAX_VIEWS}" \
    IMAGE_FRAME_SELECT="${IMAGE_FRAME_SELECT}" \
    IMAGE_USE_SOURCE_MASK="${IMAGE_USE_SOURCE_MASK}" \
    IMAGE_MASK_CROP_RESOLUTION="${IMAGE_MASK_CROP_RESOLUTION}" \
    IMAGE_COND_AGGREGATION=mean \
    COND_FUSION=image_only \
    MASK_DILATE64="${MASK_DILATE64}" \
    MASK_DILATE16="${MASK_DILATE16}" \
    CFG_DROP_PROB="${CFG_DROP_PROB}" \
    KNOWN_CLAMP_START_T="${KNOWN_CLAMP_START_T}" \
    CLAMP_INITIAL_NOISE="${CLAMP_INITIAL_NOISE}" \
    MAX_STEPS="${MAX_STEPS}" \
    SAVE_EVERY="${SAVE_EVERY}" \
    EVAL_STEPS="${EVAL_STEPS}" \
    bash trellis_point_prior_mv/scripts/run_latent_inpaint_flow.sh
    ;;

  concat_teacher_forced)
    POINT_RUN_ROOT="${POINT_RUN_ROOT:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint_flow/latent_inpaint_flow_img4_d64_0_overfit1_s500}"
    RUN_NAME="${RUN_NAME:-latent_inpaint_flow_img4_d64_0_overfit1_s500_teacher_forced}"
    GPU="${GPU}" \
    RUN_NAME="${RUN_NAME}" \
    LATENT_RUN_ROOT="${LATENT_RUN_ROOT}" \
    POINT_RUN_ROOT="${POINT_RUN_ROOT}" \
    INDICES="${INDICES:-all}" \
    USE_IMAGE_COND=1 \
    IMAGE_MAX_VIEWS="${IMAGE_MAX_VIEWS}" \
    IMAGE_FRAME_SELECT="${IMAGE_FRAME_SELECT}" \
    IMAGE_USE_SOURCE_MASK="${IMAGE_USE_SOURCE_MASK}" \
    IMAGE_MASK_CROP_RESOLUTION="${IMAGE_MASK_CROP_RESOLUTION}" \
    IMAGE_COND_AGGREGATION=mean \
    COND_FUSION=concat \
    MASK_DILATE64="${MASK_DILATE64}" \
    MASK_DILATE16="${MASK_DILATE16}" \
    T_VALUES="${T_VALUES}" \
    bash trellis_point_prior_mv/scripts/run_latent_inpaint_teacher_forced.sh
    ;;

  image_only_teacher_forced)
    POINT_RUN_ROOT="${POINT_RUN_ROOT:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint_flow/latent_inpaint_flow_img4_d64_0_imageonly_overfit1_s500}"
    RUN_NAME="${RUN_NAME:-latent_inpaint_flow_img4_d64_0_imageonly_overfit1_s500_teacher_forced}"
    GPU="${GPU}" \
    RUN_NAME="${RUN_NAME}" \
    LATENT_RUN_ROOT="${LATENT_RUN_ROOT}" \
    POINT_RUN_ROOT="${POINT_RUN_ROOT}" \
    INDICES="${INDICES:-all}" \
    USE_IMAGE_COND=1 \
    IMAGE_MAX_VIEWS="${IMAGE_MAX_VIEWS}" \
    IMAGE_FRAME_SELECT="${IMAGE_FRAME_SELECT}" \
    IMAGE_USE_SOURCE_MASK="${IMAGE_USE_SOURCE_MASK}" \
    IMAGE_MASK_CROP_RESOLUTION="${IMAGE_MASK_CROP_RESOLUTION}" \
    IMAGE_COND_AGGREGATION=mean \
    COND_FUSION=image_only \
    MASK_DILATE64="${MASK_DILATE64}" \
    MASK_DILATE16="${MASK_DILATE16}" \
    T_VALUES="${T_VALUES}" \
    bash trellis_point_prior_mv/scripts/run_latent_inpaint_teacher_forced.sh
    ;;

  first_overfit)
    RUN_NAME="${RUN_NAME:-latent_inpaint_flow_imgfirst_d64_0_concat_overfit1_s500}"
    GPU="${GPU}" \
    MODE=smoke \
    RUN_NAME="${RUN_NAME}" \
    RUN_BUILD=0 \
    RUN_TRAIN=1 \
    RUN_EVAL=1 \
    TRAIN_LATENT_ROOT="${TRAIN_LATENT_ROOT}" \
    VAL_LATENT_ROOT="${VAL_LATENT_ROOT}" \
    USE_IMAGE_COND=1 \
    IMAGE_MAX_VIEWS="${IMAGE_MAX_VIEWS}" \
    IMAGE_FRAME_SELECT="${IMAGE_FRAME_SELECT}" \
    IMAGE_USE_SOURCE_MASK="${IMAGE_USE_SOURCE_MASK}" \
    IMAGE_MASK_CROP_RESOLUTION="${IMAGE_MASK_CROP_RESOLUTION}" \
    IMAGE_COND_AGGREGATION=first \
    COND_FUSION=concat \
    MASK_DILATE64="${MASK_DILATE64}" \
    MASK_DILATE16="${MASK_DILATE16}" \
    CFG_DROP_PROB="${CFG_DROP_PROB}" \
    KNOWN_CLAMP_START_T="${KNOWN_CLAMP_START_T}" \
    CLAMP_INITIAL_NOISE="${CLAMP_INITIAL_NOISE}" \
    MAX_STEPS="${MAX_STEPS}" \
    SAVE_EVERY="${SAVE_EVERY}" \
    EVAL_STEPS="${EVAL_STEPS}" \
    bash trellis_point_prior_mv/scripts/run_latent_inpaint_flow.sh
    ;;

  first_teacher_forced)
    POINT_RUN_ROOT="${POINT_RUN_ROOT:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint_flow/latent_inpaint_flow_imgfirst_d64_0_concat_overfit1_s500}"
    RUN_NAME="${RUN_NAME:-latent_inpaint_flow_imgfirst_d64_0_concat_overfit1_s500_teacher_forced}"
    GPU="${GPU}" \
    RUN_NAME="${RUN_NAME}" \
    LATENT_RUN_ROOT="${LATENT_RUN_ROOT}" \
    POINT_RUN_ROOT="${POINT_RUN_ROOT}" \
    INDICES="${INDICES:-all}" \
    USE_IMAGE_COND=1 \
    IMAGE_MAX_VIEWS="${IMAGE_MAX_VIEWS}" \
    IMAGE_FRAME_SELECT="${IMAGE_FRAME_SELECT}" \
    IMAGE_USE_SOURCE_MASK="${IMAGE_USE_SOURCE_MASK}" \
    IMAGE_MASK_CROP_RESOLUTION="${IMAGE_MASK_CROP_RESOLUTION}" \
    IMAGE_COND_AGGREGATION=first \
    COND_FUSION=concat \
    MASK_DILATE64="${MASK_DILATE64}" \
    MASK_DILATE16="${MASK_DILATE16}" \
    T_VALUES="${T_VALUES}" \
    bash trellis_point_prior_mv/scripts/run_latent_inpaint_teacher_forced.sh
    ;;

  *)
    echo "Unsupported DIAG=${DIAG}" >&2
    echo "Use: image_only_overfit | concat_teacher_forced | image_only_teacher_forced | first_overfit | first_teacher_forced" >&2
    exit 2
    ;;
esac
