#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zjr/Tracker
PYTHON=/home/zjr/anaconda3/envs/reconviagen/bin/python

GPU=${GPU:-1}
RUN_NAME=${RUN_NAME:-projection_alignment_uv_depth_only_001}
GEOM_MODE=${GEOM_MODE:-uv_depth_only}
MATCH_DIM=${MATCH_DIM:-128}

TRAIN_MANIFEST=${TRAIN_MANIFEST:-/data/pixal3d_multiview/objaverse_sparse_mv_artraj_pbr_5000_v9_select8/train.json}
VAL_MANIFEST=${VAL_MANIFEST:-/data/pixal3d_multiview/objaverse_sparse_mv_artraj_pbr_5000_v9_select8/val.json}
IMAGE_COND_MODEL=${IMAGE_COND_MODEL:-${ROOT}/models/dinov3-vitl16-pretrain-lvd1689m}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/pixal3d_multiview/outputs/train_v9/projection_alignment_heads/${RUN_NAME}}
RUN_TRAIN=${RUN_TRAIN:-1}
CHECKPOINT=${CHECKPOINT:-}

TRAIN_INDICES=${TRAIN_INDICES:-0-511}
VAL_INDICES=${VAL_INDICES:-0-127}
MAX_EPOCHS=${MAX_EPOCHS:-4}
MAX_STEPS=${MAX_STEPS:-1024}
LR=${LR:-1e-4}
SAVE_EVERY=${SAVE_EVERY:-256}
LOG_EVERY=${LOG_EVERY:-10}

MAX_POS_PER_SAMPLE=${MAX_POS_PER_SAMPLE:-768}
NEG_PER_POS=${NEG_PER_POS:-3}
NUM_NEGATIVES=${NUM_NEGATIVES:-2}
RANK_MARGIN=${RANK_MARGIN:-0.2}
RANK_LOSS_WEIGHT=${RANK_LOSS_WEIGHT:-0.25}
RANK_SCORE_TYPE=${RANK_SCORE_TYPE:-combined}
RANK_MISSING_LOGIT=${RANK_MISSING_LOGIT:--1.0}
RANK_COVERAGE_WEIGHT=${RANK_COVERAGE_WEIGHT:-0.5}
RANK_VOXEL_WEIGHT=${RANK_VOXEL_WEIGHT:-0.5}
RANK_CONSISTENCY_SCORE_WEIGHT=${RANK_CONSISTENCY_SCORE_WEIGHT:-0.25}
RANK_MATCH_SCORE_WEIGHT=${RANK_MATCH_SCORE_WEIGHT:-0.5}
RANK_MATCH_LOGIT_SCORE_WEIGHT=${RANK_MATCH_LOGIT_SCORE_WEIGHT:-0.5}
CONSISTENCY_RANK_LOSS_WEIGHT=${CONSISTENCY_RANK_LOSS_WEIGHT:-0.0}
CONSISTENCY_POSITIVE_LOSS_WEIGHT=${CONSISTENCY_POSITIVE_LOSS_WEIGHT:-0.0}
CONSISTENCY_MARGIN=${CONSISTENCY_MARGIN:-0.05}
CONSISTENCY_MIN_VIEWS=${CONSISTENCY_MIN_VIEWS:-2}
CONSISTENCY_MISSING_SCORE=${CONSISTENCY_MISSING_SCORE:--1.0}
MATCH_CONTRASTIVE_LOSS_WEIGHT=${MATCH_CONTRASTIVE_LOSS_WEIGHT:-0.0}
MATCH_TEMPERATURE=${MATCH_TEMPERATURE:-0.07}
MATCH_TARGET_SOFT_THRESHOLD=${MATCH_TARGET_SOFT_THRESHOLD:-0.999}
MATCH_MIN_VIEWS=${MATCH_MIN_VIEWS:-3}
MATCH_MAX_VOXELS=${MATCH_MAX_VOXELS:-256}
MATCH_VISIBLE_SURFACE_ONLY=${MATCH_VISIBLE_SURFACE_ONLY:-0}
MATCH_VISIBILITY_THRESHOLD=${MATCH_VISIBILITY_THRESHOLD:-0.5}
MATCH_MASK_VALUE_THRESHOLD=${MATCH_MASK_VALUE_THRESHOLD:-0.5}
MATCH_MASK_HIT_THRESHOLD=${MATCH_MASK_HIT_THRESHOLD:-0.5}
MATCH_MIN_SUPPORT_WEIGHT=${MATCH_MIN_SUPPORT_WEIGHT:-1e-6}
MATCH_REQUIRE_VALID_DEPTH=${MATCH_REQUIRE_VALID_DEPTH:-1}
MATCH_MISSING_SCORE=${MATCH_MISSING_SCORE:--1.0}
MATCH_LOGIT_LOSS_WEIGHT=${MATCH_LOGIT_LOSS_WEIGHT:-0.0}
MATCH_ATTENTION_LOSS_WEIGHT=${MATCH_ATTENTION_LOSS_WEIGHT:-0.0}
MATCH_ATTENTION_TEMPERATURE=${MATCH_ATTENTION_TEMPERATURE:-1.0}

NEGATIVE_MODES=${NEGATIVE_MODES:-reverse,cyclic_shift1,cyclic_shift2,cross_sample,identity,noise,large_noise}
NEGATIVE_WEIGHTS=${NEGATIVE_WEIGHTS:-0.25,0.22,0.22,0.16,0.08,0.035,0.035}
EVAL_POSE_MODES=${EVAL_POSE_MODES:-reverse,cyclic_shift1,cyclic_shift2,cross_sample,identity,noise,large_noise}
EVAL_SCORE_TYPES=${EVAL_SCORE_TYPES:-support,attention,fixed_align,coverage_penalized,voxel,combined,view_consistency,combined_consistency,visible_match_consistency,combined_visible_match,visible_match_logit,combined_match_logit}

export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HUB_OFFLINE=1
export ATTN_BACKEND=flash_attn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/matplotlib
export NUMBA_CACHE_DIR=/tmp/numba_cache

cd "${ROOT}"
mkdir -p "${OUTPUT_DIR}"

echo "[config] run_name=${RUN_NAME}"
echo "[config] gpu=${GPU}"
echo "[config] geom_mode=${GEOM_MODE}"
echo "[config] output_dir=${OUTPUT_DIR}"
echo "[config] run_train=${RUN_TRAIN}"
echo "[config] checkpoint=${CHECKPOINT}"
echo "[config] train_manifest=${TRAIN_MANIFEST}"
echo "[config] val_manifest=${VAL_MANIFEST}"
echo "[config] train_indices=${TRAIN_INDICES}"
echo "[config] val_indices=${VAL_INDICES}"
echo "[config] negative_modes=${NEGATIVE_MODES}"
echo "[config] negative_weights=${NEGATIVE_WEIGHTS}"
echo "[config] rank_score_type=${RANK_SCORE_TYPE}"
echo "[config] rank_loss_weight=${RANK_LOSS_WEIGHT}"
echo "[config] consistency_rank_loss_weight=${CONSISTENCY_RANK_LOSS_WEIGHT}"
echo "[config] match_contrastive_loss_weight=${MATCH_CONTRASTIVE_LOSS_WEIGHT}"
echo "[config] match_logit_loss_weight=${MATCH_LOGIT_LOSS_WEIGHT}"
echo "[config] match_attention_loss_weight=${MATCH_ATTENTION_LOSS_WEIGHT}"
echo "[config] match_visible_surface_only=${MATCH_VISIBLE_SURFACE_ONLY}"

ARGS=(
  pixal3d_multiview/train_projection_alignment_head.py
  --train_manifest "${TRAIN_MANIFEST}" \
  --val_manifest "${VAL_MANIFEST}" \
  --output_dir "${OUTPUT_DIR}" \
  --image_cond_model "${IMAGE_COND_MODEL}" \
  --max_frames 8 \
  --train_indices "${TRAIN_INDICES}" \
  --val_indices "${VAL_INDICES}" \
  --max_epochs "${MAX_EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
  --lr "${LR}" \
  --geom_mode "${GEOM_MODE}" \
  --match_dim "${MATCH_DIM}" \
  --max_pos_per_sample "${MAX_POS_PER_SAMPLE}" \
  --neg_per_pos "${NEG_PER_POS}" \
  --negative_modes "${NEGATIVE_MODES}" \
  --negative_weights "${NEGATIVE_WEIGHTS}" \
  --num_negatives "${NUM_NEGATIVES}" \
  --eval_pose_modes "${EVAL_POSE_MODES}" \
  --eval_score_types "${EVAL_SCORE_TYPES}" \
  --rank_margin "${RANK_MARGIN}" \
  --rank_score_type "${RANK_SCORE_TYPE}" \
  --rank_missing_logit "${RANK_MISSING_LOGIT}" \
  --rank_coverage_weight "${RANK_COVERAGE_WEIGHT}" \
  --rank_voxel_weight "${RANK_VOXEL_WEIGHT}" \
  --rank_consistency_score_weight "${RANK_CONSISTENCY_SCORE_WEIGHT}" \
  --rank_match_score_weight "${RANK_MATCH_SCORE_WEIGHT}" \
  --rank_match_logit_score_weight "${RANK_MATCH_LOGIT_SCORE_WEIGHT}" \
  --consistency_rank_loss_weight "${CONSISTENCY_RANK_LOSS_WEIGHT}" \
  --consistency_positive_loss_weight "${CONSISTENCY_POSITIVE_LOSS_WEIGHT}" \
  --consistency_margin "${CONSISTENCY_MARGIN}" \
  --consistency_min_views "${CONSISTENCY_MIN_VIEWS}" \
  --consistency_missing_score "${CONSISTENCY_MISSING_SCORE}" \
  --match_contrastive_loss_weight "${MATCH_CONTRASTIVE_LOSS_WEIGHT}" \
  --match_logit_loss_weight "${MATCH_LOGIT_LOSS_WEIGHT}" \
  --match_attention_loss_weight "${MATCH_ATTENTION_LOSS_WEIGHT}" \
  --match_attention_temperature "${MATCH_ATTENTION_TEMPERATURE}" \
  --match_temperature "${MATCH_TEMPERATURE}" \
  --match_target_soft_threshold "${MATCH_TARGET_SOFT_THRESHOLD}" \
  --match_min_views "${MATCH_MIN_VIEWS}" \
  --match_max_voxels "${MATCH_MAX_VOXELS}" \
  --match_visible_surface_only "${MATCH_VISIBLE_SURFACE_ONLY}" \
  --match_visibility_threshold "${MATCH_VISIBILITY_THRESHOLD}" \
  --match_mask_value_threshold "${MATCH_MASK_VALUE_THRESHOLD}" \
  --match_mask_hit_threshold "${MATCH_MASK_HIT_THRESHOLD}" \
  --match_min_support_weight "${MATCH_MIN_SUPPORT_WEIGHT}" \
  --match_require_valid_depth "${MATCH_REQUIRE_VALID_DEPTH}" \
  --match_missing_score "${MATCH_MISSING_SCORE}" \
  --voxel_loss_weight 1.0 \
  --view_loss_weight 0.25 \
  --attn_loss_weight 0.5 \
  --rank_loss_weight "${RANK_LOSS_WEIGHT}" \
  --reg_loss_weight 0.01 \
  --log_every "${LOG_EVERY}" \
  --save_every "${SAVE_EVERY}"
)

if [[ -n "${CHECKPOINT}" ]]; then
  ARGS+=(--checkpoint "${CHECKPOINT}")
fi
if [[ "${RUN_TRAIN}" == "0" ]]; then
  ARGS+=(--eval_only)
fi

"${PYTHON}" -u "${ARGS[@]}"

cat <<EOF
[summary]
checkpoint: ${OUTPUT_DIR}/final.pt
report:     ${OUTPUT_DIR}/report.md
summary:    ${OUTPUT_DIR}/summary.json
pose csv:   ${OUTPUT_DIR}/pose_alignment_summary.csv
target csv: ${OUTPUT_DIR}/target_voxel_metrics.csv
EOF
