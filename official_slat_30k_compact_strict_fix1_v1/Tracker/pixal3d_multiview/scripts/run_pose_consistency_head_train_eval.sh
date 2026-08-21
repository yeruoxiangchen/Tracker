#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zjr/Tracker
PYTHON=/home/zjr/anaconda3/envs/reconviagen/bin/python

GPU=${GPU:-1}
HEAD_SCORE_MODE=${HEAD_SCORE_MODE:-pairwise}
HEAD_PAIR_WEIGHT_MODE=${HEAD_PAIR_WEIGHT_MODE:-support}
HEAD_PAIR_WEIGHT_THRESHOLD=${HEAD_PAIR_WEIGHT_THRESHOLD:-0.05}
RUN_NAME=${RUN_NAME:-pose_consistency_head_${HEAD_SCORE_MODE}_001}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-/data/pixal3d_multiview/objaverse_sparse_mv_artraj_pbr_5000_v9_select8/train.json}
VAL_MANIFEST=${VAL_MANIFEST:-/data/pixal3d_multiview/objaverse_sparse_mv_artraj_pbr_5000_v9_select8/val.json}
IMAGE_COND_MODEL=${IMAGE_COND_MODEL:-${ROOT}/models/dinov3-vitl16-pretrain-lvd1689m}

# Current recommended sparse checkpoint for downstream sparse sampling.
SPARSE_CHECKPOINT=${SPARSE_CHECKPOINT:-${ROOT}/pixal3d_multiview/outputs/train_v9/geometry_adapter_from_s900_s1200_001/step_600.pt}

TRAIN_DIR=${TRAIN_DIR:-${ROOT}/pixal3d_multiview/outputs/train_v9/pose_consistency_heads/${RUN_NAME}}
EVAL_ROOT=${EVAL_ROOT:-${ROOT}/pixal3d_multiview/outputs/eval_v9/${RUN_NAME}}

RUN_TRAIN=${RUN_TRAIN:-1}
RUN_SCORE_EVAL=${RUN_SCORE_EVAL:-1}
RUN_SPARSE_EVAL=${RUN_SPARSE_EVAL:-1}
RUN_BASELINE=${RUN_BASELINE:-1}

MAX_STEPS=${MAX_STEPS:-1200}
LR=${LR:-1e-4}
NUM_WORKERS=${NUM_WORKERS:-2}
SAVE_EVERY=${SAVE_EVERY:-600}
NUM_NEGATIVES=${NUM_NEGATIVES:-2}
RANKING_MARGIN=${RANKING_MARGIN:-0.08}

# identity is intentionally not a training negative; keep it for eval only.
NEGATIVE_MODES=${NEGATIVE_MODES:-cyclic_shift1,cyclic_shift2,cross_sample,reverse,noise,large_noise}
NEGATIVE_WEIGHTS=${NEGATIVE_WEIGHTS:-0.25,0.20,0.25,0.15,0.10,0.05}

SCORE_EVAL_INDICES=${SCORE_EVAL_INDICES:-0-127}
SPARSE_EVAL_INDICES=${SPARSE_EVAL_INDICES:-0-63}
SPARSE_STEPS=${SPARSE_STEPS:-30}
POSE_CONSISTENCY_ALPHA=${POSE_CONSISTENCY_ALPHA:-1.0}

export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HUB_OFFLINE=1
export ATTN_BACKEND=flash_attn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/matplotlib
export NUMBA_CACHE_DIR=/tmp/numba_cache

cd "${ROOT}"
mkdir -p "${TRAIN_DIR}" "${EVAL_ROOT}"

echo "[config] run_name=${RUN_NAME}"
echo "[config] gpu=${GPU}"
echo "[config] train_dir=${TRAIN_DIR}"
echo "[config] eval_root=${EVAL_ROOT}"
echo "[config] train_manifest=${TRAIN_MANIFEST}"
echo "[config] val_manifest=${VAL_MANIFEST}"
echo "[config] sparse_checkpoint=${SPARSE_CHECKPOINT}"
echo "[config] head_score_mode=${HEAD_SCORE_MODE}"
echo "[config] head_pair_weight_mode=${HEAD_PAIR_WEIGHT_MODE}"
echo "[config] head_pair_weight_threshold=${HEAD_PAIR_WEIGHT_THRESHOLD}"
echo "[config] negative_modes=${NEGATIVE_MODES}"
echo "[config] negative_weights=${NEGATIVE_WEIGHTS}"
echo "[config] pose_consistency_alpha=${POSE_CONSISTENCY_ALPHA}"

if [[ "${RUN_TRAIN}" == "1" ]]; then
  echo "[start] train pose consistency head"
  "${PYTHON}" -u pixal3d_multiview/train_pose_consistency_head.py \
    --train_manifest "${TRAIN_MANIFEST}" \
    --output_dir "${TRAIN_DIR}" \
    --image_cond_model "${IMAGE_COND_MODEL}" \
    --max_frames 8 \
    --batch_size 1 \
    --num_workers "${NUM_WORKERS}" \
    --max_epochs 1 \
    --max_steps "${MAX_STEPS}" \
    --lr "${LR}" \
    --weight_decay 0.01 \
    --negative_modes "${NEGATIVE_MODES}" \
    --negative_weights "${NEGATIVE_WEIGHTS}" \
    --num_negatives "${NUM_NEGATIVES}" \
    --ranking_margin "${RANKING_MARGIN}" \
    --correct_keep_target 0.65 \
    --correct_keep_weight 0.05 \
    --wrong_min_keep 0.02 \
    --wrong_min_keep_weight 0.0 \
    --head_reduced_dim 128 \
    --head_hidden_dim 256 \
    --head_dropout 0.0 \
    --head_min_gate 0.05 \
    --head_initial_logit 2.0 \
    --head_score_mode "${HEAD_SCORE_MODE}" \
    --head_pair_weight_mode "${HEAD_PAIR_WEIGHT_MODE}" \
    --head_pair_weight_threshold "${HEAD_PAIR_WEIGHT_THRESHOLD}" \
    --empty_policy zero \
    --global_fusion concat \
    --log_every 10 \
    --save_every "${SAVE_EVERY}"
  echo "[done] train pose consistency head: ${TRAIN_DIR}/final.pt"
else
  echo "[skip] train because RUN_TRAIN=${RUN_TRAIN}"
fi

HEAD_CHECKPOINT=${HEAD_CHECKPOINT:-${TRAIN_DIR}/final.pt}
if [[ ! -f "${HEAD_CHECKPOINT}" ]]; then
  echo "[error] missing head checkpoint: ${HEAD_CHECKPOINT}" >&2
  exit 1
fi

if [[ "${RUN_SCORE_EVAL}" == "1" ]]; then
  echo "[start] eval condition-level pose consistency score"
  "${PYTHON}" -u pixal3d_multiview/eval_pose_consistency_head.py \
    --manifest "${VAL_MANIFEST}" \
    --head_checkpoint "${HEAD_CHECKPOINT}" \
    --output_dir "${EVAL_ROOT}/condition_score" \
    --image_cond_model "${IMAGE_COND_MODEL}" \
    --max_frames 8 \
    --indices "${SCORE_EVAL_INDICES}" \
    --pose_modes correct,cyclic_shift1,cyclic_shift2,reverse,noise,large_noise,cross_sample,identity \
    --reference_pose correct \
    --empty_policy zero \
    --global_fusion concat
  echo "[done] condition score eval: ${EVAL_ROOT}/condition_score/score_report.md"
else
  echo "[skip] condition score eval because RUN_SCORE_EVAL=${RUN_SCORE_EVAL}"
fi

if [[ "${RUN_SPARSE_EVAL}" == "1" ]]; then
  if [[ ! -f "${SPARSE_CHECKPOINT}" ]]; then
    echo "[error] missing sparse checkpoint: ${SPARSE_CHECKPOINT}" >&2
    exit 1
  fi

  if [[ "${RUN_BASELINE}" == "1" ]]; then
    echo "[start] sparse sweep baseline without pose consistency head"
    "${PYTHON}" -u pixal3d_multiview/eval_sparse_checkpoint_sweep.py \
      --manifest "${VAL_MANIFEST}" \
      --checkpoints "${SPARSE_CHECKPOINT}" \
      --output_dir "${EVAL_ROOT}/sparse_sweep_baseline_no_head" \
      --indices "${SPARSE_EVAL_INDICES}" \
      --pose_modes correct,cyclic_shift1,cyclic_shift2,reverse,noise,large_noise,identity \
      --reference_pose correct \
      --image_cond_model "${IMAGE_COND_MODEL}" \
      --max_frames 8 \
      --steps "${SPARSE_STEPS}" \
      --empty_policy zero \
      --global_fusion concat \
      --geometry_feature_mode none \
      --view_aggregator gated \
      --geometry_adapter mlp \
    --ablation_name baseline_no_pose_consistency_head
    echo "[done] sparse baseline: ${EVAL_ROOT}/sparse_sweep_baseline_no_head/sweep_report.md"
  fi

  echo "[start] sparse sweep with pose consistency head as view-gate logit prior"
  "${PYTHON}" -u pixal3d_multiview/eval_sparse_checkpoint_sweep.py \
    --manifest "${VAL_MANIFEST}" \
    --checkpoints "${SPARSE_CHECKPOINT}" \
    --output_dir "${EVAL_ROOT}/sparse_sweep_with_head_logit_prior" \
    --indices "${SPARSE_EVAL_INDICES}" \
    --pose_modes correct,cyclic_shift1,cyclic_shift2,reverse,noise,large_noise,identity \
    --reference_pose correct \
    --image_cond_model "${IMAGE_COND_MODEL}" \
    --max_frames 8 \
    --steps "${SPARSE_STEPS}" \
    --empty_policy zero \
    --global_fusion concat \
    --geometry_feature_mode none \
    --view_aggregator gated \
    --geometry_adapter mlp \
    --pose_consistency_head "${HEAD_CHECKPOINT}" \
    --pose_consistency_alpha "${POSE_CONSISTENCY_ALPHA}" \
    --ablation_name pose_consistency_head_logit_prior
  echo "[done] sparse head logit prior: ${EVAL_ROOT}/sparse_sweep_with_head_logit_prior/sweep_report.md"
else
  echo "[skip] sparse eval because RUN_SPARSE_EVAL=${RUN_SPARSE_EVAL}"
fi

cat <<EOF
[summary]
head checkpoint: ${HEAD_CHECKPOINT}
train metrics:   ${TRAIN_DIR}/train_metrics.csv
score report:    ${EVAL_ROOT}/condition_score/score_report.md
sparse baseline: ${EVAL_ROOT}/sparse_sweep_baseline_no_head/sweep_report.md
sparse with head:${EVAL_ROOT}/sparse_sweep_with_head_logit_prior/sweep_report.md
EOF
