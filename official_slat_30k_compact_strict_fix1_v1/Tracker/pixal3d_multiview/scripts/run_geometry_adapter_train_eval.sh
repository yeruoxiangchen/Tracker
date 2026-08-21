#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zjr/Tracker
PYTHON=/home/zjr/anaconda3/envs/reconviagen/bin/python

GPU=${GPU:-1}
RUN_NAME=${RUN_NAME:-geometry_adapter_from_s900_s1200_001}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-/data/pixal3d_multiview/objaverse_sparse_mv_artraj_pbr_5000_v9_select8/train.json}
VAL_MANIFEST=${VAL_MANIFEST:-/data/pixal3d_multiview/objaverse_sparse_mv_artraj_pbr_5000_v9_select8/val.json}
IMAGE_COND_MODEL=${IMAGE_COND_MODEL:-${ROOT}/models/dinov3-vitl16-pretrain-lvd1689m}
INIT_WEIGHTS=${INIT_WEIGHTS:-${ROOT}/pixal3d_multiview/outputs/train_v9/view_gated_agg_s1200/step_900.pt}
OLD_BEST=${OLD_BEST:-${ROOT}/pixal3d_multiview/outputs/train_v9/view_gated_agg_s1200/step_900.pt}
TRAIN_DIR=${TRAIN_DIR:-${ROOT}/pixal3d_multiview/outputs/train_v9/${RUN_NAME}}
EVAL_ROOT=${EVAL_ROOT:-${ROOT}/pixal3d_multiview/outputs/eval_v9/${RUN_NAME}}

RUN_TRAIN=${RUN_TRAIN:-1}
RUN_EVAL=${RUN_EVAL:-1}
RUN_BASELINE=${RUN_BASELINE:-1}
RUN_PREVIEW=${RUN_PREVIEW:-1}

MAX_STEPS=${MAX_STEPS:-1200}
LR=${LR:-1e-4}
NUM_WORKERS=${NUM_WORKERS:-2}
SAVE_EVERY=${SAVE_EVERY:-300}
EVAL_INDICES=${EVAL_INDICES:-0-63}
PREVIEW_INDICES=${PREVIEW_INDICES:-0,1,5,10,20,30,50,80,100}

export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HUB_OFFLINE=1
export ATTN_BACKEND=flash_attn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/matplotlib
export NUMBA_CACHE_DIR=/tmp/numba_cache

cd "${ROOT}"
mkdir -p "${TRAIN_DIR}" "${EVAL_ROOT}/fixed_loss" "${EVAL_ROOT}/logs"

echo "[config] run_name=${RUN_NAME}"
echo "[config] gpu=${GPU}"
echo "[config] train_dir=${TRAIN_DIR}"
echo "[config] eval_root=${EVAL_ROOT}"
echo "[config] train_manifest=${TRAIN_MANIFEST}"
echo "[config] val_manifest=${VAL_MANIFEST}"
echo "[config] init_weights=${INIT_WEIGHTS}"

if [[ "${RUN_TRAIN}" == "1" ]]; then
  echo "[start] train geometry adapter"
  "${PYTHON}" -u pixal3d_multiview/train_sparse_multiview.py \
    --train_manifest "${TRAIN_MANIFEST}" \
    --output_dir "${TRAIN_DIR}" \
    --init_weights "${INIT_WEIGHTS}" \
    --image_cond_model "${IMAGE_COND_MODEL}" \
    --max_frames 8 \
    --batch_size 1 \
    --num_workers "${NUM_WORKERS}" \
    --max_epochs 1 \
    --max_steps "${MAX_STEPS}" \
    --lr "${LR}" \
    --weight_decay 0.01 \
    --trainable none \
    --view_aggregator gated \
    --freeze_view_aggregator \
    --geometry_adapter mlp \
    --geometry_adapter_hidden_dim 256 \
    --geometry_adapter_dropout 0.0 \
    --geometry_adapter_residual_scale 1.0 \
    --cfg_drop_prob 0.0 \
    --log_every 10 \
    --save_every "${SAVE_EVERY}" \
    --amp_dtype bf16 \
    --empty_policy zero \
    --global_fusion concat \
    --geometry_feature_mode none
  echo "[done] train geometry adapter: ${TRAIN_DIR}/final.pt"
else
  echo "[skip] train because RUN_TRAIN=${RUN_TRAIN}"
fi

if [[ "${RUN_EVAL}" != "1" ]]; then
  echo "[skip] eval because RUN_EVAL=${RUN_EVAL}"
  echo "[summary] train_dir=${TRAIN_DIR}"
  exit 0
fi

if [[ ! -f "${TRAIN_DIR}/final.pt" ]]; then
  echo "[error] missing trained checkpoint: ${TRAIN_DIR}/final.pt" >&2
  exit 1
fi

CHECKPOINTS=()
for ckpt in \
  "${TRAIN_DIR}/step_300.pt" \
  "${TRAIN_DIR}/step_600.pt" \
  "${TRAIN_DIR}/step_900.pt" \
  "${TRAIN_DIR}/step_1200.pt" \
  "${TRAIN_DIR}/final.pt"; do
  if [[ -f "${ckpt}" ]]; then
    CHECKPOINTS+=("${ckpt}")
  fi
done

if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
  echo "[error] no checkpoints found in ${TRAIN_DIR}" >&2
  exit 1
fi

CHECKPOINTS_CSV=$(IFS=,; echo "${CHECKPOINTS[*]}")
echo "[eval] checkpoints=${CHECKPOINTS_CSV}"

if [[ "${RUN_BASELINE}" == "1" ]]; then
  if [[ ! -f "${OLD_BEST}" ]]; then
    echo "[error] missing old best checkpoint: ${OLD_BEST}" >&2
    exit 1
  fi
  mkdir -p "${EVAL_ROOT}/fixed_loss_old_best"

  echo "[start] old best fixed loss baseline"
  "${PYTHON}" -u pixal3d_multiview/eval_fixed_train_loss.py \
    --train_manifest "${VAL_MANIFEST}" \
    --checkpoint "${OLD_BEST}" \
    --checkpoint_only \
    --output "${EVAL_ROOT}/fixed_loss_old_best/val_old_s900.json" \
    --image_cond_model "${IMAGE_COND_MODEL}" \
    --max_frames 8 \
    --max_samples 128 \
    --fixed_t 0.5 \
    --amp_dtype bf16 \
    --empty_policy zero \
    --global_fusion concat \
    --geometry_feature_mode none \
    --view_aggregator gated \
    --quiet
  "${PYTHON}" -u pixal3d_multiview/eval_fixed_train_loss.py \
    --train_manifest "${TRAIN_MANIFEST}" \
    --checkpoint "${OLD_BEST}" \
    --checkpoint_only \
    --output "${EVAL_ROOT}/fixed_loss_old_best/train_old_s900.json" \
    --image_cond_model "${IMAGE_COND_MODEL}" \
    --max_frames 8 \
    --max_samples 128 \
    --fixed_t 0.5 \
    --amp_dtype bf16 \
    --empty_policy zero \
    --global_fusion concat \
    --geometry_feature_mode none \
    --view_aggregator gated \
    --quiet
  echo "[done] old best fixed loss baseline"

  echo "[start] old best strong pose sweep baseline"
  "${PYTHON}" -u pixal3d_multiview/eval_sparse_checkpoint_sweep.py \
    --manifest "${VAL_MANIFEST}" \
    --checkpoints "${OLD_BEST}" \
    --output_dir "${EVAL_ROOT}/baseline_old_s900_pose_sweep_strong" \
    --indices "${EVAL_INDICES}" \
    --pose_modes correct,reverse,noise,large_noise,identity \
    --reference_pose correct \
    --image_cond_model "${IMAGE_COND_MODEL}" \
    --max_frames 8 \
    --steps 30 \
    --empty_policy zero \
    --global_fusion concat \
    --geometry_feature_mode none \
    --view_aggregator gated \
    --ablation_name old_s900_no_geometry_adapter_baseline
  echo "[done] old best strong pose sweep baseline"
else
  echo "[skip] old best baseline because RUN_BASELINE=${RUN_BASELINE}"
fi

echo "[start] geometry adapter fixed loss on val"
for ckpt in "${CHECKPOINTS[@]}"; do
  tag=$(basename "${ckpt}" .pt)
  "${PYTHON}" -u pixal3d_multiview/eval_fixed_train_loss.py \
    --train_manifest "${VAL_MANIFEST}" \
    --checkpoint "${ckpt}" \
    --checkpoint_only \
    --output "${EVAL_ROOT}/fixed_loss/val_${tag}.json" \
    --image_cond_model "${IMAGE_COND_MODEL}" \
    --max_frames 8 \
    --max_samples 128 \
    --fixed_t 0.5 \
    --amp_dtype bf16 \
    --empty_policy zero \
    --global_fusion concat \
    --geometry_feature_mode none \
    --view_aggregator gated \
    --geometry_adapter mlp \
    --quiet
done
echo "[done] geometry adapter fixed loss on val"

echo "[start] geometry adapter fixed loss on train subset"
for ckpt in "${CHECKPOINTS[@]}"; do
  tag=$(basename "${ckpt}" .pt)
  "${PYTHON}" -u pixal3d_multiview/eval_fixed_train_loss.py \
    --train_manifest "${TRAIN_MANIFEST}" \
    --checkpoint "${ckpt}" \
    --checkpoint_only \
    --output "${EVAL_ROOT}/fixed_loss/train_${tag}.json" \
    --image_cond_model "${IMAGE_COND_MODEL}" \
    --max_frames 8 \
    --max_samples 128 \
    --fixed_t 0.5 \
    --amp_dtype bf16 \
    --empty_policy zero \
    --global_fusion concat \
    --geometry_feature_mode none \
    --view_aggregator gated \
    --geometry_adapter mlp \
    --quiet
done
echo "[done] geometry adapter fixed loss on train subset"

echo "[start] geometry adapter strong pose checkpoint sweep"
"${PYTHON}" -u pixal3d_multiview/eval_sparse_checkpoint_sweep.py \
  --manifest "${VAL_MANIFEST}" \
  --checkpoints "${CHECKPOINTS_CSV}" \
  --output_dir "${EVAL_ROOT}/pose_sweep_strong" \
  --indices "${EVAL_INDICES}" \
  --pose_modes correct,reverse,noise,large_noise,identity \
  --reference_pose correct \
  --image_cond_model "${IMAGE_COND_MODEL}" \
  --max_frames 8 \
  --steps 30 \
  --empty_policy zero \
  --global_fusion concat \
  --geometry_feature_mode none \
  --view_aggregator gated \
  --geometry_adapter mlp \
  --ablation_name geometry_adapter_from_s900_strong_pose_sweep
echo "[done] geometry adapter strong pose checkpoint sweep"

if [[ "${RUN_PREVIEW}" == "1" ]]; then
  echo "[start] geometry adapter preview sparse samples"
  PREVIEW_CKPTS=()
  for ckpt in "${TRAIN_DIR}/step_900.pt" "${TRAIN_DIR}/step_1200.pt" "${TRAIN_DIR}/final.pt"; do
    if [[ -f "${ckpt}" ]]; then
      PREVIEW_CKPTS+=("${ckpt}")
    fi
  done
  for ckpt in "${PREVIEW_CKPTS[@]}"; do
    tag=$(basename "${ckpt}" .pt)
    "${PYTHON}" -u pixal3d_multiview/eval_sparse_checkpoint_sweep.py \
      --manifest "${VAL_MANIFEST}" \
      --checkpoints "${ckpt}" \
      --output_dir "${EVAL_ROOT}/preview_${tag}" \
      --indices "${PREVIEW_INDICES}" \
      --pose_modes correct,reverse,large_noise,identity \
      --reference_pose correct \
      --image_cond_model "${IMAGE_COND_MODEL}" \
      --max_frames 8 \
      --steps 50 \
      --empty_policy zero \
      --global_fusion concat \
      --geometry_feature_mode none \
      --view_aggregator gated \
      --geometry_adapter mlp \
      --save_previews \
      --ablation_name "geometry_adapter_${tag}_preview"
  done
  echo "[done] geometry adapter preview sparse samples"
else
  echo "[skip] preview because RUN_PREVIEW=${RUN_PREVIEW}"
fi

echo "[summary]"
echo "train_dir: ${TRAIN_DIR}"
echo "eval_root: ${EVAL_ROOT}"
echo "fixed loss: ${EVAL_ROOT}/fixed_loss"
echo "old best fixed loss: ${EVAL_ROOT}/fixed_loss_old_best"
echo "old best pose sweep: ${EVAL_ROOT}/baseline_old_s900_pose_sweep_strong/sweep_report.md"
echo "geometry adapter pose sweep: ${EVAL_ROOT}/pose_sweep_strong/sweep_report.md"
echo "geometry adapter pose sweep csv: ${EVAL_ROOT}/pose_sweep_strong/sweep_summary.csv"
echo "geometry adapter pairwise csv: ${EVAL_ROOT}/pose_sweep_strong/pose_pairwise.csv"
echo "geometry adapter rank csv: ${EVAL_ROOT}/pose_sweep_strong/pose_rank_summary.csv"
