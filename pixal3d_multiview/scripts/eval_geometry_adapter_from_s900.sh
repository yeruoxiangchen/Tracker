#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zjr/Tracker
PYTHON=/home/zjr/anaconda3/envs/reconviagen/bin/python

GPU=${GPU:-1}
TRAIN_DIR=${TRAIN_DIR:-${ROOT}/pixal3d_multiview/outputs/train_v9/geometry_adapter_from_s900_s1200_001}
VAL_MANIFEST=${VAL_MANIFEST:-/data/pixal3d_multiview/objaverse_sparse_mv_artraj_pbr_5000_v9_select8/val.json}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-/data/pixal3d_multiview/objaverse_sparse_mv_artraj_pbr_5000_v9_select8/train.json}
IMAGE_COND_MODEL=${IMAGE_COND_MODEL:-${ROOT}/models/dinov3-vitl16-pretrain-lvd1689m}
EVAL_ROOT=${EVAL_ROOT:-${ROOT}/pixal3d_multiview/outputs/eval_v9/geometry_adapter_from_s900_s1200_001}
OLD_BEST=${OLD_BEST:-${ROOT}/pixal3d_multiview/outputs/train_v9/view_gated_agg_s1200/step_900.pt}
RUN_BASELINE=${RUN_BASELINE:-1}
RUN_PREVIEW=${RUN_PREVIEW:-1}

export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HUB_OFFLINE=1
export ATTN_BACKEND=flash_attn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/matplotlib
export NUMBA_CACHE_DIR=/tmp/numba_cache

cd "${ROOT}"
mkdir -p "${EVAL_ROOT}/fixed_loss" "${EVAL_ROOT}/logs"

CHECKPOINTS=(
  "${TRAIN_DIR}/step_300.pt"
  "${TRAIN_DIR}/step_600.pt"
  "${TRAIN_DIR}/step_900.pt"
  "${TRAIN_DIR}/step_1200.pt"
  "${TRAIN_DIR}/final.pt"
)

for ckpt in "${CHECKPOINTS[@]}"; do
  if [[ ! -f "${ckpt}" ]]; then
    echo "[missing] ${ckpt}" >&2
    exit 1
  fi
done

echo "[eval] train_dir=${TRAIN_DIR}"
echo "[eval] eval_root=${EVAL_ROOT}"
echo "[eval] gpu=${GPU}"

if [[ "${RUN_BASELINE}" == "1" ]]; then
  if [[ ! -f "${OLD_BEST}" ]]; then
    echo "[missing] old best checkpoint: ${OLD_BEST}" >&2
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
    --output_dir "${EVAL_ROOT}/baseline_old_s900_pose_sweep_strong_0_63" \
    --indices 0-63 \
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

echo "[start] fixed loss on val"
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
echo "[done] fixed loss on val"

echo "[start] fixed loss on train subset"
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
echo "[done] fixed loss on train subset"

echo "[start] strong pose checkpoint sweep"
"${PYTHON}" -u pixal3d_multiview/eval_sparse_checkpoint_sweep.py \
  --manifest "${VAL_MANIFEST}" \
  --checkpoints "$(IFS=,; echo "${CHECKPOINTS[*]}")" \
  --output_dir "${EVAL_ROOT}/pose_sweep_strong_0_63" \
  --indices 0-63 \
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
echo "[done] strong pose checkpoint sweep"

if [[ "${RUN_PREVIEW}" == "1" ]]; then
  echo "[start] preview sparse samples for candidate checkpoints"
  for ckpt in "${TRAIN_DIR}/step_900.pt" "${TRAIN_DIR}/step_1200.pt" "${TRAIN_DIR}/final.pt"; do
    tag=$(basename "${ckpt}" .pt)
    "${PYTHON}" -u pixal3d_multiview/eval_sparse_checkpoint_sweep.py \
      --manifest "${VAL_MANIFEST}" \
      --checkpoints "${ckpt}" \
      --output_dir "${EVAL_ROOT}/preview_${tag}" \
      --indices 0,1,5,10,20,30,50,80,100 \
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
  echo "[done] preview sparse samples"
else
  echo "[skip] preview sparse samples because RUN_PREVIEW=${RUN_PREVIEW}"
fi

echo "[summary]"
echo "fixed loss: ${EVAL_ROOT}/fixed_loss"
echo "old best fixed loss: ${EVAL_ROOT}/fixed_loss_old_best"
echo "old best pose sweep: ${EVAL_ROOT}/baseline_old_s900_pose_sweep_strong_0_63/sweep_report.md"
echo "pose sweep: ${EVAL_ROOT}/pose_sweep_strong_0_63/sweep_report.md"
echo "pose sweep csv: ${EVAL_ROOT}/pose_sweep_strong_0_63/sweep_summary.csv"
echo "pairwise csv: ${EVAL_ROOT}/pose_sweep_strong_0_63/pose_pairwise.csv"
echo "rank csv: ${EVAL_ROOT}/pose_sweep_strong_0_63/pose_rank_summary.csv"
echo "previews: ${EVAL_ROOT}/preview_step_900, ${EVAL_ROOT}/preview_step_1200, ${EVAL_ROOT}/preview_final"
