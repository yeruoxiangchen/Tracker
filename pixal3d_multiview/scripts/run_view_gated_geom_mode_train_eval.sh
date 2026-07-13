#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/zjr/Tracker}
PY=${PY:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
GPU=${GPU:-1}

GEOM_MODE=${GEOM_MODE:-no_xyz}
RUN_NAME=${RUN_NAME:-view_gated_${GEOM_MODE}_from_s900_s1200_001}
RUN_TRAIN=${RUN_TRAIN:-1}
RUN_EVAL=${RUN_EVAL:-1}

TRAIN_MANIFEST=${TRAIN_MANIFEST:-/data/pixal3d_multiview/objaverse_sparse_mv_artraj_pbr_5000_v9_select8/train.json}
VAL_MANIFEST=${VAL_MANIFEST:-/data/pixal3d_multiview/objaverse_sparse_mv_artraj_pbr_5000_v9_select8/val.json}
IMAGE_COND_MODEL=${IMAGE_COND_MODEL:-${ROOT}/models/dinov3-vitl16-pretrain-lvd1689m}
INIT_WEIGHTS=${INIT_WEIGHTS:-${ROOT}/pixal3d_multiview/outputs/train_v9/view_gated_agg_s1200/step_900.pt}

TRAIN_DIR=${TRAIN_DIR:-${ROOT}/pixal3d_multiview/outputs/train_v9/${RUN_NAME}}
EVAL_ROOT=${EVAL_ROOT:-${ROOT}/pixal3d_multiview/outputs/eval_v9/${RUN_NAME}}

MAX_STEPS=${MAX_STEPS:-1200}
MAX_EPOCHS=${MAX_EPOCHS:-1}
LR=${LR:-5e-6}
CFG_DROP_PROB=${CFG_DROP_PROB:-0.0}
SAVE_EVERY=${SAVE_EVERY:-300}
LOG_EVERY=${LOG_EVERY:-20}
SPARSE_STEPS=${SPARSE_STEPS:-30}
EVAL_INDICES=${EVAL_INDICES:-0-63}
PREVIEW_INDICES=${PREVIEW_INDICES:-0,1,5,10,20,30,50,80,100}
POSE_MODES=${POSE_MODES:-correct,reverse,cyclic_shift1,cyclic_shift2,noise,large_noise,identity}

case "${GEOM_MODE}" in
  full|no_xyz|uv_depth_only|support_only) ;;
  *)
    echo "[error] GEOM_MODE should be one of: full no_xyz uv_depth_only support_only, got ${GEOM_MODE}" >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HUB_OFFLINE=1
export ATTN_BACKEND=flash_attn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/matplotlib
export NUMBA_CACHE_DIR=/tmp/numba_cache

cd "${ROOT}"
mkdir -p "${TRAIN_DIR}" "${EVAL_ROOT}/fixed_loss"

echo "[config] run_name=${RUN_NAME}"
echo "[config] gpu=${GPU}"
echo "[config] geom_mode=${GEOM_MODE}"
echo "[config] train_dir=${TRAIN_DIR}"
echo "[config] eval_root=${EVAL_ROOT}"
echo "[config] init_weights=${INIT_WEIGHTS}"
echo "[config] train_manifest=${TRAIN_MANIFEST}"
echo "[config] val_manifest=${VAL_MANIFEST}"

if [[ "${RUN_TRAIN}" == "1" ]]; then
  echo "[start] train view-gated aggregator with geom_mode=${GEOM_MODE}"
  "${PY}" -u pixal3d_multiview/train_sparse_multiview.py \
    --train_manifest "${TRAIN_MANIFEST}" \
    --output_dir "${TRAIN_DIR}" \
    --init_weights "${INIT_WEIGHTS}" \
    --image_cond_model "${IMAGE_COND_MODEL}" \
    --max_frames 8 \
    --max_epochs "${MAX_EPOCHS}" \
    --max_steps "${MAX_STEPS}" \
    --lr "${LR}" \
    --cfg_drop_prob "${CFG_DROP_PROB}" \
    --trainable none \
    --view_aggregator gated \
    --view_aggregator_geom_mode "${GEOM_MODE}" \
    --view_aggregator_reduced_dim 128 \
    --view_aggregator_hidden_dim 256 \
    --view_aggregator_dropout 0.0 \
    --view_aggregator_residual_scale 1.0 \
    --empty_policy zero \
    --global_fusion concat \
    --geometry_feature_mode none \
    --batch_size 1 \
    --num_workers 0 \
    --amp_dtype bf16 \
    --save_every "${SAVE_EVERY}" \
    --log_every "${LOG_EVERY}"
  echo "[done] train: ${TRAIN_DIR}/final.pt"
else
  echo "[skip] train because RUN_TRAIN=${RUN_TRAIN}"
fi

if [[ "${RUN_EVAL}" != "1" ]]; then
  echo "[skip] eval because RUN_EVAL=${RUN_EVAL}"
  exit 0
fi

CHECKPOINTS=(
  "${TRAIN_DIR}/step_300.pt"
  "${TRAIN_DIR}/step_600.pt"
  "${TRAIN_DIR}/step_900.pt"
  "${TRAIN_DIR}/step_1200.pt"
  "${TRAIN_DIR}/final.pt"
)

EXISTING_CHECKPOINTS=()
for ckpt in "${CHECKPOINTS[@]}"; do
  if [[ -f "${ckpt}" ]]; then
    EXISTING_CHECKPOINTS+=("${ckpt}")
  else
    echo "[missing] ${ckpt}"
  fi
done
if [[ "${#EXISTING_CHECKPOINTS[@]}" -eq 0 ]]; then
  echo "[error] no checkpoints found under ${TRAIN_DIR}" >&2
  exit 1
fi

echo "[start] fixed loss on val"
for ckpt in "${EXISTING_CHECKPOINTS[@]}"; do
  tag=$(basename "${ckpt}" .pt)
  "${PY}" -u pixal3d_multiview/eval_fixed_train_loss.py \
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
    --view_aggregator_geom_mode "${GEOM_MODE}" \
    --quiet
done
echo "[done] fixed loss on val"

echo "[start] fixed loss on train subset"
for ckpt in "${EXISTING_CHECKPOINTS[@]}"; do
  tag=$(basename "${ckpt}" .pt)
  "${PY}" -u pixal3d_multiview/eval_fixed_train_loss.py \
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
    --view_aggregator_geom_mode "${GEOM_MODE}" \
    --quiet
done
echo "[done] fixed loss on train subset"

echo "[start] pose sweep"
"${PY}" -u pixal3d_multiview/eval_sparse_checkpoint_sweep.py \
  --manifest "${VAL_MANIFEST}" \
  --checkpoints "$(IFS=,; echo "${EXISTING_CHECKPOINTS[*]}")" \
  --output_dir "${EVAL_ROOT}/pose_sweep_${EVAL_INDICES}" \
  --indices "${EVAL_INDICES}" \
  --pose_modes "${POSE_MODES}" \
  --reference_pose correct \
  --image_cond_model "${IMAGE_COND_MODEL}" \
  --max_frames 8 \
  --steps "${SPARSE_STEPS}" \
  --empty_policy zero \
  --global_fusion concat \
  --geometry_feature_mode none \
  --view_aggregator gated \
  --view_aggregator_geom_mode "${GEOM_MODE}" \
  --ablation_name "view_gated_${GEOM_MODE}_pose_sweep"
echo "[done] pose sweep"

BEST_CKPT="${TRAIN_DIR}/final.pt"
if [[ ! -f "${BEST_CKPT}" ]]; then
  LAST_CKPT_INDEX=$((${#EXISTING_CHECKPOINTS[@]} - 1))
  BEST_CKPT="${EXISTING_CHECKPOINTS[${LAST_CKPT_INDEX}]}"
fi

echo "[start] preview sparse samples: ${BEST_CKPT}"
"${PY}" -u pixal3d_multiview/eval_sparse_checkpoint_sweep.py \
  --manifest "${VAL_MANIFEST}" \
  --checkpoints "${BEST_CKPT}" \
  --output_dir "${EVAL_ROOT}/preview_final" \
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
  --view_aggregator_geom_mode "${GEOM_MODE}" \
  --save_previews \
  --ablation_name "view_gated_${GEOM_MODE}_preview"
echo "[done] preview sparse samples"

echo "[summary]"
echo "train_dir: ${TRAIN_DIR}"
echo "eval_root: ${EVAL_ROOT}"
echo "fixed_loss: ${EVAL_ROOT}/fixed_loss"
echo "pose_sweep: ${EVAL_ROOT}/pose_sweep_${EVAL_INDICES}/sweep_report.md"
echo "pose_sweep_csv: ${EVAL_ROOT}/pose_sweep_${EVAL_INDICES}/sweep_summary.csv"
echo "pairwise_csv: ${EVAL_ROOT}/pose_sweep_${EVAL_INDICES}/pose_pairwise.csv"
echo "rank_csv: ${EVAL_ROOT}/pose_sweep_${EVAL_INDICES}/pose_rank_summary.csv"
echo "preview: ${EVAL_ROOT}/preview_final"
