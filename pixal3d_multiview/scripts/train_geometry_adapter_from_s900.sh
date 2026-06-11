#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zjr/Tracker
PYTHON=/home/zjr/anaconda3/envs/reconviagen/bin/python

GPU=${GPU:-1}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-/data/pixal3d_multiview/objaverse_sparse_mv_artraj_pbr_5000_v9_select8/train.json}
IMAGE_COND_MODEL=${IMAGE_COND_MODEL:-${ROOT}/models/dinov3-vitl16-pretrain-lvd1689m}
INIT_WEIGHTS=${INIT_WEIGHTS:-${ROOT}/pixal3d_multiview/outputs/train_v9/view_gated_agg_s1200/step_900.pt}
OUT_DIR=${OUT_DIR:-${ROOT}/pixal3d_multiview/outputs/train_v9/geometry_adapter_from_s900_s1200_001}
MAX_STEPS=${MAX_STEPS:-1200}
LR=${LR:-1e-4}
NUM_WORKERS=${NUM_WORKERS:-2}

export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HUB_OFFLINE=1
export ATTN_BACKEND=flash_attn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/matplotlib
export NUMBA_CACHE_DIR=/tmp/numba_cache

cd "${ROOT}"
mkdir -p "${OUT_DIR}"

echo "[train] output=${OUT_DIR}"
echo "[train] init_weights=${INIT_WEIGHTS}"
echo "[train] gpu=${GPU} max_steps=${MAX_STEPS} lr=${LR}"

"${PYTHON}" -u pixal3d_multiview/train_sparse_multiview.py \
  --train_manifest "${TRAIN_MANIFEST}" \
  --output_dir "${OUT_DIR}" \
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
  --save_every 300 \
  --amp_dtype bf16 \
  --empty_policy zero \
  --global_fusion concat \
  --geometry_feature_mode none

echo "[done] ${OUT_DIR}/final.pt"
