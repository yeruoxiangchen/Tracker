#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

GPU=${GPU:-1}
RUN_NAME=${RUN_NAME:-native_pixal3d_sparse_compare_smoke}
MANIFEST=${MANIFEST:-/data/pixal3d_multiview/objaverse_sparse_mv_artraj_pbr_5000_v9_select8/val.json}
OUTPUT_DIR=${OUTPUT_DIR:-/home/zjr/Tracker/pixal3d_multiview/outputs/eval_v9/${RUN_NAME}}

INDICES=${INDICES:-0,1,5,10,20,30,50,80,100}
STEPS=${STEPS:-30}
MAX_FRAMES=${MAX_FRAMES:-8}
NATIVE_FRAME_INDICES=${NATIVE_FRAME_INDICES:-0,1,2}
NATIVE_IMAGE_POLICIES=${NATIVE_IMAGE_POLICIES:-crop_mask}
NATIVE_PARAM_MODES=${NATIVE_PARAM_MODES:-default,fov_narrow,fov_wide,distance_near,distance_far,scale_small,scale_large}
MULTIVIEW_POSE_MODES=${MULTIVIEW_POSE_MODES:-correct,reverse,cyclic_shift1,cyclic_shift2,noise,large_noise,identity}
SAVE_PREVIEWS=${SAVE_PREVIEWS:-0}

mkdir -p "${OUTPUT_DIR}"

CMD=(
  /home/zjr/anaconda3/envs/reconviagen/bin/python -u
  pixal3d_multiview/eval_native_pixal3d_sparse_compare.py
  --manifest "${MANIFEST}"
  --output_dir "${OUTPUT_DIR}"
  --indices "${INDICES}"
  --steps "${STEPS}"
  --max_frames "${MAX_FRAMES}"
  --image_cond_model /home/zjr/Tracker/models/dinov3-vitl16-pretrain-lvd1689m
  --native_frame_indices "${NATIVE_FRAME_INDICES}"
  --native_image_policies "${NATIVE_IMAGE_POLICIES}"
  --native_param_modes "${NATIVE_PARAM_MODES}"
  --multiview_pose_modes "${MULTIVIEW_POSE_MODES}"
)

if [[ "${SAVE_PREVIEWS}" == "1" ]]; then
  CMD+=(--save_previews)
fi

echo "[run] output_dir=${OUTPUT_DIR}"
echo "[run] gpu=${GPU} indices=${INDICES} steps=${STEPS}"

CUDA_VISIBLE_DEVICES="${GPU}" \
HF_HUB_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
"${CMD[@]}"

echo "[done] report=${OUTPUT_DIR}/report.md"
echo "[done] summary=${OUTPUT_DIR}/summary.csv"
echo "[done] pairwise=${OUTPUT_DIR}/pairwise_summary.csv"
