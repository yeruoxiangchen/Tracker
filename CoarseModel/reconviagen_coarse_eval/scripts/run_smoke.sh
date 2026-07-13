#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

GPU=${GPU:-1}
MANIFEST=${MANIFEST:-/data/pixal3d_multiview/objaverse_sparse_mv_artraj_pbr_5000_v9_select8/val.json}
SAMPLE_INDEX=${SAMPLE_INDEX:-0}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/zjr/Tracker/CoarseModel/reconviagen_coarse_eval/outputs}
PYTHON_BIN=${PYTHON_BIN:-/home/zjr/anaconda3/envs/reconviagen/bin/python}

CUDA_VISIBLE_DEVICES=${GPU} \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
PYOPENGL_PLATFORM=egl \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${PYTHON_BIN}" -u CoarseModel/reconviagen_coarse_eval/run_pipeline.py \
  --manifest "${MANIFEST}" \
  --sample_index "${SAMPLE_INDEX}" \
  --output_root "${OUTPUT_ROOT}" \
  --python_bin "${PYTHON_BIN}" \
  --max_frames 8 \
  --stages prepare,recon,mesh_eval,coarse \
  --recon_seeds 0 \
  --mesh_simplify 0.75 \
  --mesh_eval_samples 12000 \
  --template_views 16 \
  --template_inplane 3 \
  --cluster_num 512 \
  --skip_deformation

