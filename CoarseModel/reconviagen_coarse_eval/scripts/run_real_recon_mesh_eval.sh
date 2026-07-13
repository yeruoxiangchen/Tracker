#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

GPU=${GPU:-1}
PYTHON_BIN=${PYTHON_BIN:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/zjr/Tracker/CoarseModel/reconviagen_coarse_eval/outputs}
MAX_FRAMES=${MAX_FRAMES:-18}
RECON_SEEDS=${RECON_SEEDS:-0}
RECON_NUM_CANDIDATES=${RECON_NUM_CANDIDATES:-}
RECON_RESOLUTION=${RECON_RESOLUTION:-518}
MESH_SIMPLIFY=${MESH_SIMPLIFY:-0.75}
MESH_EVAL_REFERENCE=${MESH_EVAL_REFERENCE:-dataset_model}

SEED_TAG=${RECON_SEEDS//,/_}

for DATASET in GOOD_MESH_TEST reconviagen_20260520_021556; do
  DATASET_DIR="/home/zjr/Tracker/CoarseModel/datasets/${DATASET}"
  CASE_NAME="${DATASET}_fresh_recon_s${SEED_TAG}"

  echo "[real_recon_eval] dataset=${DATASET} case=${CASE_NAME}"

  CMD=(
    "${PYTHON_BIN}" -u CoarseModel/reconviagen_coarse_eval/run_pipeline.py
    --dataset_dir "${DATASET_DIR}"
    --case_name "${CASE_NAME}"
    --output_root "${OUTPUT_ROOT}"
    --python_bin "${PYTHON_BIN}"
    --stages prepare,recon,mesh_eval
    --max_frames "${MAX_FRAMES}"
    --force_recon_generate
    --recon_seeds "${RECON_SEEDS}"
    --recon_resolution "${RECON_RESOLUTION}"
    --mesh_simplify "${MESH_SIMPLIFY}"
    --mesh_eval_reference "${MESH_EVAL_REFERENCE}"
  )
  if [[ -n "${RECON_NUM_CANDIDATES}" ]]; then
    CMD+=(--recon_num_candidates "${RECON_NUM_CANDIDATES}")
  fi

  CUDA_VISIBLE_DEVICES="${GPU}" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn \
  SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib \
  NUMBA_CACHE_DIR=/tmp/numba_cache \
  XDG_CACHE_HOME=/tmp/xdg_cache \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${CMD[@]}"

  echo "[real_recon_eval] report ${OUTPUT_ROOT}/runs/${CASE_NAME}/pipeline_report.json"
  echo "[real_recon_eval] mesh stats ${OUTPUT_ROOT}/mesh_quality/${CASE_NAME}/mesh_quality_report.json"
done
