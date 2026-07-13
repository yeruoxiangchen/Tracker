#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

GPU=${GPU:-1}
PYTHON_BIN=${PYTHON_BIN:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
DATASET_DIR=${DATASET_DIR:-/home/zjr/Tracker/CoarseModel/datasets/heimei}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/zjr/Tracker/CoarseModel/reconviagen_coarse_eval/outputs}
MAX_FRAMES=${MAX_FRAMES:-18}
TRAJECTORY_MODE=${TRAJECTORY_MODE:-balanced}
ARC_SPAN_DEG=${ARC_SPAN_DEG:-90}
ARC_CENTER_FRACTION=${ARC_CENTER_FRACTION:-0.5}
RANDOM_SEED=${RANDOM_SEED:-0}
RECON_SEEDS=${RECON_SEEDS:-0}
RECON_NUM_CANDIDATES=${RECON_NUM_CANDIDATES:-}
RECON_RESOLUTION=${RECON_RESOLUTION:-518}
MESH_SIMPLIFY=${MESH_SIMPLIFY:-0.75}
MESH_EVAL_SAMPLES=${MESH_EVAL_SAMPLES:-12000}
MESH_EVAL_ICP_ITERS=${MESH_EVAL_ICP_ITERS:-8}
LINK_MODE=${LINK_MODE:-symlink}

SEED_TAG=${RECON_SEEDS//,/_}
if [[ "${MAX_FRAMES}" == "0" ]]; then
  CASE_NAME=${CASE_NAME:-heimei_${TRAJECTORY_MODE}_all_s${SEED_TAG}}
else
  CASE_NAME=${CASE_NAME:-heimei_${TRAJECTORY_MODE}${MAX_FRAMES}_s${SEED_TAG}}
fi

echo "[heimei_recon_eval] prepare ${CASE_NAME}"
"${PYTHON_BIN}" -u CoarseModel/reconviagen_coarse_eval/prepare_heimei_colmap_balanced.py \
  --dataset_dir "${DATASET_DIR}" \
  --output_root "${OUTPUT_ROOT}" \
  --case_name "${CASE_NAME}" \
  --max_frames "${MAX_FRAMES}" \
  --link_mode "${LINK_MODE}" \
  --trajectory_mode "${TRAJECTORY_MODE}" \
  --arc_span_deg "${ARC_SPAN_DEG}" \
  --arc_center_fraction "${ARC_CENTER_FRACTION}" \
  --random_seed "${RANDOM_SEED}"

WORKSPACE_DATASET="${OUTPUT_ROOT}/workspace/datasets/${CASE_NAME}"

echo "[heimei_recon_eval] recon dataset=${WORKSPACE_DATASET}"
RECON_CMD=(
  "${PYTHON_BIN}" -u CoarseModel/reconviagen_coarse_eval/run_reconviagen_mesh.py
  --dataset_dir "${WORKSPACE_DATASET}"
  --output_root "${OUTPUT_ROOT}"
  --python_bin "${PYTHON_BIN}"
  --seeds "${RECON_SEEDS}"
  --resolution "${RECON_RESOLUTION}"
  --mesh_simplify "${MESH_SIMPLIFY}"
)
if [[ -n "${RECON_NUM_CANDIDATES}" ]]; then
  RECON_CMD+=(--num_candidates "${RECON_NUM_CANDIDATES}")
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
"${RECON_CMD[@]}"

echo "[heimei_recon_eval] mesh_eval"
"${PYTHON_BIN}" -u CoarseModel/reconviagen_coarse_eval/run_pipeline.py \
  --dataset_dir "${DATASET_DIR}" \
  --case_name "${CASE_NAME}" \
  --output_root "${OUTPUT_ROOT}" \
  --stages mesh_eval \
  --mesh_eval_reference dataset_model \
  --mesh_eval_samples "${MESH_EVAL_SAMPLES}" \
  --mesh_eval_icp_iters "${MESH_EVAL_ICP_ITERS}"

echo "[heimei_recon_eval] prepared ${OUTPUT_ROOT}/runs/${CASE_NAME}/prepared_sample.json"
echo "[heimei_recon_eval] selected ${OUTPUT_ROOT}/runs/${CASE_NAME}/selected_frames.json"
echo "[heimei_recon_eval] recon ${OUTPUT_ROOT}/runs/${CASE_NAME}/recon_generation_report.json"
echo "[heimei_recon_eval] mesh ${OUTPUT_ROOT}/mesh_quality/${CASE_NAME}/mesh_quality_report.json"
