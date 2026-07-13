#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

GPU=${GPU:-1}
PYTHON_BIN=${PYTHON_BIN:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
DATASET_DIR=${DATASET_DIR:-/home/zjr/Tracker/CoarseModel/datasets/heimei}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/zjr/Tracker/CoarseModel/reconviagen_coarse_eval/outputs}
PRESET=${PRESET:-quick}
RUN_RECON=${RUN_RECON:-1}
RUN_MESH_EVAL=${RUN_MESH_EVAL:-1}
RECON_SEEDS=${RECON_SEEDS:-0}
RECON_NUM_CANDIDATES=${RECON_NUM_CANDIDATES:-}
RECON_RESOLUTION=${RECON_RESOLUTION:-518}
MESH_SIMPLIFY=${MESH_SIMPLIFY:-0.75}
MESH_EVAL_SAMPLES=${MESH_EVAL_SAMPLES:-12000}
MESH_EVAL_ICP_ITERS=${MESH_EVAL_ICP_ITERS:-8}
LINK_MODE=${LINK_MODE:-symlink}
RANDOM_SEED=${RANDOM_SEED:-0}

SEED_TAG=${RECON_SEEDS//,/_}
SWEEP_NAME=${SWEEP_NAME:-heimei_coverage_${PRESET}_s${SEED_TAG}}

case "${PRESET}" in
  quick)
    SPECS=(
      "arc70_18|arc|18|70|0.50"
      "arc140_18|arc|18|140|0.50"
      "full8|balanced|8|0|0.50"
      "full18|balanced|18|0|0.50"
    )
    ;;
  counts)
    SPECS=(
      "full6|balanced|6|0|0.50"
      "full8|balanced|8|0|0.50"
      "full12|balanced|12|0|0.50"
      "full18|balanced|18|0|0.50"
      "full32|balanced|32|0|0.50"
    )
    ;;
  trajectories)
    SPECS=(
      "sorted18|sorted_first|18|0|0.50"
      "arc70_18|arc|18|70|0.50"
      "arc140_18|arc|18|140|0.50"
      "arc220_18|arc|18|220|0.50"
      "full18|balanced|18|0|0.50"
      "elev_low18|elevation_low|18|0|0.50"
      "elev_mid18|elevation_mid|18|0|0.50"
      "elev_high18|elevation_high|18|0|0.50"
    )
    ;;
  full)
    SPECS=(
      "sorted18|sorted_first|18|0|0.50"
      "arc70_12|arc|12|70|0.50"
      "arc70_18|arc|18|70|0.50"
      "arc140_18|arc|18|140|0.50"
      "arc220_18|arc|18|220|0.50"
      "full6|balanced|6|0|0.50"
      "full8|balanced|8|0|0.50"
      "full12|balanced|12|0|0.50"
      "full18|balanced|18|0|0.50"
      "full32|balanced|32|0|0.50"
      "elev_low18|elevation_low|18|0|0.50"
      "elev_mid18|elevation_mid|18|0|0.50"
      "elev_high18|elevation_high|18|0|0.50"
    )
    ;;
  *)
    echo "Unknown PRESET=${PRESET}; use quick, counts, trajectories, full" >&2
    exit 2
    ;;
esac

CASE_NAMES=()

for SPEC in "${SPECS[@]}"; do
  IFS='|' read -r TAG TRAJECTORY_MODE MAX_FRAMES ARC_SPAN_DEG ARC_CENTER_FRACTION <<< "${SPEC}"
  CASE_NAME="heimei_${SWEEP_NAME}_${TAG}"
  CASE_NAMES+=("${CASE_NAME}")

  echo "[heimei_coverage_sweep] prepare case=${CASE_NAME} mode=${TRAJECTORY_MODE} frames=${MAX_FRAMES} arc=${ARC_SPAN_DEG}"
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

  if [[ "${RUN_RECON}" == "1" ]]; then
    WORKSPACE_DATASET="${OUTPUT_ROOT}/workspace/datasets/${CASE_NAME}"
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

    echo "[heimei_coverage_sweep] recon case=${CASE_NAME}"
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
  fi

  if [[ "${RUN_RECON}" == "1" && "${RUN_MESH_EVAL}" == "1" ]]; then
    echo "[heimei_coverage_sweep] mesh_eval case=${CASE_NAME}"
    "${PYTHON_BIN}" -u CoarseModel/reconviagen_coarse_eval/run_pipeline.py \
      --dataset_dir "${DATASET_DIR}" \
      --case_name "${CASE_NAME}" \
      --output_root "${OUTPUT_ROOT}" \
      --stages mesh_eval \
      --mesh_eval_reference dataset_model \
      --mesh_eval_samples "${MESH_EVAL_SAMPLES}" \
      --mesh_eval_icp_iters "${MESH_EVAL_ICP_ITERS}"
  fi
done

CASE_NAMES_CSV=$(IFS=,; echo "${CASE_NAMES[*]}")
"${PYTHON_BIN}" -u CoarseModel/reconviagen_coarse_eval/summarize_heimei_coverage_results.py \
  --output_root "${OUTPUT_ROOT}" \
  --sweep_name "${SWEEP_NAME}" \
  --case_names "${CASE_NAMES_CSV}"

echo "[heimei_coverage_sweep] summary ${OUTPUT_ROOT}/coverage_sweeps/${SWEEP_NAME}/summary.csv"
echo "[heimei_coverage_sweep] summary ${OUTPUT_ROOT}/coverage_sweeps/${SWEEP_NAME}/summary.json"
