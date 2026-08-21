#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
PYTHON=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
BUILD_GPUS=${BUILD_GPUS:-0,1,2,3,4,5,6,7}
MAX_OBJECTS=${MAX_OBJECTS:-0}

: "${SPLIT_MANIFEST:?set SPLIT_MANIFEST}"
: "${BASE_MANIFEST:?set BASE_MANIFEST}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="$PWD:$PWD/ReconViaGen:$PWD/ReconViaGen/wheels/vggt"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

mkdir -p "${OUTPUT_DIR}/logs"
IFS=, read -r -a GPU_ARRAY <<<"${BUILD_GPUS}"
WORKERS=${#GPU_ARRAY[@]}
if (( WORKERS <= 0 )); then
  echo "ERROR: BUILD_GPUS is empty" >&2
  exit 90
fi

echo "===== with-VGGT SS cache preflight ====="
"${PYTHON}" -u -m official_ss_with_vggt_perf_v1.build_cache \
  --mode preflight \
  --split_manifest "${SPLIT_MANIFEST}" \
  --base_manifest "${BASE_MANIFEST}" \
  --output_dir "${OUTPUT_DIR}" \
  --expected_selected_views 8 \
  --max_objects "${MAX_OBJECTS}"

echo "===== launch ${WORKERS} workers ====="
PIDS=()
for ((INDEX=0; INDEX<WORKERS; INDEX++)); do
  GPU=${GPU_ARRAY[$INDEX]}
  LOG="${OUTPUT_DIR}/logs/worker_$(printf '%02d' "${INDEX}")_gpu${GPU}.log"
  CUDA_VISIBLE_DEVICES="${GPU}" \
    "${PYTHON}" -u -m official_ss_with_vggt_perf_v1.build_cache \
      --mode materialize \
      --split_manifest "${SPLIT_MANIFEST}" \
      --base_manifest "${BASE_MANIFEST}" \
      --output_dir "${OUTPUT_DIR}" \
      --expected_selected_views 8 \
      --max_objects "${MAX_OBJECTS}" \
      --worker_index "${INDEX}" \
      --worker_count "${WORKERS}" \
      --device cuda:0 >"${LOG}" 2>&1 &
  PIDS+=("$!")
  echo "worker=${INDEX} gpu=${GPU} pid=${PIDS[-1]} log=${LOG}"
done

FAILED=0
for ((INDEX=0; INDEX<WORKERS; INDEX++)); do
  if ! wait "${PIDS[$INDEX]}"; then
    echo "ERROR: with-VGGT SS worker ${INDEX} failed" >&2
    FAILED=1
  fi
done
if (( FAILED != 0 )); then
  echo "Outputs are preserved; rerun this same command to resume." >&2
  exit 91
fi

echo "===== finalize immutable manifest ====="
"${PYTHON}" -u -m official_ss_with_vggt_perf_v1.build_cache \
  --mode finalize \
  --split_manifest "${SPLIT_MANIFEST}" \
  --base_manifest "${BASE_MANIFEST}" \
  --output_dir "${OUTPUT_DIR}" \
  --expected_selected_views 8 \
  --max_objects "${MAX_OBJECTS}"

echo "WITH-VGGT OFFICIAL SS CACHE COMPLETE: ${OUTPUT_DIR}"
