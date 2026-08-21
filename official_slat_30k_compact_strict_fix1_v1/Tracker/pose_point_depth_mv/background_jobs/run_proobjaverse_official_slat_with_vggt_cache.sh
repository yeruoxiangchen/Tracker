#!/usr/bin/env bash
set -euo pipefail

# Required environment:
#   SPLIT_MANIFEST, BASE_SLAT_MANIFEST, BASE_LIFTING_MANIFEST, OUTPUT_DIR
# Optional:
#   BUILD_GPUS=0,1,2,3,4,5,6,7  MAX_OBJECTS=0  PYTHON=...

PYTHON=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
BUILD_GPUS=${BUILD_GPUS:-0}
MAX_OBJECTS=${MAX_OBJECTS:-0}
PRETRAINED=${PRETRAINED:-Stable-X/trellis-vggt-v0-2}
VGGT_REPO=${VGGT_REPO:-Stable-X/vggt-object-v0-1}

: "${SPLIT_MANIFEST:?SPLIT_MANIFEST is required}"
: "${BASE_SLAT_MANIFEST:?BASE_SLAT_MANIFEST is required}"
: "${BASE_LIFTING_MANIFEST:?BASE_LIFTING_MANIFEST is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

IFS=, read -r -a GPU_ARRAY <<<"${BUILD_GPUS}"
WORKER_COUNT=${#GPU_ARRAY[@]}
if [ "${WORKER_COUNT}" -le 0 ]; then
  echo "ERROR: BUILD_GPUS is empty" >&2
  exit 90
fi

mkdir -p "${OUTPUT_DIR}/logs"

COMMON=(
  --split_manifest "${SPLIT_MANIFEST}"
  --base_slat_manifest "${BASE_SLAT_MANIFEST}"
  --base_lifting_manifest "${BASE_LIFTING_MANIFEST}"
  --output_dir "${OUTPUT_DIR}"
  --pretrained "${PRETRAINED}"
  --vggt_repo "${VGGT_REPO}"
  --expected_selected_views 8
  --max_objects "${MAX_OBJECTS}"
)

echo "===== with-VGGT cache preflight ====="
"${PYTHON}" -u -m \
  pose_point_depth_mv.prepare_proobjaverse_official_slat_with_vggt_sidecar \
  --mode preflight "${COMMON[@]}" \
  | tee "${OUTPUT_DIR}/logs/preflight.log"

echo "===== launch ${WORKER_COUNT} cache workers ====="
PIDS=()
for ((WORKER_INDEX=0; WORKER_INDEX<WORKER_COUNT; WORKER_INDEX++)); do
  GPU=${GPU_ARRAY[WORKER_INDEX]}
  LOG=$(printf '%s/logs/worker_%02d_gpu%s.log' \
    "${OUTPUT_DIR}" "${WORKER_INDEX}" "${GPU}")
  CUDA_VISIBLE_DEVICES="${GPU}" \
    "${PYTHON}" -u -m \
      pose_point_depth_mv.prepare_proobjaverse_official_slat_with_vggt_sidecar \
      --mode materialize \
      --device cuda:0 \
      --worker_index "${WORKER_INDEX}" \
      --worker_count "${WORKER_COUNT}" \
      "${COMMON[@]}" \
      >"${LOG}" 2>&1 &
  PID=$!
  PIDS+=("${PID}")
  echo "worker=${WORKER_INDEX} gpu=${GPU} pid=${PID} log=${LOG}"
done

RC=0
for ((WORKER_INDEX=0; WORKER_INDEX<WORKER_COUNT; WORKER_INDEX++)); do
  if ! wait "${PIDS[WORKER_INDEX]}"; then
    echo "ERROR: with-VGGT cache worker ${WORKER_INDEX} failed" >&2
    RC=1
  fi
done
if [ "${RC}" -ne 0 ]; then
  echo "Worker outputs are preserved. Correct the failure and rerun the same command." >&2
  exit 91
fi

echo "===== finalize immutable paired manifests ====="
"${PYTHON}" -u -m \
  pose_point_depth_mv.prepare_proobjaverse_official_slat_with_vggt_sidecar \
  --mode finalize "${COMMON[@]}" \
  | tee "${OUTPUT_DIR}/logs/finalize.log"

echo "WITH-VGGT CACHE COMPLETE: ${OUTPUT_DIR}"
