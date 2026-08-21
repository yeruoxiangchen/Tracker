#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
BLENDER=${BLENDER:-/tmp/blender-3.0.1-linux-x64/blender}
CATEGORIES=${CATEGORIES_ROOT:-/data/zjr/OmniObject3D/raw_scans_extracted_omni215_v1/categories}
OUT=${OUTPUT_ROOT:-/data/zjr/omniobject3d_reconviagen_style_omni200_20cat_render4_20260821_v1}
GPUS_CSV=${RENDER_GPUS:-0,1,2,3,5,6,7}
SEED=${SELECTION_SEED:-20260821}

test -x "${PY}"
test -x "${BLENDER}"
test -d "${CATEGORIES}"
mkdir -p "${OUT}/logs"

echo "===== freeze Omni200/20 scientific protocol ====="
"${PY}" -u -m pose_point_depth_mv.dataset_tools.build_reconviagen_omni200_benchmark \
  freeze \
  --categories_root "${CATEGORIES}" \
  --output_root "${OUT}" \
  --blender_path "${BLENDER}" \
  --seed "${SEED}" \
  --category_count 20 \
  --objects_per_category 10 \
  --image_size 512 \
  --focal_ratio 1.25 \
  --camera_radius 2.0 \
  --canonical_margin 0.9 \
  --blender_engine CYCLES \
  --blender_samples 16 \
  --blender_cycles_device CUDA

PROTOCOL=${OUT}/protocol.json
test -s "${PROTOCOL}"
IFS=, read -r -a GPU_ARRAY <<<"${GPUS_CSV}"
WORKERS=${#GPU_ARRAY[@]}
if (( WORKERS < 1 )); then
  echo "ERROR: RENDER_GPUS is empty" >&2
  exit 90
fi

echo "===== launch ${WORKERS} render workers on GPUs ${GPUS_CSV} ====="
pids=()
for worker in "${!GPU_ARRAY[@]}"; do
  gpu=${GPU_ARRAY[$worker]}
  log=${OUT}/logs/worker_$(printf '%02d' "${worker}")_gpu${gpu}.log
  CUDA_VISIBLE_DEVICES="${gpu}" \
  PYOPENGL_PLATFORM=egl \
  "${PY}" -u -m pose_point_depth_mv.dataset_tools.build_reconviagen_omni200_benchmark \
    worker \
    --protocol "${PROTOCOL}" \
    --worker_index "${worker}" \
    --num_workers "${WORKERS}" \
    --blender_path "${BLENDER}" \
    >"${log}" 2>&1 &
  pids+=("$!")
  echo "worker=${worker} gpu=${gpu} pid=$! log=${log}"
done

failed=0
for worker in "${!pids[@]}"; do
  if ! wait "${pids[$worker]}"; then
    echo "ERROR: worker ${worker} failed" >&2
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "At least one render worker failed; completed object reports are resumable." >&2
  exit 91
fi

echo "===== finalize frozen render manifest ====="
"${PY}" -u -m pose_point_depth_mv.dataset_tools.build_reconviagen_omni200_benchmark \
  finalize \
  --protocol "${PROTOCOL}"

echo "OMNI200/20 RENDER DATASET COMPLETE: ${OUT}"
