#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
BLENDER=${BLENDER:-/tmp/blender-3.0.1-linux-x64/blender}
BASE=${BASE_OMNI200_ROOT:-/data/zjr/omniobject3d_reconviagen_style_omni200_20cat_render4_20260821_v1}
DATA=${UNIFORM4_DATA_ROOT:-/data/zjr/omniobject3d_reconviagen_style_omni200_20cat_uniform4_idx0_6_12_18_20260821_v1}
OUT=${OUTPUT_ROOT:-/data/zjr/omniobject3d_omni200_uniform4_ss30k_slat30k_step30k_metrics_seed42_4gpu1235_20260821_v1}
GPUS_CSV=${EVAL_GPUS:-1,2,3,5}

test -x "${PY}"
test -x "${BLENDER}"
test -s "${BASE}/protocol.json"
test -s "${BASE}/manifest.json"
test -s "${BASE}/report.json"

IFS=, read -r -a GPU_ARRAY <<<"${GPUS_CSV}"
if (( ${#GPU_ARRAY[@]} != 4 )); then
  echo "ERROR: uniform4 run requires exactly four GPUs; got ${GPUS_CSV}" >&2
  exit 90
fi
if [[ "$(printf '%s\n' "${GPU_ARRAY[@]}" | sort -u | wc -l)" -ne 4 ]]; then
  echo "ERROR: duplicate GPU index in ${GPUS_CSV}" >&2
  exit 91
fi

mkdir -p "${DATA}/logs" "${OUT}/logs"
export PYTHONPATH="${PWD}:${PWD}/ReconViaGen:${PWD}/ReconViaGen/wheels/vggt"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export ATTN_BACKEND=flash_attn
export SPCONV_ALGO=native
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

echo "===== P0 derive exact frozen Omni200 uniform4 protocol ====="
"${PY}" -u -m \
  pose_point_depth_mv.dataset_tools.derive_reconviagen_omni200_uniform4_protocol \
  --base_protocol "${BASE}/protocol.json" \
  --output_protocol "${DATA}/protocol.json"

PROTOCOL=${DATA}/protocol.json
test -s "${PROTOCOL}"

echo "===== P1 render fixed uniform views [0,6,12,18] on GPUs ${GPUS_CSV} ====="
pids=()
for worker in "${!GPU_ARRAY[@]}"; do
  gpu=${GPU_ARRAY[$worker]}
  log=${DATA}/logs/worker_$(printf '%02d' "${worker}")_gpu${gpu}.log
  CUDA_VISIBLE_DEVICES="${gpu}" PYOPENGL_PLATFORM=egl \
    "${PY}" -u -m \
      pose_point_depth_mv.dataset_tools.build_reconviagen_omni200_benchmark \
      worker \
      --protocol "${PROTOCOL}" \
      --worker_index "${worker}" \
      --num_workers 4 \
      --blender_path "${BLENDER}" \
      >"${log}" 2>&1 &
  pids+=("$!")
  echo "render_worker=${worker} gpu=${gpu} pid=$! log=${log}"
done

failed=0
for worker in "${!pids[@]}"; do
  if ! wait "${pids[$worker]}"; then
    echo "ERROR: render worker ${worker} failed; completed objects remain resumable" >&2
    failed=1
  fi
done
(( failed == 0 )) || exit 92

echo "===== P2 finalize fixed-uniform4 render manifest ====="
"${PY}" -u -m pose_point_depth_mv.dataset_tools.build_reconviagen_omni200_benchmark \
  finalize --protocol "${PROTOCOL}"

echo "===== P3 SS30K + SLat30K inference and CD/F-score on the same 200 objects ====="
OMNI200_ROOT="${DATA}" \
OUTPUT_ROOT="${OUT}" \
EVAL_GPUS="${GPUS_CSV}" \
PYTHON="${PY}" \
  bash pose_point_depth_mv/background_jobs/run_omni200_ss30k_slat30k_metrics_4gpu.sh

echo "OMNI200 UNIFORM4 SS30K+SLAT30K COMPLETE: ${OUT}/aggregate_v1/report.json"
