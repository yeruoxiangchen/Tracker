#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
BLENDER=${BLENDER:-/tmp/blender-3.0.1-linux-x64/blender}
SOURCE=${DORA_SOURCE_ROOT:-/data/zjr/Dora-Bench-256_20260821_v1}
DATA=${DORA300_ROOT:-/data/zjr/dorabench_reconviagen_style_dora300_trellis40_input0_9_19_29_20260821_v1}
OUT=${OUTPUT_ROOT:-/data/zjr/dorabench_dora300_ss30k_slat30k_step30k_metrics_seed42_trellis40_input0_9_19_29_7gpu_20260821_v1}
GPUS_CSV=${DORA_GPUS:-0,1,2,3,5,6,7}
SEED=${DORA_SELECTION_SEED:-20260821}

test -x "${PY}"
test -x "${BLENDER}"
test -s "${SOURCE}/DOWNLOAD_COMPLETE.txt"
test -s "${SOURCE}/dora-bench-256.zip"
mkdir -p "${DATA}/logs" "${OUT}/logs"
export PYTHONPATH="${PWD}:${PWD}/ReconViaGen:${PWD}/ReconViaGen/wheels/vggt"

echo "===== P0 freeze Dora300 and extract only the selected 300 meshes ====="
"${PY}" -u -m pose_point_depth_mv.dataset_tools.build_reconviagen_dorabench300_benchmark \
  freeze \
  --source_root "${SOURCE}" \
  --output_root "${DATA}" \
  --blender_path "${BLENDER}" \
  --seed "${SEED}" \
  --object_count 300 \
  --image_size 1024 \
  --canonical_margin 1.0 \
  --blender_engine CYCLES \
  --blender_samples 128 \
  --blender_cycles_device CUDA

PROTOCOL=${DATA}/protocol.json
test -s "${PROTOCOL}"
IFS=, read -r -a GPU_ARRAY <<<"${GPUS_CSV}"
WORKERS=${#GPU_ARRAY[@]}
if (( WORKERS != 7 )); then
  echo "ERROR: Dora run requires seven GPUs excluding the phone-service GPU; got ${GPUS_CSV}" >&2
  exit 90
fi

wait_gpu_set() {
  local wanted=$1
  while true; do
    local busy=0
    while IFS=, read -r index used; do
      index=${index// /}
      used=${used// /}
      case ",${wanted}," in
        *,"${index}",*)
          if (( used > 512 )); then
            busy=1
            echo "[$(date -u +%Y-%m-%dT%H:%M:%S%z)] waiting: gpu=${index} used=${used}MiB"
          fi
          ;;
      esac
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    (( busy == 0 )) && break
    sleep 30
  done
}

pids=()
worker_labels=()
launch_render_worker() {
  local worker=$1
  local gpu=$2
  log=${DATA}/logs/worker_$(printf '%02d' "${worker}")_gpu${gpu}.log
  CUDA_VISIBLE_DEVICES="${gpu}" PYOPENGL_PLATFORM=egl \
    "${PY}" -u -m pose_point_depth_mv.dataset_tools.build_reconviagen_dorabench300_benchmark \
      worker \
      --protocol "${PROTOCOL}" \
      --worker_index "${worker}" \
      --num_workers "${WORKERS}" \
      --blender_path "${BLENDER}" \
      >"${log}" 2>&1 &
  pids+=("$!")
  worker_labels+=("${worker}")
  echo "render_worker=${worker} gpu=${gpu} pid=$! log=${log}"
}

echo "===== P1a start Dora workers on currently free GPUs 0,6,7 ====="
wait_gpu_set "0,6,7"
for worker in "${!GPU_ARRAY[@]}"; do
  gpu=${GPU_ARRAY[$worker]}
  case ",0,6,7," in
    *,"${gpu}",*) launch_render_worker "${worker}" "${gpu}" ;;
  esac
done

echo "===== P1b wait until Omni uniform4 releases GPUs 1,2,3,5 ====="
while tmux has-session -t omni200uniform4 2>/dev/null; do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%S%z)] Dora workers 0/6/7 active; waiting for Omni GPUs 1,2,3,5"
  sleep 30
done
wait_gpu_set "1,2,3,5"
echo "===== P1c add remaining Dora workers on released GPUs 1,2,3,5 ====="
for worker in "${!GPU_ARRAY[@]}"; do
  gpu=${GPU_ARRAY[$worker]}
  case ",1,2,3,5," in
    *,"${gpu}",*) launch_render_worker "${worker}" "${gpu}" ;;
  esac
done

failed=0
for slot in "${!pids[@]}"; do
  if ! wait "${pids[$slot]}"; then
    echo "ERROR: Dora render worker ${worker_labels[$slot]} failed; completed objects are resumable" >&2
    failed=1
  fi
done
(( failed == 0 )) || exit 91

echo "===== P2 finalize immutable Dora300 render manifest ====="
"${PY}" -u -m pose_point_depth_mv.dataset_tools.build_reconviagen_dorabench300_benchmark \
  finalize --protocol "${PROTOCOL}"

echo "===== P3 SS30K+SLat30K inference and CD/F-score ====="
DORA300_ROOT="${DATA}" OUTPUT_ROOT="${OUT}" EVAL_GPUS="${GPUS_CSV}" PYTHON="${PY}" \
  bash pose_point_depth_mv/background_jobs/run_dorabench300_ss30k_slat30k_metrics_7gpu.sh

echo "DORA-BENCH-300 PROGRAM COMPLETE: ${OUT}/aggregate_v1/report.json"
