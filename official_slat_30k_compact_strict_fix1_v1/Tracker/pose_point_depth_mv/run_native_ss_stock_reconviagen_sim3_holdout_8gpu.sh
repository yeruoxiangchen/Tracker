#!/usr/bin/env bash
set -euo pipefail

# Run eight independent mesh-evaluation workers and merge only after all rows
# have been written.  The evaluator itself is CPU-bound; CUDA_VISIBLE_DEVICES
# keeps the launcher compatible with GPU-isolated hosts and leaves one worker
# per requested card.

PYTHON_BIN="${PYTHON_BIN:-/home/zjr/anaconda3/envs/reconviagen/bin/python}"
RUN_DIR="${1:?usage: $0 RUN_DIR [GPU_CSV]}"
GPU_CSV="${2:-0,1,2,3,4,5,6,7}"
if [[ ! -f "${RUN_DIR}/run_config.json" ]]; then
  echo "missing run config: ${RUN_DIR}/run_config.json (run prepare first)" >&2
  exit 2
fi
IFS=',' read -r -a GPUS <<< "${GPU_CSV}"
if [[ "${#GPUS[@]}" -ne 8 ]]; then
  echo "expected exactly 8 GPU indices, got ${#GPUS[@]}: ${GPU_CSV}" >&2
  exit 2
fi

for worker_id in "${!GPUS[@]}"; do
  gpu="${GPUS[$worker_id]}"
  log="${RUN_DIR}/worker_${worker_id}.log"
  mkdir -p "${RUN_DIR}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u -m \
    pose_point_depth_mv.evaluate_native_ss_stock_reconviagen_sim3_holdout worker \
    --run_dir "${RUN_DIR}" --worker_id "${worker_id}" --worker_count 8 --resume \
    >"${log}" 2>&1 &
  echo "launched worker=${worker_id} gpu=${gpu} log=${log}"
done

status=0
for pid in $(jobs -p); do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "one or more workers failed; inspect ${RUN_DIR}/worker_*.log" >&2
  exit "${status}"
fi

"${PYTHON_BIN}" -u -m pose_point_depth_mv.evaluate_native_ss_stock_reconviagen_sim3_holdout \
  merge --run_dir "${RUN_DIR}"
