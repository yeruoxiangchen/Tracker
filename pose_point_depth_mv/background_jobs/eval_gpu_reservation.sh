#!/usr/bin/env bash

# Source-only helpers for holding evaluation GPUs across worker/wave gaps.
# Call start_eval_gpu_reservations once after the selected GPUs pass the idle
# gate, and call stop_eval_gpu_reservations from the parent script's EXIT trap.

declare -ag EVAL_GPU_RESERVATION_PIDS=()
EVAL_GPU_RESERVATION_STARTED=0
EVAL_GPU_RESERVATION_RUN_DIR=""

start_eval_gpu_reservations() {
  local gpu_csv=$1
  local base_dir=$2
  local label=$3
  local python_bin=$4
  local project_root=$5
  local enabled=${EVAL_GPU_RESERVATION:-1}
  local hold_mib=${EVAL_GPU_HOLD_MIB:-64}
  local ready_timeout=${EVAL_GPU_HOLD_READY_TIMEOUT_SECONDS:-120}

  if [[ "${enabled}" == "0" ]]; then
    echo "GPU reservation disabled by EVAL_GPU_RESERVATION=0"
    return 0
  fi
  if (( EVAL_GPU_RESERVATION_STARTED != 0 )); then
    return 0
  fi
  [[ "${hold_mib}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: EVAL_GPU_HOLD_MIB must be a positive integer" >&2
    return 90
  }

  local holder="${project_root}/pose_point_depth_mv/background_jobs/hold_eval_gpu.py"
  test -s "${holder}" || {
    echo "ERROR: missing GPU reservation helper: ${holder}" >&2
    return 91
  }

  local -a gpus=()
  IFS=, read -r -a gpus <<<"${gpu_csv}"
  (( ${#gpus[@]} > 0 )) || {
    echo "ERROR: empty GPU list for reservation" >&2
    return 92
  }

  EVAL_GPU_RESERVATION_RUN_DIR="${base_dir}/gpu_reservations/run_${BASHPID:-$$}"
  mkdir -p "${EVAL_GPU_RESERVATION_RUN_DIR}"
  EVAL_GPU_RESERVATION_PIDS=()

  local index gpu ready log pid
  for ((index=0; index<${#gpus[@]}; index++)); do
    gpu=${gpus[$index]}
    ready="${EVAL_GPU_RESERVATION_RUN_DIR}/gpu${gpu}.ready.json"
    log="${EVAL_GPU_RESERVATION_RUN_DIR}/gpu${gpu}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${holder}" \
      --physical-gpu "${gpu}" \
      --memory-mib "${hold_mib}" \
      --label "${label}" \
      --ready-file "${ready}" >"${log}" 2>&1 &
    pid=$!
    EVAL_GPU_RESERVATION_PIDS+=("${pid}")
    echo "GPU reservation starting: gpu=${gpu} pid=${pid} log=${log}"
  done

  local deadline=$((SECONDS + ready_timeout))
  for ((index=0; index<${#gpus[@]}; index++)); do
    gpu=${gpus[$index]}
    ready="${EVAL_GPU_RESERVATION_RUN_DIR}/gpu${gpu}.ready.json"
    pid=${EVAL_GPU_RESERVATION_PIDS[$index]}
    while [[ ! -s "${ready}" ]]; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        echo "ERROR: GPU reservation process exited before ready: gpu=${gpu} pid=${pid}" >&2
        cat "${EVAL_GPU_RESERVATION_RUN_DIR}/gpu${gpu}.log" >&2 || true
        stop_eval_gpu_reservations
        return 93
      fi
      if (( SECONDS >= deadline )); then
        echo "ERROR: timeout waiting for GPU reservation: gpu=${gpu} pid=${pid}" >&2
        stop_eval_gpu_reservations
        return 94
      fi
      sleep 0.2
    done
    echo "GPU reservation ready: gpu=${gpu} pid=${pid} ready=${ready}"
  done
  EVAL_GPU_RESERVATION_STARTED=1
}

stop_eval_gpu_reservations() {
  local pid
  for pid in "${EVAL_GPU_RESERVATION_PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${EVAL_GPU_RESERVATION_PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  if (( ${#EVAL_GPU_RESERVATION_PIDS[@]} > 0 )); then
    echo "GPU reservations released: ${EVAL_GPU_RESERVATION_PIDS[*]}"
  fi
  EVAL_GPU_RESERVATION_PIDS=()
  EVAL_GPU_RESERVATION_STARTED=0
}
