#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

GPUS_CSV=${EVAL_GPUS:-1,2,3,5}
POLL_SECONDS=${GPU_WAIT_POLL_SECONDS:-20}
IFS=, read -r -a GPUS <<<"${GPUS_CSV}"

echo "[$(date -u -Is)] queued uniform4 run; waiting for healthy and idle GPUs ${GPUS_CSV}"
while true; do
  if ! nvidia-smi --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1; then
    echo "[$(date -u -Is)] WAIT: NVIDIA driver query unavailable"
    sleep "${POLL_SECONDS}"
    continue
  fi
  busy=()
  for gpu in "${GPUS[@]}"; do
    pids=$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
    [[ -z "${pids}" ]] || busy+=("gpu${gpu}:$(tr '\n' ',' <<<"${pids}" | sed 's/,$//')")
  done
  if (( ${#busy[@]} )); then
    echo "[$(date -u -Is)] WAIT: selected GPUs busy: ${busy[*]}"
    sleep "${POLL_SECONDS}"
    continue
  fi
  break
done

echo "[$(date -u -Is)] GPU preflight PASS; starting uniform4 render + evaluation"
exec bash pose_point_depth_mv/background_jobs/run_omni200_uniform4_ss30k_slat30k_4gpu1234.sh
