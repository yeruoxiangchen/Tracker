#!/usr/bin/env bash
set -euo pipefail

echo "ERROR: this serial GPU0/3 queue is retired." >&2
echo "Run with-VGGT on GPU0,3 and 30K SLat on GPU4,5,6,7 as two parallel tmux jobs." >&2
exit 90

# One continuous GPU0/GPU3 reservation around two sequential evaluations:
#   1. with-VGGT VSS/SLat P10--P17 (completed P10 is reused),
#   2. 30K no-VGGT SLat 10K/30K/60K/70K compatibility trajectory.
# The low-memory holders remain alive across every worker/wave boundary and
# across the hand-off between the two tasks.

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
PYTHON=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
EVAL_GPUS=${EVAL_GPUS:-0,3}
POLL_SECONDS=${POLL_SECONDS:-20}
MAX_IDLE_MEMORY_MIB=${MAX_IDLE_MEMORY_MIB:-1024}

SS_ROOT=${SS_ROOT:-/data/zjr/proobjaverse_official_native_ss_train2000_with_vggt_20260817_v1}
SLAT30K_ROOT=${SLAT30K_ROOT:-/data/zjr/proobjaverse_official_slat_train29861_20260817_v1}
QUEUE_STATUS=${QUEUE_STATUS:-${SS_ROOT}/logs/gpu03_posttrain_then_slat30k_eval_v1.status}
QUEUE_EXIT_CODE=${QUEUE_EXIT_CODE:-${SS_ROOT}/logs/gpu03_posttrain_then_slat30k_eval_v1.exit_code}
RESERVATION_ROOT=${RESERVATION_ROOT:-${SS_ROOT}/logs/gpu03_posttrain_then_slat30k_reservation_v1}

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/pose_point_depth_mv/background_jobs/eval_gpu_reservation.sh"
IFS=, read -r -a GPU_ARRAY <<<"${EVAL_GPUS}"
if (( ${#GPU_ARRAY[@]} != 2 )); then
  echo "ERROR: shared evaluation queue requires exactly two GPUs" >&2
  exit 90
fi
mkdir -p "${SS_ROOT}/logs" "${SLAT30K_ROOT}/logs"

CURRENT_STAGE=preflight
write_status() {
  local state=$1 temporary="${QUEUE_STATUS}.tmp.$$"
  {
    printf 'state=%s\n' "${state}"
    printf 'stage=%s\n' "${CURRENT_STAGE}"
    printf 'gpus=%s\n' "${EVAL_GPUS}"
    printf 'time_utc=%s\n' "$(date -u -Is)"
  } >"${temporary}"
  mv "${temporary}" "${QUEUE_STATUS}"
}
finish() {
  local rc=$?
  trap - EXIT
  stop_eval_gpu_reservations
  [[ ${rc} -eq 0 ]] && write_status PASS || write_status FAIL
  printf '%s\n' "${rc}" >"${QUEUE_EXIT_CODE}"
  echo "===== GPU03 shared eval queue exit=${rc} stage=${CURRENT_STAGE} ====="
  exit "${rc}"
}
trap finish EXIT

wait_for_selected_gpus() {
  local idle gpu used
  CURRENT_STAGE=wait_for_gpu03
  write_status WAITING
  while :; do
    idle=1
    for gpu in "${GPU_ARRAY[@]}"; do
      used=$(nvidia-smi -i "${gpu}" --query-gpu=memory.used \
        --format=csv,noheader,nounits | awk 'NR == 1 {print int($1)}')
      (( used <= MAX_IDLE_MEMORY_MIB )) || idle=0
    done
    (( idle == 1 )) && return 0
    echo "[$(date -u -Is)] waiting for shared eval GPUs=${EVAL_GPUS}"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits || true
    sleep "${POLL_SECONDS}"
  done
}

wait_for_selected_gpus
start_eval_gpu_reservations \
  "${EVAL_GPUS}" "${RESERVATION_ROOT}" \
  gpu03_posttrain_then_slat30k "${PYTHON}" "${PROJECT_ROOT}"

CURRENT_STAGE=with_vggt_P10_P17
write_status RUNNING
EVAL_GPUS="${EVAL_GPUS}" EVAL_GPU_RESERVATION=0 \
  PROJECT_ROOT="${PROJECT_ROOT}" PYTHON="${PYTHON}" \
  bash "${PROJECT_ROOT}/official_ss_with_vggt_perf_v1/run_source_posttrain_2gpu03.sh"

CURRENT_STAGE=slat30k_10k30k60k70k
write_status RUNNING
EVAL_GPUS="${EVAL_GPUS}" EVAL_GPU_RESERVATION=0 \
  PROJECT="${PROJECT_ROOT}" \
  bash "${PROJECT_ROOT}/pose_point_depth_mv/background_jobs/run_proobjaverse_slat29861_legacy_eval_2gpu03.sh"

CURRENT_STAGE=complete
write_status PASS
echo "GPU03 WITH-VGGT POSTTRAIN + SLAT30K EVALUATION QUEUE COMPLETE"
