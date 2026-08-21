#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

RUN=${OBJAVERSE2K_SLAT_RUN:-/data/zjr/objaverse2k_no_vggt_slat_20260811_v1}
GPUS=${OBJAVERSE2K_SLAT_GPUS:-0,5,6,7}
POLL_SECONDS=${OBJAVERSE2K_SLAT_P2_POLL_SECONDS:-60}
P2_EXIT=${RUN}/logs/prepare_objaverse2k_slat_4gpu.exit_code
P2_STATE=${RUN}/logs/prepare_objaverse2k_slat_4gpu.state
P2_LOCK=${RUN}/logs/prepare_objaverse2k_slat_4gpu.lock
P3_SCRIPT=/home/zjr/Tracker/pose_point_depth_mv/background_jobs/run_objaverse2k_no_vggt_slat_4gpu.sh
P3_CHECKPOINT=${RUN}/slat_objaverse2135_step2000_seed42_4gpu_v1/checkpoints/step_000400.pt
STATE=${RUN}/logs/p2_to_p3_step400.state
EXIT_CODE=${RUN}/logs/p2_to_p3_step400.exit_code
LOCK=${RUN}/logs/p2_to_p3_step400.lock

if [[ ! "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "OBJAVERSE2K_SLAT_P2_POLL_SECONDS must be a positive integer" >&2
  exit 96
fi

mkdir -p "${RUN}/logs"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "P2-to-P3 step400 handoff is already running" >&2
  exit 99
fi

PHASE=starting
finish() {
  RC=$?
  trap - EXIT
  if [ "${RC}" -eq 0 ]; then
    PHASE=complete
  else
    PHASE=failed
  fi
  printf 'finished_at=%s phase=%s rc=%s\n' \
    "$(date --iso-8601=seconds)" "${PHASE}" "${RC}" >"${STATE}"
  printf '%s\n' "${RC}" >"${EXIT_CODE}"
  exit "${RC}"
}
trap finish EXIT
rm -f "${EXIT_CODE}"

log_p2_progress() {
  local LOG
  printf '[%s] P2 state: ' "$(date --iso-8601=seconds)"
  if [ -s "${P2_STATE}" ]; then
    tr '\n' ' ' <"${P2_STATE}"
  else
    printf 'missing'
  fi
  printf '\n'
  for LOG in "${RUN}"/logs/cache_train_worker_*.log; do
    [ -f "${LOG}" ] || continue
    printf '  %s: ' "$(basename "${LOG}")"
    sed -n '/\[direct_slat_cache:cond\]/h; ${x;p;}' "${LOG}"
  done
}

PHASE=waiting_p2
printf 'started_at=%s phase=%s poll_seconds=%s gpus=%s\n' \
  "$(date --iso-8601=seconds)" "${PHASE}" "${POLL_SECONDS}" "${GPUS}" >"${STATE}"

MISSING_POLLS=0
while true; do
  if [ -s "${P2_EXIT}" ]; then
    P2_RC=$(tr -d '[:space:]' <"${P2_EXIT}")
    if [[ ! "${P2_RC}" =~ ^[0-9]+$ ]]; then
      echo "invalid P2 exit code: ${P2_RC}" >&2
      exit 97
    fi
    if [ "${P2_RC}" -ne 0 ]; then
      echo "P2 failed with rc=${P2_RC}; P3 will not start" >&2
      exit 2
    fi
    echo "P2 completed successfully; validating P3 prerequisites"
    break
  fi

  if [ ! -e "${P2_LOCK}" ]; then
    MISSING_POLLS=$((MISSING_POLLS + 1))
  else
    exec {P2_FD}<>"${P2_LOCK}"
    if flock -n "${P2_FD}"; then
      flock -u "${P2_FD}"
      MISSING_POLLS=$((MISSING_POLLS + 1))
    else
      MISSING_POLLS=0
    fi
    exec {P2_FD}>&-
  fi

  if [ "${MISSING_POLLS}" -ge 3 ]; then
    echo "P2 is not running and has no exit code; P3 will not start" >&2
    exit 97
  fi

  log_p2_progress
  sleep "${POLL_SECONDS}"
done

for REQUIRED in \
  "${RUN}/slat_cache_train_seed42_merged_v1/manifest.json" \
  "${RUN}/slat_cache_train_seed42_merged_v1/_OBJAVERSE2K_SLAT_CACHE_MERGE_COMPLETE.json" \
  "${RUN}/slat_cache_dev64_seed424344_merged_v1/manifest.json" \
  "${RUN}/slat_cache_dev64_seed424344_merged_v1/_OBJAVERSE2K_SLAT_CACHE_MERGE_COMPLETE.json" \
  "${RUN}/slat_target_decoder_audit_dev32_v1/report.json" \
  "${P3_SCRIPT}"; do
  test -s "${REQUIRED}"
done

PHASE=running_p3_step400
printf 'updated_at=%s phase=%s gpus=%s\n' \
  "$(date --iso-8601=seconds)" "${PHASE}" "${GPUS}" >"${STATE}"
echo "starting P3 on GPUs ${GPUS}, run_until_step=400"

OBJAVERSE2K_SLAT_RUN="${RUN}" \
OBJAVERSE2K_SLAT_GPUS="${GPUS}" \
OBJAVERSE2K_SLAT_RUN_UNTIL_STEP=400 \
  /usr/bin/bash "${P3_SCRIPT}"

test -s "${P3_CHECKPOINT}"
echo "P3 step400 checkpoint ready: ${P3_CHECKPOINT}"
