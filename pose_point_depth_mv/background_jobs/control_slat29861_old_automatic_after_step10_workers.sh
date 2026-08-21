#!/usr/bin/env bash
set -euo pipefail

# Safely prevent the already-running multi-checkpoint scheduler from advancing
# past its current step10K GPU worker wave.  A detached guardian watches the
# four atomic worker reports and terminates the old scheduler immediately after
# all four reports exist.  It does not signal the live GPU workers.

MODE=${1:?usage: $0 arm|status|finish}
ROOT=${ROOT:-/data/zjr/proobjaverse_official_slat_train29861_20260817_v1}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/eval_legacy_protocol2128_step10k30k60k_seed424344_4gpu_targetlocked_v2}
PREDICTED=${OUTPUT_ROOT}/step_010000/legacy_dev48_predicted_training_overlap
STATE=${STATE:-${ROOT}/logs/slat30k_targetv2_disarmed_parent.state}
TMUX_SESSION=${TMUX_SESSION:-slat30k_targetv2}
GUARD_SESSION=${GUARD_SESSION:-slat30k_targetv2_guard}
GUARD_LOG=${GUARD_LOG:-${ROOT}/logs/slat30k_targetv2_guard.log}
GUARD_POLL_SECONDS=${GUARD_POLL_SECONDS:-1}
WORKER_PATTERN='pose_point_depth_mv.evaluate_proobjaverse_official_native_ss_stock_slat worker.*eval_legacy_protocol2128_step10k30k60k_seed424344_4gpu_targetlocked_v2/step_010000'
STEP30_WORKER_PATTERN='pose_point_depth_mv.evaluate_proobjaverse_official_native_ss_stock_slat worker.*eval_legacy_protocol2128_step10k30k60k_seed424344_4gpu_targetlocked_v2/step_030000'

worker_pids() {
  pgrep -f "${WORKER_PATTERN}" || true
}

worker_parent() {
  local -a workers=() parents=()
  mapfile -t workers < <(worker_pids)
  (( ${#workers[@]} > 0 )) || {
    echo "ERROR: no live step10K targetlocked worker found" >&2
    return 90
  }
  mapfile -t parents < <(
    for pid in "${workers[@]}"; do
      ps -o ppid= -p "${pid}" | awk '{print $1}'
    done | sort -u
  )
  (( ${#parents[@]} == 1 )) || {
    echo "ERROR: step10K workers do not have one scheduler parent: ${parents[*]}" >&2
    return 91
  }
  printf '%s\n' "${parents[0]}"
}

validate_saved_parent() {
  test -s "${STATE}" || {
    echo "ERROR: missing disarm state ${STATE}; run arm first" >&2
    return 92
  }
  read -r parent expected_start <"${STATE}"
  [[ "${parent}" =~ ^[0-9]+$ && "${expected_start}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: malformed disarm state ${STATE}" >&2
    return 92
  }
  kill -0 "${parent}" 2>/dev/null || {
    echo "ERROR: saved scheduler parent is no longer alive: ${parent}" >&2
    return 93
  }
  current_start=$(awk '{print $22}' "/proc/${parent}/stat")
  [[ "${current_start}" == "${expected_start}" ]] || {
    echo "ERROR: scheduler PID was reused; refusing to signal ${parent}" >&2
    return 94
  }
  printf '%s\n' "${parent}"
}

report_count() {
  local count=0
  for spec in shard0_16_28 shard1_28_40 shard2_40_52 shard3_52_64; do
    [[ -s "${PREDICTED}/${spec}/report.json" ]] && count=$((count + 1))
  done
  printf '%s\n' "${count}"
}

validate_parent_command() {
  local parent=$1 command_line
  command_line=$(ps -o args= -p "${parent}")
  [[ "${command_line}" == *run_proobjaverse_slat29861_legacy_eval_4gpu.sh* ]] || {
    echo "ERROR: refusing to control unexpected parent: ${parent} ${command_line}" >&2
    return 96
  }
}

cleanup_old_holders() {
  local -a holders=()
  local pid args
  mapfile -t holders < <(
    pgrep -f 'hold_eval_gpu.py.*slat29861_4gpu_targetlocked' || true
  )
  if (( ${#holders[@]} == 0 )); then
    return 0
  fi
  echo "EXIT trap left old holders; terminating exact validated holder PIDs: ${holders[*]}"
  for pid in "${holders[@]}"; do
    args=$(ps -o args= -p "${pid}")
    [[ "${args}" == *hold_eval_gpu.py*slat29861_4gpu_targetlocked* ]] || {
      echo "ERROR: refusing to terminate unexpected holder PID ${pid}: ${args}" >&2
      return 100
    }
    kill -TERM "${pid}"
  done
  sleep 3
  for pid in "${holders[@]}"; do
    kill -0 "${pid}" 2>/dev/null && {
      echo "ERROR: old GPU holder remains alive: ${pid}" >&2
      return 100
    }
  done
}

tmux_session_exists_exact() {
  local name=$1
  tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -Fqx -- "${name}"
}

archive_finished_state() {
  local finished_state
  finished_state="${STATE}.finished_$(date -u +%Y%m%dT%H%M%SZ)"
  mv "${STATE}" "${finished_state}"
  echo "OLD AUTOMATIC SCHEDULER STOPPED AFTER STEP10K WORKERS"
  echo "No step30K task was launched by the old scheduler."
  echo "finished_state=${finished_state}"
}

reconcile_completed_stale_state() {
  local parent expected_start reports
  reports=$(report_count)
  (( reports == 4 )) || {
    echo "ERROR: only ${reports}/4 worker reports complete; cannot reconcile" >&2
    return 98
  }
  read -r parent expected_start <"${STATE}"
  [[ "${parent}" =~ ^[0-9]+$ && "${expected_start}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: malformed stale state ${STATE}" >&2
    return 92
  }
  if kill -0 "${parent}" 2>/dev/null; then
    echo "ERROR: saved scheduler parent is still alive; normal finish is required" >&2
    return 99
  fi
  if tmux_session_exists_exact "${TMUX_SESSION}"; then
    echo "ERROR: exact old tmux session remains: ${TMUX_SESSION}" >&2
    return 99
  fi
  if [[ -n "$(worker_pids)" ]]; then
    echo "ERROR: step10K worker still exists; refusing stale-state reconciliation" >&2
    return 99
  fi
  if pgrep -f "${STEP30_WORKER_PATTERN}" >/dev/null; then
    echo "ERROR: old automatic step30K worker exists; refusing reconciliation" >&2
    return 101
  fi
  cleanup_old_holders
  archive_finished_state
}

stop_scheduler_after_reports() {
  local parent reports deadline stat current_start expected_start
  reports=$(report_count)
  (( reports == 4 )) || {
    echo "ERROR: only ${reports}/4 worker reports complete; scheduler must remain alive" >&2
    return 98
  }
  parent=$(validate_saved_parent)
  validate_parent_command "${parent}"
  read -r _ expected_start <"${STATE}"

  echo "[$(date -u -Is)] 4/4 worker reports complete; stopping old scheduler parent=${parent}"
  kill -TERM "${parent}"
  # Harmless if it was never stopped; required to deliver queued TERM if an
  # earlier control attempt left it stopped.
  kill -CONT "${parent}" 2>/dev/null || true

  deadline=$((SECONDS + 15))
  while kill -0 "${parent}" 2>/dev/null; do
    stat=$(ps -o stat= -p "${parent}" 2>/dev/null | awk '{print $1}')
    [[ "${stat}" == Z* ]] && break
    (( SECONDS < deadline )) || break
    sleep 0.2
  done
  if kill -0 "${parent}" 2>/dev/null; then
    stat=$(ps -o stat= -p "${parent}" 2>/dev/null | awk '{print $1}')
    if [[ "${stat}" != Z* ]]; then
      current_start=$(awk '{print $22}' "/proc/${parent}/stat")
      [[ "${current_start}" == "${expected_start}" ]] || {
        echo "ERROR: scheduler PID changed before final stop; refusing SIGKILL" >&2
        return 99
      }
      echo "scheduler ignored TERM after completed worker wave; sending validated KILL"
      kill -KILL "${parent}"
      sleep 1
    fi
  fi

  # At this point all worker reports are durable, so closing the old pane and
  # releasing its reservation helpers cannot interrupt unfinished GPU work.
  # tmux target names otherwise allow prefix matching.  If the old session has
  # already disappeared, a non-exact target could match and kill the guardian
  # session named `${TMUX_SESSION}_guard` before it archives STATE.
  if tmux_session_exists_exact "${TMUX_SESSION}"; then
    tmux kill-session -t "=${TMUX_SESSION}" 2>/dev/null || true
  fi
  cleanup_old_holders

  if pgrep -f "${STEP30_WORKER_PATTERN}" >/dev/null; then
    echo "ERROR: an old automatic step30K worker was observed; do not start manual 30K" >&2
    return 101
  fi
  archive_finished_state
}

launch_guard() {
  local script guard_command
  if tmux has-session -t "=${GUARD_SESSION}" 2>/dev/null; then
    echo "STEP10K COMPLETION GUARD ALREADY RUNNING: ${GUARD_SESSION}"
    return 0
  fi
  script=$(readlink -f "${BASH_SOURCE[0]}")
  mkdir -p "$(dirname "${GUARD_LOG}")"
  printf -v guard_command \
    'env ROOT=%q OUTPUT_ROOT=%q STATE=%q TMUX_SESSION=%q GUARD_SESSION=%q GUARD_LOG=%q GUARD_POLL_SECONDS=%q bash %q guard >> %q 2>&1' \
    "${ROOT}" "${OUTPUT_ROOT}" "${STATE}" "${TMUX_SESSION}" \
    "${GUARD_SESSION}" "${GUARD_LOG}" "${GUARD_POLL_SECONDS}" \
    "${script}" "${GUARD_LOG}"
  tmux new-session -d -s "${GUARD_SESSION}" "${guard_command}"
  sleep 1
  if tmux has-session -t "=${GUARD_SESSION}" 2>/dev/null; then
    echo "STEP10K COMPLETION GUARD ARMED: session=${GUARD_SESSION} log=${GUARD_LOG}"
    echo "The four current workers continue; the guardian stops the old scheduler at 4/4."
    return 0
  fi
  if grep -q 'OLD AUTOMATIC SCHEDULER STOPPED AFTER STEP10K WORKERS' "${GUARD_LOG}" 2>/dev/null; then
    echo "STEP10K workers were already complete; guardian stopped the old scheduler."
    return 0
  fi
  echo "ERROR: guard session exited unexpectedly; inspect ${GUARD_LOG}" >&2
  return 102
}

case "${MODE}" in
  arm)
    if [[ -s "${STATE}" ]]; then
      # Recover the state written by the superseded SIGSTOP implementation.
      parent=$(validate_saved_parent)
      validate_parent_command "${parent}"
      echo "reusing validated scheduler state: parent=${parent}"
    else
      parent=$(worker_parent)
      validate_parent_command "${parent}"
      start=$(awk '{print $22}' "/proc/${parent}/stat")
      temporary=${STATE}.tmp.$$
      mkdir -p "$(dirname "${STATE}")"
      printf '%s %s\n' "${parent}" "${start}" >"${temporary}"
      mv "${temporary}" "${STATE}"
    fi
    launch_guard
    ;;
  status)
    parent=$(validate_saved_parent)
    printf 'scheduler_parent=%s stat=%s (guardian target; it need not be T)\n' \
      "${parent}" "$(ps -o stat= -p "${parent}" | awk '{print $1}')"
    printf 'completed_worker_reports=%s/4\n' "$(report_count)"
    if tmux has-session -t "=${GUARD_SESSION}" 2>/dev/null; then
      printf 'completion_guard=running session=%s log=%s\n' "${GUARD_SESSION}" "${GUARD_LOG}"
    else
      printf 'completion_guard=not_running log=%s\n' "${GUARD_LOG}"
    fi
    echo "live_workers:"
    worker_pids | xargs -r ps -o pid,ppid,stat,etime,args -p
    ;;
  finish)
    if [[ ! -s "${STATE}" ]]; then
      echo "ERROR: missing state ${STATE}" >&2
      exit 92
    fi
    read -r saved_parent _ <"${STATE}"
    if kill -0 "${saved_parent}" 2>/dev/null; then
      stop_scheduler_after_reports
    else
      reconcile_completed_stale_state
    fi
    ;;
  guard)
    [[ "${GUARD_POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]] || {
      echo "ERROR: GUARD_POLL_SECONDS must be a positive integer" >&2
      exit 103
    }
    parent=$(validate_saved_parent)
    validate_parent_command "${parent}"
    echo "[$(date -u -Is)] guard started parent=${parent}"
    while (( $(report_count) < 4 )); do
      validate_saved_parent >/dev/null
      if pgrep -f "${STEP30_WORKER_PATTERN}" >/dev/null; then
        echo "ERROR: automatic step30K started before guard completion" >&2
        exit 104
      fi
      sleep "${GUARD_POLL_SECONDS}"
    done
    stop_scheduler_after_reports
    ;;
  *)
    echo "ERROR: MODE must be arm, status, or finish" >&2
    exit 89
    ;;
esac
