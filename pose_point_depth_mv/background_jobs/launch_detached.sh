#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "usage: $0 JOB_NAME LOG_FILE PID_FILE COMMAND [ARG ...]" >&2
  exit 64
fi

JOB_NAME=$1
LOG_FILE=$2
PID_FILE=$3
shift 3

mkdir -p "$(dirname "${LOG_FILE}")" "$(dirname "${PID_FILE}")"

if [ -s "${PID_FILE}" ]; then
  OLD_PID=$(tr -dc '0-9' < "${PID_FILE}")
  if [ -n "${OLD_PID}" ] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "${JOB_NAME} is already running: pid=${OLD_PID}"
    echo "log: ${LOG_FILE}"
    exit 0
  fi
fi

printf '\n[%s] launching %s\n' "$(date --iso-8601=seconds)" "${JOB_NAME}" >> "${LOG_FILE}"
nohup setsid "$@" >> "${LOG_FILE}" 2>&1 < /dev/null &
NEW_PID=$!
printf '%s\n' "${NEW_PID}" > "${PID_FILE}"

sleep 1
if kill -0 "${NEW_PID}" 2>/dev/null; then
  echo "${JOB_NAME} started in background: pid=${NEW_PID}"
  echo "log: ${LOG_FILE}"
  echo "monitor: tail -f '${LOG_FILE}'"
  exit 0
fi

set +e
wait "${NEW_PID}"
RC=$?
set -e
echo "${JOB_NAME} exited during startup: rc=${RC}" >&2
echo "inspect: ${LOG_FILE}" >&2
exit "${RC}"
