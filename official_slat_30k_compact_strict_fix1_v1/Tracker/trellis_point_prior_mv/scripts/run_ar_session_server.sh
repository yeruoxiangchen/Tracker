#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zjr/Tracker}"
PY="${PY:-/home/zjr/anaconda3/envs/foundpose/bin/python}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5000}"
TPP_GENERATE_ENABLED="${TPP_GENERATE_ENABLED:-0}"

cd "${ROOT}"

echo "[ar_session_server] server=trellis_point_prior_mv/server.py"
echo "[ar_session_server] host=${HOST} port=${PORT}"
echo "[ar_session_server] generate_enabled=${TPP_GENERATE_ENABLED} (default 0: capture/save only)"
echo "[ar_session_server] note: do not run CoarseModel/connect/server.py on the same port at the same time"

AR_SESSION_SERVER_HOST="${HOST}" \
AR_SESSION_SERVER_PORT="${PORT}" \
TPP_GENERATE_ENABLED="${TPP_GENERATE_ENABLED}" \
"${PY}" -u trellis_point_prior_mv/server.py
