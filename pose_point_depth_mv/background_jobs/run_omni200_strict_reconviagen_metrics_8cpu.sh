#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
BENCH=/data/zjr/omniobject3d_reconviagen_style_omni200_20cat_render4_20260821_v1/manifest.json
ROOT=/data/zjr/omniobject3d_omni200_strict_reconviagen_seed42_8gpu_20260821_v1
INFERENCE=${ROOT}/inference_aggregate_v1/report.json
CURRENT=/data/zjr/omniobject3d_omni200_ss30k_slat30k_step30k_metrics_seed42_20260821_v1/aggregate_v1/report.json
WORKERS=${ROOT}/metrics_8cpu_v1/workers
FINAL=${ROOT}/metrics_aggregate_v1
COUNT=${METRIC_WORKERS:-8}

test -s "${BENCH}"
test -s "${INFERENCE}"
test -s "${CURRENT}"
mkdir -p "${ROOT}/metrics_8cpu_v1/logs" "${WORKERS}"
export PYTHONPATH="${PWD}:${PWD}/ReconViaGen"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

pids=()
for ((worker=0; worker<COUNT; worker++)); do
  output=${WORKERS}/worker_$(printf '%02d' "${worker}")
  log=${ROOT}/metrics_8cpu_v1/logs/worker_$(printf '%02d' "${worker}").log
  "${PY}" -u -m pose_point_depth_mv.evaluate_omni200_strict_reconviagen_metrics \
    worker \
    --benchmark_manifest "${BENCH}" \
    --inference_aggregate "${INFERENCE}" \
    --output_dir "${output}" \
    --worker_index "${worker}" \
    --worker_count "${COUNT}" \
    --surface_points 100000 \
    --fscore_radius 0.1 \
    --seed 42 >"${log}" 2>&1 &
  pids+=("$!")
  echo "metric_worker=${worker} pid=$! log=${log}"
done

failed=0
for worker in "${!pids[@]}"; do
  if ! wait "${pids[$worker]}"; then
    echo "ERROR: metric worker ${worker} failed" >&2
    failed=1
  fi
done
(( failed == 0 )) || exit 91

"${PY}" -u -m pose_point_depth_mv.evaluate_omni200_strict_reconviagen_metrics \
  aggregate \
  --benchmark_manifest "${BENCH}" \
  --inference_aggregate "${INFERENCE}" \
  --current_report "${CURRENT}" \
  --workers_root "${WORKERS}" \
  --output_dir "${FINAL}" \
  --expected_workers "${COUNT}" \
  --surface_points 100000 \
  --fscore_radius 0.1 \
  --seed 42

echo "OMNI200 STRICT RECONVIAGEN METRICS COMPLETE: ${FINAL}/report.json"
