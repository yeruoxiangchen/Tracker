#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
OUT=${OUTPUT_ROOT:-/data/zjr/dorabench_dora299_strict_reconviagen_seed42_trellis40_input0_9_19_29_8gpu_20260821_v1}
CURRENT=${CURRENT_REPORT:-/data/zjr/dorabench_dora300_ss30k_slat30k_step30k_metrics_seed42_trellis40_input0_9_19_29_7gpu_20260821_v1/aggregate_failure_aware_v1/report.json}
SUBSET=${OUT}/protocol/dora299_current_valid_subset.json
INFERENCE=${OUT}/inference_aggregate_v1/report.json
METRIC_ROOT=${OUT}/metric_workers_model_o_v2
FINAL=${OUT}/aggregate_model_o_v2
SEED=${EVAL_SEED:-42}

test -x "${PY}"
test -s "${SUBSET}"
test -s "${INFERENCE}"
test -s "${CURRENT}"
test ! -e "${METRIC_ROOT}"
test ! -e "${FINAL}"

mkdir -p "${OUT}/logs" "${METRIC_ROOT}"
export PYTHONPATH="${PWD}:${PWD}/ReconViaGen:${PWD}/ReconViaGen/wheels/vggt"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

echo "===== CPU-only Dora299 strict ReconViaGen model-O metric replay v2 ====="
pids=()
for worker in $(seq 0 7); do
  worker_out=$(printf '%s/worker_%02d' "${METRIC_ROOT}" "${worker}")
  log=$(printf '%s/logs/model_o_v2_metric_worker_%02d.log' "${OUT}" "${worker}")
  "${PY}" -u -m pose_point_depth_mv.evaluate_dorabench299_strict_reconviagen worker \
    --subset_manifest "${SUBSET}" \
    --inference_aggregate "${INFERENCE}" \
    --output_dir "${worker_out}" \
    --worker_index "${worker}" \
    --worker_count 8 \
    --surface_points 100000 \
    --fscore_radius 0.1 \
    --seed "${SEED}" >"${log}" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if (( failed != 0 )); then
  echo "ERROR: at least one model-O v2 metric worker failed" >&2
  exit 96
fi

"${PY}" -u -m pose_point_depth_mv.evaluate_dorabench299_strict_reconviagen aggregate \
  --subset_manifest "${SUBSET}" \
  --inference_aggregate "${INFERENCE}" \
  --current_report "${CURRENT}" \
  --workers_root "${METRIC_ROOT}" \
  --output_dir "${FINAL}" \
  --expected_workers 8 \
  --surface_points 100000 \
  --fscore_radius 0.1 \
  --seed "${SEED}"

echo "DORA299 STRICT RECONVIAGEN MODEL-O METRIC V2 COMPLETE: ${FINAL}/report.json"
