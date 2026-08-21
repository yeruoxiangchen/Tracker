#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker
PY=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
DATA=${DORA300_ROOT:-/data/zjr/dorabench_reconviagen_style_dora300_trellis40_input0_9_19_29_20260821_v1}
OUT=${OUTPUT_ROOT:-/data/zjr/dorabench_dora300_ss30k_slat30k_step30k_metrics_seed42_trellis40_input0_9_19_29_7gpu_20260821_v1}

date -u +%Y-%m-%dT%H:%M:%S%z
echo "============================================================"
if [[ -s "${DATA}/protocol.json" ]]; then
  rendered=$( (find "${DATA}/objects" -mindepth 2 -maxdepth 2 -name report.json -type f 2>/dev/null || true) | wc -l )
  final=pending
  [[ -s "${DATA}/report.json" ]] && final=COMPLETE
  echo "stage=render_or_later"
  echo "render_objects=${rendered}/300"
  echo "render_images=$((rendered * 4))/1200"
  echo "render_manifest=${final}"
else
  extracted=$( (find "${DATA}/source_meshes" -name mesh.obj -type f 2>/dev/null || true) | wc -l )
  echo "stage=freeze_or_extract selected_meshes=${extracted}/300"
fi

runtime=0
[[ -s "${OUT}/00_exact_model_o_runtime/runtime_input_manifest.json" ]] && runtime=300
model_inputs=$( (find "${OUT}/workers" -path '*/01_model_inputs/objects/*/*/report.json' -type f 2>/dev/null || true) | wc -l )
meshes=$( (find "${OUT}/workers" -path '*/02_current_ss30k_slat30k/meshes/*/*/seed_42/result.json' -type f 2>/dev/null || true) | wc -l )
metrics=$( (find "${OUT}/workers" -path '*/03_metrics/objects/*/metric.json' -type f 2>/dev/null || true) | wc -l )
worker_reports=$( (find "${OUT}/workers" -path '*/03_metrics/metrics_report.json' -type f 2>/dev/null || true) | wc -l )
aggregate=pending
[[ -s "${OUT}/aggregate_v1/report.json" ]] && aggregate=COMPLETE
failure_aware_aggregate=pending
[[ -s "${OUT}/aggregate_failure_aware_v1/report.json" ]] \
  && failure_aware_aggregate=COMPLETE_299_VALID_1_REGISTERED_FAILURE
echo "------------------------------------------------------------"
echo "exact_model_o_runtime=${runtime}/300"
echo "dino_model_inputs=${model_inputs}/300"
echo "ss30k_slat30k_meshes=${meshes}/300"
echo "metric_objects=${metrics}/300"
echo "metric_worker_reports=${worker_reports}/7"
echo "aggregate=${aggregate}"
echo "failure_aware_aggregate=${failure_aware_aggregate}"
echo "------------------------------------------------------------"
if tmux has-session -t dora300eval 2>/dev/null; then
  echo "tmux=dora300eval RUNNING_OR_WAITING"
else
  echo "tmux=dora300eval EXITED_OR_COMPLETE"
fi
if tmux has-session -t dora300repair1 2>/dev/null; then
  echo "repair_tmux=dora300repair1 RUNNING"
else
  echo "repair_tmux=dora300repair1 EXITED_OR_COMPLETE"
fi
for log in "${DATA}"/logs/worker_*_gpu*.log; do
  [[ -f "${log}" ]] || continue
  line=$(grep -E '\[dora300\].*(complete|FAILED|reused)' "${log}" | tail -n 1 || true)
  [[ -n "${line}" ]] && echo "$(basename "${log}"): ${line}"
done
repair_log=${OUT}/logs/worker_01_active_point_repair_gpu1_v1.log
if [[ -f "${repair_log}" ]]; then
  line=$(grep -E '===== P[0-4]|\[real_official_slat:mesh\]|\[omni200:metric\]|REPAIR COMPLETE|Traceback|ERROR:' "${repair_log}" | tail -n 1 || true)
  [[ -n "${line}" ]] && echo "repair: ${line}"
fi
for log in "${OUT}"/logs/worker_*_gpu*.log; do
  [[ -f "${log}" ]] || continue
  line=$(grep -E '\[(real_dino_only_input|real_official_ss_slat|omni200:metric)|COMPLETE|FAILED|Traceback' "${log}" | tail -n 1 || true)
  [[ -n "${line}" ]] && echo "$(basename "${log}"): ${line}"
done
