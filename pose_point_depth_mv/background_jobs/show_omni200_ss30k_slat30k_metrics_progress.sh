#!/usr/bin/env bash
set -euo pipefail

OUT=${OUTPUT_ROOT:-/data/zjr/omniobject3d_omni200_ss30k_slat30k_step30k_metrics_seed42_20260821_v1}

runtime=0
dino=0
meshes=0
metrics=0
worker_reports=0
[[ -d "${OUT}/00_exact_model_o_runtime/objects" ]] && runtime=$(find "${OUT}/00_exact_model_o_runtime/objects" -name report.json -type f | wc -l)
[[ -d "${OUT}/workers" ]] && dino=$(find "${OUT}/workers" -path '*/01_model_inputs/objects/*/*/report.json' -type f | wc -l)
[[ -d "${OUT}/workers" ]] && meshes=$(find "${OUT}/workers" -path '*/02_current_ss30k_slat30k/meshes/*/*.obj' -type f | wc -l)
[[ -d "${OUT}/workers" ]] && metrics=$(find "${OUT}/workers" -path '*/03_metrics/objects/*/metric.json' -type f | wc -l)
[[ -d "${OUT}/workers" ]] && worker_reports=$(find "${OUT}/workers" -path '*/03_metrics/metrics_report.json' -type f | wc -l)

date -Is
echo "exact_model_o_runtime=${runtime}/200"
echo "dino_model_inputs=${dino}/200"
echo "ss30k_slat30k_meshes=${meshes}/200"
echo "metric_objects=${metrics}/200"
echo "metric_worker_reports=${worker_reports}"
if [[ -s "${OUT}/aggregate_v1/report.json" ]]; then
  echo "aggregate=COMPLETE"
  cat "${OUT}/aggregate_v1/summary.txt"
else
  echo "aggregate=pending"
fi
echo "------------------------------------------------------------"
pgrep -af '[m]anual_mesh_reconstruction.model_inputs|[m]anual_mesh_reconstruction.current_model|[e]valuate_omni200_ss30k_slat30k' || echo "no matching evaluator process"

