#!/usr/bin/env bash
set -euo pipefail

OUT=${OUTPUT_ROOT:-/data/zjr/omniobject3d_omni200_strict_reconviagen_seed42_8gpu_20260821_v1}

meshes=0
results=0
worker_reports=0
[[ -d "${OUT}" ]] && meshes=$(find "${OUT}" -path '*/meshes/*/*/seed_42/mesh_reference_o.obj' -type f | wc -l)
[[ -d "${OUT}" ]] && results=$(find "${OUT}" -path '*/meshes/*/*/seed_42/result.json' -type f | wc -l)
[[ -d "${OUT}/workers" ]] && worker_reports=$(find "${OUT}/workers" -path '*/01_strict_reconviagen/inference_manifest.json' -type f | wc -l)

date -Is
echo "strict_reconviagen_meshes=${meshes}/200"
echo "strict_reconviagen_results=${results}/200"
echo "inference_worker_reports=${worker_reports}/8"
repair_reports=0
[[ -d "${OUT}/repair_worker00_3gpu067_v1" ]] && repair_reports=$(find "${OUT}/repair_worker00_3gpu067_v1" -name inference_manifest.json -type f | wc -l)
echo "worker0_repair_reports=${repair_reports}/4"
if [[ -s "${OUT}/inference_aggregate_v1/report.json" ]]; then
  echo "inference_aggregate=COMPLETE"
else
  echo "inference_aggregate=pending"
fi
echo "------------------------------------------------------------"
if [[ -d "${OUT}/logs" ]]; then
  for log in "${OUT}"/logs/worker_*.log; do
    [[ -f "${log}" ]] || continue
    printf '%s: ' "$(basename "${log}")"
    grep -E '\[real_reconviagen\]|INFERENCE COMPLETE|Traceback|ERROR' "${log}" | tail -n 1 || echo "initializing"
  done
fi
if [[ -d "${OUT}/repair_worker00_3gpu067_v1/logs" ]]; then
  for log in "${OUT}"/repair_worker00_3gpu067_v1/logs/*.log; do
    [[ -f "${log}" ]] || continue
    printf 'repair/%s: ' "$(basename "${log}")"
    grep -E '\[real_reconviagen\]|passed|Traceback|ERROR' "${log}" | tail -n 1 || echo "initializing"
  done
fi
if [[ -d "${OUT}/repair_remaining_per_object_2gpu67_v1/logs" ]]; then
  for log in "${OUT}"/repair_remaining_per_object_2gpu67_v1/logs/*.log; do
    [[ -f "${log}" ]] || continue
    printf 'final-repair/%s: ' "$(basename "${log}")"
    grep -E '\[real_reconviagen\]|attempted=|Traceback|ERROR' "${log}" | tail -n 1 || echo "initializing"
  done
fi
echo "------------------------------------------------------------"
pgrep -af '[m]anual_mesh_reconstruction.reconviagen' || echo "no matching ReconViaGen evaluator process"
