#!/usr/bin/env bash
set -u

cd /home/zjr/Tracker

PY=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
DATA=${UNIFORM4_DATA_ROOT:-/data/zjr/omniobject3d_reconviagen_style_omni200_20cat_uniform4_idx0_6_12_18_20260821_v1}
OUT=${OUTPUT_ROOT:-/data/zjr/omniobject3d_omni200_uniform4_ss30k_slat30k_step30k_metrics_seed42_4gpu1235_20260821_v1}
MASTER=${OUT}/logs/master.log

date -u -Is
echo "uniform4_indices=0,6,12,18"
echo "============================================================"
if [[ ! -s "${DATA}/protocol.json" ]]; then
  echo "stage=waiting_for_healthy_idle_gpu_or_protocol_derivation"
else
  "${PY}" -m pose_point_depth_mv.dataset_tools.build_reconviagen_omni200_benchmark \
    status --protocol "${DATA}/protocol.json" 2>&1 || true
fi
echo "------------------------------------------------------------"
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
echo "exact_model_o_runtime=${runtime}/200"
echo "dino_model_inputs=${dino}/200"
echo "ss30k_slat30k_meshes=${meshes}/200"
echo "metric_objects=${metrics}/200"
echo "metric_worker_reports=${worker_reports}/4"
if [[ -s "${OUT}/aggregate_v1/report.json" ]]; then
  echo "aggregate=COMPLETE"
  cat "${OUT}/aggregate_v1/summary.txt"
else
  echo "aggregate=pending"
fi
echo "------------------------------------------------------------"
if tmux has-session -t omni200uniform4 2>/dev/null; then
  echo "tmux=omni200uniform4 RUNNING_OR_WAITING"
else
  echo "tmux=omni200uniform4 EXITED_OR_COMPLETE"
fi
tail -n 8 "${MASTER}" 2>/dev/null || true
echo "------------------------------------------------------------"
timeout 5s nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi unavailable or timed out"
