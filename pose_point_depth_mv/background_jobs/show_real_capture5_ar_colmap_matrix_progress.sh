#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/home/zjr/Tracker/pose_point_depth_mv/outputs2/真实采集5组_AR与COLMAPPose_ReconViaGen_vs_SS30K_SLat30K_三选帧轮廓_20260819_v1}

DATASETS=(
  20260816_035545_862_axisuv_v5
  20260812_171117_303
  20260816_040547_970_axisuv_v5
  20260811_064454_154
  20260811_090511_346
)
BRANCHES=(
  01_ar_phone_spherical8
  02_colmap_time_uniform8
  03_colmap_random8_seed20260819
  04_colmap_spherical8
)

stage() {
  local root=$1
  if [[ -s "${root}/06_current_camera_contours/report.json" ]]; then echo complete
  elif [[ -s "${root}/05_reconviagen/inference_manifest.json" ]]; then echo contour_pending
  elif [[ -s "${root}/04_current_ss30k_slat30k/inference_manifest.json" ]]; then echo reconviagen_pending
  elif [[ -s "${root}/03_dino_only_input/model_input_manifest.json" ]]; then echo current_model_pending
  elif [[ -s "${root}/02_runtime_o/runtime_input_manifest.json" ]]; then echo dino_pending
  elif [[ -s "${root}/01_raw_cache/raw_cache_report.json" ]]; then echo runtime_o_pending
  else echo pending
  fi
}

date -Is
echo "root=${ROOT}"
echo "============================================================"
complete=0
for dataset in "${DATASETS[@]}"; do
  if [[ -s "${ROOT}/objects/${dataset}/00_offline_colmap/selection_report.json" ]]; then
    colmap=complete
  else
    colmap=running_or_pending
  fi
  echo "${dataset} offline_colmap=${colmap}"
  for branch in "${BRANCHES[@]}"; do
    value=$(stage "${ROOT}/objects/${dataset}/branches/${branch}")
    [[ "${value}" == complete ]] && complete=$((complete + 1))
    printf '  %-38s %s\n' "${branch}" "${value}"
  done
done
echo "------------------------------------------------------------"
echo "completed_branches=${complete}/20"
[[ -s "${ROOT}/report.json" ]] && echo "final_report=complete" || echo "final_report=pending"
echo "------------------------------------------------------------"
tmux has-session -t real5_colmap_matrix 2>/dev/null \
  && echo "tmux=real5_colmap_matrix RUNNING" \
  || echo "tmux=real5_colmap_matrix EXITED_OR_COMPLETE"
echo "------------------------------------------------------------"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits 2>/dev/null || true
