#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/zjr/Tracker/pose_point_depth_mv/outputs2/真实采集5组_AR_COLMAP六分支_SS30K_SLat30K_runtimeO正确轮廓_20260819_v2}

date -Is
echo "root=${ROOT}"
echo "============================================================"

if [[ ! -d "${ROOT}" ]]; then
  echo "state=not_started"
  exit 0
fi

branches=0
corrected=0
contours=0
fresh_current=0
fresh_recon=0
for object_root in "${ROOT}"/objects/*; do
  [[ -d "${object_root}" ]] || continue
  name=$(basename "${object_root}")
  echo "object=${name}"
  for slug in \
    01_ar_time_uniform8 \
    02_ar_random8_seed20260819 \
    03_ar_spherical8 \
    04_colmap_time_uniform8 \
    05_colmap_random8_seed20260819 \
    06_colmap_spherical8
  do
    branch=${object_root}/branches/${slug}
    status=pending
    [[ -s "${branch}/04_current_ss30k_slat30k/inference_manifest.json" ]] && status=current
    [[ -s "${branch}/05_reconviagen/inference_manifest.json" ]] && status=${status}+recon
    if [[ -s "${branch}/06_current_camera_contours/report.json" ]]; then
      status=complete
      contours=$((contours + 1))
    fi
    [[ -s "${branch}/legacy_v1_reuse_and_axis_repair.json" ]] && corrected=$((corrected + 1))
    if [[ "${slug}" == 01_* || "${slug}" == 02_* ]]; then
      [[ -s "${branch}/04_current_ss30k_slat30k/inference_manifest.json" ]] && fresh_current=$((fresh_current + 1))
      [[ -s "${branch}/05_reconviagen/inference_manifest.json" ]] && fresh_recon=$((fresh_recon + 1))
    fi
    [[ -d "${branch}" ]] && branches=$((branches + 1))
    printf '  %-39s %s\n' "${slug}" "${status}"
  done
done

echo "------------------------------------------------------------"
echo "materialized_branch_dirs=${branches}/30"
echo "hash_bound_legacy_axis_repairs=${corrected}/20"
echo "fresh_AR_current_meshes=${fresh_current}/10"
echo "fresh_AR_reconviagen_meshes=${fresh_recon}/10"
echo "corrected_contour_reports=${contours}/30"
[[ -s "${ROOT}/report.json" ]] && echo "final_report=complete" || echo "final_report=pending"
echo "------------------------------------------------------------"
echo "live processes"
pgrep -af '[r]un_real_capture5_ss30k_slat30k_six_branch_runtime_o_v2|[i]nfer_real_proobjaverse_official_ss_slat|[i]nfer_omni_real_reconviagen|[r]ender_runtime_o_mesh_camera_contours' || echo "none"
