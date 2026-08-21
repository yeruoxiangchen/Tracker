#!/usr/bin/env bash
set -u

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_ROOT}/pose_point_depth_mv/outputs2/Objectron_ShoeCamera_GT中心引导_训练一致球面FPS8_SS30K_SLat30K_vs_ReconViaGen_20260820_v1}
UNIT=${UNIT:-tracker-objectron-gt-center-guided8-v1.service}
SLUG=01_gt_center_training_spherical_fps8

date -u
printf 'service: '
systemctl --user is-active "${UNIT}" 2>/dev/null || true

for spec in \
  "shoe:${OUTPUT_ROOT}/shoe_batch14_30_obj0" \
  "camera:${OUTPUT_ROOT}/camera_batch7_24_obj0"
do
  label=${spec%%:*}
  root=${spec#*:}
  branch=${root}/${SLUG}
  printf '%s: plan=%s dino=%s current_pose=%s current_true=%s recon=%s contours=%s/%s final=%s\n' \
    "${label}" \
    "$([[ -s "${root}/experiment_plan.json" ]] && echo 1 || echo 0)" \
    "$([[ -s "${branch}/03_model_input_pose_mask/model_input_manifest.json" ]] && echo 1 || echo 0)" \
    "$([[ -s "${branch}/05_current_pose_mask/inference_manifest.json" ]] && echo 1 || echo 0)" \
    "$([[ -s "${branch}/06_current_true_pose/inference_manifest.json" ]] && echo 1 || echo 0)" \
    "$([[ -s "${branch}/07_reconviagen_once/inference_manifest.json" ]] && echo 1 || echo 0)" \
    "$([[ -s "${branch}/08_contours_pose_mask/report.json" ]] && echo 1 || echo 0)" \
    "$([[ -s "${branch}/09_contours_true_pose/report.json" ]] && echo 1 || echo 0)" \
    "$([[ -s "${root}/report.json" ]] && echo 1 || echo 0)"
done

printf 'pair report: %s\n' "$([[ -s "${OUTPUT_ROOT}/report.json" ]] && echo COMPLETE || echo pending)"
tail -n 3 "${OUTPUT_ROOT}/logs/master.log" 2>/dev/null || true
