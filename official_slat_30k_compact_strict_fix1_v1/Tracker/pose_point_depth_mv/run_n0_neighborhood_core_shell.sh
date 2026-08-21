#!/usr/bin/env bash
set -uo pipefail

cd /home/zjr/Tracker || exit 97

CACHE=/data/ar_ss_flow_pose_lifting_overfit64_v1_20260714
RUN=pose_point_depth_mv/outputs/c0_2b_reliability_train16_s400_seed42_bf16_20260717
ROOT=${RUN}/pre_c0_2c_audit_fresh48_20260718
EXACT_DIR=${ROOT}/exact_step200_fresh48
GAUSSIAN_DIR=${ROOT}/gaussian3_step200_fresh48_maps_n0
OUT=${ROOT}/n0_core_shell_step200_fresh48
CKPT=${RUN}/checkpoints/step_000200.pt
OVERALL=0

if [ -f "${GAUSSIAN_DIR}/report.json" ] && [ -d "${GAUSSIAN_DIR}/voxel_maps" ]; then
  echo "reuse gaussian maps: ${GAUSSIAN_DIR}"
  EVAL_CODE=0
elif [ -e "${GAUSSIAN_DIR}" ]; then
  echo "incomplete gaussian output exists: ${GAUSSIAN_DIR}"
  EVAL_CODE=98
else
  CUDA_VISIBLE_DEVICES=${GPU:-1} \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn \
  SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib \
  NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/zjr/anaconda3/envs/reconviagen/bin/python -u -m \
    pose_point_depth_mv.eval_voxel_selfcal_correspondence \
    --cache_manifest "${CACHE}/manifest.json" \
    --checkpoint "${CKPT}" \
    --output_dir "${GAUSSIAN_DIR}" \
    --indices 16-63 \
    --split_name fresh48 \
    --max_samples 0 \
    --device cuda \
    --threshold 0.0 \
    --bootstrap_samples 10000 \
    --min_voxel_positive_ratio 0.60 \
    --min_per_object_positive_ratio 0.50 \
    --min_object_local_pass_rate 0.65 \
    --min_heldout_gate_positive_ratio 0.65 \
    --min_spatial_control_object_win_rate 0.65 \
    --min_spatial_control_gate_positive_ratio 0.65 \
    --min_spatial_std 1e-4 \
    --max_permutation_diff 1e-5 \
    --spatial_tolerance gaussian3 \
    --allow_spatial_tolerance_mismatch \
    --soft_gate_temperature 0.25 \
    --soft_gate_reliability_power 1.0 \
    --continuous_gate_max_scale 0.10 \
    --save_maps \
    2>&1 | tee "${GAUSSIAN_DIR}.log"
  EVAL_CODE=${PIPESTATUS[0]}
fi
echo "${EVAL_CODE}" > "${GAUSSIAN_DIR}.exit_code"
if [ "${EVAL_CODE}" -ne 0 ]; then
  OVERALL=${EVAL_CODE}
fi

if [ "${EVAL_CODE}" -eq 0 ] && [ -f "${EXACT_DIR}/report.json" ]; then
  if [ -f "${OUT}/report.json" ]; then
    /home/zjr/anaconda3/envs/reconviagen/bin/python -c '
import json, sys
r = json.load(open(sys.argv[1]))
required = (
    "neighborhood_formal_report_passed",
    "neighborhood_recomputed_decision_passed",
    "neighborhood_raw_metrics_match_report",
)
ok = r.get("passed") is True and all(r.get("checks", {}).get(k) is True for k in required)
raise SystemExit(0 if ok else 2)
' "${OUT}/report.json"
    N0_CODE=$?
    echo "reuse N0 report: ${OUT}/report.json code=${N0_CODE}"
  elif [ -e "${OUT}" ]; then
    echo "incomplete N0 output exists: ${OUT}"
    N0_CODE=98
  else
    /home/zjr/anaconda3/envs/reconviagen/bin/python -u -m \
      pose_point_depth_mv.audit_neighborhood_core_shell \
      --exact_dir "${EXACT_DIR}" \
      --neighborhood_dir "${GAUSSIAN_DIR}" \
      --output_dir "${OUT}" \
      --bootstrap_samples 10000 \
      --min_core_positive_ratio 0.60 \
      --min_shell_positive_ratio 0.50 \
      --min_core_object_pass_rate 0.65 \
      --min_shell_object_pass_rate 0.65 \
      --fail_on_decision \
      2>&1 | tee "${OUT}.log"
    N0_CODE=${PIPESTATUS[0]}
  fi
else
  echo "skip N0 core/shell audit: gaussian or exact prerequisite missing"
  N0_CODE=99
fi
echo "${N0_CODE}" > "${OUT}.exit_code"
if [ "${N0_CODE}" -ne 0 ]; then
  OVERALL=${N0_CODE}
fi

echo "${OVERALL}" > "${ROOT}/n0_runner.status"
echo "N0 complete: status=${OVERALL}"

# This script is launched with /bin/bash under nohup. Scientific FAIL is in JSON.
exit 0
