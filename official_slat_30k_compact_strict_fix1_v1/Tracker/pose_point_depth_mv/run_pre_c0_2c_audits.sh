#!/usr/bin/env bash
set -uo pipefail

cd /home/zjr/Tracker || exit 0

CACHE=/data/ar_ss_flow_pose_lifting_overfit64_v1_20260714
RUN=pose_point_depth_mv/outputs/c0_2b_reliability_train16_s400_seed42_bf16_20260717
AUDIT_ROOT=${RUN}/pre_c0_2c_audit_fresh48_20260718
STEPS=(100 200)
OVERALL_CODE=0

mkdir -p "${AUDIT_ROOT}"

run_eval() {
  local STEP=$1
  local TOLERANCE=$2
  local CKPT
  local OUT
  local CODE
  local EXTRA_ARGS=()

  CKPT=$(printf "%s/checkpoints/step_%06d.pt" "${RUN}" "${STEP}")
  OUT=${AUDIT_ROOT}/${TOLERANCE}_step${STEP}_fresh48
  if [ "${TOLERANCE}" = exact ]; then
    EXTRA_ARGS=(--spatial_tolerance exact --save_maps)
  else
    EXTRA_ARGS=(
      --spatial_tolerance gaussian3
      --allow_spatial_tolerance_mismatch
    )
  fi

  if [ ! -f "${CKPT}" ]; then
    echo "missing checkpoint: ${CKPT}"
    CODE=98
  elif [ -f "${OUT}/report.json" ]; then
    echo "reuse report: ${OUT}/report.json"
    CODE=0
  elif [ -e "${OUT}" ]; then
    echo "incomplete output exists; preserve it and skip: ${OUT}"
    CODE=98
  else
    CUDA_VISIBLE_DEVICES=1 \
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
      --output_dir "${OUT}" \
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
      "${EXTRA_ARGS[@]}" \
      2>&1 | tee "${OUT}.log"
    CODE=${PIPESTATUS[0]}
  fi

  echo "${CODE}" > "${OUT}.exit_code"
  echo "step=${STEP} tolerance=${TOLERANCE} code=${CODE}"
  if [ "${CODE}" -ne 0 ]; then
    OVERALL_CODE=${CODE}
  fi
}

for STEP in "${STEPS[@]}"; do
  run_eval "${STEP}" exact
  run_eval "${STEP}" gaussian3
done

SUMMARY=${AUDIT_ROOT}/summary
if [ -f "${SUMMARY}/report.json" ]; then
  echo "reuse summary: ${SUMMARY}/report.json"
  SUMMARY_CODE=0
elif [ -e "${SUMMARY}" ]; then
  echo "incomplete summary exists; preserve it and skip: ${SUMMARY}"
  SUMMARY_CODE=98
elif [ \
  -f "${AUDIT_ROOT}/exact_step100_fresh48/report.json" \
  ] && [ \
  -f "${AUDIT_ROOT}/exact_step200_fresh48/report.json" \
  ] && [ \
  -f "${AUDIT_ROOT}/gaussian3_step100_fresh48/report.json" \
  ] && [ \
  -f "${AUDIT_ROOT}/gaussian3_step200_fresh48/report.json" \
  ]; then
  /home/zjr/anaconda3/envs/reconviagen/bin/python -u -m \
    pose_point_depth_mv.audit_voxel_dynamics_depth_strata \
    --step100_dir "${AUDIT_ROOT}/exact_step100_fresh48" \
    --step200_dir "${AUDIT_ROOT}/exact_step200_fresh48" \
    --gaussian_step100_dir "${AUDIT_ROOT}/gaussian3_step100_fresh48" \
    --gaussian_step200_dir "${AUDIT_ROOT}/gaussian3_step200_fresh48" \
    --output_dir "${SUMMARY}" \
    --min_voxel_positive_ratio 0.60 \
    --min_object_local_pass_rate 0.65 \
    2>&1 | tee "${SUMMARY}.log"
  SUMMARY_CODE=${PIPESTATUS[0]}
else
  echo "skip summary: one or more prerequisite reports are missing"
  SUMMARY_CODE=99
fi

echo "${SUMMARY_CODE}" > "${SUMMARY}.exit_code"
if [ "${SUMMARY_CODE}" -ne 0 ]; then
  OVERALL_CODE=${SUMMARY_CODE}
fi
echo "${OVERALL_CODE}" > "${AUDIT_ROOT}/runner.status"
echo "pre-C0.2c audits complete: status=${OVERALL_CODE}"

# Scientific FAIL is recorded in reports and never closes the caller terminal.
exit 0
