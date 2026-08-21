#!/usr/bin/env bash
set -uo pipefail

cd /home/zjr/Tracker || exit 97

SEED=${1:-42}
GPU=${2:-1}
CACHE=/data/ar_ss_flow_pose_lifting_overfit64_v1_20260714
RUN=pose_point_depth_mv/outputs/c0_3_gaussian3_train16_s200_seed${SEED}_bf16_20260718
CKPT=${RUN}/checkpoints/last.pt
OVERALL=0

run_exact() {
  local SPLIT=$1
  local INDICES=$2
  local OUT=${RUN}/c0_3_exact_ablation_${SPLIT}
  local CODE
  if [ -f "${OUT}/report.json" ] && [ -d "${OUT}/voxel_maps" ]; then
    echo "reuse exact ablation: ${OUT}"
    CODE=0
  elif [ -e "${OUT}" ]; then
    echo "incomplete exact ablation exists: ${OUT}"
    CODE=98
  else
    CUDA_VISIBLE_DEVICES=${GPU} \
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
      --indices "${INDICES}" \
      --split_name "${SPLIT}" \
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
      --spatial_tolerance exact \
      --allow_spatial_tolerance_mismatch \
      --soft_gate_temperature 0.25 \
      --soft_gate_reliability_power 1.0 \
      --continuous_gate_max_scale 0.10 \
      --save_maps \
      2>&1 | tee "${OUT}.log"
    CODE=${PIPESTATUS[0]}
  fi
  echo "${CODE}" > "${OUT}.exit_code"
  if [ "${CODE}" -ne 0 ]; then
    OVERALL=${CODE}
  fi
}

run_core_shell() {
  local SPLIT=$1
  local EXACT=${RUN}/c0_3_exact_ablation_${SPLIT}
  local NEIGHBORHOOD=${RUN}/c0_3_${SPLIT}
  local OUT=${RUN}/n2_core_shell_${SPLIT}
  local CODE
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
    CODE=$?
    echo "reuse core/shell report: ${OUT} code=${CODE}"
  elif [ -e "${OUT}" ]; then
    echo "incomplete core/shell output exists: ${OUT}"
    CODE=98
  elif [ -f "${EXACT}/report.json" ] && [ -f "${NEIGHBORHOOD}/report.json" ]; then
    /home/zjr/anaconda3/envs/reconviagen/bin/python -u -m \
      pose_point_depth_mv.audit_neighborhood_core_shell \
      --exact_dir "${EXACT}" \
      --neighborhood_dir "${NEIGHBORHOOD}" \
      --output_dir "${OUT}" \
      --bootstrap_samples 10000 \
      --min_core_positive_ratio 0.60 \
      --min_shell_positive_ratio 0.50 \
      --min_core_object_pass_rate 0.65 \
      --min_shell_object_pass_rate 0.65 \
      --fail_on_decision \
      2>&1 | tee "${OUT}.log"
    CODE=${PIPESTATUS[0]}
  else
    echo "skip core/shell ${SPLIT}: prerequisite report missing"
    CODE=99
  fi
  echo "${CODE}" > "${OUT}.exit_code"
  if [ "${CODE}" -ne 0 ]; then
    OVERALL=${CODE}
  fi
}

if [ ! -f "${CKPT}" ]; then
  echo "missing C0.3 checkpoint: ${CKPT}"
  OVERALL=98
else
  run_exact train16 0-15
  run_exact fresh48 16-63
  run_core_shell train16
  run_core_shell fresh48
fi

echo "${OVERALL}" > "${RUN}/n2_runner.status"
echo "N2 seed=${SEED} complete: runtime status=${OVERALL}"

# gaussian3 is the separable [1,2,1]^3 half-voxel trilinear tolerance.
exit 0
