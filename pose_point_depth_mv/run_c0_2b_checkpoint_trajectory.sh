#!/usr/bin/env bash
set -uo pipefail

cd /home/zjr/Tracker || exit 0

CACHE=/data/ar_ss_flow_pose_lifting_overfit64_v1_20260714
RUN=pose_point_depth_mv/outputs/c0_2b_reliability_train16_s400_seed42_bf16_20260717
TRAJ=${RUN}/voxel_selfcal_v2_checkpoint_trajectory_t0
STEPS=(50 100 150 200 250 300 350 400)
OVERALL_CODE=0

mkdir -p "${TRAJ}"

run_split() {
  local STEP=$1
  local SPLIT=$2
  local INDICES=$3
  local CKPT
  local OUT
  local RUN_CODE

  CKPT=$(printf "%s/checkpoints/step_%06d.pt" "${RUN}" "${STEP}")
  OUT=${TRAJ}/step_${STEP}/${SPLIT}
  mkdir -p "$(dirname "${OUT}")"

  if [ ! -f "${CKPT}" ]; then
    RUN_CODE=98
    echo "missing checkpoint: ${CKPT}"
  elif [ -f "${OUT}/report.json" ]; then
    RUN_CODE=0
    echo "reuse report: ${OUT}/report.json"
  elif [ -e "${OUT}" ]; then
    RUN_CODE=98
    echo "incomplete output exists; no overwrite: ${OUT}"
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
      --save_maps \
      2>&1 | tee "${OUT}.log"
    RUN_CODE=${PIPESTATUS[0]}
  fi

  echo "${RUN_CODE}" > "${OUT}.exit_code"
  echo "step=${STEP} split=${SPLIT} runtime_code=${RUN_CODE}"
  if [ "${RUN_CODE}" -ne 0 ]; then
    OVERALL_CODE=${RUN_CODE}
  fi
}

for STEP in "${STEPS[@]}"; do
  run_split "${STEP}" train16 0-15
  run_split "${STEP}" fresh48 16-63
done

/home/zjr/anaconda3/envs/reconviagen/bin/python -c '
import json
from pathlib import Path

root = Path("pose_point_depth_mv/outputs/c0_2b_reliability_train16_s400_seed42_bf16_20260717/voxel_selfcal_v2_checkpoint_trajectory_t0")
for step in (50, 100, 150, 200, 250, 300, 350, 400):
    print("step={}".format(step))
    for split in ("train16", "fresh48"):
        path = root / "step_{}".format(step) / split / "report.json"
        if not path.is_file():
            print("{} MISSING".format(split))
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        primary = report["primary"]
        hard = primary["hard_margin_mean"]
        print(
            "{} passed={} mean={:.6f} ci_low={:.6f} voxel={:.4f} local_pass={:.4f}".format(
                split,
                report["passed"],
                hard["object"]["mean"],
                hard["object_bootstrap_95_ci"][0],
                primary["voxel_positive_ratio"]["object"]["mean"],
                primary["local_object_pass_rate"],
            )
        )
' > "${TRAJ}/trajectory_summary.log" 2>&1
SUMMARY_CODE=$?
echo "${SUMMARY_CODE}" > "${TRAJ}/summary.exit_code"
echo "${OVERALL_CODE}" > "${TRAJ}/runner.exit_code"
echo "trajectory runner complete: runtime=${OVERALL_CODE} summary=${SUMMARY_CODE}"
exit 0
