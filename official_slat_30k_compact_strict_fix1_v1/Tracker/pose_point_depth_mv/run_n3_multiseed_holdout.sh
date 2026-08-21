#!/usr/bin/env bash
set -uo pipefail

cd /home/zjr/Tracker || exit 97

GPU=${GPU:-1}
CACHE=/data/ar_ss_flow_pose_lifting_holdout48_v1_20260718
OUTPUT_ROOT=pose_point_depth_mv/outputs
SUMMARY=${OUTPUT_ROOT}/c0_3_gaussian3_s200_multiseed_holdout_summary_20260718
SEEDS=(42 43 44)
RUNS=()
OVERALL=0

json_pass_code() {
  /home/zjr/anaconda3/envs/reconviagen/bin/python -c '
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(98)
r = json.loads(p.read_text(encoding="utf-8"))
raise SystemExit(0 if r.get("passed") is True else 2)
' "$1"
}

c0_report_pass_code() {
  /home/zjr/anaconda3/envs/reconviagen/bin/python -c '
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(98)
r = json.loads(p.read_text(encoding="utf-8"))
schema_ok = (
    bool(r.get("checkpoint_sha256"))
    and r.get("hard_admitted_soft_weight_protocol", {}).get("formal_n3_gate") is True
    and r.get("continuous_soft_weight_protocol", {}).get("c1_ablation_only") is True
)
raise SystemExit(0 if r.get("passed") is True and schema_ok else 2)
' "$1"
}

PREREQ=0
if [ ! -f "${CACHE}/manifest.json" ]; then
  echo "missing untouched holdout cache: ${CACHE}/manifest.json"
  PREREQ=98
fi
if [ ! -f "${CACHE}.runner.status" ]; then
  echo "holdout cache runner status is missing"
  PREREQ=98
elif [ "$(cat "${CACHE}.runner.status")" -ne 0 ]; then
  echo "holdout cache audit did not pass"
  PREREQ=2
fi
for REPORT in \
  "${CACHE}/independent_audit/report.json" \
  "${CACHE}/stock_condition_audit_2_4_8/report.json"; do
  json_pass_code "${REPORT}"
  CODE=$?
  if [ "${CODE}" -ne 0 ]; then
    echo "holdout cache prerequisite report failed: ${REPORT} code=${CODE}"
    PREREQ=${CODE}
  fi
done

for SEED in "${SEEDS[@]}"; do
  RUN=${OUTPUT_ROOT}/c0_3_gaussian3_train16_s200_seed${SEED}_bf16_20260718
  RUNS+=("${RUN}")
  for SPLIT in c0_3_train16 c0_3_fresh48; do
    c0_report_pass_code "${RUN}/${SPLIT}/report.json"
    CODE=$?
    if [ "${CODE}" -ne 0 ]; then
      echo "prerequisite FAIL: seed=${SEED} split=${SPLIT} code=${CODE}"
      PREREQ=${CODE}
    fi
  done
done

if [ "${PREREQ}" -ne 0 ]; then
  echo "do not touch holdout: train/fresh multi-seed prerequisites failed"
  OVERALL=${PREREQ}
else
  for SEED in "${SEEDS[@]}"; do
    RUN=${OUTPUT_ROOT}/c0_3_gaussian3_train16_s200_seed${SEED}_bf16_20260718
    OUT=${RUN}/c0_3_holdout
    CKPT=${RUN}/checkpoints/last.pt
    if [ -f "${OUT}/report.json" ] && [ -d "${OUT}/voxel_maps" ]; then
      c0_report_pass_code "${OUT}/report.json"
      CODE=$?
      echo "reuse holdout seed=${SEED}: code=${CODE}"
    elif [ -e "${OUT}" ]; then
      echo "incomplete holdout output exists: ${OUT}"
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
        --indices all \
        --split_name holdout \
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
        --spatial_tolerance checkpoint \
        --soft_gate_temperature 0.25 \
        --soft_gate_reliability_power 1.0 \
        --continuous_gate_max_scale 0.10 \
        --save_maps \
        --fail_on_decision \
        2>&1 | tee "${OUT}.log"
      CODE=${PIPESTATUS[0]}
    fi
    echo "${CODE}" > "${OUT}.exit_code"
    if [ "${CODE}" -ne 0 ]; then
      OVERALL=${CODE}
    fi
  done
fi

if [ "${OVERALL}" -eq 0 ]; then
  if [ -f "${SUMMARY}/report.json" ]; then
    /home/zjr/anaconda3/envs/reconviagen/bin/python -c '
import json, sys
r = json.load(open(sys.argv[1]))
required = (
    "same_seed_splits_share_checkpoint_path",
    "same_seed_splits_share_checkpoint_sha256",
    "model_head_and_evidence_identity_consistent",
)
ok = (
    r.get("format") == "pose_point_depth_mv.neighborhood_voxel_selfcal_multiseed.v2"
    and r.get("passed") is True
    and all(r.get("checks", {}).get(k) is True for k in required)
)
raise SystemExit(0 if ok else 2)
' "${SUMMARY}/report.json"
    SUMMARY_CODE=$?
    echo "reuse N3 summary: code=${SUMMARY_CODE}"
  elif [ -e "${SUMMARY}" ]; then
    echo "incomplete N3 summary exists: ${SUMMARY}"
    SUMMARY_CODE=98
  else
    /home/zjr/anaconda3/envs/reconviagen/bin/python -u -m \
      pose_point_depth_mv.summarize_neighborhood_multiseed \
      --run_dirs "${RUNS[@]}" \
      --output_dir "${SUMMARY}" \
      --train_subdir c0_3_train16 \
      --fresh_subdir c0_3_fresh48 \
      --holdout_subdir c0_3_holdout \
      --expected_seeds 42,43,44 \
      --fail_on_decision \
      2>&1 | tee "${SUMMARY}.log"
    SUMMARY_CODE=${PIPESTATUS[0]}
  fi
else
  echo "skip N3 summary because one holdout report failed"
  SUMMARY_CODE=99
fi
echo "${SUMMARY_CODE}" > "${SUMMARY}.exit_code"
if [ "${SUMMARY_CODE}" -ne 0 ]; then
  OVERALL=${SUMMARY_CODE}
fi

echo "${OVERALL}" > "${SUMMARY}.runner.status"
echo "N3 multi-seed holdout complete: status=${OVERALL}"
exit 0
