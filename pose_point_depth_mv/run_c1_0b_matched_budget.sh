#!/usr/bin/env bash
set -uo pipefail

cd /home/zjr/Tracker || exit 0

N3=pose_point_depth_mv/outputs/c0_3_gaussian3_s200_multiseed_holdout_summary_20260718/report.json
SUMMARY=pose_point_depth_mv/outputs/c1_0b_matched_budget_multiseed_20260719
SEEDS=(42 43 44)
SPLITS=(fresh48 holdout)
REPORT_DIRS=()
PIPELINE_CODE=0

for SEED in "${SEEDS[@]}"; do
  RUN=pose_point_depth_mv/outputs/c0_3_gaussian3_train16_s200_seed${SEED}_bf16_20260718
  for SPLIT in "${SPLITS[@]}"; do
    C0=${RUN}/c0_3_${SPLIT}/report.json
    OUT=${RUN}/c1_0b_matched_budget_${SPLIT}_20260719
    REPORT_DIRS+=("${OUT}")

    echo
    echo "============================================================"
    echo "C1.0b seed=${SEED} split=${SPLIT}"
    echo "============================================================"
    if [ -f "${OUT}/report.json" ]; then
      echo "reuse complete C1.0b report: ${OUT}/report.json"
      echo 0 > "${OUT}.exit_code"
      continue
    fi
    if [ -e "${OUT}" ]; then
      echo "incomplete C1.0b output exists; leave untouched: ${OUT}"
      echo 98 > "${OUT}.exit_code"
      PIPELINE_CODE=98
      continue
    fi

    /home/zjr/anaconda3/envs/reconviagen/bin/python -u -m \
      pose_point_depth_mv.audit_c1_matched_budget \
      --c0_report "${C0}" \
      --output_dir "${OUT}" \
      --budget_fractions 0.05,0.10,0.20 \
      --policies hard_admitted,continuous \
      --bootstrap_samples 10000 \
      --min_object_win_rate 0.65 \
      2>&1 | tee "${OUT}.log"
    CODE=${PIPESTATUS[0]}
    echo "${CODE}" > "${OUT}.exit_code"
    echo "C1.0b seed=${SEED} split=${SPLIT} runtime code=${CODE}"
    if [ "${CODE}" -ne 0 ]; then
      PIPELINE_CODE=${CODE}
    fi
  done
done

echo
echo "============================================================"
echo "C1.0b three-seed Fresh/Holdout summary"
echo "============================================================"
if [ "${PIPELINE_CODE}" -ne 0 ]; then
  echo "skip C1.0b summary because an upstream runtime failed: ${PIPELINE_CODE}"
  echo 99 > "${SUMMARY}.exit_code"
elif [ -f "${SUMMARY}/report.json" ]; then
  echo "reuse complete C1.0b summary: ${SUMMARY}/report.json"
  echo 0 > "${SUMMARY}.exit_code"
elif [ -e "${SUMMARY}" ]; then
  echo "incomplete C1.0b summary exists; leave untouched: ${SUMMARY}"
  echo 98 > "${SUMMARY}.exit_code"
else
  /home/zjr/anaconda3/envs/reconviagen/bin/python -u -m \
    pose_point_depth_mv.summarize_c1_matched_budget \
    --n3_report "${N3}" \
    --report_dirs "${REPORT_DIRS[@]}" \
    --output_dir "${SUMMARY}" \
    2>&1 | tee "${SUMMARY}.log"
  SUMMARY_CODE=${PIPESTATUS[0]}
  echo "${SUMMARY_CODE}" > "${SUMMARY}.exit_code"
  echo "C1.0b summary runtime code=${SUMMARY_CODE}"
fi

if [ -f "${SUMMARY}/report.json" ]; then
  /home/zjr/anaconda3/envs/reconviagen/bin/python -c '
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("C1.0b integrity:", "PASS" if report["integrity_passed"] else "FAIL")
print("C1.0b route:", report["decision"]["route"])
print("selected policy:", report["decision"]["selected_policy"])
print("allowed next stage:", report["decision"]["allowed_next_stage"])
' "${SUMMARY}/report.json"
fi

echo "C1.0b runner finished; recorded runtime code=${PIPELINE_CODE}"
exit 0

