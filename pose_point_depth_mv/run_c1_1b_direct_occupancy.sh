#!/usr/bin/env bash
set -uo pipefail

cd /home/zjr/Tracker || exit 0

C1_0B=pose_point_depth_mv/outputs/c1_0b_matched_budget_multiseed_20260719/report.json
SUMMARY=pose_point_depth_mv/outputs/c1_1b_direct_occupancy_multiseed_20260719
SEEDS=(42 43 44)
REPORT_DIRS=()
PIPELINE_CODE=0

if [ ! -f "${C1_0B}" ]; then
  echo "C1.1b blocked: missing C1.0b summary ${C1_0B}"
  exit 0
fi

/home/zjr/anaconda3/envs/reconviagen/bin/python -c '
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
allowed = bool(report.get("passed")) and report["decision"]["route"] in {
    "restricted_surface_occupancy_gate_candidate",
    "auxiliary_correspondence_feature_only",
}
print("C1.0b route:", report["decision"]["route"])
raise SystemExit(0 if allowed else 3)
' "${C1_0B}"
ADMISSION_CODE=$?
if [ "${ADMISSION_CODE}" -ne 0 ]; then
  echo "C1.1b is scientifically blocked; no training was launched."
  exit 0
fi

for SEED in "${SEEDS[@]}"; do
  RUN=pose_point_depth_mv/outputs/c0_3_gaussian3_train16_s200_seed${SEED}_bf16_20260718
  TRAIN_OUT=${RUN}/c1_1b_direct_occupancy_s400_20260719
  TRAIN_C0=${RUN}/c0_3_train16/report.json

  echo
  echo "============================================================"
  echo "C1.1b train seed=${SEED}"
  echo "============================================================"
  if [ -f "${TRAIN_OUT}/train_report.json" ] && [ -f "${TRAIN_OUT}/checkpoints/last.pt" ]; then
    echo "reuse complete C1.1b train run: ${TRAIN_OUT}"
    TRAIN_CODE=0
  elif [ -e "${TRAIN_OUT}" ]; then
    echo "incomplete C1.1b train output exists; leave untouched: ${TRAIN_OUT}"
    TRAIN_CODE=98
  else
    CUDA_VISIBLE_DEVICES=1 \
    /home/zjr/anaconda3/envs/reconviagen/bin/python -u -m \
      pose_point_depth_mv.train_c1_direct_occupancy \
      --c1_0b_summary "${C1_0B}" \
      --c0_report "${TRAIN_C0}" \
      --output_dir "${TRAIN_OUT}" \
      --device cuda \
      --target_mode exact \
      --hidden_dim 64 \
      --max_steps 400 \
      --save_every 100 \
      --log_every 10 \
      --lr 1e-3 \
      --weight_decay 1e-4 \
      --seed "${SEED}" \
      --max_score_diff 1e-3 \
      2>&1 | tee "${TRAIN_OUT}.log"
    TRAIN_CODE=${PIPESTATUS[0]}
  fi
  echo "${TRAIN_CODE}" > "${TRAIN_OUT}.exit_code"
  echo "C1.1b seed=${SEED} train runtime code=${TRAIN_CODE}"
  if [ "${TRAIN_CODE}" -ne 0 ]; then
    PIPELINE_CODE=${TRAIN_CODE}
    continue
  fi

  for SPLIT in fresh48 holdout; do
    EVAL_C0=${RUN}/c0_3_${SPLIT}/report.json
    EVAL_OUT=${TRAIN_OUT}/eval_${SPLIT}
    REPORT_DIRS+=("${EVAL_OUT}")

    echo
    echo "C1.1b eval seed=${SEED} split=${SPLIT}"
    if [ -f "${EVAL_OUT}/report.json" ]; then
      echo "reuse complete C1.1b eval: ${EVAL_OUT}/report.json"
      EVAL_CODE=0
    elif [ -e "${EVAL_OUT}" ]; then
      echo "incomplete C1.1b eval output exists; leave untouched: ${EVAL_OUT}"
      EVAL_CODE=98
    else
      CUDA_VISIBLE_DEVICES=1 \
      /home/zjr/anaconda3/envs/reconviagen/bin/python -u -m \
        pose_point_depth_mv.eval_c1_direct_occupancy \
        --c0_report "${EVAL_C0}" \
        --checkpoint "${TRAIN_OUT}/checkpoints/last.pt" \
        --output_dir "${EVAL_OUT}" \
        --device cuda \
        --max_score_diff 1e-3 \
        --bootstrap_samples 10000 \
        --min_object_win_rate 0.65 \
        2>&1 | tee "${EVAL_OUT}.log"
      EVAL_CODE=${PIPESTATUS[0]}
    fi
    echo "${EVAL_CODE}" > "${EVAL_OUT}.exit_code"
    echo "C1.1b seed=${SEED} split=${SPLIT} runtime code=${EVAL_CODE}"
    if [ "${EVAL_CODE}" -ne 0 ]; then
      PIPELINE_CODE=${EVAL_CODE}
    fi
  done
done

echo
echo "============================================================"
echo "C1.1b three-seed Fresh/Holdout summary"
echo "============================================================"
if [ "${PIPELINE_CODE}" -ne 0 ]; then
  echo "skip C1.1b summary because an upstream runtime failed: ${PIPELINE_CODE}"
  echo 99 > "${SUMMARY}.exit_code"
elif [ -f "${SUMMARY}/report.json" ]; then
  echo "reuse complete C1.1b summary: ${SUMMARY}/report.json"
  echo 0 > "${SUMMARY}.exit_code"
elif [ -e "${SUMMARY}" ]; then
  echo "incomplete C1.1b summary exists; leave untouched: ${SUMMARY}"
  echo 98 > "${SUMMARY}.exit_code"
else
  /home/zjr/anaconda3/envs/reconviagen/bin/python -u -m \
    pose_point_depth_mv.summarize_c1_direct_occupancy \
    --c1_0b_summary "${C1_0B}" \
    --report_dirs "${REPORT_DIRS[@]}" \
    --output_dir "${SUMMARY}" \
    2>&1 | tee "${SUMMARY}.log"
  SUMMARY_CODE=${PIPESTATUS[0]}
  echo "${SUMMARY_CODE}" > "${SUMMARY}.exit_code"
  echo "C1.1b summary runtime code=${SUMMARY_CODE}"
fi

if [ -f "${SUMMARY}/report.json" ]; then
  /home/zjr/anaconda3/envs/reconviagen/bin/python -c '
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("C1.1b decision:", "PASS" if report["passed"] else "FAIL")
print("allowed next stage:", report["allowed_next_stage"])
' "${SUMMARY}/report.json"
fi

echo "C1.1b runner finished; recorded runtime code=${PIPELINE_CODE}"
exit 0

