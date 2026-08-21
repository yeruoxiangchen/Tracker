#!/usr/bin/env bash
set -uo pipefail

cd /home/zjr/Tracker || {
  echo "cannot enter /home/zjr/Tracker"
  exit 0
}

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
N3=pose_point_depth_mv/outputs/c0_3_gaussian3_s200_multiseed_holdout_summary_20260718/report.json
C1=pose_point_depth_mv/outputs/c1_0_target_enrichment_v2_multiseed_exact_20260719/report.json
SUMMARY=pose_point_depth_mv/outputs/c1_1_nested_occupancy_v2_multiseed_exact_20260719
RUNTIME_CODE=0
REPORT_DIRS=()

POLICY=$("${PY}" -c '
import json, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    r.get("format") != "pose_point_depth_mv.c1_enrichment_summary.v2"
    or r.get("passed") is not True
    or not r.get("admitted_policy")
):
    raise SystemExit(2)
print(r["admitted_policy"])
' "${C1}")
C1_CODE=$?
echo "${C1_CODE}" > "${C1}.c1_1_input.exit_code"
if [ "${C1_CODE}" -ne 0 ]; then
  echo "C1.1 blocked: C1.0 multi-seed target enrichment did not pass"
  exit 0
fi
echo "C1.1 admitted weight policy: ${POLICY}"

mapfile -t RUN_ROWS < <(
  "${PY}" -c '
import json, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for row in r["per_seed"]:
    print("{}\t{}".format(int(row["seed"]), row["run_dir"]))
' "${N3}"
)

for ROW in "${RUN_ROWS[@]}"; do
  IFS=$'\t' read -r SEED RUN <<< "${ROW}"
  TRAIN_C0="${RUN}/c0_3_train16/report.json"
  OUT="${RUN}/c1_1_nested_v2_${POLICY}_exact_s200"

  echo
  echo "============================================================"
  echo "C1.1 train seed=${SEED} policy=${POLICY}"
  echo "============================================================"

  if [ -f "${OUT}/train_report.json" ] && [ -f "${OUT}/checkpoints/last.pt" ]; then
    "${PY}" -c '
import json, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
ok = (
    r.get("format") == "pose_point_depth_mv.c1_nested_monotone_calibrator_checkpoint.v2"
    and r.get("passed") is True
    and int(r.get("completed_steps", -1)) == 200
    and int(r.get("training_seed", -1)) == int(sys.argv[2])
    and r.get("weight_policy") == sys.argv[3]
    and Path(r.get("source_c1_summary", "")).resolve() == Path(sys.argv[4]).resolve()
    and Path(r.get("source_c0_report", "")).resolve() == Path(sys.argv[5]).resolve()
    and set(r.get("model_metadata", {}))
        == {"M0_bias", "M1_reliability", "M2_weight_reliability"}
)
raise SystemExit(0 if ok else 2)
' "${OUT}/train_report.json" "${SEED}" "${POLICY}" "${C1}" "${TRAIN_C0}"
    TRAIN_CODE=$?
    if [ "${TRAIN_CODE}" -eq 0 ]; then
      echo "reuse complete C1.1 training run: ${OUT}"
    else
      echo "existing C1.1 training report is invalid: ${OUT}"
      RUNTIME_CODE=98
    fi
  elif [ -e "${OUT}" ]; then
    echo "incomplete C1.1 output exists; inspect or rename it: ${OUT}"
    TRAIN_CODE=98
    RUNTIME_CODE=98
  else
    "${PY}" -u -m pose_point_depth_mv.train_c1_occupancy_calibrator \
      --c1_summary "${C1}" \
      --c0_report "${TRAIN_C0}" \
      --output_dir "${OUT}" \
      --weight_policy summary \
      --target_mode exact \
      --max_steps 200 \
      --save_every 50 \
      --log_every 10 \
      --lr 0.02 \
      --seed "${SEED}" \
      2>&1 | tee "${OUT}.log"
    TRAIN_CODE=${PIPESTATUS[0]}
    if [ "${TRAIN_CODE}" -ne 0 ]; then
      RUNTIME_CODE="${TRAIN_CODE}"
    fi
  fi
  echo "${TRAIN_CODE:-0}" > "${OUT}.train.exit_code"

  if [ "${TRAIN_CODE:-0}" -ne 0 ]; then
    echo "skip evaluations for seed=${SEED}; train code=${TRAIN_CODE}"
    continue
  fi

  for SPLIT in train16 fresh48 holdout; do
    EVAL_C0="${RUN}/c0_3_${SPLIT}/report.json"
    EVAL_OUT="${OUT}/eval_${SPLIT}"
    REPORT_DIRS+=("${EVAL_OUT}")
    echo "C1.1 eval seed=${SEED} split=${SPLIT}"
    if [ -f "${EVAL_OUT}/report.json" ]; then
      "${PY}" -c '
import json, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
ok = (
    r.get("format") == "pose_point_depth_mv.c1_nested_calibrator_eval.v2"
    and int(r.get("training_seed", -1)) == int(sys.argv[2])
    and r.get("split_name") == sys.argv[3]
    and Path(r.get("checkpoint", "")).resolve() == Path(sys.argv[4]).resolve()
)
raise SystemExit(0 if ok else 2)
' "${EVAL_OUT}/report.json" "${SEED}" "${SPLIT}" "${OUT}/checkpoints/last.pt"
      EVAL_CODE=$?
      if [ "${EVAL_CODE}" -eq 0 ]; then
        echo "reuse complete C1.1 evaluation: ${EVAL_OUT}/report.json"
      else
        echo "existing C1.1 evaluation is invalid: ${EVAL_OUT}/report.json"
        RUNTIME_CODE=98
      fi
    elif [ -e "${EVAL_OUT}" ]; then
      echo "incomplete C1.1 evaluation exists: ${EVAL_OUT}"
      EVAL_CODE=98
      RUNTIME_CODE=98
    else
      "${PY}" -u -m pose_point_depth_mv.eval_c1_occupancy_calibrator \
        --c0_report "${EVAL_C0}" \
        --checkpoint "${OUT}/checkpoints/last.pt" \
        --output_dir "${EVAL_OUT}" \
        --permutation_repeats 16 \
        --bootstrap_samples 10000 \
        --min_object_win_rate 0.65 \
        2>&1 | tee "${EVAL_OUT}.log"
      EVAL_CODE=${PIPESTATUS[0]}
      if [ "${EVAL_CODE}" -ne 0 ]; then
        RUNTIME_CODE="${EVAL_CODE}"
      fi
    fi
    echo "${EVAL_CODE}" > "${EVAL_OUT}.exit_code"
  done
done

if [ "${RUNTIME_CODE}" -eq 0 ] && [ "${#REPORT_DIRS[@]}" -eq 9 ]; then
  if [ -f "${SUMMARY}/report.json" ]; then
    "${PY}" -c '
import json, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
ok = (
    r.get("format") == "pose_point_depth_mv.c1_nested_calibrator_summary.v2"
    and Path(r.get("source_c1_summary", "")).resolve() == Path(sys.argv[2]).resolve()
    and len(r.get("source_reports", [])) == 9
)
raise SystemExit(0 if ok else 2)
' "${SUMMARY}/report.json" "${C1}"
    SUMMARY_CODE=$?
    if [ "${SUMMARY_CODE}" -eq 0 ]; then
      echo "reuse complete C1.1 summary: ${SUMMARY}/report.json"
    else
      echo "existing C1.1 summary is invalid: ${SUMMARY}/report.json"
    fi
  elif [ -e "${SUMMARY}" ]; then
    echo "incomplete C1.1 summary exists: ${SUMMARY}"
    SUMMARY_CODE=98
  else
    "${PY}" -u -m pose_point_depth_mv.summarize_c1_occupancy_calibrators \
      --c1_summary "${C1}" \
      --report_dirs "${REPORT_DIRS[@]}" \
      --output_dir "${SUMMARY}" \
      2>&1 | tee "${SUMMARY}.log"
    SUMMARY_CODE=${PIPESTATUS[0]}
  fi
else
  echo "skip C1.1 summary; runtime=${RUNTIME_CODE}, reports=${#REPORT_DIRS[@]}"
  SUMMARY_CODE=99
fi
echo "${SUMMARY_CODE}" > "${SUMMARY}.exit_code"
echo "C1.1 runtime/summary codes: ${RUNTIME_CODE}/${SUMMARY_CODE}"
echo "Scientific PASS/FAIL is stored in ${SUMMARY}/report.json"
exit 0
