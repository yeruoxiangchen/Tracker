#!/usr/bin/env bash
set -uo pipefail

cd /home/zjr/Tracker || {
  echo "cannot enter /home/zjr/Tracker"
  exit 0
}

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
N3=pose_point_depth_mv/outputs/c0_3_gaussian3_s200_multiseed_holdout_summary_20260718/report.json
SUMMARY=pose_point_depth_mv/outputs/c1_0_target_enrichment_v2_multiseed_exact_20260719
RUNTIME_CODE=0
REPORT_DIRS=()

"${PY}" -c '
import json, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if r.get("passed") is not True:
    raise SystemExit(2)
' "${N3}"
N3_CODE=$?
echo "${N3_CODE}" > "${N3}.c1_0_input.exit_code"
if [ "${N3_CODE}" -ne 0 ]; then
  echo "C1.0 blocked: N3 report is missing or not passed"
  exit 0
fi

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
  for SPLIT in train16 fresh48 holdout; do
    C0="${RUN}/c0_3_${SPLIT}/report.json"
    OUT="${RUN}/c1_0_v2_enrichment_${SPLIT}_exact"
    REPORT_DIRS+=("${OUT}")

    echo
    echo "============================================================"
    echo "C1.0 seed=${SEED} split=${SPLIT}"
    echo "============================================================"

    if [ -f "${OUT}/report.json" ]; then
      "${PY}" -c '
import json, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
ok = (
    r.get("format") == "pose_point_depth_mv.c1_enrichment_report.v2"
    and int(r.get("training_seed", -1)) == int(sys.argv[2])
    and r.get("split_name") == sys.argv[3]
    and Path(r.get("source_c0_report", "")).resolve() == Path(sys.argv[4]).resolve()
)
raise SystemExit(0 if ok else 2)
' "${OUT}/report.json" "${SEED}" "${SPLIT}" "${C0}"
      CODE=$?
      if [ "${CODE}" -eq 0 ]; then
        echo "reuse complete C1.0 report: ${OUT}/report.json"
      else
        echo "invalid existing C1.0 report: ${OUT}/report.json"
        RUNTIME_CODE=98
      fi
    elif [ -e "${OUT}" ]; then
      echo "incomplete output exists; inspect or rename it: ${OUT}"
      CODE=98
      RUNTIME_CODE=98
    else
      "${PY}" -u -m pose_point_depth_mv.audit_c1_target_enrichment \
        --c0_report "${C0}" \
        --output_dir "${OUT}" \
        --formal_target_mode exact \
        --admission_policies hard_admitted,continuous \
        --permutation_repeats 16 \
        --bootstrap_samples 10000 \
        --min_object_win_rate 0.65 \
        2>&1 | tee "${OUT}.log"
      CODE=${PIPESTATUS[0]}
      if [ "${CODE}" -ne 0 ]; then
        RUNTIME_CODE="${CODE}"
      fi
    fi
    echo "${CODE:-0}" > "${OUT}.exit_code"
    echo "C1.0 seed=${SEED} split=${SPLIT} runtime code=${CODE:-0}"
  done
done

if [ "${RUNTIME_CODE}" -eq 0 ]; then
  if [ -f "${SUMMARY}/report.json" ]; then
    "${PY}" -c '
import json, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
ok = (
    r.get("format") == "pose_point_depth_mv.c1_enrichment_summary.v2"
    and Path(r.get("source_n3_report", "")).resolve() == Path(sys.argv[2]).resolve()
    and len(r.get("source_reports", [])) == 9
)
raise SystemExit(0 if ok else 2)
' "${SUMMARY}/report.json" "${N3}"
    SUMMARY_CODE=$?
    if [ "${SUMMARY_CODE}" -eq 0 ]; then
      echo "reuse complete C1.0 summary: ${SUMMARY}/report.json"
    else
      echo "existing C1.0 summary is invalid: ${SUMMARY}/report.json"
    fi
  elif [ -e "${SUMMARY}" ]; then
    echo "incomplete C1.0 summary exists: ${SUMMARY}"
    SUMMARY_CODE=98
  else
    "${PY}" -u -m pose_point_depth_mv.summarize_c1_target_enrichment \
      --n3_report "${N3}" \
      --report_dirs "${REPORT_DIRS[@]}" \
      --output_dir "${SUMMARY}" \
      2>&1 | tee "${SUMMARY}.log"
    SUMMARY_CODE=${PIPESTATUS[0]}
  fi
else
  echo "skip C1.0 summary because runtime code=${RUNTIME_CODE}"
  SUMMARY_CODE=99
fi
echo "${SUMMARY_CODE}" > "${SUMMARY}.exit_code"
echo "C1.0 runtime/summary codes: ${RUNTIME_CODE}/${SUMMARY_CODE}"
echo "Scientific PASS/FAIL is stored in ${SUMMARY}/report.json"

# Always return success to the interactive parent shell. Runtime and scientific
# statuses are preserved in explicit files and JSON reports above.
exit 0
