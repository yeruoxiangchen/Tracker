#!/usr/bin/env bash
set -uo pipefail

cd /home/zjr/Tracker || exit 97

SEED=${1:-42}
OUTPUT_ROOT=pose_point_depth_mv/outputs
SUMMARY=${OUTPUT_ROOT}/c0_3_gaussian3_s200_multiseed_holdout_summary_20260718
RUN=${OUTPUT_ROOT}/c0_3_gaussian3_train16_s200_seed${SEED}_bf16_20260718
OUT=${RUN}/n4_c1_gate_manifest

if [ -f "${OUT}/manifest.json" ]; then
  /home/zjr/anaconda3/envs/reconviagen/bin/python -c '
import json, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
required = (
    "source_run_is_exact_n3_run",
    "source_protocol_signature_matches_n3",
    "source_checkpoint_matches_n3",
    "source_checkpoint_sha256_matches_n3",
)
ok = (
    r.get("format") == "pose_point_depth_mv.c1_gate_manifest.v2"
    and r.get("passed") is True
    and all(r.get("checks", {}).get(k) is True for k in required)
)
raise SystemExit(0 if ok else 2)
' "${OUT}/manifest.json"
CODE=$?
echo "reuse N4 C1 gate manifest: code=${CODE}"
elif [ -e "${OUT}" ]; then
  echo "incomplete N4 output exists: ${OUT}"
  CODE=98
else
  /home/zjr/anaconda3/envs/reconviagen/bin/python -u -m \
    pose_point_depth_mv.prepare_c1_neighborhood_gates \
    --multiseed_report "${SUMMARY}/report.json" \
    --run_dir "${RUN}" \
    --map_subdir c0_3_train16 \
    --output_dir "${OUT}" \
    --expected_seed "${SEED}" \
    --fail_on_decision \
    2>&1 | tee "${OUT}.log"
  CODE=${PIPESTATUS[0]}
fi

echo "${CODE}" > "${OUT}.exit_code"
echo "N4 gate preparation complete: status=${CODE}"

# N4 only exports validated inputs. Decoder and Flow remain frozen/off.
exit 0
