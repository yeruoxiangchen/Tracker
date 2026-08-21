#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
PYTHON=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-21600}
POLL_SECONDS=${POLL_SECONDS:-20}

cd "${PROJECT_ROOT}"
export PYTHONPATH="$PWD:$PWD/ReconViaGen:$PWD/ReconViaGen/wheels/vggt"
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba_cache}

DEV=${PROJECT_ROOT}/pose_point_depth_mv/outputs2/ProObjaverse_Dev48固定随机2组_ReconViaGen_vs_VSS2k_VSLat15k_20260818_v1
OMNI=${PROJECT_ROOT}/pose_point_depth_mv/outputs2/OmniPlant012冻结8视图_ReconViaGen_vs_VSS2k_VSLat15k_相机轮廓_20260818_v1
CASE1=${DEV}/case_01_dev_index_38_0f5256cf8b1e/case_report.json
CASE2=${DEV}/case_02_dev_index_23_3f819e58019e/case_report.json
OMNI_SUMMARY=${OMNI}/summary.json
COMPLETE=${PROJECT_ROOT}/pose_point_depth_mv/outputs2/logs_with_vggt_qualitative_20260818_v1/ALL_COMPLETE.json

START=$(date +%s)
while [ ! -s "${CASE1}" ] || [ ! -s "${CASE2}" ] || [ ! -s "${OMNI_SUMMARY}" ]; do
  NOW=$(date +%s)
  if (( NOW - START > TIMEOUT_SECONDS )); then
    echo "ERROR: qualitative completion guard timed out" >&2
    exit 90
  fi
  printf '[%s] case1=%s case2=%s omni=%s\n' \
    "$(date -Is)" \
    "$([ -s "${CASE1}" ] && echo complete || echo pending)" \
    "$([ -s "${CASE2}" ] && echo complete || echo pending)" \
    "$([ -s "${OMNI_SUMMARY}" ] && echo complete || echo pending)"
  sleep "${POLL_SECONDS}"
done

"${PYTHON}" -u -m pose_point_depth_mv.finalize_with_vggt_qualitative_outputs dev \
  --output_dir "${DEV}"

"${PYTHON}" - "${DEV}" "${OMNI}" "${COMPLETE}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

dev, omni, destination = map(Path, sys.argv[1:])
payload = {
    "format": "pose_point_depth_mv.with_vggt_qualitative_dev2_omni_complete.v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "passed": True,
    "dev_summary": str((dev / "summary.json").resolve(strict=True)),
    "omni_summary": str((omni / "summary.json").resolve(strict=True)),
}
destination.parent.mkdir(parents=True, exist_ok=True)
temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
os.replace(temporary, destination)
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY

echo "WITH-VGGT QUALITATIVE COMPLETION GUARD PASS"
