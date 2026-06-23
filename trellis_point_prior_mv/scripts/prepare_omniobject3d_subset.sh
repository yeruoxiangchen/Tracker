#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zjr/Tracker}"
PY="${PY:-/home/zjr/anaconda3/envs/reconviagen/bin/python}"

OMNI_SOURCE_ROOT="${OMNI_SOURCE_ROOT:-/data/OmniObject3D}"
OMNI_OUTPUT_ROOT="${OMNI_OUTPUT_ROOT:-/data/trellis_point_prior_mv/omniobject3d_subset}"
DEMO_OUTPUT_ROOT="${DEMO_OUTPUT_ROOT:-${ROOT}/trellis_point_prior_mv/demo_assets/omniobject3d_subset}"

MAX_OBJECTS="${MAX_OBJECTS:-20}"
MAX_FRAMES="${MAX_FRAMES:-32}"
MIN_FRAMES="${MIN_FRAMES:-8}"
CATEGORIES="${CATEGORIES:-}"
SCAN_DEPTH="${SCAN_DEPTH:-4}"
ALLOW_FULL_MASKS="${ALLOW_FULL_MASKS:-0}"
ALLOW_MISSING_POSE="${ALLOW_MISSING_POSE:-0}"
OVERWRITE="${OVERWRITE:-1}"

if [[ ! -d "${OMNI_SOURCE_ROOT}" ]]; then
  echo "[prepare_omniobject3d_subset][ERROR] OmniObject3D source root not found: ${OMNI_SOURCE_ROOT}" >&2
  echo "Download/extract OmniObject3D under /data first, or set OMNI_SOURCE_ROOT." >&2
  exit 2
fi

EXTRA_ARGS=()
if [[ "${ALLOW_FULL_MASKS}" == "1" ]]; then
  EXTRA_ARGS+=(--allow_full_masks)
fi
if [[ "${ALLOW_MISSING_POSE}" == "1" ]]; then
  EXTRA_ARGS+=(--allow_missing_pose)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

"${PY}" -u "${ROOT}/trellis_point_prior_mv/prepare_omniobject3d_subset.py" \
  --source_root "${OMNI_SOURCE_ROOT}" \
  --output_root "${OMNI_OUTPUT_ROOT}" \
  --demo_output_root "${DEMO_OUTPUT_ROOT}" \
  --max_objects "${MAX_OBJECTS}" \
  --max_frames "${MAX_FRAMES}" \
  --min_frames "${MIN_FRAMES}" \
  --scan_depth "${SCAN_DEPTH}" \
  --categories "${CATEGORIES}" \
  "${EXTRA_ARGS[@]}"

echo "[prepare_omniobject3d_subset] manifest=${OMNI_OUTPUT_ROOT}/dataset_manifest.json"
echo "[prepare_omniobject3d_subset] demo=${DEMO_OUTPUT_ROOT}"
