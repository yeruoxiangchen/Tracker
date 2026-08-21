#!/usr/bin/env bash
set -euo pipefail

# Versioned replacement for the failed v1 strict comparison.  It preserves the
# v1 tree, reuses only its verified Native-SS coordinate artifacts, and binds
# every R/A/B/C surface metric to the exact target NPZ recorded by strict R.

PROJECT=${PROJECT:-/home/zjr/Tracker}
ROOT=${ROOT:-/data/zjr/proobjaverse_official_slat_train29861_20260817_v1}
EVAL_GPUS=${EVAL_GPUS:-4,5,6,7}
V1_ROOT=${V1_ROOT:-${ROOT}/eval_legacy_protocol2128_step10k30k60k70k_seed424344_4gpu_v1}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/eval_legacy_protocol2128_step10k30k60k_seed424344_4gpu_targetlocked_v2}
MASTER_LOG=${MASTER_LOG:-${ROOT}/logs/eval_legacy_protocol2128_step10k30k60k_4gpu_targetlocked_v2.launcher.log}

test -d "${V1_ROOT}/step_010000/legacy_dev48_predicted_training_overlap"

# The two completed 10K GT-support diagnostics do not depend on the target NPZ
# used by the strict R/A/B/C comparison.  Preserve their exact reports/artifacts
# by hard-linking the finalized directories into v2 instead of spending GPU time
# recomputing them.  The old v1 tree remains untouched.
for group in train64_gt legacy_dev64_gt_training_overlap; do
  source_group="${V1_ROOT}/step_010000/${group}"
  target_group="${OUTPUT_ROOT}/step_010000/${group}"
  test -s "${source_group}/aggregate_v1/report.json"
  if [[ ! -e "${target_group}" ]]; then
    mkdir -p "$(dirname "${target_group}")"
    cp -al "${source_group}" "${target_group}"
  fi
  test -s "${target_group}/aggregate_v1/report.json"
done
mkdir -p "${OUTPUT_ROOT}"
printf '%s\n' \
  "10K finalized GT-support groups hard-linked from: ${V1_ROOT}" \
  "Only Dev48 predicted-support mesh metrics are recomputed against frozen strict target NPZs." \
  >"${OUTPUT_ROOT}/TARGET_LOCKED_V2_PROVENANCE.txt"

exec env \
  PROJECT="${PROJECT}" \
  EVAL_GPUS="${EVAL_GPUS}" \
  EVAL_TAG=4gpu_targetlocked \
  STEPS=10000,30000,60000 \
  PRESERVE_INTERRUPTED_OUTPUTS=1 \
  SS_COORD_REUSE_ROOT="${V1_ROOT}/step_010000/legacy_dev48_predicted_training_overlap" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  MASTER_LOG="${MASTER_LOG}" \
  EVAL_GPU_RESERVATION_ROOT="${ROOT}/logs/gpu_reservations_slat30k_4gpu_targetlocked_v2" \
  bash "${PROJECT}/pose_point_depth_mv/background_jobs/run_proobjaverse_slat29861_legacy_eval_4gpu.sh"
