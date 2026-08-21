#!/usr/bin/env bash
set -euo pipefail

# Re-run one real dataset with official Native-SS + trained Native-SLat.
#
# The only required argument may be either:
#   1. a real dataset directory; or
#   2. an older reconstruction directory containing reconstruction_report.json.
#
# The generic default is step25000.  MODEL_STEP=8000 remains available for the
# fixed earlier diagnostic comparison.

INPUT_PATH=${1:?"usage: $0 DATASET_OR_OLD_RECONSTRUCTION_DIR"}

cd /home/zjr/Tracker

PY=${PY:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
GPU=${GPU:-0}
MODEL_STEP=${MODEL_STEP:-25000}
SEED=${SEED:-42}
RUN_TAG=${RUN_TAG:-spherical_v1}
DRY_RUN=${DRY_RUN:-0}
DIAGNOSTIC_BYPASS_POSE_MASK_QUALITY=${DIAGNOSTIC_BYPASS_POSE_MASK_QUALITY:-0}

case "${DIAGNOSTIC_BYPASS_POSE_MASK_QUALITY}" in
  0|1) ;;
  *)
    echo "ERROR: DIAGNOSTIC_BYPASS_POSE_MASK_QUALITY must be 0 or 1" >&2
    exit 64
    ;;
esac

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/zjr/Tracker/pose_point_depth_mv/outputs/可视AR}
SS_RUN=/data/zjr/proobjaverse_official_native_ss_train2000_20260815_v1
SOURCE=/data/zjr/proobjaverse_official_slat_train2000_20260813_v1

SS_REPORT=${SS_RUN}/dev64_step2000_eval16_64_seed424344_6gpu_v1/aggregate_v1/report.json
STOCK_FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json

case "${MODEL_STEP}" in
  8000)
    SLAT_CKPT=${SOURCE}/B_condition_lora_train2000_step8000_seed42_4gpu_v1/checkpoints/step_008000.pt
    EXPECTED_SLAT_SHA=49edb3bbdbd86b10c5eea14e9c80a9996076b6fd65a459db12b130b6560bda4d
    BRIDGE=${SS_RUN}/dev48_newss2000_stock_and_slat8000_mesh_seed424344_5gpu_v1/aggregate_v1/report.json
    ;;
  25000)
    SLAT_CKPT=/data/zjr/slat_train2000_trajectory_archives/slat_train2000_trajectory_step10000_25000_strict_fix1_v1/checkpoints/step_025000.pt
    EXPECTED_SLAT_SHA=5092422900fe7d1e467684f0168aaa2cce67c754f6a48ff33d91c3772b2bcf58
    BRIDGE=${SOURCE}/eval_trajectory_step15000_20000_25000_seed424344_5gpu_strict_fix1_v1/step_025000/dev48_predicted/aggregate_v1/report.json
    ;;
  *)
    echo "ERROR: MODEL_STEP must be 8000 or 25000; got ${MODEL_STEP}" >&2
    exit 64
    ;;
esac

for REQUIRED in "${PY}" "${SS_REPORT}" "${SLAT_CKPT}" "${STOCK_FREEZE}"; do
  if ! test -s "${REQUIRED}"; then
    echo "ERROR: missing required file: ${REQUIRED}" >&2
    exit 65
  fi
done

if ! test -s "${BRIDGE}"; then
  echo "ERROR: step${MODEL_STEP} cross-deployment bridge is not complete:" >&2
  echo "${BRIDGE}" >&2
  if test "${MODEL_STEP}" = 25000; then
    echo "For the currently available diagnostic, rerun with MODEL_STEP=8000." >&2
  fi
  exit 66
fi

ACTUAL_SLAT_SHA=$(sha256sum "${SLAT_CKPT}" | awk '{print $1}')
if test "${ACTUAL_SLAT_SHA}" != "${EXPECTED_SLAT_SHA}"; then
  echo "ERROR: SLat checkpoint SHA256 differs" >&2
  echo "expected=${EXPECTED_SLAT_SHA}" >&2
  echo "actual=${ACTUAL_SLAT_SHA}" >&2
  exit 67
fi

# Resolve the actual dataset and preserve its geometry/canonicalization mode.
# Every newly named run defaults to the versioned official-style spherical
# farthest-point selector.  Set VIEW_SELECTION_POLICY explicitly only when an
# old selector must be reproduced as a diagnostic.
mapfile -t RESOLVED < <(
  "${PY}" - "${INPUT_PATH}" "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

input_path = Path(sys.argv[1]).expanduser().resolve(strict=True)
output_root = Path(sys.argv[2]).expanduser().resolve()

report_path = None
if input_path.is_file():
    if input_path.name != "reconstruction_report.json":
        raise SystemExit(f"input file must be reconstruction_report.json: {input_path}")
    report_path = input_path
elif (input_path / "reconstruction_report.json").is_file():
    report_path = input_path / "reconstruction_report.json"

report = None
if report_path is not None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    dataset = Path(report["dataset_dir"]).expanduser().resolve(strict=True)
else:
    dataset = input_path
    historical = output_root / "reconstructions" / dataset.name / "reconstruction_report.json"
    if historical.is_file():
        report_path = historical
        report = json.loads(historical.read_text(encoding="utf-8"))

if not dataset.is_dir():
    raise SystemExit(f"resolved dataset is not a directory: {dataset}")

if report is not None:
    deployment = dict(report.get("deployment") or {})
    geometry = str(deployment.get("geometry_mode", "pose_mask"))
    policy = "object_spherical_farthest_valid_mask"
    gravity = deployment.get("gravity_up_w")
    min_points = int(deployment.get("min_object_points", 32))
    min_observations = int(deployment.get("min_mask_observations", 2))
    min_support = float(deployment.get("min_mask_support_ratio", 0.50))
    policy_source = "optimized_spherical_default; geometry_from=" + str(report_path)
elif (dataset / "capture_report.json").is_file():
    geometry = "pose_mask"
    policy = "object_spherical_farthest_valid_mask"
    gravity = [0.0, 1.0, 0.0]
    min_points = 32
    min_observations = 2
    min_support = 0.50
    policy_source = "phone_capture_default"
else:
    geometry = "point_mask"
    policy = "object_spherical_farthest_valid_mask"
    gravity = None
    min_points = 32
    min_observations = 2
    min_support = 0.50
    policy_source = "point_mask_optimized_spherical_default"

print(dataset)
print(geometry)
print(policy)
print("" if gravity is None else " ".join(str(float(v)) for v in gravity))
print(min_points)
print(min_observations)
print(min_support)
print(policy_source)
PY
)

DATASET=${RESOLVED[0]}
GEOMETRY_MODE=${GEOMETRY_MODE:-${RESOLVED[1]}}
VIEW_SELECTION_POLICY=${VIEW_SELECTION_POLICY:-${RESOLVED[2]}}
GRAVITY_UP_W=${GRAVITY_UP_W:-${RESOLVED[3]}}
MIN_OBJECT_POINTS=${MIN_OBJECT_POINTS:-${RESOLVED[4]}}
MIN_MASK_OBSERVATIONS=${MIN_MASK_OBSERVATIONS:-${RESOLVED[5]}}
MIN_MASK_SUPPORT_RATIO=${MIN_MASK_SUPPORT_RATIO:-${RESOLVED[6]}}
POLICY_SOURCE=${RESOLVED[7]}

DATASET_NAME=$(basename "${DATASET}")
SAFE_DATASET_NAME=$(printf '%s' "${DATASET_NAME}" | tr -c 'A-Za-z0-9._-' '_')
SESSION_ID=${SESSION_ID:-real_official_slat_step${MODEL_STEP}_retest_${SAFE_DATASET_NAME}_seed${SEED}_${RUN_TAG}}
OUTPUT_DIR=${OUTPUT_ROOT}/reconstructions/${SESSION_ID}

COMMAND=(
  "${PY}" -u -m pose_point_depth_mv.reconstruct_real_proobjaverse_official_ss_slat
  --dataset_dir "${DATASET}"
  --session_id "${SESSION_ID}"
  --output_root "${OUTPUT_ROOT}"
  --gpu "${GPU}"
  --geometry_mode "${GEOMETRY_MODE}"
  --view_selection_policy "${VIEW_SELECTION_POLICY}"
  --selected_view_count 8
  --min_object_points "${MIN_OBJECT_POINTS}"
  --min_mask_observations "${MIN_MASK_OBSERVATIONS}"
  --min_mask_support_ratio "${MIN_MASK_SUPPORT_RATIO}"
  --native_ss_report "${SS_REPORT}"
  --native_slat_checkpoint "${SLAT_CKPT}"
  --expected_slat_step "${MODEL_STEP}"
  --cross_deployment_bridge_report "${BRIDGE}"
  --stock_slat_freeze "${STOCK_FREEZE}"
  --seed "${SEED}"
  --amp_dtype bf16
)

if test -n "${GRAVITY_UP_W}"; then
  read -r -a GRAVITY_VALUES <<<"${GRAVITY_UP_W}"
  COMMAND+=(--gravity_up_w "${GRAVITY_VALUES[@]}")
fi

if test "${DIAGNOSTIC_BYPASS_POSE_MASK_QUALITY}" = 1; then
  COMMAND+=(--diagnostic_bypass_pose_mask_quality)
fi

echo "============================================================"
echo "Real official Native-SS + SLat retest"
echo "============================================================"
echo "input_path=${INPUT_PATH}"
echo "dataset=${DATASET}"
echo "policy_source=${POLICY_SOURCE}"
echo "geometry_mode=${GEOMETRY_MODE}"
echo "view_selection_policy=${VIEW_SELECTION_POLICY}"
echo "gravity_up_w=${GRAVITY_UP_W:-none}"
echo "model_step=${MODEL_STEP}"
echo "checkpoint_sha256=${ACTUAL_SLAT_SHA}"
echo "diagnostic_bypass_pose_mask_quality=${DIAGNOSTIC_BYPASS_POSE_MASK_QUALITY}"
echo "session_id=${SESSION_ID}"
echo "output_dir=${OUTPUT_DIR}"
printf 'command='
printf '%q ' "${COMMAND[@]}"
printf '\n'

if test "${DRY_RUN}" = 1; then
  echo "DRY RUN PASS"
  exit 0
fi

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ATTN_BACKEND=flash_attn
export SPCONV_ALGO=native

"${COMMAND[@]}"

REPORT=${OUTPUT_DIR}/reconstruction_report.json
test -s "${REPORT}"
"${PY}" - "${REPORT}" "${MODEL_STEP}" "${EXPECTED_SLAT_SHA}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
expected_step = int(sys.argv[2])
expected_sha = sys.argv[3]
assert report["passed"] is True
assert report["identity"]["expected_slat_step"] == expected_step
assert report["identity"]["native_slat_checkpoint_sha256"] == expected_sha
print(json.dumps({
    "passed": True,
    "run_dir": report["run_dir"],
    "runtime_o_obj": report["meshes"]["runtime_o_obj"],
    "world_obj": report["meshes"]["world_obj"],
    "world_glb": report["meshes"]["world_glb"],
    "preview": report["previews"]["contact_sheet"],
    "formal_claim_allowed": report["formal_claim_allowed"],
}, indent=2, ensure_ascii=False))
PY

echo "REAL DATA RETEST COMPLETE"
