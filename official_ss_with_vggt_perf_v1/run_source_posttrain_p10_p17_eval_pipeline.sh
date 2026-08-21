#!/usr/bin/env bash
set -euo pipefail

# Post-training evaluation pipeline for the official with-VGGT VSS + V route.
# Scientific-negative reports are preserved and do not count as program
# crashes. Identity/runtime-integrity failures stop the pipeline immediately.

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
PYTHON=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
SS_ROOT=${SS_ROOT:-/data/zjr/proobjaverse_official_native_ss_train2000_with_vggt_20260817_v1}
SLAT_ROOT=${SLAT_ROOT:-/data/zjr/proobjaverse_official_slat_train2000_20260813_v1}
SLAT_RUN=${SLAT_RUN:-${SLAT_ROOT}/V_with_vggt_train2000_step15000_seed42_8gpu_strict_perf_v1_v1}
SLAT_STEPS=${SLAT_STEPS:-10000,15000}
EVAL_GPUS=${EVAL_GPUS:-0,1,2,3,4,5,6,7}
EVAL_TAG=${EVAL_TAG:-8gpu}
EVALUATE_TRAIN64=${EVALUATE_TRAIN64:-1}
POLL_SECONDS=${POLL_SECONDS:-30}
MAX_IDLE_MEMORY_MIB=${MAX_IDLE_MEMORY_MIB:-1024}
EVAL_GPU_RESERVATION_ROOT=${EVAL_GPU_RESERVATION_ROOT:-${SS_ROOT}/logs/P10_P17_gpu_reservations_${EVAL_TAG}}

SS_TRAIN=${SS_TRAIN:-${SS_ROOT}/VSS_with_vggt_train2000_step2000_seed42_8gpu_v1}
SS_CHECKPOINT=${SS_CHECKPOINT:-${SS_TRAIN}/checkpoints/step_002000.pt}
SS_TRAIN_CACHE=${SS_TRAIN_CACHE:-${SS_ROOT}/cache_train2000_official_ss_with_vggt_sidecar_historical_v2_fix2_v1/with_vggt_ss_manifest.json}
SS_DEV_CACHE=${SS_DEV_CACHE:-${SS_ROOT}/cache_dev64_official_ss_with_vggt_sidecar_historical_v2_fix2_v1/with_vggt_ss_manifest.json}
CALIBRATION_ROOT=${CALIBRATION_ROOT:-${SS_ROOT}/dev64_VSS_step2000_calibrate0_16_seed424344_v1}
CALIBRATION=${CALIBRATION:-${CALIBRATION_ROOT}/calibration.json}
SS_DEV48_ROOT=${SS_DEV48_ROOT:-${SS_ROOT}/dev48_VSS_step2000_seed424344_6gpu_v1}
VSS_REPORT=${VSS_REPORT:-${SS_DEV48_ROOT}/aggregate_v1/report.json}

SLAT_TRAIN64_CACHE=${SLAT_TRAIN64_CACHE:-${SLAT_ROOT}/cache_train64_protocol2128_views8_with_vggt_sidecar_v1}
SLAT_DEV64_CACHE=${SLAT_DEV64_CACHE:-${SLAT_ROOT}/cache_dev64_protocol2128_views8_with_vggt_sidecar_v1}
STOCK_FREEZE=${STOCK_FREEZE:-/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json}

POST_STATUS=${POST_STATUS:-${SS_ROOT}/logs/P10_P17_posttrain_eval_v1.status}
POST_EXIT_CODE=${POST_EXIT_CODE:-${SS_ROOT}/logs/P10_P17_posttrain_eval_v1.exit_code}
POST_SUMMARY=${POST_SUMMARY:-${SS_ROOT}/P10_P17_posttrain_eval_v1_summary.json}

mkdir -p "${SS_ROOT}/logs"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$PWD:$PWD/ReconViaGen:$PWD/ReconViaGen/wheels/vggt"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export ATTN_BACKEND=${ATTN_BACKEND:-flash_attn}
export SPCONV_ALGO=${SPCONV_ALGO:-native}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba_cache}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Keep a tiny, idle CUDA context on every selected GPU for the lifetime of the
# complete post-training pipeline.  Evaluation workers can exit between shards
# and waves without making a card appear free to other users.
source "${PROJECT_ROOT}/pose_point_depth_mv/background_jobs/eval_gpu_reservation.sh"

IFS=, read -r -a GPU_ARRAY <<<"${EVAL_GPUS}"
EVAL_GPU_COUNT=${#GPU_ARRAY[@]}
if (( EVAL_GPU_COUNT != 8 && EVAL_GPU_COUNT != 4 && EVAL_GPU_COUNT != 2 )); then
  echo "ERROR: post-training pipeline requires exactly 8, 4, or 2 GPU indices" >&2
  exit 90
fi
IFS=, read -r -a STEP_ARRAY <<<"${SLAT_STEPS}"
if (( ${#STEP_ARRAY[@]} <= 0 )); then
  echo "ERROR: SLAT_STEPS is empty" >&2
  exit 91
fi
for step in "${STEP_ARRAY[@]}"; do
  case "${step}" in
    10000|15000) ;;
    *) echo "ERROR: registered with-VGGT SLat eval steps are 10000/15000; got=${step}" >&2; exit 92 ;;
  esac
done
case "${EVALUATE_TRAIN64}" in
  0|1) ;;
  *) echo "ERROR: EVALUATE_TRAIN64 must be 0 or 1" >&2; exit 92 ;;
esac

CURRENT_STAGE=preflight
write_status() {
  local state=$1
  local temporary="${POST_STATUS}.tmp.$$"
  {
    printf 'state=%s\n' "${state}"
    printf 'stage=%s\n' "${CURRENT_STAGE}"
    printf 'time_utc=%s\n' "$(date -u -Is)"
  } >"${temporary}"
  mv "${temporary}" "${POST_STATUS}"
}
finish() {
  local status=$?
  trap - EXIT
  stop_eval_gpu_reservations
  if (( status == 0 )); then
    CURRENT_STAGE=complete
    write_status PASS
  else
    write_status FAIL
  fi
  printf '%s\n' "${status}" >"${POST_EXIT_CODE}"
  echo "===== posttrain eval exit=${status} stage=${CURRENT_STAGE} time=$(date -u -Is) ====="
  exit "${status}"
}
trap finish EXIT

gpu_memory_mib() {
  nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk 'NR == 1 {print int($1)}'
}
resources_are_idle() {
  local gpu used
  # Gate only the explicitly selected devices.  Unrelated trainers on other
  # physical GPUs must not block a two-GPU evaluation route.
  for gpu in "${GPU_ARRAY[@]}"; do
    if ! used=$(gpu_memory_mib "${gpu}"); then
      return 1
    fi
    if (( used > MAX_IDLE_MEMORY_MIB )); then
      return 1
    fi
  done
  return 0
}
wait_for_idle_resources() {
  local label=$1
  CURRENT_STAGE="wait_${label}"
  write_status WAITING
  while ! resources_are_idle; do
    echo "[$(date -u -Is)] waiting for exclusive eval GPUs (${label})"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits || true
    sleep "${POLL_SECONDS}"
  done
  echo "[$(date -u -Is)] exclusive eval GPU gate PASS (${label})"
}

validate_ss_training() {
  "${PYTHON}" - "${SS_TRAIN}/report.json" <<'PY'
import json
import sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
assert r["format"] == "official_ss_with_vggt_perf_v1.native_ss_genrecon.v2"
assert r["completed"] is True and r["passed"] is True and r["step"] == 2000
assert r["global_micro_samples"] == 16000
assert r["data_identity"]["object_count"] == 2000
assert r["model_summary"]["stock_floor"] == "VSS0"
assert r["model_summary"]["runtime_input_policy"]["ddp_device_ids"] is None
print({"passed": True, "ss_training_report": sys.argv[1]})
PY
}

validate_calibration() {
  "${PYTHON}" - "${CALIBRATION}" <<'PY'
import json
import sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
assert r["format"] == "official_ss_with_vggt_perf_v1.calibration.v1"
assert r["passed"] is True and r["selected"] is not None
p = r["protocol"]
assert p["checkpoint_step"] == 2000
assert p["object_start"] == 0 and p["object_end"] == 16
assert p["joint_seeds"] == [42, 43, 44]
assert p["weights"] == "ema"
print({"passed": True, "cfg_strength": r["selected"]["cfg_strength"]})
PY
}

validate_ss_dev48() {
  "${PYTHON}" -m official_ss_with_vggt_perf_v1.artifact_validation \
    vss-aggregate \
    --report "${VSS_REPORT}" \
    --checkpoint "${SS_CHECKPOINT}" \
    --calibration "${CALIBRATION}" \
    --expected-objects 48 \
    --joint-seeds 42,43,44 \
    --checkpoint-step 2000
}

validate_slat_dev_cache() {
  "${PYTHON}" - "${SLAT_DEV64_CACHE}/report.json" <<'PY'
import json
import sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
assert r["format"] == "pose_point_depth_mv.proobjaverse_official_slat_with_vggt_cache.v1"
assert r["passed"] is True and r["complete"] is True
assert r["split"] == "dev" and r["object_count"] == 64
assert r["selected_views"] == 8 and r["vggt_forward_call_count"] == 64
assert r["vggt_model_executed"] is True
assert r["vggt_camera_consumed"] is False
assert r["known_K_T_replaced"] is False
assert r["same_frozen_view_ids_as_base"] is True
assert r["native_ss_changed"] is False and r["base_cache_rewritten"] is False
print({"passed": True, "pair_identity": r["pair_identity"]})
PY
}

validate_endpoint_step() {
  local step=$1
  local root="${SLAT_ROOT}/eval_with_vggt_VSS2000_Vstep${step}_predicted_support_seed424344_${EVAL_TAG}_v1"
  local pad checkpoint
  pad=$(printf '%06d' "${step}")
  checkpoint="${SLAT_RUN}/checkpoints/step_${pad}.pt"
  if (( EVALUATE_TRAIN64 == 1 )); then
    "${PYTHON}" -m official_ss_with_vggt_perf_v1.artifact_validation \
      endpoint-aggregate \
      --report "${root}/train64_predicted/aggregate_v1/report.json" \
      --split train --object-start 0 --object-end 64 --expected-objects 64 \
      --joint-seeds 42,43,44 --slat-step "${step}" \
      --slat-checkpoint "${checkpoint}" --ss-cache "${SS_TRAIN_CACHE}" \
      --vss-report "${VSS_REPORT}"
  fi
  "${PYTHON}" -m official_ss_with_vggt_perf_v1.artifact_validation \
    endpoint-aggregate \
    --report "${root}/dev48_predicted/aggregate_v1/report.json" \
    --split dev --object-start 16 --object-end 64 --expected-objects 48 \
    --joint-seeds 42,43,44 --slat-step "${step}" \
    --slat-checkpoint "${checkpoint}" --ss-cache "${SS_DEV_CACHE}" \
    --vss-report "${VSS_REPORT}"
}

CURRENT_STAGE=preflight
write_status RUNNING
test -s "${SS_TRAIN}/report.json"
for checkpoint_step in 500 1000 1500 2000; do
  checkpoint_pad=$(printf '%06d' "${checkpoint_step}")
  test -s "${SS_TRAIN}/checkpoints/step_${checkpoint_pad}.pt"
done
test -s "${SS_TRAIN}/checkpoints/last.pt"
if (( EVALUATE_TRAIN64 == 1 )); then
  test -s "${SS_TRAIN_CACHE}"
  test -s "${SLAT_TRAIN64_CACHE}/with_vggt_slat_manifest.json"
  test -s "${SLAT_TRAIN64_CACHE}/with_vggt_lifting_manifest.json"
fi
test -s "${SS_DEV_CACHE}"
test -s "${STOCK_FREEZE}"
validate_ss_training
for step in "${STEP_ARRAY[@]}"; do
  pad=$(printf '%06d' "${step}")
  test -s "${SLAT_RUN}/checkpoints/step_${pad}.pt"
done

wait_for_idle_resources before_P10
start_eval_gpu_reservations \
  "${EVAL_GPUS}" "${EVAL_GPU_RESERVATION_ROOT}" \
  "with_vggt_posttrain_${EVAL_TAG}" "${PYTHON}" "${PROJECT_ROOT}"

CURRENT_STAGE=P10_calibrate_VSS_CFG
write_status RUNNING
if [[ -s "${CALIBRATION}" ]]; then
  validate_calibration
elif [[ -e "${CALIBRATION_ROOT}" ]]; then
  echo "ERROR: incomplete calibration output exists: ${CALIBRATION_ROOT}" >&2
  exit 93
else
  CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" "${PYTHON}" -u \
    -m official_ss_with_vggt_perf_v1.evaluate \
      --mode calibrate \
      --cache_manifest "${SS_DEV_CACHE}" \
      --checkpoint "${SS_CHECKPOINT}" \
      --output_dir "${CALIBRATION_ROOT}" \
      --object_start 0 --object_end 16 \
      --joint_seeds 42,43,44 \
      --candidate_cfg_strengths 1,3,5 \
      --weights ema --steps 25 --cfg_interval 0.5,1.0 \
      --guidance_rescale 0.0 --rescale_t 3.0 --amp_dtype bf16 \
      --min_iou_gain_mean -1 --min_recall_gain_mean -1 \
      --min_latent_mse_gain_mean -1 --min_count_ratio 0.1 --max_count_ratio 10
  test -s "${CALIBRATION}"
  validate_calibration
fi

wait_for_idle_resources before_P11

CURRENT_STAGE=P11_VSS_heldout_Dev48
write_status RUNNING
if [[ -s "${VSS_REPORT}" ]]; then
  validate_ss_dev48
elif [[ -e "${SS_DEV48_ROOT}" ]]; then
  echo "ERROR: incomplete VSS Dev48 output exists: ${SS_DEV48_ROOT}" >&2
  exit 94
else
  if (( EVAL_GPU_COUNT == 2 )); then
    CACHE_MANIFEST="${SS_DEV_CACHE}" \
    CHECKPOINT="${SS_CHECKPOINT}" \
    CALIBRATION="${CALIBRATION}" \
    OUTPUT_ROOT="${SS_DEV48_ROOT}" \
    EVAL_GPUS="${EVAL_GPUS}" \
    PROJECT_ROOT="${PROJECT_ROOT}" PYTHON="${PYTHON}" \
      bash "${PROJECT_ROOT}/official_ss_with_vggt_perf_v1/run_eval_dev48_2gpu.sh"
  elif (( EVAL_GPU_COUNT == 4 )); then
    CACHE_MANIFEST="${SS_DEV_CACHE}" \
    CHECKPOINT="${SS_CHECKPOINT}" \
    CALIBRATION="${CALIBRATION}" \
    OUTPUT_ROOT="${SS_DEV48_ROOT}" \
    EVAL_GPUS="${EVAL_GPUS}" \
    PROJECT_ROOT="${PROJECT_ROOT}" PYTHON="${PYTHON}" \
      bash "${PROJECT_ROOT}/official_ss_with_vggt_perf_v1/run_eval_dev48_4gpu.sh"
  else
    CACHE_MANIFEST="${SS_DEV_CACHE}" \
    CHECKPOINT="${SS_CHECKPOINT}" \
    CALIBRATION="${CALIBRATION}" \
    OUTPUT_ROOT="${SS_DEV48_ROOT}" \
    EVAL_GPUS="${GPU_ARRAY[0]},${GPU_ARRAY[1]},${GPU_ARRAY[2]},${GPU_ARRAY[3]},${GPU_ARRAY[4]},${GPU_ARRAY[5]}" \
    PROJECT_ROOT="${PROJECT_ROOT}" PYTHON="${PYTHON}" \
      bash "${PROJECT_ROOT}/official_ss_with_vggt_perf_v1/run_eval_dev48_6gpu.sh"
  fi
  test -s "${VSS_REPORT}"
  validate_ss_dev48
fi

wait_for_idle_resources before_P13

CURRENT_STAGE=P13_build_Dev64_with_VGGT_SLat_sidecar
write_status RUNNING
if [[ -s "${SLAT_DEV64_CACHE}/report.json" ]] && validate_slat_dev_cache; then
  echo "Dev64 with-VGGT SLat sidecar already complete; reusing immutable cache"
else
  SPLIT_MANIFEST="${SLAT_ROOT}/protocol2128_train2000_v1/dev.json" \
  BASE_SLAT_MANIFEST="${SLAT_ROOT}/cache_dev64_protocol2128_views8_v1/slat_manifest.json" \
  BASE_LIFTING_MANIFEST="${SLAT_ROOT}/cache_dev64_protocol2128_views8_v1/lifting_manifest.json" \
  OUTPUT_DIR="${SLAT_DEV64_CACHE}" \
  BUILD_GPUS="${EVAL_GPUS}" MAX_OBJECTS=0 PYTHON="${PYTHON}" \
    bash "${PROJECT_ROOT}/pose_point_depth_mv/background_jobs/run_proobjaverse_official_slat_with_vggt_cache.sh"
  validate_slat_dev_cache
fi

for step in "${STEP_ARRAY[@]}"; do
  wait_for_idle_resources "before_endpoint_step${step}"
  CURRENT_STAGE="P15_endpoint_Train64_Dev48_step${step}"
  write_status RUNNING
  SLAT_STEP="${step}" \
  SLAT_ROOT="${SLAT_ROOT}" SS_ROOT="${SS_ROOT}" SLAT_RUN="${SLAT_RUN}" \
  VSS_RUN="${SS_TRAIN}" VSS_REPORT="${VSS_REPORT}" \
  STOCK_FREEZE="${STOCK_FREEZE}" EVAL_GPUS="${EVAL_GPUS}" \
  EVALUATE_TRAIN64="${EVALUATE_TRAIN64}" \
  OUTPUT_ROOT="${SLAT_ROOT}/eval_with_vggt_VSS2000_Vstep${step}_predicted_support_seed424344_${EVAL_TAG}_v1" \
  PROJECT_ROOT="${PROJECT_ROOT}" PYTHON="${PYTHON}" \
    bash "${PROJECT_ROOT}/official_ss_with_vggt_perf_v1/run_predicted_support_endpoint_train64_dev48.sh"
  validate_endpoint_step "${step}"
done

CURRENT_STAGE=write_summary
write_status RUNNING
"${PYTHON}" - "${VSS_REPORT}" "${POST_SUMMARY}" "${SLAT_ROOT}" "${SLAT_STEPS}" "${EVAL_TAG}" "${EVALUATE_TRAIN64}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

vss_path, output_path, slat_root, step_csv, eval_tag, evaluate_train64_text = sys.argv[1:]
evaluate_train64 = bool(int(evaluate_train64_text))
vss = json.load(open(vss_path, encoding="utf-8"))
steps = [int(value) for value in step_csv.split(",") if value]
payload = {
    "format": "official_ss_with_vggt_perf_v1.posttrain_eval_summary.v1",
    "completed": True,
    "program_integrity_passed": True,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "vss_dev48": {
        "report": vss_path,
        "science_passed": vss["passed"],
        "objects": vss["object_count"],
        "records": vss["record_count"],
    },
    "evaluation_scope": {
        "train64_predicted_support": evaluate_train64,
        "dev48_predicted_support": True,
        "slat_gt_support_input": False,
        "official_gt_mesh_metric_reference": True,
    },
    "endpoint_steps": {},
}
for step in steps:
    root = Path(slat_root) / (
        f"eval_with_vggt_VSS2000_Vstep{step}_predicted_support_seed424344_{eval_tag}_v1"
    )
    dev_path = root / "dev48_predicted/aggregate_v1/report.json"
    dev = json.load(open(dev_path, encoding="utf-8"))
    step_payload = {
        "dev48_report": str(dev_path),
        "dev48_runtime_integrity_passed": dev["passed"],
        "dev48_full_endpoint_science_passed": dev["decision"][
            "native_ss_trained_slat_end_to_end_passed"
        ],
    }
    if evaluate_train64:
        train_path = root / "train64_predicted/aggregate_v1/report.json"
        train = json.load(open(train_path, encoding="utf-8"))
        step_payload.update({
            "train64_report": str(train_path),
            "train64_runtime_integrity_passed": train["passed"],
            "train64_full_endpoint_gate_passed": train["decision"][
                "native_ss_trained_slat_end_to_end_passed"
            ],
        })
    payload["endpoint_steps"][str(step)] = step_payload
Path(output_path).write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY

echo "P10--P17 WITH-VGGT SS/SLAT POSTTRAIN EVALUATION COMPLETE"
