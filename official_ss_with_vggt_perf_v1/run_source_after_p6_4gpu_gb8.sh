#!/usr/bin/env bash
set -euo pipefail

# Continue only after P6/P6-repair has produced identity-valid full caches.
# Uses GPUs 0--3 so the independent 30K checkpoint evaluation can use 4--7.

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
PYTHON=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
RUN_ROOT=${RUN_ROOT:-/data/zjr/proobjaverse_official_native_ss_train2000_with_vggt_20260817_v1}
PIPELINE_GPUS=${PIPELINE_GPUS:-0,1,2,3}
POLL_SECONDS=${POLL_SECONDS:-30}
MAX_IDLE_MEMORY_MIB=${MAX_IDLE_MEMORY_MIB:-1024}

TRAIN_CACHE=${TRAIN_CACHE:-${RUN_ROOT}/cache_train2000_official_ss_with_vggt_sidecar_historical_v2_fix2_v1}
DEV_CACHE=${DEV_CACHE:-${RUN_ROOT}/cache_dev64_official_ss_with_vggt_sidecar_historical_v2_fix2_v1}
TRAIN_OUTPUT=${TRAIN_OUTPUT:-${RUN_ROOT}/VSS_with_vggt_train2000_step2000_seed42_4gpu_gb8_v1}
TRAIN_LOG=${TRAIN_LOG:-${RUN_ROOT}/logs/VSS_with_vggt_train2000_step2000_seed42_4gpu_gb8_v1.log}
STATUS=${STATUS:-${RUN_ROOT}/logs/P7_P17_4gpu_gb8_v1.status}
EXIT_CODE=${EXIT_CODE:-${RUN_ROOT}/logs/P7_P17_4gpu_gb8_v1.exit_code}

mkdir -p "${RUN_ROOT}/logs"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$PWD:$PWD/ReconViaGen:$PWD/ReconViaGen/wheels/vggt"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export ATTN_BACKEND=${ATTN_BACKEND:-flash_attn}
export SPCONV_ALGO=${SPCONV_ALGO:-native}

IFS=, read -r -a GPU_ARRAY <<<"${PIPELINE_GPUS}"
if (( ${#GPU_ARRAY[@]} != 4 )); then
  echo "ERROR: PIPELINE_GPUS must contain exactly four GPUs" >&2
  exit 90
fi

CURRENT_STAGE=preflight
write_status() {
  local state=$1 temporary="${STATUS}.tmp.$$"
  {
    printf 'state=%s\n' "${state}"
    printf 'stage=%s\n' "${CURRENT_STAGE}"
    printf 'time_utc=%s\n' "$(date -u -Is)"
  } >"${temporary}"
  mv "${temporary}" "${STATUS}"
}
finish() {
  local rc=$?
  trap - EXIT
  [[ ${rc} -eq 0 ]] && write_status PASS || write_status FAIL
  printf '%s\n' "${rc}" >"${EXIT_CODE}"
  echo "===== four-GPU P7--P17 exit=${rc} stage=${CURRENT_STAGE} ====="
  exit "${rc}"
}
trap finish EXIT

wait_for_gpus() {
  local label=$1 idle gpu used
  CURRENT_STAGE="wait_${label}"
  write_status WAITING
  while :; do
    idle=1
    for gpu in "${GPU_ARRAY[@]}"; do
      used=$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | awk 'NR==1 {print int($1)}')
      (( used <= MAX_IDLE_MEMORY_MIB )) || idle=0
    done
    (( idle == 1 )) && break
    echo "[$(date -u -Is)] waiting for GPUs=${PIPELINE_GPUS} (${label})"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
    sleep "${POLL_SECONDS}"
  done
}

CURRENT_STAGE=P7_validate_repaired_caches
write_status RUNNING
RUN_ROOT="${RUN_ROOT}" PROJECT_ROOT="${PROJECT_ROOT}" PYTHON="${PYTHON}" \
  bash "${PROJECT_ROOT}/official_ss_with_vggt_perf_v1/validate_full_caches.sh"
test -s "${TRAIN_CACHE}/with_vggt_ss_manifest.json"
test -s "${DEV_CACHE}/with_vggt_ss_manifest.json"

if [[ -e "${TRAIN_OUTPUT}" ]]; then
  echo "ERROR: four-GPU fresh output already exists; never overwrite/resume automatically: ${TRAIN_OUTPUT}" >&2
  exit 91
fi

wait_for_gpus before_P8_4gpu
CURRENT_STAGE=P8_train_4gpu_global_batch8
write_status RUNNING
set +e
CACHE_MANIFEST="${TRAIN_CACHE}/with_vggt_ss_manifest.json" \
OUTPUT_DIR="${TRAIN_OUTPUT}" \
TRAIN_GPUS="${PIPELINE_GPUS}" \
PROJECT_ROOT="${PROJECT_ROOT}" \
  bash "${PROJECT_ROOT}/official_ss_with_vggt_perf_v1/run_train_4gpu_gb8.sh" \
  2>&1 | tee "${TRAIN_LOG}"
train_rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "${train_rc}" >"${TRAIN_LOG}.exit_code"
(( train_rc == 0 )) || exit "${train_rc}"

CURRENT_STAGE=P8_final_guard
write_status RUNNING
"${PYTHON}" - "${TRAIN_OUTPUT}/report.json" <<'PY'
import json
import sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
assert r["format"] == "official_ss_with_vggt_perf_v1.native_ss_genrecon.v2"
assert r["completed"] is True and r["passed"] is True and r["step"] == 2000
assert r["global_micro_samples"] == 16000
assert r["data_identity"]["object_count"] == 2000
assert r["model_summary"]["stock_floor"] == "VSS0"
assert r["model_summary"]["fresh_initialization_only"] is True
assert r["model_summary"]["runtime_input_policy"]["ddp_device_ids"] is None
distributed = r["model_summary"]["optimization"]["distributed"]
assert distributed["world_size"] == 4
assert distributed["per_rank_grad_accum"] == 2
assert distributed["global_effective_batch"] == 8
print({"passed": True, "step": r["step"], "distributed": distributed})
PY

for step in 500 1000 1500 2000; do
  pad=$(printf '%06d' "${step}")
  test -s "${TRAIN_OUTPUT}/checkpoints/step_${pad}.pt"
done

CURRENT_STAGE=P10_P17_posttrain_4gpu
write_status RUNNING
SS_ROOT="${RUN_ROOT}" \
SS_TRAIN="${TRAIN_OUTPUT}" \
SS_CHECKPOINT="${TRAIN_OUTPUT}/checkpoints/step_002000.pt" \
SS_TRAIN_CACHE="${TRAIN_CACHE}/with_vggt_ss_manifest.json" \
SS_DEV_CACHE="${DEV_CACHE}/with_vggt_ss_manifest.json" \
CALIBRATION_ROOT="${RUN_ROOT}/dev64_VSS_step2000_4gpu_gb8_calibrate0_16_seed424344_v1" \
CALIBRATION="${RUN_ROOT}/dev64_VSS_step2000_4gpu_gb8_calibrate0_16_seed424344_v1/calibration.json" \
SS_DEV48_ROOT="${RUN_ROOT}/dev48_VSS_step2000_seed424344_4gpu_gb8_v1" \
VSS_REPORT="${RUN_ROOT}/dev48_VSS_step2000_seed424344_4gpu_gb8_v1/aggregate_v1/report.json" \
POST_STATUS="${RUN_ROOT}/logs/P10_P17_posttrain_eval_4gpu_gb8_v1.status" \
POST_EXIT_CODE="${RUN_ROOT}/logs/P10_P17_posttrain_eval_4gpu_gb8_v1.exit_code" \
POST_SUMMARY="${RUN_ROOT}/P10_P17_posttrain_eval_4gpu_gb8_v1_summary.json" \
EVAL_GPUS="${PIPELINE_GPUS}" \
EVAL_TAG=4gpu_ss4gpu_gb8 \
SLAT_STEPS=10000,15000 \
PROJECT_ROOT="${PROJECT_ROOT}" PYTHON="${PYTHON}" \
  bash "${PROJECT_ROOT}/official_ss_with_vggt_perf_v1/run_source_posttrain_p10_p17_eval_pipeline.sh"

CURRENT_STAGE=complete
write_status PASS
echo "WITH-VGGT SS P7--P17 FOUR-GPU/GLOBAL-BATCH-8 PIPELINE COMPLETE"
