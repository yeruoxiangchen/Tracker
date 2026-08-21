#!/usr/bin/env bash
set -euo pipefail

# Unattended source-server pipeline.  It may be launched while the with-VGGT
# SLat job is still running: this process only polls until all requested GPUs
# are genuinely idle.  It then establishes fresh exclusive 2-GPU DDP evidence,
# builds/validates the full caches, and starts the formal 8-GPU SS trajectory.

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
PYTHON=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
TORCHRUN=${TORCHRUN:-/home/zjr/anaconda3/envs/reconviagen/bin/torchrun}
RUN_ROOT=${RUN_ROOT:-/data/zjr/proobjaverse_official_native_ss_train2000_with_vggt_20260817_v1}
PIPELINE_GPUS=${PIPELINE_GPUS:-0,1,2,3,4,5,6,7}
POSTTRAIN_SLAT_STEPS=${POSTTRAIN_SLAT_STEPS:-10000,15000}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_IDLE_MEMORY_MIB=${MAX_IDLE_MEMORY_MIB:-1024}

P4_REPORT=${P4_REPORT:-${RUN_ROOT}/VSS_train8_step2_seed42_1gpu_smoke_step0exact_v2_v1/report.json}
P5_CACHE=${P5_CACHE:-${RUN_ROOT}/cache_train8_official_ss_with_vggt_sidecar_historical_v2_fix2_v1/with_vggt_ss_manifest.json}
# The earlier non-exclusive v2_v1 directory is immutable failure evidence.
P5_OUTPUT=${P5_OUTPUT:-${RUN_ROOT}/VSS_train8_step2_seed42_2gpu_ddp_smoke_step0exact_v2_exclusive_v1}
P5_LOG=${P5_LOG:-${RUN_ROOT}/logs/VSS_train8_step2_seed42_2gpu_ddp_smoke_step0exact_v2_exclusive_v1.log}

TRAIN_CACHE=${TRAIN_CACHE:-${RUN_ROOT}/cache_train2000_official_ss_with_vggt_sidecar_historical_v2_fix2_v1}
DEV_CACHE=${DEV_CACHE:-${RUN_ROOT}/cache_dev64_official_ss_with_vggt_sidecar_historical_v2_fix2_v1}
TRAIN_OUTPUT=${TRAIN_OUTPUT:-${RUN_ROOT}/VSS_with_vggt_train2000_step2000_seed42_8gpu_v1}
PIPELINE_STATUS=${PIPELINE_STATUS:-${RUN_ROOT}/logs/P5retry_P6_P17_pipeline_v2.status}
PIPELINE_EXIT_CODE=${PIPELINE_EXIT_CODE:-${RUN_ROOT}/logs/P5retry_P6_P17_pipeline_v2.exit_code}

mkdir -p "${RUN_ROOT}/logs"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$PWD:$PWD/ReconViaGen:$PWD/ReconViaGen/wheels/vggt"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export ATTN_BACKEND=${ATTN_BACKEND:-flash_attn}
export SPCONV_ALGO=${SPCONV_ALGO:-native}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

IFS=, read -r -a GPU_ARRAY <<<"${PIPELINE_GPUS}"
if (( ${#GPU_ARRAY[@]} != 8 )); then
  echo "ERROR: pipeline requires exactly 8 GPU indices" >&2
  exit 90
fi

CURRENT_STAGE=preflight
write_status() {
  local state=$1
  local temporary="${PIPELINE_STATUS}.tmp.$$"
  {
    printf 'state=%s\n' "${state}"
    printf 'stage=%s\n' "${CURRENT_STAGE}"
    printf 'time_utc=%s\n' "$(date -u -Is)"
  } >"${temporary}"
  mv "${temporary}" "${PIPELINE_STATUS}"
}
finish() {
  local status=$?
  trap - EXIT
  if (( status == 0 )); then
    CURRENT_STAGE=complete
    write_status PASS
  else
    write_status FAIL
  fi
  printf '%s\n' "${status}" >"${PIPELINE_EXIT_CODE}"
  echo "===== pipeline exit=${status} stage=${CURRENT_STAGE} time=$(date -u -Is) ====="
  exit "${status}"
}
trap finish EXIT

validate_smoke_report() {
  local report=$1
  local expected_world=$2
  "${PYTHON}" - "${report}" "${expected_world}" <<'PY'
import json
import sys

report_path, expected_world = sys.argv[1], int(sys.argv[2])
r = json.load(open(report_path, encoding="utf-8"))
assert r["format"] == "official_ss_with_vggt_perf_v1.native_ss_genrecon.v2"
assert r["completed"] is True and r["passed"] is True and r["step"] == 2
assert r["global_micro_samples"] == 16
assert r["data_identity"]["object_count"] == 8
assert r["initial_stock_audit"]["passed"] is True
assert r["initial_stock_audit"]["conditional_max_abs"] == 0.0
assert r["initial_stock_audit"]["unconditional_max_abs"] == 0.0
runtime = r["model_summary"]["runtime_input_policy"]
assert runtime["ddp_device_ids"] is None
assert runtime["complete_lifting_sample_stays_cpu_until_projection"] is True
distributed = r["model_summary"]["optimization"]["distributed"]
assert distributed["world_size"] == expected_world
assert distributed["global_effective_batch"] == 8
print({"passed": True, "report": report_path, "expected_world": expected_world})
PY
}

gpu_memory_mib() {
  nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk 'NR == 1 {print int($1)}'
}

resources_are_idle() {
  local gpu used
  if pgrep -af '[o]fficial_slat_with_vggt_perf_v1.train_proobjaverse_official' >/dev/null; then
    return 1
  fi
  if pgrep -af '[p]ose_point_depth_mv.train_proobjaverse_official_slat_condition_lora' >/dev/null; then
    return 1
  fi
  if pgrep -af '[o]fficial_ss_with_vggt_perf_v1.train' >/dev/null; then
    return 1
  fi
  for gpu in "${GPU_ARRAY[@]}"; do
    if ! used=$(gpu_memory_mib "${gpu}"); then
      echo "ERROR: unable to query GPU${gpu} memory" >&2
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
    echo "[$(date -u -Is)] waiting for exclusive GPUs (${label}); threshold=${MAX_IDLE_MEMORY_MIB} MiB"
    pgrep -af '[o]fficial_slat_with_vggt_perf_v1.train_proobjaverse_official|[o]fficial_ss_with_vggt_perf_v1.train' || true
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits || true
    sleep "${POLL_SECONDS}"
  done
  echo "[$(date -u -Is)] exclusive GPU gate PASS (${label})"
}

CURRENT_STAGE=preflight
write_status RUNNING
test -s "${P4_REPORT}"
test -s "${P5_CACHE}"
validate_smoke_report "${P4_REPORT}" 1

# Never spend an hour rebuilding caches if a partial formal trajectory already
# occupies the immutable fresh output identity.
if [[ -e "${TRAIN_OUTPUT}" ]]; then
  echo "ERROR: formal fresh output already exists; never auto-resume/reuse it: ${TRAIN_OUTPUT}" >&2
  exit 91
fi

wait_for_idle_resources before_P5_retry

CURRENT_STAGE=P5_exclusive_retry
write_status RUNNING
if [[ -e "${P5_OUTPUT}" ]]; then
  if [[ -s "${P5_OUTPUT}/report.json" ]] && \
      validate_smoke_report "${P5_OUTPUT}/report.json" 2; then
    echo "P5 exclusive evidence already PASS; reusing immutable report"
  else
    echo "ERROR: P5 exclusive output exists but is not a complete PASS: ${P5_OUTPUT}" >&2
    exit 92
  fi
else
  set +e
  CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]},${GPU_ARRAY[1]}" \
    "${TORCHRUN}" --standalone --nproc_per_node=2 \
    -m official_ss_with_vggt_perf_v1.train \
      --allow_short_smoke \
      --cache_manifest "${P5_CACHE}" \
      --output_dir "${P5_OUTPUT}" \
      --max_steps 2 \
      --save_every 1 \
      --log_every 1 \
      --grad_accum 4 \
      --num_workers 0 2>&1 | tee "${P5_LOG}"
  p5_status=${PIPESTATUS[0]}
  set -e
  if (( p5_status != 0 )); then
    echo "ERROR: exclusive P5 retry failed rc=${p5_status}; P6/P7/P8 will not run" >&2
    exit "${p5_status}"
  fi
  test -s "${P5_OUTPUT}/checkpoints/step_000002.pt"
  test -s "${P5_OUTPUT}/report.json"
  validate_smoke_report "${P5_OUTPUT}/report.json" 2
fi

wait_for_idle_resources before_P6

CURRENT_STAGE=P6_build_train2000_and_dev64
write_status RUNNING
if [[ -s "${TRAIN_CACHE}/report.json" && -s "${DEV_CACHE}/report.json" ]] && \
    RUN_ROOT="${RUN_ROOT}" PROJECT_ROOT="${PROJECT_ROOT}" PYTHON="${PYTHON}" \
      bash "${PROJECT_ROOT}/official_ss_with_vggt_perf_v1/validate_full_caches.sh"; then
  echo "P6 caches already complete and identity-valid; skipping materialization"
else
  BUILD_GPUS="${PIPELINE_GPUS}" \
  RUN_ROOT="${RUN_ROOT}" \
  PROJECT_ROOT="${PROJECT_ROOT}" \
    bash "${PROJECT_ROOT}/official_ss_with_vggt_perf_v1/run_source_build_full_caches_after_slat.sh"
fi

CURRENT_STAGE=P7_validate_full_caches
write_status RUNNING
RUN_ROOT="${RUN_ROOT}" PROJECT_ROOT="${PROJECT_ROOT}" PYTHON="${PYTHON}" \
  bash "${PROJECT_ROOT}/official_ss_with_vggt_perf_v1/validate_full_caches.sh"
du -sh "${TRAIN_CACHE}" "${DEV_CACHE}"

wait_for_idle_resources before_P8

CURRENT_STAGE=P8_train_step2000
write_status RUNNING
RUN_ROOT="${RUN_ROOT}" \
TRAIN_GPUS="${PIPELINE_GPUS}" \
CACHE_MANIFEST="${TRAIN_CACHE}/with_vggt_ss_manifest.json" \
OUTPUT_DIR="${TRAIN_OUTPUT}" \
PROJECT_ROOT="${PROJECT_ROOT}" \
  bash "${PROJECT_ROOT}/official_ss_with_vggt_perf_v1/run_source_train2000_after_slat.sh"

CURRENT_STAGE=P8_final_guard
write_status RUNNING
TRAIN_LOG="${RUN_ROOT}/logs/VSS_with_vggt_train2000_step2000_seed42_8gpu_v1.log"
test "$(cat "${TRAIN_LOG}.exit_code")" = 0
for checkpoint_step in 500 1000 1500 2000; do
  checkpoint_pad=$(printf '%06d' "${checkpoint_step}")
  test -s "${TRAIN_OUTPUT}/checkpoints/step_${checkpoint_pad}.pt"
done
test -s "${TRAIN_OUTPUT}/checkpoints/last.pt"
test -s "${TRAIN_OUTPUT}/report.json"
"${PYTHON}" - "${TRAIN_OUTPUT}/report.json" <<'PY'
import json
import sys

r = json.load(open(sys.argv[1], encoding="utf-8"))
assert r["format"] == "official_ss_with_vggt_perf_v1.native_ss_genrecon.v2"
assert r["completed"] is True and r["passed"] is True and r["step"] == 2000
assert r["global_micro_samples"] == 16000
assert r["model_summary"]["stock_floor"] == "VSS0"
assert r["model_summary"]["fresh_initialization_only"] is True
assert r["model_summary"]["runtime_input_policy"]["ddp_device_ids"] is None
assert r["model_summary"]["vggt_camera_consumed"] is False
assert r["data_identity"]["object_count"] == 2000
print({
    "passed": True,
    "step": r["step"],
    "global_micro_samples": r["global_micro_samples"],
    "elapsed_seconds": r["elapsed_seconds"],
    "checkpoint": r["checkpoint"],
})
PY

CURRENT_STAGE=P10_P17_posttrain_SS_SLat_evaluation
write_status RUNNING
SS_ROOT="${RUN_ROOT}" \
EVAL_GPUS="${PIPELINE_GPUS}" \
SLAT_STEPS="${POSTTRAIN_SLAT_STEPS}" \
PROJECT_ROOT="${PROJECT_ROOT}" \
PYTHON="${PYTHON}" \
  bash "${PROJECT_ROOT}/official_ss_with_vggt_perf_v1/run_source_posttrain_p10_p17_eval_pipeline.sh"

echo "P5 RETRY + P6--P17 TRAINING/EVALUATION PIPELINE COMPLETE"
