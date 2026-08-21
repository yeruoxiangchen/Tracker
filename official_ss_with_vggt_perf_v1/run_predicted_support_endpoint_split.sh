#!/usr/bin/env bash
set -euo pipefail

# Required: SLAT_CACHE_MANIFEST, SLAT_LIFTING_MANIFEST, SS_CACHE_MANIFEST,
# VSS_REPORT, V_CHECKPOINT, STOCK_FREEZE, OUTPUT_ROOT, OBJECT_START,
# OBJECT_END, EXPECTED_OBJECTS, EXPECTED_V_STEP, EVALUATION_SPLIT.

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
PYTHON=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
EVAL_GPUS=${EVAL_GPUS:-0,1,2,3,4,5,6,7}
JOINT_SEEDS=${JOINT_SEEDS:-42,43,44}
SURFACE_SAMPLES=${SURFACE_SAMPLES:-20000}
REUSE_INDEPENDENT_ROOT=${REUSE_INDEPENDENT_ROOT:-}
REUSE_V_CHECKPOINT=${REUSE_V_CHECKPOINT:-}
REUSE_EXPECTED_V_STEP=${REUSE_EXPECTED_V_STEP:-}

for NAME in \
  SLAT_CACHE_MANIFEST SLAT_LIFTING_MANIFEST SS_CACHE_MANIFEST VSS_REPORT \
  V_CHECKPOINT STOCK_FREEZE OUTPUT_ROOT OBJECT_START OBJECT_END \
  EXPECTED_OBJECTS EXPECTED_V_STEP EVALUATION_SPLIT
do
  if [ -z "${!NAME:-}" ]; then
    echo "ERROR: ${NAME} is required" >&2
    exit 90
  fi
done
case "${EVALUATION_SPLIT}" in
  train|dev) ;;
  *) echo "ERROR: EVALUATION_SPLIT must be train or dev" >&2; exit 90 ;;
esac

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

for PATH_VALUE in \
  "${SLAT_CACHE_MANIFEST}" "${SLAT_LIFTING_MANIFEST}" \
  "${SS_CACHE_MANIFEST}" "${VSS_REPORT}" "${V_CHECKPOINT}" "${STOCK_FREEZE}"
do
  test -s "${PATH_VALUE}"
done
V_CHECKPOINT_SHA256=$(sha256sum "${V_CHECKPOINT}" | awk '{print $1}')
SS_CACHE_SHA256=$(sha256sum "${SS_CACHE_MANIFEST}" | awk '{print $1}')
VSS_REPORT_SHA256=$(sha256sum "${VSS_REPORT}" | awk '{print $1}')

reuse_value_count=0
[[ -n "${REUSE_INDEPENDENT_ROOT}" ]] && reuse_value_count=$((reuse_value_count + 1))
[[ -n "${REUSE_V_CHECKPOINT}" ]] && reuse_value_count=$((reuse_value_count + 1))
[[ -n "${REUSE_EXPECTED_V_STEP}" ]] && reuse_value_count=$((reuse_value_count + 1))
if (( reuse_value_count != 0 && reuse_value_count != 3 )); then
  echo "ERROR: checkpoint-independent reuse requires root/checkpoint/step together" >&2
  exit 93
fi
if (( reuse_value_count == 3 )); then
  test -d "${REUSE_INDEPENDENT_ROOT}"
  test -s "${REUSE_V_CHECKPOINT}"
fi

if (( OBJECT_START < 0 || OBJECT_END <= OBJECT_START )); then
  echo "ERROR: invalid object range [${OBJECT_START},${OBJECT_END})" >&2
  exit 91
fi
if (( OBJECT_END - OBJECT_START != EXPECTED_OBJECTS )); then
  echo "ERROR: range size differs from EXPECTED_OBJECTS" >&2
  exit 92
fi

IFS=, read -r -a GPU_ARRAY <<<"${EVAL_GPUS}"
WORKERS=${#GPU_ARRAY[@]}
if (( WORKERS <= 0 || WORKERS > EXPECTED_OBJECTS )); then
  echo "ERROR: invalid EVAL_GPUS worker count=${WORKERS}" >&2
  exit 93
fi

mkdir -p "${OUTPUT_ROOT}/logs"
PIDS=()
REPORTS=()

validate_worker_report() {
  local REPORT=$1
  local START=$2
  local END=$3
  "${PYTHON}" -m official_ss_with_vggt_perf_v1.artifact_validation \
    endpoint-worker \
    --report "${REPORT}" \
    --split "${EVALUATION_SPLIT}" \
    --object-start "${START}" \
    --object-end "${END}" \
    --joint-seeds "${JOINT_SEEDS}" \
    --slat-step "${EXPECTED_V_STEP}" \
    --slat-checkpoint "${V_CHECKPOINT}" \
    --ss-cache "${SS_CACHE_MANIFEST}" \
    --vss-report "${VSS_REPORT}" \
    --slat-checkpoint-sha256 "${V_CHECKPOINT_SHA256}" \
    --ss-cache-sha256 "${SS_CACHE_SHA256}" \
    --vss-report-sha256 "${VSS_REPORT_SHA256}"
}

validate_aggregate_report() {
  local REPORT=$1
  "${PYTHON}" -m official_ss_with_vggt_perf_v1.artifact_validation \
    endpoint-aggregate \
    --report "${REPORT}" \
    --split "${EVALUATION_SPLIT}" \
    --object-start "${OBJECT_START}" \
    --object-end "${OBJECT_END}" \
    --expected-objects "${EXPECTED_OBJECTS}" \
    --joint-seeds "${JOINT_SEEDS}" \
    --slat-step "${EXPECTED_V_STEP}" \
    --slat-checkpoint "${V_CHECKPOINT}" \
    --ss-cache "${SS_CACHE_MANIFEST}" \
    --vss-report "${VSS_REPORT}" \
    --slat-checkpoint-sha256 "${V_CHECKPOINT_SHA256}" \
    --ss-cache-sha256 "${SS_CACHE_SHA256}" \
    --vss-report-sha256 "${VSS_REPORT_SHA256}"
}

run_shard() {
  local INDEX=$1
  local GPU=$2
  local START=$3
  local END=$4
  local OUT=$5
  local LOG=$6
  local ATTEMPT=0
  while :; do
    if [ -s "${OUT}/report.json" ]; then
      # Reports produced before the completed-science-negative annotation fix
      # have a valid base hash but no with-VGGT endpoint contract.  Let the
      # wrapper reopen them once with --resume: the base evaluator returns
      # without GPU work, then the wrapper adds the content-addressed endpoint
      # annotation.  Fully annotated reports retain the fast validation path.
      if "${PYTHON}" - "${OUT}/report.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(
    0
    if payload.get("with_vggt_endpoint_contract") is not None
    and payload.get("evaluation_split") in {"train", "dev"}
    else 1
)
PY
      then
        validate_worker_report "${OUT}/report.json" "${START}" "${END}"
        echo "worker=${INDEX} identity-valid report complete: ${OUT}/report.json"
        return 0
      fi
      echo "worker=${INDEX} repairing missing endpoint annotation: ${OUT}/report.json" \
        >>"${LOG}"
    fi
    ATTEMPT=$((ATTEMPT + 1))
    if (( ATTEMPT > 128 )); then
      echo "ERROR: worker=${INDEX} exceeded recorded-failure restart limit" >&2
      return 94
    fi
    RESUME_ARGS=()
    if [ -e "${OUT}" ]; then
      RESUME_ARGS+=(--resume)
    fi
    set +e
    CUDA_VISIBLE_DEVICES="${GPU}" \
      "${PYTHON}" -u -m official_ss_with_vggt_perf_v1.evaluate_ss_slat \
        worker \
        --cache_manifest "${SLAT_CACHE_MANIFEST}" \
        --lifting_cache_manifest "${SLAT_LIFTING_MANIFEST}" \
        --ss_cache_manifest "${SS_CACHE_MANIFEST}" \
        --native_ss_report "${VSS_REPORT}" \
        --stock_slat_freeze "${STOCK_FREEZE}" \
        --trained_slat_checkpoint "${V_CHECKPOINT}" \
        --trained_slat_weights ema \
        --expected_trained_slat_step "${EXPECTED_V_STEP}" \
        --output_dir "${OUT}" \
        --weights ema \
        --joint_seeds "${JOINT_SEEDS}" \
        --object_start "${START}" \
        --object_end "${END}" \
        --surface_samples "${SURFACE_SAMPLES}" \
        --amp_dtype bf16 \
        --restart_after_recorded_failure \
        "${RESUME_ARGS[@]}" >>"${LOG}" 2>&1
    RC=$?
    set -e
    if (( RC == 0 )); then
      validate_worker_report "${OUT}/report.json" "${START}" "${END}"
      return 0
    fi
    if (( RC == 75 )); then
      echo "worker=${INDEX} recorded model topology failure; CUDA process restart attempt=${ATTEMPT}" >>"${LOG}"
      continue
    fi
    if (( RC == 2 )) && [ -s "${OUT}/report.json" ]; then
      validate_worker_report "${OUT}/report.json" "${START}" "${END}"
      echo "worker=${INDEX} completed with recorded invalid model output; preserve for aggregate" >>"${LOG}"
      return 0
    fi
    echo "ERROR: worker=${INDEX} program failure rc=${RC}; log=${LOG}" >&2
    return "${RC}"
  done
}

for ((INDEX=0; INDEX<WORKERS; INDEX++)); do
  GPU=${GPU_ARRAY[$INDEX]}
  START=$((OBJECT_START + EXPECTED_OBJECTS * INDEX / WORKERS))
  END=$((OBJECT_START + EXPECTED_OBJECTS * (INDEX + 1) / WORKERS))
  OUT="${OUTPUT_ROOT}/shard${INDEX}_${START}_${END}"
  LOG="${OUTPUT_ROOT}/logs/shard${INDEX}_${START}_${END}_gpu${GPU}.log"
  if (( reuse_value_count == 3 )) && [[ ! -s "${OUT}/report.json" ]]; then
    SOURCE_OUT="${REUSE_INDEPENDENT_ROOT}/shard${INDEX}_${START}_${END}"
    "${PYTHON}" -m official_ss_with_vggt_perf_v1.artifact_validation \
      endpoint-worker \
      --report "${SOURCE_OUT}/report.json" \
      --split "${EVALUATION_SPLIT}" \
      --object-start "${START}" \
      --object-end "${END}" \
      --joint-seeds "${JOINT_SEEDS}" \
      --slat-step "${REUSE_EXPECTED_V_STEP}" \
      --slat-checkpoint "${REUSE_V_CHECKPOINT}" \
      --ss-cache "${SS_CACHE_MANIFEST}" \
      --vss-report "${VSS_REPORT}"
    "${PYTHON}" -m official_ss_with_vggt_perf_v1.reuse_endpoint_artifacts \
      --source_worker "${SOURCE_OUT}" \
      --target_worker "${OUT}" \
      --source_step "${REUSE_EXPECTED_V_STEP}" \
      --target_step "${EXPECTED_V_STEP}" \
      --source_checkpoint "${REUSE_V_CHECKPOINT}" \
      --target_checkpoint "${V_CHECKPOINT}" >>"${LOG}" 2>&1
  fi
  REPORTS+=("${OUT}/report.json")
  run_shard "${INDEX}" "${GPU}" "${START}" "${END}" "${OUT}" "${LOG}" &
  PIDS+=("$!")
  echo "worker=${INDEX} gpu=${GPU} range=[${START},${END}) pid=${PIDS[-1]} log=${LOG}"
done

FAILED=0
for ((INDEX=0; INDEX<WORKERS; INDEX++)); do
  if ! wait "${PIDS[$INDEX]}"; then
    echo "ERROR: endpoint worker ${INDEX} failed" >&2
    FAILED=1
  fi
done
if (( FAILED != 0 )); then
  echo "Worker outputs are preserved; rerun with the same variables to resume." >&2
  exit 95
fi

REPORT_CSV=$(IFS=,; echo "${REPORTS[*]}")
FINAL="${OUTPUT_ROOT}/aggregate_v1"
if [ -s "${FINAL}/report.json" ]; then
  validate_aggregate_report "${FINAL}/report.json"
  echo "reuse identity-valid aggregate: ${FINAL}/report.json"
else
  test ! -e "${FINAL}"
  set +e
  "${PYTHON}" -u -m official_ss_with_vggt_perf_v1.evaluate_ss_slat \
    aggregate \
    --cache_manifest "${SLAT_CACHE_MANIFEST}" \
    --lifting_cache_manifest "${SLAT_LIFTING_MANIFEST}" \
    --ss_cache_manifest "${SS_CACHE_MANIFEST}" \
    --shard_reports "${REPORT_CSV}" \
    --output_dir "${FINAL}" \
    --joint_seeds "${JOINT_SEEDS}" \
    --expected_objects "${EXPECTED_OBJECTS}" \
    --object_start "${OBJECT_START}" \
    --object_end "${OBJECT_END}" \
    --bootstrap_samples 5000
  SCIENCE_RC=$?
  set -e
  printf '%s\n' "${SCIENCE_RC}" >"${OUTPUT_ROOT}/aggregate_v1.science_exit_code"
  if (( SCIENCE_RC != 0 && SCIENCE_RC != 3 )); then
    echo "ERROR: endpoint aggregate program failed rc=${SCIENCE_RC}" >&2
    exit 96
  fi
  validate_aggregate_report "${FINAL}/report.json"
fi

echo "WITH-VGGT PREDICTED-SUPPORT ENDPOINT COMPLETE: ${FINAL}/report.json"
