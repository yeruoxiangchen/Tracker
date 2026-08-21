#!/usr/bin/env bash
set -euo pipefail

# Resume one predicted-support endpoint worker.  This is intentionally a
# worker-only helper: it never writes an aggregate, so it can safely repair one
# shard while the other shards continue running.

for NAME in \
  PROJECT_ROOT PYTHON GPU SLAT_CACHE_MANIFEST SLAT_LIFTING_MANIFEST \
  SS_CACHE_MANIFEST VSS_REPORT STOCK_FREEZE V_CHECKPOINT EXPECTED_V_STEP \
  OUTPUT_DIR OBJECT_START OBJECT_END LOG
do
  if [ -z "${!NAME:-}" ]; then
    echo "ERROR: ${NAME} is required" >&2
    exit 90
  fi
done

JOINT_SEEDS=${JOINT_SEEDS:-42,43,44}
SURFACE_SAMPLES=${SURFACE_SAMPLES:-20000}
MAX_RESTARTS=${MAX_RESTARTS:-128}

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

mkdir -p "$(dirname "${LOG}")"
ATTEMPT=0
while :; do
  if [ -s "${OUTPUT_DIR}/report.json" ]; then
    echo "worker report already complete: ${OUTPUT_DIR}/report.json" >>"${LOG}"
    exit 0
  fi
  ATTEMPT=$((ATTEMPT + 1))
  if (( ATTEMPT > MAX_RESTARTS )); then
    echo "ERROR: worker exceeded CUDA restart limit=${MAX_RESTARTS}" >>"${LOG}"
    exit 94
  fi
  RESUME_ARGS=()
  if [ -e "${OUTPUT_DIR}" ]; then
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
      --output_dir "${OUTPUT_DIR}" \
      --weights ema \
      --joint_seeds "${JOINT_SEEDS}" \
      --object_start "${OBJECT_START}" \
      --object_end "${OBJECT_END}" \
      --surface_samples "${SURFACE_SAMPLES}" \
      --amp_dtype bf16 \
      --restart_after_recorded_failure \
      "${RESUME_ARGS[@]}" >>"${LOG}" 2>&1
  RC=$?
  set -e
  case "${RC}" in
    0)
      exit 0
      ;;
    75)
      echo "recorded decoder failure; fresh CUDA restart attempt=${ATTEMPT}" \
        >>"${LOG}"
      ;;
    2)
      if [ -s "${OUTPUT_DIR}/report.json" ]; then
        echo "worker completed with registered model-output failures" >>"${LOG}"
        exit 0
      fi
      echo "ERROR: rc=2 without worker report" >>"${LOG}"
      exit 2
      ;;
    *)
      echo "ERROR: worker program failure rc=${RC}" >>"${LOG}"
      exit "${RC}"
      ;;
  esac
done
