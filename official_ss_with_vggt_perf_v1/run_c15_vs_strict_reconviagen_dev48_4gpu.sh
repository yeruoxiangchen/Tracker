#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/home/zjr/Tracker}
PY=${PY:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
EVAL_GPUS=${EVAL_GPUS:-4,5,6,7}
ROOT=${ROOT:-/data/zjr/proobjaverse_official_slat_train2000_20260813_v1}
OUT=${OUT:-${ROOT}/eval_with_vggt_VSS2000_Vstep15000_vs_strict_reconviagen_targetlocked_seed424344_4gpu_v1}

CACHE=${ROOT}/cache_dev64_protocol2128_views8_with_vggt_sidecar_v1/with_vggt_slat_manifest.json
LIFTING=${ROOT}/cache_dev64_protocol2128_views8_with_vggt_sidecar_v1/with_vggt_lifting_manifest.json
SS_CACHE=/data/zjr/proobjaverse_official_native_ss_train2000_with_vggt_20260817_v1/cache_dev64_official_ss_with_vggt_sidecar_historical_v2_fix2_v1/with_vggt_ss_manifest.json
CKPT=${ROOT}/V_with_vggt_train2000_step15000_seed42_8gpu_strict_perf_v1_v1/checkpoints/step_015000.pt
STOCK_FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json
SOURCE_ROOT=${ROOT}/eval_with_vggt_VSS2000_Vstep15000_predicted_support_seed424344_2gpu03_manual_v4/dev48_predicted
SOURCE_REPORTS=${SOURCE_ROOT}/shard0_16_40/report.json,${SOURCE_ROOT}/shard1_40_64/report.json
R_ROOT=${ROOT}/eval_dev64_reconviagen_original_seed424344_quantitative_v1
R_REPORTS=${R_ROOT}/worker_00_of_05/report.json,${R_ROOT}/worker_01_of_05/report.json,${R_ROOT}/worker_02_of_05/report.json,${R_ROOT}/worker_03_of_05/report.json,${R_ROOT}/worker_04_of_05/report.json
DEV_SPLIT=${ROOT}/protocol2128_train2000_v1/dev.json
CACHE_REPORT=${ROOT}/cache_dev64_protocol2128_views8_v1/report.json
TARGET_REPORT=${ROOT}/eval_dev64_B_scale_step4000_seed424344_v1/report.json
TARGET_MESH_ROOT=${ROOT}/eval_dev64_B_scale_step4000_seed424344_v1/targets
PAIRED_TARGET_ROOT=${ROOT}/eval_trajectory_step15000_20000_25000_seed424344_5gpu_strict_fix1_v1/step_025000/dev48_predicted
PAIRED_TARGET_CACHE_ROOTS=${PAIRED_TARGET_ROOT}/shard0_16_26/target_mesh_cache,${PAIRED_TARGET_ROOT}/shard1_26_36/target_mesh_cache,${PAIRED_TARGET_ROOT}/shard2_36_46/target_mesh_cache,${PAIRED_TARGET_ROOT}/shard3_46_55/target_mesh_cache,${PAIRED_TARGET_ROOT}/shard4_55_64/target_mesh_cache

cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PROJECT}/ReconViaGen:${PROJECT}/ReconViaGen/wheels/vggt"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export ATTN_BACKEND=${ATTN_BACKEND:-flash_attn}
export SPCONV_ALGO=${SPCONV_ALGO:-native}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba_cache}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

for PATH_VALUE in "${PY}" "${CACHE}" "${LIFTING}" "${SS_CACHE}" "${CKPT}" "${STOCK_FREEZE}"; do
  test -s "${PATH_VALUE}"
done
IFS=, read -r -a GPUS <<<"${EVAL_GPUS}"
if [ "${#GPUS[@]}" -ne 4 ]; then
  echo "ERROR: EVAL_GPUS must contain exactly four GPU ids" >&2
  exit 90
fi
mkdir -p "${OUT}/logs"

PIDS=()
REPORTS=()
STARTS=(16 28 40 52)
ENDS=(28 40 52 64)
run_worker() {
  local SLOT=$1 GPU=$2 START=$3 END=$4 WORKER_OUT=$5 LOG=$6
  local ATTEMPT=0
  while :; do
    ATTEMPT=$((ATTEMPT + 1))
    if (( ATTEMPT > 64 )); then
      echo "ERROR: worker ${SLOT} exceeded restart limit" >&2
      return 95
    fi
    RESUME=()
    [ -e "${WORKER_OUT}" ] && RESUME=(--resume)
    set +e
    CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u -m \
      official_ss_with_vggt_perf_v1.evaluate_c_vs_strict_reconviagen worker \
      --cache_manifest "${CACHE}" \
      --lifting_cache_manifest "${LIFTING}" \
      --ss_cache_manifest "${SS_CACHE}" \
      --source_endpoint_reports "${SOURCE_REPORTS}" \
      --strict_recon_reports "${R_REPORTS}" \
      --trained_slat_checkpoint "${CKPT}" \
      --expected_trained_slat_step 15000 \
      --stock_slat_freeze "${STOCK_FREEZE}" \
      --output_dir "${WORKER_OUT}" \
      --object_start "${START}" --object_end "${END}" \
      --joint_seeds 42,43,44 --surface_samples 20000 --amp_dtype bf16 \
      "${RESUME[@]}" >>"${LOG}" 2>&1
    RC=$?
    set -e
    if (( RC == 0 )); then return 0; fi
    if (( RC == 75 )); then
      echo "worker=${SLOT} restarting after recorded CUDA decoder failure attempt=${ATTEMPT}" >>"${LOG}"
      continue
    fi
    echo "ERROR: worker=${SLOT} failed rc=${RC}; log=${LOG}" >&2
    return "${RC}"
  done
}

for SLOT in 0 1 2 3; do
  START=${STARTS[$SLOT]}
  END=${ENDS[$SLOT]}
  WORKER_OUT=${OUT}/shard${SLOT}_${START}_${END}
  LOG=${OUT}/logs/shard${SLOT}_${START}_${END}_gpu${GPUS[$SLOT]}.log
  REPORTS+=("${WORKER_OUT}/report.json")
  run_worker "${SLOT}" "${GPUS[$SLOT]}" "${START}" "${END}" "${WORKER_OUT}" "${LOG}" &
  PIDS+=("$!")
  echo "worker=${SLOT} gpu=${GPUS[$SLOT]} range=[${START},${END}) pid=${PIDS[-1]} log=${LOG}"
done

FAILED=0
for SLOT in 0 1 2 3; do
  if ! wait "${PIDS[$SLOT]}"; then FAILED=1; fi
done
if (( FAILED != 0 )); then
  echo "C worker failure; outputs are resumable" >&2
  exit 96
fi

REPORT_CSV=$(IFS=,; echo "${REPORTS[*]}")
FINAL=${OUT}/aggregate_v1
if [ -s "${FINAL}/report.json" ]; then
  echo "reuse aggregate: ${FINAL}/report.json"
else
  test ! -e "${FINAL}"
  "${PY}" -u -m official_ss_with_vggt_perf_v1.evaluate_c_vs_strict_reconviagen \
    aggregate \
    --dev_split "${DEV_SPLIT}" \
    --cache_report "${CACHE_REPORT}" \
    --target_report "${TARGET_REPORT}" \
    --target_mesh_root "${TARGET_MESH_ROOT}" \
    --paired_target_cache_roots "${PAIRED_TARGET_CACHE_ROOTS}" \
    --strict_recon_reports "${R_REPORTS}" \
    --candidate_reports "${REPORT_CSV}" \
    --object_start 16 --object_end 64 \
    --bootstrap_samples 5000 \
    --output_dir "${FINAL}"
fi

echo "STRICT C-R COMPLETE: ${FINAL}/report.json"
