#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
PYTHON=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
EVAL_GPUS=${EVAL_GPUS:-0,1,2,3,4,5,6,7}
SLAT_STEP=${SLAT_STEP:-10000}
EVALUATE_TRAIN64=${EVALUATE_TRAIN64:-0}

SLAT_ROOT=${SLAT_ROOT:-/data/zjr/proobjaverse_official_slat_train2000_20260813_v1}
SS_ROOT=${SS_ROOT:-/data/zjr/proobjaverse_official_native_ss_train2000_with_vggt_20260817_v1}
SLAT_RUN=${SLAT_RUN:-${SLAT_ROOT}/V_with_vggt_train2000_step15000_seed42_8gpu_strict_perf_v1_v1}
VSS_RUN=${VSS_RUN:-${SS_ROOT}/VSS_with_vggt_train2000_step2000_seed42_8gpu_v1}
VSS_REPORT=${VSS_REPORT:-${SS_ROOT}/dev48_VSS_step2000_seed424344_6gpu_v1/aggregate_v1/report.json}
STOCK_FREEZE=${STOCK_FREEZE:-/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json}
SS_TRAIN_CACHE=${SS_TRAIN_CACHE:-${SS_ROOT}/cache_train2000_official_ss_with_vggt_sidecar_historical_v2_fix2_v1/with_vggt_ss_manifest.json}
SS_DEV_CACHE=${SS_DEV_CACHE:-${SS_ROOT}/cache_dev64_official_ss_with_vggt_sidecar_historical_v2_fix2_v1/with_vggt_ss_manifest.json}

STEP_PAD=$(printf '%06d' "${SLAT_STEP}")
V_CHECKPOINT=${V_CHECKPOINT:-${SLAT_RUN}/checkpoints/step_${STEP_PAD}.pt}
OUTPUT_ROOT=${OUTPUT_ROOT:-${SLAT_ROOT}/eval_with_vggt_VSS2000_Vstep${SLAT_STEP}_predicted_support_seed424344_8gpu_v1}
REFERENCE_SLAT_STEP=${REFERENCE_SLAT_STEP:-10000}
REFERENCE_OUTPUT_ROOT=${REFERENCE_OUTPUT_ROOT:-${OUTPUT_ROOT/Vstep${SLAT_STEP}/Vstep${REFERENCE_SLAT_STEP}}}
REFERENCE_STEP_PAD=$(printf '%06d' "${REFERENCE_SLAT_STEP}")
REFERENCE_V_CHECKPOINT=${REFERENCE_V_CHECKPOINT:-${SLAT_RUN}/checkpoints/step_${REFERENCE_STEP_PAD}.pt}
case "${EVALUATE_TRAIN64}" in
  0|1) ;;
  *) echo "ERROR: EVALUATE_TRAIN64 must be 0 or 1" >&2; exit 90 ;;
esac

COMMON=(
  PROJECT_ROOT="${PROJECT_ROOT}"
  PYTHON="${PYTHON}"
  EVAL_GPUS="${EVAL_GPUS}"
  VSS_REPORT="${VSS_REPORT}"
  V_CHECKPOINT="${V_CHECKPOINT}"
  STOCK_FREEZE="${STOCK_FREEZE}"
  EXPECTED_V_STEP="${SLAT_STEP}"
)
REUSE_COMMON=()
if (( SLAT_STEP != REFERENCE_SLAT_STEP )); then
  test -s "${REFERENCE_V_CHECKPOINT}"
  REUSE_COMMON=(
    REUSE_V_CHECKPOINT="${REFERENCE_V_CHECKPOINT}"
    REUSE_EXPECTED_V_STEP="${REFERENCE_SLAT_STEP}"
  )
fi

if (( EVALUATE_TRAIN64 == 1 )); then
  echo "===== optional wave: Train64 predicted-support fit diagnosis ====="
  TRAIN_REUSE=()
  if (( SLAT_STEP != REFERENCE_SLAT_STEP )); then
    TRAIN_REUSE=(REUSE_INDEPENDENT_ROOT="${REFERENCE_OUTPUT_ROOT}/train64_predicted")
  fi
  env "${COMMON[@]}" "${REUSE_COMMON[@]}" "${TRAIN_REUSE[@]}" \
    SLAT_CACHE_MANIFEST="${SLAT_ROOT}/cache_train64_protocol2128_views8_with_vggt_sidecar_v1/with_vggt_slat_manifest.json" \
    SLAT_LIFTING_MANIFEST="${SLAT_ROOT}/cache_train64_protocol2128_views8_with_vggt_sidecar_v1/with_vggt_lifting_manifest.json" \
    SS_CACHE_MANIFEST="${SS_TRAIN_CACHE}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}/train64_predicted" \
    EVALUATION_SPLIT=train \
    OBJECT_START=0 OBJECT_END=64 EXPECTED_OBJECTS=64 \
    bash "${PROJECT_ROOT}/official_ss_with_vggt_perf_v1/run_predicted_support_endpoint_split.sh"
else
  echo "===== Train64 skipped by registered Dev48-only scope ====="
fi

echo "===== held-out Dev48 predicted-support generalization ====="
DEV_REUSE=()
if (( SLAT_STEP != REFERENCE_SLAT_STEP )); then
  DEV_REUSE=(REUSE_INDEPENDENT_ROOT="${REFERENCE_OUTPUT_ROOT}/dev48_predicted")
fi
env "${COMMON[@]}" "${REUSE_COMMON[@]}" "${DEV_REUSE[@]}" \
  SLAT_CACHE_MANIFEST="${SLAT_ROOT}/cache_dev64_protocol2128_views8_with_vggt_sidecar_v1/with_vggt_slat_manifest.json" \
  SLAT_LIFTING_MANIFEST="${SLAT_ROOT}/cache_dev64_protocol2128_views8_with_vggt_sidecar_v1/with_vggt_lifting_manifest.json" \
  SS_CACHE_MANIFEST="${SS_DEV_CACHE}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}/dev48_predicted" \
  EVALUATION_SPLIT=dev \
  OBJECT_START=16 OBJECT_END=64 EXPECTED_OBJECTS=48 \
  bash "${PROJECT_ROOT}/official_ss_with_vggt_perf_v1/run_predicted_support_endpoint_split.sh"

echo "WITH-VGGT DEV48 PREDICTED-SUPPORT TEST COMPLETE: ${OUTPUT_ROOT}"
