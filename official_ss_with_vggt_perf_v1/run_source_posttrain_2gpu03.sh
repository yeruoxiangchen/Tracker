#!/usr/bin/env bash
set -euo pipefail

# Resume the already-trained four-GPU/global-batch-8 route at P10--P17 using
# physical GPUs 0 and 3.  The completed P10 calibration is identity-validated
# and reused; new P11/P15 outputs have a distinct two-GPU runtime identity.

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
PYTHON=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
RUN_ROOT=${RUN_ROOT:-/data/zjr/proobjaverse_official_native_ss_train2000_with_vggt_20260817_v1}
SLAT_ROOT=${SLAT_ROOT:-/data/zjr/proobjaverse_official_slat_train2000_20260813_v1}
EVAL_GPUS=${EVAL_GPUS:-0,3}

exec env \
  SS_ROOT="${RUN_ROOT}" \
  SS_TRAIN="${RUN_ROOT}/VSS_with_vggt_train2000_step2000_seed42_4gpu_gb8_v1" \
  SS_CHECKPOINT="${RUN_ROOT}/VSS_with_vggt_train2000_step2000_seed42_4gpu_gb8_v1/checkpoints/step_002000.pt" \
  SS_TRAIN_CACHE="${RUN_ROOT}/cache_train2000_official_ss_with_vggt_sidecar_historical_v2_fix2_v1/with_vggt_ss_manifest.json" \
  SS_DEV_CACHE="${RUN_ROOT}/cache_dev64_official_ss_with_vggt_sidecar_historical_v2_fix2_v1/with_vggt_ss_manifest.json" \
  CALIBRATION_ROOT="${RUN_ROOT}/dev64_VSS_step2000_4gpu_gb8_calibrate0_16_seed424344_v1" \
  CALIBRATION="${RUN_ROOT}/dev64_VSS_step2000_4gpu_gb8_calibrate0_16_seed424344_v1/calibration.json" \
  SS_DEV48_ROOT="${RUN_ROOT}/dev48_VSS_step2000_seed424344_2gpu03_hold_v2" \
  VSS_REPORT="${RUN_ROOT}/dev48_VSS_step2000_seed424344_2gpu03_hold_v2/aggregate_v1/report.json" \
  POST_STATUS="${RUN_ROOT}/logs/P10_P17_posttrain_eval_2gpu03_hold_v2.status" \
  POST_EXIT_CODE="${RUN_ROOT}/logs/P10_P17_posttrain_eval_2gpu03_hold_v2.exit_code" \
  POST_SUMMARY="${RUN_ROOT}/P10_P17_posttrain_eval_2gpu03_hold_v2_summary.json" \
  EVAL_GPU_RESERVATION_ROOT="${RUN_ROOT}/logs/P10_P17_gpu_reservations_2gpu03_hold_v2" \
  EVAL_GPUS="${EVAL_GPUS}" \
  EVAL_TAG=2gpu03_hold_v2 \
  EVALUATE_TRAIN64=0 \
  SLAT_STEPS=10000,15000 \
  SLAT_ROOT="${SLAT_ROOT}" \
  PROJECT_ROOT="${PROJECT_ROOT}" \
  PYTHON="${PYTHON}" \
  bash "${PROJECT_ROOT}/official_ss_with_vggt_perf_v1/run_source_posttrain_p10_p17_eval_pipeline.sh"
