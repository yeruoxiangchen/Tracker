#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
PYTHON=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
DEV_GPUS=${DEV_GPUS:-5,6}
OMNI_GPU=${OMNI_GPU:-7}

cd "${PROJECT_ROOT}"
export PYTHONPATH="$PWD:$PWD/ReconViaGen:$PWD/ReconViaGen/wheels/vggt"
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba_cache}

IFS=, read -r DEV_GPU_1 DEV_GPU_2 EXTRA <<<"${DEV_GPUS}"
if [ -z "${DEV_GPU_1}" ] || [ -z "${DEV_GPU_2}" ] || [ -n "${EXTRA:-}" ]; then
  echo "ERROR: DEV_GPUS must contain exactly two GPU indices" >&2
  exit 90
fi

SLAT_ROOT=/data/zjr/proobjaverse_official_slat_train2000_20260813_v1
DEV_OUT=${PROJECT_ROOT}/pose_point_depth_mv/outputs2/ProObjaverse_Dev48固定随机2组_ReconViaGen_vs_VSS2k_VSLat15k_20260818_v1
OMNI_OUT=${PROJECT_ROOT}/pose_point_depth_mv/outputs2/OmniPlant012冻结8视图_ReconViaGen_vs_VSS2k_VSLat15k_相机轮廓_20260818_v1
LOG_ROOT=${PROJECT_ROOT}/pose_point_depth_mv/outputs2/logs_with_vggt_qualitative_20260818_v1
mkdir -p "${LOG_ROOT}"

if [ ! -s "${DEV_OUT}/selection.json" ]; then
  test ! -e "${DEV_OUT}"
  ENDPOINT_ROOT=${SLAT_ROOT}/eval_with_vggt_VSS2000_Vstep15000_predicted_support_seed424344_2gpu03_manual_v4/dev48_predicted
  RECON_ROOT=${SLAT_ROOT}/eval_dev64_reconviagen_original_seed424344_quantitative_v1
  ENDPOINT_REPORTS=(
    "${ENDPOINT_ROOT}/shard0_16_40/report.json"
    "${ENDPOINT_ROOT}/shard1_40_64/report.json"
  )
  RECON_REPORTS=()
  for index in 00 01 02 03 04; do
    RECON_REPORTS+=("${RECON_ROOT}/worker_${index}_of_05/report.json")
  done
  ARGS=()
  for path in "${ENDPOINT_REPORTS[@]}"; do ARGS+=(--endpoint_reports "${path}"); done
  for path in "${RECON_REPORTS[@]}"; do ARGS+=(--recon_reports "${path}"); done
  "${PYTHON}" -u -m pose_point_depth_mv.export_proobjaverse_dev_with_vggt_qualitative prepare \
    --dev_split "${SLAT_ROOT}/protocol2128_train2000_v1/dev.json" \
    --cache_report "${SLAT_ROOT}/cache_dev64_protocol2128_views8_v1/report.json" \
    "${ARGS[@]}" \
    --output_dir "${DEV_OUT}" \
    --object_start 16 --count 2 --random_seed 20260818
fi

PIDS=()
GPU="${DEV_GPU_1}" SELECTION="${DEV_OUT}/selection.json" SELECTION_POSITION=1 \
  bash pose_point_depth_mv/background_jobs/run_proobjaverse_dev_with_vggt_qualitative_case.sh \
  >"${LOG_ROOT}/dev_case01_gpu${DEV_GPU_1}.log" 2>&1 &
PIDS+=("$!")
echo "started Dev case01 pid=${PIDS[-1]} GPU=${DEV_GPU_1}"

GPU="${DEV_GPU_2}" SELECTION="${DEV_OUT}/selection.json" SELECTION_POSITION=2 \
  bash pose_point_depth_mv/background_jobs/run_proobjaverse_dev_with_vggt_qualitative_case.sh \
  >"${LOG_ROOT}/dev_case02_gpu${DEV_GPU_2}.log" 2>&1 &
PIDS+=("$!")
echo "started Dev case02 pid=${PIDS[-1]} GPU=${DEV_GPU_2}"

GPU="${OMNI_GPU}" OUTPUT="${OMNI_OUT}" \
  bash pose_point_depth_mv/background_jobs/run_omni_holdout_with_vggt_qualitative.sh \
  >"${LOG_ROOT}/omni_gpu${OMNI_GPU}.log" 2>&1 &
PIDS+=("$!")
echo "started Omni replay pid=${PIDS[-1]} GPU=${OMNI_GPU}"

FAILED=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then FAILED=1; fi
done
if (( FAILED != 0 )); then
  echo "ERROR: at least one qualitative branch failed; outputs are preserved" >&2
  exit 91
fi

"${PYTHON}" -u -m pose_point_depth_mv.finalize_with_vggt_qualitative_outputs dev \
  --output_dir "${DEV_OUT}"

echo "============================================================"
echo "WITH-VGGT QUALITATIVE DEV2 + OMNI COMPLETE"
echo "Dev : ${DEV_OUT}"
echo "Omni: ${OMNI_OUT}"
echo "============================================================"
