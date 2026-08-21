#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
PYTHON=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
EVAL_GPUS=${EVAL_GPUS:-0,1,2,3,4,5}

: "${CACHE_MANIFEST:?set CACHE_MANIFEST}"
: "${CHECKPOINT:?set CHECKPOINT}"
: "${CALIBRATION:?set CALIBRATION}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="$PWD:$PWD/ReconViaGen:$PWD/ReconViaGen/wheels/vggt"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export ATTN_BACKEND=${ATTN_BACKEND:-flash_attn}
export SPCONV_ALGO=${SPCONV_ALGO:-native}

IFS=, read -r -a GPU_ARRAY <<<"${EVAL_GPUS}"
if (( ${#GPU_ARRAY[@]} != 6 )); then
  echo "ERROR: Dev48 script requires six GPUs" >&2
  exit 90
fi
STARTS=(16 24 32 40 48 56)
ENDS=(24 32 40 48 56 64)
mkdir -p "${OUTPUT_ROOT}/logs"
PIDS=()
REPORTS=()
for INDEX in 0 1 2 3 4 5; do
  GPU=${GPU_ARRAY[$INDEX]}
  START=${STARTS[$INDEX]}
  END=${ENDS[$INDEX]}
  OUT="${OUTPUT_ROOT}/shard${INDEX}_${START}_${END}"
  LOG="${OUTPUT_ROOT}/logs/shard${INDEX}_gpu${GPU}.log"
  test ! -e "${OUT}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
    "${PYTHON}" -u -m official_ss_with_vggt_perf_v1.evaluate \
      --mode evaluate \
      --cache_manifest "${CACHE_MANIFEST}" \
      --checkpoint "${CHECKPOINT}" \
      --calibration "${CALIBRATION}" \
      --output_dir "${OUT}" \
      --object_start "${START}" \
      --object_end "${END}" \
      --joint_seeds 42,43,44 \
      --weights ema \
      --steps 25 \
      --cfg_interval 0.5,1.0 \
      --guidance_rescale 0.0 \
      --rescale_t 3.0 \
      --amp_dtype bf16 \
      --bootstrap_samples 5000 \
      --min_iou_gain_mean -1 \
      --min_iou_win_rate 0 \
      --min_recall_gain_mean -1 \
      --min_latent_mse_gain_mean -1 \
      --min_count_ratio 0.01 \
      --max_count_ratio 100 \
      --min_pose_control_iou_advantage -1 >"${LOG}" 2>&1 &
  PIDS+=("$!")
  REPORTS+=("${OUT}/report.json")
  echo "worker=${INDEX} gpu=${GPU} range=[${START},${END}) pid=${PIDS[-1]} log=${LOG}"
done

FAILED=0
for INDEX in 0 1 2 3 4 5; do
  if ! wait "${PIDS[$INDEX]}"; then
    echo "ERROR: evaluation shard ${INDEX} failed" >&2
    FAILED=1
  fi
done
if (( FAILED != 0 )); then
  echo "Shard outputs are preserved; inspect logs before any retry." >&2
  exit 91
fi

REPORT_CSV=$(IFS=,; echo "${REPORTS[*]}")
set +e
"${PYTHON}" -u -m official_ss_with_vggt_perf_v1.aggregate \
  --shard_reports "${REPORT_CSV}" \
  --output_dir "${OUTPUT_ROOT}/aggregate_v1" \
  --expected_objects 48 \
  --bootstrap_samples 5000 \
  --min_iou_gain_mean 0 \
  --min_iou_win_rate 0.5 \
  --min_recall_gain_mean 0 \
  --min_latent_mse_gain_mean 0 \
  --min_count_ratio 0.85 \
  --max_count_ratio 1.20 \
  --min_pose_control_iou_advantage 0
SCIENCE_RC=$?
set -e
printf '%s\n' "${SCIENCE_RC}" >"${OUTPUT_ROOT}/aggregate_v1.science_exit_code"
if (( SCIENCE_RC != 0 && SCIENCE_RC != 3 )); then
  echo "ERROR: aggregate program failed rc=${SCIENCE_RC}" >&2
  exit 92
fi

echo "WITH-VGGT OFFICIAL SS DEV48 PROGRAM COMPLETE: ${OUTPUT_ROOT}/aggregate_v1/report.json"
echo "SCIENCE_RC=${SCIENCE_RC} (0=all registered gates pass; 3=science gate fails)"
