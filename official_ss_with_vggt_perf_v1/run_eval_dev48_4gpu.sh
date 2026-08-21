#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
PYTHON=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
EVAL_GPUS=${EVAL_GPUS:-0,1,2,3}

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
if (( ${#GPU_ARRAY[@]} != 4 )); then
  echo "ERROR: Dev48 four-GPU route requires exactly four GPUs" >&2
  exit 90
fi
STARTS=(16 28 40 52)
ENDS=(28 40 52 64)
mkdir -p "${OUTPUT_ROOT}/logs"
PIDS=()
REPORTS=()
for index in 0 1 2 3; do
  gpu=${GPU_ARRAY[$index]}
  start=${STARTS[$index]}
  end=${ENDS[$index]}
  out="${OUTPUT_ROOT}/shard${index}_${start}_${end}"
  log="${OUTPUT_ROOT}/logs/shard${index}_gpu${gpu}.log"
  test ! -e "${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON}" -u -m official_ss_with_vggt_perf_v1.evaluate \
      --mode evaluate \
      --cache_manifest "${CACHE_MANIFEST}" \
      --checkpoint "${CHECKPOINT}" \
      --calibration "${CALIBRATION}" \
      --output_dir "${out}" \
      --object_start "${start}" \
      --object_end "${end}" \
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
      --min_pose_control_iou_advantage -1 >"${log}" 2>&1 &
  PIDS+=("$!")
  REPORTS+=("${out}/report.json")
  echo "worker=${index} gpu=${gpu} range=[${start},${end}) pid=${PIDS[-1]} log=${log}"
done

failed=0
for index in 0 1 2 3; do
  if ! wait "${PIDS[$index]}"; then
    echo "ERROR: VSS Dev48 shard ${index} failed" >&2
    failed=1
  fi
done
(( failed == 0 )) || exit 91

report_csv=$(IFS=,; echo "${REPORTS[*]}")
set +e
"${PYTHON}" -u -m official_ss_with_vggt_perf_v1.aggregate \
  --shard_reports "${report_csv}" \
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
science_rc=$?
set -e
printf '%s\n' "${science_rc}" >"${OUTPUT_ROOT}/aggregate_v1.science_exit_code"
if (( science_rc != 0 && science_rc != 3 )); then
  echo "ERROR: VSS aggregate program failed rc=${science_rc}" >&2
  exit 92
fi
echo "WITH-VGGT OFFICIAL SS DEV48 PROGRAM COMPLETE: ${OUTPUT_ROOT}/aggregate_v1/report.json"
echo "SCIENCE_RC=${science_rc} (0=registered gates pass; 3=science gate fails)"
