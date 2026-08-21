#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
GPUS=${OBJAVERSE2K_SLAT_EVAL_GPUS:-0,5,6,7}
STEP=${OBJAVERSE2K_SLAT_EVAL_STEP:-400}
RUN=${OBJAVERSE2K_SLAT_RUN:-/data/zjr/objaverse2k_no_vggt_slat_20260811_v1}
STEP_PAD=$(printf '%06d' "${STEP}")
DEV_CACHE=${RUN}/slat_cache_dev64_seed424344_merged_v1/manifest.json
DEV_LIFT=${RUN}/split_dev64_v1/dev/lifting_manifest.json
SS_RUN=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1
SS_REPORT=${SS_RUN}/ss_eval_synthetic_dev32_fixedcfg3_count125_v3/report.json
STOCK_FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json
M8=${SS_RUN}/slat_mixed_step2000_seed42_2gpu_v1/checkpoints/step_002000.pt
CANDIDATE=${RUN}/slat_objaverse2135_step2000_seed42_4gpu_v1/checkpoints/step_${STEP_PAD}.pt
OUT=${RUN}/eval_dev64_step${STEP_PAD}_stock_m8_objaverse2k_v1
STATE=${RUN}/logs/eval_dev64_step${STEP_PAD}_4gpu.state
EXIT_CODE=${RUN}/logs/eval_dev64_step${STEP_PAD}_4gpu.exit_code
LOCK=${RUN}/logs/eval_dev64_step${STEP_PAD}_4gpu.lock

IFS=',' read -r -a GPU_ARRAY <<<"${GPUS}"
if [ "${#GPU_ARRAY[@]}" -ne 4 ] || [ "$(printf '%s\n' "${GPU_ARRAY[@]}" | sort -u | wc -l)" -ne 4 ]; then
  echo "OBJAVERSE2K_SLAT_EVAL_GPUS requires four distinct GPUs" >&2
  exit 96
fi
if [[ ! "${STEP}" =~ ^[0-9]+$ ]] || [ "${STEP}" -le 0 ]; then
  echo "OBJAVERSE2K_SLAT_EVAL_STEP must be positive" >&2
  exit 96
fi
for REQUIRED in "${DEV_CACHE}" "${DEV_LIFT}" "${SS_REPORT}" "${STOCK_FREEZE}" "${M8}" "${CANDIDATE}"; do
  test -s "${REQUIRED}"
done

mkdir -p "${RUN}/logs" "${OUT}"
exec 9>"${LOCK}"
if ! flock -n 9; then echo "Objaverse2K dev64 evaluation is already running" >&2; exit 99; fi
finish() {
  RC=$?
  trap - EXIT
  printf '%s\n' "${RC}" >"${EXIT_CODE}"
  printf 'finished_at=%s rc=%s\n' "$(date --iso-8601=seconds)" "${RC}" >"${STATE}"
  exit "${RC}"
}
trap finish EXIT
printf 'started_at=%s state=running gpus=%s step=%s\n' \
  "$(date --iso-8601=seconds)" "${GPUS}" "${STEP}" >"${STATE}"
rm -f "${EXIT_CODE}"

LABELS=(m8 m8 objaverse2k objaverse2k)
WORKERS=(0 1 0 1)
CHECKPOINTS=("${M8}" "${M8}" "${CANDIDATE}" "${CANDIDATE}")
pids=()
for SLOT in 0 1 2 3; do
  LABEL=${LABELS[${SLOT}]}
  WORKER=${WORKERS[${SLOT}]}
  WORKER_OUT=${OUT}/${LABEL}_worker_${WORKER}
  if [ -s "${WORKER_OUT}/report.json" ]; then
    continue
  fi
  RESUME=()
  if [ -e "${WORKER_OUT}" ]; then RESUME=(--resume); fi
  CUDA_VISIBLE_DEVICES="${GPU_ARRAY[${SLOT}]}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PY}" -u -m pose_point_depth_mv.evaluate_objaverse2k_no_vggt_slat worker \
    --cache_manifest "${DEV_CACHE}" \
    --lifting_cache_manifest "${DEV_LIFT}" \
    --checkpoint "${CHECKPOINTS[${SLOT}]}" \
    --model_label "${LABEL}" \
    --native_ss_report "${SS_REPORT}" \
    --stock_slat_freeze "${STOCK_FREEZE}" \
    --output_dir "${WORKER_OUT}" \
    --weights ema --joint_seeds 42,43,44 --noise_seed 20260811 \
    --worker_index "${WORKER}" --num_workers 2 --expected_objects 64 \
    --surface_samples 20000 --amp_dtype bf16 "${RESUME[@]}" \
    >"${RUN}/logs/eval_dev64_step${STEP_PAD}_${LABEL}_worker_${WORKER}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for PID in "${pids[@]}"; do
  if ! wait "${PID}"; then failed=1; fi
done
if [ "${failed}" -ne 0 ]; then
  echo "one or more dev64 evaluation workers failed" >&2
  exit 2
fi

if [ ! -s "${OUT}/comparison/report.json" ]; then
  "${PY}" -u -m pose_point_depth_mv.evaluate_objaverse2k_no_vggt_slat aggregate \
    --m8_reports "${OUT}/m8_worker_0/report.json,${OUT}/m8_worker_1/report.json" \
    --objaverse2k_reports "${OUT}/objaverse2k_worker_0/report.json,${OUT}/objaverse2k_worker_1/report.json" \
    --output_dir "${OUT}/comparison" \
    --expected_workers 2 --expected_objects 64 --bootstrap_samples 10000
fi

test -s "${OUT}/comparison/report.json"
test -s "${OUT}/comparison/summary.txt"
