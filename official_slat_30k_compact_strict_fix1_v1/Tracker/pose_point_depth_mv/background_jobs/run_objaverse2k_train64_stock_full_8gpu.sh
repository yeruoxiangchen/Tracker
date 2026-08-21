#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
GPUS=${OBJ2K_TRAIN64_GPUS:-0,1,2,3,4,5,6,7}
RUN=${OBJAVERSE2K_SLAT_RUN:-/data/zjr/objaverse2k_no_vggt_slat_20260811_v1}
ROOT=${OBJ2K_TRAIN64_ROOT:-/data/zjr/objaverse2k_train64_stock_full_20260813_v1}
SELECTION_SEED=${OBJ2K_TRAIN64_SELECTION_SEED:-20260813}
TRAIN_CACHE=${RUN}/slat_cache_train_seed42_merged_v1/manifest.json
TRAIN_LIFT=${RUN}/split_dev64_v1/train/lifting_manifest.json
CHECKPOINT=${RUN}/slat_objaverse2135_step2000_seed42_8gpu_bs8_v1/checkpoints/step_002000.pt
SS_RUN=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1
SS_REPORT=${SS_RUN}/ss_eval_synthetic_dev32_fixedcfg3_count125_v3/report.json
STOCK_FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json
LOG_DIR=${ROOT}/logs
EVALUATION=${ROOT}/comparison
STATE=${LOG_DIR}/run.state
EXIT_CODE=${LOG_DIR}/run.exit_code
LOCK=${LOG_DIR}/run.lock
TRAIN_LOCK=${RUN}/logs/slat_objaverse2135_8gpu_bs8_training.lock
EXPECTED_OBJECTS=64
EXPECTED_WORKERS=8

IFS=',' read -r -a GPU_ARRAY <<<"${GPUS}"
if [ "${#GPU_ARRAY[@]}" -ne 8 ] || \
   [ "$(printf '%s\n' "${GPU_ARRAY[@]}" | sort -u | wc -l)" -ne 8 ]; then
  echo "OBJ2K_TRAIN64_GPUS requires eight distinct GPUs" >&2
  exit 96
fi
if [[ ! "${SELECTION_SEED}" =~ ^[0-9]+$ ]]; then
  echo "OBJ2K_TRAIN64_SELECTION_SEED must be a non-negative integer" >&2
  exit 96
fi
for REQUIRED in \
  "${PY}" "${TRAIN_CACHE}" "${TRAIN_LIFT}" "${CHECKPOINT}" \
  "${SS_REPORT}" "${STOCK_FREEZE}"; do
  test -s "${REQUIRED}"
done

mkdir -p "${LOG_DIR}"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "Objaverse2K train64 Stock-vs-Full is already running" >&2
  exit 99
fi
exec 8>"${TRAIN_LOCK}"
if ! flock -n 8; then
  echo "Objaverse2K eight-GPU training or evaluation is still running" >&2
  exit 99
fi
finish() {
  RC=$?
  trap - EXIT
  printf '%s\n' "${RC}" >"${EXIT_CODE}"
  printf 'finished_at=%s rc=%s\n' "$(date --iso-8601=seconds)" "${RC}" >"${STATE}"
  exit "${RC}"
}
trap finish EXIT
printf 'started_at=%s state=running gpus=%s objects=%s selection_seed=%s\n' \
  "$(date --iso-8601=seconds)" "${GPUS}" "${EXPECTED_OBJECTS}" \
  "${SELECTION_SEED}" >"${STATE}"
rm -f "${EXIT_CODE}"

PIDS=()
for WORKER in 0 1 2 3 4 5 6 7; do
  WORKER_OUT=${ROOT}/worker_${WORKER}
  if [ -s "${WORKER_OUT}/report.json" ]; then
    continue
  fi
  RESUME=()
  if [ -e "${WORKER_OUT}" ]; then RESUME=(--resume); fi
  CUDA_VISIBLE_DEVICES="${GPU_ARRAY[${WORKER}]}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PY}" -u -m pose_point_depth_mv.evaluate_objaverse2k_no_vggt_slat worker \
    --cache_manifest "${TRAIN_CACHE}" \
    --lifting_cache_manifest "${TRAIN_LIFT}" \
    --checkpoint "${CHECKPOINT}" \
    --model_label objaverse2k \
    --native_ss_report "${SS_REPORT}" \
    --stock_slat_freeze "${STOCK_FREEZE}" \
    --output_dir "${WORKER_OUT}" \
    --weights ema --joint_seeds 42 --noise_seed 20260813 \
    --worker_index "${WORKER}" --num_workers "${EXPECTED_WORKERS}" \
    --expected_objects "${EXPECTED_OBJECTS}" \
    --training_overlap --object_selection_seed "${SELECTION_SEED}" \
    --fixed_axis_evaluation \
    --surface_samples 20000 --amp_dtype bf16 "${RESUME[@]}" \
    >"${LOG_DIR}/worker_${WORKER}.log" 2>&1 &
  PIDS+=("$!")
done
FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then FAILED=1; fi
done
if [ "${FAILED}" -ne 0 ]; then
  echo "At least one train64 worker failed; inspect ${LOG_DIR}" >&2
  exit 2
fi

REPORTS=()
for WORKER in 0 1 2 3 4 5 6 7; do
  REPORT=${ROOT}/worker_${WORKER}/report.json
  test -s "${REPORT}"
  REPORTS+=("${REPORT}")
done
REPORT_CSV=$(IFS=,; echo "${REPORTS[*]}")
"${PY}" -u -m pose_point_depth_mv.summarize_objaverse2k_train_stock_full \
  --worker_reports "${REPORT_CSV}" \
  --output_dir "${EVALUATION}" \
  --expected_workers "${EXPECTED_WORKERS}" \
  --expected_objects "${EXPECTED_OBJECTS}" \
  --bootstrap_samples 10000 --resume

test -s "${EVALUATION}/report.json"
test -s "${EVALUATION}/summary.txt"
cat "${EVALUATION}/summary.txt"
