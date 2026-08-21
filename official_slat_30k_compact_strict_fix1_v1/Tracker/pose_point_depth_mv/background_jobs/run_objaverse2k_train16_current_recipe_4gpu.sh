#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
TORCHRUN=${TORCHRUN:-/home/zjr/anaconda3/envs/reconviagen/bin/torchrun}
GPUS=${OBJ2K_TRAIN16_GPUS:-0,1,4,5}
ROOT=${OBJ2K_TRAIN16_ROOT:-/data/zjr/objaverse2k_train16_current_recipe_20260814_v1}
SELECTION_SEED=${OBJ2K_TRAIN16_SELECTION_SEED:-20260814}
SOURCE=/data/zjr/objaverse2k_no_vggt_slat_20260811_v1
TRAIN_CACHE=${SOURCE}/slat_cache_train_seed42_merged_v1/manifest.json
TRAIN_LIFT=${SOURCE}/split_dev64_v1/train/lifting_manifest.json
TARGET_AUDIT=${SOURCE}/slat_target_decoder_audit_dev32_v1/report.json
SS_REPORT=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1/ss_eval_synthetic_dev32_fixedcfg3_count125_v3/report.json
STOCK_FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json
SELECTION=${ROOT}/selection
TRAIN_OUT=${ROOT}/B_condition_lora_train16_step200_seed42_4gpu_v1
EVAL_ROOT=${ROOT}/eval_train16_B_condition_lora_step200_seed42_4gpu_v1
SUMMARY=${ROOT}/train16_fit_decision_v1
LOG_DIR=${ROOT}/logs
STATE=${LOG_DIR}/run.state
EXIT_CODE=${LOG_DIR}/run.exit_code
LOCK=${LOG_DIR}/run.lock
EXPECTED_OBJECTS=16
EXPECTED_WORKERS=4

IFS=',' read -r -a GPU_ARRAY <<<"${GPUS}"
if [ "${#GPU_ARRAY[@]}" -ne 4 ] || \
   [ "$(printf '%s\n' "${GPU_ARRAY[@]}" | sort -u | wc -l)" -ne 4 ]; then
  echo "OBJ2K_TRAIN16_GPUS requires four distinct GPU ids" >&2
  exit 96
fi
for REQUIRED in \
  "${PY}" "${TORCHRUN}" "${TRAIN_CACHE}" "${TRAIN_LIFT}" \
  "${TARGET_AUDIT}" "${SS_REPORT}" "${STOCK_FREEZE}"; do
  test -s "${REQUIRED}"
done

mkdir -p "${LOG_DIR}"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "Objaverse2K Train16 current-recipe experiment is already running" >&2
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
printf 'started_at=%s state=running gpus=%s world_size=4 grad_accum=2 global_batch=8\n' \
  "$(date --iso-8601=seconds)" "${GPUS}" >"${STATE}"
rm -f "${EXIT_CODE}"

"${PY}" -u -m pose_point_depth_mv.freeze_objaverse2k_slat_train16 \
  --cache_manifest "${TRAIN_CACHE}" \
  --lifting_cache_manifest "${TRAIN_LIFT}" \
  --output_dir "${SELECTION}" \
  --selection_seed "${SELECTION_SEED}" --object_count "${EXPECTED_OBJECTS}" \
  --resume
INDICES=$(tr -d '[:space:]' <"${SELECTION}/indices.txt")
test -n "${INDICES}"

if [ -s "${TRAIN_OUT}/checkpoints/step_000200.pt" ] && \
   [ -s "${TRAIN_OUT}/report.json" ]; then
  echo "reuse complete Objaverse2K Train16 training: ${TRAIN_OUT}"
else
  START=()
  if [ -s "${TRAIN_OUT}/checkpoints/last.pt" ]; then
    START=(--resume "${TRAIN_OUT}/checkpoints/last.pt")
  elif [ -d "${TRAIN_OUT}" ] && \
       [ -z "$(find "${TRAIN_OUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    rmdir "${TRAIN_OUT}"
  elif [ -e "${TRAIN_OUT}" ]; then
    echo "partial Train16 output has no resumable last.pt: ${TRAIN_OUT}" >&2
    exit 98
  fi
  CUDA_VISIBLE_DEVICES="${GPUS}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${TORCHRUN}" --standalone --nproc_per_node=4 \
    -m pose_point_depth_mv.train_native_slat_genrecon_no_vggt \
    --architecture v2 \
    --cache_manifest "${TRAIN_CACHE}" \
    --lifting_cache_manifest "${TRAIN_LIFT}" \
    --target_decoder_audit "${TARGET_AUDIT}" \
    --native_ss_report "${SS_REPORT}" \
    --stock_slat_freeze "${STOCK_FREEZE}" \
    --output_dir "${TRAIN_OUT}" \
    --pretrained Stable-X/trellis-vggt-v0-2 \
    --indices "${INDICES}" "${START[@]}" \
    --max_steps 200 --save_every 50 --log_every 10 \
    --grad_accum 2 --num_workers 0 --seed 42 \
    --lora_rank 8 --lora_alpha 16 --condition_channels 1024 \
    --new_lr 1e-4 --lora_lr 3e-5 --new_weight_decay 0.01 \
    --grad_clip 1.0 --warmup_ratio 0.02 --ema_decay 0.9995 \
    --p_uncond 0.1 --t_schedule uniform \
    --min_condition_views 1 --max_condition_views 8 \
    --stock_context_views all \
    --amp_dtype bf16 --gradient_checkpointing --verify_cache_hashes \
    >"${LOG_DIR}/train.log" 2>&1
fi
CHECKPOINT=${TRAIN_OUT}/checkpoints/step_000200.pt
test -s "${CHECKPOINT}"

PIDS=()
for WORKER in 0 1 2 3; do
  WORKER_OUT=${EVAL_ROOT}/worker_${WORKER}
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
    --checkpoint "${CHECKPOINT}" --model_label objaverse2k \
    --native_ss_report "${SS_REPORT}" \
    --stock_slat_freeze "${STOCK_FREEZE}" \
    --output_dir "${WORKER_OUT}" --weights ema \
    --joint_seeds 42 --noise_seed 20260814 \
    --worker_index "${WORKER}" --num_workers "${EXPECTED_WORKERS}" \
    --expected_objects "${EXPECTED_OBJECTS}" \
    --training_overlap --object_selection_seed "${SELECTION_SEED}" \
    --fixed_axis_evaluation --surface_samples 20000 --amp_dtype bf16 \
    "${RESUME[@]}" >"${LOG_DIR}/eval_worker_${WORKER}.log" 2>&1 &
  PIDS+=("$!")
done
FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then FAILED=1; fi
done
if [ "${FAILED}" -ne 0 ]; then
  echo "At least one Train16 evaluation worker failed; inspect ${LOG_DIR}" >&2
  exit 2
fi

REPORTS=()
for WORKER in 0 1 2 3; do
  REPORT=${EVAL_ROOT}/worker_${WORKER}/report.json
  test -s "${REPORT}"
  REPORTS+=("${REPORT}")
done
REPORT_CSV=$(IFS=,; echo "${REPORTS[*]}")
"${PY}" -u -m pose_point_depth_mv.summarize_objaverse2k_train_stock_full \
  --worker_reports "${REPORT_CSV}" --output_dir "${SUMMARY}" \
  --expected_workers "${EXPECTED_WORKERS}" \
  --expected_objects "${EXPECTED_OBJECTS}" \
  --bootstrap_samples 10000 --resume

test -s "${SUMMARY}/report.json"
test -s "${SUMMARY}/summary.txt"
cat "${SUMMARY}/summary.txt"
