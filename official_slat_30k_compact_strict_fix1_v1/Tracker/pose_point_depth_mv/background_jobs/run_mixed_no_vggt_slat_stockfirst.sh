#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

TORCHRUN=/home/zjr/anaconda3/envs/reconviagen/bin/torchrun
GPUS=${STOCK_CONTEXT_TRAIN_GPUS:-4,5}
RUN_UNTIL_STEP=${STOCK_CONTEXT_RUN_UNTIL_STEP:-2000}
SOURCE=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1
ROOT=/data/zjr/mixed_no_vggt_slat_stock_context_2x2_20260811_v1
MIXED_SLAT=${SOURCE}/manifests/mixed_slat_synth868_real376_v1.json
MIXED_LIFT=${SOURCE}/manifests/mixed_ss_lifting_synth868_real376_v1.json
CONTRACT=${SOURCE}/contracts/slat_real_full_ema_v1.json
PARENT=/data/zjr/native_v2_real500_domain_adapt_20260806_v2/slat_v2_real_step1000_seed42_2gpu_v2/checkpoints/last.pt
SS_REPORT=${SOURCE}/ss_eval_synthetic_dev32_fixedcfg3_count125_v3/report.json
AUDIT=${SOURCE}/slat_target_decoder_audit32_v1/report.json
FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json
OUT=${ROOT}/stockfirst_train_step2000_seed42_2gpu_v1
STATE=${ROOT}/logs/stockfirst_training.state
EXIT_CODE=${ROOT}/logs/stockfirst_training.exit_code
LOCK=${ROOT}/logs/stockfirst_training.lock

mkdir -p "${ROOT}/logs"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "duplicate Stock-first SLat training job" >&2
  exit 99
fi
finish() {
  RC=$?
  trap - EXIT
  printf '%s\n' "${RC}" > "${EXIT_CODE}"
  printf 'finished_at=%s rc=%s\n' "$(date --iso-8601=seconds)" "${RC}" > "${STATE}"
  exit "${RC}"
}
trap finish EXIT
printf 'started_at=%s state=running gpus=%s stock_context=first run_until_step=%s max_steps=2000\n' \
  "$(date --iso-8601=seconds)" "${GPUS}" "${RUN_UNTIL_STEP}" > "${STATE}"
rm -f "${EXIT_CODE}"

if [ "$(awk -F, '{print NF}' <<<"${GPUS}")" -ne 2 ]; then
  echo "STOCK_CONTEXT_TRAIN_GPUS requires two GPUs" >&2
  exit 96
fi
if ! [[ "${RUN_UNTIL_STEP}" =~ ^[0-9]+$ ]] || \
   [ "${RUN_UNTIL_STEP}" -lt 1 ] || [ "${RUN_UNTIL_STEP}" -gt 2000 ]; then
  echo "STOCK_CONTEXT_RUN_UNTIL_STEP must be an integer in [1,2000]" >&2
  exit 96
fi
for REQUIRED in "${MIXED_SLAT}" "${MIXED_LIFT}" "${CONTRACT}" "${PARENT}" \
  "${SS_REPORT}" "${AUDIT}" "${FREEZE}"; do
  test -s "${REQUIRED}"
done
if [ -s "${OUT}/report.json" ] && [ -s "${OUT}/checkpoints/step_002000.pt" ]; then
  echo "Stock-first SLat training already complete: ${OUT}"
  exit 0
fi
STAGE_CKPT=$(printf '%s/checkpoints/step_%06d.pt' "${OUT}" "${RUN_UNTIL_STEP}")
STAGE_REPORT=$(printf '%s/stage_report_step_%06d.json' "${OUT}" "${RUN_UNTIL_STEP}")
if [ -s "${STAGE_CKPT}" ] && { [ "${RUN_UNTIL_STEP}" -lt 2000 ] && \
   [ -s "${STAGE_REPORT}" ] || [ "${RUN_UNTIL_STEP}" -eq 2000 ] && \
   [ -s "${OUT}/report.json" ]; }; then
  echo "Stock-first SLat stage already complete: step=${RUN_UNTIL_STEP} ${OUT}"
  exit 0
fi
if [ -s "${OUT}/checkpoints/last.pt" ]; then
  START=(--resume "${OUT}/checkpoints/last.pt")
elif [ -e "${OUT}" ]; then
  echo "Stock-first output exists without resumable last.pt" >&2
  exit 98
else
  START=(--init_checkpoint "${PARENT}" --init_weights ema)
fi

CUDA_VISIBLE_DEVICES="${GPUS}" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${TORCHRUN}" --standalone --nproc_per_node=2 \
  -m pose_point_depth_mv.train_native_slat_genrecon_no_vggt_mixed \
  --architecture v2 \
  --cache_manifest "${MIXED_SLAT}" \
  --lifting_cache_manifest "${MIXED_LIFT}" \
  --migration_contract "${CONTRACT}" \
  --target_decoder_audit "${AUDIT}" \
  --native_ss_report "${SS_REPORT}" \
  --stock_slat_freeze "${FREEZE}" \
  --output_dir "${OUT}" \
  --pretrained Stable-X/trellis-vggt-v0-2 \
  --indices all \
  "${START[@]}" \
  --max_steps 2000 --run_until_step "${RUN_UNTIL_STEP}" \
  --save_every 200 --log_every 10 \
  --grad_accum 4 --num_workers 0 --seed 42 \
  --lora_rank 8 --lora_alpha 16 --condition_channels 1024 \
  --new_lr 1e-4 --lora_lr 3e-5 --new_weight_decay 0.01 \
  --grad_clip 1.0 --warmup_ratio 0.02 --ema_decay 0.9995 \
  --p_uncond 0.1 --t_logit_mean 1.0 --t_logit_std 1.0 \
  --min_condition_views 1 --max_condition_views 16 \
  --stock_context_views first \
  --amp_dtype bf16 --gradient_checkpointing --verify_cache_hashes

test -s "${STAGE_CKPT}"
if [ "${RUN_UNTIL_STEP}" -eq 2000 ]; then
  test -s "${OUT}/report.json"
else
  test -s "${STAGE_REPORT}"
fi
