#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
GPU=${MIXED_NO_VGGT_SS_GPU:-6}
RUN=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1
MIXED=${RUN}/manifests/mixed_ss_lifting_synth868_real376_v1.json
CONTRACT=${RUN}/contracts/ss_real_full_ema_v1.json
PARENT=/data/zjr/native_v2_real500_domain_adapt_20260806_v2/ss_real_step1000_seed42_2gpu_v2/checkpoints/last.pt
OUT=${RUN}/ss_mixed_step2000_seed42_1gpu_v1
STATE=${RUN}/logs/M3_ss_training.state
EXIT_CODE=${RUN}/logs/M3_ss_training.exit_code
LOCK=${RUN}/logs/M3_ss_training.lock

mkdir -p "${RUN}/logs"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "M3 refused: another mixed SS job holds ${LOCK}" >&2
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
printf 'started_at=%s state=running gpu=%s\n' "$(date --iso-8601=seconds)" "${GPU}" > "${STATE}"
rm -f "${EXIT_CODE}"

for REQUIRED in "${MIXED}" "${CONTRACT}" "${PARENT}"; do test -s "${REQUIRED}"; done
if [ -s "${OUT}/report.json" ] && [ -s "${OUT}/checkpoints/step_002000.pt" ]; then
  echo "M3 already complete: ${OUT}"
  exit 0
fi
if [ -s "${OUT}/checkpoints/last.pt" ]; then
  START=(--resume "${OUT}/checkpoints/last.pt")
elif [ -e "${OUT}" ]; then
  echo "M3 output exists without resumable last.pt: ${OUT}" >&2
  exit 98
else
  START=(--init_checkpoint "${PARENT}" --init_weights ema)
fi

CUDA_VISIBLE_DEVICES="${GPU}" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${PY}" -u -m pose_point_depth_mv.train_native_ss_genrecon_no_vggt_mixed \
  --cache_manifest "${MIXED}" \
  --migration_contract "${CONTRACT}" \
  --output_dir "${OUT}" \
  --pretrained Stable-X/trellis-vggt-v0-2 \
  --indices all \
  "${START[@]}" \
  --max_steps 2000 --save_every 200 --log_every 10 \
  --grad_accum 8 --num_workers 0 --seed 42 \
  --lora_rank 8 --lora_alpha 16 --condition_channels 1024 \
  --new_lr 5e-5 --lora_lr 1e-5 --new_weight_decay 0.01 \
  --grad_clip 1.0 --warmup_ratio 0.02 --ema_decay 0.9995 \
  --p_uncond 0.1 --t_logit_mean 1.0 --t_logit_std 1.0 \
  --min_condition_views 1 --max_condition_views 8 \
  --amp_dtype bf16 --gradient_checkpointing

test -s "${OUT}/report.json"
test -s "${OUT}/checkpoints/step_002000.pt"

