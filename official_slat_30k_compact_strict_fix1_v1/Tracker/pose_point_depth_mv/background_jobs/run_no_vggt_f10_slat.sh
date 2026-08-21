#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
GPU=${NO_VGGT_SLAT_GPU:-${NO_VGGT_SLAT_GPUS:-0}}
RUN=/data/zjr/native_ss_no_vggt_mixed1k_20260807_v1
TRAIN_CACHE=${RUN}/slat_cache_train868_seed42_v1/manifest.json
TRAIN_LIFT=${RUN}/lifting_train868_dino_only_v1/lifting_manifest.json
TARGET_AUDIT=${RUN}/slat_target_decoder_audit32_v1/report.json
SS_REPORT=${RUN}/ss_eval_final32_step2000_ema_sourcebalanced_v2/report.json
STOCK_FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json
PARENT=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/train868_step2000_seed42_2gpu_v1/checkpoints/step_002000.pt
OUT=${RUN}/slat868_step2000_seed42_1gpu_v1
STATE=${RUN}/logs/F10_slat_training_background.state
EXIT_CODE=${RUN}/logs/F10_slat_training_background.exit_code
LOCK=${RUN}/logs/F10_slat_training_background.lock

mkdir -p "${RUN}/logs"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "F10 start refused: another F10 background job holds ${LOCK}" >&2
  exit 99
fi

finish() {
  RC=$?
  trap - EXIT
  printf '%s\n' "${RC}" > "${EXIT_CODE}"
  printf 'finished_at=%s rc=%s\n' "$(date --iso-8601=seconds)" "${RC}" > "${STATE}"
  echo "F10 background job finished: rc=${RC}"
  exit "${RC}"
}
trap finish EXIT

printf 'started_at=%s state=running gpu=%s\n' \
  "$(date --iso-8601=seconds)" "${GPU}" > "${STATE}"
rm -f "${EXIT_CODE}"

if [[ ! "${GPU}" =~ ^[0-9]+$ ]]; then
  echo "NO_VGGT_SLAT_GPU must contain exactly one non-negative GPU index" >&2
  exit 96
fi

if pgrep -u "$(id -u)" -af -- '-m pose_point_depth_mv\.train_native_slat_genrecon_no_vggt( |$)' >/dev/null; then
  echo "F10 start refused: a legacy no-VGGT SLat training process is still running" >&2
  pgrep -u "$(id -u)" -af -- '-m pose_point_depth_mv\.train_native_slat_genrecon_no_vggt( |$)' >&2
  exit 99
fi

for REQUIRED in "${TRAIN_CACHE}" "${TRAIN_LIFT}" "${TARGET_AUDIT}" "${SS_REPORT}" "${STOCK_FREEZE}" "${PARENT}"; do
  test -s "${REQUIRED}"
done

"${PY}" - "${TARGET_AUDIT}" "${SS_REPORT}" <<'PY'
import json
import sys

audit = json.load(open(sys.argv[1], encoding="utf-8"))
ss = json.load(open(sys.argv[2], encoding="utf-8"))
assert audit["passed"] is True and audit["summary"]["object_count"] == 32
assert ss["passed"] is True
print({"target_decoder_audit": True, "formal_ss_final32": True})
PY

if [ -s "${OUT}/report.json" ] && [ -s "${OUT}/checkpoints/step_002000.pt" ]; then
  echo "F10 already complete: ${OUT}"
  exit 0
fi

if [ -s "${OUT}/checkpoints/last.pt" ]; then
  START_ARGS=(--resume "${OUT}/checkpoints/last.pt")
  echo "F10 resuming from ${OUT}/checkpoints/last.pt"
elif [ -e "${OUT}" ]; then
  echo "F10 output exists without a resumable last.pt: ${OUT}" >&2
  exit 98
else
  START_ARGS=(--init_checkpoint "${PARENT}" --init_weights ema)
  echo "F10 initializing from v2 Full SLat EMA"
fi

CUDA_VISIBLE_DEVICES="${GPU}" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${PY}" -u -m pose_point_depth_mv.train_native_slat_genrecon_no_vggt \
  --architecture v2 \
  --cache_manifest "${TRAIN_CACHE}" \
  --lifting_cache_manifest "${TRAIN_LIFT}" \
  --target_decoder_audit "${TARGET_AUDIT}" \
  --native_ss_report "${SS_REPORT}" \
  --stock_slat_freeze "${STOCK_FREEZE}" \
  --output_dir "${OUT}" \
  --pretrained Stable-X/trellis-vggt-v0-2 \
  --indices all \
  "${START_ARGS[@]}" \
  --max_steps 2000 --save_every 200 --log_every 10 \
  --grad_accum 8 --num_workers 0 --seed 42 \
  --lora_rank 8 --lora_alpha 16 --condition_channels 1024 \
  --new_lr 1e-4 --lora_lr 3e-5 --new_weight_decay 0.01 \
  --grad_clip 1.0 --warmup_ratio 0.02 --ema_decay 0.9995 \
  --p_uncond 0.1 --t_logit_mean 1.0 --t_logit_std 1.0 \
  --min_condition_views 1 --max_condition_views 16 \
  --amp_dtype bf16 --gradient_checkpointing --verify_cache_hashes

test -s "${OUT}/report.json"
test -s "${OUT}/checkpoints/step_002000.pt"
