#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
TORCHRUN=/home/zjr/anaconda3/envs/reconviagen/bin/torchrun
GPUS=${TRAIN_GPUS:-4,5}
ADAPT=/data/zjr/native_v2_real500_domain_adapt_20260806_v2
CACHE=${ADAPT}/slat_cache_train_adaptedss_seed42_v2/manifest.json
LIFT=${ADAPT}/cache_train_real_runtime_o_v2/lifting_manifest.json
AUDIT=${ADAPT}/slat_target_decoder_audit_train32_v2/report.json
SS_REPORT=${ADAPT}/ss_step1000_eval_dev16_32_seed424344_v2/report.json
FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json
PARENT=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/train868_step2000_seed42_2gpu_v1/checkpoints/step_002000.pt
GATE=${ADAPT}/S6_NATIVE_SLAT_TRAINING_READY.json
OUT=${ADAPT}/slat_v2_real_step1000_seed42_2gpu_v2
STATE=${ADAPT}/logs/S6_real_slat_background.state
EXIT_CODE=${ADAPT}/logs/S6_real_slat_background.exit_code
LOCK=${ADAPT}/logs/S6_real_slat_background.lock

mkdir -p "${ADAPT}/logs"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "S6 resume refused: another S6 background job holds ${LOCK}" >&2
  exit 99
fi

finish() {
  RC=$?
  trap - EXIT
  printf '%s\n' "${RC}" > "${EXIT_CODE}"
  printf 'finished_at=%s rc=%s\n' "$(date --iso-8601=seconds)" "${RC}" > "${STATE}"
  echo "S6 background job finished: rc=${RC}"
  exit "${RC}"
}
trap finish EXIT

printf 'started_at=%s state=running gpus=%s\n' \
  "$(date --iso-8601=seconds)" "${GPUS}" > "${STATE}"
rm -f "${EXIT_CODE}"

if [ "$(awk -F, '{print NF}' <<<"${GPUS}")" -ne 2 ]; then
  echo "TRAIN_GPUS must contain exactly two comma-separated GPU indices" >&2
  exit 96
fi

if pgrep -u "$(id -u)" -af -- '-m pose_point_depth_mv\.train_native_slat_genrecon( |$)' >/dev/null; then
  echo "S6 resume refused: a legacy v2 SLat training process is still running" >&2
  pgrep -u "$(id -u)" -af -- '-m pose_point_depth_mv\.train_native_slat_genrecon( |$)' >&2
  exit 99
fi

for REQUIRED in "${CACHE}" "${LIFT}" "${AUDIT}" "${SS_REPORT}" "${FREEZE}" "${PARENT}" "${GATE}"; do
  test -s "${REQUIRED}"
done

"${PY}" - "${CACHE}" "${AUDIT}" "${GATE}" <<'PY'
import json
import sys

cache, audit, gate = (json.load(open(path, encoding="utf-8")) for path in sys.argv[1:])
assert cache["materialized"] is True and cache["object_count"] >= 350
assert audit["passed"] is True and audit["summary"]["object_count"] == 32
assert gate["passed"] is True and gate["native_slat_training_ready"] is True
print({"cache_objects": cache["object_count"], "decoder_audit": True, "training_ready": True})
PY

if [ -s "${OUT}/report.json" ] && [ -s "${OUT}/checkpoints/step_001000.pt" ]; then
  echo "S6 already complete: ${OUT}"
  exit 0
fi

if [ -s "${OUT}/checkpoints/last.pt" ]; then
  START_ARGS=(--resume "${OUT}/checkpoints/last.pt")
  echo "S6 resuming from ${OUT}/checkpoints/last.pt"
elif [ -e "${OUT}" ]; then
  echo "S6 output exists without a resumable last.pt: ${OUT}" >&2
  exit 98
else
  START_ARGS=(--init_checkpoint "${PARENT}" --init_weights ema)
  echo "S6 starting from v2 Full SLat EMA"
fi

CUDA_VISIBLE_DEVICES="${GPUS}" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${TORCHRUN}" --standalone --nproc_per_node=2 \
  -m pose_point_depth_mv.train_native_slat_genrecon \
  --architecture v2 \
  --cache_manifest "${CACHE}" \
  --lifting_cache_manifest "${LIFT}" \
  --target_decoder_audit "${AUDIT}" \
  --native_ss_report "${SS_REPORT}" \
  --stock_slat_freeze "${FREEZE}" \
  --output_dir "${OUT}" \
  "${START_ARGS[@]}" \
  --max_steps 1000 --save_every 250 --log_every 10 --grad_accum 4 \
  --seed 42 --new_lr 5e-5 --lora_lr 1e-5 \
  --min_condition_views 1 --max_condition_views 8 \
  --amp_dtype bf16 --gradient_checkpointing --verify_cache_hashes

test -s "${OUT}/report.json"
test -s "${OUT}/checkpoints/step_001000.pt"
