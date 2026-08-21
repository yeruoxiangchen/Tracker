#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

TORCHRUN=/home/zjr/anaconda3/envs/reconviagen/bin/torchrun
PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
GPUS=${OBJAVERSE2K_SLAT_GPUS:-0,5,6,7}
RUN=${OBJAVERSE2K_SLAT_RUN:-/data/zjr/objaverse2k_no_vggt_slat_20260811_v1}
RUN_UNTIL_STEP=${OBJAVERSE2K_SLAT_RUN_UNTIL_STEP:-0}
SPLIT=${RUN}/split_dev64_v1
TRAIN_CACHE=${RUN}/slat_cache_train_seed42_merged_v1/manifest.json
TRAIN_LIFT=${SPLIT}/train/lifting_manifest.json
TARGET_AUDIT=${RUN}/slat_target_decoder_audit_dev32_v1/report.json
SS_RUN=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1
SS_REPORT=${SS_RUN}/ss_eval_synthetic_dev32_fixedcfg3_count125_v3/report.json
SS_AUDIT64=/data/zjr/objaverse2k_frozen_ss_audit64_20260811_v1/aggregate_v1/report.json
STOCK_FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json
PARENT=${SS_RUN}/slat_mixed_step2000_seed42_2gpu_v1/checkpoints/step_002000.pt
OUT=${RUN}/slat_objaverse2135_step2000_seed42_4gpu_v1
STATE=${RUN}/logs/slat_objaverse2135_training.state
EXIT_CODE=${RUN}/logs/slat_objaverse2135_training.exit_code
LOCK=${RUN}/logs/slat_objaverse2135_training.lock

IFS=',' read -r -a GPU_ARRAY <<<"${GPUS}"
if [ "${#GPU_ARRAY[@]}" -ne 4 ] || [ "$(printf '%s\n' "${GPU_ARRAY[@]}" | sort -u | wc -l)" -ne 4 ]; then
  echo "OBJAVERSE2K_SLAT_GPUS requires four distinct GPUs" >&2
  exit 96
fi
if [[ ! "${RUN_UNTIL_STEP}" =~ ^[0-9]+$ ]] || [ "${RUN_UNTIL_STEP}" -gt 2000 ]; then
  echo "OBJAVERSE2K_SLAT_RUN_UNTIL_STEP must be 0 or lie in 1..2000" >&2
  exit 96
fi

mkdir -p "${RUN}/logs"
exec 9>"${LOCK}"
if ! flock -n 9; then echo "Objaverse2K SLat training is already running" >&2; exit 99; fi
finish() {
  RC=$?
  trap - EXIT
  printf '%s\n' "${RC}" >"${EXIT_CODE}"
  printf 'finished_at=%s rc=%s\n' "$(date --iso-8601=seconds)" "${RC}" >"${STATE}"
  exit "${RC}"
}
trap finish EXIT
printf 'started_at=%s state=running gpus=%s run_until_step=%s\n' \
  "$(date --iso-8601=seconds)" "${GPUS}" "${RUN_UNTIL_STEP}" >"${STATE}"
rm -f "${EXIT_CODE}"

for REQUIRED in \
  "${TRAIN_CACHE}" "${TRAIN_LIFT}" "${TARGET_AUDIT}" "${SS_REPORT}" \
  "${SS_AUDIT64}" "${STOCK_FREEZE}" "${PARENT}" \
  "${RUN}/slat_cache_train_seed42_merged_v1/_OBJAVERSE2K_SLAT_CACHE_MERGE_COMPLETE.json"; do
  test -s "${REQUIRED}"
done

"${PY}" - "${TRAIN_CACHE}" "${TRAIN_LIFT}" "${TARGET_AUDIT}" "${SS_AUDIT64}" "${PARENT}" <<'PY'
import json, sys, torch
cache, lift, audit, ss_audit = [json.load(open(path, encoding="utf-8")) for path in sys.argv[1:5]]
parent = torch.load(sys.argv[5], map_location="cpu")
assert cache["materialized"] is True and cache["object_count"] == 2135
assert lift["object_count"] == 2135 and lift["objaverse2k_split"]["name"] == "train"
assert audit["passed"] is True and audit["summary"]["object_count"] >= 32
assert ss_audit["passed"] is True and ss_audit["formal"] is False
assert ss_audit["object_count"] == 64 and all(ss_audit["checks"].values())
assert parent["format"] == "pose_point_depth_mv.native_slat_genrecon_no_vggt.v1"
assert parent["step"] == 2000
assert parent["args"].get("stock_context_views", "all") == "all"
print({"objects": 2135, "ss_audit64": "passed", "parent": "M8 step2000 EMA"})
PY

if [ -s "${OUT}/report.json" ] && [ -s "${OUT}/checkpoints/step_002000.pt" ]; then
  echo "Objaverse2K SLat training already complete: ${OUT}"
  exit 0
fi
if [ -s "${OUT}/checkpoints/last.pt" ]; then
  START=(--resume "${OUT}/checkpoints/last.pt")
elif [ -e "${OUT}" ]; then
  echo "training output exists without resumable last.pt: ${OUT}" >&2
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
"${TORCHRUN}" --standalone --nproc_per_node=4 \
  -m pose_point_depth_mv.train_native_slat_genrecon_no_vggt \
  --architecture v2 \
  --cache_manifest "${TRAIN_CACHE}" \
  --lifting_cache_manifest "${TRAIN_LIFT}" \
  --target_decoder_audit "${TARGET_AUDIT}" \
  --native_ss_report "${SS_REPORT}" \
  --stock_slat_freeze "${STOCK_FREEZE}" \
  --output_dir "${OUT}" \
  --pretrained Stable-X/trellis-vggt-v0-2 \
  --indices all "${START[@]}" \
  --max_steps 2000 --run_until_step "${RUN_UNTIL_STEP}" \
  --save_every 200 --log_every 10 \
  --grad_accum 2 --num_workers 0 --seed 42 \
  --lora_rank 8 --lora_alpha 16 --condition_channels 1024 \
  --new_lr 1e-4 --lora_lr 3e-5 --new_weight_decay 0.01 \
  --grad_clip 1.0 --warmup_ratio 0.02 --ema_decay 0.9995 \
  --p_uncond 0.1 --t_logit_mean 1.0 --t_logit_std 1.0 \
  --min_condition_views 1 --max_condition_views 16 \
  --stock_context_views all \
  --amp_dtype bf16 --gradient_checkpointing --verify_cache_hashes

if [ "${RUN_UNTIL_STEP}" -eq 0 ] || [ "${RUN_UNTIL_STEP}" -eq 2000 ]; then
  test -s "${OUT}/report.json"
  test -s "${OUT}/checkpoints/step_002000.pt"
else
  test -s "${OUT}/checkpoints/step_$(printf '%06d' "${RUN_UNTIL_STEP}").pt"
fi
