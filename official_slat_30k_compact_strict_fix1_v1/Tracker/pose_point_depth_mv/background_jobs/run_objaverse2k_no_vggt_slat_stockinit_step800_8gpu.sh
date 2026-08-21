#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

TORCHRUN=/home/zjr/anaconda3/envs/reconviagen/bin/torchrun
PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
GPUS=${OBJAVERSE2K_STOCKINIT_GPUS:-0,1,2,3,4,5,6,7}
RUN=${OBJAVERSE2K_SLAT_RUN:-/data/zjr/objaverse2k_no_vggt_slat_20260811_v1}
RUN_UNTIL_STEP=800
SPLIT=${RUN}/split_dev64_v1
TRAIN_CACHE=${RUN}/slat_cache_train_seed42_merged_v1/manifest.json
TRAIN_LIFT=${SPLIT}/train/lifting_manifest.json
TARGET_AUDIT=${RUN}/slat_target_decoder_audit_dev32_v1/report.json
SS_RUN=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1
SS_REPORT=${SS_RUN}/ss_eval_synthetic_dev32_fixedcfg3_count125_v3/report.json
SS_AUDIT64=/data/zjr/objaverse2k_frozen_ss_audit64_20260811_v1/aggregate_v1/report.json
STOCK_FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json
OUT=${RUN}/slat_objaverse2135_stockinit_step2000_seed42_8gpu_bs8_v1
STATE=${RUN}/logs/slat_objaverse2135_stockinit_step800_8gpu.state
EXIT_CODE=${RUN}/logs/slat_objaverse2135_stockinit_step800_8gpu.exit_code
LOCK=${RUN}/logs/slat_objaverse2135_stockinit_step800_8gpu.lock
FOUR_GPU_LOCK=${RUN}/logs/slat_objaverse2135_stockinit_step800_4gpu.lock
M8_EIGHT_GPU_LOCK=${RUN}/logs/slat_objaverse2135_8gpu_bs8_training.lock
GENERAL_TRAINING_LOCK=${RUN}/logs/slat_objaverse2135_training.lock

IFS=',' read -r -a GPU_ARRAY <<<"${GPUS}"
if [ "${#GPU_ARRAY[@]}" -ne 8 ] || \
   [ "$(printf '%s\n' "${GPU_ARRAY[@]}" | sort -u | wc -l)" -ne 8 ]; then
  echo "OBJAVERSE2K_STOCKINIT_GPUS requires eight distinct GPUs" >&2
  exit 96
fi

for REQUIRED in \
  "${TRAIN_CACHE}" "${TRAIN_LIFT}" "${TARGET_AUDIT}" "${SS_REPORT}" \
  "${SS_AUDIT64}" "${STOCK_FREEZE}" \
  "${RUN}/slat_cache_train_seed42_merged_v1/_OBJAVERSE2K_SLAT_CACHE_MERGE_COMPLETE.json"; do
  test -s "${REQUIRED}"
done

mkdir -p "${RUN}/logs"
exec 6>"${GENERAL_TRAINING_LOCK}"
if ! flock -n 6; then
  echo "another Objaverse2K SLat training process is still running" >&2
  exit 97
fi
exec 7>"${FOUR_GPU_LOCK}"
if ! flock -n 7; then
  echo "the four-GPU Stock-init process is still running" >&2
  exit 97
fi
exec 8>"${M8_EIGHT_GPU_LOCK}"
if ! flock -n 8; then
  echo "the M8-init eight-GPU process is still running" >&2
  exit 97
fi
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "Objaverse2K Stock-init step800 eight-GPU training is already running" >&2
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
printf 'started_at=%s state=running gpus=%s world_size=8 grad_accum=1 global_batch=8 run_until_step=%s initialization=stock\n' \
  "$(date --iso-8601=seconds)" "${GPUS}" "${RUN_UNTIL_STEP}" >"${STATE}"
rm -f "${EXIT_CODE}"

"${PY}" - "${TRAIN_CACHE}" "${TRAIN_LIFT}" "${TARGET_AUDIT}" "${SS_AUDIT64}" <<'PY'
import json, sys
cache, lift, audit, ss_audit = [
    json.load(open(path, encoding="utf-8")) for path in sys.argv[1:]
]
assert cache["materialized"] is True and cache["object_count"] == 2135
assert lift["object_count"] == 2135
assert lift["objaverse2k_split"]["name"] == "train"
assert audit["passed"] is True and audit["summary"]["object_count"] >= 32
assert ss_audit["passed"] is True and ss_audit["formal"] is False
assert ss_audit["object_count"] == 64 and all(ss_audit["checks"].values())
print({
    "objects": 2135,
    "ss_audit64": "passed",
    "initialization": "fresh Stock-equivalent zero adapter",
    "world_size": 8,
    "global_batch": 8,
})
PY

if [ -s "${OUT}/checkpoints/step_000800.pt" ] && \
   [ -s "${OUT}/stage_report_step_000800.json" ]; then
  echo "Objaverse2K Stock-init step800 eight-GPU training already complete: ${OUT}"
else
  if [ -s "${OUT}/checkpoints/last.pt" ]; then
    START=(--resume "${OUT}/checkpoints/last.pt")
  elif [ -d "${OUT}" ] && \
       [ -z "$(find "${OUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    # torchrun may create the output directory before the first checkpoint.
    # Fresh training requires rank 0 to create it with exist_ok=False.
    rmdir "${OUT}"
    echo "removed empty pre-checkpoint output directory: ${OUT}"
    START=()
  elif [ -e "${OUT}" ]; then
    echo "Stock-init eight-GPU output exists without resumable last.pt: ${OUT}" >&2
    exit 98
  else
    # No --init_checkpoint is intentional: fresh v2 adapters are exactly
    # Stock-equivalent before their first optimizer update.
    START=()
  fi

  CUDA_VISIBLE_DEVICES="${GPUS}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${TORCHRUN}" --standalone --nproc_per_node=8 \
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
    --grad_accum 1 --num_workers 0 --seed 42 \
    --lora_rank 8 --lora_alpha 16 --condition_channels 1024 \
    --new_lr 1e-4 --lora_lr 3e-5 --new_weight_decay 0.01 \
    --grad_clip 1.0 --warmup_ratio 0.02 --ema_decay 0.9995 \
    --p_uncond 0.1 --t_logit_mean 1.0 --t_logit_std 1.0 \
    --min_condition_views 1 --max_condition_views 16 \
    --stock_context_views all \
    --amp_dtype bf16 --gradient_checkpointing --verify_cache_hashes
fi

test -s "${OUT}/checkpoints/step_000800.pt"
test -s "${OUT}/stage_report_step_000800.json"
"${PY}" - "${OUT}/checkpoints/step_000800.pt" \
  "${OUT}/stage_report_step_000800.json" "${OUT}" <<'PY'
import json, sys, torch
checkpoint = torch.load(sys.argv[1], map_location="cpu")
stage = json.load(open(sys.argv[2], encoding="utf-8"))
output_dir = sys.argv[3]
args = checkpoint["args"]
assert checkpoint["format"] == "pose_point_depth_mv.native_slat_genrecon_no_vggt.v1"
assert checkpoint["step"] == 800 and args["max_steps"] == 2000
assert args["grad_accum"] == 1 and args["seed"] == 42
assert args["init_checkpoint"] == ""
assert args["output_dir"] == output_dir
assert "initialization" not in checkpoint["model_summary"]
assert stage["stage_complete"] is True and stage["step"] == 800
assert stage["initial_stock_audit"]["passed"] is True
print({
    "step": 800,
    "max_steps": 2000,
    "initialization": "Stock-equivalent",
    "world_size": 8,
    "global_batch": 8,
    "initial_stock_audit": "passed",
})
PY
