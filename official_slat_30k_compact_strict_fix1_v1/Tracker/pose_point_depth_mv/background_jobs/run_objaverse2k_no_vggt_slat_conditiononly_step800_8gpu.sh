#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

TORCHRUN=/home/zjr/anaconda3/envs/reconviagen/bin/torchrun
PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
GPUS=${OBJAVERSE2K_CONDITIONONLY_GPUS:-0,1,2,3,4,5,6,7}
NUM_WORKERS=${OBJAVERSE2K_CONDITIONONLY_NUM_WORKERS:-2}
GRADIENT_CHECKPOINTING=${OBJAVERSE2K_CONDITIONONLY_GRADIENT_CHECKPOINTING:-1}
RUN=${OBJAVERSE2K_SLAT_RUN:-/data/zjr/objaverse2k_no_vggt_slat_20260811_v1}
MAX_STEPS=800
SPLIT=${RUN}/split_dev64_v1
TRAIN_CACHE=${RUN}/slat_cache_train_seed42_merged_v1/manifest.json
TRAIN_LIFT=${SPLIT}/train/lifting_manifest.json
TARGET_AUDIT=${RUN}/slat_target_decoder_audit_dev32_v1/report.json
SS_RUN=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1
SS_REPORT=${SS_RUN}/ss_eval_synthetic_dev32_fixedcfg3_count125_v3/report.json
SS_AUDIT64=/data/zjr/objaverse2k_frozen_ss_audit64_20260811_v1/aggregate_v1/report.json
STOCK_FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json
OUT=${RUN}/slat_objaverse2135_stockinit_conditiononly_step800_seed42_8gpu_bs8_v1
STATE=${RUN}/logs/slat_objaverse2135_conditiononly_step800_8gpu.state
EXIT_CODE=${RUN}/logs/slat_objaverse2135_conditiononly_step800_8gpu.exit_code
LOCK=${RUN}/logs/slat_objaverse2135_conditiononly_step800_8gpu.lock
GENERAL_TRAINING_LOCK=${RUN}/logs/slat_objaverse2135_training.lock

IFS=',' read -r -a GPU_ARRAY <<<"${GPUS}"
if [ "${#GPU_ARRAY[@]}" -ne 8 ] || \
   [ "$(printf '%s\n' "${GPU_ARRAY[@]}" | sort -u | wc -l)" -ne 8 ]; then
  echo "OBJAVERSE2K_CONDITIONONLY_GPUS requires eight distinct GPUs" >&2
  exit 96
fi
if ! [[ "${NUM_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "OBJAVERSE2K_CONDITIONONLY_NUM_WORKERS must be a positive integer" >&2
  exit 96
fi
if [ "${GRADIENT_CHECKPOINTING}" != 0 ] && \
   [ "${GRADIENT_CHECKPOINTING}" != 1 ]; then
  echo "OBJAVERSE2K_CONDITIONONLY_GRADIENT_CHECKPOINTING must be 0 or 1" >&2
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
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "Objaverse2K condition-only training is already running" >&2
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
printf 'started_at=%s state=validating gpus=%s world_size=8 global_batch=8 max_steps=%s architecture=condition-only stock_context=all num_workers=%s gradient_checkpointing=%s\n' \
  "$(date --iso-8601=seconds)" "${GPUS}" "${MAX_STEPS}" \
  "${NUM_WORKERS}" "${GRADIENT_CHECKPOINTING}" >"${STATE}"
rm -f "${EXIT_CODE}"

# Hash every cache artifact once. Repeating this in all eight ranks only adds I/O.
"${PY}" - "${TRAIN_CACHE}" "${TRAIN_LIFT}" "${TARGET_AUDIT}" "${SS_AUDIT64}" <<'PY'
import json
import sys

from pose_point_depth_mv.native_3d_condition import NativeConditionSLatDataset

cache_path, lift_path, audit_path, ss_audit_path = sys.argv[1:]
cache, lift, audit, ss_audit = [
    json.load(open(path, encoding="utf-8"))
    for path in (cache_path, lift_path, audit_path, ss_audit_path)
]
assert cache["materialized"] is True and cache["object_count"] == 2135
assert lift["object_count"] == 2135
assert lift["objaverse2k_split"]["name"] == "train"
assert audit["passed"] is True and audit["summary"]["object_count"] >= 32
assert ss_audit["passed"] is True and ss_audit["formal"] is False
assert ss_audit["object_count"] == 64 and all(ss_audit["checks"].values())
dataset = NativeConditionSLatDataset(
    cache_path, lift_path, indices="all", verify_hashes=True
)
assert len({str(row["object_uid"]) for row in dataset.rows}) == 2135
print({
    "objects": 2135,
    "cache_hash_scan": "passed_once_before_torchrun",
    "ss_audit64": "passed",
    "initialization": "fresh Stock-equivalent zero condition",
    "architecture": "condition-only",
    "stock_context_views": "all",
    "world_size": 8,
    "global_batch": 8,
})
PY

printf 'started_at=%s state=running gpus=%s world_size=8 global_batch=8 max_steps=%s architecture=condition-only stock_context=all num_workers=%s gradient_checkpointing=%s\n' \
  "$(date --iso-8601=seconds)" "${GPUS}" "${MAX_STEPS}" \
  "${NUM_WORKERS}" "${GRADIENT_CHECKPOINTING}" >"${STATE}"

if [ -s "${OUT}/checkpoints/step_000800.pt" ] && [ -s "${OUT}/report.json" ]; then
  echo "Objaverse2K condition-only step800 training already complete: ${OUT}"
else
  if [ -s "${OUT}/checkpoints/last.pt" ]; then
    START=(--resume "${OUT}/checkpoints/last.pt")
  elif [ -d "${OUT}" ] && \
       [ -z "$(find "${OUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    rmdir "${OUT}"
    echo "removed empty pre-checkpoint output directory: ${OUT}"
    START=()
  elif [ -e "${OUT}" ]; then
    echo "condition-only output exists without resumable last.pt: ${OUT}" >&2
    exit 98
  else
    START=()
  fi

  CHECKPOINT_ARGS=()
  if [ "${GRADIENT_CHECKPOINTING}" = 1 ]; then
    CHECKPOINT_ARGS=(--gradient_checkpointing)
  fi

  CUDA_VISIBLE_DEVICES="${GPUS}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${TORCHRUN}" --standalone --nproc_per_node=8 \
    -m pose_point_depth_mv.train_native_slat_condition_only_no_vggt \
    --cache_manifest "${TRAIN_CACHE}" \
    --lifting_cache_manifest "${TRAIN_LIFT}" \
    --target_decoder_audit "${TARGET_AUDIT}" \
    --native_ss_report "${SS_REPORT}" \
    --stock_slat_freeze "${STOCK_FREEZE}" \
    --output_dir "${OUT}" \
    --pretrained Stable-X/trellis-vggt-v0-2 \
    --indices all "${START[@]}" \
    --max_steps "${MAX_STEPS}" --save_every 200 --log_every 10 \
    --grad_accum 1 --num_workers "${NUM_WORKERS}" --seed 42 \
    --condition_channels 1024 --condition_lr 1e-4 \
    --condition_weight_decay 0.01 \
    --grad_clip 1.0 --warmup_steps 40 --ema_decay 0.9995 \
    --p_uncond 0.1 --t_logit_mean 1.0 --t_logit_std 1.0 \
    --min_condition_views 1 --max_condition_views 16 \
    --amp_dtype bf16 "${CHECKPOINT_ARGS[@]}"
fi

test -s "${OUT}/checkpoints/step_000800.pt"
test -s "${OUT}/report.json"
"${PY}" - "${OUT}/checkpoints/step_000800.pt" "${OUT}/report.json" \
  "${OUT}" "${NUM_WORKERS}" "${GRADIENT_CHECKPOINTING}" <<'PY'
import json
import sys

import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
report = json.load(open(sys.argv[2], encoding="utf-8"))
output_dir = sys.argv[3]
num_workers = int(sys.argv[4])
gradient_checkpointing = bool(int(sys.argv[5]))
args = checkpoint["args"]
summary = checkpoint["model_summary"]
expected_format = "pose_point_depth_mv.native_slat_condition_only_no_vggt.v1"
assert checkpoint["format"] == expected_format and report["format"] == expected_format
assert checkpoint["step"] == 800 and args["max_steps"] == 800
assert args["grad_accum"] == 1 and args["seed"] == 42
assert args["num_workers"] == num_workers
assert args["gradient_checkpointing"] is gradient_checkpointing
assert args["verify_cache_hashes"] is False
assert args["output_dir"] == output_dir
assert args["warmup_steps"] == 40
assert summary["flow_lora"]["present"] is False
assert summary["flow_lora"]["parameter_count"] == 0
assert summary["context_view_fusion"]["trainable"] is False
assert summary["vggt_model_executed"] is False
assert set(checkpoint["model_trainable_state"]).issubset({
    name for name in checkpoint["model_trainable_state"]
    if name.startswith("aggregator.") or name.startswith("block_condition.")
})
assert report["passed"] is True and report["initial_stock_audit"]["passed"] is True
print({
    "step": 800,
    "initialization": "Stock-equivalent",
    "architecture": "condition-only",
    "lora_parameters": 0,
    "stock_context_views": "all",
    "world_size": 8,
    "global_batch": 8,
    "gradient_checkpointing": gradient_checkpointing,
    "initial_stock_audit": "passed",
})
PY
