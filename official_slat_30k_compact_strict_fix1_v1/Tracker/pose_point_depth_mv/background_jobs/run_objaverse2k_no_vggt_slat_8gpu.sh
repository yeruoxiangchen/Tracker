#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

TORCHRUN=/home/zjr/anaconda3/envs/reconviagen/bin/torchrun
PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
GPUS=${OBJAVERSE2K_SLAT_8GPU_GPUS:-0,1,2,3,4,5,6,7}
RUN=${OBJAVERSE2K_SLAT_RUN:-/data/zjr/objaverse2k_no_vggt_slat_20260811_v1}
SPLIT=${RUN}/split_dev64_v1
TRAIN_CACHE=${RUN}/slat_cache_train_seed42_merged_v1/manifest.json
TRAIN_LIFT=${SPLIT}/train/lifting_manifest.json
TARGET_AUDIT=${RUN}/slat_target_decoder_audit_dev32_v1/report.json
SS_RUN=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1
SS_REPORT=${SS_RUN}/ss_eval_synthetic_dev32_fixedcfg3_count125_v3/report.json
SS_AUDIT64=/data/zjr/objaverse2k_frozen_ss_audit64_20260811_v1/aggregate_v1/report.json
STOCK_FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json
PARENT=${SS_RUN}/slat_mixed_step2000_seed42_2gpu_v1/checkpoints/step_002000.pt
OUT=${RUN}/slat_objaverse2135_step2000_seed42_8gpu_bs8_v1
STATE=${RUN}/logs/slat_objaverse2135_8gpu_bs8_training.state
EXIT_CODE=${RUN}/logs/slat_objaverse2135_8gpu_bs8_training.exit_code
LOCK=${RUN}/logs/slat_objaverse2135_8gpu_bs8_training.lock
OLD_LOCK=${RUN}/logs/slat_objaverse2135_training.lock

IFS=',' read -r -a GPU_ARRAY <<<"${GPUS}"
if [ "${#GPU_ARRAY[@]}" -ne 8 ] || \
   [ "$(printf '%s\n' "${GPU_ARRAY[@]}" | sort -u | wc -l)" -ne 8 ]; then
  echo "OBJAVERSE2K_SLAT_8GPU_GPUS requires eight distinct GPUs" >&2
  exit 96
fi
mkdir -p "${RUN}/logs"

# Hold the old lock too, so a four-GPU process cannot overlap this run even
# when it was launched in the foreground rather than through systemd.
exec 8>"${OLD_LOCK}"
if ! flock -n 8; then
  echo "the four-GPU Objaverse2K training process is still running" >&2
  exit 97
fi
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "Objaverse2K eight-GPU SLat training is already running" >&2
  exit 99
fi

for PARTITION in 0 1 2 3; do
  UNIT=tracker-objaverse5k-direct-dino-p${PARTITION}-v1.service
  if systemctl --user is-active --quiet "${UNIT}"; then
    echo "DINO partition is still active: ${UNIT}" >&2
    exit 97
  fi
done
if systemctl --user is-active --quiet \
  tracker-objaverse5k-direct-dino-shards-v1.service; then
  echo "legacy DINO watcher is still active" >&2
  exit 97
fi
if [ "$(find /data/zjr/objaverse5k_direct_dino_cache_shards_20260811_v1/shards \
  -name _DINO_ONLY_LIFTING_COMPLETE.json | wc -l)" -ne 16 ]; then
  echo "Objaverse5K DINO cache is not complete (requires 16/16)" >&2
  exit 97
fi

finish() {
  RC=$?
  trap - EXIT
  printf '%s\n' "${RC}" >"${EXIT_CODE}"
  printf 'finished_at=%s rc=%s\n' "$(date --iso-8601=seconds)" "${RC}" >"${STATE}"
  exit "${RC}"
}
trap finish EXIT
printf 'started_at=%s state=running gpus=%s world_size=8 grad_accum=1 global_batch=8 target_step=2000\n' \
  "$(date --iso-8601=seconds)" "${GPUS}" >"${STATE}"
rm -f "${EXIT_CODE}"

for REQUIRED in \
  "${TRAIN_CACHE}" "${TRAIN_LIFT}" "${TARGET_AUDIT}" "${SS_REPORT}" \
  "${SS_AUDIT64}" "${STOCK_FREEZE}" "${PARENT}" \
  "${RUN}/slat_cache_train_seed42_merged_v1/_OBJAVERSE2K_SLAT_CACHE_MERGE_COMPLETE.json"; do
  test -s "${REQUIRED}"
done

"${PY}" - "${TRAIN_CACHE}" "${TRAIN_LIFT}" "${TARGET_AUDIT}" \
  "${SS_AUDIT64}" "${PARENT}" <<'PY'
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
print({"objects": 2135, "parent": "M8 step2000 EMA", "global_batch": 8})
PY

if [ -s "${OUT}/report.json" ] && [ -s "${OUT}/checkpoints/step_002000.pt" ]; then
  echo "Objaverse2K eight-GPU SLat training already complete: ${OUT}"
  exit 0
fi
if [ -s "${OUT}/checkpoints/last.pt" ]; then
  START=(--resume "${OUT}/checkpoints/last.pt")
elif [ -e "${OUT}" ]; then
  echo "eight-GPU output exists without resumable last.pt: ${OUT}" >&2
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
  --max_steps 2000 --run_until_step 0 \
  --save_every 200 --log_every 10 \
  --grad_accum 1 --num_workers 0 --seed 42 \
  --lora_rank 8 --lora_alpha 16 --condition_channels 1024 \
  --new_lr 1e-4 --lora_lr 3e-5 --new_weight_decay 0.01 \
  --grad_clip 1.0 --warmup_ratio 0.02 --ema_decay 0.9995 \
  --p_uncond 0.1 --t_logit_mean 1.0 --t_logit_std 1.0 \
  --min_condition_views 1 --max_condition_views 16 \
  --stock_context_views all \
  --amp_dtype bf16 --gradient_checkpointing --verify_cache_hashes

test -s "${OUT}/report.json"
test -s "${OUT}/checkpoints/step_002000.pt"
