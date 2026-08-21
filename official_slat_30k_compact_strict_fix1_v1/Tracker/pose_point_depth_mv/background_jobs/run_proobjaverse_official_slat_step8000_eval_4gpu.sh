#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
EVAL_GPUS=${EVAL_GPUS:-0,2,3,4}
RUN=/data/zjr/proobjaverse_official_slat_train2000_20260813_v1
CKPT=${RUN}/B_condition_lora_train2000_step8000_seed42_4gpu_v1/checkpoints/step_008000.pt
TRAIN_CACHE=${RUN}/cache_train2000_protocol2128_views8_v1
DEV_CACHE=${RUN}/cache_dev64_protocol2128_views8_v1
SS_REPORT=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1/ss_eval_synthetic_dev32_fixedcfg3_count125_v3/report.json
STOCK_FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json

if [ "$(awk -F, '{print NF}' <<<"${EVAL_GPUS}")" -ne 4 ]; then
  echo "blocked: EVAL_GPUS must contain exactly four GPU ids" >&2
  exit 2
fi
if [ "$(tr ',' '\n' <<<"${EVAL_GPUS}" | sort -u | wc -l)" -ne 4 ]; then
  echo "blocked: EVAL_GPUS must contain four distinct GPU ids" >&2
  exit 2
fi

for path in \
  "${CKPT}" \
  "${TRAIN_CACHE}/slat_manifest.json" \
  "${TRAIN_CACHE}/lifting_manifest.json" \
  "${DEV_CACHE}/slat_manifest.json" \
  "${DEV_CACHE}/lifting_manifest.json" \
  "${SS_REPORT}" \
  "${STOCK_FREEZE}"
do
  test -s "${path}"
done

IFS=, read -r GPU0 GPU1 GPU2 GPU3 <<<"${EVAL_GPUS}"
started=0
reused=0

for SPEC in \
  "train64 0 ${GPU0} 0 32 ${TRAIN_CACHE}" \
  "train64 1 ${GPU1} 32 64 ${TRAIN_CACHE}" \
  "dev64 0 ${GPU2} 0 32 ${DEV_CACHE}" \
  "dev64 1 ${GPU3} 32 64 ${DEV_CACHE}"
do
  read -r SPLIT SHARD DEVICE START END CACHE <<<"${SPEC}"
  OUT_ROOT=${RUN}/eval_${SPLIT}_B_scale_step8000_seed424344_4gpu_v1
  OUT=${OUT_ROOT}/shard${SHARD}_${START}_${END}
  LOG=${RUN}/logs/Q1_eval_${SPLIT}_step8000_shard${SHARD}.log
  UNIT=tracker-proobj-slat-step8000-${SPLIT}-shard${SHARD}-v1

  if [ -s "${OUT}/report.json" ]; then
    echo "reuse ${SPLIT} shard${SHARD}: ${OUT}/report.json"
    reused=$((reused + 1))
  elif systemctl --user is-active --quiet "${UNIT}.service"; then
    echo "already running: ${UNIT}.service"
    reused=$((reused + 1))
  elif [ -e "${OUT}" ]; then
    echo "blocked: partial immutable output exists: ${OUT}" >&2
    exit 2
  else
    mkdir -p "${OUT_ROOT}" "${RUN}/logs"
    systemd-run --user \
      --unit="${UNIT}" \
      --collect \
      --property=WorkingDirectory=/home/zjr/Tracker \
      --property=StandardOutput=append:${LOG} \
      --property=StandardError=append:${LOG} \
      /usr/bin/env \
        CUDA_VISIBLE_DEVICES="${DEVICE}" \
        HF_HUB_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 \
        ATTN_BACKEND=flash_attn \
        SPCONV_ALGO=native \
        MPLCONFIGDIR=/tmp/matplotlib \
        NUMBA_CACHE_DIR=/tmp/numba_cache \
        TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "${PY}" -u -m pose_point_depth_mv.evaluate_proobjaverse_official_slat_gt_support \
        --arm condition_lora \
        --cache_manifest "${CACHE}/slat_manifest.json" \
        --lifting_cache_manifest "${CACHE}/lifting_manifest.json" \
        --checkpoint "${CKPT}" \
        --native_ss_report "${SS_REPORT}" \
        --stock_slat_freeze "${STOCK_FREEZE}" \
        --output_dir "${OUT}" \
        --weights ema \
        --joint_seeds 42,43,44 \
        --max_objects 64 \
        --object_start "${START}" \
        --object_end "${END}" \
        --surface_samples 20000
    echo "started ${SPLIT} shard${SHARD} on physical GPU ${DEVICE}: ${UNIT}.service"
    echo "log: ${LOG}"
    started=$((started + 1))
  fi
done

echo "launcher complete: started=${started} reused_or_running=${reused}"
echo "current terminal remains open"
