#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PARTITION=${1:-}
if [[ ! "${PARTITION}" =~ ^[0-3]$ ]]; then
  echo "usage: $0 PARTITION  # 0..3 maps to GPU 1..4" >&2
  exit 96
fi

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
WORK=/data/zjr/objaverse5k_direct_dino_cache_shards_20260811_v1
WIDE=/data/zjr/objaverse5k_single_subject_render512_wideorbit_20260811_v2
GPU=$((PARTITION + 1))
FIRST=$((8 + PARTITION))
SECOND=$((12 + PARTITION))
SHARDS=${FIRST},${SECOND}
RENDER_UNIT=objaverse5k-wide-render-w${PARTITION}-20260811-v2.service
DINO_UNIT=tracker-objaverse5k-direct-dino-p${PARTITION}-v1
LOG=${WORK}/logs/direct_dino_p${PARTITION}_gpu${GPU}.log

# The legacy all-shard watcher has no cross-process claim lock and must not run
# alongside partitioned workers.
if systemctl --user is-active --quiet \
  tracker-objaverse5k-direct-dino-shards-v1.service; then
  echo "refusing to overlap the legacy --shards all DINO watcher" >&2
  exit 97
fi

if systemctl --user is-active --quiet "${RENDER_UNIT}"; then
  echo "render worker ${PARTITION} is still active on GPU ${GPU}" >&2
  exit 98
fi
for SHARD in "${FIRST}" "${SECOND}"; do
  MARKER=${WIDE}/objaverse/shard_$(printf '%03d' "${SHARD}")/_WORKER_COMPLETE.json
  test -s "${MARKER}" || {
    echo "render shard is not complete: ${MARKER}" >&2
    exit 98
  }
done

mkdir -p "${WORK}/logs"
nvidia-smi -i "${GPU}" \
  --query-gpu=index,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits

if systemctl --user is-active --quiet "${DINO_UNIT}.service"; then
  echo "already active: ${DINO_UNIT}.service"
  exit 0
fi

systemctl --user reset-failed "${DINO_UNIT}.service" >/dev/null 2>&1 || true
systemd-run --user \
  --unit="${DINO_UNIT}" \
  --collect \
  --property=WorkingDirectory=/home/zjr/Tracker \
  --property="StandardOutput=append:${LOG}" \
  --property="StandardError=append:${LOG}" \
  /usr/bin/env \
    CUDA_VISIBLE_DEVICES="${GPU}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    ATTN_BACKEND=flash_attn \
    SPCONV_ALGO=native \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PY}" -u -m \
    pose_point_depth_mv.dataset_tools.process_objaverse5k_cache_shards \
      --stage dino \
      --work_root "${WORK}" \
      --python "${PY}" \
      --shards "${SHARDS}" \
      --expected_shards 2 \
      --device cuda \
      --dino_model dinov2_vitl14_reg \
      --ss_context_tokens 4096 \
      --log_every 10 \
      --poll_seconds 60 \
      --watch

echo "started ${DINO_UNIT}.service: GPU ${GPU}, shards ${SHARDS}"
echo "log: ${LOG}"
