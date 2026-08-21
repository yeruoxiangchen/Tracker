#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker
ROOT=${OUTPUT_ROOT:-/data/zjr/dorabench_dora299_trellis_s_m_seed42_trellis40_input0_9_19_29_8gpu_20260821_v1}
mkdir -p "${ROOT}"

echo "===== TRELLIS-S then TRELLIS-M frozen Dora299 queue ====="
echo "S=stochastic four-image mode; M=multidiffusion four-image mode"

echo "===== wait for all eight CUDA devices ====="
while true; do
  gpu_count=$(
    /home/zjr/anaconda3/envs/reconviagen/bin/python -c \
      'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)' \
      2>/dev/null || printf '0\n'
  )
  if nvidia-smi >/dev/null 2>&1 && [[ "${gpu_count}" == "8" ]]; then
    echo "CUDA_RECOVERED=$(date -u -Is) devices=8"
    break
  fi
  echo "CUDA_WAIT=$(date -u -Is) visible_devices=${gpu_count}"
  sleep 15
done

echo "===== exact-contract one-object TRELLIS-S smoke ====="
SMOKE_KEY=$(
  /home/zjr/anaconda3/envs/reconviagen/bin/python -c \
    "import json; p=json.load(open('/data/zjr/dorabench_dora299_strict_reconviagen_seed42_trellis40_input0_9_19_29_8gpu_20260821_v1/protocol/dora299_current_valid_subset.json')); r=p['objects'][0]; print(r['category']+':'+r['uid'])"
)
export PYTHONPATH="${PWD}:${PWD}/ReconViaGen"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 ATTN_BACKEND=flash_attn SPCONV_ALGO=native
CUDA_VISIBLE_DEVICES=0 /home/zjr/anaconda3/envs/reconviagen/bin/python -u -m \
  pose_point_depth_mv.evaluate_dorabench299_trellis_baselines \
  inference-worker \
  --subset_manifest /data/zjr/dorabench_dora299_strict_reconviagen_seed42_trellis40_input0_9_19_29_8gpu_20260821_v1/protocol/dora299_current_valid_subset.json \
  --baseline trellis_s --output_root "${ROOT}/trellis_s" --device cuda \
  --object "${SMOKE_KEY}" --pretrained microsoft/TRELLIS-image-large \
  --model_revision 25e0d31ffbebe4b5a97464dd851910efc3002d96 \
  --seed 42 --ss_steps 30 --ss_cfg 7.5 --slat_steps 12 --slat_cfg 3.0
echo "TRELLIS_S_EXACT_SMOKE=PASS object=${SMOKE_KEY}"

BASELINE=trellis_s OUTPUT_ROOT="${ROOT}/trellis_s" \
  bash pose_point_depth_mv/background_jobs/run_dorabench299_trellis_baseline_8gpu.sh

BASELINE=trellis_m OUTPUT_ROOT="${ROOT}/trellis_m" \
  bash pose_point_depth_mv/background_jobs/run_dorabench299_trellis_baseline_8gpu.sh

echo "DORA299 TRELLIS-S AND TRELLIS-M COMPLETE: ${ROOT}"
