#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
TORCHRUN=${TORCHRUN:-/home/zjr/anaconda3/envs/reconviagen/bin/torchrun}
TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3,4,5,6,7}

: "${CACHE_MANIFEST:?set CACHE_MANIFEST}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="$PWD:$PWD/ReconViaGen:$PWD/ReconViaGen/wheels/vggt"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export ATTN_BACKEND=${ATTN_BACKEND:-flash_attn}
export SPCONV_ALGO=${SPCONV_ALGO:-native}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

IFS=, read -r -a GPU_ARRAY <<<"${TRAIN_GPUS}"
WORLD_SIZE=${#GPU_ARRAY[@]}
if (( WORLD_SIZE != 8 )); then
  echo "ERROR: formal contract requires exactly 8 visible GPUs" >&2
  exit 90
fi

ARGS=(
  --cache_manifest "${CACHE_MANIFEST}"
  --output_dir "${OUTPUT_DIR}"
  --max_steps 2000
  --save_every 500
  --log_every 10
  --grad_accum 1
  --num_workers 2
  --seed 42
  --lora_rank 8
  --lora_alpha 16
  --condition_channels 1024
  --new_lr 5e-5
  --lora_lr 1e-5
  --new_weight_decay 0.01
  --grad_clip 1.0
  --warmup_ratio 0.02
  --ema_decay 0.9995
  --p_uncond 0.1
  --t_logit_mean 1.0
  --t_logit_std 1.0
  --min_condition_views 1
  --max_condition_views 8
  --amp_dtype bf16
  --gradient_checkpointing
)
# The formal 2K trajectory has four immutable science anchors. ``last.pt`` is
# only a convenience pointer and must never be the sole surviving artifact.
REQUIRED_CHECKPOINT_STEPS=(500 1000 1500 2000)
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  ARGS+=(--resume "${RESUME_CHECKPOINT}")
else
  if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "ERROR: fresh output already exists: ${OUTPUT_DIR}" >&2
    exit 91
  fi
fi

CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
  "${TORCHRUN}" --standalone --nproc_per_node=8 \
    -m official_ss_with_vggt_perf_v1.train "${ARGS[@]}"

for STEP in "${REQUIRED_CHECKPOINT_STEPS[@]}"; do
  PAD=$(printf '%06d' "${STEP}")
  CHECKPOINT="${OUTPUT_DIR}/checkpoints/step_${PAD}.pt"
  if [[ ! -s "${CHECKPOINT}" ]]; then
    echo "ERROR: formal retained SS checkpoint is missing: ${CHECKPOINT}" >&2
    exit 92
  fi
done
test -s "${OUTPUT_DIR}/checkpoints/last.pt"
echo "SS retained checkpoints PASS: 500,1000,1500,2000"
