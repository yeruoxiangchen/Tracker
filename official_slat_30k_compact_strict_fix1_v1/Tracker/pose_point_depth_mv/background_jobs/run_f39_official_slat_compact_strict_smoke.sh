#!/usr/bin/env bash
set -euo pipefail

# Functional CUDA/DDP smoke for the isolated compact-v2 + strict-fix1 tree.
# This script never selects scientific artifacts implicitly: the three frozen
# evidence paths must be supplied by the operator and are checked before launch.

source /root/miniconda3/etc/profile.d/conda.sh
conda activate reconviagen-bw

CODE_ROOT=${CODE_ROOT:-/root/Tracker_30k_compact_strict_fix1_v1}
COMPACT_ROOT=${COMPACT_ROOT:-/root/autodl-tmp/proobjaverse_official_slat_compact_v2_pilot256_8gpu_v1}
CACHE_MANIFEST=${CACHE_MANIFEST:-${COMPACT_ROOT}/slat_manifest.json}
LIFTING_MANIFEST=${LIFTING_MANIFEST:-${COMPACT_ROOT}/lifting_manifest.json}
: "${DECODER_AUDIT:?export DECODER_AUDIT=/absolute/path/to/frozen/report.json}"
: "${NATIVE_SS_REPORT:?export NATIVE_SS_REPORT=/absolute/path/to/frozen/report.json}"
: "${STOCK_SLAT_FREEZE:?export STOCK_SLAT_FREEZE=/absolute/path/to/frozen/freeze.json}"

WORLD_SIZE=${SMOKE_WORLD_SIZE:-1}
RUN_UNTIL_STEP=${RUN_UNTIL_STEP:-2}
MAX_STEPS=${MAX_STEPS:-5}
case "${WORLD_SIZE}" in
  1)
    DEFAULT_GPUS=0
    GRAD_ACCUM=8
    ;;
  2)
    DEFAULT_GPUS=0,1
    GRAD_ACCUM=4
    ;;
  8)
    DEFAULT_GPUS=0,1,2,3,4,5,6,7
    GRAD_ACCUM=1
    ;;
  *)
    echo "ERROR: SMOKE_WORLD_SIZE must be 1, 2, or 8" >&2
    exit 80
    ;;
esac
VISIBLE_GPUS=${VISIBLE_GPUS:-${DEFAULT_GPUS}}
OUTPUT_DIR=${OUTPUT_DIR:-/root/autodl-tmp/official_slat_compact_strict_fix1_smoke_w${WORLD_SIZE}_v1}
LOG=${LOG:-${OUTPUT_DIR}.log}

PY=/root/miniconda3/envs/reconviagen-bw/bin/python
TORCHRUN=/root/miniconda3/envs/reconviagen-bw/bin/torchrun

test -d "${CODE_ROOT}"
test -s "${CACHE_MANIFEST}"
test -s "${LIFTING_MANIFEST}"
test -s "${DECODER_AUDIT}"
test -s "${NATIVE_SS_REPORT}"
test -s "${STOCK_SLAT_FREEZE}"
test ! -e "${OUTPUT_DIR}"
test ! -e "${LOG}"

export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUMM_CUDA_ARCH_LIST=12.0
export SPCONV_ALGO=native
export ATTN_BACKEND=flash_attn
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_HOME=${HF_HOME:-/root/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=${TORCH_HOME:-/root/.cache/torch}
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${CODE_ROOT}:${CODE_ROOT}/ReconViaGen:${CODE_ROOT}/ReconViaGen/wheels/vggt"

cd "${CODE_ROOT}"

echo "============================================================"
echo "F39 official 30K compact-v2 + strict-fix1 CUDA smoke"
echo "world_size=${WORLD_SIZE} grad_accum=${GRAD_ACCUM} global_batch=$((WORLD_SIZE * GRAD_ACCUM))"
echo "visible_gpus=${VISIBLE_GPUS}"
echo "cache=${CACHE_MANIFEST}"
echo "lifting=${LIFTING_MANIFEST}"
echo "output=${OUTPUT_DIR}"
echo "============================================================"

set +e
CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}" \
  "${TORCHRUN}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${WORLD_SIZE}" \
  -m pose_point_depth_mv.train_proobjaverse_official_slat_condition_lora \
  --architecture v2 \
  --cache_manifest "${CACHE_MANIFEST}" \
  --lifting_cache_manifest "${LIFTING_MANIFEST}" \
  --target_decoder_audit "${DECODER_AUDIT}" \
  --native_ss_report "${NATIVE_SS_REPORT}" \
  --stock_slat_freeze "${STOCK_SLAT_FREEZE}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --run_until_step "${RUN_UNTIL_STEP}" \
  --save_every 1 \
  --log_every 1 \
  --grad_accum "${GRAD_ACCUM}" \
  --num_workers 2 \
  --prefetch_factor 2 \
  --persistent_workers \
  --pin_memory \
  --torch_num_threads 2 \
  --torch_num_interop_threads 1 \
  --seed 42 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --condition_channels 1024 \
  --view_fusion_hidden_dim 64 \
  --geometry_logit_scale_init 1.0 \
  --new_lr 1e-4 \
  --lora_lr 3e-5 \
  --new_weight_decay 0.01 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --grad_clip 1.0 \
  --warmup_ratio 0.02 \
  --ema_decay 0.9995 \
  --amp_dtype bf16 \
  --p_uncond 0.1 \
  --t_schedule uniform \
  --min_condition_views 1 \
  --max_condition_views 8 \
  --stock_context_views all \
  --gradient_checkpointing \
  2>&1 | tee "${LOG}"
RC=${PIPESTATUS[0]}
set -e

echo "CUDA_SMOKE_RC=${RC}"
echo "log=${LOG}"
exit "${RC}"
