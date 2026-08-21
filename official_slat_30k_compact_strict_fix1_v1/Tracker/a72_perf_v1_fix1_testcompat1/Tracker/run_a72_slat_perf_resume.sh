#!/usr/bin/env bash
# Isolated A72 Native-SLat fix1 resume/smoke launcher.  Extract this tree to a
# new /root/Tracker_perf_v1_fix1 directory; never overlay another runtime tree.
set -euo pipefail

TRACKER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${TRACKER_ROOT}"

: "${RESUME_CHECKPOINT:?set RESUME_CHECKPOINT to the immutable source checkpoint}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to a new persistent output directory}"
: "${START_STEP:?set START_STEP to the checkpoint optimizer step}"
: "${RUN_UNTIL_STEP:?set RUN_UNTIL_STEP to the smoke/stage boundary}"

MAX_STEPS=${MAX_STEPS:-20000}
PERF_PROFILE=${PERF_PROFILE:-strict}
NUM_WORKERS=${NUM_WORKERS:-2}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-2}
TORCH_NUM_INTEROP_THREADS=${TORCH_NUM_INTEROP_THREADS:-1}
LOG_EVERY=${LOG_EVERY:-1}
SAVE_EVERY=${SAVE_EVERY:-1000}

RUN=${RUN:-/data/zjr/proobjaverse_official_slat_train2000_20260813_v1}
CACHE=${CACHE:-${RUN}/cache_train2000_protocol2128_views8_v1}
SSROOT=${SSROOT:-/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1}
NATIVE_SS_REPORT=${NATIVE_SS_REPORT:-${SSROOT}/ss_eval_synthetic_dev32_fixedcfg3_count125_v3/report.json}
STOCK_SLAT_FREEZE=${STOCK_SLAT_FREEZE:-/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json}
DECODER_AUDIT=${DECODER_AUDIT:-${RUN}/decoder_audit32_protocol2128_v1/report.json}
TORCHRUN=${TORCHRUN:-torchrun}
PYTHON=${PYTHON:-python}

if (( RUN_UNTIL_STEP <= START_STEP )); then
  echo "RUN_UNTIL_STEP must exceed START_STEP" >&2
  exit 2
fi
if (( MAX_STEPS < RUN_UNTIL_STEP )); then
  echo "MAX_STEPS must be at least RUN_UNTIL_STEP" >&2
  exit 2
fi
case "${PERF_PROFILE}" in
  strict)
    CACHE_CHECK_ARGS=()
    ;;
  audited-cache)
    CACHE_CHECK_ARGS=(--skip_redundant_cache_finite_checks)
    ;;
  *)
    echo "PERF_PROFILE must be strict or audited-cache" >&2
    exit 2
    ;;
esac

for required in \
  "${CACHE}/slat_manifest.json" \
  "${CACHE}/lifting_manifest.json" \
  "${DECODER_AUDIT}" \
  "${NATIVE_SS_REPORT}" \
  "${STOCK_SLAT_FREEZE}" \
  "${RESUME_CHECKPOINT}"; do
  test -s "${required}"
done

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "OUTPUT_DIR already exists; fix1 requires a new, never-resumed smoke output:" >&2
  echo "${OUTPUT_DIR}" >&2
  exit 3
fi

export PYTHONPATH="${TRACKER_ROOT}:${TRACKER_ROOT}/ReconViaGen:${TRACKER_ROOT}/ReconViaGen/wheels/vggt"
export HF_HOME=${HF_HOME:-/root/autodl-fs/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=${TORCH_HOME:-/root/.cache/torch}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export ATTN_BACKEND=${ATTN_BACKEND:-flash_attn}
export SPCONV_ALGO=${SPCONV_ALGO:-native}
export OMP_NUM_THREADS=${TORCH_NUM_THREADS}
export MKL_NUM_THREADS=${TORCH_NUM_THREADS}
export OPENBLAS_NUM_THREADS=${A72_OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${A72_NUMEXPR_NUM_THREADS:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" -m pose_point_depth_mv.preflight_proobjaverse_official_slat_resume \
  --cache_manifest "${CACHE}/slat_manifest.json" \
  --lifting_cache_manifest "${CACHE}/lifting_manifest.json" \
  --target_decoder_audit "${DECODER_AUDIT}" \
  --native_ss_report "${NATIVE_SS_REPORT}" \
  --stock_slat_freeze "${STOCK_SLAT_FREEZE}" \
  --resume "${RESUME_CHECKPOINT}" \
  --max_steps "${MAX_STEPS}" \
  --world_size 8 \
  --grad_accum 1 \
  --allow_resume_max_steps_extension \
  --allow_resume_topology_change \
  --allow_resume_data_path_relocation \
  | tee "${OUTPUT_DIR}/preflight.log"

"${TORCHRUN}" --standalone --nnodes=1 --nproc_per_node=8 \
  -m pose_point_depth_mv.train_proobjaverse_official_slat_condition_lora \
  --architecture v2 \
  --cache_manifest "${CACHE}/slat_manifest.json" \
  --lifting_cache_manifest "${CACHE}/lifting_manifest.json" \
  --target_decoder_audit "${DECODER_AUDIT}" \
  --native_ss_report "${NATIVE_SS_REPORT}" \
  --stock_slat_freeze "${STOCK_SLAT_FREEZE}" \
  --output_dir "${OUTPUT_DIR}" \
  --pretrained Stable-X/trellis-vggt-v0-2 \
  --indices all \
  --resume "${RESUME_CHECKPOINT}" \
  --max_steps "${MAX_STEPS}" \
  --run_until_step "${RUN_UNTIL_STEP}" \
  --save_every "${SAVE_EVERY}" \
  --log_every "${LOG_EVERY}" \
  --grad_accum 1 \
  --num_workers "${NUM_WORKERS}" \
  --prefetch_factor "${PREFETCH_FACTOR}" \
  --persistent_workers \
  --pin_memory \
  --torch_num_threads "${TORCH_NUM_THREADS}" \
  --torch_num_interop_threads "${TORCH_NUM_INTEROP_THREADS}" \
  "${CACHE_CHECK_ARGS[@]}" \
  --seed 42 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --condition_channels 1024 \
  --new_lr 1e-4 \
  --lora_lr 3e-5 \
  --new_weight_decay 0.01 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --grad_clip 1.0 \
  --warmup_steps -1 \
  --warmup_ratio 0.02 \
  --ema_decay 0.9995 \
  --amp_dtype bf16 \
  --amp_init_scale 8192 \
  --p_uncond 0.1 \
  --t_logit_mean 1.0 \
  --t_logit_std 1.0 \
  --t_schedule uniform \
  --min_condition_views 1 \
  --max_condition_views 8 \
  --stock_context_views all \
  --gradient_checkpointing \
  --allow_resume_max_steps_extension \
  --allow_resume_topology_change \
  --allow_resume_data_path_relocation \
  2>&1 | tee "${OUTPUT_DIR}/train.log"

printf -v PADDED_STEP '%06d' "${RUN_UNTIL_STEP}"
if (( RUN_UNTIL_STEP == MAX_STEPS )); then
  REPORT=${OUTPUT_DIR}/report.json
else
  REPORT=${OUTPUT_DIR}/stage_report_step_${PADDED_STEP}.json
fi

"${PYTHON}" -m pose_point_depth_mv.summarize_a72_slat_perf \
  --report "${REPORT}" \
  --start_step "${START_STEP}" \
  --end_step "${RUN_UNTIL_STEP}" \
  --discard_first 2 \
  | tee "${OUTPUT_DIR}/performance_summary.json"
