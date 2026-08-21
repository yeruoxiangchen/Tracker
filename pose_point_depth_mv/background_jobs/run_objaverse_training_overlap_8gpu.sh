#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
ROOT=${TRAIN_OVERLAP_ROOT:-/data/zjr/objaverse_training_overlap_20260813_v1}
GPU_CSV=${TRAIN_OVERLAP_GPUS:-0,1,2,3,4,5,6,7}
IFS=, read -r -a GPUS <<<"${GPU_CSV}"
if [ "${#GPUS[@]}" -ne 8 ]; then
  echo "TRAIN_OVERLAP_GPUS must contain exactly 8 comma-separated GPU IDs" >&2
  exit 96
fi

PRETRAINED=Stable-X/trellis-vggt-v0-2
MIXED=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1
OBJ2K=/data/zjr/objaverse2k_no_vggt_slat_20260811_v1
STOCK=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json
SS=${MIXED}/ss_mixed_step2000_seed42_1gpu_v1/checkpoints/step_002000.pt
M8=${MIXED}/slat_mixed_step2000_seed42_2gpu_v1/checkpoints/step_002000.pt
OBJ2K_SLAT=${OBJ2K}/slat_objaverse2135_step2000_seed42_8gpu_bs8_v1/checkpoints/step_002000.pt

OBJ2K_LIFT=${OBJ2K}/split_dev64_v1/train/lifting_manifest.json
OBJ2K_CACHE=${OBJ2K}/slat_cache_train_seed42_merged_v1/manifest.json
MIXED_LIFT=/data/zjr/native_ss_no_vggt_mixed1k_20260807_v1/lifting_train868_dino_only_v1/lifting_manifest.json
MIXED_CACHE=${MIXED}/slat_cache_synthetic868_finalss_seed42_v1/manifest.json
MIXED_LIFT_META=${MIXED}/manifests/mixed_ss_lifting_synth868_real376_v1.json
MIXED_SLAT_META=${MIXED}/manifests/mixed_slat_synth868_real376_v1.json

mkdir -p "${ROOT}/logs"
for REQUIRED in \
  "${PY}" "${STOCK}" "${SS}" "${M8}" "${OBJ2K_SLAT}" \
  "${OBJ2K_LIFT}" "${OBJ2K_CACHE}" "${MIXED_LIFT}" "${MIXED_CACHE}" \
  "${MIXED_LIFT_META}" "${MIXED_SLAT_META}"; do
  test -s "${REQUIRED}"
done

prepare_group() {
  local NAME=$1
  local SCOPE=$2
  local LIFT=$3
  local CACHE=$4
  local GROUP=${ROOT}/${NAME}
  local EXTRA=()
  if [ "${SCOPE}" = mixed_objaverse_train ]; then
    EXTRA=(
      --mixed_lifting_meta_manifest "${MIXED_LIFT_META}"
      --mixed_slat_meta_manifest "${MIXED_SLAT_META}"
    )
  fi
  "${PY}" -u -m pose_point_depth_mv.select_objaverse_training_overlap_subset \
    --source_scope "${SCOPE}" \
    --lifting_manifest "${LIFT}" \
    --slat_manifest "${CACHE}" \
    --output_selection "${GROUP}/selection.json" \
    --output_lifting_subset "${GROUP}/lifting_subset.json" \
    --count 16 --seed 42 --resume "${EXTRA[@]}"
  "${PY}" -u -m pose_point_depth_mv.prepare_objaverse16_no_vggt_model_inputs \
    --selection_manifest "${GROUP}/selection.json" \
    --lifting_manifest "${GROUP}/lifting_subset.json" \
    --output_dir "${GROUP}/model_inputs" --resume
}

prepare_group objaverse2k_train16 objaverse2k_train "${OBJ2K_LIFT}" "${OBJ2K_CACHE}"
prepare_group mixed_objaverse_train16 mixed_objaverse_train "${MIXED_LIFT}" "${MIXED_CACHE}"

run_worker() {
  local GPU=$1
  local NAME=$2
  local MODEL_LABEL=$3
  local SLAT=$4
  local WORKER=$5
  local GROUP=${ROOT}/${NAME}
  local NATIVE=${GROUP}/native_worker_${WORKER}
  local RECON=${GROUP}/recon_worker_${WORKER}
  local LOG=${ROOT}/logs/${NAME}_worker_${WORKER}.log
  {
    CUDA_VISIBLE_DEVICES="${GPU}" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
    MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
    TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" -u -m pose_point_depth_mv.infer_objaverse_training_overlap_native \
      --model_label "${MODEL_LABEL}" \
      --worker_index "${WORKER}" --num_workers 4 \
      --model_input_manifest "${GROUP}/model_inputs/model_input_manifest.json" \
      --native_ss_checkpoint "${SS}" \
      --native_slat_checkpoint "${SLAT}" \
      --stock_slat_freeze "${STOCK}" \
      --output_dir "${NATIVE}" \
      --pretrained "${PRETRAINED}" \
      --seeds 42 --weights ema --device cuda --amp_dtype bf16

    CUDA_VISIBLE_DEVICES="${GPU}" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
    MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
    TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" -u -m pose_point_depth_mv.infer_objaverse16_reconviagen \
      --selection_manifest "${GROUP}/selection.json" \
      --source_lifting_manifest "${GROUP}/lifting_subset.json" \
      --output_dir "${RECON}" \
      --pretrained "${PRETRAINED}" \
      --seeds 42 --device cuda --low_vram --multiimage_algo multidiffusion \
      --worker_index "${WORKER}" --num_workers 4 --resume
  } >"${LOG}" 2>&1
}

PIDS=()
for WORKER in 0 1 2 3; do
  run_worker "${GPUS[${WORKER}]}" objaverse2k_train16 objaverse2k_slat \
    "${OBJ2K_SLAT}" "${WORKER}" &
  PIDS+=("$!")
done
for WORKER in 0 1 2 3; do
  GPU_INDEX=$((WORKER + 4))
  run_worker "${GPUS[${GPU_INDEX}]}" mixed_objaverse_train16 m8 \
    "${M8}" "${WORKER}" &
  PIDS+=("$!")
done

FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then FAILED=1; fi
done
if [ "${FAILED}" -ne 0 ]; then
  echo "At least one inference worker failed; inspect ${ROOT}/logs" >&2
  exit 2
fi

evaluate_group() {
  local NAME=$1
  local MODEL_LABEL=$2
  local GROUP=${ROOT}/${NAME}
  local ARGS=()
  for WORKER in 0 1 2 3; do
    ARGS+=(
      --native_manifest "${GROUP}/native_worker_${WORKER}/inference_manifest.json"
      --reconviagen_manifest "${GROUP}/recon_worker_${WORKER}/inference_manifest.json"
    )
  done
  "${PY}" -u -m pose_point_depth_mv.evaluate_objaverse_training_overlap \
    --selection_manifest "${GROUP}/selection.json" \
    --native_label "${MODEL_LABEL}" "${ARGS[@]}" \
    --output_dir "${GROUP}/evaluation_20k" \
    --surface_samples 20000 --fscore_thresholds 0.01,0.02,0.05 --resume
}

evaluate_group objaverse2k_train16 objaverse2k_slat
evaluate_group mixed_objaverse_train16 m8

echo "Objaverse2K report: ${ROOT}/objaverse2k_train16/evaluation_20k/report.json"
echo "M8 Objaverse-only report: ${ROOT}/mixed_objaverse_train16/evaluation_20k/report.json"
