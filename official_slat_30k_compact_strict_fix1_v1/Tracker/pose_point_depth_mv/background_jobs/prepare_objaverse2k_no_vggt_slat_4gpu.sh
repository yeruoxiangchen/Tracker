#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
GPUS=${OBJAVERSE2K_SLAT_GPUS:-0,5,6,7}
RUN=${OBJAVERSE2K_SLAT_RUN:-/data/zjr/objaverse2k_no_vggt_slat_20260811_v1}
SOURCE_LIFT=/data/zjr/objaverse5k_direct_dino_cache_shards_20260811_v1/merged_limited8_stage08_v1/lifting_manifest.json
AUDIT_LIFT=/data/zjr/objaverse2k_frozen_ss_audit64_20260811_v1/selection/lifting_manifest.json
RENDER_ROOT=/data/zjr/objaverse5k_single_subject_render512_20260810_v1/objaverse
SPLIT=${RUN}/split_dev64_v1
TARGET_ROOT=${RUN}/lh_slats_native_mv16_v1
TRAIN_SHARDS=${RUN}/slat_cache_train_seed42_shards_v1
DEV_SHARDS=${RUN}/slat_cache_dev64_seed424344_shards_v1
TRAIN_CACHE=${RUN}/slat_cache_train_seed42_merged_v1
DEV_CACHE=${RUN}/slat_cache_dev64_seed424344_merged_v1
TARGET_AUDIT=${RUN}/slat_target_decoder_audit_dev32_v1
SS_RUN=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1
SS_CKPT=${SS_RUN}/ss_mixed_step2000_seed42_1gpu_v1/checkpoints/step_002000.pt
SS_REPORT=${SS_RUN}/ss_eval_synthetic_dev32_fixedcfg3_count125_v3/report.json
ENCODER=/data/zjr/models/microsoft_TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16
DINO_REPO=/home/zjr/.cache/torch/hub/facebookresearch_dinov2_main
DINO_CKPT=/home/zjr/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth
MESH_DEC=/home/zjr/.cache/huggingface/hub/models--Stable-X--trellis-vggt-v0-2/snapshots/647659a5ad5fbf67e22793e7b5e2cee4b30c5d13/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors
STATE=${RUN}/logs/prepare_objaverse2k_slat_4gpu.state
EXIT_CODE=${RUN}/logs/prepare_objaverse2k_slat_4gpu.exit_code
LOCK=${RUN}/logs/prepare_objaverse2k_slat_4gpu.lock

IFS=',' read -r -a GPU_ARRAY <<<"${GPUS}"
if [ "${#GPU_ARRAY[@]}" -ne 4 ]; then
  echo "OBJAVERSE2K_SLAT_GPUS must contain exactly four GPU indices" >&2
  exit 96
fi
declare -A GPU_SEEN=()
for GPU in "${GPU_ARRAY[@]}"; do
  if [[ ! "${GPU}" =~ ^[0-9]+$ ]] || [ -n "${GPU_SEEN[${GPU}]:-}" ]; then
    echo "OBJAVERSE2K_SLAT_GPUS must contain four distinct non-negative indices" >&2
    exit 96
  fi
  GPU_SEEN[${GPU}]=1
done

RENDER_MANIFESTS=()
for INDEX in 0 1 2 3 4 5 6 7; do
  RENDER_MANIFESTS+=("${RENDER_ROOT}/shard_$(printf '%03d' "${INDEX}")/manifest.json")
done
RENDER_CSV=$(IFS=,; echo "${RENDER_MANIFESTS[*]}")

mkdir -p "${RUN}/logs"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "Objaverse2K SLat preparation is already running" >&2
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
printf 'started_at=%s state=running gpus=%s\n' \
  "$(date --iso-8601=seconds)" "${GPUS}" >"${STATE}"
rm -f "${EXIT_CODE}"

for REQUIRED in \
  "${SOURCE_LIFT}" "${AUDIT_LIFT}" "${SS_CKPT}" "${SS_REPORT}" \
  "${ENCODER}.json" "${ENCODER}.safetensors" "${DINO_REPO}/hubconf.py" \
  "${DINO_CKPT}" "${MESH_DEC}" "${RENDER_MANIFESTS[@]}"; do
  test -s "${REQUIRED}"
done

if [ ! -s "${SPLIT}/_OBJAVERSE2K_SLAT_SPLIT_COMPLETE.json" ]; then
  if [ -e "${SPLIT}" ]; then
    echo "split output exists without completion marker: ${SPLIT}" >&2
    exit 98
  fi
  "${PY}" -u -m pose_point_depth_mv.objaverse2k_slat_pipeline prepare \
    --source_lifting_manifest "${SOURCE_LIFT}" \
    --audit_lifting_manifest "${AUDIT_LIFT}" \
    --render_manifests "${RENDER_CSV}" \
    --output_dir "${SPLIT}" \
    --dev_objects 64 --seed 20260811 --num_workers 4 \
    --expected_source_objects 2199 --expected_source_samples 4137 \
      --expected_audit_objects 64
fi

if [ ! -s "${SPLIT}/native_target_preflight.json" ]; then
  "${PY}" -u -m pose_point_depth_mv.objaverse2k_slat_pipeline preflight-targets \
    --split_bundle "${SPLIT}" \
    --render_manifests "${RENDER_CSV}" \
    --min_native_sequence_target_iou 0.75
fi

run_target_workers() {
  mkdir -p "${TARGET_ROOT}" "${RUN}/logs"
  local pids=()
  for RANK in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="${GPU_ARRAY[${RANK}]}" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
    MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
    TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" -u -m pose_point_depth_mv.build_local_lh_slats \
      --render_manifests "${RENDER_CSV}" \
      --lifting_manifests "${SPLIT}/train/lifting_manifest.json,${SPLIT}/dev/lifting_manifest.json" \
      --output_dir "${TARGET_ROOT}" \
      --encoder_prefix "${ENCODER}" \
      --mesh_decoder_weights "${MESH_DEC}" \
      --dinov2_repo "${DINO_REPO}" \
      --dinov2_checkpoint "${DINO_CKPT}" \
      --dinov2_model dinov2_vitl14_reg \
      --target_contract native_objaverse_render_v1 \
      --min_native_sequence_target_iou 0.75 \
      --device cuda --image_size 518 --view_batch_size 2 \
      --min_views 8 --max_views 16 \
      --min_visible_view_fraction_mean 0.25 \
      --min_mask_support_view_fraction_mean 0.10 \
      --min_mask_support_ge2_ratio 0.50 \
      --min_mask_support_ge4_ratio 0.20 \
      --max_zero_mask_support_ratio 0.25 \
      --min_camera_corruption_mask_support_drop_mean 0.05 \
      --min_camera_corruption_mask_support_drop_median 0.02 \
      --min_camera_corruption_positive_object_rate 0.70 \
      --rank "${RANK}" --world_size 4 --resume \
      >"${RUN}/logs/lh_slats_rank_${RANK}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for PID in "${pids[@]}"; do
    if ! wait "${PID}"; then failed=1; fi
  done
  if [ "${failed}" -ne 0 ]; then
    echo "one or more local lh-slats workers failed" >&2
    return 2
  fi
}

if [ ! -s "${TARGET_ROOT}/_OBJAVERSE2K_LOCAL_LH_SLATS_COMPLETE.json" ]; then
  run_target_workers
  "${PY}" -u -m pose_point_depth_mv.objaverse2k_slat_pipeline finalize-targets \
    --split_bundle "${SPLIT}" --target_root "${TARGET_ROOT}" --world_size 4
fi

run_cache_workers() {
  local SPLIT_NAME=$1
  local SEEDS=$2
  local SHARD_ROOT=$3
  mkdir -p "${SHARD_ROOT}"
  local pids=()
  for WORKER in 0 1 2 3; do
    local WORKER_PAD
    WORKER_PAD=$(printf '%03d' "${WORKER}")
    local WORKER_OUT=${SHARD_ROOT}/worker_${WORKER_PAD}
    local INDICES
    INDICES=$(tr -d '[:space:]' <"${SPLIT}/${SPLIT_NAME}/worker_${WORKER_PAD}_indices.txt")
    if [ -s "${WORKER_OUT}/manifest.json" ]; then
      continue
    fi
    CUDA_VISIBLE_DEVICES="${GPU_ARRAY[${WORKER}]}" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
    MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
    TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" -u -m pose_point_depth_mv.build_direct_slat_cache_no_vggt \
      --lifting_manifest "${SPLIT}/${SPLIT_NAME}/lifting_manifest.json" \
      --slat_root "${TARGET_ROOT}" \
      --flow_checkpoint "${SS_CKPT}" \
      --native_ss_report "${SS_REPORT}" \
      --condition_arch native_ss_genrecon_v2 \
      --output_dir "${WORKER_OUT}" \
      --indices "${INDICES}" \
      --ss_seeds "${SEEDS}" \
      --expected_ss_step 2000 --amp_dtype bf16 \
      --require_all_objects --resume \
      >"${RUN}/logs/cache_${SPLIT_NAME}_worker_${WORKER_PAD}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for PID in "${pids[@]}"; do
    if ! wait "${PID}"; then failed=1; fi
  done
  if [ "${failed}" -ne 0 ]; then
    echo "one or more ${SPLIT_NAME} cache workers failed" >&2
    return 2
  fi
}

merge_cache() {
  local SPLIT_NAME=$1
  local SEEDS=$2
  local SHARD_ROOT=$3
  local MERGED=$4
  if [ -s "${MERGED}/_OBJAVERSE2K_SLAT_CACHE_MERGE_COMPLETE.json" ]; then
    return 0
  fi
  if [ -e "${MERGED}" ]; then
    echo "merged cache exists without completion marker: ${MERGED}" >&2
    return 98
  fi
  "${PY}" -u -m pose_point_depth_mv.objaverse2k_slat_pipeline merge-cache \
    --split_bundle "${SPLIT}" --split "${SPLIT_NAME}" \
    --input_dirs "${SHARD_ROOT}/worker_000,${SHARD_ROOT}/worker_001,${SHARD_ROOT}/worker_002,${SHARD_ROOT}/worker_003" \
    --output_dir "${MERGED}" --ss_seeds "${SEEDS}"
}

run_cache_workers train 42 "${TRAIN_SHARDS}"
merge_cache train 42 "${TRAIN_SHARDS}" "${TRAIN_CACHE}"
run_cache_workers dev 42,43,44 "${DEV_SHARDS}"
merge_cache dev 42,43,44 "${DEV_SHARDS}" "${DEV_CACHE}"

if [ ! -s "${TARGET_AUDIT}/report.json" ]; then
  if [ -e "${TARGET_AUDIT}" ]; then
    AUDIT_RESUME=(--resume)
  else
    AUDIT_RESUME=()
  fi
  CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PY}" -u -m pose_point_depth_mv.audit_direct_slat_targets \
    --cache_manifest "${DEV_CACHE}/manifest.json" \
    --output_dir "${TARGET_AUDIT}" \
    --pretrained Stable-X/trellis-vggt-v0-2 \
    --max_objects 32 --surface_samples 20000 \
    --max_chamfer_l1 0.10 --min_mesh_success_rate 1.0 \
    --decision_profile strict "${AUDIT_RESUME[@]}" \
    >"${RUN}/logs/slat_target_decoder_audit_dev32.log" 2>&1
fi

"${PY}" - "${TRAIN_CACHE}/manifest.json" "${DEV_CACHE}/manifest.json" "${TARGET_AUDIT}/report.json" <<'PY'
import json, sys
train, dev, audit = [json.load(open(path, encoding="utf-8")) for path in sys.argv[1:]]
assert train["materialized"] is True and train["object_count"] == 2135
assert dev["materialized"] is True and dev["object_count"] == 64
assert set(row["object_uid"] for row in train["samples"]).isdisjoint(
    row["object_uid"] for row in dev["samples"]
)
assert audit["passed"] is True and audit["summary"]["object_count"] == 32
print({"train_objects": 2135, "dev_objects": 64, "target_audit": "passed"})
PY
