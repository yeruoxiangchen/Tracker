#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
GPU=${MIXED_NO_VGGT_SLAT_CACHE_GPU:-6}
RUN=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1
SS_CKPT=${RUN}/ss_mixed_step2000_seed42_1gpu_v1/checkpoints/step_002000.pt
SS_REPORT=${RUN}/ss_eval_synthetic_dev32_fixedcfg3_count125_v3/report.json
SYN_LIFT=/data/zjr/native_ss_no_vggt_mixed1k_20260807_v1/lifting_train868_dino_only_v1/lifting_manifest.json
REAL_LIFT=${RUN}/lifting_real376_dino_only_v1/lifting_manifest.json
MIXED_LIFT=${RUN}/manifests/mixed_ss_lifting_synth868_real376_v1.json
SYN_ROOT=/data/zjr/native3d_condition_reviewed1k_inputs_20260730_v3/lh_slats_train_val_v2
REAL_ROOT=/data/zjr/native_v2_real500_domain_adapt_20260806_v2/cache_train_real_runtime_o_v2/lh_slats
SYN_CACHE=${RUN}/slat_cache_synthetic868_finalss_seed42_v1
REAL_CACHE=${RUN}/slat_cache_real376_finalss_seed42_v1
MIXED_SLAT=${RUN}/manifests/mixed_slat_synth868_real376_v1.json
VAL_LIFT=/data/zjr/native_ss_no_vggt_mixed1k_20260807_v1/lifting_val64_dino_only_v1/lifting_manifest.json
VAL_TARGET=${RUN}/slat_val32_target_only_v1
AUDIT=${RUN}/slat_target_decoder_audit32_v1
SLAT_PARENT_RUN=/data/zjr/native_v2_real500_domain_adapt_20260806_v2/slat_v2_real_step1000_seed42_2gpu_v2
SLAT_PARENT=${SLAT_PARENT_RUN}/checkpoints/last.pt
SLAT_PARENT_REPORT=${SLAT_PARENT_RUN}/report.json
SLAT_CONTRACT=${RUN}/contracts/slat_real_full_ema_v1.json
STATE=${RUN}/logs/M7_prepare_slat_data.state
EXIT_CODE=${RUN}/logs/M7_prepare_slat_data.exit_code
LOCK=${RUN}/logs/M7_prepare_slat_data.lock

mkdir -p "${RUN}/logs" "${RUN}/manifests" "${RUN}/contracts"
exec 9>"${LOCK}"
if ! flock -n 9; then echo "M7 duplicate job" >&2; exit 99; fi
finish() {
  RC=$?
  trap - EXIT
  printf '%s\n' "${RC}" > "${EXIT_CODE}"
  printf 'finished_at=%s rc=%s\n' "$(date --iso-8601=seconds)" "${RC}" > "${STATE}"
  exit "${RC}"
}
trap finish EXIT
printf 'started_at=%s state=running gpu=%s\n' "$(date --iso-8601=seconds)" "${GPU}" > "${STATE}"
rm -f "${EXIT_CODE}"

for REQUIRED in "${SS_CKPT}" "${SS_REPORT}" "${SYN_LIFT}" "${REAL_LIFT}" "${MIXED_LIFT}" "${SLAT_PARENT}" "${SLAT_PARENT_REPORT}"; do
  test -s "${REQUIRED}"
done
for REQUIRED in "${SYN_ROOT}" "${REAL_ROOT}"; do test -d "${REQUIRED}"; done

build_cache() {
  local lift=$1
  local slat_root=$2
  local output=$3
  if [ -s "${output}/manifest.json" ]; then return 0; fi
  local resume=()
  if [ -e "${output}" ]; then resume=(--resume); fi
  CUDA_VISIBLE_DEVICES="${GPU}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PY}" -u -m pose_point_depth_mv.build_direct_slat_cache_no_vggt \
    --lifting_manifest "${lift}" \
    --slat_root "${slat_root}" \
    --flow_checkpoint "${SS_CKPT}" \
    --native_ss_report "${SS_REPORT}" \
    --condition_arch native_ss_genrecon_v2 \
    --output_dir "${output}" \
    --ss_seeds 42 \
    --expected_ss_step 2000 \
    --amp_dtype bf16 \
    --require_all_objects \
    "${resume[@]}"
}

build_cache "${SYN_LIFT}" "${SYN_ROOT}" "${SYN_CACHE}"
build_cache "${REAL_LIFT}" "${REAL_ROOT}" "${REAL_CACHE}"

"${PY}" -u -m pose_point_depth_mv.dataset_tools.build_mixed_no_vggt_manifest \
  slat \
  --synthetic_manifest "${SYN_CACHE}/manifest.json" \
  --real_manifest "${REAL_CACHE}/manifest.json" \
  --lifting_manifest "${MIXED_LIFT}" \
  --output "${MIXED_SLAT}"

if [ ! -s "${VAL_TARGET}/manifest.json" ]; then
  RESUME=()
  if [ -e "${VAL_TARGET}" ]; then RESUME=(--resume); fi
  "${PY}" -u -m pose_point_depth_mv.build_direct_slat_cache_no_vggt \
    --lifting_manifest "${VAL_LIFT}" \
    --slat_root "${SYN_ROOT}" \
    --flow_checkpoint "${SS_CKPT}" \
    --native_ss_report "${SS_REPORT}" \
    --condition_arch native_ss_genrecon_v2 \
    --output_dir "${VAL_TARGET}" \
    --indices 32-63 \
    --ss_seeds 42,43,44 \
    --expected_ss_step 2000 \
    --amp_dtype bf16 \
    --require_all_objects \
    --target_only \
    "${RESUME[@]}"
fi

if [ ! -s "${AUDIT}/report.json" ]; then
  CUDA_VISIBLE_DEVICES="${GPU}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  "${PY}" -u -m pose_point_depth_mv.audit_direct_slat_targets \
    --cache_manifest "${VAL_TARGET}/manifest.json" \
    --output_dir "${AUDIT}" \
    --pretrained Stable-X/trellis-vggt-v0-2 \
    --max_objects 32 \
    --surface_samples 20000 \
    --max_chamfer_l1 0.10 \
    --min_mesh_success_rate 1.0 \
    --decision_profile strict
fi

"${PY}" -u -m pose_point_depth_mv.dataset_tools.build_real_full_no_vggt_migration_contract \
  --stage slat \
  --parent_checkpoint "${SLAT_PARENT}" \
  --parent_report "${SLAT_PARENT_REPORT}" \
  --output "${SLAT_CONTRACT}" \
  --min_real_objects 350

"${PY}" - "${MIXED_SLAT}" "${MIXED_LIFT}" "${AUDIT}/report.json" <<'PY'
from pose_point_depth_mv.mixed_no_vggt_data import MixedNativeConditionSLatDataset
import json, sys

dataset = MixedNativeConditionSLatDataset(sys.argv[1], sys.argv[2])
audit = json.load(open(sys.argv[3], encoding="utf-8"))
assert audit["passed"] is True and audit["summary"]["object_count"] == 32
object_count = len({row["object_uid"] for row in dataset.rows})
domain_objects = {
    name: len({row["object_uid"] for row in value.rows})
    for name, value in dataset.domain_datasets.items()
}
assert object_count == 1244
assert domain_objects == {"synthetic": 868, "real": 376}
print({"passed": True, "samples": len(dataset), "objects": object_count,
       "domain_objects": domain_objects})
PY
