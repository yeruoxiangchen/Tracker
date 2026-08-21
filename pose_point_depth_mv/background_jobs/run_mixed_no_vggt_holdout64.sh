#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
PIXAL_PY=/home/zjr/anaconda3/envs/pixal3d/bin/python
GPU=${MIXED_NO_VGGT_HOLDOUT_GPU:-6}
JOB_TAG=${M11_JOB_TAG:-M11_holdout64}
ALLOW_ALIGNMENT_QUALITY_WARNINGS=${M11_ALLOW_ALIGNMENT_QUALITY_WARNINGS:-0}
ROOT=/data/zjr/omni_real_video500_download_20260804_v2
ADAPT=/data/zjr/native_v2_real500_domain_adapt_20260806_v2
RUN=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1
SPLIT=${ROOT}/D6_novel500_dev64_holdout64_v3_pilotfree_eval/holdout.json
DEPLOY=${RUN}/contracts/final_no_vggt_deployment_benchmark32_v1.json
CONSUMED=${RUN}/contracts/M11_HOLDOUT64_CONSUMPTION_STARTED.json
INV=${ROOT}/M11A_holdout64_extraction_inventory_v1.json
RAW=${ROOT}/M11B_holdout64_raw_cache_v1
RUNTIME=${ROOT}/M11C_holdout64_runtime_o_v1
ALIGN=${ROOT}/M11D_holdout64_scan_to_colmap_w_v1
ADJ=${ALIGN}/alignment_adjudicated_v1.json
LABEL=${ROOT}/M11E_holdout64_mesh_o_labels_v1
FULL_MODEL=${ROOT}/M11F_holdout64_native_v2_model_inputs_v1
DINO_MODEL=${ROOT}/M11G_holdout64_dino_only_model_inputs_v1
NO_VGGT=${ROOT}/M11H_holdout64_native_no_vggt_mixed1244_seed42_v1
REAL_FULL=${ROOT}/M11I_holdout64_native_v2_realadapt_step1000_seed42_v1
SYNTH_FULL=${ROOT}/M11J_holdout64_native_v2_parent_seed42_v1
RECON=${ROOT}/M11K_holdout64_reconviagen_original_seed42_v1
PIXAL=${ROOT}/M11L_holdout64_pixal3d_official_seed42_v1
EVAL=${ROOT}/M11M_holdout64_fiveway_no_vggt_mixed1244_seed42_v1
ALIGNMENT_WARNING_CONTRACT=${RUN}/contracts/M11_ALIGNMENT_QUALITY_WARNING_ALL64_v1.json
SS_FINAL=${RUN}/ss_mixed_step2000_seed42_1gpu_v1/checkpoints/step_002000.pt
SLAT_FINAL=${RUN}/slat_mixed_step2000_seed42_2gpu_v1/checkpoints/step_002000.pt
SS_CONTRACT=${RUN}/contracts/ss_real_full_ema_v1.json
SLAT_CONTRACT=${RUN}/contracts/slat_real_full_ema_v1.json
SS_REAL=${ADAPT}/ss_real_step1000_seed42_2gpu_v2/checkpoints/step_001000.pt
SLAT_REAL=${ADAPT}/slat_v2_real_step1000_seed42_2gpu_v2/checkpoints/step_001000.pt
SS_SYNTH=/data/zjr/native3d_condition_ss_mixed1k_20260801_v1/ss868_sourceholdout_seed42_v1/checkpoints/step_002000.pt
SLAT_SYNTH=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/train868_step2000_seed42_2gpu_v1/checkpoints/step_002000.pt
FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json
PIXAL_MODEL=/home/zjr/.cache/huggingface/hub/models--TencentARC--Pixal3D/snapshots/0b31f9160aa400719af409098bff7936a932f726
NAF_ROOT=/data/zjr/models/valeoai_NAF_37f2dfc180f2de53d98bd601109c0da0dd6b0f43
STATE=${RUN}/logs/${JOB_TAG}.state
EXIT_CODE=${RUN}/logs/${JOB_TAG}.exit_code
LOCK=${RUN}/logs/M11_holdout64.lock

mkdir -p "${RUN}/logs" "${RUN}/contracts"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "M11 refused: another final Holdout64 job holds ${LOCK}" >&2
  exit 99
fi
finish() {
  RC=$?
  trap - EXIT
  printf '%s\n' "${RC}" > "${EXIT_CODE}"
  printf 'finished_at=%s rc=%s\n' "$(date --iso-8601=seconds)" "${RC}" > "${STATE}"
  exit "${RC}"
}
trap finish EXIT
printf 'started_at=%s state=running gpu=%s\n' \
  "$(date --iso-8601=seconds)" "${GPU}" > "${STATE}"
rm -f "${EXIT_CODE}"

for REQUIRED in "${SPLIT}" "${DEPLOY}" "${SS_FINAL}" "${SLAT_FINAL}" \
                "${SS_CONTRACT}" "${SLAT_CONTRACT}" "${SS_REAL}" \
                "${SLAT_REAL}" "${SS_SYNTH}" "${SLAT_SYNTH}" "${FREEZE}" \
                "${PIXAL_MODEL}/pipeline.json" "${NAF_ROOT}/repo/hubconf.py" \
                "${NAF_ROOT}/naf_release.pth" "${NAF_ROOT}/source_manifest.sha256"; do
  test -s "${REQUIRED}"
done

# The old Full-only S8 and this no-VGGT M11 must not both claim first consumption.
if [ ! -s "${CONSUMED}" ] && \
   { [ -e "${ROOT}/S8A_holdout64_extraction_inventory_v1.json" ] || \
     [ -e "${ROOT}/S8K_holdout64_fourway_realadapt_step1000_seed42_v1" ]; }; then
  echo "M11 blocked: Holdout64 was already touched by the older S8 route" >&2
  exit 95
fi

"${PY}" - "${DEPLOY}" "${CONSUMED}" <<'PY'
import json
import sys
from pathlib import Path

from pose_point_depth_mv.dataset_tools.freeze_mixed_no_vggt_deployment import (
    FORMAT,
)
from pose_point_depth_mv.omni_real_benchmark_common import atomic_json, sha256_file

deployment_path = Path(sys.argv[1]).resolve()
marker_path = Path(sys.argv[2]).resolve()
deployment = json.load(open(deployment_path, encoding="utf-8"))
assert deployment["format"] == FORMAT
assert deployment["passed"] is True and deployment["holdout64_unlocked"] is True
assert deployment["holdout64_consumed"] is False
for stage in ("ss", "slat"):
    frozen = deployment["binding"][stage]
    checkpoint = Path(frozen["checkpoint"]).resolve()
    assert checkpoint.is_file()
    assert sha256_file(checkpoint) == frozen["checkpoint_sha256"]
payload = {
    "format": "pose_point_depth_mv.mixed_no_vggt_holdout64_consumption.v1",
    "deployment": str(deployment_path),
    "deployment_sha256": sha256_file(deployment_path),
    "holdout64_consumed": True,
    "selection_after_this_marker_forbidden": True,
    "passed": True,
}
if marker_path.exists():
    assert json.load(open(marker_path, encoding="utf-8")) == payload
else:
    atomic_json(marker_path, payload)
print(payload)
PY

if [ ! -s "${INV}" ]; then
  "${PY}" -m pose_point_depth_mv.dataset_tools.freeze_omni_real_raw_split \
    inventory --source_split "${SPLIT}" --output "${INV}"
fi
if [ ! -s "${RAW}/raw_cache_report.json" ]; then
  "${PY}" -u -m pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache \
    extract-cache --inventory "${INV}" --output_dir "${RAW}"
fi
if [ ! -s "${RUNTIME}/runtime_input_manifest.json" ]; then
  RESUME=()
  if [ -e "${RUNTIME}" ]; then RESUME=(--resume); fi
  "${PY}" -u -m pose_point_depth_mv.dataset_tools.prepare_omni_real_runtime_inputs \
    --raw_cache_report "${RAW}/raw_cache_report.json" \
    --output_dir "${RUNTIME}" --selected_view_count 8 \
    "${RESUME[@]}"
fi
if [ ! -s "${ALIGN}/coarse_alignment_manifest.json" ]; then
  RESUME=()
  if [ -e "${ALIGN}" ]; then RESUME=(--resume); fi
  set +e
  "${PY}" -u -m pose_point_depth_mv.dataset_tools.align_omni_real_mesh_to_colmap \
    --raw_cache_report "${RAW}/raw_cache_report.json" \
    --output_dir "${ALIGN}" \
    "${RESUME[@]}"
  ALIGN_RC=$?
  set -e
  if [ "${ALIGN_RC}" -ne 0 ] && [ "${ALIGN_RC}" -ne 2 ]; then
    exit "${ALIGN_RC}"
  fi
fi
if [ ! -s "${ADJ}" ]; then
  set +e
  "${PY}" -u -m pose_point_depth_mv.dataset_tools.adjudicate_omni_real_mesh_alignment \
    --source_alignment_manifest "${ALIGN}/coarse_alignment_manifest.json" \
    --output "${ADJ}" --expected_objects 64
  ADJ_RC=$?
  set -e
  if [ "${ADJ_RC}" -ne 0 ] && [ "${ADJ_RC}" -ne 2 ]; then
    exit "${ADJ_RC}"
  fi
fi

LABEL_WARNING_ARGS=()
if [ "${ALLOW_ALIGNMENT_QUALITY_WARNINGS}" = 1 ]; then
  "${PY}" - "${ADJ}" "${ALIGNMENT_WARNING_CONTRACT}" \
    "${LABEL}" "${FULL_MODEL}" "${DINO_MODEL}" "${NO_VGGT}" \
    "${REAL_FULL}" "${SYNTH_FULL}" "${RECON}" "${PIXAL}" "${EVAL}" <<'PY'
import json
import sys
from pathlib import Path

from pose_point_depth_mv.dataset_tools.adjudicate_omni_real_mesh_alignment import (
    MANIFEST_FORMAT,
)
from pose_point_depth_mv.omni_real_benchmark_common import atomic_json, sha256_file

source_path = Path(sys.argv[1]).resolve()
contract_path = Path(sys.argv[2]).resolve()
source = json.load(open(source_path, encoding="utf-8"))
rows = list(source.get("objects", []))
warnings = [row for row in rows if row.get("automatic_passed") is not True]
assert source.get("format") == MANIFEST_FORMAT
assert source.get("passed") is False
assert source.get("failures") == []
assert source.get("selected_object_count") == 64
assert source.get("completed_object_count") == 64
assert source.get("automatic_pass_count") == 63
assert len(warnings) == 1
warning = warnings[0]
warning_key = f"{warning['category']}:{warning['object_id']}"
assert warning_key == "egg:egg_044"
assert warning.get("alignment_quality_checks", {}).get("median_normalized") is True
assert warning.get("alignment_quality_checks", {}).get("inlier_rate_3pct") is False

payload = {
    "format": "pose_point_depth_mv.m11_alignment_quality_warning_contract.v1",
    "source_alignment_manifest": str(source_path),
    "source_alignment_manifest_sha256": sha256_file(source_path),
    "selected_object_count": 64,
    "completed_object_count": 64,
    "alignment_quality_pass_count": 63,
    "alignment_quality_warning_count": 1,
    "alignment_quality_warning_records": [
        {
            "object_key": warning_key,
            "median_normalized": float(warning["median_normalized"]),
            "inlier_rate_3pct": float(warning["inlier_rate_3pct"]),
            "p90_normalized": float(
                warning.get(
                    "p90_normalized_diagnostic", warning.get("p90_normalized")
                )
            ),
            "alignment_quality_checks": warning["alignment_quality_checks"],
            "alignment_cache": warning["cache_npz"],
            "alignment_cache_sha256": sha256_file(Path(warning["cache_npz"])),
        }
    ],
    "primary_population": "all64",
    "sensitivity_populations": ["reliable63", "low_confidence1"],
    "object_deleted_or_replaced": False,
    "alignment_transform_refit": False,
    "model_input_changed": False,
    "decision_used_model_output_or_metric": False,
    "scope_guard": (
        "Post-consumption coverage amendment made from label-front-end evidence "
        "only. All 64 frozen objects and fitted transforms remain unchanged. "
        "All64 is primary; Reliable63 and LowConfidence1 are disclosures."
    ),
    "passed": True,
}
if contract_path.exists():
    assert json.load(open(contract_path, encoding="utf-8")) == payload
else:
    for output in map(Path, sys.argv[3:]):
        assert not output.exists(), (
            "warning contract must be frozen before downstream M11 outputs: "
            f"{output}"
        )
    atomic_json(contract_path, payload)
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY
  LABEL_WARNING_ARGS=(
    --include_alignment_quality_warnings
    --max_alignment_quality_warnings 1
  )
else
  "${PY}" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] is True and p["automatic_pass_count"]==64' "${ADJ}"
fi

if [ ! -s "${LABEL}/runtime_o_label_manifest.json" ]; then
  RESUME=()
  if [ -e "${LABEL}" ]; then RESUME=(--resume); fi
  "${PY}" -u -m pose_point_depth_mv.dataset_tools.prepare_omni_real_label_cache \
    --runtime_input_manifest "${RUNTIME}/runtime_input_manifest.json" \
    --alignment_manifest "${ADJ}" --output_dir "${LABEL}" \
    "${LABEL_WARNING_ARGS[@]}" \
    "${RESUME[@]}"
fi

prepare_model_inputs() {
  local MODULE=$1
  local OUT=$2
  if [ -s "${OUT}/model_input_manifest.json" ]; then return 0; fi
  local resume=()
  if [ -e "${OUT}" ]; then resume=(--resume); fi
  CUDA_VISIBLE_DEVICES="${GPU}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  "${PY}" -u -m "${MODULE}" \
    --runtime_input_manifest "${RUNTIME}/runtime_input_manifest.json" \
    --output_dir "${OUT}" --device cuda \
    "${resume[@]}"
}
prepare_model_inputs pose_point_depth_mv.dataset_tools.prepare_omni_real_model_inputs "${FULL_MODEL}"
prepare_model_inputs pose_point_depth_mv.dataset_tools.prepare_omni_real_dino_only_model_inputs "${DINO_MODEL}"

run_full() {
  local SS=$1
  local SLAT=$2
  local OUT=$3
  if [ -s "${OUT}/inference_manifest.json" ]; then return 0; fi
  CUDA_VISIBLE_DEVICES="${GPU}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PY}" -u -m pose_point_depth_mv.infer_omni_real_native_v2 \
    --model_input_manifest "${FULL_MODEL}/model_input_manifest.json" \
    --native_ss_checkpoint "${SS}" --native_slat_checkpoint "${SLAT}" \
    --stock_slat_freeze "${FREEZE}" --output_dir "${OUT}" \
    --seeds 42 --weights ema --amp_dtype bf16 --device cuda
}

if [ ! -s "${NO_VGGT}/inference_manifest.json" ]; then
  CUDA_VISIBLE_DEVICES="${GPU}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PY}" -u -m pose_point_depth_mv.infer_omni_real_native_no_vggt_mixed \
    --model_input_manifest "${DINO_MODEL}/model_input_manifest.json" \
    --native_ss_checkpoint "${SS_FINAL}" --native_slat_checkpoint "${SLAT_FINAL}" \
    --ss_migration_contract "${SS_CONTRACT}" \
    --slat_migration_contract "${SLAT_CONTRACT}" \
    --stock_slat_freeze "${FREEZE}" --output_dir "${NO_VGGT}" \
    --seeds 42 --weights ema --amp_dtype bf16 --device cuda
fi
run_full "${SS_REAL}" "${SLAT_REAL}" "${REAL_FULL}"
run_full "${SS_SYNTH}" "${SLAT_SYNTH}" "${SYNTH_FULL}"

if [ ! -s "${RECON}/inference_manifest.json" ]; then
  CUDA_VISIBLE_DEVICES="${GPU}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn SPCONV_ALGO=native MPLCONFIGDIR=/tmp/matplotlib \
  NUMBA_CACHE_DIR=/tmp/numba_cache TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PY}" -u -m pose_point_depth_mv.infer_omni_real_reconviagen \
    --runtime_input_manifest "${RUNTIME}/runtime_input_manifest.json" \
    --output_dir "${RECON}" --seeds 42 --device cuda --low_vram
fi
if [ ! -s "${PIXAL}/inference_manifest.json" ]; then
  CUDA_VISIBLE_DEVICES="${GPU}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa \
  MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PIXAL_PY}" -u -m pose_point_depth_mv.infer_omni_real_pixal3d \
    --runtime_input_manifest "${RUNTIME}/runtime_input_manifest.json" \
    --output_dir "${PIXAL}" --model_path "${PIXAL_MODEL}" \
    --naf_repo "${NAF_ROOT}/repo" --naf_checkpoint "${NAF_ROOT}/naf_release.pth" \
    --naf_source_manifest "${NAF_ROOT}/source_manifest.sha256" \
    --seeds 42 --device cuda --low_vram --resolution 1024 \
    --max_num_tokens 49152 --sampling_steps 12 \
    --isolate_objects --isolate_batch_size 1
fi

if [ ! -s "${EVAL}/report.json" ]; then
  "${PY}" -u -m pose_point_depth_mv.evaluate_omni_real_no_vggt_final \
    --label_manifest "${LABEL}/runtime_o_label_manifest.json" \
    --no_vggt_manifest "${NO_VGGT}/inference_manifest.json" \
    --real_full_manifest "${REAL_FULL}/inference_manifest.json" \
    --synthetic_full_manifest "${SYNTH_FULL}/inference_manifest.json" \
    --reconviagen_manifest "${RECON}/inference_manifest.json" \
    --pixal3d_manifest "${PIXAL}/inference_manifest.json" \
    --output_dir "${EVAL}" --protocol_scope formal_holdout64 \
    --frozen_split_manifest "${SPLIT}" --expected_objects 64 \
    --surface_samples 20000
fi

"${PY}" - "${EVAL}/report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["passed"] is True and report["formal"] is True
assert report["holdout64_consumed"] is True
assert report["formal_holdout_binding"]["passed"] is True
quality = report["label_quality_protocol"]
assert quality["primary_population"] == "all_objects"
if quality["low_confidence_object_count"]:
    assert quality["reliable_object_count"] == 63
    assert quality["low_confidence_object_keys"] == ["egg:egg_044"]
print({
    "formal_protocol_passed": True,
    "no_vggt_decision": report["no_vggt_decision"],
    "report": sys.argv[1],
})
PY
