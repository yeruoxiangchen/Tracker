#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
PYTHON=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
GPU=${GPU:?GPU is required}
SELECTION=${SELECTION:?SELECTION is required}
SELECTION_POSITION=${SELECTION_POSITION:?SELECTION_POSITION is required}

cd "${PROJECT_ROOT}"
export PYTHONPATH="$PWD:$PWD/ReconViaGen:$PWD/ReconViaGen/wheels/vggt"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export ATTN_BACKEND=${ATTN_BACKEND:-flash_attn}
export SPCONV_ALGO=${SPCONV_ALGO:-native}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba_cache}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

readarray -t VALUES < <(
  "${PYTHON}" -c '
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
rows=[r for r in p["selections"] if int(r["selection_position"])==int(sys.argv[2])]
assert len(rows)==1
r=rows[0]
print(r["dev_index"])
print(r["case_dir"])
' "${SELECTION}" "${SELECTION_POSITION}"
)
DEV_INDEX=${VALUES[0]}
CASE_DIR=${VALUES[1]}
ENDPOINT=${CASE_DIR}/_audited_endpoint_worker

SLAT_ROOT=/data/zjr/proobjaverse_official_slat_train2000_20260813_v1
SS_ROOT=/data/zjr/proobjaverse_official_native_ss_train2000_with_vggt_20260817_v1
SLAT_CACHE=${SLAT_ROOT}/cache_dev64_protocol2128_views8_with_vggt_sidecar_v1/with_vggt_slat_manifest.json
SLAT_LIFTING=${SLAT_ROOT}/cache_dev64_protocol2128_views8_with_vggt_sidecar_v1/with_vggt_lifting_manifest.json
SS_CACHE=${SS_ROOT}/cache_dev64_official_ss_with_vggt_sidecar_historical_v2_fix2_v1/with_vggt_ss_manifest.json
VSS_REPORT=${SS_ROOT}/dev48_VSS_step2000_seed424344_2gpu03_manual_v3/aggregate_v1/report.json
V_CHECKPOINT=${SLAT_ROOT}/V_with_vggt_train2000_step15000_seed42_8gpu_strict_perf_v1_v1/checkpoints/step_015000.pt
STOCK_FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json

for path in "${SELECTION}" "${SLAT_CACHE}" "${SLAT_LIFTING}" "${SS_CACHE}" "${VSS_REPORT}" "${V_CHECKPOINT}" "${STOCK_FREEZE}"; do
  test -s "${path}"
done

if [ ! -s "${ENDPOINT}/report.json" ]; then
  RESUME=()
  if [ -e "${ENDPOINT}" ]; then RESUME+=(--resume); fi
  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u -m \
    official_ss_with_vggt_perf_v1.evaluate_ss_slat worker \
      --cache_manifest "${SLAT_CACHE}" \
      --lifting_cache_manifest "${SLAT_LIFTING}" \
      --ss_cache_manifest "${SS_CACHE}" \
      --native_ss_report "${VSS_REPORT}" \
      --stock_slat_freeze "${STOCK_FREEZE}" \
      --trained_slat_checkpoint "${V_CHECKPOINT}" \
      --trained_slat_weights ema \
      --expected_trained_slat_step 15000 \
      --expected_checkpoint_training_membership all_disjoint \
      --output_dir "${ENDPOINT}" \
      --weights ema \
      --joint_seeds 42 \
      --object_start "${DEV_INDEX}" \
      --object_end "$((DEV_INDEX + 1))" \
      --surface_samples 20000 \
      --amp_dtype bf16 \
      --save_meshes \
      "${RESUME[@]}"
  ENDPOINT_RC=$?
  set -e
  if (( ENDPOINT_RC == 2 )) && [ ! -s "${ENDPOINT}/report.json" ]; then
    echo "ERROR: endpoint returned rc=2 without a completed report" >&2
    exit 92
  fi
  if (( ENDPOINT_RC != 0 && ENDPOINT_RC != 2 )); then
    echo "ERROR: audited endpoint worker failed rc=${ENDPOINT_RC}" >&2
    exit "${ENDPOINT_RC}"
  fi
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u -m \
  pose_point_depth_mv.export_proobjaverse_dev_with_vggt_qualitative \
  reconviagen \
  --selection "${SELECTION}" \
  --selection_position "${SELECTION_POSITION}" \
  --device cuda \
  --low_vram \
  --resume

if [ ! -s "${CASE_DIR}/case_report.json" ]; then
  "${PYTHON}" -u -m pose_point_depth_mv.export_proobjaverse_dev_with_vggt_qualitative \
    finalize \
    --selection "${SELECTION}" \
    --selection_position "${SELECTION_POSITION}" \
    --endpoint_worker "${ENDPOINT}"
fi

echo "DEV QUALITATIVE CASE COMPLETE: ${CASE_DIR}"
