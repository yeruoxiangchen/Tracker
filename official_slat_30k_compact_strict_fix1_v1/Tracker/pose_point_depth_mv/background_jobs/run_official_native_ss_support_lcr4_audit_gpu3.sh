#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=${PY:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
GPU=${GPU:-3}
MIN_FREE_GPU_MIB=${MIN_FREE_GPU_MIB:-22000}
SOURCE=/data/zjr/proobjaverse_official_slat_train2000_20260813_v1
RUN=/data/zjr/proobjaverse_official_native_ss_train2000_20260815_v1
CACHE=${SOURCE}/cache_dev64_protocol2128_views8_v1/slat_manifest.json
LIFTING=${SOURCE}/cache_dev64_protocol2128_views8_v1/lifting_manifest.json
AGG=${RUN}/dev48_newss2000_stock_and_slat8000_mesh_seed424344_5gpu_v1/aggregate_v1/report.json
SS_REPORT=${RUN}/dev64_step2000_eval16_64_seed424344_6gpu_v1/aggregate_v1/report.json
FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json
SLAT_CKPT=${SOURCE}/B_condition_lora_train2000_step8000_seed42_4gpu_v1/checkpoints/step_008000.pt
AUDIT=${RUN}/dev48_support_explosion1_lcr_worst3_audit_20260815_v1
SELECTION=${AUDIT}/selection.json
RERUN=${AUDIT}/reruns
FINAL=${AUDIT}/audit_report.json

for REQUIRED in "${PY}" "${CACHE}" "${LIFTING}" "${AGG}" "${SS_REPORT}" "${FREEZE}" "${SLAT_CKPT}"; do
  test -e "${REQUIRED}"
done
FREE_MIB=$(nvidia-smi -i "${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
if ! [[ "${FREE_MIB}" =~ ^[0-9]+$ ]] || [ "${FREE_MIB}" -lt "${MIN_FREE_GPU_MIB}" ]; then
  echo "audit blocked: GPU ${GPU} free=${FREE_MIB:-?} MiB required=${MIN_FREE_GPU_MIB} MiB"
  exit 95
fi

mkdir -p "${AUDIT}" "${RERUN}" "${RUN}/logs"
${PY} -u -m pose_point_depth_mv.audit_proobjaverse_native_ss_support_lcr_cases \
  select \
  --aggregate_report "${AGG}" \
  --cache_manifest "${CACHE}" \
  --rerun_root "${RERUN}" \
  --output "${SELECTION}" \
  --lcr_count 3 \
  --expected_support_objects 1

mapfile -t CASES < <(${PY} -c '
import json,sys
r=json.load(open(sys.argv[1]))
for x in r["cases"]:
    print("\t".join(str(x[key]) for key in (
        "label", "kind", "object_index", "object_end", "rerun_output"
    )))
' "${SELECTION}")
if [ "${#CASES[@]}" -ne 4 ]; then
  echo "audit blocked: expected exactly four selected cases"
  exit 2
fi

for CASE in "${CASES[@]}"; do
  IFS=$'\t' read -r LABEL KIND START END OUT <<<"${CASE}"
  echo "[support_lcr_audit] start ${LABEL} kind=${KIND} slice=[${START},${END})"
  if [ -s "${OUT}/report.json" ]; then
    echo "[support_lcr_audit] reuse ${OUT}/report.json"
    continue
  fi
  RESUME=()
  if [ -d "${OUT}" ]; then RESUME=(--resume); fi
  set +e
  CUDA_VISIBLE_DEVICES=${GPU} \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ${PY} -u -m pose_point_depth_mv.evaluate_proobjaverse_official_native_ss_stock_slat \
    worker \
    --cache_manifest "${CACHE}" \
    --lifting_cache_manifest "${LIFTING}" \
    --native_ss_report "${SS_REPORT}" \
    --stock_slat_freeze "${FREEZE}" \
    --trained_slat_checkpoint "${SLAT_CKPT}" \
    --trained_slat_weights ema \
    --expected_trained_slat_step 8000 \
    --output_dir "${OUT}" \
    --object_start "${START}" --object_end "${END}" \
    --joint_seeds 42,43,44 \
    --weights ema --surface_samples 20000 --amp_dtype bf16 \
    --save_meshes "${RESUME[@]}"
  RC=$?
  set -e
  test -s "${OUT}/report.json"
  ${PY} -c '
import json,sys
r=json.load(open(sys.argv[1])); kind=sys.argv[2]; rc=int(sys.argv[3])
assert r["complete"] is True and r["object_count"] == 1 and r["record_count"] == 3
if kind == "lcr_worst":
    assert rc == 0 and r["passed"] is True
else:
    assert rc in (0,2) and r["passed"] is False
print({"case":kind,"runtime_passed":r["passed"],"worker_rc":rc})
' "${OUT}/report.json" "${KIND}" "${RC}"
done

${PY} -u -m pose_point_depth_mv.audit_proobjaverse_native_ss_support_lcr_cases \
  finalize \
  --selection "${SELECTION}" \
  --output "${FINAL}"

echo "audit report: ${FINAL}"
echo "audit summary: ${AUDIT}/summary.txt"
