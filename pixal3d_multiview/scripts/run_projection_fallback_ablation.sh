#!/usr/bin/env bash
set -u

ROOT=${ROOT:-/home/zjr/Tracker}
PYTHON=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
GPU=${GPU:-1}
MANIFEST=${MANIFEST:-/data/pixal3d_multiview/objaverse_sparse_mv_artraj_pbr_5000_v9_select8/val.json}
CHECKPOINT=${CHECKPOINT:-/home/zjr/Tracker/pixal3d_multiview/outputs/train_v9/objaverse_5000_v9_select8_proj_only_e6/final.pt}
IMAGE_COND_MODEL=${IMAGE_COND_MODEL:-/home/zjr/Tracker/models/dinov3-vitl16-pretrain-lvd1689m}
RUN_ID=${RUN_ID:-projection_fallback_ablation_$(date -u +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/zjr/Tracker/pixal3d_multiview/outputs/eval_v9/${RUN_ID}}
INDICES_SMALL=${INDICES_SMALL:-0,1,5,10,20,30,50,80}
INDICES_SAMPLING=${INDICES_SAMPLING:-0,1,5,10,20,30,50,80,100}
MAX_SAMPLES=${MAX_SAMPLES:-64}
STEPS=${STEPS:-30}

mkdir -p "${OUTPUT_ROOT}/logs"
REPORT="${OUTPUT_ROOT}/测试记录.md"
FAILURES="${OUTPUT_ROOT}/失败汇总.txt"
: > "${FAILURES}"

timestamp() {
  date -u "+%Y-%m-%d %H:%M:%S UTC"
}

record() {
  echo "$1" | tee -a "${REPORT}"
}

run_step() {
  local name="$1"
  shift
  local log="${OUTPUT_ROOT}/logs/${name}.log"
  record ""
  record "## ${name}"
  record ""
  record "- 时间: $(timestamp)"
  record "- 消融: 建议1，去掉大量 zero projected feature"
  record "- 日志: \`${log}\`"
  record "- 命令: \`$*\`"
  if "$@" > "${log}" 2>&1; then
    record "- 状态: completed"
    record "- 完成时间: $(timestamp)"
  else
    local code=$?
    record "- 状态: failed (${code})"
    record "- 失败时间: $(timestamp)"
    {
      echo "[${name}] exit_code=${code}"
      tail -n 80 "${log}"
      echo ""
    } >> "${FAILURES}"
  fi
}

cd "${ROOT}" || exit 1
cat > "${REPORT}" <<EOF
# Projection Fallback 消融测试记录

- 开始时间: $(timestamp)
- 消融: 建议1，去掉大量 zero projected feature
- 输出目录: \`${OUTPUT_ROOT}\`
- manifest: \`${MANIFEST}\`
- checkpoint: \`${CHECKPOINT}\`
- GPU: \`${GPU}\`

EOF

COMMON_ENV=(
  env
  CUDA_VISIBLE_DEVICES="${GPU}"
  HF_HUB_OFFLINE=1
  ATTN_BACKEND=flash_attn
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  MPLCONFIGDIR=/tmp/matplotlib
  NUMBA_CACHE_DIR=/tmp/numba_cache
)

run_step adapter_zero \
  "${COMMON_ENV[@]}" "${PYTHON}" -u pixal3d_multiview/eval_condition_adapter_stats.py \
    --manifest "${MANIFEST}" \
    --output_dir "${OUTPUT_ROOT}/adapter_zero" \
    --image_cond_model "${IMAGE_COND_MODEL}" \
    --indices "${INDICES_SMALL}" \
    --max_frames 8 \
    --empty_policy zero \
    --ablation_name "建议1_zero_projected_feature_baseline_zero"

run_step adapter_soft \
  "${COMMON_ENV[@]}" "${PYTHON}" -u pixal3d_multiview/eval_condition_adapter_stats.py \
    --manifest "${MANIFEST}" \
    --output_dir "${OUTPUT_ROOT}/adapter_soft" \
    --image_cond_model "${IMAGE_COND_MODEL}" \
    --indices "${INDICES_SMALL}" \
    --max_frames 8 \
    --empty_policy soft \
    --fallback_weight 1.0 \
    --support_confidence_power 1.0 \
    --ablation_name "建议1_zero_projected_feature_soft_fallback"

for policy in zero soft; do
  for pose in correct shuffle identity; do
    extra=()
    if [[ "${policy}" == "soft" ]]; then
      extra+=(--fallback_weight 1.0 --support_confidence_power 1.0)
    fi
    run_step "fixed_${policy}_${pose}" \
      "${COMMON_ENV[@]}" "${PYTHON}" -u pixal3d_multiview/eval_fixed_train_loss.py \
        --train_manifest "${MANIFEST}" \
        --checkpoint "${CHECKPOINT}" \
        --checkpoint_only \
        --output "${OUTPUT_ROOT}/fixed_loss/${policy}_${pose}.json" \
        --image_cond_model "${IMAGE_COND_MODEL}" \
        --max_frames 8 \
        --max_samples "${MAX_SAMPLES}" \
        --fixed_t 0.5 \
        --amp_dtype bf16 \
        --pose_mode "${pose}" \
        --empty_policy "${policy}" \
        "${extra[@]}" \
        --ablation_name "建议1_zero_projected_feature_${policy}_${pose}" \
        --quiet
  done
done

for policy in zero soft; do
  for pose in correct shuffle; do
    extra=()
    if [[ "${policy}" == "soft" ]]; then
      extra+=(--fallback_weight 1.0 --support_confidence_power 1.0)
    fi
    run_step "sparse_${policy}_${pose}" \
      "${COMMON_ENV[@]}" "${PYTHON}" -u pixal3d_multiview/eval_sparse_sampling_batch.py \
        --manifest "${MANIFEST}" \
        --checkpoint "${CHECKPOINT}" \
        --output_dir "${OUTPUT_ROOT}/sparse_sampling/${policy}_${pose}" \
        --image_cond_model "${IMAGE_COND_MODEL}" \
        --indices "${INDICES_SAMPLING}" \
        --max_frames 8 \
        --steps "${STEPS}" \
        --seed 1234 \
        --pose_mode "${pose}" \
        --empty_policy "${policy}" \
        "${extra[@]}" \
        --ablation_name "建议1_zero_projected_feature_${policy}_${pose}" \
        --quiet
  done
done

record ""
record "## 汇总"
record ""
record "- 结束时间: $(timestamp)"
record "- adapter zero: \`${OUTPUT_ROOT}/adapter_zero/summary.json\`"
record "- adapter soft: \`${OUTPUT_ROOT}/adapter_soft/summary.json\`"
record "- fixed loss: \`${OUTPUT_ROOT}/fixed_loss\`"
record "- sparse sampling: \`${OUTPUT_ROOT}/sparse_sampling\`"
record "- 失败汇总: \`${FAILURES}\`"

if [[ -s "${FAILURES}" ]]; then
  record "- 总状态: 有失败，查看失败汇总。"
else
  record "- 总状态: 全部完成。"
fi
