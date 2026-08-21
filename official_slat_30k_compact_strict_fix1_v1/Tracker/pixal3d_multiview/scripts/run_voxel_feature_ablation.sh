#!/usr/bin/env bash
set -u
set -o pipefail

ROOT=/home/zjr/Tracker
PY=${PY:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
GPU=${GPU:-1}
RUN_NAME=${RUN_NAME:-voxel_feature_ablation_128_v1}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-/data/pixal3d_multiview/objaverse_sparse_mv_artraj_pbr_5000_v9_select8/train.json}
VAL_MANIFEST=${VAL_MANIFEST:-/data/pixal3d_multiview/objaverse_sparse_mv_artraj_pbr_5000_v9_select8/val.json}
IMAGE_COND_MODEL=${IMAGE_COND_MODEL:-/home/zjr/Tracker/models/dinov3-vitl16-pretrain-lvd1689m}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/zjr/Tracker/pixal3d_multiview/outputs/eval_v9/${RUN_NAME}}
TRAIN_INDICES=${TRAIN_INDICES:-0-127}
VAL_INDICES=${VAL_INDICES:-0-127}
MAX_FRAMES=${MAX_FRAMES:-8}
EPOCHS=${EPOCHS:-20}
MAX_POS_PER_SAMPLE=${MAX_POS_PER_SAMPLE:-512}
NEG_PER_POS=${NEG_PER_POS:-3}
BATCH_SIZE=${BATCH_SIZE:-1024}
REDUCED_DIM=${REDUCED_DIM:-128}
HIDDEN_DIM=${HIDDEN_DIM:-256}
LR=${LR:-1e-4}
USE_STATS=${USE_STATS:-0}
MODES=${MODES:-"feature_only geometry_only support_only feature_geometry feature_support geometry_support full"}

mkdir -p "${OUTPUT_ROOT}"
cd "${ROOT}" || exit 1

echo "[config] output_root=${OUTPUT_ROOT}"
echo "[config] modes=${MODES}"
echo "[config] train_indices=${TRAIN_INDICES} val_indices=${VAL_INDICES}"

failures=()
for mode in ${MODES}; do
  out="${OUTPUT_ROOT}/${mode}"
  mkdir -p "${out}"
  echo "[start] mode=${mode} output=${out}"
  stats_args=()
  if [[ "${USE_STATS}" == "0" ]]; then
    stats_args+=(--no_stats)
  fi

  if CUDA_VISIBLE_DEVICES="${GPU}" \
    HF_HUB_OFFLINE=1 \
    ATTN_BACKEND=flash_attn \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    "${PY}" -u pixal3d_multiview/train_voxel_feature_diagnostic_head.py \
      --train_manifest "${TRAIN_MANIFEST}" \
      --val_manifest "${VAL_MANIFEST}" \
      --output_dir "${out}" \
      --image_cond_model "${IMAGE_COND_MODEL}" \
      --device cuda \
      --train_indices "${TRAIN_INDICES}" \
      --val_indices "${VAL_INDICES}" \
      --max_frames "${MAX_FRAMES}" \
      --epochs "${EPOCHS}" \
      --max_pos_per_sample "${MAX_POS_PER_SAMPLE}" \
      --neg_per_pos "${NEG_PER_POS}" \
      --batch_size "${BATCH_SIZE}" \
      --reduced_dim "${REDUCED_DIM}" \
      --hidden_dim "${HIDDEN_DIM}" \
      --lr "${LR}" \
      --feature_ablation "${mode}" \
      "${stats_args[@]}" \
      2>&1 | tee "${out}/run.log"; then
    echo "[done] mode=${mode}"
  else
    echo "[failed] mode=${mode}"
    failures+=("${mode}")
  fi
done

"${PY}" -c "
import json
from pathlib import Path
root = Path('${OUTPUT_ROOT}')
rows = []
for path in sorted(root.glob('*/summary.json')):
    data = json.load(open(path))
    mode = path.parent.name
    train = data.get('train_metrics', {})
    val = data.get('val_metrics', {})
    rows.append([
        mode,
        train.get('auc'),
        val.get('auc'),
        val.get('ap'),
        val.get('score_gap'),
        val.get('precision@0.5'),
        val.get('recall@0.5'),
        val.get('target_score_mean'),
        val.get('non_target_score_mean'),
    ])
out = root / 'summary.tsv'
with out.open('w') as f:
    f.write('mode\\ttrain_auc\\tval_auc\\tval_ap\\tval_gap\\tval_p50\\tval_r50\\tval_target_score\\tval_non_target_score\\n')
    for row in rows:
        f.write('\\t'.join('' if x is None else str(x) for x in row) + '\\n')
print(f'[summary] {out}')
"

if [[ "${#failures[@]}" -gt 0 ]]; then
  printf '[failures]'
  printf ' %s' "${failures[@]}"
  printf '\n'
  exit 1
fi

echo "[all_done] ${OUTPUT_ROOT}"
