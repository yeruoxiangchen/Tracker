#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zjr/Tracker
PYTHON=/home/zjr/anaconda3/envs/reconviagen/bin/python

GPU=${GPU:-1}
RUN_NAME=${RUN_NAME:-pose_head_reranker_s200}
VAL_MANIFEST=${VAL_MANIFEST:-/data/pixal3d_multiview/objaverse_sparse_mv_artraj_pbr_5000_v9_select8/val.json}
IMAGE_COND_MODEL=${IMAGE_COND_MODEL:-${ROOT}/models/dinov3-vitl16-pretrain-lvd1689m}
HEAD_CHECKPOINT=${HEAD_CHECKPOINT:-${ROOT}/pixal3d_multiview/outputs/train_v9/pose_consistency_heads/pose_consistency_pairwise_centered_s200/final.pt}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/pixal3d_multiview/outputs/eval_v9/${RUN_NAME}}

INDICES=${INDICES:-0-31}
CANDIDATE_POSE_MODES=${CANDIDATE_POSE_MODES:-correct,cyclic_shift1,cyclic_shift2,reverse,noise,large_noise}
REFERENCE_POSE=${REFERENCE_POSE:-correct}
INPUT_POSE=${INPUT_POSE:-correct}
SELECTION_METRIC=${SELECTION_METRIC:-score}
SCORE_THRESHOLD=${SCORE_THRESHOLD:-}
MARGIN_THRESHOLD=${MARGIN_THRESHOLD:-0.05}

export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HUB_OFFLINE=1
export ATTN_BACKEND=flash_attn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/matplotlib
export NUMBA_CACHE_DIR=/tmp/numba_cache

cd "${ROOT}"
mkdir -p "${OUTPUT_DIR}"

echo "[config] run_name=${RUN_NAME}"
echo "[config] gpu=${GPU}"
echo "[config] val_manifest=${VAL_MANIFEST}"
echo "[config] head_checkpoint=${HEAD_CHECKPOINT}"
echo "[config] output_dir=${OUTPUT_DIR}"
echo "[config] indices=${INDICES}"
echo "[config] candidate_pose_modes=${CANDIDATE_POSE_MODES}"
echo "[config] reference_pose=${REFERENCE_POSE}"
echo "[config] input_pose=${INPUT_POSE}"
echo "[config] selection_metric=${SELECTION_METRIC}"
echo "[config] score_threshold=${SCORE_THRESHOLD:-<disabled>}"
echo "[config] margin_threshold=${MARGIN_THRESHOLD}"

THRESHOLD_ARGS=()
if [[ -n "${SCORE_THRESHOLD}" ]]; then
  THRESHOLD_ARGS+=(--score_threshold "${SCORE_THRESHOLD}")
fi

"${PYTHON}" -u pixal3d_multiview/eval_pose_head_reranker.py \
  --manifest "${VAL_MANIFEST}" \
  --head_checkpoint "${HEAD_CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --image_cond_model "${IMAGE_COND_MODEL}" \
  --max_frames 8 \
  --indices "${INDICES}" \
  --candidate_pose_modes "${CANDIDATE_POSE_MODES}" \
  --reference_pose "${REFERENCE_POSE}" \
  --input_pose "${INPUT_POSE}" \
  --selection_metric "${SELECTION_METRIC}" \
  "${THRESHOLD_ARGS[@]}" \
  --margin_threshold "${MARGIN_THRESHOLD}" \
  --empty_policy zero \
  --global_fusion concat

cat <<EOF
[summary]
report:    ${OUTPUT_DIR}/rerank_report.md
decisions: ${OUTPUT_DIR}/rerank_decisions.csv
selection: ${OUTPUT_DIR}/rerank_selection.jsonl
scores:    ${OUTPUT_DIR}/candidate_scores_ranked.csv
EOF
