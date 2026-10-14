#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN=/home/zjr/anaconda3/envs/mpp_llava/bin/python
AUDIT_DIR="$PROJECT_ROOT/outputs/evidence_case_audit_20261009"
MPP_INIT="$PROJECT_ROOT/outputs/mpp_feedback_sft_control_40k_2epoch_v2/final"
UPSTREAM="$PROJECT_ROOT/../mpp_llava_video_workspace/third_party/MPP-LLaVA"
VISUAL_ROOT="$PROJECT_ROOT/../mpp_llava_video_workspace/checkpoints"
export OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJECT_ROOT:$UPSTREAM:${PYTHONPATH:-}"
for condition in full duplicate roi; do
  "$PYTHON_BIN" -u scripts/generate_answers.py \
    --upstream-root "$UPSTREAM" --checkpoint-root /tmp/zjr_mpp_fast/checkpoints \
    --visual-checkpoint-root "$VISUAL_ROOT" --bridge-init checkpoints/visual_bridge_init.pt \
    --feature-stage eva --precision-protocol fp32_trainables_v1 \
    --task-instruction-version legacy_v1 --max-txt-len 1536 --qformer-mode auto \
    --adapter "$MPP_INIT/adapter" --projector "$MPP_INIT/projector.pt" \
    --qformer-adapter "$MPP_INIT/qformer_adapter.pt" \
    --data "$AUDIT_DIR/data.jsonl" --region-views "$AUDIT_DIR/views.jsonl" \
    --cache-dir /tmp/zjr_mpp_fast/cache/lingoqa_5f_eva_v1 \
    --region-cache-dir datasets/cache/lingoqa_vlm_evidence_eva_v1/dev \
    --region-condition "$condition" --output-dir "$AUDIT_DIR/$condition" \
    --batch-size 1 --max-new-tokens 96
  "$PYTHON_BIN" -u scripts/evaluate_answers.py \
    --predictions "$AUDIT_DIR/$condition/predictions.jsonl" \
    --output-dir "$AUDIT_DIR/$condition/scored" --lingo-judge
done
