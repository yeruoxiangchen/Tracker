#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-/home/zjr/anaconda3/envs/mpp_llava/bin/python}"
OUTPUT_DIR="${PROMPT_ABLATION_DIR:-outputs/task_instruction_ablation_v1}"
TRAIN_DIR="${TRAIN_DIR:-outputs/mpp_lingoqa_lora_gc_full_v1}"
model="${1:?usage: run_task_instruction_ablation.sh base|epoch03}"
adapter=()
case "$model" in
  base) ;;
  epoch03) adapter=(--adapter "$TRAIN_DIR/epoch-03/adapter") ;;
  *) echo "unknown model: $model" >&2; exit 2 ;;
esac
export PYTHONPATH=".:../mpp_llava_video_workspace/third_party/MPP-LLaVA"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
"$PYTHON_BIN" -u scripts/generate_answers.py \
  --data "$OUTPUT_DIR/diagnostic.jsonl" \
  --cache-dir "${LINGO_CACHE_DIR:-/tmp/zjr_mpp_fast/cache/lingoqa_5f_mpp}" \
  --upstream-root ../mpp_llava_video_workspace/third_party/MPP-LLaVA \
  --checkpoint-root "${MPP_CHECKPOINT_ROOT:-/tmp/zjr_mpp_fast/checkpoints}" \
  --output-dir "$OUTPUT_DIR/$model" \
  --task-instruction-version driving_perspective_v1 \
  "${adapter[@]}"
"$PYTHON_BIN" -u scripts/evaluate_answers.py \
  --predictions "$OUTPUT_DIR/$model/predictions.jsonl" \
  --output-dir "$OUTPUT_DIR/$model/scored" --lingo-judge
