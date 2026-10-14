#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zjr/anaconda3/envs/mpp_llava/bin/python}"
UPSTREAM_ROOT="${MPP_UPSTREAM_ROOT:-/home/zjr/Tracker/internship_project_lab/mpp_llava_video_workspace/third_party/MPP-LLaVA}"
CHECKPOINT_ROOT="${MPP_CHECKPOINT_ROOT:-/tmp/zjr_mpp_fast/checkpoints}"
CACHE_DIR="${LINGO_QFORMER_CACHE_DIR:-/tmp/zjr_mpp_fast/cache/lingoqa_5f_qformer_v1}"
DATA_DIR="${LINGO_PROCESSED_DIR:-$PROJECT_ROOT/datasets/processed/lingoqa_full_v2}"
OUTPUT_ROOT="${RECHECK_OUTPUT_ROOT:-$PROJECT_ROOT/outputs/projector_sft_eval}"

model="${1:?usage: run_projector_recheck.sh MODEL; MODEL=base|old_lora|joint}"
export PYTHONPATH="$PROJECT_ROOT:$UPSTREAM_ROOT"

model_args=()
case "$model" in
  base)
    splits=(action scenery)
    ;;
  old_lora)
    splits=(eval action scenery)
    model_args=(
      --adapter "$PROJECT_ROOT/outputs/mpp_lingoqa_full_v2_lora_1epoch/final_adapter"
    )
    ;;
  joint)
    splits=(action scenery)
    model_args=(
      --adapter "$PROJECT_ROOT/outputs/mpp_lingoqa_projector_lora_1epoch_v1/final_adapter"
      --projector "$PROJECT_ROOT/outputs/mpp_lingoqa_projector_lora_1epoch_v1/final_projector.pt"
    )
    ;;
  *)
    echo "unknown model: $model" >&2
    exit 2
    ;;
esac

for split in "${splits[@]}"; do
  case "$split" in
    eval) data="$DATA_DIR/eval.jsonl" ;;
    action) data="$DATA_DIR/dev_action.jsonl" ;;
    scenery) data="$DATA_DIR/dev_scenery.jsonl" ;;
  esac

  output_dir="$OUTPUT_ROOT/${model}_${split}"
  echo "Starting generation: model=$model split=$split data=$data" >&2

  "$PYTHON_BIN" -u "$PROJECT_ROOT/scripts/generate_answers.py" \
    --data "$data" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$output_dir" \
    --upstream-root "$UPSTREAM_ROOT" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --feature-stage qformer \
    --task-instruction-version legacy_v1 \
    --batch-size 1 \
    --max-new-tokens 96 \
    --max-records 0 \
    "${model_args[@]}"

  echo "Starting scoring: model=$model split=$split" >&2
  "$PYTHON_BIN" -u "$PROJECT_ROOT/scripts/evaluate_answers.py" \
    --predictions "$output_dir/predictions.jsonl" \
    --output-dir "$output_dir/scored" \
    --lingo-judge
done

echo "Completed recheck: model=$model" >&2
