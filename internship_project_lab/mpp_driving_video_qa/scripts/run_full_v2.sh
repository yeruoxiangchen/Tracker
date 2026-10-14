#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zjr/anaconda3/envs/mpp_llava/bin/python}"
RAW_ROOT="${LINGO_RAW_ROOT:-/data/zjr/datasets/lingoqa}"
DATA_DIR="${LINGO_PROCESSED_DIR:-$PROJECT_ROOT/datasets/processed/lingoqa_full_v2}"
CACHE_DIR="${LINGO_CACHE_DIR:-/tmp/zjr_mpp_fast/cache/lingoqa_5f_mpp}"
OUTPUT_DIR="${SFT_OUTPUT_DIR:-$PROJECT_ROOT/outputs/mpp_lingoqa_full_v2_lora_1epoch}"
ASSET_ROOT="${MPP_ASSET_ROOT:-/tmp/zjr_mpp_fast}"
if [[ ! -d "$ASSET_ROOT/checkpoints" ]]; then
  ASSET_ROOT="/home/zjr/Tracker/internship_project_lab/mpp_llava_video_workspace"
fi
UPSTREAM_ROOT="${MPP_UPSTREAM_ROOT:-/home/zjr/Tracker/internship_project_lab/mpp_llava_video_workspace/third_party/MPP-LLaVA}"
CHECKPOINT_ROOT="${MPP_CHECKPOINT_ROOT:-$ASSET_ROOT/checkpoints}"
VISUAL_CHECKPOINT_ROOT="${MPP_VISUAL_CHECKPOINT_ROOT:-/home/zjr/Tracker/internship_project_lab/mpp_llava_video_workspace/checkpoints}"

export PYTHON_BIN MPP_UPSTREAM_ROOT="$UPSTREAM_ROOT" MPP_CHECKPOINT_ROOT="$CHECKPOINT_ROOT"
export LINGO_PROCESSED_DIR="$DATA_DIR" LINGO_CACHE_DIR="$CACHE_DIR" SFT_OUTPUT_DIR="$OUTPUT_DIR"
export TASK_INSTRUCTION_VERSION="legacy_v1"

action="${1:-help}"
case "$action" in
  prepare)
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/prepare_lingoqa.py" \
      --train-input "action=$RAW_ROOT/action/train.parquet" \
      --train-input "scenery=$RAW_ROOT/scenery/train.parquet" \
      --eval-input "$RAW_ROOT/evaluation/val.parquet" \
      --frames-root-map "action=$RAW_ROOT/action/extracted" \
      --frames-root-map "scenery=$RAW_ROOT/scenery/extracted" \
      --frames-root-map "evaluation=$RAW_ROOT/evaluation/extracted" \
      --fixed-dev-data "action=$PROJECT_ROOT/datasets/processed/lingoqa_v1/dev.jsonl" \
      --dev-segments-by-source scenery=100 \
      --train-size 0 --max-qa-per-segment 0 --max-dev-qa-per-segment 12 \
      --dev-segments 0 --seed 42 --check-files --output-dir "$DATA_DIR"
    ;;
  cache)
    export CACHE_EXISTING_POLICY="${CACHE_EXISTING_POLICY:-validate}"
    visual_checkpoint="$VISUAL_CHECKPOINT_ROOT/ckpt/blip2/blip2_pretrained_flant5xxl.pth"
    projection_index="$CHECKPOINT_ROOT/mpp-qwen-next-sft/sharded/model.safetensors.index.json"
    if [[ ! -f "$visual_checkpoint" ]]; then
      echo "missing visual checkpoint: $visual_checkpoint" >&2
      exit 2
    fi
    if [[ ! -f "$projection_index" ]]; then
      echo "missing MPP projection index: $projection_index" >&2
      exit 2
    fi
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
      "$PYTHON_BIN" -m torch.distributed.run \
      --nproc_per_node=4 --master_port="${MASTER_PORT:-29641}" \
      "$PROJECT_ROOT/scripts/cache_visual_features.py" \
      --data "$DATA_DIR/train.jsonl" --data "$DATA_DIR/dev.jsonl" \
      --data "$DATA_DIR/eval.jsonl" --cache-dir "$CACHE_DIR" \
      --existing-policy "$CACHE_EXISTING_POLICY" \
      --upstream-root "$UPSTREAM_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" \
      --visual-checkpoint-root "$VISUAL_CHECKPOINT_ROOT"
    ;;
  cache_status)
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/report_cache_progress.py" \
      --data-dir "$DATA_DIR" --cache-dir "$CACHE_DIR"
    ;;
  train)
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/validate_full_v2.py" \
      --data-dir "$DATA_DIR" --cache-dir "$CACHE_DIR"
    export SFT_EPOCHS="${SFT_EPOCHS:-1}"
    export SFT_SAVE_EVERY="${SFT_SAVE_EVERY:-5000}"
    export MASTER_PORT="${MASTER_PORT:-29642}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
      bash "$PROJECT_ROOT/scripts/run_pipeline.sh" train
    ;;
  status)
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/report_progress.py" --output-dir "$OUTPUT_DIR"
    ;;
  eval)
    model="${2:?usage: run_full_v2.sh eval MODEL SPLIT; MODEL=base|old|new}"
    split="${3:?usage: run_full_v2.sh eval MODEL SPLIT; SPLIT=action|scenery|eval}"
    case "$model" in
      base) adapter="" ;;
      old) adapter="$PROJECT_ROOT/outputs/mpp_lingoqa_lora_gc_full_v1/final_adapter" ;;
      new) adapter="$OUTPUT_DIR/final_adapter" ;;
      checkpoint-*) adapter="$OUTPUT_DIR/$model/adapter" ;;
      *) echo "unknown model: $model" >&2; exit 2 ;;
    esac
    case "$split" in
      action) data="$DATA_DIR/dev_action.jsonl" ;;
      scenery) data="$DATA_DIR/dev_scenery.jsonl" ;;
      eval) data="$DATA_DIR/eval.jsonl" ;;
      *) echo "unknown split: $split" >&2; exit 2 ;;
    esac
    eval_output="$PROJECT_ROOT/outputs/full_v2_eval/${model}_${split}"
    adapter_args=()
    if [[ -n "$adapter" ]]; then adapter_args+=(--adapter "$adapter"); fi
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONPATH="$PROJECT_ROOT:$UPSTREAM_ROOT" \
      "$PYTHON_BIN" "$PROJECT_ROOT/scripts/generate_answers.py" \
      --data "$data" --cache-dir "$CACHE_DIR" --output-dir "$eval_output" \
      --task-instruction-version legacy_v1 "${adapter_args[@]}" \
      --upstream-root "$UPSTREAM_ROOT" --checkpoint-root "$CHECKPOINT_ROOT"
    ;;
  score)
    model="${2:?usage: run_full_v2.sh score MODEL SPLIT}"
    split="${3:?usage: run_full_v2.sh score MODEL SPLIT}"
    eval_output="$PROJECT_ROOT/outputs/full_v2_eval/${model}_${split}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONPATH="$PROJECT_ROOT" \
      "$PYTHON_BIN" "$PROJECT_ROOT/scripts/evaluate_answers.py" \
      --predictions "$eval_output/predictions.jsonl" \
      --output-dir "$eval_output/scored" --lingo-judge
    ;;
  report)
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/report_full_v2.py" \
      --eval-root "$PROJECT_ROOT/outputs/full_v2_eval" \
      --output "$PROJECT_ROOT/outputs/full_v2_eval/report.json"
    ;;
  *)
    echo "usage: $0 {prepare|cache|cache_status|train|status|eval MODEL SPLIT|score MODEL SPLIT|report}" >&2
    exit 2
    ;;
esac
