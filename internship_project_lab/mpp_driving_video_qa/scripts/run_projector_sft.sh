#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zjr/anaconda3/envs/mpp_llava/bin/python}"
DATA_DIR="${LINGO_PROCESSED_DIR:-$PROJECT_ROOT/datasets/processed/lingoqa_full_v2}"
CACHE_DIR="${LINGO_QFORMER_CACHE_DIR:-/tmp/zjr_mpp_fast/cache/lingoqa_5f_qformer_v1}"
OUTPUT_DIR="${SFT_OUTPUT_DIR:-$PROJECT_ROOT/outputs/mpp_lingoqa_projector_lora_1epoch_v1}"
UPSTREAM_ROOT="${MPP_UPSTREAM_ROOT:-/home/zjr/Tracker/internship_project_lab/mpp_llava_video_workspace/third_party/MPP-LLaVA}"
PERSISTENT_ASSETS="/home/zjr/Tracker/internship_project_lab/mpp_llava_video_workspace/checkpoints"
CHECKPOINT_ROOT="${MPP_CHECKPOINT_ROOT:-/tmp/zjr_mpp_fast/checkpoints}"
if [[ ! -f "$CHECKPOINT_ROOT/mpp-qwen-next-sft/sharded/model.safetensors.index.json" ]]; then
  CHECKPOINT_ROOT="$PERSISTENT_ASSETS"
fi
VISUAL_CHECKPOINT_ROOT="${MPP_VISUAL_CHECKPOINT_ROOT:-$PERSISTENT_ASSETS}"

export PYTHON_BIN
export MPP_UPSTREAM_ROOT="$UPSTREAM_ROOT"
export MPP_CHECKPOINT_ROOT="$CHECKPOINT_ROOT"
export MPP_VISUAL_CHECKPOINT_ROOT="$VISUAL_CHECKPOINT_ROOT"
export LINGO_PROCESSED_DIR="$DATA_DIR"
export LINGO_CACHE_DIR="$CACHE_DIR"
export SFT_OUTPUT_DIR="$OUTPUT_DIR"
export MPP_FEATURE_STAGE=qformer
export MPP_TRAIN_PROJECTOR=1
export TASK_INSTRUCTION_VERSION=legacy_v1

action="${1:-help}"
case "$action" in
  cache)
    export CACHE_EXISTING_POLICY="${CACHE_EXISTING_POLICY:-validate}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
      MASTER_PORT="${MASTER_PORT:-29651}" \
      bash "$PROJECT_ROOT/scripts/run_pipeline.sh" cache
    ;;
  cache_status)
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/report_cache_progress.py" \
      --data-dir "$DATA_DIR" --cache-dir "$CACHE_DIR"
    ;;
  train)
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/validate_full_v2.py" \
      --data-dir "$DATA_DIR" --cache-dir "$CACHE_DIR" \
      --feature-stage qformer
    export SFT_EPOCHS="${SFT_EPOCHS:-1}"
    export SFT_SAVE_EVERY="${SFT_SAVE_EVERY:-5000}"
    export SFT_GRAD_ACCUMULATION="${SFT_GRAD_ACCUMULATION:-4}"
    export MPP_PROJECTOR_LR="${MPP_PROJECTOR_LR:-2e-5}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
      MASTER_PORT="${MASTER_PORT:-29652}" \
      bash "$PROJECT_ROOT/scripts/run_pipeline.sh" train
    ;;
  status)
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/report_progress.py" --output-dir "$OUTPUT_DIR"
    ;;
  eval)
    model="${2:?usage: run_projector_sft.sh eval MODEL SPLIT; MODEL=base|joint}"
    split="${3:?usage: run_projector_sft.sh eval MODEL SPLIT; SPLIT=action|scenery|eval}"
    case "$model" in
      base) model_args=() ;;
      joint)
        model_args=(--adapter "$OUTPUT_DIR/final_adapter"
                    --projector "$OUTPUT_DIR/final_projector.pt")
        ;;
      checkpoint-*)
        model_args=(--adapter "$OUTPUT_DIR/$model/adapter"
                    --projector "$OUTPUT_DIR/$model/projector.pt")
        ;;
      *) echo "unknown model: $model" >&2; exit 2 ;;
    esac
    case "$split" in
      action) data="$DATA_DIR/dev_action.jsonl" ;;
      scenery) data="$DATA_DIR/dev_scenery.jsonl" ;;
      eval) data="$DATA_DIR/eval.jsonl" ;;
      *) echo "unknown split: $split" >&2; exit 2 ;;
    esac
    eval_output="$PROJECT_ROOT/outputs/projector_sft_eval/${model}_${split}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      PYTHONPATH="$PROJECT_ROOT:$UPSTREAM_ROOT" \
      "$PYTHON_BIN" "$PROJECT_ROOT/scripts/generate_answers.py" \
      --data "$data" --cache-dir "$CACHE_DIR" --output-dir "$eval_output" \
      --feature-stage qformer --task-instruction-version legacy_v1 \
      --upstream-root "$UPSTREAM_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" \
      "${model_args[@]}"
    ;;
  score)
    model="${2:?usage: run_projector_sft.sh score MODEL SPLIT}"
    split="${3:?usage: run_projector_sft.sh score MODEL SPLIT}"
    eval_output="$PROJECT_ROOT/outputs/projector_sft_eval/${model}_${split}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONPATH="$PROJECT_ROOT" \
      "$PYTHON_BIN" "$PROJECT_ROOT/scripts/evaluate_answers.py" \
      --predictions "$eval_output/predictions.jsonl" \
      --output-dir "$eval_output/scored" --lingo-judge
    ;;
  *)
    echo "usage: $0 {cache|cache_status|train|status|eval MODEL SPLIT|score MODEL SPLIT}" >&2
    exit 2
    ;;
esac
