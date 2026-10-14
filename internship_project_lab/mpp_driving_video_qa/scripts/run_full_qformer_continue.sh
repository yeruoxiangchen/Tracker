#!/usr/bin/env bash
# One new training stage initialized from the completed full-data model.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zjr/anaconda3/envs/mpp_llava/bin/python}"
PARENT_DIR="${CONTINUE_PARENT_DIR:-$PROJECT_ROOT/outputs/mpp_lingoqa_qformer_all_1epoch_v1}"
PARENT_EVAL_DIR="${CONTINUE_PARENT_EVAL_DIR:-$PROJECT_ROOT/outputs/full_qformer_scale_eval_v1}"
export FULL_QFORMER_DATA_DIR="${CONTINUE_DATA_DIR:-$PROJECT_ROOT/datasets/processed/lingoqa_qformer_all_v1}"
export FULL_QFORMER_TRAIN_DIR="${CONTINUE_TRAIN_DIR:-$PROJECT_ROOT/outputs/mpp_lingoqa_qformer_continue_1epoch_v1}"
export FULL_QFORMER_EVAL_DIR="${CONTINUE_EVAL_DIR:-$PROJECT_ROOT/outputs/full_qformer_continue_eval_v1}"
export SFT_PARENT_RUN="$PARENT_DIR"
export SFT_INIT_ADAPTER="$PARENT_DIR/final_adapter"
export SFT_INIT_PROJECTOR="$PARENT_DIR/final_projector.pt"
export SFT_INIT_QFORMER="$PARENT_DIR/final_qformer_adapter.pt"
export SFT_EPOCHS=1 SFT_MAX_STEPS=0 QFORMER_MODE=full SAVE_EVERY=6000
export NPROC_PER_NODE=4 GRAD_ACCUMULATION=4
export SFT_LR="${SFT_LR:-2.5e-6}" PROJECTOR_LR="${PROJECTOR_LR:-2.5e-6}" QFORMER_LR="${QFORMER_LR:-5e-6}"
export SFT_SEED="${SFT_SEED:-43}" SFT_WARMUP_RATIO="${SFT_WARMUP_RATIO:-0.03}"
export MASTER_PORT="${MASTER_PORT:-29687}"
models=(checkpoint-006000 checkpoint-012000 checkpoint-018000 checkpoint-024000 final)
report_args=(--data-dir "$FULL_QFORMER_DATA_DIR" --train-dir "$FULL_QFORMER_TRAIN_DIR"
  --eval-dir "$FULL_QFORMER_EVAL_DIR" --parent-dir "$PARENT_DIR" --parent-eval-dir "$PARENT_EVAL_DIR")
report() { "$PYTHON_BIN" scripts/report_full_qformer_continue.py "${report_args[@]}" --action "$1"; }
case "${1:-help}" in
  check) report check ;;
  train)
    report check
    bash scripts/run_full_qformer_sft.sh train
    ;;
  status|eval_status) report "$1" ;;
  eval)
    model="${2:?checkpoint-N or final}"; split="${3:?action or scenery}"
    if [[ "$split" != action && "$split" != scenery ]]; then
      echo "Use eval_selected for official evaluation after development selection." >&2; exit 2
    fi
    bash scripts/run_full_qformer_sft.sh eval "$model" "$split"
    ;;
  eval_dev)
    report check
    for model in "${models[@]}"; do
      for split in action scenery; do bash scripts/run_full_qformer_sft.sh eval "$model" "$split"; done
    done
    report report
    ;;
  report) report report ;;
  eval_selected)
    report report
    selected="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_model"])' "$FULL_QFORMER_EVAL_DIR/report.json")"
    if [[ "$selected" != baseline ]]; then bash scripts/run_full_qformer_sft.sh eval "$selected" eval; fi
    report report_eval
    ;;
  *) echo "usage: $0 check|train|status|eval MODEL action/scenery|eval_dev|report|eval_selected|eval_status" >&2; exit 2 ;;
esac
