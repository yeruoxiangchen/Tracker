#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zjr/anaconda3/envs/mpp_llava/bin/python}"
UPSTREAM="${MPP_UPSTREAM_ROOT:-$PROJECT_ROOT/../mpp_llava_video_workspace/third_party/MPP-LLaVA}"
CHECKPOINT_ROOT="${MPP_CHECKPOINT_ROOT:-/tmp/zjr_mpp_fast/checkpoints}"
if [[ ! -f "$CHECKPOINT_ROOT/mpp-qwen-next-sft/sharded/model.safetensors.index.json" ]]; then
  CHECKPOINT_ROOT="$PROJECT_ROOT/../mpp_llava_video_workspace/checkpoints"
fi
SOURCE_DIR="${OPSD_SOURCE_DIR:-$PROJECT_ROOT/datasets/processed/lingoqa_full_v2}"
DATA_DIR="${OPSD_DATA_DIR:-$PROJECT_ROOT/datasets/processed/lingoqa_answer_opsd_v1}"
CACHE_DIR="${MPP_QFORMER_CACHE_DIR:-/tmp/zjr_mpp_fast/cache/lingoqa_5f_qformer_v1}"
INIT_DIR="${OPSD_INIT_DIR:-$PROJECT_ROOT/outputs/mpp_lingoqa_projector_lora_1epoch_v1}"
PROBE_DIR="${OPSD_PROBE_DIR:-$PROJECT_ROOT/outputs/answer_opsd_probe_v1}"
TRAIN_DIR="${OPSD_TRAIN_DIR:-$PROJECT_ROOT/outputs/mpp_lingoqa_answer_opsd_v1}"
CONTROL_DIR="${OPSD_CONTROL_DIR:-$PROJECT_ROOT/outputs/mpp_lingoqa_answer_opsd_sft_control_v1}"
EVAL_DIR="${OPSD_EVAL_DIR:-$PROJECT_ROOT/outputs/answer_opsd_eval_v1}"
export PYTHONPATH="$PROJECT_ROOT:$UPSTREAM${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
common=(--upstream-root "$UPSTREAM" --checkpoint-root "$CHECKPOINT_ROOT" --feature-stage qformer)
initial=(--adapter "$INIT_DIR/final_adapter" --projector "$INIT_DIR/final_projector.pt")
score() {
  "$PYTHON_BIN" -u scripts/evaluate_answers.py --predictions "$1/predictions.jsonl" \
    --output-dir "$1/scored" --lingo-judge
}
case "${1:-help}" in
  prepare)
    "$PYTHON_BIN" scripts/prepare_answer_opsd.py --source-dir "$SOURCE_DIR" --output-dir "$DATA_DIR" \
      --size "${OPSD_PROBE_SIZE:-2000}" --seed "${SEED:-42}"
    ;;
  smoke)
    "$PYTHON_BIN" -u scripts/smoke_answer_opsd.py --data "$DATA_DIR/probe.jsonl" \
      --cache-dir "$CACHE_DIR" --upstream-root "$UPSTREAM" --checkpoint-root "$CHECKPOINT_ROOT" \
      "${initial[@]}" --output-dir "${OPSD_SMOKE_DIR:-$PROJECT_ROOT/outputs/answer_opsd_real_smoke_v1}"
    ;;
  probe_student|probe_teacher)
    context=view; role=student
    if [[ "$1" == probe_teacher ]]; then context=answer_hint; role=teacher; fi
    "$PYTHON_BIN" -u scripts/compare_teachers.py generate --backend mpp \
      --data "$DATA_DIR/probe.jsonl" --output-dir "$PROBE_DIR/$role" --cache-dir "$CACHE_DIR" \
      "${common[@]}" "${initial[@]}" --view full --teacher-context "$context" \
      --sample --num-samples "${OPSD_NUM_SAMPLES:-4}" --seed "${SEED:-42}" \
      --task-instruction-version legacy_v1 --max-new-tokens 96
    score "$PROBE_DIR/$role"
    ;;
  probe)
    bash "$0" probe_student
    bash "$0" probe_teacher
    bash "$0" select
    ;;
  select)
    "$PYTHON_BIN" scripts/prepare_opd_data.py --train "$DATA_DIR/probe.jsonl" \
      --student-scored "$PROBE_DIR/student/scored/scored_predictions.jsonl" \
      --teacher-scored "$PROBE_DIR/teacher/scored/scored_predictions.jsonl" \
      --output "$DATA_DIR/train.jsonl" --max-records "${OPSD_TRAIN_SIZE:-1000}" \
      --num-samples "${OPSD_NUM_SAMPLES:-4}" --seed "${SEED:-42}" --replay-fraction 0.3
    ;;
  train|train_control)
    mode=opd; destination="$TRAIN_DIR"
    if [[ "$1" == train_control ]]; then mode=sft_control; destination="$CONTROL_DIR"; fi
    extra=()
    if [[ -n "${OPSD_RESUME_FROM:-}" ]]; then extra+=(--resume-from "$OPSD_RESUME_FROM"); fi
    "$PYTHON_BIN" -u -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE:-4}" \
      --master_port="${MASTER_PORT:-29675}" scripts/train_opd.py \
      --data "$DATA_DIR/train.jsonl" --cache-dir "$CACHE_DIR" --teacher-cache-dir "$CACHE_DIR" \
      --output-dir "$destination" "${common[@]}" --training-mode "$mode" \
      --init-adapter "$INIT_DIR/final_adapter" --projector "$INIT_DIR/final_projector.pt" \
      --teacher-adapter "$INIT_DIR/final_adapter" --teacher-projector "$INIT_DIR/final_projector.pt" \
      --teacher-view full --teacher-context answer_hint --objective jsd \
      --lr "${OPSD_LR:-2e-6}" --projector-lr "${OPSD_PROJECTOR_LR:-2e-6}" \
      --epochs "${OPSD_EPOCHS:-1}" --max-steps "${OPSD_MAX_STEPS:-200}" \
      --grad-accumulation "${GRAD_ACCUMULATION:-4}" --sft-weight 0.2 \
      --save-every 50 --seed "${SEED:-42}" "${extra[@]}"
    ;;
  eval)
    kind="${2:?specify joint|sft_control|opsd}"; split="${3:?specify action|scenery|eval}"
    case "$split" in
      action|scenery) data_path="$DATA_DIR/dev_${split}.jsonl" ;;
      eval) data_path="$DATA_DIR/eval.jsonl" ;;
      *) exit 2 ;;
    esac
    case "$kind" in
      joint) weights=("${initial[@]}") ;;
      sft_control) weights=(--adapter "$CONTROL_DIR/final/adapter" --projector "$CONTROL_DIR/final/projector.pt") ;;
      opsd) weights=(--adapter "$TRAIN_DIR/final/adapter" --projector "$TRAIN_DIR/final/projector.pt") ;;
      *) exit 2 ;;
    esac
    destination="$EVAL_DIR/${kind}_${split}"
    "$PYTHON_BIN" -u scripts/generate_answers.py --data "$data_path" --cache-dir "$CACHE_DIR" \
      --output-dir "$destination" "${common[@]}" "${weights[@]}" \
      --task-instruction-version legacy_v1 --batch-size 1 --max-new-tokens 96
    score "$destination"
    ;;
  eval_dev)
    for kind in joint sft_control opsd; do
      for split in action scenery; do bash "$0" eval "$kind" "$split"; done
    done
    bash "$0" report
    ;;
  status|report)
    "$PYTHON_BIN" scripts/report_answer_opsd.py --action "$1" --data-dir "$DATA_DIR" \
      --probe-dir "$PROBE_DIR" --train-dir "$TRAIN_DIR" --control-dir "$CONTROL_DIR" \
      --eval-dir "$EVAL_DIR" --num-samples "${OPSD_NUM_SAMPLES:-4}"
    ;;
  *) echo "Usage: bash scripts/run_answer_opsd.sh prepare|smoke|probe|probe_student|probe_teacher|select|train|train_control|eval MODEL SPLIT|eval_dev|status|report" ;;
esac
