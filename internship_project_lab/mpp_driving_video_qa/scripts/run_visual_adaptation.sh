#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zjr/anaconda3/envs/mpp_llava/bin/python}"
UPSTREAM="${MPP_UPSTREAM_ROOT:-$PROJECT_ROOT/../mpp_llava_video_workspace/third_party/MPP-LLaVA}"
PERSISTENT="$PROJECT_ROOT/../mpp_llava_video_workspace/checkpoints"
CHECKPOINT_ROOT="${MPP_CHECKPOINT_ROOT:-/tmp/zjr_mpp_fast/checkpoints}"
if [[ ! -f "$CHECKPOINT_ROOT/mpp-qwen-next-sft/sharded/model.safetensors.index.json" ]]; then
  CHECKPOINT_ROOT="$PERSISTENT"
fi
VISUAL_ROOT="${MPP_VISUAL_CHECKPOINT_ROOT:-$PERSISTENT}"
BRIDGE_INIT="${MPP_BRIDGE_INIT:-$PROJECT_ROOT/checkpoints/visual_bridge_init.pt}"
DATA_DIR="${VISUAL_DATA_DIR:-$PROJECT_ROOT/datasets/processed/lingoqa_visual20k_v1}"
CACHE_DIR="${MPP_EVA_CACHE_DIR:-/tmp/zjr_mpp_fast/cache/lingoqa_5f_eva_v1}"
TEACHER_CACHE="${MPP_TEACHER_CACHE_DIR:-/tmp/zjr_mpp_fast/cache/lingoqa_5f_eva_crop_v1}"
JOINT_DIR="$PROJECT_ROOT/outputs/mpp_lingoqa_projector_lora_1epoch_v1"
SFT_DIR="${VISUAL_SFT_DIR:-$PROJECT_ROOT/outputs/mpp_lingoqa_qformer_lora20k_v1}"
OPD_DIR="${MPP_OPD_DIR:-$PROJECT_ROOT/outputs/mpp_lingoqa_opd_pilot_v1}"
PROBE_DIR="${TEACHER_PROBE_DIR:-$PROJECT_ROOT/outputs/teacher_probe_v1}"
SELECTION_DIR="${OPD_SELECTION_DIR:-$PROJECT_ROOT/outputs/opd_selection_v1}"
EVAL_DIR="${VISUAL_EVAL_DIR:-$PROJECT_ROOT/outputs/visual_adaptation_eval}"
export PYTHONPATH="$PROJECT_ROOT:$UPSTREAM${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC="${NPROC_PER_NODE:-4}"
common=(--upstream-root "$UPSTREAM" --checkpoint-root "$CHECKPOINT_ROOT"
        --visual-checkpoint-root "$VISUAL_ROOT" --bridge-init "$BRIDGE_INIT" --feature-stage eva)

weights_for() {
  case "$1" in
    joint) MODEL_ADAPTER="$JOINT_DIR/final_adapter"; MODEL_PROJECTOR="$JOINT_DIR/final_projector.pt"; MODEL_QFORMER="" ;;
    visual) MODEL_ADAPTER="$SFT_DIR/final_adapter"; MODEL_PROJECTOR="$SFT_DIR/final_projector.pt"; MODEL_QFORMER="$SFT_DIR/final_qformer_adapter.pt" ;;
    opd) MODEL_ADAPTER="$OPD_DIR/final/adapter"; MODEL_PROJECTOR="$OPD_DIR/final/projector.pt"; MODEL_QFORMER="$OPD_DIR/final/qformer_adapter.pt" ;;
    *) echo "unknown model: $1" >&2; exit 2 ;;
  esac
  weights=(--adapter "$MODEL_ADAPTER" --projector "$MODEL_PROJECTOR")
  if [[ -n "$MODEL_QFORMER" ]]; then weights+=(--qformer-adapter "$MODEL_QFORMER"); fi
}
cache_data() {
  local destination="$1" view="$2"
  shift 2
  "$PYTHON_BIN" -u -m torch.distributed.run --nproc_per_node="$NPROC" \
    --master_port="${MASTER_PORT:-29671}" scripts/cache_visual_features.py \
    --cache-dir "$destination" --view "$view" "${common[@]}" "$@"
}
score() {
  "$PYTHON_BIN" -u scripts/evaluate_answers.py --predictions "$1/predictions.jsonl" \
    --output-dir "$1/scored" --lingo-judge
}
teacher_generate() {
  local data_path="$1" destination="$2" view="$3" cache_path="$4"
  "$PYTHON_BIN" -u scripts/compare_teachers.py generate --data "$data_path" \
    --output-dir "$destination" --backend mpp --view "$view" --cache-dir "$cache_path" \
    "${common[@]}" "${weights[@]}"
  score "$destination"
}
case "${1:-help}" in
  prepare)
    "$PYTHON_BIN" -u scripts/prepare_visual_adaptation.py --output-dir "$DATA_DIR" \
      --train-size "${VISUAL_TRAIN_SIZE:-20000}" --teacher-train-size "${TEACHER_TRAIN_SIZE:-1000}"
    "$PYTHON_BIN" -u -m mpp_video_qa.visual_bridge \
      --source "$VISUAL_ROOT/ckpt/blip2/blip2_pretrained_flant5xxl.pth" --output "$BRIDGE_INIT"
    ;;
  cache)
    cache_data "$CACHE_DIR" full --data "$DATA_DIR/train.jsonl" \
      --data "$DATA_DIR/dev.jsonl" --data "$DATA_DIR/eval.jsonl"
    ;;
  cache_status)
    "$PYTHON_BIN" scripts/report_cache_progress.py --data-dir "$DATA_DIR" --cache-dir "$CACHE_DIR"
    ;;
  teacher_cache)
    cache_data "$CACHE_DIR" full --data "$DATA_DIR/teacher_probe.jsonl"
    cache_data "$TEACHER_CACHE" full_crop --data "$DATA_DIR/teacher_probe.jsonl"
    ;;
  teacher_compare)
    weights_for joint
    teacher_generate "$DATA_DIR/teacher_probe.jsonl" "$PROBE_DIR/joint" full "$CACHE_DIR"
    teacher_generate "$DATA_DIR/teacher_probe.jsonl" "$PROBE_DIR/joint_crop" full_crop "$TEACHER_CACHE"
    ;;
  teacher_strong)
    : "${STRONG_TEACHER_MODEL:?set a local HF video/image VLM model path}"
    STRONG_PYTHON="${STRONG_TEACHER_PYTHON:-/home/zjr/anaconda3/envs/chartvlm/bin/python}"
    "$STRONG_PYTHON" -u scripts/compare_teachers.py generate \
      --data "$DATA_DIR/teacher_probe.jsonl" --output-dir "$PROBE_DIR/strong" \
      --backend hf_vlm --model "$STRONG_TEACHER_MODEL" --view full
    score "$PROBE_DIR/strong"
    ;;
  teacher_report)
    reports=(joint="$PROBE_DIR/joint/scored/scored_predictions.jsonl"
             joint_crop="$PROBE_DIR/joint_crop/scored/scored_predictions.jsonl")
    if [[ -f "$PROBE_DIR/strong/scored/scored_predictions.jsonl" ]]; then
      reports+=(strong="$PROBE_DIR/strong/scored/scored_predictions.jsonl")
    fi
    "$PYTHON_BIN" scripts/compare_teachers.py report --data "$DATA_DIR/teacher_probe.jsonl" \
      --output-dir "$PROBE_DIR" --baseline joint --scored "${reports[@]}"
    ;;
  train)
    init=(--init-adapter "$JOINT_DIR/final_adapter" --projector "$JOINT_DIR/final_projector.pt")
    if [[ -n "${SFT_RESUME_FROM:-}" ]]; then init=(--resume-from "$SFT_RESUME_FROM"); fi
    train_bridge=(--train-qformer)
    if [[ "${TRAIN_QFORMER:-1}" == "0" ]]; then train_bridge=(); fi
    "$PYTHON_BIN" -u -m torch.distributed.run --nproc_per_node="$NPROC" \
      --master_port="${MASTER_PORT:-29672}" scripts/train_sft.py \
      --data "$DATA_DIR/train.jsonl" --dev-data "$DATA_DIR/dev.jsonl" \
      --cache-dir "$CACHE_DIR" --output-dir "$SFT_DIR" "${common[@]}" "${init[@]}" \
      --train-projector "${train_bridge[@]}" --qformer-r 8 --qformer-alpha 16 \
      --qformer-lr "${QFORMER_LR:-1e-5}" --lr "${SFT_LR:-5e-6}" --projector-lr "${PROJECTOR_LR:-5e-6}" \
      --epochs "${SFT_EPOCHS:-1}" --batch-size 1 --grad-accumulation "${GRAD_ACCUMULATION:-4}" \
      --gradient-checkpointing --max-dev-records 64 --save-every 200 \
      --max-steps "${SFT_MAX_STEPS:-0}"
    ;;
  eval)
    kind="${2:?specify joint|visual|opd}"; split="${3:?specify action|scenery|eval}"
    case "$split" in
      action|scenery) data_path="$DATA_DIR/dev_${split}.jsonl" ;;
      eval) data_path="$DATA_DIR/eval.jsonl" ;;
      *) exit 2 ;;
    esac
    weights_for "$kind"
    destination="$EVAL_DIR/${kind}_${split}"
    "$PYTHON_BIN" -u scripts/generate_answers.py --data "$data_path" --cache-dir "$CACHE_DIR" \
      --output-dir "$destination" "${common[@]}" "${weights[@]}" --batch-size 1 --max-new-tokens 96
    score "$destination"
    if [[ "$kind" != "joint" ]]; then
      baseline="$PROJECT_ROOT/outputs/projector_sft_eval/joint_${split}/scored/scored_predictions.jsonl"
      if [[ -f "$EVAL_DIR/joint_${split}/scored/scored_predictions.jsonl" ]]; then
        baseline="$EVAL_DIR/joint_${split}/scored/scored_predictions.jsonl"
      fi
      "$PYTHON_BIN" scripts/compare_runs.py --baseline "$baseline" \
        --candidate "$destination/scored/scored_predictions.jsonl" \
        --output "$EVAL_DIR/joint_vs_${kind}_${split}.json"
    fi
    ;;
  opd_cache)
    cache_data "$CACHE_DIR" full --data "$DATA_DIR/teacher_train_probe.jsonl"
    cache_data "$TEACHER_CACHE" full_crop --data "$DATA_DIR/teacher_train_probe.jsonl"
    ;;
  opd_select)
    weights_for "${OPD_STUDENT_KIND:-visual}"
    teacher_generate "$DATA_DIR/teacher_train_probe.jsonl" "$SELECTION_DIR/student" full "$CACHE_DIR"
    teacher_generate "$DATA_DIR/teacher_train_probe.jsonl" "$SELECTION_DIR/teacher" full_crop "$TEACHER_CACHE"
    "$PYTHON_BIN" scripts/prepare_opd_data.py --train "$DATA_DIR/train.jsonl" \
      --student-scored "$SELECTION_DIR/student/scored/scored_predictions.jsonl" \
      --teacher-scored "$SELECTION_DIR/teacher/scored/scored_predictions.jsonl" \
      --output "$DATA_DIR/opd_train.jsonl" --max-records "${OPD_TRAIN_SIZE:-1000}"
    ;;
  opd)
    weights_for "${OPD_STUDENT_KIND:-visual}"
    extra=(--init-adapter "$MODEL_ADAPTER" --projector "$MODEL_PROJECTOR"
           --teacher-adapter "$MODEL_ADAPTER" --teacher-projector "$MODEL_PROJECTOR")
    if [[ -n "$MODEL_QFORMER" ]]; then
      extra+=(--qformer-adapter "$MODEL_QFORMER" --teacher-qformer-adapter "$MODEL_QFORMER")
    fi
    if [[ "${OPD_TRAIN_QFORMER:-1}" == "1" ]]; then extra+=(--train-qformer); fi
    if [[ -n "${OPD_RESUME_FROM:-}" ]]; then extra+=(--resume-from "$OPD_RESUME_FROM"); fi
    "$PYTHON_BIN" -u -m torch.distributed.run --nproc_per_node="$NPROC" \
      --master_port="${MASTER_PORT:-29673}" scripts/train_opd.py \
      --data "$DATA_DIR/opd_train.jsonl" --cache-dir "$CACHE_DIR" --teacher-cache-dir "$TEACHER_CACHE" \
      --output-dir "$OPD_DIR" "${common[@]}" "${extra[@]}" \
      --max-steps "${OPD_MAX_STEPS:-200}" --epochs "${OPD_EPOCHS:-1}" \
      --grad-accumulation "${GRAD_ACCUMULATION:-4}" --qformer-lr "${OPD_QFORMER_LR:-2e-6}"
    ;;
  status)
    "$PYTHON_BIN" scripts/report_progress.py --output-dir "$SFT_DIR"
    ;;
  opd_status)
    "$PYTHON_BIN" scripts/report_progress.py --output-dir "$OPD_DIR"
    ;;
  *)
    echo "usage: $0 prepare|cache|cache_status|teacher_cache|teacher_compare|teacher_strong|teacher_report|train|eval MODEL SPLIT|status|opd_cache|opd_select|opd|opd_status" >&2
    exit 2
    ;;
esac

