#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
OLD_DATA=/data/pixal3d_multiview/objaverse_sparse_mv_artraj_pbr_5000_v9_select8
OLD_ROOT=/data/reconvggt_pointpose_v9_odsplit_20260712
NEW_DATA=/data/pixal3d_multiview/objaverse_sparse_mv_artraj_pbr_5000_v9_select8_ssfixed_v1_20260712
ROOT=/data/reconvggt_pointpose_v9_ssfixed_odsplit_20260712
REPAIR_SCRIPT=pixal3d_multiview/dataset_tools/repair_object_level_ss_dataset.py
SMOKE_OUT=/home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/pointpose_ss_lora_ssfixed_single_s5

if [[ "$OLD_DATA" == "$NEW_DATA" || "$OLD_ROOT" == "$ROOT" ]]; then
  echo "Refusing to use identical source/output roots" >&2
  exit 2
fi
if [[ -e "$NEW_DATA" || -e "$ROOT" ]]; then
  echo "Output root already exists. Use a new versioned path, or run the repair command manually with --resume." >&2
  echo "NEW_DATA=$NEW_DATA" >&2
  echo "ROOT=$ROOT" >&2
  exit 2
fi

$PY -m py_compile "$REPAIR_SCRIPT"
mkdir -p "$ROOT/logs"

# 1) Rebuild object-level deterministic SS latents and rewrite dataset/split manifests.
CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY -u "$REPAIR_SCRIPT" \
  --source_dataset_root "$OLD_DATA" \
  --output_dataset_root "$NEW_DATA" \
  --source_experiment_root "$OLD_ROOT" \
  --output_experiment_root "$ROOT" \
  --dataset_manifests train.json,val.json \
  --split_names train,val,holdout \
  --encoder_pretrained microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16 \
  --decoder_pretrained Stable-X/trellis-vggt-v0-2 \
  --target_mode decoder_projected \
  --surface_points 160000 \
  --voxel_resolution 64 \
  --canonical_margin 0.9 \
  --seed 42 \
  --device cuda \
  --latent_dtype float16 \
  --observation_mode symlink \
  --log_every 25 \
  2>&1 | tee "$ROOT/logs/31_ss_repair.log"

$PY - <<PY
import json
from pathlib import Path
p = Path("$ROOT/ss_repair_report.json")
r = json.loads(p.read_text())
assert r["passed"], r["same_object_consistency_failures"][:10]
assert r["target_mode"] == "decoder_projected"
for path in [
    Path("$NEW_DATA/train.json"),
    Path("$NEW_DATA/val.json"),
    Path("$ROOT/manifests/train.json"),
    Path("$ROOT/manifests/val.json"),
    Path("$ROOT/manifests/holdout.json"),
]:
    assert path.is_file(), path
print({k: r[k] for k in ("passed", "object_count", "sample_count", "target_mode")})
PY

# 2) Rebuild priors. They depend on target_coords and must not be reused.
for SPLIT in train val holdout; do
  $PY -u trellis_point_prior_mv/build_point_prior_dataset.py \
    --source_manifest "$ROOT/manifests/${SPLIT}.json" \
    --output_dir "$ROOT/priors/${SPLIT}" \
    --indices all \
    --max_frames 8 \
    --seed 42 \
    --grid_transform pixal3d_rotation \
    --num_prior_views_choices 2,4,8 \
    --point_count_choices 50,100,300,800,1500 \
    --min_support 1 \
    --min_support_ratio 0.45 \
    --dropout_min 0.0 \
    --dropout_max 0.65 \
    --coord_jitter 1 \
    --outlier_ratio 0.03 \
    --front_depth_epsilon 0.02 \
    2>&1 | tee "$ROOT/logs/32_3_prior_${SPLIT}.log"
done

# 3) 32.4 physical cache.
for SPLIT in train val holdout; do
  $PY -u reconvggt_ar_adapter_a/build_pointpose_ss_cache.py \
    --source_manifest "$ROOT/manifests/${SPLIT}.json" \
    --prior_manifest "$ROOT/priors/${SPLIT}/manifest.json" \
    --output_dir "$ROOT/cache/${SPLIT}" \
    --indices all \
    --log_every 50 \
    2>&1 | tee "$ROOT/logs/32_4_cache_${SPLIT}.log"
done

# 4) 32.4a independent audit, including strict decoder round-trip.
for SPLIT in train val holdout; do
  CUDA_VISIBLE_DEVICES=0 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn \
  SPCONV_ALGO=native \
  $PY -u reconvggt_ar_adapter_a/audit_pointpose_ss_cache.py \
    --cache_manifest "$ROOT/cache/${SPLIT}/manifest.json" \
    --output_dir "$ROOT/cache/${SPLIT}/independent_audit" \
    --check_decoder \
    --pretrained Stable-X/trellis-vggt-v0-2 \
    --device cuda \
    --decoder_min_iou 0.99 \
    2>&1 | tee "$ROOT/logs/32_4a_audit_${SPLIT}.log"
done

$PY - <<PY
import json
from pathlib import Path
for split in ("train", "val", "holdout"):
    path = Path("$ROOT/cache") / split / "independent_audit" / "report.json"
    if not path.exists():
        candidates = list(path.parent.glob("*.json"))
        if len(candidates) == 1:
            path = candidates[0]
    report = json.loads(path.read_text())
    assert report.get("passed") is True, (split, path, report.get("summary"))
    print(split, "PASS", path)
PY

# 5) 32.4b object-balanced overfit64 manifest.
$PY -u reconvggt_ar_adapter_a/build_object_subset_manifest.py \
  --source_manifest "$ROOT/cache/train/manifest.json" \
  --output_manifest "$ROOT/cache/train/overfit64.json" \
  --object_count 64 \
  --sequences_per_object 1 \
  --seed 42 \
  2>&1 | tee "$ROOT/logs/32_4b_overfit64.log"

$PY - <<PY
import json
from pathlib import Path
p = Path("$ROOT/cache/train/overfit64.json")
r = json.loads(p.read_text())
samples = r["samples"] if isinstance(r, dict) else r
objects = {str(x.get("object_uid") or str(x["uid"]).split("_seq", 1)[0]) for x in samples}
assert len(samples) == 64, len(samples)
assert len(objects) == 64, len(objects)
print({"sample_count": len(samples), "unique_object_count": len(objects)})
PY

# 6) 32.5 single-GPU, five-optimizer-step smoke.
rm -rf "$SMOKE_OUT"
CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY -u reconvggt_ar_adapter_a/train_pointpose_ss_lora.py \
  --cache_manifest "$ROOT/cache/train/manifest.json" \
  --output_dir "$SMOKE_OUT" \
  --pretrained Stable-X/trellis-vggt-v0-2 \
  --indices 0-7 \
  --max_steps 5 \
  --save_every 5 \
  --log_every 1 \
  --num_workers 0 \
  --seed 42 \
  --lr 2e-5 \
  --grad_accum 1 \
  --grad_clip 1.0 \
  --amp_dtype fp16 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --physical_hidden_dim 256 \
  --physical_heads 8 \
  --bridge_train_last_blocks 0 \
  --gradient_checkpointing \
  --t_schedule uniform \
  --drop_all_prob 0.1 \
  --physical_drop_prob 0.1 \
  --occupancy_weight 0.0 \
  2>&1 | tee "$ROOT/logs/32_5_smoke.log"

echo "Pipeline completed."
echo "Repaired dataset: $NEW_DATA"
echo "Experiment root:  $ROOT"
echo "Smoke output:     $SMOKE_OUT"