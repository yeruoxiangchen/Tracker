# ReconViaGen + CoarseModel Synthetic Evaluation

This folder is an external test harness. It does not modify the existing
CoarseModel or ReconViaGen implementation files.

Default outputs are written under:

```text
CoarseModel/reconviagen_coarse_eval/outputs
```

The pipeline has four stages:

1. `prepare`: convert one Pixal3D v9 synthetic sample into a CoarseModel-style
   dataset with `rgb`, `masks`, and `sparse/0` COLMAP text cameras.
2. `recon`: run `ReconViaGen/rebuild_mesh_from_coarse_dataset.py` to generate
   `reconstructed_object.glb`, then install it as the test object mesh.
3. `mesh_eval`: by default compare ReconViaGen mesh with the Pixal3D sparse
   target coords from `ss_latents/*.npz` after normalized Sim3 ICP alignment.
   This avoids loading large textured Objaverse GLBs during smoke tests. Use
   `--mesh_eval_reference glb` only when you explicitly want source-GLB surface
   evaluation and have enough memory headroom.
4. `coarse`: generate templates/repre for the generated mesh and run the same
   core path used by `CoarseModel/estimation_4stage_defo_fin.py`.

Smoke command:

```bash
cd /home/zjr/Tracker

GPU=1 \
SAMPLE_INDEX=0 \
bash CoarseModel/reconviagen_coarse_eval/scripts/run_smoke.sh
```

Safer staged commands:

```bash
cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
MANIFEST=/data/pixal3d_multiview/objaverse_sparse_mv_artraj_pbr_5000_v9_select8/val.json
OUT=/home/zjr/Tracker/CoarseModel/reconviagen_coarse_eval/outputs

# 1. Prepare only: no GPU model, no template generation.
$PY -u CoarseModel/reconviagen_coarse_eval/run_pipeline.py \
  --manifest "$MANIFEST" \
  --sample_index 0 \
  --output_root "$OUT" \
  --stages prepare \
  --max_frames 8

# 2. ReconViaGen only: generates and installs reconstructed_object.glb.
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
$PY -u CoarseModel/reconviagen_coarse_eval/run_pipeline.py \
  --manifest "$MANIFEST" \
  --sample_index 0 \
  --output_root "$OUT" \
  --python_bin "$PY" \
  --stages recon \
  --recon_seeds 0 \
  --mesh_simplify 0.75

# 3. Mesh quality only: default sparse target coords reference.
$PY -u CoarseModel/reconviagen_coarse_eval/run_pipeline.py \
  --manifest "$MANIFEST" \
  --sample_index 0 \
  --output_root "$OUT" \
  --stages mesh_eval \
  --mesh_eval_reference sparse \
  --mesh_eval_samples 8000 \
  --mesh_eval_icp_iters 8

# 4. CoarseModel 4-stage smoke: uses the same core functions as
# CoarseModel/estimation_4stage_defo_fin.py, outputs stay under $OUT.
CUDA_VISIBLE_DEVICES=1 PYOPENGL_PLATFORM=egl \
$PY -u CoarseModel/reconviagen_coarse_eval/run_pipeline.py \
  --manifest "$MANIFEST" \
  --sample_index 0 \
  --output_root "$OUT" \
  --stages coarse \
  --template_views 16 \
  --template_inplane 3 \
  --cluster_num 512 \
  --skip_deformation
```

For a fuller CoarseModel test, remove `--skip_deformation` and increase
`--template_views/--template_inplane`. Do that after one smoke case succeeds.

## Existing CoarseModel Datasets

The harness can also run on existing datasets such as:

```text
CoarseModel/datasets/GOOD_MESH_TEST
CoarseModel/datasets/reconviagen_20260520_021556
```

For these datasets, `prepare` creates a workspace copy with symlinked input
frames and copied `models/`, so original datasets are not modified. By default,
`recon` reuses `reconviagen_output/reconstructed_object.glb` if present instead
of rerunning ReconViaGen.

To test ReconViaGen's current mesh generation ability on the two real datasets,
use the fresh-generation script below. It adds `--force_recon_generate`, writes
new outputs under this harness, and compares the generated mesh with the
dataset's existing `models/<dataset>.obj` after normalized Sim3/ICP alignment.
That model is a practical reference for regression testing, not a strict GT.

```bash
cd /home/zjr/Tracker

GPU=1 \
MAX_FRAMES=18 \
RECON_SEEDS=0 \
bash CoarseModel/reconviagen_coarse_eval/scripts/run_real_recon_mesh_eval.sh
```

Use `MAX_FRAMES=0` to keep all frames from the dataset instead of the first 18.

For a slightly stronger candidate-selection test, use two seeds:

```bash
cd /home/zjr/Tracker

GPU=1 \
MAX_FRAMES=18 \
RECON_SEEDS=0,1 \
bash CoarseModel/reconviagen_coarse_eval/scripts/run_real_recon_mesh_eval.sh
```

Fresh-generation result locations:

```text
CoarseModel/reconviagen_coarse_eval/outputs/runs/<CASE>/pipeline_report.json
CoarseModel/reconviagen_coarse_eval/outputs/runs/<CASE>/recon_generation_report.json
CoarseModel/reconviagen_coarse_eval/outputs/reconviagen/<CASE>/<TIMESTAMP>/rebuild_report.json
CoarseModel/reconviagen_coarse_eval/outputs/reconviagen/<CASE>/<TIMESTAMP>/reconstructed_object.glb
CoarseModel/reconviagen_coarse_eval/outputs/reconviagen/<CASE>/<TIMESTAMP>/reconstructed_object.mp4
CoarseModel/reconviagen_coarse_eval/outputs/mesh_quality/<CASE>/mesh_quality_report.json
```

Set `MESH_EVAL_REFERENCE=basic` when you only want intrinsic mesh statistics
and do not want to compare against the existing dataset model.

The following commands are reuse-baseline checks. They do not test fresh mesh
generation unless `--force_recon_generate` is added.

```bash
cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
OUT=/home/zjr/Tracker/CoarseModel/reconviagen_coarse_eval/outputs

for DATASET in GOOD_MESH_TEST reconviagen_20260520_021556; do
  $PY -u CoarseModel/reconviagen_coarse_eval/run_pipeline.py \
    --dataset_dir "/home/zjr/Tracker/CoarseModel/datasets/${DATASET}" \
    --output_root "$OUT" \
    --stages prepare \
    --max_frames 8

  $PY -u CoarseModel/reconviagen_coarse_eval/run_pipeline.py \
    --dataset_dir "/home/zjr/Tracker/CoarseModel/datasets/${DATASET}" \
    --output_root "$OUT" \
    --stages recon

  $PY -u CoarseModel/reconviagen_coarse_eval/run_pipeline.py \
    --dataset_dir "/home/zjr/Tracker/CoarseModel/datasets/${DATASET}" \
    --output_root "$OUT" \
    --stages mesh_eval \
    --mesh_eval_reference basic
done
```

Then run CoarseModel smoke. This reuses existing
`CoarseModel/results/templates/v1/<DATASET>` and
`CoarseModel/results/object_repre/<DATASET>/v1` through workspace symlinks,
which avoids regenerating templates when `pyrender` is missing:

```bash
cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
OUT=/home/zjr/Tracker/CoarseModel/reconviagen_coarse_eval/outputs

CUDA_VISIBLE_DEVICES=1 PYOPENGL_PLATFORM=egl \
$PY -u CoarseModel/reconviagen_coarse_eval/run_pipeline.py \
  --dataset_dir /home/zjr/Tracker/CoarseModel/datasets/GOOD_MESH_TEST \
  --output_root "$OUT" \
  --stages coarse \
  --template_version v1 \
  --repre_version v1 \
  --reuse_existing_coarse_assets \
  --asset_link_mode symlink \
  --skip_deformation

CUDA_VISIBLE_DEVICES=1 PYOPENGL_PLATFORM=egl \
$PY -u CoarseModel/reconviagen_coarse_eval/run_pipeline.py \
  --dataset_dir /home/zjr/Tracker/CoarseModel/datasets/reconviagen_20260520_021556 \
  --output_root "$OUT" \
  --stages coarse \
  --template_version v1 \
  --repre_version v1 \
  --reuse_existing_coarse_assets \
  --asset_link_mode symlink \
  --skip_deformation
```

If you actually want to regenerate templates in this environment, install
`pyrender` first or run in an environment where `import pyrender` works.
