# Pose-Aligned Generative Object Reconstruction

This directory is the maintained model package for the paper endpoint. It
contains the training, cache preparation, evaluation, and inference code for:

- no-VGGT Native-SS, checkpoint step 30,000, EMA weights;
- no-VGGT Native-SLat v2, checkpoint step 30,000, EMA weights;
- the frozen Stock Mesh decoder;
- 29,861 usable training objects from the frozen ProObjaverse 30K selection.

Historical experiments, generated outputs, presentation assets, and the
`manual_mesh_reconstruction` demo are intentionally not part of this package.
The complete former package is preserved without filtering at
`../archive/pose_point_depth_mv_full_20260903/`.

## Independence guarantee

Source code in this package neither imports `pose_point_depth_mv` nor reads
files from that source directory. Run the static and import checks with:

```bash
/home/zjr/anaconda3/envs/reconviagen/bin/python \
  -m pose_aligned_reconstruction.verify_package --imports
```

Some constants still start with `pose_point_depth_mv.`. They are immutable
format identifiers embedded in existing checkpoints, caches, and reports.
Changing them would break compatibility with the paper's frozen artifacts;
they are data-schema names, not Python imports or source paths.

Runtime dependencies that remain shared at the Tracker repository level are
the current TRELLIS/ReconViaGen stack plus `ar_ss_flow`,
`reconvggt_ar_adapter_a`, and `trellis_point_prior_mv`. None of those imports
the archived model package on the current code path.

## Frozen paper endpoint

Validate sizes, scientific bindings, and all SHA-256 digests (including both
roughly 0.6 GB checkpoints):

```bash
/home/zjr/anaconda3/envs/reconviagen/bin/python \
  -m pose_aligned_reconstruction.current_30k --full-hash
```

The canonical paths and hashes are defined in `current_30k.py`. The two
training runs were configured for longer schedules, but the paper endpoint is
the explicitly frozen step-30,000 snapshot from each run:

| stage | original run contract | frozen paper snapshot |
| --- | ---: | ---: |
| Native-SS | 60,000 maximum steps | 30,000 EMA |
| Native-SLat v2 | 150,000 maximum steps | 30,000 EMA |

## Inference

Prepare posed real-capture input using the three front-end commands (inspect
their required arguments with `--help`):

```bash
python -m pose_aligned_reconstruction.dataset_tools.prepare_omni_real_video_cache --help
python -m pose_aligned_reconstruction.dataset_tools.prepare_omni_real_runtime_inputs --help
python -m pose_aligned_reconstruction.dataset_tools.prepare_omni_real_dino_only_model_inputs --help
```

Then run the frozen endpoint. Checkpoint/report arguments are deliberately
bound by the wrapper and cannot drift accidentally:

```bash
python -m pose_aligned_reconstruction.infer_current \
  --model_input_manifest /path/to/model_input_manifest.json \
  --output_dir /path/to/output \
  --device cuda --seeds 42
```

Use `--object CATEGORY:OBJECT_ID` to restrict a smoke run to one object.

## Training

The maintained entry points are:

```bash
python -m pose_aligned_reconstruction.materialize_proobjaverse_official_ss_targets --help
python -m pose_aligned_reconstruction.train_proobjaverse_official_native_ss_no_vggt --help
python -m pose_aligned_reconstruction.prepare_proobjaverse_official_slat_compact_cache --help
python -m pose_aligned_reconstruction.train_proobjaverse_official_native_slat_no_vggt --help
```

The lower-level `train_native_slat_genrecon_no_vggt` entry point intentionally
retains the older direct-SLat audit contract. For the paper's official
ProObjaverse targets, use the explicit official entry point shown above; it
binds the matching official decoder-audit validator.

For exact continuation, use `--resume` with the frozen checkpoint and the same
cache identity. The checkpoint validators reject changes to model, data, and
optimization identity unless a narrowly scoped relocation/extension flag is
explicitly supplied. The original caches live under `/root/autodl-tmp` on the
training machine and are not duplicated in this source package.

The frozen Native-SS profile uses rank 8/alpha 16 LoRA, 1--8 conditioning
views, learning rates `1e-4` and `3e-5`, warmup 600, EMA 0.9995, BF16, and
gradient checkpointing. Native-SLat v2 uses the same main optimization values,
uniform timestep sampling, all-view stock cross-attention, warmup 3,000, and
gradient checkpointing. Both original runs used eight processes, gradient
accumulation 1, and seed 42.

## Tests

No third-party test runner is required:

```bash
/home/zjr/anaconda3/envs/reconviagen/bin/python -m unittest discover \
  -s pose_aligned_reconstruction -p 'test_*.py' -v
```

See `VALIDATION.md` for the post-archive test record and end-to-end GPU smoke
comparison against the previously frozen output.
