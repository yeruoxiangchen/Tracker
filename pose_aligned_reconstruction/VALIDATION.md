# Validation record (2026-09-03)

All checks below were run after the top-level `pose_point_depth_mv` directory
had been moved to `archive/pose_point_depth_mv_full_20260903`. Therefore a
successful import or execution could not silently fall back to the former
Python package.

## Source independence

- 75 Python files inspected by AST at validation time.
- Zero imports of `pose_point_depth_mv`.
- Zero source paths into the archived package.
- Zero symlinks into the archive.
- The SS trainer, SLat trainer, compact-cache builder, and frozen inference
  entry point all imported successfully.
- No `pose_point_depth_mv` module appeared in `sys.modules` during the import
  smoke test.
- Of the copied Python files, 66 are byte-identical to their archived source
  after normalizing only the package namespace. Three counterparts contain
  intentional changes: the new package initializer, the `run(args)` extraction
  used by the frozen inference wrapper, and compact-cache dispatch in the
  official Native-SS trainer. Five Python files are new: the frozen asset and
  inference wrappers, the verifier, the explicit official Native-SLat trainer,
  and the compact-dispatch regression test. A sixth new file regression-tests
  the official Native-SLat entry-point binding.

The retained `pose_point_depth_mv.*` strings are frozen artifact schema IDs,
not imports. Existing 30K checkpoints and manifests require those IDs.

## Automated tests

Command:

```bash
/home/zjr/anaconda3/envs/reconviagen/bin/python -m unittest discover \
  -s pose_aligned_reconstruction -p 'test_*.py' -q
```

Result: 68 tests passed. One CUDA-specific unit test was skipped in the
sandboxed CPU run. The separate GPU end-to-end test below exercised CUDA.

## Frozen artifact validation

`python -m pose_aligned_reconstruction.current_30k --full-hash` passed for all
seven bound assets. This included full reads of both 30K checkpoints:

- Native-SS: `042a1b5467b05975584aeb571dec6ffaed5096edcc6abe4aa88600c9c9506b7f`
- Native-SLat: `da8d058bbb1a917a5b91cd338d60f7d7ec15d7a5a211c86250ed514d8c0a0371`

The frozen report and cross-deployment binding also passed their scientific
identity checks.

## GPU training smoke tests

Both paper stages completed a real one-step training cycle on one RTX 3090:
model construction, cached input loading, forward, backward, gradient clipping,
AdamW update, EMA update, report generation, checkpoint save, and strict
checkpoint revalidation.

Native-SS used the official compact Dev64 cache and sampled six conditioning
views. Its flow loss was `0.0689209`; the pre-clip gradient norm was `0.250135`,
with nonzero LoRA (`0.0204200`) and condition (`0.249300`) gradients. The saved
step-1 checkpoint passed the no-VGGT Native-SS validator.

Native-SLat v2 used an official no-VGGT ProObjaverse cache with the same data
and audit schemas as the 30K run, sampled six views and 12,267 active points.
Its flow loss was `0.557998`; the pre-clip gradient norm was `1.57831`, with
nonzero LoRA (`0.223115`) and condition (`0.974791`) gradients. The saved
step-1 checkpoint passed the no-VGGT Native-SLat validator.

These tests exposed and fixed two missing entry-point adapters in the former
local snapshot: official compact-SS dataset dispatch and official-SLat decoder
audit dispatch. Both adapters now live inside `pose_aligned_reconstruction`;
neither modifies nor depends on the archived package. The full 29,861-object
training cache remains on the original training machine, so this host validated
the exact training code paths with smaller formal-schema caches rather than
starting another 30K optimization run.

## End-to-end GPU inference

The new `infer_current` entry point ran on one RTX 3090 with seed 42 for
`asparagus:omni_asparagus_005`, using an existing held-out model-input manifest.
It loaded both 30K EMA checkpoints, sampled Native-SS, sampled Native-SLat, and
decoded a mesh successfully.

- Native-SS coordinates: 1,336, exactly equal to the historical `.npz` array.
- Mesh: watertight, winding-consistent, one component, no boundary or
  non-manifold edges.
- New/old vertex counts: 32,580 / 32,528.
- New/old face counts: 65,156 / 65,052.
- Symmetric nearest-vertex mean distance: `8.56e-5` of object-bbox diagonal.
- 95th-percentile nearest-vertex distance: `3.46e-4` in the normalized object
  frame.
- Vertex Hausdorff distance: `0.00341` of object-bbox diagonal.

The OBJ byte hash is not identical because the sparse CUDA/mesh decoding path
is floating-point nondeterministic. The discrete SS result, checkpoint hashes,
sampler hash, seed, topology checks, and near-identical geometry all match the
frozen execution contract.

Smoke output was written to
`/tmp/pose_aligned_reconstruction_gpu_smoke_20260903`; it is diagnostic and is
not part of the source package.
