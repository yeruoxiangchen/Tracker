# Audit archive

This directory keeps the research audit trail that explains how the maintained
30K endpoint was selected and which earlier approaches failed. The copied audit
documents are preserved verbatim; historical paths and conclusions inside them
have not been rewritten after the package split.

## Canonical audit

The latest and most complete record is [`../审计.md`](../审计.md):

- 30,912 lines, through Chapter 192;
- source snapshot: `archive/pose_point_depth_mv_full_20260903/审计.md`;
- source SHA-256:
  `2805a9e6cd94a6c3823acfba6c810b4aecae20fd1fc28fd932ddb76f2b31f71f`.

Use this file first when checking the current paper endpoint, dataset split,
benchmark contract, failed branches, or the AR deployment path.

## Failed-experiment records

[`历史失败实验/`](历史失败实验/) contains the eight source reports gathered on
2026-07-28 plus their cross-experiment index. They cover:

1. AR SS Flow;
2. Pixal3D multiview architecture evolution;
3. the five-level Pixal3D sparse-structure tests;
4. PointsTo3D single-image conditioning and 8-view-prior scale-up;
5. ReconVGGT AR Adapter stages A/B;
6. TRELLIS point-prior PixalV9 smoke tests;
7. TRELLIS latent-concatenation sanity tests;
8. TRELLIS latentVal32 and masked-latent inpainting.

The index records every original path and SHA-256 and distinguishes engineering
success from demonstrated model benefit.

## Historical snapshots

[`历史审计快照/`](历史审计快照/) retains three independently useful checkpoints:

- `审计_截至Holdout64_20260810.md`: the shorter audit snapshot through Chapter
  121, copied from `yxc/审计.md`;
- `审计_官方30K打包快照_20260816.md`: the official-30K packaging snapshot
  through Chapter 152;
- `BUILD_AUDIT_20260816.md`: the distinct with-VGGT build/package audit.

These snapshots are not substitutes for the canonical audit. They are kept to
make the chronology and the exact state available at earlier decisions
reviewable.

## Integrity

`SHA256SUMS.txt` hashes every Markdown document in this audit archive, including
the canonical `../审计.md`. Run from `pose_aligned_reconstruction` with:

```bash
sha256sum -c audits/SHA256SUMS.txt
```
