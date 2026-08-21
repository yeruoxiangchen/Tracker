# Dataset Download Helpers

Canonical implementations have moved to
`pose_point_depth_mv/dataset_tools/`. The Python files left in this directory
are compatibility entrypoints for historical commands.

These scripts keep large training data under `/data` instead of the Tracker
workspace.

## Objaverse

Fetch only metadata/UIDs:

```bash
/home/zjr/anaconda3/envs/reconviagen/bin/python \
  pose_point_depth_mv/dataset_tools/download_objaverse.py \
  --output-root /data/Objaverse \
  --metadata-only
```

Smoke-test a small download:

```bash
/home/zjr/anaconda3/envs/reconviagen/bin/python \
  pose_point_depth_mv/dataset_tools/download_objaverse.py \
  --output-root /data/Objaverse \
  --max-objects 10 \
  --download-processes 4
```

Download a selected UID file:

```bash
/home/zjr/anaconda3/envs/reconviagen/bin/python \
  pose_point_depth_mv/dataset_tools/download_objaverse.py \
  --output-root /data/Objaverse \
  --uids-file /data/Objaverse/my_uids.txt \
  --download-processes 8
```

## OmniObject3D

Official OpenXLab repo: `omniobject3d/OmniObject3D-New`.

OpenXLab requires login before listing or downloading this dataset:

```bash
/home/zjr/anaconda3/envs/reconviagen/bin/openxlab login
```

Alternatively, export `OPENXLAB_AK` and `OPENXLAB_SK` before running the helper.

List files:

```bash
/home/zjr/anaconda3/envs/reconviagen/bin/python \
  pose_point_depth_mv/dataset_tools/download_omniobject3d.py list
```

Download a subset, for example point clouds:

```bash
/home/zjr/anaconda3/envs/reconviagen/bin/python \
  pose_point_depth_mv/dataset_tools/download_omniobject3d.py download \
  --source-path /raw/point_clouds/ply_files \
  --target-path /data/OmniObject3D/raw
```

Download the whole compressed dataset:

```bash
/home/zjr/anaconda3/envs/reconviagen/bin/python \
  pose_point_depth_mv/dataset_tools/download_omniobject3d.py get \
  --target-path /data/OmniObject3D/raw
```

The whole OmniObject3D compressed release is about 1.2TB, so prefer a subset
until the training conversion pipeline is fixed.
