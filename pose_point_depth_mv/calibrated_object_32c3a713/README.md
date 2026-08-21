# Calibrated object 32c3a713

这是从冻结 Direct-SLAT validation pool 复制出的可查看资产，原始 `/data`
文件没有被移动或修改。

## 标识

- UID: `32c3a7137a8a4191ba087895d19496e8_seq001`
- object UID: `32c3a7137a8a4191ba087895d19496e8`
- split: validation
- Direct 输入视图: `0,1,3,5`
- 相机: pixel-space `K` + `c2w`

## 查看

- `object_contact_sheet.png`: 自动裁剪后的 8-view 外观总览
- `input_contact_sheet.png`: 保留原始 512x512 framing 的总览
- `thumbnails/`: 8 张原始 masked RGBA
- `masks/`: 8 张原始 mask
- `direct_inputs/`: Direct 实际使用的 4 张 RGB 副本
- `meshes/model_source.glb`: 带内嵌材质的源 GLB
- `meshes/model_canonical.obj`: 按冻结 latent normalization 导出的 OBJ
- `meshes/material.mtl`、`meshes/Material.002.png`: canonical OBJ 的材质和纹理
- `metadata.json`: 每个视图的 K/c2w、hash、统计量和 `/data` provenance

## 选择依据

该对象只按输入和 GT 属性选择，没有查看旧 step800 的结果：

- source Mesh: 5,644 vertices / 9,676 faces
- target SLAT: 6,948 points
- masked RGB std mean: 58.1462
- 16-bin unique colors: 301
- Direct view count: 4
- 与 Direct 训练池的 object UID 和 source GLB SHA-256 均零重叠

完整 Direct feature/condition/support cache 体积约 78 MB，继续保留在 `/data`；
本目录只复制与 Bunny 目录对应的可视资产，并在 `metadata.json` 中记录缓存路径。
