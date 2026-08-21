# Bunny Racer 人工审查工具

这个目录把 `pose_point_depth_mv/bunny/` 的五张实拍缩略图和参考扫描 Mesh
冻结成一个独立人工审查协议。它的目标是让 Pixal3D、ReconViaGen stock 和不断变化的
训练模型输出相同的产物契约：

```text
pose_point_depth_mv/bunny/outputs/bunny_human_review_20260727_v1/
methods/<method_id>/
  result.json
  mesh*.obj/glb

comparison/<review_id>/
  normal_side_by_side.mp4
  normal_contact_sheet.png
  methods/<method_id>/normal_*
  index.html
  report.json
```

## 输入语义

- Pixal3D 固定只使用 `single_view_index` 一张 RGBA；
- ReconViaGen stock 使用五张 RGBA，由 VGGT 做未标定多视图聚合；
- 训练模型通过原生 ReconViaGen LoRA loader、JSON 命令适配器或 Mesh 注册器接入；
- 参考 `model.obj` 只用于人工审查，默认禁止命令适配器将它作为模型输入。

五张原始图没有相机内外参、深度、稀疏点云或 TM2W。当前 Direct-SS/Direct-SLAT Full
需要这些 AR 物理输入，因此不能把原始 Bunny RGB-only 输入伪装成完整 Full。若使用
模拟传感器/参考 Mesh 采样点云，适配器必须显式设置
`reference_mesh_declared_as_model_input=true`，运行时也必须额外确认；这种结果应命名为
oracle/sensor-simulation，不能与照片输入混为一谈。

## 可替换训练后端

`trained_adapter_template.json` 的 `command` 是 argv 数组，不经过 shell。运行时会生成：

```text
methods/<method_id>/adapter_context.json
```

当前推理代码从该文件读取冻结图片、mask、协议哈希和输出目录，最后把 Mesh 写到
`expected_mesh`。训练代码、checkpoint 格式和参数可以变化；统一比较层只验证 Mesh、
代码、checkpoint 与协议哈希。

也可以先用任何当前推理入口生成 GLB/OBJ，再通过 `trained_adapter register` 复制并绑定。
这条路径适合推理代码正在频繁变化的阶段。

`direct_slat_cache_adapter_template.json` 进一步给出了当前
`export_direct_slat_mesh_pairs` 的具体适配方式。它要求 Bunny 已经拥有 materialized
Direct-SLAT cache；原始五张照片本身不满足这个条件。

## 显示规则

统一渲染只改变内存中的显示副本：每个 Mesh 独立 bbox 居中并各向同性缩放，然后使用
同一 yaw 转台、固定 pitch/FOV 渲染法线图。原始 Mesh 文件不会被改动。这个显示规则
适合比较完整度、薄结构、洞、组件、左右/背面形状，不表示世界尺度或 TM2W 一致。

完整可执行命令见：

```text
pose_point_depth_mv/bunny/Bunny人工审查三模型重建命令_20260727.txt
```

如果只需要恢复原版 Pixal3D 与 ReconViaGen stock 的几何结果并立即生成三列法线对比，
使用：

```text
pose_point_depth_mv/bunny/Bunny_Pixal3D_ReconViaGen恢复与渲染命令_20260728.txt
```

如果已经为 Bunny 准备了带相机、稀疏点/TM2W 来源声明的 materialized
Direct-SLAT cache，可运行旧版 Direct-SLAT step800 后半段，并生成 Reference /
Pixal3D / ReconViaGen stock / Direct Full 四列法线对比：

```text
pose_point_depth_mv/bunny/Bunny已有DirectSLATCache_旧版Step800四模型渲染命令_20260728.txt
```

该命令不是 RGB-to-Mesh 完整重建入口，也不会从五张未标定缩略图伪造 Direct Full
输入。若 cache 的物理输入由 reference Mesh 模拟，必须选择脚本中的
`reference_oracle` 模式并显式批准，产物也会使用 oracle 方法名。
