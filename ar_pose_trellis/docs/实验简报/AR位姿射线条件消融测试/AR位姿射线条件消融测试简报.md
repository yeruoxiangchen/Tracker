# AR 位姿射线条件消融测试简报

## 当前网络结构

当前 `ar_pose_trellis` 的目标不是直接输出最终优化 mesh，而是验证：在 TRELLIS sparse structure 生成阶段，加入手机 AR 相机位姿是否能改善第一步 coarse sparse coords。后续 mesh 对齐、尺度估计、M2W 估计和几何细化仍然由 `CoarseModel` 流程处理。

当前训练和测试只评价 sparse structure / coarse coords，不评价 SLAT、mesh 纹理、mesh 几何细节或最终位姿优化精度。

整体结构如下：

| 模块 | 状态 | 作用 |
|---|---|---|
| `image_cond_model` | 冻结 | TRELLIS 原始 DINO 图像编码器，提取多视图 patch feature |
| `ss_encoder` | 冻结 | 将 target sparse occupancy 编码成 TRELLIS sparse structure latent，作为训练监督 |
| `sparse_structure_flow_model` | LoRA 微调 | TRELLIS sparse structure flow，学习从噪声生成 sparse latent |
| `ARDinoRayCond` | 训练 | 新增条件网络，将 DINO 图像特征和 AR pose/ray/mask 特征融合成 sparse flow 的 condition |

训练时的监督来自合成数据的 `target_coords`。代码先把 `target_coords` 填成 `64^3` occupancy，再经过冻结的 `ss_encoder` 得到 target latent；训练目标是让 sparse structure flow 在 condition 约束下预测该 latent 的 flow matching 速度场。

## AR Pose Ray Condition 包含的信息

当前没有使用 VGGT。`ARDinoRayCond` 的输入来自图像、mask、相机内参和 AR 相机外参，核心是把每个 DINO patch 对应的相机射线和相机姿态编码进去。

每个 view、每个 patch 的 pose/ray feature 是 16 维：

| 特征 | 维度 | 含义 |
|---|---:|---|
| `rays_world` | 3 | 该 patch 像素中心经过相机内参反投影、再由相机旋转变换后的世界/参考坐标系射线方向 |
| `origins` | 3 | 相机中心位置，默认转成 reference-relative pose 后再归一化 |
| `right` | 3 | 相机坐标系 x 轴方向 |
| `up` | 3 | 相机坐标系 y 轴方向 |
| `forward` | 3 | 相机坐标系 z/forward 方向，受 `camera_forward_sign` 控制 |
| `mask_patch` | 1 | mask 下采样到 DINO patch 网格后的前景占比 |

相机位姿默认不是直接使用绝对世界坐标，而是使用 reference-relative pose：

```text
c2w_i -> inv(c2w_ref) @ c2w_i
```

这样做的目的是让 condition 表达多视图相对运动，而不是让网络死记某个世界坐标系下的绝对相机位置。

`ARDinoRayCond` 的融合方式是：

```text
DINO patch feature -> Linear(1024, 1024)
pose/ray feature  -> Linear(16, 1024) -> SiLU -> Linear(1024, 1024)

context = image_context + pose_scale * pose_context
```

然后用 4096 个可学习的 multiview condition tokens 通过 4 层 cross-attention block 从 `context` 中聚合多视图信息，输出给 TRELLIS sparse structure flow。

## 本次消融实验

本次完整实验路径：

```text
/home/zjr/Tracker/ar_pose_trellis/runs/pose_condition_experiments/pose_condition_reuse_image_pose_003_gpu1
```

评测输出路径：

```text
/home/zjr/Tracker/ar_pose_trellis/benchmark_outputs/pose_condition_experiments/pose_condition_reuse_image_pose_003_gpu1
```

本次共有 48 个任务：

- `image_pose`：复用已有 checkpoint，不重训
- `image_only`：重新训练完成
- `pose_only`：重新训练完成
- 所有训练和评测任务均完成，失败数为 0

三个核心消融模型：

| 变体 | 输入条件 |
|---|---|
| `image_pose` | DINO 图像特征 + AR pose/ray/mask condition |
| `image_only` | 只使用 DINO 图像特征，不使用 AR pose/ray/mask |
| `pose_only` | 只使用 AR pose/ray/mask，不使用 DINO 图像特征 |

## Raw Sparse Coords 结果

不加 visual hull logits prior 时，三组模型在 `correct` pose 下的核心结果如下：

| 模型 | IoU | F2 | Chamfer | pred coords |
|---|---:|---:|---:|---:|
| `image_pose` | 0.0255 | 0.1650 | 7.354 | 3719 |
| `image_only` | 0.0302 | 0.1717 | 7.256 | 5567 |
| `pose_only` | 0.0250 | 0.1563 | 7.088 | 3740 |

结论：当前 learned AR pose ray condition 没有超过纯 DINO 图像条件。`image_only` 反而是 raw sparse coords 上最强的 baseline，说明当前的 pose/ray token 融合方式没有稳定改善 coarse sparse structure。

## Pose 扰动敏感性

如果 AR pose condition 起到了强作用，`correct` 应该明显优于 `identity / shuffle / noise`。但本次结果中差距很小。

`image_pose raw`：

| mode | IoU | F2 | Chamfer |
|---|---:|---:|---:|
| `correct` | 0.0255 | 0.1650 | 7.354 |
| `identity` | 0.0232 | 0.1579 | 7.403 |
| `shuffle` | 0.0242 | 0.1591 | 7.432 |
| `noise` | 0.0254 | 0.1644 | 7.337 |

`pose_only raw`：

| mode | IoU | F2 | Chamfer |
|---|---:|---:|---:|
| `correct` | 0.0250 | 0.1563 | 7.088 |
| `identity` | 0.0249 | 0.1542 | 7.161 |
| `shuffle` | 0.0248 | 0.1544 | 7.157 |
| `noise` | 0.0249 | 0.1554 | 7.092 |

结论：当前网络对 pose 的敏感性不足。尤其 `pose_only` 中 correct / identity / shuffle / noise 几乎没有拉开，说明仅靠 learned ray condition 很难让 sparse structure flow 自动学出强几何约束。

## Visual Hull Logits Prior 的结果

加入 visual hull logits prior 后，正确位姿和错误位姿的差距明显扩大，尤其 `weight=80`：

| 模型 | mode | IoU | F2 | Chamfer | pred coords |
|---|---|---:|---:|---:|---:|
| `image_pose + vh_w80` | `correct` | 0.0276 | 0.1662 | 7.279 | 3970 |
| `image_pose + vh_w80` | `identity` | 0.0131 | 0.1363 | 7.617 | 1710 |
| `pose_only + vh_w80` | `correct` | 0.0273 | 0.1630 | 7.000 | 3922 |
| `pose_only + vh_w80` | `identity` | 0.0145 | 0.1316 | 7.471 | 2097 |
| `image_only + vh_w80` | `correct` | 0.0318 | 0.1750 | 7.205 | 5938 |
| `image_only + vh_w80` | `identity` | 0.0227 | 0.1519 | 7.441 | 3583 |

这说明 AR pose 本身并不是没有价值。正确的相机位姿和 mask 能通过投影形成明确的 visual hull / voxel projection prior，并且这个显式几何 prior 能有效压低明显错误 pose 的结果。

## 当前结论

1. 当前 `ARDinoRayCond` 这种 learned pose/ray token 融合方式还没有证明能改善 TRELLIS sparse coords。
2. `image_only` 是目前更强的 learned baseline，说明 DINO 图像条件仍然主导 sparse structure 生成。
3. `pose_only` 几乎无法区分 correct / identity / shuffle / noise，说明单靠 AR ray token 学 sparse coords 的能力不足。
4. visual hull logits prior 明显有效，说明 AR pose + mask 的显式几何投影约束值得继续做。
5. 下一步不应继续原样加长训练当前结构，而应把 visual hull / voxel projection prior 作为显式 sparse logits prior 或额外几何通道接入，再比较：
   - `image_only`
   - `image_only + visual_hull prior`
   - `image_pose token only`
   - `image_pose token + visual_hull prior`

一句话总结：当前实验没有证明 learned AR pose ray condition 本身有效，但证明了 AR pose + mask 的显式投影几何先验有效；后续创新点应从“可解释的几何 prior 接入 sparse structure 生成”继续推进。

## 补充结论：AR Pose Ray 与 Visual Hull 的边界

后续在 `GOOD_MESH_TEST` 上又测试了两类更直接的几何接入方式：

1. 将 visual hull 作为 logits prior 加到 sparse structure logits 上。
2. 直接用 `mask + AR pose` 生成 visual hull surface coords，跳过当前 learned sparse network，再把 coords 送入 TRELLIS SLAT flow。

直接 visual hull coords 测试输出路径：

```text
/home/zjr/Tracker/ar_pose_trellis/outputs/good_mesh_tests/direct_visual_hull_coords_sweep
```

该测试可以跑通，说明 `visual_hull coords -> SLAT` 这条工程路径是可行的；但输出 mesh 仍然明显破碎、片状、薄壳化，没有得到稳定的粗 mesh。也就是说，问题不只是当前 sparse network 学得不稳定。

当前更准确的结论是：

1. AR pose ray condition 有相机位姿和射线信息，但它只是 learned token condition，没有直接绑定到每个 3D voxel / sparse coords 的局部图像证据上。
2. visual hull 可以提供显式几何候选区域，但它只有 silhouette 包络约束，不能提供凹陷、背面、局部表面和纹理信息。
3. 直接把 visual hull coords 交给原始 TRELLIS SLAT flow 不够，因为 SLAT 没有针对这种 AR visual hull coords 分布训练，也没有在每个 coords 上看到对应的多视图图像特征。
4. 因此，AR pose 和 visual hull 不是没用，而是当前接入方式太弱；后续应该把它们转成 per-voxel / per-sparse-coord 的多视图投影图像特征，而不是只作为全局 pose token 或简单 coords 约束。

下一步更合理的方向是 Pixal3D 风格的投影条件：

```text
visual hull / AR pose 给候选 sparse coords
+
将每个 coords 投影到多视图图像
+
采样 DINO/RGB/mask 特征
+
把 per-coord projected features 接入 SLAT 或 sparse/SLAT adapter
```

这比继续只调 `pose_scale`、visual hull 权重或 sparse threshold 更有意义。

## 后续 Pixal3D 改造的代码落点

不建议直接在 `Pixal3D/` 源码里改主流程。`Pixal3D/` 应该保留为参考实现，主要用来借鉴它的 3D grid/image feature projection、project attention 和 sparse projected feature 的接入方式。

后续改造应放在：

```text
/home/zjr/Tracker/ar_pose_trellis
```

理由：

1. 当前目标是服务 `/home/zjr/Tracker/CoarseModel/connect/server.py` 接收的手机 AR 图像、mask、相机位姿，并生成一个 coarse mesh；这个工程入口已经在 `ar_pose_trellis`。
2. 现有训练、测试、GOOD_MESH_TEST、benchmark、visual hull prior、direct visual hull coords 都已经在 `ar_pose_trellis` 下，继续放这里便于对比实验。
3. `Pixal3D` 的默认相机/物体规范坐标假设和当前 AR 输入不完全一致，直接改 Pixal3D 会把参考代码和实验代码混在一起，后续难维护。
4. 更合适的做法是在 `ar_pose_trellis` 下新建 Pixal3D-style 模块，例如：

```text
ar_pose_trellis/projected_condition.py
ar_pose_trellis/pixal3d_style_pipeline.py
ar_pose_trellis/train_projected_slat.py
ar_pose_trellis/scripts/run_good_mesh_projected_condition.sh
```

也就是说：`Pixal3D/` 作为参考，不直接改；核心实验和后续可接入 server 的代码继续放在 `ar_pose_trellis/`。

## 为什么当前 AR Pose Ray 约束较弱

当前 AR Pose Ray condition 确实已经接入 sparse structure flow，但它的约束形式偏弱。原因不是相机位姿没有价值，而是当前接入方式只提供了“相机射线描述”，没有把几何关系转成每个 3D 位置上的图像证据。

当前 AR Pose Ray condition 做的是：

```text
每个图像 patch 的 ray direction / camera origin / right-up-forward / mask_patch
-> MLP
-> 与 DINO patch feature 相加
-> cross-attention 聚合成 condition tokens
-> 输入 sparse structure flow
```

它告诉模型的是：

```text
这些图像 patch 来自这些相机射线
```

但它没有直接告诉模型：

```text
某个 sparse coord / voxel 投影到哪几张图的哪些 patch
这些 patch 是否在 mask 内
这些 patch 的 DINO/RGB/mask 特征是什么
这些多视图 patch 是否在看同一个 3D 点
这个 3D 点的局部几何和纹理应该是什么
```

因此它是 soft condition。模型可以利用它，也可以在训练中主要依赖 DINO 图像先验而弱化甚至忽略 pose/ray token。本次 `correct / identity / shuffle / noise` 差距较小，就说明当前网络对 pose 的敏感性不足。

## 为什么 ReconViaGen 的 VGGT Tokens 更强

ReconViaGen 中的 VGGT `aggregated_tokens_list` 不是普通相机位姿 token，也不是简单的 ray encoding。它来自已经训练好的 VGGT aggregator，本身就包含多视图几何推理后的中间表示。

VGGT aggregated tokens 通常隐含了：

```text
多视图图像匹配
相机相对关系
深度线索
point map / 点云线索
track / correspondence 线索
跨视图一致性
局部图像纹理和语义
```

ReconViaGen 还同时把 VGGT tokens 接入两个阶段：

```text
VGGT tokens + DINO image cond -> sparse structure condition
VGGT tokens + DINO image cond -> SLAT condition
```

也就是说，VGGT 不只影响 coarse coords，也影响 coords 上的 SLAT latent 生成。当前 `ar_pose_trellis` 的 AR Pose Ray 主要影响 sparse structure，而 SLAT 仍然基本沿用 TRELLIS 原始 image condition，所以即使 sparse coords 被 pose 或 visual hull 稍微约束，SLAT 阶段仍然缺少几何对齐的局部图像证据。

两者的本质区别可以概括为：

| 方法 | 输入信息 | 约束强度 |
|---|---|---|
| AR Pose Ray | 相机矩阵、ray direction、mask patch 比例 | 几何参数级，软约束 |
| Visual Hull | mask + pose 投影得到候选 voxel | silhouette 包络级，中等约束 |
| VGGT Tokens | 多视图图像经过几何预训练模型聚合后的 dense tokens | 图像匹配 + 隐式几何级，强约束 |

因此，当前实验不能说明 AR pose 没价值，只能说明“把 AR pose 写成 ray token 后直接喂给 sparse flow”不够。若希望 AR pose 接近 VGGT 的约束效果，需要把它变成 per-voxel / per-sparse-coord 的投影图像特征：

```text
sparse coords / visual hull coords
-> 用 AR pose 投影到多视图图像
-> 采样 DINO/RGB/mask feature
-> 聚合成 per-coord projected feature
-> 接入 sparse flow 和 SLAT flow
```

这也是后续转向 Pixal3D-style projected condition 的主要原因。

---

以下保留原始 `稀疏位姿消融测试简报.md` 内容，作为历史测试记录。

# 稀疏位姿消融测试简报

## 测试目的

验证 AR camera pose 是否能改善 TRELLIS sparse structure 的粗结构 coords 生成。

本测试只评价 sparse coords，不评价 SLAT、mesh 几何细节或纹理质量。

## 使用数据

- 数据集：Objaverse 合成 meshrgb 数据
- 数据目录：`/data/ar_pose_trellis/objaverse_pose_1000_meshrgb_s2`
- 测试集清单：`/home/zjr/Tracker/ar_pose_trellis/outputs/benchmarks/objaverse_meshrgb_val_testsets.json`
- 测试样本数：8 个 validation case
- 每个 case 测 4 组 pose ablation，共 32 条结果
- checkpoint：`/home/zjr/Tracker/ar_pose_trellis/checkpoints/sparse_image_pose_meshrgb_s2_e4.ckpt`

## 四组输入含义

| mode | 含义 |
|---|---|
| `correct` | 正确图像/mask + 正确相机位姿 |
| `identity` | 正确图像/mask + 所有相机位姿替换成单位矩阵 |
| `shuffle` | 正确图像/mask + 同一序列内相机位姿随机打乱 |
| `noise` | 正确图像/mask + 正确位姿上加随机旋转和平移扰动 |

## 指标含义

| 指标 | 含义 |
|---|---|
| `IoU` | 预测 sparse coords 与 target coords 的 voxel 交并比，越高越好 |
| `F2` | 2 voxel 距离阈值下的 F-score，越高越好 |
| `Chamfer` | 预测 coords 与 target coords 的平均最近距离，越低越好 |

## 不加 visual hull prior

报告路径：

`/home/zjr/Tracker/ar_pose_trellis/outputs/benchmarks/sparse_pose_ablation_raw/sparse_pose_ablation_report.json`

| mode | pred coords | IoU | F2 | Chamfer |
|---|---:|---:|---:|---:|
| `correct` | 3719.0 | 0.02554 | 0.16497 | 7.35450 |
| `identity` | 3669.2 | 0.02318 | 0.15794 | 7.40288 |
| `shuffle` | 3777.1 | 0.02418 | 0.15908 | 7.43203 |
| `noise` | 3788.2 | 0.02541 | 0.16438 | 7.33662 |

## 加 visual hull logits prior

设置：`--visual_hull_prior_weight 40`

报告路径：

`/home/zjr/Tracker/ar_pose_trellis/outputs/benchmarks/sparse_pose_ablation_visual_hull_w40/sparse_pose_ablation_report.json`

| mode | pred coords | IoU | F2 | Chamfer |
|---|---:|---:|---:|---:|
| `correct` | 3798.2 | 0.02597 | 0.16495 | 7.33569 |
| `identity` | 3234.4 | 0.02091 | 0.15214 | 7.48224 |
| `shuffle` | 3834.5 | 0.02410 | 0.15750 | 7.44230 |
| `noise` | 3756.8 | 0.02498 | 0.16164 | 7.35536 |

## 主要结论

1. 不加 visual hull prior 时，`correct` 只比 `identity/shuffle/noise` 略好，说明当前模型主要还是靠图像先验，pose condition 的区分度较弱。
2. 加 visual hull logits prior 后，`identity` 明显变差，`correct - identity` 的差距变大：
   - IoU gap：`0.00236 -> 0.00505`
   - F2 gap：`0.00704 -> 0.01281`
   - Chamfer gap：`-0.04838 -> -0.14655`
3. `correct` 仍然没有明显拉开 `shuffle/noise`，说明 visual hull prior 是有效但较弱的几何约束，只能排除明显错误 pose，不能稳定识别图像和 pose 是否一一严格对应。

一句话总结：visual hull logits prior 有助于压低明显错误的相机位姿，但当前还不足以证明 AR pose condition 已经强力改善 sparse coords。

## 数据如何得到

不加 visual hull prior 的等价复现命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
/home/zjr/anaconda3/envs/reconviagen/bin/python \
  ar_pose_trellis/benchmark/evaluate_sparse_pose_ablation.py \
  --testsets /home/zjr/Tracker/ar_pose_trellis/outputs/benchmarks/objaverse_meshrgb_val_testsets.json \
  --checkpoint /home/zjr/Tracker/ar_pose_trellis/checkpoints/sparse_image_pose_meshrgb_s2_e4.ckpt \
  --output_root /home/zjr/Tracker/ar_pose_trellis/outputs/benchmarks/sparse_pose_ablation_raw \
  --max_frames 8 \
  --ss_steps 12 \
  --ss_min_coords 0 \
  --cond_fp16 \
  --inprocess \
  --continue_on_error
```

加 visual hull logits prior 的命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
/home/zjr/anaconda3/envs/reconviagen/bin/python \
  ar_pose_trellis/benchmark/evaluate_sparse_pose_ablation.py \
  --testsets /home/zjr/Tracker/ar_pose_trellis/outputs/benchmarks/objaverse_meshrgb_val_testsets.json \
  --checkpoint /home/zjr/Tracker/ar_pose_trellis/checkpoints/sparse_image_pose_meshrgb_s2_e4.ckpt \
  --output_root /home/zjr/Tracker/ar_pose_trellis/outputs/benchmarks/sparse_pose_ablation_visual_hull_w40 \
  --max_frames 8 \
  --ss_steps 12 \
  --ss_min_coords 0 \
  --cond_fp16 \
  --visual_hull_prior_weight 40 \
  --inprocess \
  --continue_on_error
```
