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

---

以下保留原始 `稀疏位姿消融测试简报.md` 内容，作为历史测试记录。

# 稀疏位姿消融测试简报

## 测试目的

验证 AR camera pose 是否能改善 TRELLIS sparse structure 的粗结构 coords 生成。

本测试只评价 sparse coords，不评价 SLAT、mesh 几何细节或纹理质量。

## 使用数据

- 数据集：Objaverse 合成 meshrgb 数据
- 数据目录：`/data/ar_pose_trellis/objaverse_pose_1000_meshrgb_s2`
- 测试集清单：`/home/zjr/Tracker/ar_pose_trellis/benchmark_outputs/objaverse_meshrgb_selected_val_testsets.json`
- 测试样本数：8 个 validation case
- 每个 case 测 4 组 pose ablation，共 32 条结果
- checkpoint：`/home/zjr/Tracker/ar_pose_trellis/runs/ss_arpose_meshrgb_1000_s2_e4/last.ckpt`

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

`/home/zjr/Tracker/ar_pose_trellis/benchmark_outputs/sparse_pose_ablation_meshrgb_e4_raw/sparse_pose_ablation_report.json`

| mode | pred coords | IoU | F2 | Chamfer |
|---|---:|---:|---:|---:|
| `correct` | 3719.0 | 0.02554 | 0.16497 | 7.35450 |
| `identity` | 3669.2 | 0.02318 | 0.15794 | 7.40288 |
| `shuffle` | 3777.1 | 0.02418 | 0.15908 | 7.43203 |
| `noise` | 3788.2 | 0.02541 | 0.16438 | 7.33662 |

## 加 visual hull logits prior

设置：`--visual_hull_prior_weight 40`

报告路径：

`/home/zjr/Tracker/ar_pose_trellis/benchmark_outputs/sparse_pose_ablation_meshrgb_e4_vhprior_w40_inprocess/sparse_pose_ablation_report.json`

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
  --testsets /home/zjr/Tracker/ar_pose_trellis/benchmark_outputs/objaverse_meshrgb_selected_val_testsets.json \
  --checkpoint /home/zjr/Tracker/ar_pose_trellis/runs/ss_arpose_meshrgb_1000_s2_e4/last.ckpt \
  --output_root /home/zjr/Tracker/ar_pose_trellis/benchmark_outputs/sparse_pose_ablation_meshrgb_e4_raw \
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
  --testsets /home/zjr/Tracker/ar_pose_trellis/benchmark_outputs/objaverse_meshrgb_selected_val_testsets.json \
  --checkpoint /home/zjr/Tracker/ar_pose_trellis/runs/ss_arpose_meshrgb_1000_s2_e4/last.ckpt \
  --output_root /home/zjr/Tracker/ar_pose_trellis/benchmark_outputs/sparse_pose_ablation_meshrgb_e4_vhprior_w40_inprocess \
  --max_frames 8 \
  --ss_steps 12 \
  --ss_min_coords 0 \
  --cond_fp16 \
  --visual_hull_prior_weight 40 \
  --inprocess \
  --continue_on_error
```
