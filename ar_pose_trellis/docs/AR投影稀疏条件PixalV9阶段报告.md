# AR 投影稀疏条件 Pixal3D Multiview V9 阶段报告

时间：2026-06-16  
代码目录：`/home/zjr/Tracker/ar_pose_trellis`  
数据集：`/data/pixal3d_multiview/objaverse_sparse_mv_artraj_pbr_5000_v9_select8`

## 1. 当前目标

当前目标不是继续优化旧的 `ARDinoRayCond`，而是在 TRELLIS sparse flow 阶段加入 AR pose 显式投影条件：

```text
multi-view RGB / mask / camera pose
  -> project sparse latent grid points to views
  -> sample DINO patch features + mask support + visibility/depth statistics
  -> ARProjectedSparseCond
  -> sparse_structure_flow_model
  -> sparse coords
```

这一阶段只改 sparse flow 条件，不改 SLAT。mesh 质量仍可能受 SLAT 阶段限制。

## 2. 数据集选择

本轮主线切换为 `pixal3d_multiview` 构建的 v9 数据，而不是旧 `ar_pose_trellis` 数据。

原因：

- v9 manifest 已包含质量过滤后的 8-view selection、mask、camera pose、render 统计和 projection 统计。
- v9 数据已有预编码 sparse latent target：`ss_latents/*.npz` 中的 `z`，形状为 `[8,16,16,16]`。
- 训练时可以直接用该 `z` 作为 sparse flow matching target，避免每次从 `target_coords` 重新编码。

重要坐标系：

```text
target_coords / ss_latent: pixal3d_sparse_structure
camera / render space: pixal3d_rotated_render_space
```

因此 projected condition 必须使用：

```text
--projected_grid_transform pixal3d_rotation
```

否则 sparse latent grid 点和相机/mask 不在同一坐标系中。

## 3. 当前模型结构

新增条件模块：

```text
ARProjectedSparseCond
```

主要逻辑：

1. 生成 TRELLIS sparse latent grid centers，默认 `16^3 = 4096` 个 token。
2. 若使用 pixal3d_multiview v9 数据，先对 grid centers 应用 `PIXAL3D_ROTATION`。
3. 将每个 grid point 投影到每个 view。
4. 采样：
   - DINO patch feature
   - mask support
   - visible view count
   - support ratio
   - depth mean/std
   - xyz grid position
5. 对 visible/support feature 做聚合。
6. 输出 `[B,4096,1024]` condition tokens 给 TRELLIS sparse flow。

当前 gating：

```text
projected_min_support = 2.0
projected_min_support_ratio = 0.5
```

低 support voxel 不给 appearance feature，只保留 grid/geometry 信息，避免 mask 外或弱 support 区域污染条件。

## 4. 代码改动

新增文件：

- `ar_pose_trellis/projected_condition.py`
  - 新增 `ARProjectedSparseCond`
  - 支持 `pixal3d_rotation`
  - 支持 weak support gate
  - 避免 zero-mask / weak-support appearance 泄漏

- `ar_pose_trellis/diagnose_projected_sparse_alignment.py`
  - 诊断 target coords/random coords 投影到 mask 的支持率
  - 支持 pixal3d_multiview manifest
  - 支持 `--grid_transform pixal3d_rotation`

- `ar_pose_trellis/evaluate_projected_sparse_pixal_manifest.py`
  - 直接评测 pixal3d_multiview `val.json`
  - 输出 `report.json` / `report.csv`
  - 失败时输出 `failure.json`

- `ar_pose_trellis/scripts/run_projected_pixal_v9_stage1.sh`
  - 统一封装 diagnostic / smoke train / smoke eval / s200 train / s200 eval
  - 避免长命令粘贴截断

修改文件：

- `ar_pose_trellis/train_ss_ar_pose.py`
  - 新增 `pixal3d_multiview_manifest` dataset format
  - 直接读取 `ss_latent z`
  - 新增 projected condition 参数

- `ar_pose_trellis/pipeline.py`
  - pipeline 支持 `condition_mode=projected`
  - 支持 `projected_grid_transform`

- `ar_pose_trellis/generate_ar_pose_mesh.py`
  - 推理入口支持 projected condition 参数

- `ar_pose_trellis/benchmark/evaluate_sparse_pose_ablation.py`
  - 旧 ablation 入口透传 projected 参数

- `ar_pose_trellis/__init__.py`
  - 导出 `ARProjectedSparseCond`

## 5. 投影诊断结果

命令：

```bash
GPU=1 MODE=smoke bash ar_pose_trellis/scripts/run_projected_pixal_v9_stage1.sh
```

诊断输出：

```text
ar_pose_trellis/outputs/diagnostics/projected_alignment_pixal_v9_train_0_31_rot_strict_v2.json
```

32 个 train 样本统计：

| group | supported_ratio | support_ratio_mean | visible_mean |
|---|---:|---:|---:|
| target | 0.7381 | 0.6897 | 7.6864 |
| random | 0.4083 | 0.4084 | 6.3301 |
| target - random | +0.3298 | +0.2813 | +1.3562 |

判断：

```text
pixal3d_rotation 后，target sparse coords 投影到 mask 的支持明显高于 random coords。
说明 AR pose + mask 的 projected sparse signal 是存在的。
```

## 6. Smoke 训练结果

训练输出：

```text
ar_pose_trellis/outputs/training_runs/ss_projected_pixal_v9_rot_smoke_v2/last.ckpt
```

训练配置：

```text
dataset_format = pixal3d_multiview_manifest
condition_mode = projected
projected_grid_transform = pixal3d_rotation
projected_min_support = 2.0
projected_min_support_ratio = 0.5
num_views = 8
max_steps = 20
lr = 5e-5
cfg_drop_prob = 0.1
```

训练日志摘要：

```text
Trainable params: 27.8M
Non-trainable params: 919M
Total params: 947M
train_loss at step 20: 0.299
```

判断：

```text
训练链路正常，checkpoint 正常生成。
20 step smoke 只用于验证链路和初步 pose sensitivity，不用于最终判断模型收益。
```

## 7. Smoke 测试结果

测试输出：

```text
ar_pose_trellis/outputs/benchmarks/ss_projected_pixal_v9_rot_smoke_v2_val8/report.json
ar_pose_trellis/outputs/benchmarks/ss_projected_pixal_v9_rot_smoke_v2_val8/report.csv
```

8 个 val 样本，pose modes：

```text
correct, identity, shuffle, noise
```

聚合结果：

| pose | IoU mean | recall mean | precision mean | pred_unique mean | intersection mean |
|---|---:|---:|---:|---:|---:|
| correct | 0.0173 | 0.0260 | 0.1054 | 2979.0 | 229.1 |
| identity | 0.0470 | 0.0989 | 0.0890 | 10668.0 | 965.9 |
| shuffle | 0.0076 | 0.0090 | 0.1139 | 1094.1 | 93.5 |
| noise | 0.0101 | 0.0139 | 0.1130 | 1573.5 | 128.3 |

关键现象：

```text
correct > shuffle
correct > noise
identity > correct
```

解释：

- `correct > shuffle/noise` 是正向信号，说明 projected condition 已经有弱 pose sensitivity。
- `identity` 高于 `correct` 是当前主要问题。
- `identity` 的 `pred_unique mean = 10668`，明显大于 `correct = 2979`，接近 target 的 `9776` 量级。
- 当前 threshold=0 的 sparse overlap 受预测点数量影响很大，identity 可能因为预测更 dense 而获得更高 IoU/recall，不代表它真的更懂几何。

## 8. 当前结论

当前阶段不能说模型已经成功，但也不能说路线失败。

已经成立的部分：

```text
1. pixal3d_multiview v9 数据的 projected geometry signal 存在。
2. pixal3d_rotation 是必要坐标变换。
3. projected sparse 训练和测试链路已经跑通。
4. 20-step smoke 中 correct 已经压过 shuffle/noise。
```

尚未成立的部分：

```text
1. correct 尚未压过 identity。
2. 当前 threshold=0 overlap 指标受预测点数影响，identity 的 dense prediction 会虚高。
3. 20-step 训练太短，不能判断模型收益上限。
4. 只改 sparse，尚未验证 SLAT/mesh 质量。
```

当前判断：

```text
AR projected sparse condition 有继续测试价值，但需要更公平的评测和更长训练。
```

## 9. 下一步建议

### 9.1 先跑 s200

Smoke 已满足最低继续条件：

```text
correct > shuffle
correct > noise
```

下一步执行：

```bash
cd /home/zjr/Tracker
GPU=1 MODE=s200 bash ar_pose_trellis/scripts/run_projected_pixal_v9_stage1.sh
```

结果路径：

```text
ar_pose_trellis/outputs/training_runs/ss_projected_pixal_v9_rot_s200_v2/last.ckpt
ar_pose_trellis/outputs/benchmarks/ss_projected_pixal_v9_rot_s200_v2_val32/report.json
ar_pose_trellis/outputs/benchmarks/ss_projected_pixal_v9_rot_s200_v2_val32/report.csv
```

### 9.2 增加 fixed top-k / matched-count sparse eval

当前 `threshold=0` 会让不同 pose 产生不同数量的 coords：

```text
correct: 约 2979
identity: 约 10668
```

这使得 identity 在 IoU/recall 上占便宜。

建议新增评测：

```text
固定 pred coords 数量，例如 top-k = target_unique 或 top-k = 4096/8192。
再计算 IoU / recall / precision。
```

这样才能判断 identity 是否真的更好，还是只是点数多。

### 9.3 增加 visual hull prior 对照

建议在 s200 后增加：

```text
visual_hull_prior_weight = 20 / 40 / 80
```

观察：

```text
correct 是否更稳定
identity 是否仍然靠 dense coords 虚高
shuffle/noise 是否被压低
```

### 9.4 若 s200 仍有正向信号，再进入 SLAT

只改 sparse 不一定能改善最终 mesh。

如果 s200 结果满足：

```text
correct > shuffle/noise
correct 在 fixed top-k 下接近或超过 identity
```

再进入第二阶段：

```text
ARProjectedSLATCond
```

即对 active sparse coords 做多视图投影采样，让 SLAT 阶段也看到 AR pose/RGB/mask 约束。

## 10. 当前不建议做的事

暂时不建议：

```text
1. 继续优化旧 ARDinoRayCond 作为主线。
2. 继续加 pose head / reranker 作为主线。
3. 在没有 fixed top-k eval 前，仅凭 threshold=0 的 identity IoU 判断失败。
4. 直接把 sparse 结果用于 mesh 成败判断，因为 SLAT 尚未接入 AR projected condition。
```

## 11. 当前推荐判断标准

s200 后重点看：

```text
1. correct vs shuffle
2. correct vs noise / large_noise
3. correct vs identity 的 fixed top-k 指标
4. pred_unique 是否稳定在合理范围
5. visual hull prior 是否改善 correct 且不让 identity 虚高
```

阶段成功标准：

```text
correct 在正常错误 pose 上稳定领先，并且 fixed-count eval 下不被 identity 明显压过。
```

如果达不到该标准，则 projected sparse 只能作为辅助 prior/gate，不应继续作为唯一模型改进主线。

## 12. s200 训练测试结果补充

运行结果：

```text
checkpoint:
/home/zjr/Tracker/ar_pose_trellis/outputs/training_runs/ss_projected_pixal_v9_rot_s200_v2/last.ckpt

eval report:
/home/zjr/Tracker/ar_pose_trellis/outputs/benchmarks/ss_projected_pixal_v9_rot_s200_v2_val32/report.json
```

评测设置：

```text
indices: 0-31
pose_modes: correct, identity, shuffle, noise, large_noise
ss_steps: 12
ss_threshold: 0.0
condition_mode: projected
projected_grid_transform: pixal3d_rotation
projected_min_support: 2.0
projected_min_support_ratio: 0.5
visual_hull_prior_weight: 0.0
```

### 12.1 汇总指标

| pose | count | IoU mean | IoU median | recall mean | precision mean | pred_unique mean | intersection mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| correct | 32 | 0.0221 | 0.0216 | 0.0329 | 0.0824 | 3819.1 | 305.6 |
| identity | 32 | 0.0143 | 0.0121 | 0.0158 | 0.1481 | 929.6 | 138.1 |
| shuffle | 32 | 0.0197 | 0.0183 | 0.0357 | 0.0580 | 5050.7 | 328.2 |
| noise | 32 | 0.0115 | 0.0068 | 0.0200 | 0.0409 | 3664.4 | 168.1 |
| large_noise | 32 | 0.0229 | 0.0167 | 0.0314 | 0.1112 | 2856.0 | 261.0 |

相比 smoke：

| pose | smoke IoU mean | s200 IoU mean | delta | smoke pred_unique | s200 pred_unique |
|---|---:|---:|---:|---:|---:|
| correct | 0.0173 | 0.0221 | +0.0048 | 2979.0 | 3819.1 |
| identity | 0.0470 | 0.0143 | -0.0327 | 10668.0 | 929.6 |
| shuffle | 0.0076 | 0.0197 | +0.0120 | 1094.1 | 5050.7 |
| noise | 0.0101 | 0.0115 | +0.0013 | 1573.5 | 3664.4 |

### 12.2 Correct-vs-Wrong

按每个 sample 的 IoU 比较：

| wrong pose | correct wins | ties | mean delta | median delta |
|---|---:|---:|---:|---:|
| identity | 21/32 | 0 | +0.0078 | +0.0090 |
| shuffle | 18/32 | 4 | +0.0025 | +0.0014 |
| noise | 19/32 | 2 | +0.0106 | +0.0103 |
| large_noise | 13/32 | 0 | -0.0008 | -0.0007 |

按 IoU top1 统计：

```text
correct: 7/32
large_noise: 12/32
shuffle: 5/32
noise: 5/32
identity: 3/32

correct rank mean: 2.59
correct rank median: 3.0
```

空预测情况：

```text
correct zero pred: 4/32
identity zero pred: 0/32
shuffle zero pred: 5/32
noise zero pred: 9/32
large_noise zero pred: 0/32
```

### 12.3 当前结论

这次 s200 不能再说是完全失败。相比 smoke，有三个正向信号：

```text
1. correct IoU 从 0.0173 提升到 0.0221。
2. identity 的异常虚高被明显压下，从 0.0470 降到 0.0143。
3. correct 对 identity/noise 的 per-sample 胜率已经明显高于随机。
```

但也不能说 sparse projected condition 已经学成了可靠 pose-sensitive 约束，原因是：

```text
1. large_noise 的 mean IoU = 0.0229，略高于 correct 的 0.0221。
2. shuffle 的 mean IoU = 0.0197，已经很接近 correct。
3. correct 的 IoU top1 只有 7/32。
4. correct 还有 4/32 空预测，说明 threshold=0 下输出稳定性不足。
5. 当前仍只评估 sparse coords，没有进入 SLAT / mesh 质量判断。
```

因此阶段判断是：

```text
有训练收益，但收益还没有强到可以证明 AR pose projected sparse 已经是主线成功方案。
当前更像是 sparse flow 开始利用投影条件，但 pose 区分能力仍不稳定。
```

### 12.4 下一步建议

第一优先级不是继续盲目加训练步数，而是先补一个 fixed-count sparse eval。

当前评测使用：

```text
ss_threshold = 0.0
```

不同 pose 产生的 `pred_unique` 差异很大：

```text
correct: 3819
identity: 930
shuffle: 5051
noise: 3664
large_noise: 2856
```

这会把 IoU、recall 和 precision 混在一起：有些 wrong pose 是因为预测点更多而占 recall 便宜，有些 pose 是因为点更少而 precision 变高。因此下一步应新增或运行：

```text
fixed top-k eval:
1. topk = 4096
2. topk = 8192
3. topk = target_unique
```

目标是回答一个更干净的问题：

```text
在预测点数一致时，correct 是否稳定优于 shuffle / noise / large_noise / identity。
```

第二优先级是 visual hull prior sweep，但要放在 fixed-count eval 之后。建议配置：

```text
visual_hull_prior_weight = 10 / 20 / 40
visual_hull_min_visible_views = 1
visual_hull_mask_threshold = 0.5
```

观察标准：

```text
1. correct IoU/recall 是否提升。
2. shuffle / large_noise 是否被压低。
3. correct zero pred 是否减少。
4. pred_unique 是否不再剧烈摆动。
```

第三优先级才是更长训练，例如 s1000。只有 fixed-count eval 显示 correct 稳定领先时，s1000 才有意义。否则继续训练可能只是强化当前不稳定的 pose/volume 偏置。

暂时不建议立刻进入 mesh：

```text
sparse stage 当前只证明了弱正向信号。
SLAT 还没有接入 projected condition。
如果直接生成 mesh，失败时无法区分是 sparse 没学好，还是 SLAT 阶段丢掉了 AR pose 信息。
```

当前推荐顺序：

```text
1. 新增 fixed-count sparse eval。
2. 用 s200 checkpoint 跑 fixed-count eval。
3. 如果 correct 在 fixed-count 下仍领先，再跑 visual hull prior sweep。
4. 如果 visual hull 有正向收益，再跑 s1000。
5. s1000 稳定后，再实现 ARProjectedSLATCond 并进入 mesh 评测。
```

## 13. Fixed-count sparse eval 结果补充

运行结果：

```text
/home/zjr/Tracker/ar_pose_trellis/outputs/benchmarks/ss_projected_pixal_v9_rot_s200_v2_val32_fixed_topk/report.json
```

设置：

```text
checkpoint: ss_projected_pixal_v9_rot_s200_v2/last.ckpt
indices: 0-31
pose_modes: correct, identity, shuffle, noise, large_noise
fixed_topk: 4096, 8192, target_unique
visual_hull_prior_weight: 0.0
```

这次评测直接从 decoder logits 选 fixed top-k，不再使用 `ss_threshold=0.0` 的可变点数，也不使用 `ss_min_coords` fallback。它更直接地测试 learned projected condition 的 spatial ranking 能力。

### 13.1 Top-k 汇总

| selection | pose | IoU mean | recall mean | precision mean | pred_unique mean |
|---|---|---:|---:|---:|---:|
| topk_4096 | correct | 0.0240 | 0.0368 | 0.0736 | 4096 |
| topk_4096 | identity | 0.0337 | 0.0499 | 0.1052 | 4096 |
| topk_4096 | shuffle | 0.0160 | 0.0242 | 0.0498 | 4096 |
| topk_4096 | noise | 0.0175 | 0.0266 | 0.0545 | 4096 |
| topk_4096 | large_noise | 0.0281 | 0.0431 | 0.0876 | 4096 |
| topk_8192 | correct | 0.0300 | 0.0600 | 0.0631 | 8192 |
| topk_8192 | identity | 0.0270 | 0.0539 | 0.0567 | 8192 |
| topk_8192 | shuffle | 0.0234 | 0.0466 | 0.0483 | 8192 |
| topk_8192 | noise | 0.0222 | 0.0452 | 0.0455 | 8192 |
| topk_8192 | large_noise | 0.0329 | 0.0653 | 0.0705 | 8192 |
| topk_target_unique | correct | 0.0315 | 0.0598 | 0.0598 | 9108.8 |
| topk_target_unique | identity | 0.0274 | 0.0524 | 0.0524 | 9108.8 |
| topk_target_unique | shuffle | 0.0248 | 0.0472 | 0.0472 | 9108.8 |
| topk_target_unique | noise | 0.0214 | 0.0412 | 0.0412 | 9108.8 |
| topk_target_unique | large_noise | 0.0340 | 0.0647 | 0.0647 | 9108.8 |

### 13.2 Rank / top1

| selection | metric | correct top1 | top1 rate | rank mean | rank median | most frequent top1 |
|---|---|---:|---:|---:|---:|---|
| topk_4096 | IoU | 4/32 | 0.125 | 3.00 | 3.00 | identity 14/32 |
| topk_8192 | IoU | 6/32 | 0.188 | 2.69 | 2.00 | large_noise 11/32 |
| topk_target_unique | IoU | 8/32 | 0.250 | 2.66 | 2.50 | large_noise 13/32 |

### 13.3 Paired IoU delta

`mean_delta = correct - wrong`，越大越好。

| selection | wrong pose | correct wins | win rate | mean delta | median delta |
|---|---|---:|---:|---:|---:|
| topk_4096 | identity | 10/32 | 0.313 | -0.0097 | -0.0057 |
| topk_4096 | large_noise | 13/32 | 0.406 | -0.0041 | -0.0032 |
| topk_4096 | noise | 18/32 | 0.563 | +0.0065 | +0.0033 |
| topk_4096 | shuffle | 18/32 | 0.563 | +0.0080 | +0.0021 |
| topk_8192 | identity | 19/32 | 0.594 | +0.0030 | +0.0091 |
| topk_8192 | large_noise | 14/32 | 0.438 | -0.0029 | -0.0005 |
| topk_8192 | noise | 18/32 | 0.563 | +0.0077 | +0.0076 |
| topk_8192 | shuffle | 18/32 | 0.563 | +0.0066 | +0.0016 |
| topk_target_unique | identity | 19/32 | 0.594 | +0.0040 | +0.0069 |
| topk_target_unique | large_noise | 11/32 | 0.344 | -0.0026 | -0.0037 |
| topk_target_unique | noise | 20/32 | 0.625 | +0.0101 | +0.0077 |
| topk_target_unique | shuffle | 20/32 | 0.625 | +0.0067 | +0.0022 |

### 13.4 当前判断

fixed-count 结果比 threshold=0 更明确：当前 learned projected sparse condition 没有学到稳定的 pose-sensitive spatial ranking。

主要问题：

```text
1. correct top1 最高只有 8/32，即 25%。
2. topk_4096 下 identity 明显强于 correct。
3. topk_8192 和 target_unique 下 large_noise 仍然强于 correct。
4. correct 只对 noise / shuffle 有弱优势，但优势不大。
5. fixed-count 已排除了 pred_unique 差异，因此问题不是单纯 threshold calibration。
```

这说明当前失败点更接近训练目标和 pose negative 监督不足，而不是推理阈值问题。

### 13.5 下一步选择：wrong-pose ranking 优先

在 fixed-count 结果出来后，下一步不应优先做 visual hull prior sweep 作为主线。

原因：

```text
visual hull prior 是显式几何后验，会直接改 logits。
如果它提升 correct rank，只能说明 mask + pose 的硬几何 prior 有用，
不能证明当前 projected condition 自己学会了 pose-sensitive 排序。
```

当前更应该先加 wrong-pose ranking / contrastive sparse loss：

```text
同一个样本、同一组 images/masks：
correct pose -> logits_correct
wrong pose   -> logits_wrong

要求：
score(logits_correct, target_sparse) >
score(logits_wrong, target_sparse) + margin
```

推荐先做轻量训练目标，不直接大改模型结构：

```text
1. 保留当前 ARProjectedSparseCond。
2. 每个 batch 额外构造 1-2 个 wrong pose：
   identity / shuffle / large_noise / noise。
3. 对 correct 和 wrong 分别跑 sparse flow。
4. 用 fixed top-k 或 target coords 上的 logit margin 做 ranking loss。
5. 总 loss = flow matching loss + lambda_rank * ranking_loss。
```

ranking score 建议优先用 target coords 上的 logits，而不是 IoU 后处理：

```text
target_logit_mean = mean(logits[target_coords])
background_logit_mean = mean(logits[random_non_target_coords])
score = target_logit_mean - background_logit_mean

loss_rank = relu(margin - score_correct + score_wrong)
```

这样训练信号直接作用在 decoder logits 的排序能力上，而不是等采样后才评估。

visual hull prior sweep 仍然可以保留，但定位应改为：

```text
辅助诊断 / deployment prior 上限测试，
不作为 learned projected condition 是否成功的主判据。
```

推荐顺序改为：

```text
1. 先实现 wrong-pose ranking loss。
2. 用 s200/s500 ranking 训练复测 fixed top-k。
3. 如果 correct top1/rank 明显改善，再跑 visual hull prior sweep 看推理期 prior 是否锦上添花。
4. 如果 ranking 仍无效，再考虑模型结构问题，而不是继续堆 prior。
```

## 14. Wrong-pose ranking s200 结果与跨项目结论

本轮在 `ss_projected_pixal_v9_rot_s200_v2/last.ckpt` 基础上继续训练 wrong-pose ranking：

```text
ranking modes: identity, shuffle, large_noise, noise
ranking weight: 0.2
ranking margin: 0.05
ranking negatives per step: 1
eval: fixed_topk = 4096, 8192, target_unique
```

结果路径：

```text
smoke:
/home/zjr/Tracker/ar_pose_trellis/outputs/benchmarks/ss_projected_pixal_v9_rot_s200_rank_smoke_v1_val8_fixed_topk/report.json

s200:
/home/zjr/Tracker/ar_pose_trellis/outputs/benchmarks/ss_projected_pixal_v9_rot_s200_rank_s200_v1_val32_fixed_topk/report.json
```

### 14.1 Ranking s200 相比 baseline fixed-topk

| selection | run | correct top1 | top1 rate | rank mean | rank median | most frequent top1 |
|---|---|---:|---:|---:|---:|---|
| topk_4096 | baseline | 4/32 | 0.125 | 3.000 | 3.000 | identity 14/32 |
| topk_4096 | ranking s200 | 11/32 | 0.344 | 2.000 | 2.000 | correct 11/32 |
| topk_8192 | baseline | 6/32 | 0.188 | 2.688 | 2.000 | large_noise 11/32 |
| topk_8192 | ranking s200 | 11/32 | 0.344 | 2.000 | 2.000 | correct 11/32 |
| topk_target_unique | baseline | 8/32 | 0.250 | 2.656 | 2.500 | large_noise 13/32 |
| topk_target_unique | ranking s200 | 12/32 | 0.375 | 2.031 | 2.000 | correct 12/32 |

Ranking s200 的正向变化是明确的：

```text
1. correct top1 从 4/32、6/32、8/32 提升到 11/32、11/32、12/32。
2. rank mean 从 2.66-3.00 降到约 2.00。
3. large_noise 不再系统性压过 correct。
4. identity 被明显压下，ranking s200 中 identity 的 fixed-topk IoU 近似为 0。
```

### 14.2 但 correct 几何本身没有明显变好

| selection | baseline correct IoU | ranking s200 correct IoU | delta |
|---|---:|---:|---:|
| topk_4096 | 0.0240 | 0.0254 | +0.0014 |
| topk_8192 | 0.0300 | 0.0276 | -0.0024 |
| topk_target_unique | 0.0315 | 0.0290 | -0.0025 |

也就是说，ranking loss 改善了相对排序，但没有让 correct pose 下的 sparse coords 更接近 target。它主要学到了：

```text
压低 identity / large_noise 这类 wrong pose，
而不是增强 correct pose 的真实三维结构生成能力。
```

这和 fixed top-k paired delta 一致：

| selection | wrong pose | ranking s200 win rate | mean delta |
|---|---|---:|---:|
| topk_4096 | identity | 0.781 | +0.0254 |
| topk_4096 | large_noise | 0.594 | +0.0124 |
| topk_4096 | shuffle | 0.500 | +0.0059 |
| topk_4096 | noise | 0.500 | +0.0027 |
| topk_8192 | identity | 0.781 | +0.0276 |
| topk_8192 | large_noise | 0.594 | +0.0143 |
| topk_8192 | shuffle | 0.531 | +0.0036 |
| topk_8192 | noise | 0.469 | +0.0020 |
| topk_target_unique | identity | 0.781 | +0.0290 |
| topk_target_unique | large_noise | 0.594 | +0.0155 |
| topk_target_unique | shuffle | 0.500 | +0.0014 |
| topk_target_unique | noise | 0.500 | +0.0032 |

对 identity / large_noise 的收益明显；对 shuffle / noise 基本接近 50% 胜率，说明模型仍没有稳定学到真实的 view-pose correspondence。

### 14.3 结合 pixal3d_multiview 五层报告

`pixal3d_multiview/outputs/五层测试流程报告/多视图稀疏结构五层测试报告.md` 的核心结论是：

```text
当前 multiview sparse checkpoint 有训练收益；
但相机位姿约束仍然偏弱；
主要瓶颈是 multiview projection adapter / sparse flow 难以把 pose 当作强几何条件使用。
```

后续 pose head / pairwise / front-depth / alpha sweep 的结论也类似：

```text
pose-sensitive 信号在 scorer/reranker 层面存在；
但传导到 sparse coords / sparse view-gate prior 时收益很弱；
继续调 alpha、pair_weight_mode、head 小结构的边际收益很低。
```

本轮 `ar_pose_trellis` 的 wrong-pose ranking 与该趋势一致：

```text
1. 模型修改不是完全无效，ranking 确实能改变 pose 排序。
2. 但 sparse 结构质量没有同步提升。
3. correct top1 最高只有 12/32，仍远不到“可靠 pose-sensitive sparse generator”。
4. correct IoU 仍在 0.025-0.029 量级，不足以直接进入 mesh 成功判断。
```

因此现在可以更明确地说：

```text
如果“改模型”指继续在 sparse stage 里改 adapter / prior / ranking loss，
当前确实仍然呈现边际收益偏低的趋势。

但这不等于 AR pose 或多视图信息没有价值；
它说明价值没有稳定传导到 sparse flow 的 coarse coords 生成上。
```

### 14.4 对下一步的判断

不建议继续直接做：

```text
ranking s500 / s1000
继续加大 ranking_weight
继续堆更多 wrong pose
继续把 visual hull prior 当主线救 sparse IoU
```

原因是当前 s200 已经显示：

```text
ranking 能压低 wrong pose，
但 correct sparse geometry 没有提升。
```

继续加训练步数很可能进一步学习 wrong-pose shortcut，而不是生成更好的 correct sparse structure。

下一步建议分两条线：

**路线 A：最小复核，确认不是 ranking 超参问题**

只做一个小规模复核，不作为主线：

```text
RANKING_WEIGHT = 0.05 或 0.1
RANKING_MODES = shuffle,noise,large_noise
去掉 identity，因为 identity 太容易形成 out-of-distribution shortcut
```

通过标准：

```text
1. correct IoU 不下降；
2. correct top1/rank 继续提升；
3. correct vs shuffle/noise 的 paired delta 明显转正；
4. 不只是 identity 被压到 0。
```

如果这个复核仍然不能提高 correct IoU，则停止继续 sparse-stage ranking。

**路线 B：转向最终目标，更建议作为主线**

把精力转向 mesh / reprojection / SLAT 条件：

```text
1. 不再只用 sparse IoU 判断成败。
2. 让多视图 RGB/mask/pose 进入 SLAT 或 mesh refinement，而不是只进 sparse flow。
3. 对生成 mesh 做 multi-view mask reprojection / visible feature consistency。
4. 若存在多个 pose 或 sparse/mesh hypothesis，再用 scorer/ranking 做 rerank，而不是期望 sparse flow 本身完全解决 pose-sensitive 约束。
```

更具体的下一步建议：

```text
第一优先级：实现 ARProjectedSLATCond 或 mesh-level reprojection loss / eval。
第二优先级：把 ranking/head 作为 pose/sparse/mesh hypothesis reranker。
第三优先级：只做一个去掉 identity 的 low-weight ranking 复核。
不建议继续把 sparse-stage adapter/head 小改作为主线。
```
