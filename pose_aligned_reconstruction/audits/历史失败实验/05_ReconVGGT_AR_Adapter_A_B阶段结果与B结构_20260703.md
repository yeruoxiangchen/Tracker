# ReconVGGT AR Adapter A/B 阶段记录 2026-07-03

## A 阶段结论：AR session + SLAT sanity 已通过

已完成两组 A 阶段测试：

```text
reconvggt_ar_adapter_a/outputs/ar_20260617_vggt_token_sanity_mv4
reconvggt_ar_adapter_a/outputs/ar_20260617_vggt_token_sanity_mv4_slat
```

AR session 测试使用真实手机采集图像：

```text
/home/zjr/Tracker/ReconViaGen/ar_tracker/data/20260617_070746_829
```

它的目的不是评估 mesh 质量，而是确认 ReconViaGen + VGGT 在真实 AR 图片输入下的 token layout 是否能支撑后续 projection-aware adapter：

```text
1. aggregated_tokens_list 是否保留 view 维；
2. token 是否能还原成 per-view spatial grid；
3. 前 5 个 token 是否应按 prefix/global/register token 处理；
4. zero-init adapter 插入后 sparse condition 是否完全不变；
5. --check_slat 时 SLAT condition 是否也完全不变。
```

关键结果：

```text
image_count = 4
input_tensor = [4, 3, 518, 518]
image_cond raw = [4, 1374, 1024]
image_cond normalized = [1, 4, 1374, 1024]
prefix_tokens = 5
spatial_tokens = 1369 = 37 * 37
pixel_per_token = 14.0
```

选中的 VGGT 层 `[4, 11, 17, 23]` 都是：

```text
[B, V, T, C] = [1, 4, 1374, 2048]
```

zero-init sparse-only：

```text
selected_token_max_abs_diff = 0
ss_cond_max_abs_diff = 0
passed = true
```

zero-init with SLAT：

```text
selected_token_max_abs_diff = 0
ss_cond_max_abs_diff = 0
slat_cond_max_abs_diff = 0
passed = true
```

结论：

```text
VGGT aggregated tokens 在当前 ReconViaGen 路径下保留 view 维和 37x37 spatial token grid。
前 5 个 token 应作为 prefix/global/register tokens，不参与 naive spatial projection。
A 阶段 adapter 插入点对 sparse 和 SLAT condition 都是安全的。
```

## B 阶段结构

B 阶段新增 projection-aware spatial token adapter，但仍然先做 zero-init sanity，不直接训练。

结构：

```text
AR poses.txt + selected image names
  -> per-view 37x37 token-grid pose/ray/intrinsics features
  -> zero-init MLP
  -> VGGT selected layers [4,11,17,23] spatial-token bias
  -> ReconViaGen get_ss_cond / get_slat_cond
```

具体约束：

```text
1. prefix/global/register tokens [0:5] 原样复制，不改；
2. spatial tokens [5:] 必须是 37*37；
3. projection features shape 必须是 [B, V, 1369, F]；
4. final projection layer zero-init；
5. zero-init 后 selected token diff、ss_cond diff、slat_cond diff 都必须为 0。
```

当前 B 阶段 feature 内容：

```text
normalized token center x/y/r
camera-space ray
quaternion-rotated ray descriptor
normalized camera position
quaternion xyzw
resized intrinsics fx/fy/cx/cy
view index + valid flag
```

这一步还不是最终训练结构。它先解决一个基础问题：

```text
VGGT token 是否能被真实 AR pose 以 per-view spatial token grid 的方式安全调制。
```

如果 B zero-init sanity 不通过，不能继续训练；如果通过，下一步才考虑训练 adapter 或把 AR point-prior 投影统计加入 feature。

## 新增/修改代码

新增：

```text
reconvggt_ar_adapter_a/projection_token_features.py
reconvggt_ar_adapter_a/run_b_projection_sanity.py
```

修改：

```text
reconvggt_ar_adapter_a/token_adapter.py
```

新增类/函数：

```text
ProjectionAwareSpatialTokenAdapter
parse_ar_pose_file()
select_pose_records()
build_pose_token_features()
summarize_pose_features()
run_b_projection_sanity.py
```

## 下一步判断

B 阶段第一步只看：

```text
zero_init_sanity.passed == true
ss_cond_max_abs_diff == 0
slat_cond_max_abs_diff == 0
prefix_token_max_abs_diff 全 0
projection_features shape == [1, 4, 1369, F]
```

通过后，再做：

```text
1. AR point-prior projection occupancy/support feature 接入；
2. adapter-only 小样本训练；
3. sparse/SLAT 分支分别评估；
4. 最后才进入 ReconViaGen mesh downstream。
```

## 本次验证状态

已完成静态验证：

```bash
/home/zjr/anaconda3/envs/reconviagen/bin/python -m py_compile \
  reconvggt_ar_adapter_a/token_adapter.py \
  reconvggt_ar_adapter_a/projection_token_features.py \
  reconvggt_ar_adapter_a/run_b_projection_sanity.py
```

结果：

```text
通过。
```

已完成 CLI 导入检查：

```bash
/home/zjr/anaconda3/envs/reconviagen/bin/python reconvggt_ar_adapter_a/run_b_projection_sanity.py --help
```

结果：

```text
通过，参数正常显示。
```

我在工具环境中尝试直接运行 B zero-init sanity，但该环境里 PyTorch 报：

```text
torch.cuda.is_available() == false
RuntimeError: No CUDA GPUs are available
```

同时 `nvidia-smi` 可以看到 GPU，因此这更像当前工具执行环境的 CUDA runtime 可见性问题，不是 B 阶段代码逻辑错误。请在你之前能正常运行 ReconViaGen / TRELLIS 的终端里执行 `命令说明.txt` 的 7.1 命令。

## B zero-init 实跑结果

你运行完成的 B sanity report：

```text
reconvggt_ar_adapter_a/outputs/ar_20260617_b_projection_token_sanity_mv4_slat/report.json
```

关键结果：

```text
projection_features.shape = [1, 4, 1369, 22]
feature_dim = 22
matched_pose_names = frame_0000.jpg ... frame_0003.jpg
selected_token_max_abs_diff = 0
prefix_token_max_abs_diff = 0
ss_cond_max_abs_diff = 0
slat_cond_max_abs_diff = 0
zero_init_sanity.passed = true
```

结论：

```text
B 阶段 pose/ray/intrinsics projection-aware token adapter 插入点成立。
它不会改变 sparse condition，也不会改变 SLAT condition。
prefix/global/register tokens 也保持不变。
```

这说明可以进入 B1：把真实 AR sparse point-prior 投影统计接入 token-grid feature。

## B1 新增：point-prior projection token feature

本次新增/修改：

```text
reconvggt_ar_adapter_a/projection_token_features.py
reconvggt_ar_adapter_a/run_b_projection_sanity.py
```

新增能力：

```text
1. 支持两种 pose 格式：
   - 完整 AR pose: image, position, euler, quaternion, intrinsics
   - 简化 pose: image, position, euler

2. 支持 point-prior 输入：
   - --points3d_txt
   - --point_prior_npz
   - --colmap_sparse_dir

3. 推荐优先使用 --colmap_sparse_dir：
   cameras.txt / images.txt / points3D.txt 的坐标约定明确，
   可直接用 COLMAP world-to-camera qvec/tvec 投影。
```

B1 point-prior token feature：

```text
per-view 37x37 token feature 追加 7 维：
  hit
  log_count
  mean_conf
  mean_inv_depth
  max_inv_depth
  mean_u_norm
  mean_v_norm
```

CPU 级别 feature 构建测试已通过：

```text
prepared session:
  ar_session_20260617_075401_819

COLMAP sparse dir:
  .../sparse_ar_streaming/0

points3D count:
  820

output feature:
  [1, 7, 1369, 7]

per_view_inside_count:
  [820, 820, 820, 820, 732, 818, 817]

inside_count_total:
  5647
```

这说明 B1 的 point-prior projection path 是有效的。下一步应先跑 `命令说明.txt` 的 8.1，确认：

```text
projection_features.shape = [1, 7, 1369, 29]
pose feature dim = 22
point feature dim = 7
ss_cond_max_abs_diff = 0
slat_cond_max_abs_diff = 0
zero_init_sanity.passed = true
```

如果 8.1 通过，才进入 B2 adapter-only 训练；否则先修 projection feature / image-pose matching。

## 8.1 实跑结果：B1 point-prior projection sanity 已通过

你运行完成的 B1 report：

```text
reconvggt_ar_adapter_a/outputs/ar_20260617_b1_pointprior_colmap_projection_sanity_mv7_slat/report.json
```

关键结果：

```text
projection_features.shape = [1, 7, 1369, 29]
pose_projection_features.shape = [1, 7, 1369, 22]
point_count = 820
per_view_inside_count = [820, 820, 820, 820, 732, 818, 817]
inside_count_total = 5647
selected_token_max_abs_diff = 0
prefix_token_max_abs_diff = 0
ss_cond_max_abs_diff = 0
slat_cond_max_abs_diff = 0
zero_init_sanity.passed = true
```

结论：

```text
B1 的 AR point-prior projection feature path 成立。
COLMAP sparse dir 能把点云可靠投到 per-view 37x37 token grid。
zero-init adapter 仍然不改变 sparse/SLAT condition。
```

因此可以进入 B2。

## B2 结构：adapter-only signal-injection smoke

B2 不直接宣称 mesh 质量提升，也不直接训练 ReconViaGen 主干。它先验证一个更小的问题：

```text
point-prior projection feature 是否能让 adapter 学到受控的 spatial-token residual；
这个 residual 是否集中在 point-hit tokens；
训练后 ss/slat condition drift 是否可控。
```

B2 训练目标：

```text
input:
  frozen VGGT aggregated tokens
  pose/ray/intrinsics token features
  point-prior projection token features

trainable:
  ProjectionAwareSpatialTokenAdapter only

loss:
  hit token 的 directional score -> 接近 small target
  miss token 的 directional score -> 接近 0
  加轻量 bias L2

metrics:
  score_hit_mean
  score_miss_mean
  score_separation
  energy_hit_mean
  energy_miss_mean
  energy_separation
  final ss_cond_max_abs_diff
  final slat_cond_max_abs_diff
```

这个训练的意义：

```text
energy-only/norm-only loss 在 zero-init bias=0 处没有有效方向，可能无法启动。
B2 因此改为 fixed random direction 的 directional score loss。
这样 adapter 仍可从严格 zero-init 开始，但 hit token 有非零梯度方向。

如果 B2 连 hit/miss token bias separation 都学不出来，说明 projection feature/adapter 结构本身有问题。
如果能学出来但 ss/slat drift 过大，说明 injection 太强，需要更小 target scale 或 block-wise/gated 注入。
如果能学出来且 drift 可控，下一步才考虑接入实际 sparse/mesh downstream。
```

本次新增：

```text
reconvggt_ar_adapter_a/train_b_projection_adapter.py
run_b_projection_sanity.py 支持 --load_adapter，用于检查 trained adapter condition drift
run_b_projection_sanity.py 在 --load_adapter 时额外输出 eval_mode=loaded_adapter_drift、drift_eval、adapter_energy_stats
run_b_projection_sanity.py 会从 checkpoint 的 target_dirs 复算 adapter_score_stats
```

## B2 修正：避免 zero-init energy-only 零梯度

原 B2 草案如果只用：

```text
loss = ((bias_energy - target) ** 2).mean()
```

会有一个关键问题：

```text
adapter zero-init -> bias = 0
energy/norm 在零点没有方向
训练可能完全学不动
```

已修正为：

```text
每层固定一个 random unit direction
score = dot(bias, direction)
loss = ((score - target) ** 2 * miss_weight).mean() + bias_l2
```

这样：

```text
bias = 0 时 score = 0
target > 0 的 hit token 对 bias 有非零梯度
adapter 仍可从严格 zero-init 启动
```

同时保留 energy 作为评估指标：

```text
energy_hit_mean
energy_miss_mean
energy_separation
```

训练稳定性修正：

```text
adapter 参数保持 fp32
projection_features 保持 fp32
adapter 输出在 forward 末尾再 cast 到 VGGT token dtype
```

验证：

```text
fake zero-init adapter unit test:
  bias_absmax_before = 0.0
  last_weight_grad_absmax = 0.0044729211
  last_bias_grad_absmax = 0.0109949801
```

这说明当前 B2 directional loss 可以从 zero-init 启动，不再有 energy-only loss 的零梯度问题。

## B2 判断注意事项

```text
1. trained adapter 后 zero_init_sanity.passed=false 是正常现象。
   adapter 已经不是 zero-init。

2. 理想训练后应看到：
   selected_token_max_abs_diff > 0
   prefix_token_max_abs_diff = 0
   ss/slat diff > 0 但可控

3. 如果 trained adapter 后 ss/slat diff 仍完全为 0，
   说明 adapter 可能没有真正影响 ReconViaGen condition。

4. score separation 是主判断：
   adapter_score_stats.hit_mean > adapter_score_stats.miss_mean

5. energy separation 是辅助判断：
   adapter_energy_stats.hit_mean > adapter_energy_stats.miss_mean

6. hit_ratio 会影响解释。
   hit_ratio 接近 1 时 miss token 太少，hit/miss separation 的意义变弱。
```

## B2 实跑结果分析

已完成：

```text
reconvggt_ar_adapter_a/outputs/ar_20260617_b2_pointprior_adapteronly_s100
reconvggt_ar_adapter_a/outputs/ar_20260617_b2_pointprior_adapteronly_s100/trained_adapter_drift_eval
```

训练配置核心：

```text
max_steps = 100
lr = 1e-4
bias_target_scale = 0.02
direction_seed = 1234
trainable = ProjectionAwareSpatialTokenAdapter only
```

### 结果

B1 输入仍然正确：

```text
projection_features.shape = [1, 7, 1369, 29]
point_count = 820
inside_count_total = 5647
hit_ratio = 0.17416258
```

训练从 zero-init 成功启动：

```text
step 1:
  score_hit_mean = 0
  score_miss_mean = 0
  energy_hit_mean ~= 1e-4
  energy_miss_mean ~= 1e-4

step 100:
  score_hit_mean = 0.01330
  score_miss_mean = -0.00110
  score_separation = 0.01440
  energy_hit_mean = 0.0003570
  energy_miss_mean = 0.0001024
  energy_separation = 0.0002545
```

9.2 drift eval 复算一致：

```text
eval_mode = loaded_adapter_drift
target_dirs_loaded = true
adapter_score_stats.hit_mean = 0.01304
adapter_score_stats.miss_mean = -0.00102
adapter_score_stats.separation = 0.01406
adapter_energy_stats.hit_mean = 0.000350
adapter_energy_stats.miss_mean = 0.000101
adapter_energy_stats.separation = 0.000249
```

adapter 确实影响了 selected spatial tokens，但没有碰 prefix tokens：

```text
selected_token_max_abs_diff:
  layer4  = 0.00160
  layer11 = 0.00168
  layer17 = 0.00160
  layer23 = 0.00176

prefix_token_max_abs_diff = 0 for all selected layers
```

condition drift：

```text
ss_cond_max_abs_diff = 0.25
slat_cond_max_abs_diff = 0.03125
```

### 结论

```text
B2 成功证明：point-prior projection token feature 能驱动 adapter 学到 hit/miss 分离的 spatial-token residual。
这说明 B 路线不是“feature 没信号”或“adapter 学不动”。
```

但当前 `bias_target_scale=0.02` 已经带来明确 ss/slat drift：

```text
ss_cond diff = 0.25
slat diff = 0.03125
```

这不是错误，因为 trained adapter 本来就应该改变 condition；但在进入 sparse/SLAT sampling 或 mesh 前，需要先确定更温和的注入强度。

### 下一步建议

优先做 B2 scale ablation，而不是立刻 B3 downstream：

```text
B2 scale=0.005
B2 scale=0.010
```

判断目标：

```text
score separation 仍 > 0
energy separation 仍 > 0
prefix diff = 0
ss/slat drift 明显低于 scale=0.02
```

如果 `0.005` 已有清楚 score/energy separation 且 drift 更小，优先用它进入 B3。

如果 `0.005` 太弱、`0.010` 可控，选 `0.010`。

如果两者都没有有效 separation，再回到 `0.020`，但 B3 时要加更保守的 adapter scale/gate。

## 10.1-10.2 实跑结果：B2 scale ablation

你已经完成：

```text
scale020 = ar_20260617_b2_pointprior_adapteronly_s100
scale005 = ar_20260617_b2_pointprior_adapteronly_s100_scale005
scale010 = ar_20260617_b2_pointprior_adapteronly_s100_scale010
```

三组都成功加载 adapter checkpoint，并且都保留了 prefix token 不变：

```text
prefix_token_max_abs_diff = 0
target_dirs_loaded = true
eval_mode = loaded_adapter_drift
```

### 结果对比

```text
scale020:
  score_hit_mean = 0.01304
  score_miss_mean = -0.00102
  score_separation = 0.01406
  energy_hit_mean = 0.000350
  energy_miss_mean = 0.000101
  energy_separation = 0.000249
  selected_token_diff ~= 0.00160-0.00176
  ss_cond_max_abs_diff = 0.25
  slat_cond_max_abs_diff = 0.03125

scale005:
  score_hit_mean = 0.00328
  score_miss_mean = -0.00041
  score_separation = 0.00369
  energy_hit_mean = 0.000397
  energy_miss_mean = 0.000147
  energy_separation = 0.000250
  selected_token_diff ~= 0.00357-0.00377
  ss_cond_max_abs_diff = 0.25
  slat_cond_max_abs_diff = 0.03125

scale010:
  score_hit_mean = 0.00702
  score_miss_mean = -0.00103
  score_separation = 0.00805
  energy_hit_mean = 0.000272
  energy_miss_mean = 0.000111
  energy_separation = 0.000161
  selected_token_diff ~= 0.00259-0.00284
  ss_cond_max_abs_diff = 0.25
  slat_cond_max_abs_diff = 0.03125
```

### 关键判断

这轮 ablation 没有出现预期的“scale 越小，condition max drift 越小”：

```text
scale020 / scale005 / scale010:
  ss_cond_max_abs_diff 都是 0.25
  slat_cond_max_abs_diff 都是 0.03125
```

因此不能继续把 `ss_cond_max_abs_diff` 和 `slat_cond_max_abs_diff` 的单个 max 值当作唯一选型标准。它们可能已经被下游 condition module 的局部最大响应、fp16/量化粒度、或者某个少量 token 的最大扰动主导；max 值对注入强度不够敏感。

更细看 selected token diff，`scale005` 反而最大：

```text
scale020 selected diff ~= 0.0017
scale010 selected diff ~= 0.0028
scale005 selected diff ~= 0.0037
```

这说明训练动态不是单调的。降低 target scale 后，AdamW + directional loss + L2 的平衡点没有简单缩小 adapter 输出；所以 `scale005` 不能被认为是更温和版本。

### 当前最合理选择

如果下一步必须选一个 B3 候选，我建议选：

```text
scale010
```

理由：

```text
1. score separation = 0.00805，明显高于 scale005 的 0.00369；
2. selected token diff 低于 scale005；
3. energy separation 仍为正；
4. 比 scale020 更保守，避免一开始用最强注入；
5. prefix token 仍完全不变。
```

`scale020` 是 stronger signal 备选。如果 `scale010` downstream 没有效果，可以回到 `scale020`，但不建议第一轮 B3 直接用最强注入。

`scale005` 暂时不建议进入 B3：它的 score separation 太弱，而且 selected token diff 反而最大。

### 下一步建议

不要直接跑大规模 mesh。下一步建议分两步：

```text
第一步：补 drift 分布统计
  当前只看 max_abs_diff 不够。
  建议在 run_b_projection_sanity.py 里增加：
    mean_abs_diff
    p95_abs_diff
    p99_abs_diff
    l2_mean_diff
  分别统计 selected token、ss_cond、slat_cond。

第二步：B3 小样本 downstream smoke
  用 scale010 checkpoint。
  先只跑 1 个 AR session / 少量 view。
  目标不是证明 mesh 提升，而是确认：
    adapter 注入不会让 sparse/SLAT sampling 崩；
    projection-aware residual 是否能影响 sparse 或 mesh 的可见区域。
```

B3 第一轮不建议直接训练 ReconViaGen 主干。当前 B2 只证明了 condition residual 可学习，还没有证明它能改善生成。因此 B3 应先做 eval-time adapter injection smoke，再决定是否训练主干或做更强的 block-wise adapter。

## 11.3 实跑结果：B2 drift 分布统计

你已经复跑了三档 trained adapter drift stats：

```text
scale020
scale010
scale005
```

三组共同确认：

```text
eval_mode = loaded_adapter_drift
target_dirs_loaded = true
prefix_token_max_abs_diff = 0 for layers 4,11,17,23
```

这说明 adapter 只作用在 spatial tokens，没有污染 prefix/global/register tokens。

### scale020

```text
score_separation = 0.014055
energy_separation = 0.000249

ss_cond:
  max  = 0.25
  mean = 0.0008378
  p95  = 0.002441
  p99  = 0.004639
  rmse = 0.001473

slat_cond:
  max  = 0.03125
  mean = 0.00002779
  p95  = 0
  p99  = 0.000977
  rmse = 0.000320

selected token:
  layer mean roughly 0.000084-0.000123
  layer p99 roughly 0.000481-0.000589
```

`scale020` 的优点是 signal 最强，缺点是 ss/slat 分布扰动也偏大。

### scale010

```text
score_separation = 0.008052
energy_separation = 0.000161

ss_cond:
  max  = 0.25
  mean = 0.0007862
  p95  = 0.001953
  p99  = 0.004028
  rmse = 0.001404

slat_cond:
  max  = 0.03125
  mean = 0.00002497
  p95  = 0
  p99  = 0.000977
  rmse = 0.000296

selected token:
  layer mean roughly 0.000061-0.000107
  layer p99 roughly 0.000339-0.000597
```

`scale010` 的 score separation 明显高于 `scale005`，同时 ss/slat mean、p95、p99、rmse 基本都低于 `scale020`。这是目前最平衡的一档。

### scale005

```text
score_separation = 0.003689
energy_separation = 0.000250

ss_cond:
  max  = 0.25
  mean = 0.0008542
  p95  = 0.002441
  p99  = 0.004883
  rmse = 0.001511

slat_cond:
  max  = 0.03125
  mean = 0.00002497
  p95  = 0
  p99  = 0.000977
  rmse = 0.000292

selected token:
  layer max roughly 0.00357-0.00377
  layer p99 roughly 0.000769-0.000883
```

`scale005` 并没有变成更温和的选择。它的 score separation 太弱，而且 selected-token max/p99 和 ss p99/rmse 反而更差。

### 结论

可以进入 B3，但只进入：

```text
B3 eval-time adapter injection smoke
```

不建议直接进入：

```text
ReconViaGen 主干训练
大规模 mesh sweep
直接宣称质量提升
```

B3 第一轮候选：

```text
scale010
```

理由：

```text
1. score_separation = 0.008052，信号足够；
2. 比 scale005 明显强；
3. ss_cond mean / p95 / p99 / rmse 都低于 scale020；
4. slat_cond mean / rmse 低于 scale020；
5. prefix tokens 保持 0 diff；
6. selected token 分布整体比 scale005 更稳。
```

`scale020` 保留为 stronger-signal 对照，不作为 B3 第一选择。

### 下一步建议

B3 目标应非常窄：

```text
验证 trained projection-aware adapter 注入后：
1. ReconViaGen sparse / SLAT condition 能正常跑；
2. sparse / mesh 不崩；
3. 与 no-adapter baseline 相比，输出确实发生可测变化；
4. 如果有几何评估，再观察可见区域是否朝 AR point-prior 靠近。
```

第一轮 B3 不要求质量提升，只要求跑通并证明 adapter injection 对 downstream 有可控影响。

推荐顺序：

```text
1. B3-0：no-adapter baseline，固定同一 AR session / same views / same seed。
2. B3-1：scale010 adapter eval-time injection，同样设置。
3. B3-2：如果 B3-1 不崩，再补 scale020 stronger-signal 对照。
4. 只在 B3-1/B3-2 有可解释差异后，再考虑训练主干或更复杂 adapter。
```

如果 B3-1 完全无变化，说明当前 token residual 还没有传导到生成结果，需要改注入位置或增加 adapter scale/gate。

如果 B3-1 明显破坏输出，说明 condition residual 仍太强或方向不对，需要更小 runtime scale，而不是继续训练更强 adapter。

如果 B3-1 有可控变化、B3-2 更强但不崩，才说明这条 ReconVGGT AR adapter 路线有进入真实 downstream 优化的价值。

## 12.2-12.3 实跑结果：B3 adapter injection 已传导到 sparse / mesh

你已经完成：

```text
B3-0 no-adapter sparse baseline
B3-1 scale010 adapter sparse, runtime_scale=1.0
B3-1 scale010 adapter sparse, runtime_scale=0.5
B3-0 no-adapter mesh
B3-1 scale010 adapter mesh, runtime_scale=0.5
```

### Sparse 结果

```text
no-adapter baseline:
  coord_count = 79219
  component_count = 6
  largest_component_ratio = 0.99951

scale010 adapter, runtime_scale=1.0:
  coord_count = 45935
  component_count = 13
  largest_component_ratio = 0.97240

scale010 adapter, runtime_scale=0.5:
  coord_count = 57808
  component_count = 10
  largest_component_ratio = 0.99355
```

与 no-adapter baseline 的坐标集合对比：

```text
scale010, runtime_scale=1.0:
  IoU = 0.36415
  baseline_keep = 0.42173
  adapter_keep = 0.72731
  baseline_only = 45810
  adapter_only = 12526

scale010, runtime_scale=0.5:
  IoU = 0.42385
  baseline_keep = 0.51490
  adapter_keep = 0.70561
  baseline_only = 38429
  adapter_only = 17018
```

这说明 B3 adapter injection 已经明确传导到 ReconViaGen sparse sampling。它不是只在 condition 数值上变化，而是改变了最终 sparse coords。

但 `runtime_scale=1.0` 明显过强：

```text
coord_count 从 79219 降到 45935
component_count 从 6 增到 13
largest_component_ratio 从 0.9995 降到 0.9724
```

`runtime_scale=0.5` 更稳：

```text
coord_count = 57808
component_count = 10
largest_component_ratio = 0.99355
```

它仍然显著改变 sparse，但没有像 scale=1.0 那样明显破坏连通性。

### Mesh 结果

```text
no-adapter mesh:
  sparse coord_count = 79219
  mesh vertex_count = 1336970
  mesh face_count = 2674452

scale010 adapter mesh, runtime_scale=0.5:
  sparse coord_count = 57808
  mesh vertex_count = 1011680
  mesh face_count = 2023900
```

mesh 规模随 sparse 明显收缩：

```text
vertex_count 下降约 24.3%
face_count 下降约 24.3%
```

这进一步说明 adapter 注入对 downstream mesh 有实质影响。

当前还不能说质量提升，因为还没有做 mask-aware object-centric 输入、点云距离、可见区域一致性、黑/暗片指标或视觉检查。但 B3 的最低目标已经达成：

```text
adapter injection 不仅能跑通，而且能改变 sparse 和 mesh。
```

### 关于 mask

本轮 B3 没有使用 mask。脚本读的是：

```text
${DATASET}/images
```

并直接转 RGB。这个设置适合做“adapter 是否传导到 downstream”的 smoke test，因为它尽量少引入变量。

但最终目标是 object-centric mesh，因此后续必须使用 mask。当前数据集确实有：

```text
${DATASET}/masks/frame_*.png
```

不过要注意一个关键问题：

```text
不能直接用 mask+crop，然后继续用当前 projection token feature。
```

原因是当前 projection feature 是按原始相机内参和原图坐标投影到 37x37 token grid：

```text
original image / camera intrinsics -> resized full-frame 518x518 -> 37x37 tokens
```

如果对输入图像做 crop，图像 token 坐标系变了，但 projection feature 仍然在原图坐标系，二者会错位。

因此下一步 mask 方案应先用：

```text
mask apply, no crop
```

也就是只把背景置黑，保持整图坐标系不变。这样 image tokens 和 projection tokens 仍然对齐。

后续如果要做真正 `mask+crop`，需要同步把 projection feature 的坐标也按 crop window 重映射，这应作为单独版本实现。

### 结论

B3 可以继续，但下一步不要直接扩训练。推荐先做 mask-aware B3：

```text
1. masked no-adapter sparse baseline
2. masked scale010 runtime_scale=0.5 sparse
3. masked sparse 对比
4. 如果不崩，再 masked mesh smoke
```

mask 模式先使用：

```text
mask_mode = apply
mask_background = black
不 crop
```

如果 masked B3 中 adapter 仍然能产生可控变化，并且 sparse/mesh 不崩，才进入更有意义的 object-centric 评估。

## 13.2-13.5 实跑结果：masked B3 object-centric smoke

你已经完成：

```text
masked no-adapter sparse
masked scale010 runtime_scale=0.5 sparse
masked no-adapter mesh
masked scale010 runtime_scale=0.5 mesh
```

这里的 mask 使用方式是：

```text
mask_mode = apply
mask_background = black
不 crop
```

平均前景比例：

```text
foreground_ratio_mean = 0.17688
foreground_ratio_min  = 0.13503
foreground_ratio_max  = 0.21764
```

### Masked Sparse

```text
masked no-adapter:
  coord_count = 9991
  component_count = 4
  largest_component_ratio = 0.96657

masked scale010, runtime_scale=0.5:
  coord_count = 10067
  component_count = 6
  largest_component_ratio = 0.96672
```

坐标集合对比：

```text
masked no-adapter vs masked scale010 a0.5:
  IoU = 0.96975
  baseline_keep = 0.98839
  adapter_keep = 0.98093
  baseline_only = 116
  adapter_only = 192
```

这说明在 masked object-centric 输入下，adapter injection 仍然传导到了 sparse，但影响很小。它不是完全无效，但远小于 unmasked 条件下的变化。

### Masked Mesh

```text
masked no-adapter mesh:
  sparse coord_count = 9991
  vertex_count = 159264
  face_count = 318512

masked scale010 a0.5 mesh:
  sparse coord_count = 10064
  vertex_count = 161614
  face_count = 323208
```

mesh 规模变化：

```text
vertex_count +2350 约 +1.48%
face_count   +4696 约 +1.47%
```

坐标集合对比：

```text
masked no-adapter mesh vs masked scale010 a0.5 mesh:
  IoU = 0.95163
  baseline_keep = 0.97878
  adapter_keep = 0.97168
  baseline_only = 212
  adapter_only = 285
```

这说明 adapter 对 masked mesh 也有影响，但仍是轻微扰动。

### Mask 本身的影响

mask 对 ReconVGGT 的影响非常大：

```text
unmasked no-adapter sparse:
  coord_count = 79219
  component_count = 6
  largest_component_ratio = 0.99951

masked no-adapter sparse:
  coord_count = 9991
  component_count = 4
  largest_component_ratio = 0.96657
```

mesh 同样大幅收缩：

```text
unmasked no-adapter mesh:
  vertex_count = 1336970
  face_count = 2674452

masked no-adapter mesh:
  vertex_count = 159264
  face_count = 318512
```

unmasked 与 masked sparse 的坐标 IoU 很低：

```text
unmasked no-adapter vs masked no-adapter:
  IoU = 0.05332

unmasked scale010 a0.5 vs masked scale010 a0.5:
  IoU = 0.05102
```

这说明当前 object-centric 行为主要由 mask 输入决定，而不是 adapter 决定。

### 结论

masked B3 的核心结论：

```text
1. mask-aware full-frame input 能把 ReconVGGT 输出从大范围背景结构压到更 object-centric 的 sparse/mesh；
2. scale010 adapter a0.5 在 masked 条件下没有崩；
3. adapter 对 masked sparse/mesh 有可测变化；
4. 但这个变化很小，当前主要因素是 mask，而不是 point-prior adapter。
```

因此，现在还不能说 point-prior adapter 已经改善了 masked object mesh。更准确的说法是：

```text
point-prior projection adapter 已经成功接入 ReconVGGT downstream；
在 unmasked 条件下影响很强；
在 masked object-centric 条件下影响较弱，需要增强或改评价指标。
```

### 下一步建议

不要马上训练 ReconVGGT 主干。当前应先回答两个问题：

```text
Q1: masked 条件下 adapter 是否只是 scale 太弱？
Q2: adapter 改动是否朝 AR point-prior 更一致，而不是只改变 coord_count？
```

建议下一步按以下顺序：

```text
1. masked stronger-signal sparse sweep:
   scale010 runtime_scale=1.0
   scale020 runtime_scale=0.5
   scale020 runtime_scale=1.0

2. 给 B3 report 加 point-prior alignment metrics:
   sparse coord 到 AR/SLAM points 的 nearest-neighbor distance
   within radius ratio
   projection support / mask support

3. 只对最有希望的一档跑 masked mesh。

4. 如果 stronger signal 仍然只有极小变化：
   当前 adapter 注入位置/训练目标不足，应改成更直接的 geometry-aware condition，而不是继续加大训练。
```

目前最优先的是第 2 点：没有 point-prior alignment，只看 coord_count / component / mesh vertex count 无法判断 adapter 改动是不是“更对”。如果只是多了 192 个 coords，但不知道它们是否靠近 AR point-prior，就不能作为论文结论。

短期建议：

```text
先补 B3 point-prior alignment metrics；
再跑 masked stronger-signal sparse sweep；
最后才决定是否进入 mesh 或训练。
```

## 14.6.2 masked scale020 a0.5 mesh 结果

### 实验配置

本轮只补跑了 sparse sweep 中最有信息量的一档：

```text
adapter checkpoint:
  ar_20260617_b2_pointprior_adapteronly_s100/checkpoints/last.ckpt

runtime_scale:
  0.5

input:
  mask_mode=apply
  mask_background=black
  full-frame，不 crop

output:
  ar_20260617_b3_masked_scale020_mesh_seed42_a050_metrics
```

### sparse / prior alignment

与 masked no-adapter baseline 对比：

```text
masked no-adapter:
  coord_count = 9991
  component_count = 4
  largest_component_ratio = 0.966570
  within_prior_radius_ratio = 0.311080
  projection_any_mask_hit_ratio = 0.499750
  visible_outside_mask_event_ratio = 0.695431

masked scale020 a0.5:
  coord_count = 10091
  component_count = 4
  largest_component_ratio = 0.966505
  within_prior_radius_ratio = 0.312853
  projection_any_mask_hit_ratio = 0.500446
  visible_outside_mask_event_ratio = 0.697245
```

变化幅度：

```text
coord_count:
  +100

within_prior_radius_ratio:
  +0.001773

projection_any_mask_hit_ratio:
  +0.000696

visible_outside_mask_event_ratio:
  +0.001814  # 变差
```

结论：adapter 确实带来可测变化，但 prior/mask 改善非常弱，而且 outside-mask event 同时变差。这不是一个干净的 point-prior alignment 增益。

### mesh 对比

```text
masked no-adapter mesh:
  vertex_count = 159264
  face_count = 318512
  bbox_extent = [0.4439, 0.5710, 0.9871]

masked scale010 a0.5 mesh:
  vertex_count = 161614
  face_count = 323208
  bbox_extent = [0.4442, 0.5711, 0.8989]

masked scale020 a0.5 mesh:
  vertex_count = 163096
  face_count = 326152
  bbox_extent = [0.4449, 0.5759, 0.8975]
```

scale020 a0.5 相比 baseline：

```text
vertex_count:
  +3832

face_count:
  +7640

xy extent:
  基本不变

z extent:
  从 0.9871 降到 0.8975
```

这更像轻微增加/改动局部 sparse 后导致 mesh 尺寸和 z 方向范围变化，不足以说明 mesh 几何更贴近 AR point-prior。

### 是否继续 14.6

不建议继续完整 14.6 mesh sweep。

```text
14.6.1 scale010 a1.0:
  sparse 改善不干净，outside-mask 更差，不值得跑 mesh。

14.6.3 scale020 a1.0:
  更 aggressive，outside-mask 更差，也不值得跑 mesh。
```

当前 B3 的结论已经足够：

```text
1. masked input 是 object-centric 输出的主要来源；
2. B2 point-prior token bias adapter 已经接通 downstream；
3. adapter 对 sparse/mesh 有可测但很小的影响；
4. 这个影响没有稳定转化成 AR point-prior / mask-support 改善。
```

### 是否进入 B4

建议进入 B4，但不是扩大 B3 mesh，也不是继续调 runtime scale。

进入 B4 的理由是：

```text
B2 的训练目标是 token-level direction-score proxy；
它能让 adapter 影响 VGGT tokens，
但没有强约束最终 sparse 更贴近 AR point-prior / mask support。
```

所以 B4 应该改成：

```text
adapter 仍然小规模、zero-init；
但训练/选择目标改成直接面向 sparse/prior alignment：
  within_prior_radius_ratio
  projection_any_mask_hit_ratio
  visible_outside_mask_event_ratio
  sparse component / coord count stability
```

第一版 B4 不建议做大训练。建议先做最小 smoke：

```text
1. 冻结 ReconVGGT / TRELLIS 主干；
2. 只训练或搜索 adapter runtime/update；
3. 每隔少量 step 或候选 scale 跑 sparse-only；
4. 用 prior-alignment report 作为 early selection；
5. 只有 sparse 指标出现明确改善后再跑 mesh。
```

如果 B4 的 sparse/prior-alignment objective 仍然无法带来稳定提升，就应停止 eval-time token-bias 路线，转向：

```text
1. crop-aware object-centric projection；
2. 训练期接入 VGGT/SLAT condition；
3. 更直接的 geometry-aware sparse/mesh refinement。
```

## 15. 第 14 节总评与下一步路线

### 14 的最终结论

第 14 节已经回答了当前最关键的问题：

```text
B2 projection-token adapter 是否在 masked object-centric ReconVGGT 路径中产生了有效几何改善？
```

答案是：

```text
没有形成足够强、足够干净的几何改善。
```

证据如下：

```text
1. masked input 本身把输出从 unmasked 的百万级 mesh 压到 16 万 vertex 左右；
   这说明 object-centric 行为主要来自 mask，而不是 adapter。

2. adapter 确实接通 downstream：
   sparse / mesh 都有可测变化。

3. 但 adapter 的 prior-alignment 改善非常弱：
   within_prior_radius_ratio 只提升约 +0.0018；
   projection_any_mask_hit_ratio 只提升约 +0.0007。

4. 同时 visible_outside_mask_event_ratio 变差：
   说明新增/改变的 sparse 并不是稳定地向 mask-supported / point-prior-supported 区域收敛。

5. mesh 侧只是轻微改变 vertex/face 和 bbox；
   还不能证明 AR point-prior 改善了物体几何。
```

因此，不建议继续做：

```text
1. 14.6 的完整 mesh sweep；
2. 继续调 runtime_scale；
3. 继续堆 B2 direction-score token adapter；
4. 直接对 ReconVGGT / TRELLIS 做大规模微调。
```

当前最准确的阶段性结论是：

```text
AR point-prior -> VGGT token bias -> ReconVGGT downstream 这条链路已经打通；
但 B2 的 proxy supervision 太弱，不能稳定转化为 sparse/mesh 的 point-prior alignment。
```

### 我认为下一步的核心问题

当前不是“adapter 有没有影响”的问题，而是：

```text
adapter 影响的 sparse delta 是否是正确的？
```

现在整体指标只告诉我们最终 sparse 稍微变了，但还没有回答：

```text
adapter 新增的 voxel 是否更靠近 AR prior？
adapter 删除的 voxel 是否更远离 AR prior？
adapter 新增的 voxel 是否更受 mask projection 支持？
adapter 是否只是改变整体采样轨迹，而不是做几何选择？
```

如果这些 delta-level 问题不先回答，直接进入 B4 训练会很容易变成黑箱。

### 下一步优先级

我建议按下面顺序推进。

#### 第一步：B4.0 delta-level prior alignment 诊断

先不训练，先把 B3 报告补强：

```text
对比 baseline coords 与 adapter coords：
  added = adapter - baseline
  removed = baseline - adapter
  kept = adapter ∩ baseline

分别统计：
  prior_distance_mean / median / p90
  within_prior_radius_ratio
  projection_any_mask_hit_ratio
  projection_support_mean
  visible_outside_mask_event_ratio
  component attribution
```

判断标准：

```text
如果 added 比 removed 更靠近 prior、更高 mask support、更低 outside-mask：
  adapter 虽弱但方向正确，可以进入 B4 训练。

如果 added 和 removed 没有几何优势，甚至 added 更差：
  B2 token-bias 路线本身不是有效几何控制，不应继续加训练。
```

这是下一步最应该先做的诊断。

#### 第二步：crop-aware object-centric projection

当前 masked 输入是：

```text
full-frame + black background
不 crop
```

这样做的好处是 projection grid 仍然和原图坐标对齐；坏处是物体只占约 17% 前景，projection signal 在 37x37 token grid 上很稀疏。

如果要让 point-prior 更强地影响 object mesh，后续需要实现：

```text
mask bbox crop / padding / resize
同步重映射 camera intrinsics 和 projected token coordinates
projection feature 在 crop 后 token grid 上重新计算
```

这一步很重要，因为 ReconVGGT / TRELLIS 本来更偏 object-centric 输入。full-frame black background 虽然安全，但可能把有效点云约束稀释得太厉害。

#### 第三步：B4.1 sparse/prior-alignment objective smoke

只有 B4.0 证明 adapter delta 方向至少有一点正确，才进入训练型 B4。

第一版 B4 不应该大训练，只做 sparse-only smoke：

```text
冻结 ReconVGGT / TRELLIS 主干；
只训练小 adapter；
不跑 mesh；
固定少量 seeds；
每隔少量 step 跑 sparse；
用 prior-alignment 指标 early select。
```

训练目标不再只用 token-level direction score，而要显式靠近 downstream 几何指标：

```text
希望提升：
  within_prior_radius_ratio
  projection_any_mask_hit_ratio
  projection_keep_ratio

希望降低：
  visible_outside_mask_event_ratio
  small isolated components

希望保持：
  coord_count 不爆炸
  component_count 不碎
  largest_component_ratio 不退化
```

如果不能做完全可微的 sparse sampler 训练，第一版也可以做更保守的候选选择：

```text
训练多个小 adapter / 多个 scale / 多个 layer-set；
全部 sparse-only；
按 prior-alignment report 选择；
只对胜出配置跑 mesh。
```

#### 第四步：如果 B4.1 仍无效，停止 eval-time token bias

如果 B4.0 / B4.1 证明 adapter delta 没有稳定几何方向，则应停止当前路线：

```text
point-prior token bias adapter
eval-time injection
direction-score proxy
```

后续应转向更直接的路线：

```text
1. crop-aware object-centric projection；
2. VGGT aggregated tokens 的 geometry adapter 训练期接入；
3. SLAT / sparse decoder 侧的 geometry-aware conditioning；
4. mesh 后处理或 sparse refinement 中显式使用 AR prior。
```

### 当前我的建议

短期不建议继续跑 mesh，也不建议直接开始重训练。

我建议下一步就是：

```text
B4.0:
  实现 baseline-vs-adapter delta prior-alignment report；
  对已经跑完的 scale010 / scale020 masked sparse 结果复算 added / removed / kept 指标；
  判断 adapter 改动方向是否几何正确。
```

如果 B4.0 结论是 positive，再进入：

```text
B4.1:
  sparse-only adapter selection / light training；
  目标从 token proxy 改为 prior-alignment selection。
```

如果 B4.0 结论是 negative，则当前 B2/B3 路线不值得继续，应该优先做 crop-aware projection 或换到更直接的 sparse/SLAT geometry conditioning。

## 16. B4.1 mask-aware gated adapter 结果

### 实验目的

B4.1 把旧 adapter 从：

```text
pose + unmasked COLMAP point projection -> token bias
```

改成：

```text
pose
+ mask-filtered object point projection
+ token foreground mask
-> gated token bias
```

目标是验证：

```text
新增 voxel 是否更靠近 AR prior；
新增 voxel 是否更有 mask support；
新增 voxel 是否不再增加 outside-mask event。
```

### 16.7 汇总结论

旧 adapter 中最干净的是 `scale010 a0.5`：

```text
changed voxels:
  added = 16
  removed = 16

added_minus_removed_within_prior_radius_ratio = +0.1875
added_minus_removed_projection_any_mask_hit_ratio = +0.125
added_minus_removed_visible_outside_mask_event_ratio = -0.00742
```

但这档只改变 16 个 voxel，影响太小，不能作为 mesh 改善路线。

旧 `scale020 a0.5` 和 `scale020 a1.0` 的问题是：

```text
scale020 a0.5:
  added 更靠近 prior；
  mask hit 没有优势；
  outside-mask 变差。

scale020 a1.0:
  added 更靠近 prior、mask hit 更高；
  outside-mask 仍明显变差。
```

B4.1 mask-aware 的结果：

```text
maskaware a0.5:
  added = 297
  removed = 174
  added_minus_removed_within_prior_radius_ratio = +0.0320
  added_minus_removed_prior_distance_mean = -0.4965
  added_minus_removed_projection_any_mask_hit_ratio = +0.0121
  added_minus_removed_visible_outside_mask_event_ratio = +0.0192

maskaware a1.0:
  added = 382
  removed = 244
  added_minus_removed_within_prior_radius_ratio = -0.1336
  added_minus_removed_prior_distance_mean = +1.1630
  added_minus_removed_projection_any_mask_hit_ratio = -0.2833
  added_minus_removed_visible_outside_mask_event_ratio = +0.0467
```

### 判断

`maskaware a0.5` 有一点进步：

```text
新增 voxel 比 removed voxel 更靠近 AR prior；
新增 voxel 的 mask_hit 也略高。
```

但它仍然失败在最关键的约束：

```text
outside-mask event 继续变差。
```

`maskaware a1.0` 则是明确失败：

```text
更远离 prior；
mask_hit 下降；
outside-mask 更差。
```

因此当前结论是：

```text
1. 停止旧 adapter 路线的 mesh / runtime sweep；
2. 不跑 maskaware a0.5 mesh；
3. 丢弃 maskaware a1.0；
4. 进入更严格的 mask-aware ablation。
```

### 下一步

下一步不应该扩大训练，而应该收紧 gate：

```text
1. gate 从 binary token hit 改成 foreground ratio:
   adapter_gate_feature_index = -2

2. gate 加强：
   adapter_gate_power = 2.0 或 3.0

3. token 前景阈值提高：
   token_mask_min_ratio = 0.10 或 0.15

4. 降低 bias 目标强度：
   bias_target_scale = 0.01

5. miss_weight 提高：
   miss_weight = 1.0
```

判断标准保持不变：

```text
added_minus_removed_within_prior_radius_ratio > 0
added_minus_removed_projection_any_mask_hit_ratio > 0
added_minus_removed_visible_outside_mask_event_ratio <= 0
adapter_minus_baseline_visible_outside_mask_event_ratio <= 0
```


如果 strict foreground-ratio gate 仍然让 outside-mask 变差，就说明 per-view token gate 仍不够，需要转 B4.2：

```text
多视图一致的 object point support：
  先按 3D point 的多视图 mask support 过滤点云；
  再投影到 token；
  而不是每个 view 单独过滤后直接产生 token bias。
```

## 十七、2026-07-07 strict foreground-ratio gate 结果

### 运行状态

17 节命令检查结果：

```text
17.2 p2 / token_min=0.10 训练：已完成
17.3 p3 / token_min=0.15 训练：已完成
17.4 p2 a0.5 sparse eval：已完成
17.4 p2 a1.0 sparse eval：原缺失，已补跑完成
17.5 p3 a0.5 sparse eval：已完成
17.5 p3 a1.0 sparse eval：已完成
17.6 p2 a0.5 delta：已完成
17.6 p2 a1.0 delta：原缺失，已补跑完成
17.6 p3 a0.5 delta：已完成
17.6 p3 a1.0 delta：原缺失，已补跑完成
```

### 训练结果

两组 strict foreground-ratio gate 都能正常训练，初始 zero-diff 仍保持：

```text
p2 / token_min=0.10:
  score separation = 0.007633
  energy separation = 0.000235
  hit_ratio = 0.111865
  gate mean = 0.176914
  gate nonzero_ratio = 0.207242
  ss/slat initial diff = 0

p3 / token_min=0.15:
  score separation = 0.008040
  energy separation = 0.000233
  hit_ratio = 0.111865
  gate mean = 0.176914
  gate nonzero_ratio = 0.207242
  ss/slat initial diff = 0
```

说明这次问题不在训练启动或 adapter 接线；adapter 确实学到了 mask-gated token bias。

### sparse eval

四组 sparse eval 的整体 sparse 指标：

```text
p2 a0.5:
  coord_count = 10126
  component_count = 4
  within_prior_radius_ratio = 0.315129
  projection_any_mask_hit_ratio = 0.501383
  visible_outside_mask_event_ratio = 0.697572

p2 a1.0:
  coord_count = 10126
  component_count = 4
  within_prior_radius_ratio = 0.315129
  projection_any_mask_hit_ratio = 0.501383
  visible_outside_mask_event_ratio = 0.697572

p3 a0.5:
  coord_count = 10118
  component_count = 5
  within_prior_radius_ratio = 0.313402
  projection_any_mask_hit_ratio = 0.501384
  visible_outside_mask_event_ratio = 0.697261

p3 a1.0:
  coord_count = 10053
  component_count = 6
  within_prior_radius_ratio = 0.312345
  projection_any_mask_hit_ratio = 0.501243
  visible_outside_mask_event_ratio = 0.696172
```

`p2 a0.5` 和 `p2 a1.0` 输出完全一致，说明在该 checkpoint 下 runtime scale 0.5 到 1.0 没有改变 decoded sparse set；这可能来自 bias 在当前采样/阈值下未跨过更多 sparse 决策边界。

### delta 诊断

和 masked no-adapter baseline 对比：

```text
p2 a0.5 / a1.0:
  added = 363
  removed = 228
  set IoU = 0.942921
  added_minus_removed_within_prior_radius_ratio = +0.091634
  added_minus_removed_prior_distance_mean = -0.761807
  added_minus_removed_projection_any_mask_hit_ratio = +0.025881
  added_minus_removed_visible_outside_mask_event_ratio = +0.006309
  adapter_minus_baseline_visible_outside_mask_event_ratio = +0.002141

p3 a0.5:
  added = 305
  removed = 178
  set IoU = 0.953089
  added_minus_removed_within_prior_radius_ratio = +0.042807
  added_minus_removed_prior_distance_mean = -0.516398
  added_minus_removed_projection_any_mask_hit_ratio = +0.028366
  added_minus_removed_visible_outside_mask_event_ratio = +0.012425
  adapter_minus_baseline_visible_outside_mask_event_ratio = +0.001830

p3 a1.0:
  added = 100
  removed = 38
  set IoU = 0.986324
  added_minus_removed_within_prior_radius_ratio = -0.006316
  added_minus_removed_prior_distance_mean = -0.655739
  added_minus_removed_projection_any_mask_hit_ratio = +0.035789
  added_minus_removed_visible_outside_mask_event_ratio = +0.047223
  adapter_minus_baseline_visible_outside_mask_event_ratio = +0.000741
```

### 结论

strict foreground-ratio gate 没有达到继续条件。

它改善了两类指标：

```text
1. added 点通常比 removed 点更靠近 AR prior；
2. added 点通常有更高 projection mask hit。
```

但它没有解决核心失败点：

```text
visible_outside_mask_event_ratio 仍然为正。
```

也就是说 adapter 的改变方向仍会增加 mask 外可见事件。对于以物体为中心的 AR reconstruction，这比 prior 距离小幅改善更关键。当前 per-view token gate 仍然只是在“投影到某个 view 的局部 token”上做 gating，没有保证 3D point 本身是多视图一致的 object point。

因此当前判断是：

```text
停止旧 adapter / B4.1 per-view token gate 路线的 mesh 和 sweep；
不继续扩大 runtime scale；
不继续加训练步数；
转入 B4.2 多视图一致 object point support。
```

### 下一步建议

B4.2 应该把过滤顺序前移到 3D point 级别：

```text
1. 对每个 COLMAP / AR prior 3D point，投影到所有可用 view；
2. 统计该 3D point 在多少 view 内落在 object mask 内；
3. 只保留满足 min_support_views / min_support_ratio 的 object-supported 3D points；
4. 再把这些 3D points 投影到 token grid；
5. token bias 只由多视图一致 object point support 生成。
```

建议先实现一个只做诊断的版本：

```text
B4.2a:
  不训练；
  输出 point-level mask support 分布；
  比较原 unfiltered points、per-view filtered points、multi-view object-supported points。

B4.2b:
  用 B4.2a 的 object-supported points 重新训练 adapter；
  只跑 sparse + delta；
  不跑 mesh。
```

判断标准不变：

```text
added_minus_removed_within_prior_radius_ratio > 0
added_minus_removed_projection_any_mask_hit_ratio > 0
added_minus_removed_visible_outside_mask_event_ratio <= 0
adapter_minus_baseline_visible_outside_mask_event_ratio <= 0
```

## 十八、2026-07-07 B4.2 多视图一致 object point support 结果

### 运行状态

18 节全部完成：

```text
18.2 support2 / ratio0.50 训练：已完成
18.3 support3 / ratio0.60 训练：已完成
18.4 support2 sparse eval：a0.5 / a1.0 已完成
18.5 support3 sparse eval：a0.5 / a1.0 已完成
18.6 delta 诊断：四组已完成
```

### 训练端 point support 过滤

B4.2 的 point-level 多视图 mask support 过滤确实生效：

```text
support2 / ratio0.50:
  point_count_before = 820
  point_count_after = 707
  kept_ratio = 0.8622
  kept_mask_hit_count_mean = 5.4767
  kept_support_ratio_mean = 0.7909
  score separation = 0.00736
  energy separation = 0.000240

support3 / ratio0.60:
  point_count_before = 820
  point_count_after = 493
  kept_ratio = 0.6012
  kept_mask_hit_count_mean = 6.1907
  kept_support_ratio_mean = 0.8913
  score separation = 0.00724
  energy separation = 0.000285
```

说明：

```text
1. 多视图 object point support 过滤没有代码接线问题；
2. support3 比 support2 更严格，保留点更少、更 object-consistent；
3. adapter 仍然能训练出 hit/miss separation。
```

### sparse eval

整体 sparse 指标如下：

```text
support2 a0.5:
  coord_count = 10126
  component_count = 4
  within_prior_radius_ratio = 0.31355
  projection_any_mask_hit_ratio = 0.50049
  visible_outside_mask_event_ratio = 0.69734

support2 a1.0:
  coord_count = 10086
  component_count = 5
  within_prior_radius_ratio = 0.31241
  projection_any_mask_hit_ratio = 0.50050
  visible_outside_mask_event_ratio = 0.69741

support3 a0.5:
  coord_count = 10073
  component_count = 6
  within_prior_radius_ratio = 0.31222
  projection_any_mask_hit_ratio = 0.49945
  visible_outside_mask_event_ratio = 0.69751

support3 a1.0:
  coord_count = 10090
  component_count = 6
  within_prior_radius_ratio = 0.31229
  projection_any_mask_hit_ratio = 0.50059
  visible_outside_mask_event_ratio = 0.69723
```

相比 B4.1 strict gate，B4.2 没有带来清晰的整体 sparse 改善。support3 更严格，但没有让 decoded sparse 更贴合 object mask。

### delta 诊断

和 masked no-adapter baseline 对比：

```text
support2 a0.5:
  added = 333
  removed = 198
  set IoU = 0.94857
  added_minus_removed_within_prior_radius_ratio = +0.03331
  added_minus_removed_projection_any_mask_hit_ratio = -0.01024
  added_minus_removed_visible_outside_mask_event_ratio = +0.00626
  adapter_minus_baseline_visible_outside_mask_event_ratio = +0.00191

support2 a1.0:
  added = 286
  removed = 191
  set IoU = 0.95359
  added_minus_removed_within_prior_radius_ratio = +0.01992
  added_minus_removed_projection_any_mask_hit_ratio = -0.00073
  added_minus_removed_visible_outside_mask_event_ratio = +0.02861
  adapter_minus_baseline_visible_outside_mask_event_ratio = +0.00198

support3 a0.5:
  added = 241
  removed = 159
  set IoU = 0.96091
  added_minus_removed_within_prior_radius_ratio = +0.02299
  added_minus_removed_projection_any_mask_hit_ratio = -0.04776
  added_minus_removed_visible_outside_mask_event_ratio = +0.04553
  adapter_minus_baseline_visible_outside_mask_event_ratio = +0.00208

support3 a1.0:
  added = 288
  removed = 189
  set IoU = 0.95359
  added_minus_removed_within_prior_radius_ratio = +0.00926
  added_minus_removed_projection_any_mask_hit_ratio = +0.00132
  added_minus_removed_visible_outside_mask_event_ratio = +0.02236
  adapter_minus_baseline_visible_outside_mask_event_ratio = +0.00180
```

### 判断

B4.2 没有达到继续条件。

继续条件是：

```text
added_minus_removed_within_prior_radius_ratio > 0
added_minus_removed_projection_any_mask_hit_ratio > 0
added_minus_removed_visible_outside_mask_event_ratio <= 0
adapter_minus_baseline_visible_outside_mask_event_ratio <= 0
```

实际结果：

```text
1. within_prior_radius_ratio 多数为正，但幅度比 B4.1 strict p2 小；
2. projection_any_mask_hit_ratio 在 support2/support3 中多数为负或接近 0；
3. visible_outside_mask_event_ratio 四组全部为正；
4. adapter_minus_baseline_visible_outside_mask_event_ratio 四组全部为正。
```

所以结论是：

```text
B4.2 的 point-level object support 过滤本身是正确的，
但 token-bias adapter 仍然不能稳定把 sparse 变化导向 object mask 内。
```

这说明失败点已经不只是“输入点云里混入背景点”。即使先过滤到多视图一致 object-supported 3D points，adapter 对 VGGT token 的微小 bias 仍然表现为局部扰动，而不是可靠的 object-aware sparse generation constraint。

### 与 B4.1 对比

B4.1 strict p2 a0.5：

```text
added_minus_removed_within_prior_radius_ratio = +0.09163
added_minus_removed_projection_any_mask_hit_ratio = +0.02588
added_minus_removed_visible_outside_mask_event_ratio = +0.00631
```

B4.2 support2 a0.5：

```text
added_minus_removed_within_prior_radius_ratio = +0.03331
added_minus_removed_projection_any_mask_hit_ratio = -0.01024
added_minus_removed_visible_outside_mask_event_ratio = +0.00626
```

B4.2 support2 a0.5 的 outside 只和 B4.1 strict p2 a0.5 接近，没有真正改善；同时 prior/mask-hit 方向性反而变弱。因此不建议继续沿 support 阈值做 sweep。

### 下一步建议

停止以下路线：

```text
1. 不跑 B4.2 mesh；
2. 不继续 runtime scale sweep；
3. 不继续 support2/support3 阈值 sweep；
4. 不继续只训练 positive token-bias adapter。
```

下一步应转 B5：显式负约束或后验过滤。

优先级建议：

```text
B5.1 sparse 后验 object-mask / visual-hull filter
  对 no-adapter 或 adapter sparse coords 做 deterministic filtering；
  直接删除 mask 外可见事件高的 coords；
  先证明 AR mask/pose/prior 能稳定改善 sparse set。

B5.2 positive/negative 双通道 token adapter
  不只给 object-supported token 加正 bias；
  同时给 outside-mask / low-support token 加负 bias；
  训练目标显式包含 outside suppression。

B5.3 再考虑 mesh
  只有当 sparse delta 满足 outside <= 0 后，
  再跑 mesh，否则 mesh 解释不干净。
```

当前最干净、最快的下一步是 B5.1：

```text
直接对 sparse coords 做 AR mask / visual hull 后验过滤。
```

原因：

```text
1. B4.1/B4.2 已经证明 token-bias adapter 改变方向不稳定；
2. 后验 filter 可以直接回答 AR pose + mask + prior 是否能改善 sparse coords；
3. 如果后验 filter 都不能降低 outside-mask 事件，继续训练 adapter 没意义；
4. 如果后验 filter 有效，再把它蒸馏/学习到 adapter 或 sampler 中才有依据。
```

## 十九、2026-07-07 B5.0/B5.1 SS-condition residual smoke

### 目的

本节执行修正后的 B5 路线：

```text
1. 不直接用 decoded sparse coords loss 训练；
2. 先确认 loss / metric 可微性边界；
3. cond_delta 不只吃 cond_base，而是用 AR/prior projection features 构造 cond-shaped residual；
4. 第一版只改 SS condition，不碰 SLAT；
5. 先做 eval-time residual candidate search，不训练；
6. 后续如果训练，只训练 adapter / LoRA，不 full fine-tune sparse_structure_vggt_cond。
```

新增脚本：

```text
reconvggt_ar_adapter_a/run_b5_ss_cond_residual_smoke.py
```

### B5.0 接口与可微性确认

运行：

```text
ar_20260617_b5_sscond_residual_support2_r050_seed42
```

关键 shape：

```text
cond_base:
  shape = [1, 4096, 1024]
  dtype = float16
  rms = 1.590278
  abs_max = 169.625

projection_features:
  shape = [1, 7, 1369, 31]
  feature_dim = 31

AR/prior residual:
  shape = [1, 4096, 1024]
  rms = 1.590278
  residual_seed = 20260707
  token_mode = random
```

point-level support filter：

```text
point_count_before = 820
point_count_after = 707
kept_ratio = 0.8622
min_support_views = 2
min_support_ratio = 0.50
kept_support_ratio_mean = 0.7909
```

可微性结论：

```text
teacher-forced flow / pred_x0 loss:
  可以作为后续训练的可微代理。

decoded coords / IoU / component / visual-hull:
  当前路径中经过 sampling + threshold / argwhere；
  本节只作为 eval 诊断，不作为训练 loss。
```

### B5.1 eval-time residual candidate search

baseline：

```text
scale = 0
coord_count = 10029
component_count = 6
largest_component_ratio = 0.9672
```

candidate 结果：

```text
scale +0.02:
  coord_count = 10041
  set_iou_vs_baseline = 0.8359
  added = 903
  removed = 891
  added_minus_removed_within_prior_radius_ratio = +0.1593
  added_minus_removed_prior_distance_mean = -1.1506
  added_minus_removed_projection_any_mask_hit_ratio = +0.2532
  added_minus_removed_visible_outside_mask_event_ratio = -0.2141
  adapter_minus_baseline_visible_outside_mask_event_ratio = -0.0177

scale -0.02:
  coord_count = 10002
  set_iou_vs_baseline = 0.7625
  added = 1336
  removed = 1363
  added_minus_removed_within_prior_radius_ratio = +0.2587
  added_minus_removed_prior_distance_mean = -1.7252
  added_minus_removed_projection_any_mask_hit_ratio = +0.3484
  added_minus_removed_visible_outside_mask_event_ratio = -0.1681
  adapter_minus_baseline_visible_outside_mask_event_ratio = -0.0228

scale +0.05:
  coord_count = 15588
  component_count = 35
  added_minus_removed_within_prior_radius_ratio = -0.1322
  added_minus_removed_prior_distance_mean = +2.6426
  added_minus_removed_visible_outside_mask_event_ratio = -0.1330

scale -0.05:
  coord_count = 12901
  component_count = 10
  added_minus_removed_within_prior_radius_ratio = -0.1257
  added_minus_removed_prior_distance_mean = +2.2062
  adapter_minus_baseline_visible_outside_mask_event_ratio = +0.0047
```

### 结论

B5.0/B5.1 是目前最强的正信号。

小尺度 SS-condition residual 已经能让 sparse delta 同时满足：

```text
1. added 点比 removed 点更靠近 AR prior；
2. added 点有更高 projection/mask hit；
3. added-minus-removed outside-mask event 下降；
4. adapter 整体 outside-mask event 不上升，反而下降。
```

这和 B4 token-bias adapter 的结果不同。B4 多轮都无法稳定降低 outside-mask event，而 B5 在 `scale ±0.02` 上直接出现正确方向，说明：

```text
AR/prior 信号放在 sparse_structure_vggt_cond 输出后的 SS condition 空间，比放在 VGGT token 空间更有效。
```

同时，大尺度 residual 已经开始过扰动：

```text
scale ±0.05 虽然可能降低 outside-mask，但 prior 方向变差、coord 数和 component 明显异常。
```

因此后续不应该扩大 scale，而应该围绕小尺度 residual 做验证和训练。

### 下一步

先补一个 B5.1 稳定性验证：

```text
1. 多 residual_seed；
2. 只扫小尺度：+0.01, -0.01, +0.02, -0.02, +0.03, -0.03；
3. 继续只跑 sparse，不碰 SLAT。
```

如果多 seed 下仍有稳定正方向，再进入 B5.2：

```text
freeze VGGT
freeze sparse_structure_vggt_cond
freeze sparse flow
train zero-init SS cond residual adapter
训练 loss 先用 teacher-forced sparse flow / pred_x0 可微代理
decoded coords / visual-hull 继续只作为 eval
```

## 二十、2026-07-08 B5.1 多 seed 小尺度稳定性验证

### 运行范围

汇总了三组 residual seed：

```text
20260707:
  residual_scales = 0, ±0.02, ±0.05

20260708:
  residual_scales = 0, ±0.01, ±0.02, ±0.03

20260709:
  residual_scales = 0, ±0.01, ±0.02, ±0.03
```

共同设置：

```text
SS-only；
不碰 SLAT；
不训练；
support2_r050；
seed42 sparse sampling；
decoded coords 只作为 eval 诊断。
```

### 通过标准

本节继续使用 B5.1 的 candidate 判断：

```text
added_minus_removed_within_prior_radius_ratio > 0
added_minus_removed_projection_any_mask_hit_ratio > 0
added_minus_removed_visible_outside_mask_event_ratio <= 0
adapter_minus_baseline_visible_outside_mask_event_ratio <= 0
```

### 跨 seed 汇总

```text
scale -0.02:
  pass = 3 / 3
  avg_iou_vs_baseline = 0.8156
  avg_added_minus_removed_within_prior_radius_ratio = +0.2956
  avg_added_minus_removed_projection_any_mask_hit_ratio = +0.2825
  avg_added_minus_removed_visible_outside_mask_event_ratio = -0.1307
  avg_adapter_minus_baseline_visible_outside_mask_event_ratio = -0.0141
  avg_coord_count = 9894
  avg_component_count = 6.33

scale +0.02:
  pass = 3 / 3
  avg_iou_vs_baseline = 0.8355
  avg_added_minus_removed_within_prior_radius_ratio = +0.1461
  avg_added_minus_removed_projection_any_mask_hit_ratio = +0.1528
  avg_added_minus_removed_visible_outside_mask_event_ratio = -0.1037
  avg_adapter_minus_baseline_visible_outside_mask_event_ratio = -0.0077
  avg_coord_count = 10263
  avg_component_count = 8.00

scale +0.01:
  pass = 2 / 2
  avg_iou_vs_baseline = 0.8931
  avg_added_minus_removed_within_prior_radius_ratio = +0.0942
  avg_added_minus_removed_projection_any_mask_hit_ratio = +0.1103
  avg_added_minus_removed_visible_outside_mask_event_ratio = -0.1339
  avg_adapter_minus_baseline_visible_outside_mask_event_ratio = -0.0056
  avg_coord_count = 9766
  avg_component_count = 6.00

scale -0.01:
  pass = 1 / 2
  20260708 通过；
  20260709 mask / outside 方向失败。

scale +0.03:
  pass = 1 / 2
  20260709 通过；
  20260708 prior 方向失败，coord_count/component 明显偏大。

scale -0.03:
  pass = 1 / 2
  20260709 通过；
  20260708 prior/outside/adapter_outside 失败。

scale ±0.05:
  不稳定且过扰动；
  prior 方向或 component/coord_count 明显异常。
```

### 结论

`scale ±0.02` 是当前最稳定的 B5 eval-time residual 区间。

其中：

```text
scale -0.02:
  prior/mask 方向最强；
  outside-mask 也稳定下降；
  coord_count 没有膨胀；
  component_count 接近 baseline；
  但 set IoU 相对 baseline 更低，说明扰动更强。

scale +0.02:
  三个 seed 也全部通过；
  扰动略温和；
  但 prior/mask 改善幅度约为 -0.02 的一半；
  个别 seed component_count 到 13，稳定性略弱。

scale +0.01:
  两个 seed 均通过；
  最保守；
  但改善幅度较小。
```

因此：

```text
主推荐训练中心：scale -0.02
保守对照：scale +0.01 或 +0.02
不建议：±0.03 / ±0.05
```

### 下一步建议

可以进入 B5.2，但训练目标必须是可微代理，不是 decoded coords loss。

建议第一版 B5.2：

```text
1. freeze VGGT；
2. freeze sparse_structure_vggt_cond；
3. freeze sparse_structure_flow_model；
4. train zero-init SS cond residual adapter；
5. adapter 输入：
   cond_base + AR/prior cond-shaped encoding；
6. target behavior:
   先模仿 eval-time scale -0.02 residual direction；
   再接 teacher-forced sparse flow / pred_x0 可微 loss；
7. decoded sparse coords / visual-hull / outside-mask 仍只作为 eval。
```

更稳的实现顺序：

```text
B5.2a:
  train adapter to reproduce the deterministic -0.02 residual direction；
  验证训练版输出和 eval-time residual sparse delta 一致。

B5.2b:
  加 teacher-forced flow / pred_x0 可微代理 loss；
  观察是否比纯 mimic 更稳。
```

## 二十、2026-07-08 B5.2a / B5.2b SS condition residual adapter 训练结果

### 修改和新增代码

新增：

```text
reconvggt_ar_adapter_a/train_b5_ss_cond_residual_adapter.py
```

核心结构：

```text
SSCondResidualAdapter:
  输入:
    cond_base
    ar_cond_encoding

  形式:
    concat(cond_base, ar_cond_encoding)
    -> LayerNorm
    -> Linear
    -> SiLU
    -> zero-init Linear
    -> delta_cond

  输出:
    cond_adapter = cond_base + delta_cond
```

设计边界：

```text
freeze VGGT；
freeze sparse_structure_vggt_cond / get_ss_cond 原始 bridge；
freeze sparse_structure_flow_model；
只训练 get_ss_cond 输出后的 SS condition residual adapter；
第一版只改 sparse-structure condition，不碰 SLAT；
decoded sparse coords / prior alignment / outside-mask 只用于 eval，不作为训练 loss。
```

同时更新：

```text
reconvggt_ar_adapter_a/命令说明.txt
```

新增了 `20.1` 到 `20.4` 的 B5.2a / B5.2b 训练、评估和汇总命令。

### 代码检查

已运行：

```bash
/home/zjr/anaconda3/envs/reconviagen/bin/python -m py_compile \
  reconvggt_ar_adapter_a/train_b5_ss_cond_residual_adapter.py \
  reconvggt_ar_adapter_a/run_b5_ss_cond_residual_smoke.py
```

结果：

```text
通过。
```

### 运行命令

完整命令已写入：

```text
reconvggt_ar_adapter_a/命令说明.txt
```

本次实际运行：

```text
20.2 B5.2a：condition-space mimic -0.02 residual
20.3 B5.2b：mimic + frozen sparse-flow / x0 proxy
20.4 B5.2 汇总
```

输出目录：

```text
B5.2a:
  /home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/ar_20260617_b52a_sscond_mimic_m0p02_s100

B5.2b:
  /home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/ar_20260617_b52b_sscond_mimic_flowproxy_m0p02_s50
```

### B5.2a 结果

配置：

```text
target_scale = -0.02
max_steps = 100
lr = 1e-4
mimic_weight = 1.0
flow_proxy_weight = 0.0
x0_proxy_weight = 0.0
```

训练后 adapter 与 target residual 的 condition-space 拟合：

```text
mse = 0.0006389545
l1 = 0.0187853780
cosine_mean = 0.9034823
norm_ratio_mean = 1.5727441
```

说明：

```text
adapter 学到了 residual 方向；
但 norm_ratio > 1，说明输出幅度明显超过 deterministic -0.02 target。
```

sparse eval：

```text
baseline:
  coord_count = 10029
  component_count = 6
  largest_component_ratio = 0.9672

target_residual:
  coord_count = 10002
  component_count = 6
  iou_vs_baseline = 0.7625
  added_minus_removed_within_prior_radius_ratio = +0.2587
  added_minus_removed_projection_any_mask_hit_ratio = +0.3484
  added_minus_removed_visible_outside_mask_event_ratio = -0.1681
  adapter_minus_baseline_visible_outside_mask_event_ratio = -0.0228

adapter_residual:
  coord_count = 9864
  component_count = 7
  iou_vs_baseline = 0.7689
  added_minus_removed_within_prior_radius_ratio = +0.3023
  added_minus_removed_projection_any_mask_hit_ratio = +0.3449
  added_minus_removed_visible_outside_mask_event_ratio = -0.1486
  adapter_minus_baseline_visible_outside_mask_event_ratio = -0.0190
```

adapter residual 与 target residual 的 set compare：

```text
target_count = 10002
adapter_count = 9864
kept_count = 9630
removed_count = 372
added_count = 234
iou = 0.9408
```

结论：

```text
B5.2a 成功证明：
  trainable zero-init SS residual adapter 可以复现 B5.1 deterministic -0.02 residual 的有效方向。

但第一版 s100 / lr=1e-4 有过冲：
  condition norm_ratio = 1.57；
  adapter sparse delta 比 target residual 更强；
  outside-mask 仍下降，但不如 target residual 干净。
```

### B5.2b 结果

配置：

```text
resume = B5.2a last.ckpt
max_steps = 50
lr = 5e-5
mimic_weight = 1.0
flow_proxy_weight = 0.25
x0_proxy_weight = 0.05
t_values = 0.5,0.75,0.9
```

训练后 adapter 与 target residual 的 condition-space 拟合：

```text
mse = 0.0006927646
l1 = 0.0199668650
cosine_mean = 0.9318753
norm_ratio_mean = 1.6756603
```

说明：

```text
flow/x0 proxy 让方向 cosine 更高；
但没有抑制幅度，norm_ratio 从 1.57 进一步升到 1.68；
proxy 第一版不是 stabilizer，更像沿同一有效方向继续放大。
```

sparse eval：

```text
baseline:
  coord_count = 10029
  component_count = 6
  largest_component_ratio = 0.9672

target_residual:
  coord_count = 10002
  component_count = 6
  iou_vs_baseline = 0.7625
  added_minus_removed_within_prior_radius_ratio = +0.2587
  added_minus_removed_projection_any_mask_hit_ratio = +0.3484
  added_minus_removed_visible_outside_mask_event_ratio = -0.1681
  adapter_minus_baseline_visible_outside_mask_event_ratio = -0.0228

adapter_residual:
  coord_count = 10078
  component_count = 8
  iou_vs_baseline = 0.7584
  added_minus_removed_within_prior_radius_ratio = +0.3021
  added_minus_removed_projection_any_mask_hit_ratio = +0.4016
  added_minus_removed_visible_outside_mask_event_ratio = -0.1790
  adapter_minus_baseline_visible_outside_mask_event_ratio = -0.0255
```

adapter residual 与 target residual 的 set compare：

```text
target_count = 10002
adapter_count = 10078
kept_count = 9306
removed_count = 696
added_count = 772
iou = 0.8637
```

结论：

```text
B5.2b 的 sparse direction 更强：
  prior 方向 +0.3021；
  mask-hit 方向 +0.4016；
  outside-mask 下降 -0.1790；
  adapter overall outside 下降 -0.0255。

但它明显偏离 target residual：
  adapter_vs_target IoU 从 B5.2a 的 0.9408 降到 0.8637；
  component_count 从 6/7 增到 8；
  norm_ratio 到 1.68。

所以 B5.2b 第一版证明 proxy 有梯度、有作用，但也证明当前 proxy 权重 / 学习率 / resume 策略会放大 residual，而不是稳定复现 target。
```

### 总结

当前 B5.2 的关键结论：

```text
1. B5.2a 成立：
   SS cond residual adapter 可以训练，并且能学到 B5.1 的有效 residual 方向。

2. B5.2b 成立但过强：
   frozen sparse-flow / pred_x0 proxy 可以推动结果朝 prior/mask 更强方向变化；
   但第一版会放大 residual，导致和 target residual 的 set overlap 明显下降。

3. 这条路线比 token adapter 更有希望：
   token adapter 对 ss/slat condition 的影响小且方向不稳；
   SS cond residual adapter 直接作用在 sparse_structure_vggt_cond 输出后，已经能稳定改变 sparse set，并且方向符合 prior/mask/outside 指标。
```

### 下一步建议

不要继续直接加大 B5.2b。

建议先做稳定化：

```text
B5.2c:
  从零开始或从 B5.2a step 较早 checkpoint 开始；
  lr 降到 2e-5；
  max_steps 20 或 30；
  save_every 10；
  加 delta_norm penalty，让 norm_ratio 接近 1.0；
  flow_proxy_weight 降到 0.05；
  x0_proxy_weight 降到 0.01。
```

判断标准：

```text
adapter_vs_target IoU >= 0.93；
added_minus_removed_projection_any_mask_hit_ratio > +0.30；
added_minus_removed_visible_outside_mask_event_ratio < -0.15；
component_count 不高于 7。
```

如果 B5.2c 稳住：

```text
进入 B5.3:
  不再只用单个 AR session；
  在多个 AR session / synthetic heldout 上训练同一个 SS residual adapter；
  用 deterministic residual 或 teacher proxy 做弱监督；
  再评估是否能泛化。
```

如果 B5.2c 仍过冲：

```text
保留 B5.2a 作为 proof-of-concept；
停止 flow proxy；
改成 candidate search / sparse posterior filtering / learned reranker。
```

## 二十一、2026-07-08 B5.2c 拆分稳定化实验

### 用户建议采纳

B5.2c 没有直接做成：

```text
mimic + norm + flow + x0
```

而是拆成：

```text
B5.2c-1:
  mimic + norm stabilization；
  不加 flow / x0 proxy。

B5.2c-2:
  在 c-1 稳住后；
  加弱 flow / x0 proxy。
```

这个拆法是必要的，因为 B5.2b 已经证明：

```text
proxy 有梯度、有作用；
但第一版 proxy 会放大 residual；
如果不先单独确认 norm stabilization，会混淆问题来源。
```

### 代码修改

修改：

```text
reconvggt_ar_adapter_a/train_b5_ss_cond_residual_adapter.py
```

新增内容：

```text
1. delta_norm_loss()

2. 新增参数：
   --delta_norm_weight
   --delta_norm_target_ratio
   --delta_norm_loss_mode {over,target}

3. rows / log 中新增：
   delta_norm_loss
   delta_norm_ratio_mean

4. report 的 differentiability_note 中新增 B5.2c 说明。
```

当前使用的 norm 形式：

```text
delta_norm_loss_mode = over
delta_norm_target_ratio = 1.05

loss 只惩罚：
  norm(delta) / norm(target_delta) > 1.05
```

这样做的原因：

```text
zero-init 时不强行拉大 delta；
只防止 B5.2a / B5.2b 那种 residual 过冲。
```

命令已写入：

```text
reconvggt_ar_adapter_a/命令说明.txt
  21.1 代码检查
  21.2 B5.2c-1
  21.3 B5.2c-2
  21.4 汇总
```

### B5.2c-1 结果

配置：

```text
max_steps = 60
lr = 2e-5
mimic_weight = 1.0
delta_norm_weight = 0.05
delta_norm_target_ratio = 1.05
flow_proxy_weight = 0.0
x0_proxy_weight = 0.0
```

训练收敛：

```text
final mse = 0.0001627345
l1 = 0.00883744
cosine_mean = 0.9172106
norm_ratio_mean = 0.9635206
```

最后一行训练日志：

```text
step = 60
loss = 0.0001622094
mimic_loss = 0.0001622094
delta_norm_loss = 0.0
delta_norm_ratio_mean = 0.9580631
flow_proxy_loss = 0.0
x0_proxy_loss = 0.0
```

对比 B5.2a：

```text
B5.2a norm_ratio = 1.5727
B5.2c-1 norm_ratio = 0.9635
```

说明：

```text
norm stabilization 有效；
residual 幅度已经不再过冲。
```

sparse eval：

```text
target_residual:
  coord_count = 10002
  component_count = 6
  iou_vs_baseline = 0.7625
  add-rem-prior = +0.2587
  add-rem-mask = +0.3484
  add-rem-outside = -0.1681
  adapter-outside = -0.0228

adapter_residual:
  coord_count = 10059
  component_count = 7
  iou_vs_baseline = 0.7853
  add-rem-prior = +0.3245
  add-rem-mask = +0.3449
  add-rem-outside = -0.1338
  adapter-outside = -0.0165
```

adapter vs target set compare：

```text
iou = 0.9286
target_count = 10002
adapter_count = 10059
removed = 343
added = 400
```

判断：

```text
优点：
  norm 稳住；
  prior 方向更强；
  mask-hit 方向基本保持；
  component_count = 7，可接受。

不足：
  adapter_vs_target IoU = 0.9286，略低于 0.93；
  add-rem-outside = -0.1338，未达到 -0.15；
  outside-mask 改善弱于 target residual。
```

因此 B5.2c-1 是：

```text
幅度稳定通过；
方向指标未完全通过。
```

### B5.2c-2 结果

配置：

```text
resume = B5.2c-1 last.ckpt
max_steps = 30
lr = 2e-5
mimic_weight = 1.0
delta_norm_weight = 0.05
delta_norm_target_ratio = 1.05
flow_proxy_weight = 0.05
x0_proxy_weight = 0.01
t_values = 0.5,0.75,0.9
```

训练收敛：

```text
final mse = 0.0001687449
l1 = 0.00821216
cosine_mean = 0.9140549
norm_ratio_mean = 0.8660222
```

最后一行训练日志：

```text
step = 30
loss = 0.0001689558
mimic_loss = 0.0001688187
delta_norm_loss = 0.0
delta_norm_ratio_mean = 0.8624086
flow_proxy_loss = 2.35e-06
x0_proxy_loss = 1.95e-06
```

说明：

```text
弱 proxy 没有造成过冲；
但它也没有补回 c-1 缺失的 outside-mask 方向；
反而让 residual 幅度更小，norm_ratio 从 0.9635 降到 0.8660。
```

sparse eval：

```text
target_residual:
  coord_count = 10002
  component_count = 6
  iou_vs_baseline = 0.7625
  add-rem-prior = +0.2587
  add-rem-mask = +0.3484
  add-rem-outside = -0.1681
  adapter-outside = -0.0228

adapter_residual:
  coord_count = 10086
  component_count = 7
  iou_vs_baseline = 0.7939
  add-rem-prior = +0.3093
  add-rem-mask = +0.2841
  add-rem-outside = -0.1065
  adapter-outside = -0.0129
```

adapter vs target set compare：

```text
iou = 0.9208
target_count = 10002
adapter_count = 10086
removed = 372
added = 456
```

判断：

```text
B5.2c-2 不如 B5.2c-1。

弱 proxy:
  没有带来更好的 target matching；
  没有增强 outside-mask suppression；
  mask-hit 方向从 +0.3449 降到 +0.2841；
  adapter_vs_target IoU 从 0.9286 降到 0.9208。
```

### B5.2c 总结

对比：

```text
B5.2a:
  norm_ratio = 1.5727
  adapter_vs_target IoU = 0.9408
  add-rem-mask = +0.3449
  add-rem-outside = -0.1486
  component = 7

B5.2b:
  norm_ratio = 1.6757
  adapter_vs_target IoU = 0.8637
  add-rem-mask = +0.4016
  add-rem-outside = -0.1790
  component = 8

B5.2c-1:
  norm_ratio = 0.9635
  adapter_vs_target IoU = 0.9286
  add-rem-mask = +0.3449
  add-rem-outside = -0.1338
  component = 7

B5.2c-2:
  norm_ratio = 0.8660
  adapter_vs_target IoU = 0.9208
  add-rem-mask = +0.2841
  add-rem-outside = -0.1065
  component = 7
```

结论：

```text
B5.2c-1 证明 norm stabilization 是有效的；
B5.2c-2 证明当前弱 proxy 不值得继续加；
当前 B5.2c 还没有达到进入 B5.3 的标准。
```

### 是否进入 B5.3

当前不建议进入 B5.3。

原因：

```text
B5.3 意味着扩到多 AR session / synthetic heldout 上训练同一个 SS residual adapter。
但当前单 session 上还没有同时满足：
  adapter_vs_target IoU >= 0.93；
  mask-hit 方向 > +0.30；
  outside-mask 方向 < -0.15；
  component_count <= 7。

B5.2c-1 只差一点，但 outside-mask suppression 不够；
B5.2c-2 则进一步退化。
```

下一步更合理的是小范围 B5.2c 参数修正，而不是进入 5.3：

```text
1. 保留 B5.2c-1 作为当前最稳 checkpoint；
2. 不继续 B5.2c-2 proxy；
3. 只扫 mimic + norm：
   lr = 3e-5
   max_steps = 50
   delta_norm_target_ratio = 1.10
   delta_norm_weight = 0.02 或 0.03

目标：
  让 norm_ratio 接近 1.05-1.15；
  同时把 outside-mask 改善拉回到 <= -0.15。
```

如果这个小 sweep 仍不过：

```text
停止 SS cond residual adapter 的单样本拟合路线；
转向 sparse posterior filtering / learned candidate reranker；
或者把 residual target 从随机 cond-shaped direction 改成更显式的 outside-mask negative encoding。
```

## 二十二、2026-07-08 B5.2c 两套稳定配置对比

### 本次运行的两套配置

按讨论跑了两套配置，均为：

```text
from scratch；
flow_proxy_weight = 0.0；
x0_proxy_weight = 0.0；
只做 mimic + norm stabilization。
```

配置 A：

```text
output:
  /home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/ar_20260617_b52c_over110_w002_lr3e5_s50

lr = 3e-5
max_steps = 50
delta_norm_loss_mode = over
delta_norm_target_ratio = 1.10
delta_norm_weight = 0.02
```

配置 B：

```text
output:
  /home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/ar_20260617_b52c_target100_w002_lr2e5_s80

lr = 2e-5
max_steps = 80
delta_norm_loss_mode = target
delta_norm_target_ratio = 1.0
delta_norm_weight = 0.02
```

### 配置 A 结果

condition-space 拟合：

```text
mse = 0.0002933906
l1 = 0.0126530
cosine_mean = 0.8758141
norm_ratio_mean = 1.1147243
```

最后训练日志：

```text
step = 50
loss = 0.0003010002
mimic_loss = 0.0002949110
delta_norm_loss = 0.00030446
delta_norm_ratio_mean = 1.1174488
```

sparse / direction：

```text
adapter_vs_target IoU = 0.9354
coord_count = 10031
component_count = 7
largest_component_ratio = 0.9677
iou_vs_baseline = 0.7807

added_minus_removed_within_prior_radius_ratio = +0.3386
added_minus_removed_projection_any_mask_hit_ratio = +0.3676
added_minus_removed_visible_outside_mask_event_ratio = -0.1451
adapter_minus_baseline_visible_outside_mask_event_ratio = -0.0184
```

判断：

```text
配置 A 相比 B5.2c-1 明显改善：
  adapter_vs_target IoU 从 0.9286 -> 0.9354；
  mask-hit 从 +0.3449 -> +0.3676；
  outside 从 -0.1338 -> -0.1451。

但 outside-mask suppression 仍未达到严格门槛 -0.15。
```

### 配置 B 结果

condition-space 拟合：

```text
mse = 0.0003337086
l1 = 0.0131566
cosine_mean = 0.8441149
norm_ratio_mean = 1.0500610
```

最后训练日志：

```text
step = 80
loss = 0.0003882603
mimic_loss = 0.0003345844
delta_norm_loss = 0.0026838
delta_norm_ratio_mean = 1.0518054
```

sparse / direction：

```text
adapter_vs_target IoU = 0.9546
coord_count = 9984
component_count = 7
largest_component_ratio = 0.9665
iou_vs_baseline = 0.7733

added_minus_removed_within_prior_radius_ratio = +0.3381
added_minus_removed_projection_any_mask_hit_ratio = +0.3618
added_minus_removed_visible_outside_mask_event_ratio = -0.1478
adapter_minus_baseline_visible_outside_mask_event_ratio = -0.0191
```

判断：

```text
配置 B 是目前最稳的 SS residual adapter 配置。

优点：
  adapter_vs_target IoU = 0.9546，是当前最高；
  norm_ratio = 1.05，幅度稳定；
  coord_count 接近 target_residual；
  component_count = 7，可接受；
  prior / mask-hit 方向均强于 target_residual；
  outside-mask suppression 接近 -0.15。

不足：
  cosine 只有 0.8441，方向不如 c-1/c-2；
  outside = -0.1478，严格门槛 -0.15 还差 0.0022。
```

### 与旧配置对比

```text
B5.2c-1:
  norm = 0.9635
  adapter_vs_target IoU = 0.9286
  add-rem-mask = +0.3449
  add-rem-outside = -0.1338
  component = 7

配置 A:
  norm = 1.1147
  adapter_vs_target IoU = 0.9354
  add-rem-mask = +0.3676
  add-rem-outside = -0.1451
  component = 7

配置 B:
  norm = 1.0501
  adapter_vs_target IoU = 0.9546
  add-rem-mask = +0.3618
  add-rem-outside = -0.1478
  component = 7

B5.2c-2 weak proxy:
  norm = 0.8660
  adapter_vs_target IoU = 0.9208
  add-rem-mask = +0.2841
  add-rem-outside = -0.1065
  component = 7
```

### 结论

两套配置都比原 B5.2c-1 更好。

其中：

```text
配置 A:
  mask-hit 方向最强；
  但 norm 稍偏高，target set matching 不如 B。

配置 B:
  整体最稳；
  adapter_vs_target IoU 最高；
  norm 最接近 1.0；
  outside-mask suppression 最接近门槛。
```

所以当前首选 checkpoint 是：

```text
/home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/ar_20260617_b52c_target100_w002_lr2e5_s80/checkpoints/last.ckpt
```

### 是否进入 B5.3

严格判断：

```text
暂不进入 B5.3。
```

原因：

```text
配置 B 已经非常接近通过；
但 outside-mask suppression = -0.1478，严格门槛 -0.15 仍差 0.0022。
```

工程判断：

```text
配置 B 可以视作接近可用的候选；
但在扩到多 AR session / synthetic heldout 之前，最好补一个 very-small sweep。
```

下一步建议：

```text
只基于配置 B 做小范围修正，不再碰 proxy：

1. target mode:
   lr = 2e-5
   max_steps = 100
   delta_norm_weight = 0.02
   delta_norm_target_ratio = 1.05

2. target mode:
   lr = 2e-5
   max_steps = 80
   delta_norm_weight = 0.03
   delta_norm_target_ratio = 1.0

如果其中任意一个达到：
  adapter_vs_target IoU >= 0.93；
  add-rem-mask > +0.30；
  add-rem-outside <= -0.15；
  component_count <= 7；

就可以进入 B5.3。
```

## 二十三、2026-07-08 B-like correction checkpoint sweep 与 safe teacher

### 执行顺序

按以下顺序执行：

```text
第一步：
  跑两套 B-like small correction。

第二步：
  对两套新配置和旧配置 B 的 saved checkpoints 做 sweep。

第三步：
  如果 strong teacher 稳定过线，再跑 safe teacher 20260709/+0.01。
```

### 代码修改

新增：

```text
reconvggt_ar_adapter_a/eval_b5_ss_cond_residual_checkpoint_sweep.py
```

作用：

```text
一次加载 ReconViaGen / VGGT / TRELLIS pipeline；
一次采样 baseline 和 target_residual；
随后逐个加载 adapter checkpoint；
用相同 ss_noise 采样 adapter_residual；
输出每个 checkpoint 的：
  adapter_vs_target IoU
  norm_ratio
  mask-hit direction
  outside-mask direction
  component_count
  pass/fail
```

pass 标准：

```text
adapter_vs_target IoU >= 0.93
added_minus_removed_projection_any_mask_hit_ratio > +0.30
added_minus_removed_visible_outside_mask_event_ratio <= -0.15
component_count <= 7
```

命令已写入：

```text
reconvggt_ar_adapter_a/命令说明.txt
  22.1 代码检查
  22.2 B-like correction A
  22.3 B-like correction B
  22.4 saved checkpoint sweep
  22.5 safe teacher 20260709 / +0.01
```

### 第一步：两套 B-like correction

配置 A：

```text
output:
  /home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/ar_20260617_b52c_target105_w002_lr2e5_s100

target_scale = -0.02
residual_seed = 20260707
lr = 2e-5
max_steps = 100
delta_norm_loss_mode = target
delta_norm_target_ratio = 1.05
delta_norm_weight = 0.02
```

last checkpoint：

```text
mse = 0.0003516165
l1 = 0.0137336
cosine = 0.8457273
norm_ratio = 1.0964086
adapter_vs_target IoU = 0.9638
coord_count = 9929
component_count = 7
mask-hit direction = +0.3544
outside-mask direction = -0.1517
adapter outside delta = -0.0200
prior direction = +0.3157
```

配置 B：

```text
output:
  /home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/ar_20260617_b52c_target100_w003_lr2e5_s80

target_scale = -0.02
residual_seed = 20260707
lr = 2e-5
max_steps = 80
delta_norm_loss_mode = target
delta_norm_target_ratio = 1.0
delta_norm_weight = 0.03
```

last checkpoint：

```text
mse = 0.0003243549
l1 = 0.0127838
cosine = 0.8417941
norm_ratio = 1.0128345
adapter_vs_target IoU = 0.9531
coord_count = 9994
component_count = 7
mask-hit direction = +0.3643
outside-mask direction = -0.1470
adapter outside delta = -0.0189
prior direction = +0.3434
```

结论：

```text
配置 A 的 last checkpoint 已经严格过线；
配置 B 的 last checkpoint 非常稳，但 outside-mask 仍略弱。
```

### 第二步：saved checkpoint sweep

sweep 输出：

```text
/home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/ar_20260617_b52c_strong_teacher_checkpoint_sweep
```

参与 sweep：

```text
旧配置 B:
  ar_20260617_b52c_target100_w002_lr2e5_s80

新配置 A:
  ar_20260617_b52c_target105_w002_lr2e5_s100

新配置 B:
  ar_20260617_b52c_target100_w003_lr2e5_s80
```

sweep 总结果：

```text
pass_count = 5 / 29

pass_labels:
  old_B step60
  new_A step80
  new_A step90
  new_A step100
  new_A last

best_by_target_iou:
  old_B step30
  但 component_count = 8，不通过。

best_by_outside:
  old_B step60
```

过线 checkpoint：

```text
old_B step60:
  target_iou = 0.9449
  norm = 1.0017
  mask = +0.3779
  outside = -0.1541
  component = 7

new_A step80:
  target_iou = 0.9655
  norm = 1.1020
  mask = +0.3511
  outside = -0.1503
  component = 7

new_A step90:
  target_iou = 0.9603
  norm = 1.0933
  mask = +0.3569
  outside = -0.1512
  component = 7

new_A step100:
  target_iou = 0.9638
  norm = 1.0964
  mask = +0.3544
  outside = -0.1517
  component = 7

new_A last:
  target_iou = 0.9638
  norm = 1.0964
  mask = +0.3544
  outside = -0.1517
  component = 7
```

判断：

```text
strong teacher 已稳定过线。

特别是 new_A:
  step80 / step90 / step100 / last 连续过线；
  不是单个 checkpoint 偶然过线。
```

因此按预设进入第三步。

当前 strong teacher 首选 checkpoint：

```text
/home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/ar_20260617_b52c_target105_w002_lr2e5_s100/checkpoints/last.ckpt
```

更保守但 outside 最强的 checkpoint：

```text
/home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/ar_20260617_b52c_target100_w002_lr2e5_s80/checkpoints/adapter_step_000060.ckpt
```

### 第三步：safe teacher 20260709 / +0.01

配置：

```text
output:
  /home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/ar_20260617_b52c_safe_teacher_p0p01_r20260709_target100_w002_lr2e5_s80

target_scale = +0.01
residual_seed = 20260709
lr = 2e-5
max_steps = 80
delta_norm_loss_mode = target
delta_norm_target_ratio = 1.0
delta_norm_weight = 0.02
```

condition-space 拟合：

```text
mse = 0.0001027701
l1 = 0.0072479
cosine = 0.7933881
norm_ratio = 0.9826842
adapter_vs_target IoU = 0.9558
```

safe teacher target_residual 本身：

```text
coord_count = 10077
component_count = 6
iou_vs_baseline = 0.9335
mask-hit direction = +0.1748
outside-mask direction = -0.1955
adapter outside delta = -0.0057
prior direction = +0.1098
```

训练出的 adapter_residual：

```text
coord_count = 10271
component_count = 7
iou_vs_baseline = 0.9610
mask-hit direction = +0.1531
outside-mask direction = -0.0612
adapter outside delta = -0.0018
prior direction = -0.0947
```

判断：

```text
safe teacher target_residual 是一个“温和但 outside-mask 很好”的 residual；
但训练出的 adapter 没有复现它的关键方向。

主要问题：
  prior direction 从 target 的 +0.1098 变成 adapter 的 -0.0947；
  outside-mask suppression 从 target 的 -0.1955 退到 -0.0612；
  mask-hit direction 也从 +0.1748 退到 +0.1531。
```

因此：

```text
safe teacher +0.01 / 20260709 不作为下一阶段主线。
```

### 本节最终结论

当前可以确认：

```text
1. strong teacher 20260707 / -0.02 是有效训练目标；
2. B-like correction A 稳定过线；
3. old_B step60 是最保守、outside 最强的过线 checkpoint；
4. safe teacher 20260709 / +0.01 不适合作为当前训练目标。
```

下一步建议：

```text
不要再继续 safe teacher。

进入 B5.3 的候选有两个：

A. strong teacher 主线:
   使用 new_A last checkpoint / 配置作为训练 recipe；
   扩到多 AR session / synthetic heldout；
   目标是验证是否泛化。

B. conservative strong teacher:
   使用 old_B step60 作为 checkpoint/recipe；
   更稳、更低 norm；
   outside-mask 最强；
   适合先做小规模多样本验证。
```

我更建议 B5.3 第一轮用：

```text
recipe:
  target_scale = -0.02
  residual_seed = 20260707
  delta_norm_loss_mode = target
  delta_norm_target_ratio = 1.05
  delta_norm_weight = 0.02
  lr = 2e-5
  max_steps = 100
  checkpoint selection:
    prefer last if stable；
    but save all step checkpoints and sweep。
```

并且不要再用 flow/x0 proxy，直到多样本 strong teacher 能稳定过线。

## 二十四、2026-07-08 B5.3 multi-session strong teacher smoke

### 目的

B5.2 已经证明在单个 AR session 上，`residual_seed=20260707 / target_scale=-0.02` 的 strong teacher 可以被 SS-condition residual adapter 稳定拟合，并且能产生较好的 mask/outside 方向。

本节验证这个结论是否能跨 session 成立：

```text
训练:
  ar_20260617_075401_819
  ar_20260624_010530_849

holdout:
  ar_20260624_011149_207

模型:
  freeze VGGT
  freeze sparse_structure_vggt_cond / get_ss_cond
  freeze sparse flow
  只训练 SS condition residual adapter

训练目标:
  target_delta = ar_cond_encoding * -0.02
  residual_seed = 20260707
  mimic_weight = 1.0
  delta_norm_weight = 0.02
  delta_norm_loss_mode = target
  delta_norm_target_ratio = 1.05
  no flow proxy
  no x0 proxy
```

新增代码：

```text
reconvggt_ar_adapter_a/train_b5_ss_cond_residual_multisession.py
reconvggt_ar_adapter_a/configs/b53_multisession_train2_holdout1.json
```

运行输出：

```text
/home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/ar_b53_multisession_strong_teacher_train2_holdout1_s100/report.json
/home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/ar_b53_multisession_strong_teacher_train2_holdout1_s100/report.md
/home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/ar_b53_multisession_strong_teacher_train2_holdout1_s100/checkpoints/last.ckpt
```

### 训练结果

100 step 后：

```text
loss = 0.000434742
mimic_loss = 0.000430357
delta_norm_ratio_mean = 1.03521
delta_target_cosine = 0.79294
```

训练没有数值问题，adapter 能学到一个平均方向，但 cosine 只有约 `0.793`，低于单 session 里更接近 teacher 的情况。

### sparse eval 结果

严格通过条件：

```text
adapter_vs_target_iou >= 0.93
mask_direction > 0.30
outside_direction <= -0.15
component_count <= 7
```

结果：

```text
ar_20260617_075401_819 [train]
  adapter_vs_target_iou = 0.97379
  component = 7
  mask_direction = +0.33101
  outside_direction = -0.14616
  passed = false

ar_20260624_010530_849 [train]
  adapter_vs_target_iou = 0.95396
  component = 1
  mask_direction = 0.0
  outside_direction = +0.00684
  passed = false

ar_20260624_011149_207 [holdout]
  adapter_vs_target_iou = 0.93645
  component = 1
  mask_direction = +0.12722
  outside_direction = -0.00325
  passed = false
```

对比 target residual：

```text
ar_20260617_075401_819:
  target mask_direction = +0.34841
  target outside_direction = -0.16809

ar_20260624_010530_849:
  target mask_direction = 0.0
  target outside_direction = -0.06859

ar_20260624_011149_207:
  target mask_direction = +0.05236
  target outside_direction = +0.03639
```

### 结论

B5.3 没有通过 multi-session strict criterion。

关键判断：

```text
1. 第一个 train session 接近 B5.2 行为，但 outside_direction = -0.14616，
   距离严格阈值 -0.15 仍差一点。

2. 第二个 train session 的 mask_direction = 0，
   outside_direction 甚至变成 +0.00684。

3. holdout 虽然 adapter_vs_target_iou = 0.93645 过线，
   但 mask_direction 只有 +0.12722，
   outside_direction 只有 -0.00325，几乎没有 outside-mask suppression。

4. 更重要的是，target residual 本身在新 session 上也不稳定：
   holdout target outside_direction = +0.03639，
   说明 20260707 / -0.02 这个 deterministic teacher 不是跨 session 稳定的几何规则。
```

因此，B5.2 的 strong teacher 暂时只能视为：

```text
single-session effective direction
```

不能直接作为可泛化训练目标扩展。

### 下一步建议

停止继续在当前 deterministic random residual teacher 上扩大训练。

后续应转向更有物理含义的 teacher / loss：

```text
1. 不再用 random channel direction 作为 teacher；
2. 用由 object mask / visual hull / AR point support 明确定义的 cond-shaped residual；
3. 或者绕开 teacher mimic，直接训练一个 SS-condition residual adapter，
   但 loss 必须来自可解释的 sparse objective proxy：
     - outside-mask negative objective；
     - visual-hull consistency；
     - prior support attraction；
     - sparse coords correction 的可微近似。
```

如果仍保留 B5 路线，下一步应是：

```text
B5.4:
  构造 mask/visual-hull/prior-support driven residual teacher；
  先 eval-time candidate search；
  再做 adapter mimic；
  不再使用随机 residual seed 作为核心 teacher。
```

## 二十五、2026-07-08 B5.4 physical SS-grid feature encoding + candidate search

### 目的

B5.3 证明 deterministic random residual teacher 不能跨 session 泛化。本节把 B5 改成物理特征主导：

```text
object mask / visual hull / point support / prior distance
  -> physical_token_features on SS 16^3 grid
  -> physical gate
  -> delta_cond = physical_gate[token] * normalized cond_base[token, channel] * scale
```

重要限制：

```text
1. 不训练；
2. 不使用 random channel direction；
3. 不碰 VGGT / get_ss_cond / sparse flow / SLAT；
4. 只做 eval-time residual candidate search；
5. channel basis 来自 cond_base 自身，空间 gate 来自物理特征。
```

新增代码：

```text
reconvggt_ar_adapter_a/run_b54_physical_ssgrid_smoke.py
```

命令已写入：

```text
reconvggt_ar_adapter_a/命令说明.txt
```

输出：

```text
/home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/ar_b54_physical_ssgrid_train2_holdout1_condunit/report.json
/home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/ar_b54_physical_ssgrid_train2_holdout1_condunit/report.md
```

### physical feature sanity

本节特征对齐的是 `cond_base=[B,4096,1024]` 的 SS 16³ token，而不是 2D VGGT token。

三个 session 的 physical feature 非退化：

```text
ar_20260617_075401_819 [train]
  support_gate_nonzero_ratio = 0.23926
  outside_gate_nonzero_ratio = 0.59326
  visual_hull_inside_ratio = 0.12549
  prior_within_radius_ratio = 0.06812
  mask_hit_ratio = 0.23926
  support_gate_vs_mask_hit corr = 0.50420
  support_gate_vs_prior_within corr = 0.68324

ar_20260624_010530_849 [train]
  support_gate_nonzero_ratio = 0.59570
  outside_gate_nonzero_ratio = 0.98535
  visual_hull_inside_ratio = 0.05688
  prior_within_radius_ratio = 0.03589
  mask_hit_ratio = 0.59570
  support_gate_vs_mask_hit corr = 0.32458
  support_gate_vs_prior_within corr = 0.60595

ar_20260624_011149_207 [holdout]
  support_gate_nonzero_ratio = 0.43555
  outside_gate_nonzero_ratio = 0.86157
  visual_hull_inside_ratio = 0.02295
  prior_within_radius_ratio = 0.02588
  mask_hit_ratio = 0.43555
  support_gate_vs_mask_hit corr = 0.34268
  support_gate_vs_prior_within corr = 0.65406
```

判断：

```text
physical_token_features 是有信息量的；
support/prior/mask/visual-hull 特征不是常数；
support_gate 与 mask/prior 有正相关；
outside_gate 与 visual_hull_inside 大多负相关。
```

因此 B5.4 的 feature encoding 本身成立。

### candidate search 结果

候选：

```text
basis = cond_unit
gates = support_gate, outside_gate, contrast_gate, visual_hull_gate, prior_gate
scales = +0.005, -0.005, +0.01, -0.01
```

aggregate top candidates：

```text
cond_unit_outside_gate_p0p01
  good_direction_session_count = 1 / 3
  mean_mask = +0.14224
  mean_outside = -0.07124
  mean_prior = +0.05128
  min_mask = 0.0
  max_outside = +0.00840
  mean_iou = 0.80217

cond_unit_outside_gate_p0p005
  good_direction_session_count = 1 / 3
  mean_mask = +0.14012
  mean_outside = -0.07471
  mean_prior = +0.08871
  min_mask = 0.0
  max_outside = -0.00711
  mean_iou = 0.85347

cond_unit_visual_hull_gate_m0p01
  good_direction_session_count = 1 / 3
  mean_mask = +0.13221
  mean_outside = -0.03623
  mean_prior = +0.12879
  min_mask = 0.0
  max_outside = +0.00464
  mean_iou = 0.85258

cond_unit_contrast_gate_m0p005
  good_direction_session_count = 1 / 3
  mean_mask = +0.08651
  mean_outside = -0.05806
  mean_prior = +0.05548
  min_mask = 0.0
  max_outside = -0.00440
  mean_iou = 0.85898
```

其中最值得保留观察的是：

```text
cond_unit_outside_gate_p0p005
```

原因：

```text
1. 三个 session 的 outside direction 全部 <= 0；
2. mean_mask 为正；
3. mean_prior 为正；
4. mean_iou 仍有 0.85347，没有完全破坏 baseline。
```

但它仍不满足训练条件：

```text
ar_20260617_075401_819:
  mask = +0.20267
  outside = -0.15602
  prior = +0.21196

ar_20260624_010530_849:
  mask = 0.0
  outside = -0.00711
  prior = +0.09294

ar_20260624_011149_207:
  mask = +0.21769
  outside = -0.06101
  prior = -0.03876
```

问题：

```text
holdout prior direction 仍为负；
第二个 train session 的 mask direction 为 0；
good_direction_session_count 只有 1 / 3。
```

### 结论

B5.4 推进后的结论是：

```text
1. physical SS-grid feature encoding 是正确方向；
2. 它比 random residual teacher 更可解释；
3. outside_gate 类候选确实能产生跨 session 的 outside suppression 倾向；
4. 但单一 physical gate + cond_unit channel basis 还不足以形成稳定 teacher；
5. 当前没有 candidate 可以直接进入 adapter mimic training。
```

因此：

```text
不要训练 B5.4 当前候选；
不要把 cond_unit_outside_gate_p0p005 当作最终 teacher；
但可以把它作为 B5.4b 的起点。
```

### 下一步建议

进入 B5.4b，而不是直接 B5.5 differentiable sparse proxy loss。

B5.4b 应做：

```text
1. linear physical gate composition:
   gate = a * outside_gate + b * support_gate + c * prior_gate + d * contrast_gate

2. 小网格搜索，不训练：
   目标是同时满足：
     outside <= 0 across all sessions
     mask >= 0 across all sessions
     prior >= 0 across all sessions
     iou 不过度下降

3. 增加 channel basis ablation:
   cond_unit
   cond_centered_unit
   maybe low-rank cond PCA basis

4. 只有当 B5.4b 找到 multi-session 稳定 candidate 后，
   才进入 adapter mimic training。
```

更激进的 differentiable sparse proxy loss 仍然保留，但排在后面：

```text
B5.5:
  outside negative
  visual-hull negative
  prior support positive
  baseline preservation
  delta norm
  smoothness proxy
```

当前不建议直接进入 B5.5，因为 B5.4 已经说明：

```text
物理特征有用；
但 teacher/candidate 组合还没有稳定。
```

## 25. B5.4b physical gate composition + basis ablation

时间：2026-07-08

### 目的

B5.4 的单 gate 结果说明：

```text
physical SS-grid feature encoding 是正确方向；
outside_gate 类候选有 outside suppression 倾向；
但单一 physical gate + cond_unit channel basis 不能形成跨 session 稳定 teacher。
```

因此 B5.4b 继续保持“不训练”，只做 eval-time candidate search：

```text
1. 组合 physical gates：
   outside_gate / support_gate / prior_gate / contrast_gate / visual_hull_gate

2. 比较 channel basis：
   cond_unit
   cond_centered_unit

3. 判断是否存在 train2 + holdout1 都稳定的 residual teacher。
```

### 修改代码

修改文件：

```text
reconvggt_ar_adapter_a/run_b54_physical_ssgrid_smoke.py
```

主要改动：

```text
1. 新增 --candidate_formulas；
   支持形如：
     outside_prior:outside_gate=1,prior_gate=0.5

2. 新增组合 gate 解析：
   _parse_gate_formulas()
   _compose_gate()

3. _candidate_delta() 支持 gate_weights；
   可以用多个 physical feature 线性组合生成 spatial gate。

4. basis_modes 支持 cond_centered_unit；
   用 centered cond direction 作为 residual channel basis 对照。
```

代码检查通过：

```bash
/home/zjr/anaconda3/envs/reconviagen/bin/python -m py_compile \
  reconvggt_ar_adapter_a/run_b54_physical_ssgrid_smoke.py
```

### 运行命令

命令已写入：

```text
reconvggt_ar_adapter_a/命令说明.txt
```

对应章节：

```text
25. B5.4b physical gate composition + basis ablation
```

实际输出：

```text
reconvggt_ar_adapter_a/outputs/ar_b54b_physical_combo_train2_holdout1_v2/report.json
```

### 汇总结果

总 candidate 数：

```text
64
```

严格过线 candidate：

```text
0
```

严格条件为：

```text
good_direction_session_count == 3
min_mask_direction >= 0
max_outside_direction <= 0
min_prior_direction >= 0
```

report judgment：

```text
B5.4 found at least one physical candidate with non-random spatial gates;
inspect aggregate good_direction_session_count before training.
No candidate is consistently good across all sessions, so do not train mimic yet.
```

Top aggregate：

```text
cond_unit_outside_vh_p0p01
  good = 2
  mean_mask = +0.11634
  min_mask = 0.0
  mean_outside = -0.03705
  max_outside = +0.00520
  mean_prior = +0.07246
  min_prior = +0.00712
  mean_iou = 0.78835

cond_centered_unit_outside_vh_p0p005
  good = 1
  mean_mask = +0.14852
  min_mask = 0.0
  mean_outside = -0.03449
  max_outside = -0.01359
  mean_prior = +0.01226
  min_prior = -0.04629
  mean_iou = 0.72215

cond_unit_outside_p0p01
  good = 1
  mean_mask = +0.14110
  min_mask = 0.0
  mean_outside = -0.07136
  max_outside = +0.00840
  mean_prior = +0.05073
  min_prior = -0.03557
  mean_iou = 0.80131

cond_unit_outside_prior_p0p005
  good = 1
  mean_mask = +0.14042
  min_mask = 0.0
  mean_outside = -0.08263
  max_outside = -0.00753
  mean_prior = +0.00231
  min_prior = -0.04487
  mean_iou = 0.85687

cond_unit_outside_p0p005
  good = 1
  mean_mask = +0.13745
  min_mask = 0.0
  mean_outside = -0.07476
  max_outside = -0.00711
  mean_prior = +0.08929
  min_prior = -0.03704
  mean_iou = 0.85264
```

### 分 session 观察

第一个 train session：

```text
ar_20260617_075401_819

best candidate:
  cond_unit_outside_support_prior_m0p01
  mask = +0.33607
  outside = -0.13208
  prior = +0.37858
  iou = 0.81886
```

第二个 train session：

```text
ar_20260624_010530_849

best candidates 的 mask direction 基本为 0；
outside 多数可控；
prior 有小幅正向；
但没有明显 token-mask gain。

example:
  cond_unit_outside_contrast_p0p005
  mask = 0.0
  outside = -0.00798
  prior = +0.09408
  iou = 0.68361
```

holdout session：

```text
ar_20260624_011149_207

best candidate:
  cond_unit_outside_contrast_p0p005
  mask = +0.22671
  outside = -0.05755
  prior = -0.02752
  iou = 0.96977
```

关键问题：

```text
1. train1 可获得很强 mask / outside / prior 三向改善；
2. train2 的 mask direction 经常为 0，说明物理 gate 对该 session 的可控增益弱；
3. holdout 的 mask / outside 可改善，但 prior direction 仍可能为负；
4. 没有 candidate 同时满足 mask、outside、prior 三个方向在所有 session 上一致。
```

### 结论

B5.4b 的结论是：

```text
1. linear physical gate composition 比单 gate 更可解释，但仍没有稳定 teacher；
2. cond_centered_unit 没有显著优于 cond_unit；
3. outside + visual_hull / outside + contrast 类组合有局部潜力；
4. 但 train2 与 holdout 的方向冲突仍然存在；
5. 不能进入 adapter mimic training。
```

因此：

```text
停止 B5.4 系列的 teacher mimic 尝试；
不要用当前 64 个 candidate 中任意一个作为监督目标；
不要继续扩大同类 formula sweep。
```

### 下一步建议

进入 B5.5，而不是继续 B5.4c。

B5.5 的方向应从“找一个离散 residual teacher”改为“直接优化可解释物理 proxy”，但仍保持第一版保守：

```text
1. freeze 原 ReconViaGen bridge；
2. 只训练 SS-condition residual adapter；
3. 不碰 SLAT；
4. 使用可微 proxy，而不是 sparse argmax/坐标集合 loss；
5. loss 直接约束 cond_delta 在物理区域的符号/强度：
   outside negative
   visual-hull negative
   prior/object support positive
   baseline preservation
   delta norm
   spatial smoothness
```

原因：

```text
B5.4/B5.4b 已经证明物理 feature 本身不是随机的；
失败点在于 eval-time hand-crafted teacher 不够稳定；
下一步应让小 adapter 学习如何组合 physical features 和 cond_base，
而不是继续人工枚举 residual formula。
```

## 26. B5.5 SS-condition physical proxy residual adapter

### 26.1 本节目的

B5.4/B5.4b 已经说明：手工构造 eval-time residual teacher 不稳定，不能直接进入 mimic training。

本节按新的路线实现并测试 B5.5：

```text
freeze 原 ReconViaGen VGGT / sparse_structure_vggt_cond / sparse flow / sparse decoder；
在 sparse_structure_vggt_cond 输出后训练一个 zero-init SS-condition residual adapter；
第一版只改 SS，不碰 SLAT；
使用物理 feature 作为输入：
  point support / prior distance / mask hit / outside visible / visual hull；
先测试可微 proxy 是否可训练，再看 sparse sampling 的 delta 方向。
```

### 26.2 新增/修改代码

新增：

```text
reconvggt_ar_adapter_a/train_b55_physical_proxy_adapter.py
```

主要结构：

```text
PhysicalSSCondResidualAdapter:
  input:
    cond_base: sparse_structure_vggt_cond 输出的 cond token
    physical_features: 16^3 SS-grid 上的物理 feature
    active_gate: 由 prior/mask/visual-hull 得到的 token gate

  output:
    cond_delta，zero-init，叠加到 cond_base 后只用于 sparse structure sampling
```

本节中途增加的稳定性保护：

```text
1. delta_clip_abs；
2. grad_clip_norm；
3. backward_loss_scale；
4. non-finite loss / grad skip；
5. adapter 输入、hidden activation 的 nan_to_num / clamp；
6. nan_to_num_grads 调试选项；
7. aggregate strict 判断不再把 no-op(iou=1.0) 算作通过。
```

### 26.3 代码检查命令

```bash
cd /home/zjr/Tracker

/home/zjr/anaconda3/envs/reconviagen/bin/python -m py_compile \
  reconvggt_ar_adapter_a/train_b55_physical_proxy_adapter.py
```

结果：

```text
py_compile 通过。
```

### 26.4 B5.5a：decoder-logit differentiable proxy

第一版直接让 loss 穿过 frozen sparse flow + frozen sparse decoder logits。

运行命令：

```bash
cd /home/zjr/Tracker

CUDA_VISIBLE_DEVICES=5 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/zjr/anaconda3/envs/reconviagen/bin/python -u reconvggt_ar_adapter_a/train_b55_physical_proxy_adapter.py \
  --sessions_json reconvggt_ar_adapter_a/configs/b53_multisession_train2_holdout1.json \
  --output_dir /home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/ar_b55_physical_proxy_train2_holdout1_s20_lr1e4 \
  --pretrained Stable-X/trellis-vggt-v0-2 \
  --max_views 7 \
  --device cuda \
  --low_vram \
  --mask_mode apply \
  --mask_background black \
  --mask_threshold 127 \
  --patch_start_idx 5 \
  --seed 42 \
  --num_samples 1 \
  --ss_steps 12 \
  --ss_cfg_strength 7.5 \
  --ss_guidance_rescale 0.5 \
  --ss_rescale_t 3.0 \
  --ss_grid_side 16 \
  --sparse_resolution 64 \
  --prior_radius 4.0 \
  --physical_vh_min_visible_views 1 \
  --physical_vh_min_support_ratio 0.5 \
  --projection_min_support_views 1 \
  --projection_min_support_ratio 0.0 \
  --visual_hull_min_visible_views 1 \
  --visual_hull_min_support_ratio 0.0 \
  --use_prior_score_positive \
  --visual_hull_active_weight 0.25 \
  --hidden_dim 256 \
  --max_steps 20 \
  --lr 1.0e-4 \
  --t_values 0.5,0.75,0.9 \
  --train_runtime_scale 1.0 \
  --eval_runtime_scales 0.25,0.5,1.0 \
  --margin_pos 0.003 \
  --margin_neg 0.001 \
  --pos_weight 1.0 \
  --neg_weight 2.0 \
  --preserve_weight 0.1 \
  --delta_norm_weight 0.02 \
  --smooth_weight 0.01 \
  --log_every 5 \
  --save_every 20
```

结果：

```text
step 1:
  loss=1.10e-05
  delta_norm≈6.32e-13

step 5 起：
  loss=nan
  pos=nan
  neg=nan
  delta_norm=nan

eval:
  adapter_ap0p25 / ap0p5 / ap1 全部把 sparse 结果推成空集：
    mean_iou=0.0
    component_count=0
```

结论：

```text
decoder-logit proxy 直接反传不稳定；
当前 fp16 sparse flow/decoder 链路下，不能把它作为 B5.5 第一版训练目标。
```

### 26.5 B5.5a 稳定性复测：backward loss scale

运行命令：

```bash
cd /home/zjr/Tracker

CUDA_VISIBLE_DEVICES=5 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/zjr/anaconda3/envs/reconviagen/bin/python -u reconvggt_ar_adapter_a/train_b55_physical_proxy_adapter.py \
  --sessions_json reconvggt_ar_adapter_a/configs/b53_multisession_train2_holdout1.json \
  --output_dir /home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/ar_b55_physical_proxy_train2_holdout1_s5_backscale1e6 \
  --pretrained Stable-X/trellis-vggt-v0-2 \
  --max_views 7 \
  --device cuda \
  --low_vram \
  --mask_mode apply \
  --mask_background black \
  --mask_threshold 127 \
  --patch_start_idx 5 \
  --seed 42 \
  --num_samples 1 \
  --ss_steps 12 \
  --ss_cfg_strength 7.5 \
  --ss_guidance_rescale 0.5 \
  --ss_rescale_t 3.0 \
  --ss_grid_side 16 \
  --sparse_resolution 64 \
  --prior_radius 4.0 \
  --physical_vh_min_visible_views 1 \
  --physical_vh_min_support_ratio 0.5 \
  --projection_min_support_views 1 \
  --projection_min_support_ratio 0.0 \
  --visual_hull_min_visible_views 1 \
  --visual_hull_min_support_ratio 0.0 \
  --use_prior_score_positive \
  --visual_hull_active_weight 0.25 \
  --hidden_dim 256 \
  --max_steps 5 \
  --lr 2.0e-5 \
  --t_values 0.5,0.75,0.9 \
  --train_runtime_scale 0.25 \
  --eval_runtime_scales 0.05,0.1,0.25 \
  --margin_pos 0.002 \
  --margin_neg 0.001 \
  --pos_weight 1.0 \
  --neg_weight 2.0 \
  --preserve_weight 0.2 \
  --delta_norm_weight 0.05 \
  --smooth_weight 0.02 \
  --delta_clip_abs 0.05 \
  --grad_clip_norm 1.0 \
  --backward_loss_scale 1e-6 \
  --log_every 1 \
  --save_every 5
```

结果：

```text
step 1-5:
  loss finite
  但全部 skipped_nonfinite_grad=True
  grad_norm=nan

eval:
  iou=1.0 是 no-op，不是有效改进；
  修正后的 aggregate 判据不再把这种 no-op 算作通过。
```

结论：

```text
把 backward loss scale 降到 1e-6 仍不能得到有限梯度；
decoder-logit proxy 的问题不是单纯梯度幅度过大，而是反传链路本身不适合作为当前第一版。
```

### 26.6 B5.5b：condition-basis physical proxy

为避免 decoder 反传不稳定，加入 `--proxy_loss_mode cond_basis`。

核心思想：

```text
不穿过 sparse decoder；
用 cond_base 自身归一化方向作为 channel basis；
positive physical token 希望 cond_delta 在 cond_base 方向为正；
negative physical token 希望 cond_delta 在 cond_base 方向为负；
仍然通过真实 sparse sampling 做最终 eval。
```

运行命令：

```bash
cd /home/zjr/Tracker

CUDA_VISIBLE_DEVICES=5 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/zjr/anaconda3/envs/reconviagen/bin/python -u reconvggt_ar_adapter_a/train_b55_physical_proxy_adapter.py \
  --sessions_json reconvggt_ar_adapter_a/configs/b53_multisession_train2_holdout1.json \
  --output_dir /home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/ar_b55_condbasis_proxy_train2_holdout1_s20_lr1e4 \
  --pretrained Stable-X/trellis-vggt-v0-2 \
  --max_views 7 \
  --device cuda \
  --low_vram \
  --mask_mode apply \
  --mask_background black \
  --mask_threshold 127 \
  --patch_start_idx 5 \
  --seed 42 \
  --num_samples 1 \
  --ss_steps 12 \
  --ss_cfg_strength 7.5 \
  --ss_guidance_rescale 0.5 \
  --ss_rescale_t 3.0 \
  --ss_grid_side 16 \
  --sparse_resolution 64 \
  --prior_radius 4.0 \
  --physical_vh_min_visible_views 1 \
  --physical_vh_min_support_ratio 0.5 \
  --projection_min_support_views 1 \
  --projection_min_support_ratio 0.0 \
  --visual_hull_min_visible_views 1 \
  --visual_hull_min_support_ratio 0.0 \
  --use_prior_score_positive \
  --visual_hull_active_weight 0.25 \
  --hidden_dim 256 \
  --proxy_loss_mode cond_basis \
  --max_steps 20 \
  --lr 1.0e-4 \
  --t_values 0.5,0.75,0.9 \
  --train_runtime_scale 1.0 \
  --eval_runtime_scales 0.25,0.5,1.0 \
  --cond_basis_pos_target 0.01 \
  --cond_basis_neg_target 0.005 \
  --pos_weight 1.0 \
  --neg_weight 2.0 \
  --preserve_weight 0.2 \
  --delta_norm_weight 0.05 \
  --smooth_weight 0.02 \
  --delta_clip_abs 0.05 \
  --grad_clip_norm 1.0 \
  --backward_loss_scale 1.0 \
  --log_every 5 \
  --save_every 20
```

结果：

```text
step 1-20:
  loss=1.50e-04
  但全部 skipped_nonfinite_grad=True
  grad_norm=nan

eval:
  iou=1.0
  mask/outside/prior direction 全为 0
  这是 no-op，不是有效结果。
```

### 26.7 B5.5b finite 化与 gradient-clean 调试

继续做两处稳定性修正：

```text
1. adapter 输入 cond_base / physical_features 做 nan_to_num 和 clamp；
2. MLP 每层 hidden activation 做 nan_to_num 和 clamp；
3. 增加 --nan_to_num_grads 调试选项，观察是否存在可用有限梯度。
```

最终 gradient-clean 短测命令：

```bash
cd /home/zjr/Tracker

CUDA_VISIBLE_DEVICES=5 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/zjr/anaconda3/envs/reconviagen/bin/python -u reconvggt_ar_adapter_a/train_b55_physical_proxy_adapter.py \
  --sessions_json reconvggt_ar_adapter_a/configs/b53_multisession_train2_holdout1.json \
  --output_dir /home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/ar_b55_condbasis_proxy_train2_holdout1_s5_gradclean \
  --pretrained Stable-X/trellis-vggt-v0-2 \
  --max_views 7 \
  --device cuda \
  --low_vram \
  --mask_mode apply \
  --mask_background black \
  --mask_threshold 127 \
  --patch_start_idx 5 \
  --seed 42 \
  --num_samples 1 \
  --ss_steps 12 \
  --ss_cfg_strength 7.5 \
  --ss_guidance_rescale 0.5 \
  --ss_rescale_t 3.0 \
  --ss_grid_side 16 \
  --sparse_resolution 64 \
  --prior_radius 4.0 \
  --physical_vh_min_visible_views 1 \
  --physical_vh_min_support_ratio 0.5 \
  --projection_min_support_views 1 \
  --projection_min_support_ratio 0.0 \
  --visual_hull_min_visible_views 1 \
  --visual_hull_min_support_ratio 0.0 \
  --use_prior_score_positive \
  --visual_hull_active_weight 0.25 \
  --hidden_dim 256 \
  --proxy_loss_mode cond_basis \
  --max_steps 5 \
  --lr 1.0e-4 \
  --t_values 0.5,0.75,0.9 \
  --train_runtime_scale 1.0 \
  --eval_runtime_scales 0.25,0.5,1.0 \
  --cond_basis_pos_target 0.01 \
  --cond_basis_neg_target 0.005 \
  --pos_weight 1.0 \
  --neg_weight 2.0 \
  --preserve_weight 0.2 \
  --delta_norm_weight 0.05 \
  --smooth_weight 0.02 \
  --delta_clip_abs 0.05 \
  --grad_clip_norm 1.0 \
  --nan_to_num_grads \
  --log_every 1 \
  --save_every 5
```

结果：

```text
step 1-5:
  loss=1.50e-04
  grad=0.0
  sanitized_grad=662812
  delta_norm≈6.32e-13
  dpos=0
  dneg=0

eval:
  adapter_ap0p25 / ap0p5 / ap1:
    mean_iou=1.0
    mask=0
    outside=0
    prior=0
  仍然是 no-op。
```

### 26.8 本节结论

```text
B5.5a decoder-logit differentiable proxy:
  不可用。直接训练会 NaN；缩小 LR / scale / backward loss scale 后仍然 non-finite grad。

B5.5b condition-basis proxy:
  也没有形成有效训练。即便绕开 decoder，梯度仍大面积 non-finite；
  gradient-clean 后可用梯度为 0，adapter 没有实际更新。

当前 B5.5 的关键问题不是 candidate 不够好，
而是“从 cond residual adapter 到有效 SS 物理约束”的可训练路径仍未打通。
```

需要特别注意：

```text
早期 report 中出现过 strict=3/3、iou=1.0 的结果，
这是旧 aggregate 判据把 no-op 当作通过；
已修正为必须发生 sparse set 改变才计入 strict。
```

### 26.9 下一步建议

不要继续在当前 B5.5 adapter MLP 上加步数或 sweep。理由：

```text
1. decoder-logit proxy 数值不稳定；
2. condition-basis proxy 不能产生有效梯度；
3. no-op 已经被误判过一次，继续 sweep 容易制造假阳性。
```

更合理的下一步是换训练位置，而不是继续修这个 adapter：

```text
路线 1：SS-condition residual 改成显式 low-rank delta basis
  不再让 MLP 自己产生 1024-delta；
  使用有限、可控、已归一化的 K 个 basis；
  只学习每个 physical token 的 scalar gate。

路线 2：直接训练 sparse output correction
  在 sparse coords / occupancy 后处理层做可解释 correction；
  loss 用 prior/mask/visual-hull，避免穿过 sparse decoder。

路线 3：若坚持学习 ReconViaGen 内部 bridge
  不建议 full 解冻 get_ss_cond；
  先做 sparse_structure_vggt_cond 后的小 LoRA / low-rank residual，
  但训练目标必须先解决 finite gradient 与 no-op 判据问题。
```

当前最推荐：

```text
B5.6 = low-rank physical residual basis。

核心改变：
  cond_delta[token] = scalar_gate(physical_features[token]) * learned_basis[k]

优点：
  梯度只穿过 scalar gate 和少量 basis；
  不再让大 MLP 直接输出 1024-delta；
  可以显式限制 delta norm；
  可以先用 eval-time basis sweep，再训练 gate。
```

## 27. B5.6 low-rank physical residual basis 实验

### 27.1 本节目的

上一节 B5.5 的大 MLP residual adapter 失败点是：

```text
1. decoder-logit proxy 反传不稳定；
2. condition-basis proxy 也出现大面积 non-finite grad；
3. gradient-clean 后 grad=0，adapter 实际没有更新。
```

因此本节按 26.9 的建议改成 B5.6：

```text
cond_delta[token] = scalar_gate(physical_features[token]) * learned_basis[k]
```

目标是验证：

```text
1. 不再让 MLP 直接输出 1024-delta；
2. 只学习少量 scalar gate + low-rank basis；
3. gate loss 不穿过 sparse decoder；
4. 真实 sparse sampling 仍用于最终 eval。
```

### 27.2 修改代码

修改文件：

```text
reconvggt_ar_adapter_a/train_b55_physical_proxy_adapter.py
```

新增结构：

```text
LowRankPhysicalSSCondResidualAdapter
```

核心实现：

```text
physical_features -> gate_mlp -> gates [B,T,R]
learned basis [R,C]
delta = einsum(gates, normalize(basis))
delta *= active_gate
```

新增参数：

```text
--adapter_type lowrank_basis
--lowrank_rank
--lowrank_basis_init_std
--proxy_loss_mode lowrank_gate
--gate_pos_target
--gate_neg_target
--gate_l2_weight
```

新增训练 loss：

```text
positive token:
  gate_score -> +gate_pos_target

negative/outside token:
  gate_score -> -gate_neg_target

neutral token:
  gate_score -> 0

regularization:
  gate_l2
  delta_norm
  smoothness
```

中途修改：

```text
第一次 low-rank gate 仍然 non-finite grad；
随后去掉 low-rank gate 的 LayerNorm，改成 raw physical features 直接进 Linear/SiLU/Linear；
再补 rawphys 和 rawphys+gradient-clean 短测。
```

### 27.3 代码检查命令

```bash
cd /home/zjr/Tracker

/home/zjr/anaconda3/envs/reconviagen/bin/python -m py_compile \
  reconvggt_ar_adapter_a/train_b55_physical_proxy_adapter.py
```

结果：

```text
py_compile 通过。
```

### 27.4 B5.6 low-rank gate s20

命令：

```bash
cd /home/zjr/Tracker

CUDA_VISIBLE_DEVICES=5 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/zjr/anaconda3/envs/reconviagen/bin/python -u reconvggt_ar_adapter_a/train_b55_physical_proxy_adapter.py \
  --sessions_json reconvggt_ar_adapter_a/configs/b53_multisession_train2_holdout1.json \
  --output_dir /home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/ar_b56_lowrank_gate_train2_holdout1_s20_rank4_lr1e4 \
  --pretrained Stable-X/trellis-vggt-v0-2 \
  --max_views 7 \
  --device cuda \
  --low_vram \
  --mask_mode apply \
  --mask_background black \
  --mask_threshold 127 \
  --patch_start_idx 5 \
  --seed 42 \
  --num_samples 1 \
  --ss_steps 12 \
  --ss_cfg_strength 7.5 \
  --ss_guidance_rescale 0.5 \
  --ss_rescale_t 3.0 \
  --ss_grid_side 16 \
  --sparse_resolution 64 \
  --prior_radius 4.0 \
  --physical_vh_min_visible_views 1 \
  --physical_vh_min_support_ratio 0.5 \
  --projection_min_support_views 1 \
  --projection_min_support_ratio 0.0 \
  --visual_hull_min_visible_views 1 \
  --visual_hull_min_support_ratio 0.0 \
  --use_prior_score_positive \
  --visual_hull_active_weight 0.25 \
  --adapter_type lowrank_basis \
  --hidden_dim 128 \
  --lowrank_rank 4 \
  --lowrank_basis_init_std 0.02 \
  --proxy_loss_mode lowrank_gate \
  --max_steps 20 \
  --lr 1.0e-4 \
  --t_values 0.5,0.75,0.9 \
  --train_runtime_scale 1.0 \
  --eval_runtime_scales 1.0,2.0,4.0 \
  --gate_pos_target 0.02 \
  --gate_neg_target 0.01 \
  --pos_weight 1.0 \
  --neg_weight 2.0 \
  --preserve_weight 0.2 \
  --gate_l2_weight 0.01 \
  --delta_norm_weight 0.05 \
  --smooth_weight 0.02 \
  --delta_clip_abs 0.1 \
  --grad_clip_norm 1.0 \
  --log_every 5 \
  --save_every 20
```

结果：

```text
output:
  /home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/ar_b56_lowrank_gate_train2_holdout1_s20_rank4_lr1e4/report.json

adapter:
  LowRankPhysicalSSCondResidualAdapter
  rank=4
  hidden_dim=128

train:
  step 1-20:
    loss=6.00e-04
    skipped_nonfinite_grad=True
    grad_norm=nan

eval:
  adapter_ap1 / ap2 / ap4:
    mean_iou=1.0
    mask=0
    outside=0
    prior=0
    strict=0/3
```

结论：

```text
low-rank basis 结构本身没有解决 non-finite grad；
因为所有 step 都被 skipped_nonfinite_grad 跳过，所以 eval 是 no-op。
```

### 27.5 B5.6 raw physical features s5

为了排除 LayerNorm 导致 non-finite grad，本节把 low-rank gate 内的 LayerNorm 去掉，直接使用已经 nan_to_num/clamp 的 raw physical features。

命令：

```bash
cd /home/zjr/Tracker

CUDA_VISIBLE_DEVICES=5 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/zjr/anaconda3/envs/reconviagen/bin/python -u reconvggt_ar_adapter_a/train_b55_physical_proxy_adapter.py \
  --sessions_json reconvggt_ar_adapter_a/configs/b53_multisession_train2_holdout1.json \
  --output_dir /home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/ar_b56_lowrank_gate_rawphys_train2_holdout1_s5_rank4_lr1e4 \
  --pretrained Stable-X/trellis-vggt-v0-2 \
  --max_views 7 \
  --device cuda \
  --low_vram \
  --mask_mode apply \
  --mask_background black \
  --mask_threshold 127 \
  --patch_start_idx 5 \
  --seed 42 \
  --num_samples 1 \
  --ss_steps 12 \
  --ss_cfg_strength 7.5 \
  --ss_guidance_rescale 0.5 \
  --ss_rescale_t 3.0 \
  --ss_grid_side 16 \
  --sparse_resolution 64 \
  --prior_radius 4.0 \
  --physical_vh_min_visible_views 1 \
  --physical_vh_min_support_ratio 0.5 \
  --projection_min_support_views 1 \
  --projection_min_support_ratio 0.0 \
  --visual_hull_min_visible_views 1 \
  --visual_hull_min_support_ratio 0.0 \
  --use_prior_score_positive \
  --visual_hull_active_weight 0.25 \
  --adapter_type lowrank_basis \
  --hidden_dim 128 \
  --lowrank_rank 4 \
  --lowrank_basis_init_std 0.02 \
  --proxy_loss_mode lowrank_gate \
  --max_steps 5 \
  --lr 1.0e-4 \
  --t_values 0.5,0.75,0.9 \
  --train_runtime_scale 1.0 \
  --eval_runtime_scales 1.0,2.0,4.0 \
  --gate_pos_target 0.02 \
  --gate_neg_target 0.01 \
  --pos_weight 1.0 \
  --neg_weight 2.0 \
  --preserve_weight 0.2 \
  --gate_l2_weight 0.01 \
  --delta_norm_weight 0.05 \
  --smooth_weight 0.02 \
  --delta_clip_abs 0.1 \
  --grad_clip_norm 1.0 \
  --log_every 1 \
  --save_every 5
```

结果：

```text
output:
  /home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/ar_b56_lowrank_gate_rawphys_train2_holdout1_s5_rank4_lr1e4/report.json

train:
  step 1-5:
    loss=6.00e-04
    skipped_nonfinite_grad=True
    grad_norm=nan

eval:
  adapter_ap1 / ap2 / ap4:
    mean_iou=1.0
    mask=0
    outside=0
    prior=0
    strict=0/3
```

结论：

```text
去掉 LayerNorm 仍然 non-finite grad；
问题不是 LayerNorm。
```

### 27.6 B5.6 raw physical + gradient-clean s5

命令：

```bash
cd /home/zjr/Tracker

CUDA_VISIBLE_DEVICES=5 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/zjr/anaconda3/envs/reconviagen/bin/python -u reconvggt_ar_adapter_a/train_b55_physical_proxy_adapter.py \
  --sessions_json reconvggt_ar_adapter_a/configs/b53_multisession_train2_holdout1.json \
  --output_dir /home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/ar_b56_lowrank_gate_rawphys_train2_holdout1_s5_gradclean \
  --pretrained Stable-X/trellis-vggt-v0-2 \
  --max_views 7 \
  --device cuda \
  --low_vram \
  --mask_mode apply \
  --mask_background black \
  --mask_threshold 127 \
  --patch_start_idx 5 \
  --seed 42 \
  --num_samples 1 \
  --ss_steps 12 \
  --ss_cfg_strength 7.5 \
  --ss_guidance_rescale 0.5 \
  --ss_rescale_t 3.0 \
  --ss_grid_side 16 \
  --sparse_resolution 64 \
  --prior_radius 4.0 \
  --physical_vh_min_visible_views 1 \
  --physical_vh_min_support_ratio 0.5 \
  --projection_min_support_views 1 \
  --projection_min_support_ratio 0.0 \
  --visual_hull_min_visible_views 1 \
  --visual_hull_min_support_ratio 0.0 \
  --use_prior_score_positive \
  --visual_hull_active_weight 0.25 \
  --adapter_type lowrank_basis \
  --hidden_dim 128 \
  --lowrank_rank 4 \
  --lowrank_basis_init_std 0.02 \
  --proxy_loss_mode lowrank_gate \
  --max_steps 5 \
  --lr 1.0e-4 \
  --t_values 0.5,0.75,0.9 \
  --train_runtime_scale 1.0 \
  --eval_runtime_scales 1.0,2.0,4.0 \
  --gate_pos_target 0.02 \
  --gate_neg_target 0.01 \
  --pos_weight 1.0 \
  --neg_weight 2.0 \
  --preserve_weight 0.2 \
  --gate_l2_weight 0.01 \
  --delta_norm_weight 0.05 \
  --smooth_weight 0.02 \
  --delta_clip_abs 0.1 \
  --grad_clip_norm 1.0 \
  --nan_to_num_grads \
  --log_every 1 \
  --save_every 5
```

结果：

```text
output:
  /home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/ar_b56_lowrank_gate_rawphys_train2_holdout1_s5_gradclean/report.json

train:
  step 1-5:
    loss=6.00e-04
    sanitized_grad=6532
    grad=0.0
    delta_norm≈6.32e-13
    dpos=0
    dneg=0

eval:
  adapter_ap1 / ap2 / ap4:
    mean_iou=1.0
    mask=0
    outside=0
    prior=0
    strict=0/3
```

结论：

```text
gradient-clean 后仍然 grad=0；
说明 non-finite 梯度被清掉后没有剩余有效梯度；
low-rank gate 没有实际更新，eval 仍然 no-op。
```

### 27.7 B5.6 结论

```text
B5.6 证明：
  把 dense 1024-delta MLP 改成 low-rank physical residual basis，
  仍然不能解决当前训练路径的 non-finite gradient 问题。

更细地说：
  1. rank4 low-rank gate s20: 全部 skipped_nonfinite_grad；
  2. 去掉 LayerNorm 后 raw physical s5: 仍然 skipped_nonfinite_grad；
  3. raw physical + gradient-clean: sanitized_grad=6532，但 grad=0，仍然 no-op。
```

因此，当前失败不应再归因于：

```text
adapter 输出维度太大；
LayerNorm；
low-rank basis 容量不足；
eval scale 太小。
```

更可能的问题是：

```text
当前 physical feature / loss mask / gate objective 存在会导致反向梯度非有限的项；
或者 loss 的有效监督区域与 gate 初始化组合导致只有非有限梯度，没有可用有限梯度。
```

### 27.8 下一步建议

不要继续 B5.6 加步数、加 rank、加 eval scale。下一步应该先做 B5.7 gradient source audit，而不是继续训练：

```text
B5.7:
  1. 对每个 train session dump physical_features 的逐通道 finite/min/max/mean/std；
  2. dump pos16/neg16/neutral16 的 sum/min/max；
  3. 做一个完全脱离 TRELLIS/VGGT 的 toy backward：
       gates = Linear(raw_features)
       loss = current lowrank_gate_loss
     检查是否仍然 non-finite；
  4. 打印具体哪个参数 first non-finite grad；
  5. 若 toy backward 正常，再逐步加 active_gate / delta_norm / smoothness；
  6. 只有确定 finite gradient source 后，才回到 sparse sampling eval。
```

当前建议优先实现：

```text
train_b57_gradient_source_audit.py
```

不要再直接训练 adapter，否则只会继续产生 no-op report。

## 28. B5.7 梯度源审计与 B5.8 safe-norm lowrank smoke

### 28.1 目的

上一节 B5.6 的现象是：

```text
low-rank gate 仍然 skipped_nonfinite_grad；
nan_to_num_grads 后 sanitized_grad=6532，但 grad=0；
eval 仍然 no-op。
```

本节先不继续加 rank / step / eval scale，而是按 27.8 做梯度源审计：

```text
1. 检查 physical_features 是否有 NaN/Inf；
2. 检查 pos16/neg16/neutral16/active16 是否有限；
3. 脱离 TRELLIS/VGGT/sampler 做 toy Linear gate backward；
4. 复用 actual LowRank gate loss 做 backward；
5. 找出 non-finite gradient 的来源；
6. 修复后再跑一个短 B5.8 smoke。
```

### 28.2 本节新增和修改代码

新增：

```text
reconvggt_ar_adapter_a/train_b57_gradient_source_audit.py
```

用途：

```text
不加载 ReconViaGen / VGGT / sparse sampler；
只读取同一份 b53_multisession_train2_holdout1.json；
构造 physical_features 和 loss masks；
分别测试：
  toy_linear_zero_gate_only
  toy_linear_zero_gate_delta_norm
  toy_linear_zero_gate_delta_norm_smooth
  toy_linear_small_gate_only
  actual_lowrank_full_loss
并打印 first non-finite grad 参数。
```

修改：

```text
reconvggt_ar_adapter_a/train_b55_physical_proxy_adapter.py
```

具体改动：

```text
1. _smoothness_loss 不再对 zero residual 求 sqrt energy；
   改成对 squared energy 做平滑。

2. 新增 _delta_norm_stats(delta, cond_base)；
   delta_norm_loss = mean(delta^2) / mean(cond_base^2)
   delta_norm_ratio 只用于日志，不参与反传。

3. decoder_logits / cond_basis / lowrank_gate 三个分支全部改用 safe squared norm。
```

原因：

```text
zero-init residual 下，sqrt(mean(delta^2)).clamp_min(...)
在 delta=0 附近会导致 NaN gradient；
这正是 B5.6 non-finite grad 的来源。
```

### 28.3 B5.7 pre-fix 审计命令

```bash
cd /home/zjr/Tracker

CUDA_VISIBLE_DEVICES=5 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/zjr/anaconda3/envs/reconviagen/bin/python -u reconvggt_ar_adapter_a/train_b57_gradient_source_audit.py \
  --sessions_json reconvggt_ar_adapter_a/configs/b53_multisession_train2_holdout1.json \
  --output_dir /home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/ar_b57_gradient_source_audit_train2_holdout1 \
  --device cuda \
  --ss_grid_side 16 \
  --sparse_resolution 64 \
  --prior_radius 4.0 \
  --mask_threshold 127 \
  --physical_vh_min_visible_views 1 \
  --physical_vh_min_support_ratio 0.5 \
  --visual_hull_active_weight 0.25 \
  --hidden_dim 128 \
  --lowrank_rank 4 \
  --lowrank_basis_init_std 0.02 \
  --gate_pos_target 0.02 \
  --gate_neg_target 0.01 \
  --pos_weight 1.0 \
  --neg_weight 2.0 \
  --preserve_weight 0.1 \
  --gate_l2_weight 0.01 \
  --delta_norm_weight 0.02 \
  --smooth_weight 0.01
```

结果：

```text
all_features_finite=True
toy_all_finite=False
actual_lowrank_all_finite=False

toy_linear_zero_gate_only:
  loss finite
  grad finite

toy_linear_zero_gate_delta_norm:
  loss finite
  grad non-finite
  first = linear.weight[0,0] = nan

actual_lowrank_full_loss:
  loss finite
  grad non-finite
  nonfinite_grad = 6532
  first = basis[0,0] = nan
```

物理特征本身是有限的：

```text
ar_20260617_075401_819:
  pos16 sum=242.44
  neg16 sum=1898.90
  active16 sum=2060.95
  all mask finite

ar_20260624_010530_849:
  pos16 sum=178.67
  neg16 sum=3359.28
  active16 sum=3415.14
  all mask finite
```

结论：

```text
B5.6 失败不是 physical feature 有 NaN/Inf；
也不是 gate-only objective 无梯度；
核心错误是 delta_norm / smoothness 中的 sqrt-style zero residual norm。
```

### 28.4 safe-norm 修复后 B5.7 审计命令

```bash
cd /home/zjr/Tracker

CUDA_VISIBLE_DEVICES=5 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/zjr/anaconda3/envs/reconviagen/bin/python -u reconvggt_ar_adapter_a/train_b57_gradient_source_audit.py \
  --sessions_json reconvggt_ar_adapter_a/configs/b53_multisession_train2_holdout1.json \
  --output_dir /home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/ar_b57_gradient_source_audit_train2_holdout1_safenorm \
  --device cuda \
  --ss_grid_side 16 \
  --sparse_resolution 64 \
  --prior_radius 4.0 \
  --mask_threshold 127 \
  --physical_vh_min_visible_views 1 \
  --physical_vh_min_support_ratio 0.5 \
  --visual_hull_active_weight 0.25 \
  --hidden_dim 128 \
  --lowrank_rank 4 \
  --lowrank_basis_init_std 0.02 \
  --gate_pos_target 0.02 \
  --gate_neg_target 0.01 \
  --pos_weight 1.0 \
  --neg_weight 2.0 \
  --preserve_weight 0.1 \
  --gate_l2_weight 0.01 \
  --delta_norm_weight 0.02 \
  --smooth_weight 0.01
```

结果：

```text
all_features_finite=True
toy_all_finite=True
actual_lowrank_all_finite=True

toy_linear_zero_gate_only:
  grad finite

toy_linear_zero_gate_delta_norm:
  grad finite

toy_linear_zero_gate_delta_norm_smooth:
  grad finite

actual_lowrank_full_loss:
  grad finite
  nonfinite_grad=0
```

结论：

```text
safe squared norm 修复了 B5.6 的非有限梯度源。
```

### 28.5 B5.8 safe-norm lowrank gate s5 smoke 命令

```bash
cd /home/zjr/Tracker

CUDA_VISIBLE_DEVICES=5 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/zjr/anaconda3/envs/reconviagen/bin/python -u reconvggt_ar_adapter_a/train_b55_physical_proxy_adapter.py \
  --sessions_json reconvggt_ar_adapter_a/configs/b53_multisession_train2_holdout1.json \
  --output_dir /home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/ar_b58_lowrank_gate_safenorm_train2_holdout1_s5_rank4_lr1e4 \
  --pretrained Stable-X/trellis-vggt-v0-2 \
  --max_views 7 \
  --device cuda \
  --low_vram \
  --mask_mode apply \
  --mask_background black \
  --mask_threshold 127 \
  --patch_start_idx 5 \
  --seed 42 \
  --num_samples 1 \
  --ss_steps 12 \
  --ss_cfg_strength 7.5 \
  --ss_guidance_rescale 0.5 \
  --ss_rescale_t 3.0 \
  --ss_grid_side 16 \
  --sparse_resolution 64 \
  --prior_radius 4.0 \
  --physical_vh_min_visible_views 1 \
  --physical_vh_min_support_ratio 0.5 \
  --projection_min_support_views 1 \
  --projection_min_support_ratio 0.0 \
  --visual_hull_min_visible_views 1 \
  --visual_hull_min_support_ratio 0.0 \
  --use_prior_score_positive \
  --visual_hull_active_weight 0.25 \
  --adapter_type lowrank_basis \
  --hidden_dim 128 \
  --lowrank_rank 4 \
  --lowrank_basis_init_std 0.02 \
  --proxy_loss_mode lowrank_gate \
  --max_steps 5 \
  --lr 1.0e-4 \
  --t_values 0.5,0.75,0.9 \
  --train_runtime_scale 1.0 \
  --eval_runtime_scales 1.0,2.0,4.0 \
  --gate_pos_target 0.02 \
  --gate_neg_target 0.01 \
  --pos_weight 1.0 \
  --neg_weight 2.0 \
  --preserve_weight 0.2 \
  --gate_l2_weight 0.01 \
  --delta_norm_weight 0.05 \
  --smooth_weight 0.02 \
  --delta_clip_abs 0.1 \
  --grad_clip_norm 1.0 \
  --log_every 1 \
  --save_every 5
```

训练结果：

```text
step 1:
  loss=6.0000e-04
  grad=0.240226
  delta_norm=0
  dpos=0
  dneg=0

step 2:
  loss=4.5243e-04
  grad=0.257567
  delta_norm=3.084e-04
  dpos=0.001658
  dneg=-0.007989

step 3:
  loss=3.5577e-04
  grad=0.079602
  delta_norm=2.952e-04
  dpos=0.008382
  dneg=-0.005404

step 4:
  loss=3.8906e-04
  grad=0.184348
  delta_norm=2.767e-04
  dpos=0.013988
  dneg=-0.002122

step 5:
  loss=3.9562e-04
  grad=0.211845
  delta_norm=2.844e-04
  dpos=0.015536
  dneg=-0.002122
```

这说明：

```text
safe-norm 后训练不再 skipped；
loss 有下降；
adapter 不再是 no-op。
```

Sparse eval 汇总：

```text
adapter_ap2:
  strict=1/3
  mean_iou=0.995542
  mean_mask=0.115338
  mean_outside=-0.035166
  mean_prior=0.061091
  max_outside=0.065119

adapter_ap4:
  strict=1/3
  mean_iou=0.994992
  mean_mask=-0.053476
  mean_outside=-0.002367
  mean_prior=-0.156035
  max_outside=0.040365

adapter_ap1:
  strict=0/3
  mean_iou=0.996059
  mean_mask=0.024691
  mean_outside=0.019999
  mean_prior=-0.072751
  max_outside=0.080548
```

Per-session 关键结果：

```text
adapter_ap2:
  ar_20260617_075401_819 train:
    mask=+0.3460
    outside=-0.1866
    prior=+0.1486
    strict pass

  ar_20260624_010530_849 train:
    mask=0.0
    outside=+0.0651
    prior=+0.2222
    outside fail

  ar_20260624_011149_207 holdout:
    mask=0.0
    outside=+0.0160
    prior=-0.1875
    outside/prior fail
```

### 28.6 本节结论

本节解决了一个真实代码错误：

```text
旧 delta_norm / smoothness 使用 sqrt-style residual norm；
zero-init residual 下会产生 NaN gradient；
这是 B5.6 sanitized_grad=6532 且 no-op 的直接原因。
```

修复后：

```text
B5.7 safe-norm 审计全部 finite；
B5.8 s5 训练可正常更新；
因此 lowrank physical proxy 训练路径现在数值上可运行。
```

但 B5.8 还没有证明路线有效：

```text
s5 只在 1/3 session 上满足 strict direction；
holdout 仍然 prior/outside 方向不稳；
当前 lowrank gate objective 仍不能直接作为成功路线。
```

### 28.7 下一步建议

现在不要回到 B5.6 的旧结果，也不要直接加很长训练。建议按以下顺序：

```text
1. B5.8b: safe-norm lowrank gate s20；
   目标只看训练是否稳定，以及 strict 是否从 1/3 提升。

2. 如果 s20 仍然 holdout 不稳：
   调整 loss，而不是继续加步数：
     - 增大 outside negative 权重；
     - 降低 prior positive 权重；
     - 或把 pos target 只给 mask+prior 同时满足的 token。

3. 如果 s20 出现 train-only 改善、holdout 恶化：
   停止 direct physical proxy loss；
   转向更监督明确的 SS-condition residual teacher 或直接 sparse coord correction。
```

当前最合理的下一条命令是：

```text
B5.8b safe-norm lowrank gate s20
```

## 29. B5.9 互斥 surface candidate：selected7/fullscan 三 seed 结果

### 29.1 实验状态

本节完成了以下六组 candidate search：

```text
selected7 feature scope:
  seed 42 / 43 / 44

fullscan feature scope:
  seed 42 / 43 / 44

canonical evaluation scope:
  六组实验全部固定为 fullscan
```

两个 multi-seed 汇总均完整读取三个 seed，且配置一致性检查通过：

```text
selected7:
  seeds=[42,43,44]
  configuration_invariant_passed=true
  passing_candidates=[]

fullscan:
  seeds=[42,43,44]
  configuration_invariant_passed=true
  passing_candidates=[]
```

因此本节不是文件缺失或汇总脚本失败，而是没有同一个 candidate 能同时通过三个 session 和三个 seed。

### 29.2 mapping、neutral 与 surface label 检查

代码正确性条件均通过：

```text
16^3 token <-> 64^3 block tensor round-trip: pass
surface_contrast neutral leakage: 0
positive/negative overlap: 0
evaluation frame scope: fullscan
```

selected7 feature scope 的标签数：

| session | split | positive | negative | neutral |
|---|---:|---:|---:|---:|
| ar_20260617_075401_819 | train | 137 | 1021 | 2938 |
| ar_20260624_010530_849 | train | 24 | 937 | 3135 |
| ar_20260624_011149_207 | validation | 11 | 648 | 3437 |

fullscan feature scope 的标签数：

| session | split | positive | negative | neutral |
|---|---:|---:|---:|---:|
| ar_20260617_075401_819 | train | 137 | 1021 | 2938 |
| ar_20260624_010530_849 | train | 22 | 1829 | 2245 |
| ar_20260624_011149_207 | validation | 12 | 1205 | 2879 |

第一组 session 本身只有 7 个 frame，因此 selected7/fullscan 标签相同。后两组 fullscan 增加了大量 reliable-outside token，但没有产生跨 seed 稳定 candidate。

### 29.3 strict check 分解

每个 scope 共计：

```text
12 candidates x 3 seeds x 3 sessions = 108 candidate-session rows
```

通过数量如下：

| check | selected7 | fullscan |
|---|---:|---:|
| direction_passed | 20 / 108 | 24 / 108 |
| changed_ratio_passed | 72 / 108 | 68 / 108 |
| set_iou_passed | 92 / 108 | 88 / 108 |
| absolute_outside_passed | 65 / 108 | 56 / 108 |
| structure_passed | 107 / 108 | 107 / 108 |
| all strict checks | 14 / 108 | 15 / 108 |

这说明：

```text
component/coord 判据不是主要失败源；
set IoU 和结构保持大部分可通过；
主要瓶颈是 mask/outside/prior 三个物理方向无法同时、稳定地满足。
```

按 session 汇总 strict pass：

| scope | train 819 | train 849 | validation 207 |
|---|---:|---:|---:|
| selected7 | 4 / 36 | 0 / 36 | 10 / 36 |
| fullscan | 4 / 36 | 1 / 36 | 10 / 36 |

`train 849` 是最明显的瓶颈，但问题不只来自它。`train 819` 有 137 个 positive token，direction 仍随 seed 变号，因此不能把失败完全归因于 validation positive 过少。

### 29.4 最接近的 candidate 也不稳定

`cond_centered_unit_surface_m0p005` 是两个 scope 中综合最接近的配置之一。

selected7：

```text
mean changed ratio = 0.05596
max component delta = 0
direction pass = 2 / 9 candidate-session rows
strict pass = 2 / 9
```

fullscan：

```text
mean changed ratio = 0.07114
max component delta = 0
direction pass = 4 / 9 candidate-session rows
strict pass = 3 / 9
```

同一 candidate 的典型变号：

```text
fullscan / train 819:
  seed42: mask=+0.2447 outside=-0.0839 prior=+0.2140
  seed43: mask=-0.1407 outside=+0.0813 prior=-0.0887
  seed44: mask=+0.0781 outside=-0.0323 prior=-0.1806
```

这不是 scale 太小导致的 no-op。changed ratio 已经达到几个百分点；真正问题是 `cond_unit` / `cond_centered_unit` channel direction 与 occupancy 修正方向没有稳定对应关系。

### 29.5 本节结论

```text
B5.9 FAIL。

互斥 spatial labels、neutral=0、mapping 和 canonical evaluation 已经修正；
正负 scale、两种 basis、三个强度、三个 session、三个 seed 都已覆盖；
但没有跨 session、跨 seed 的稳定 deterministic channel direction。
```

因此：

```text
不要按现有 30.1 -> 30.2 直接进入 adapter 训练；
不要继续扩大 deterministic scale sweep；
不要把少数 candidate-session strict pass 当成机制成立。
```

### 29.6 下一步建议

下一步应新增 **B5.9c frozen flow-gradient channel-direction audit**，仍然不训练 adapter：

```text
1. 在 baseline condition 和相同 high-t state 上，
   对 exclusive positive/outside occupancy proxy 求 dL/d(cond)。

2. 对两个 train session、validation、三个 seed 分别报告：
   - positive / negative / neutral gradient norm；
   - 同 session 跨 seed cosine similarity；
   - 跨 session cosine similarity；
   - gradient low-rank SVD explained variance。

3. 用负梯度或共享 rank-r gradient basis 形成 eval-only residual，
   继续使用 B5.9 的 fullscan canonical metrics 和 strict checks。
```

判断分支：

```text
gradient direction 跨 seed/session 稳定，且 eval-only residual 过线：
  才进入 B5.10 low-rank adapter 学习。

gradient 本身跨 seed/session 不稳定：
  停止 SS-condition channel residual 路线；
  转 synthetic supervised occupancy / sparse coordinate correction，
  或直接在 sparse occupancy/coords 层施加物理约束。
```

当前 30.1 connectivity audit 只能证明梯度能传播，不能回答梯度方向是否共享稳定，因此不能替代 B5.9c。

### 29.7 sampler 路径复核

2026-07-11 重新核对实际 pretrained 配置与 Python MRO：

```text
Stable-X/trellis-vggt-v0-2 sparse sampler:
  FlowEulerGuidanceIntervalSampler

有效继承路径:
  FlowEulerGuidanceIntervalSampler
  -> FlowEulerSampler.sample

结论:
  该路径使用调用方传入的 noise；
  flow_euler.py 中重新 torch.randn_like(x_1) 的实现属于 FlowMatchingSampler，
  不是当前 sparse sampler 的运行路径。
```

因此 29.1-29.6 的 B5.9 结果不需要因“内部重采样”而作废。为了防止未来配置或 sampler 实现变化，评估代码新增：

```text
native pipeline + explicitly supplied noise；
同一 baseline condition/noise 连续采样两次；
不重置 RNG；
coords 必须逐项完全相同，否则 hard fail。
```

这项检查是确定性保护，不替换 ReconViaGen 原 sampler，也不手写另一套 Euler 采样器。

## 30. B5.9c frozen flow-gradient channel-direction audit 实现

### 30.1 目的

B5.9 已证明手工 `cond_unit / cond_centered_unit` channel basis 无法跨 session/seed 稳定工作。B5.9c 不立即训练 adapter，而是直接询问 frozen ReconViaGen flow/decoder：

```text
在 stock trajectory 的 t=0.90 状态、sampler-consistent CFG 下，
exclusive positive / reliable-outside / neutral proxy loss
对 SS condition 的负梯度方向是什么？
只用 train 梯度拟合的方向能否跨 seed、跨 session 并泛化到 validation？
```

### 30.2 新增代码

新增：

```text
reconvggt_ar_adapter_a/audit_b59c_flow_gradient_direction.py
```

主要实现：

```text
1. frozen bridge / sparse flow / sparse decoder；不创建 optimizer；
2. 对 3 sessions x 3 seeds 分别计算 dL/d(cond)；
3. 报告 positive/negative/neutral token gradient RMS；
4. 分别报告 train-only 和 all-record diagnostic 的同 session 跨 seed、跨 session channel cosine；
5. 分别做两套 SVD；all-record SVD 只用于诊断，不构造正式候选；
6. shared_train_channel 只由 2 train sessions x 3 seeds 拟合；
7. 构造 oracle_full、record_rank1、shared_train_rank1 三类 eval-only residual；
8. validation 梯度不参与 shared_train_rank1 拟合，但 validation rollout 参与泛化判断；
9. 在相同 initial noise 下执行真实 sparse rollout 和 B5.9 strict checks；
10. 每个 session/seed baseline 都执行 fixed-noise repeat hard check。
```

修改：

```text
run_b54_physical_ssgrid_smoke.py
  增加统一 supplied-noise sampling helper 和 baseline repeat 保护。

train_b55_physical_proxy_adapter.py
  B5.10 rollout eval 复用同一 supplied-noise helper，并执行 baseline repeat 保护。
```

### 30.3 验证

已完成：

```text
三个修改文件 py_compile: PASS
B5.9c --help CLI: PASS
supplied-noise helper CPU mock:
  同 noise 连续调用 coords 完全相同；
  helper 使用 clone，不修改调用方 noise；
  PASS

train-only leakage synthetic audit:
  固定 train vectors，分别替换两组 validation vectors；
  shared_train_channel 保持不变；
  all-record diagnostic channel 随 validation 改变；
  PASS
```

CUDA 模型级 B5.9c 尚未在本次代码修改中运行；完整命令已追加到 `命令说明.txt` 第 31 节。

### 30.4 下一步判定

```text
shared_train_rank1 在全部 9 条 session/seed 记录上 strict pass，
且 validation_gradient_used_for_fit=false：
  才进入 B5.10b low-rank adapter s5。

只有 oracle_full / record_rank1 通过：
  说明存在局部物理方向，但不存在共享 rank-1 basis；
  暂停全局 adapter，研究 rank-2/rank-4 或 session-conditioned basis。

oracle_full 也不能稳定通过：
  说明 SS-condition 层的局部一阶方向不能转化为稳定 rollout 修正；
  停止 SS-condition residual，转 sparse occupancy/coordinate 直接监督。
```

## 31. B5.9c train-only flow-gradient audit 结果

### 31.1 数据隔离与确定性检查

运行目录：

```text
reconvggt_ar_adapter_a/outputs/
  ar_b59c_trainfit_flowgrad_stocktraj_cfg_selected7_t090_seed42_43_44_v2
```

关键正确性条件全部满足：

```text
sampler_class = FlowEulerGuidanceIntervalSampler
fixed_noise_repeat = true（9 / 9）

shared direction fit:
  fit_split = train
  fit_record_count = 6
  fit_sessions = 2 train sessions
  validation_gradient_used_for_fit = false

evaluation:
  6 train records
  3 validation records
```

因此本次失败不是 initial-noise 漂移，也不是 validation 梯度泄漏。`shared_train_rank1` 是仅由两组 train session 的 6 条梯度拟合，然后在完整 9 条 rollout 上评估。

### 31.2 train 梯度方向不具备稳定 rank-1 共享结构

train-only alignment：

| 指标 | 结果 |
|---|---:|
| within-session channel cosine mean | 0.2158 |
| within-session channel cosine min | -0.3735 |
| within-session full-gradient cosine mean | 0.0938 |
| across-train-session channel cosine | -0.3713 |
| rank-1 explained variance | 0.3944 |
| rank-4 explained variance | 0.8907 |

两个 train session 呈现不同状态：

```text
train 819:
  三个 seed 的 channel cosine = -0.3299 / -0.3735 / +0.1555
  同一 session 内方向已经频繁反号。

train 849:
  三个 seed 的 channel cosine = +0.6211 / +0.6707 / +0.5508
  session 内相对稳定。

两个 train session 均值方向:
  cosine = -0.3713
```

这说明不存在可由当前两个 train session 支持的稳定全局 rank-1 channel direction。`rank-4 explained variance=0.8907` 只能说明六条拟合 channel vector 的能量可由较低维子空间描述，不能证明 rank-4 residual 会形成正确的物理 rollout。

### 31.3 surface-gated rank-1 本身不是 oracle 梯度的有效近似

每条记录中，`surface_contrast x fitted_channel` 对完整 token-wise oracle direction 的拟合 cosine 仅为：

```text
min approximately 0.0343
max approximately 0.0591
```

也就是说，oracle gradient 的主要结构不是：

```text
一个全局 channel vector
x
positive/negative surface scalar gate
```

它具有明显的 token/state/session 依赖。即使把共享 basis 从 rank-1 扩到 rank-4，当前简单 spatial gate 仍不能表达主要梯度结构。

### 31.4 rollout 结果

| candidate | strict pass | train | validation | direction pass |
|---|---:|---:|---:|---:|
| shared_train_rank1, scale 0.005 | 3 / 9 | 2 / 6 | 1 / 3 | 4 / 9 |
| record_rank1, scale 0.005 | 2 / 9 | 1 / 6 | 1 / 3 | 5 / 9 |
| shared_train_rank1, scale 0.0025 | 1 / 9 | 0 / 6 | 1 / 3 | 2 / 9 |
| oracle_full, scale 0.0025 | 0 / 9 | 0 / 6 | 0 / 3 | 5 / 9 |
| oracle_full, scale 0.005 | 0 / 9 | 0 / 6 | 0 / 3 | 3 / 9 |
| record_rank1, scale 0.0025 | 0 / 9 | 0 / 6 | 0 / 3 | 1 / 9 |

最好的 `shared_train_rank1_p0p005` 仍有明显不稳定性：

```text
mean changed ratio = 0.1420
max changed ratio = 0.4133

failure counts over 9 records:
  direction = 5
  changed ratio = 3
  set IoU = 3
  absolute outside = 2
  structure = 0
```

因此结构连通性不是主因。主要问题是物理方向随 seed/session 改变，并且 condition residual 经完整 flow rollout 后会被不一致地放大。

`oracle_full` 的结果还暴露了尺度问题：

```text
scale 0.0025:
  mean changed ratio = 0.5375
  max changed ratio = 1.2088

scale 0.005:
  mean changed ratio = 0.9181
  max changed ratio = 1.5633
```

所以不能把 `oracle_full=0/9` 单独解释为“一阶梯度完全没有局部价值”；当前归一化后尺度已经远离小扰动区间。但这不改变主结论：train-only shared rank-1 在合理 changed-ratio 范围内也无法稳定泛化，当前 B5.10 全局 SS-condition adapter 没有训练依据。

### 31.5 最终结论

```text
B5.9c FAIL。

已排除：
  validation 梯度泄漏；
  sampler initial-noise 不一致；
  单纯 component/coord 结构判据过严。

已确认：
  train 内跨 seed 方向不稳定；
  train session 之间方向反相关；
  surface-gated global channel basis 对 oracle gradient 拟合极弱；
  condition residual 经 12-step rollout 后产生过大且方向不稳定的 topology 变化。
```

因此停止：

```text
现有 B5.10 low-rank SS-condition residual 训练；
shared rank-1 的继续 scale sweep；
直接依据 rank4 explained variance 进入 rank-4 adapter；
继续增加 B5.x 训练步数。
```

### 31.6 下一步：B6 direct occupancy-logit correction

下一阶段不再在 `get_ss_cond` 输出上施加 residual，而是在 frozen sparse rollout 完成后直接修正 decoder occupancy logits：

```text
images / VGGT / ReconViaGen bridge
  -> frozen sparse flow rollout
  -> final sparse latent
  -> frozen sparse decoder logits L_base

AR point prior + poses + masks
  -> 64^3 physical fields
     positive surface support
     reliable outside evidence
     mask / visual-hull support
     normalized prior distance

L_corrected = L_base + DeltaL(physical fields, L_base)
```

优先执行 B6.0，不训练：

```text
1. 修改 sparse sampler，使一次 frozen rollout 同时缓存 final latent 和 L_base；
2. 所有 candidate 共用同一个 L_base，不再重复运行 flow；
3. 使用 neutral=0 的直接 logit residual：
     DeltaL = alpha_pos * positive64 - alpha_neg * negative64；
4. alpha_pos / alpha_neg 独立小范围 sweep；
5. 在 3 sessions x 3 seeds 上复用 B5.9 strict checks；
6. validation 只评估，不参与 alpha 选择；alpha 仅由 train 选择。
```

B6.0 的意义是验证：

```text
物理 evidence 在真正控制 occupancy 的位置是否存在稳定方向。
```

只有 B6.0 train-selected candidate 能泛化到 validation，才进入 B6.1 学习：

```text
freeze ReconViaGen；
训练 zero-init 3D residual-logit head；
使用 synthetic Objectverse GT occupancy；
损失包含 GT occupancy、outside negative、prior positive、neutral preservation；
严格拆分 train / validation / unused holdout；
real AR session 只做最终无 GT 物理指标评估。
```

如果 B6.0 直接 logit residual 仍无法产生跨 session 稳定方向，则不再训练 occupancy adapter，转为显式 sparse coordinate filtering/reranking，并把学习模块放到候选选择而不是生成 flow 内部。

## 32. PointPose object-level SS 数据修复与训练 smoke（2026-07-12）

### 32.1 旧 32.4/32.4a

旧 cache 的路径、shape、finite、K/T 和坐标映射检查正常，失败全部来自保存的 `target_coords`
与 `z -> SS decoder -> threshold_0 coords` 不一致：

| split | samples | failures | min IoU | mean IoU |
|---|---:|---:|---:|---:|
| train | 1259 | 4 | 0.780639 | 0.999678 |
| val | 175 | 2 | 0.922738 | 0.999089 |
| holdout | 182 | 0 | 0.999483 | 0.999986 |

诊断排除了 xyz 轴交换、翻转和整体平移；错误主要位于正确表面的 1 voxel 邻域，rank-cut 和
oracle-count 也无法恢复。这说明部分 mesh voxel occupancy 不能被 SS VAE 精确 round-trip。

### 32.2 同 object 的 sequence-level 随机 GT

旧 builder 为每个 sequence 重新随机执行 surface sampling。同一物体的不同 sequence 因而保存了
不同 occupancy、`target_coords` 和 `z`。全量只读统计：

```text
multi-sequence objects = 463
different z pairs = 456 / 463
exact z pairs = 7 / 463

target IoU min    = 0.843460
target IoU median = 0.970881
target IoU p90    = 0.997463

IoU < 0.90 = 3 / 463
IoU < 0.95 = 86 / 463
IoU < 0.99 = 348 / 463

z mean-abs-diff median = 0.096982
z mean-abs-diff max    = 0.624462
z max-abs-diff max     = 7.03125
```

这对 CFM 是实质监督噪声：变化的图像/pose/prior 本应映射到同一个 canonical full-shape target，
但旧数据加入了无法由观测推断的 Monte-Carlo voxelization 差异。它会增加 conditional vector field
方差、消耗 LoRA 容量并削弱多 sequence 的物体不变性监督。旧 prior 又由对应 target 派生，可能让
模型学习 synthetic prior 的离散捷径，而不是可迁移到真实 AR 点云的关系。

旧 root 停止用于正式训练：

```text
/data/reconvggt_pointpose_v9_odsplit_20260712
```

### 32.3 Object-level SS 修复

修复流程改为每个 object 一个稳定 seed、一次 surface sampling、一份 deterministic `z`，所有 sequence
共享相同 `z/target_coords`；图像、pose 和 sparse prior 仍按 sequence 变化。

```text
objects = 1153
sequences = 1616
same_object_consistency_failures = []
target_mode = decoder_projected
```

新 root：

```text
/data/reconvggt_pointpose_v9_ssfixed_odsplit_20260712
```

重新构建 prior/cache 后：

| split | samples | audit | decoder min/mean IoU |
|---|---:|---|---:|
| train | 1259 | PASS | 1.0 / 1.0 |
| val | 175 | PASS | 1.0 / 1.0 |
| holdout | 182 | PASS | 1.0 / 1.0 |

所有 nonfinite、越界、缺文件、latent shape、mapping 和 decoder failure 计数均为 0。overfit64 manifest
也满足 64 samples / 64 unique objects。

`decoder_projected` 定义为：

```text
mesh occupancy -> SS encoder -> z -> SS decoder threshold -> target_coords
```

所以 IoU=1 证明 latent 与 SS threshold target 自洽，不等于与原始 mesh surface 完全一致。修复数据
保留了 `mesh_target_coords`；SS flow 指标使用 decoder-projected target，物理/mesh 指标仍需对比
`mesh_target_coords`。

### 32.4 Repaired dataset FP16 s5

输出：

```text
reconvggt_ar_adapter_a/outputs/pointpose_ss_lora_ssfixed_single_s5
```

接线审计通过：120 个 LoRA modules 覆盖 24/24 flow blocks，VGGT/image encoder/decoder/bridge/base flow
均冻结，physical trainable=7,385,088，LoRA trainable=5,111,808。

step 1 在 `t=0.9877` 发生 FP16 overflow；GradScaler 从 65536 降至 32768 并跳过 update，但旧训练器
错误地增加了 global step。因此日志 5 step 实际只有 4 次成功 optimizer update。steps 2-5 finite，
step 3 physical encoder/cross-attention 获得非零梯度，最终 checkpoint 全部 finite。

裁决：数据和接线 PASS，旧 step accounting FAIL。

### 32.5 BF16 clean smoke

输出：

```text
reconvggt_ar_adapter_a/outputs/pointpose_ss_lora_ssfixed_single_s5_bf16_clean
```

实际 `train_report.json` 记录 `amp_dtype=bf16`、两个 drop probability 都为 0。粘贴命令缺少显式
`--amp_dtype bf16`，与当前默认 fp16 不一致；后续必须保存完整实际命令。

```text
steps 1-3: finite，physical/LoRA 梯度路径正常
step 4: physical gradient NaN，LoRA gradient Inf
step 5: loss/delta/tokens/gradients 全部 NaN
```

BF16 没有 GradScaler，step 4 的非有限梯度被直接写入参数。最终 checkpoint：

```text
trainable tensors = 262
nonfinite trainable tensors = 262
```

该 checkpoint 是 poisoned negative run，禁止 eval/resume。BF16 当前不能视为 FP16 的稳定替代。

### 32.6 当前裁决与训练门槛

```text
数据层面：PASS
模型接线：PASS
数值训练：FAIL，暂不进入八卡 overfit64 s200
```

训练器必须做到：

1. backward 后、clip 前检查所有 trainable gradient finite；
2. nonfinite 时不能调用 `clip_grad_norm_`，避免 Inf 乘零生成 NaN；
3. FP16 根据 scaler 前后值识别 skipped update；
4. skipped/nonfinite update 不增加 `global_step`；
5. BF16/FP32 nonfinite 默认 hard fail；
6. optimizer step 后检查参数 finite；
7. checkpoint 前再次 finite audit，拒绝 poisoned state；
8. 日志记录 applied/skipped、overflow count、AMP dtype 和 scaler。

下一轮先跑 repaired root 上的 FP16 s20：initial scale 16384、drop=0、nonfinite hard fail。只有连续完成
20 个真实 finite optimizer updates、保存 checkpoint nonfinite count=0，才进入 overfit64 s200。

### 32.7 Finite-update guard 实现与验证

`train_pointpose_ss_lora.py` 已增加：完整 loss/condition/gradient finite 检查、FP16 skipped-step
识别、有效更新计数、参数更新后 finite 检查、保存前拒绝 nonfinite checkpoint、AMP/scaler/overflow
日志，以及 `error/skip` 两种策略。另新增 `audit_pointpose_training_run.py`，独立核对更新数、step、
overflow 次数和 checkpoint trainable state。

在 repaired cache、GPU 1、indices 0-1 上完成 FP16 s5 实测：

```text
lr = 1e-5
amp_init_scale = 16384
drop_all_prob = 0
physical_drop_prob = 0

applied optimizer updates = 5 / 5
nonfinite attempts = 0
scaler = 16384 -> 16384
checkpoint trainable tensors = 262
checkpoint nonfinite tensors = 0
independent finite audit = PASS
```

第 1 步只有 zero-init output projection 获得 physical 梯度；第 2 步起 physical encoder 和
cross-attention 梯度均非零，符合 zero-init 预期。该结果证明新数值守卫和 repaired data 的短程训练可用，
但 5 steps 不能替代正式 s20 门槛；下一步仍按命令说明第 33.2/33.3 节运行 s20，再决定是否启动八卡
overfit64 s200。

### 32.8 第二轮运行级审查修复

进一步审查发现：`update_finite=false` 可能来自 forward diagnostic，而不一定来自 GradScaler 检测到的
gradient Inf；此时调用 `scaler.step()` 仍可能真实更新参数。现已修正为任何 nonfinite 分支都绝不调用
optimizer step，并分别记录 `forward_finite` 与 `gradient_finite`。

同时完成：

1. optimizer `exp_avg/exp_avg_sq/step` 等 state 的递归 finite 检查；
2. checkpoint 前同时审计 model、optimizer 和 scaler state；
3. audit 工具检查 `start_step + applied_updates = completed_step = checkpoint_step`；
4. audit 工具检查 history skip 数、update flags、optimizer/scaler finite；
5. eval 从 checkpoint 自动恢复 `amp_dtype`，修复五路评估的 Namespace 缺字段问题；
6. s20/s200 命令启用 `set -euo pipefail`，s20 改为 nonfinite hard fail。

最终代码在 repaired cache 上再次完成 FP16 s2：2/2 更新成功，forward/gradient/model/optimizer/scaler
全部 finite，增强版独立 audit PASS。当前可以进入 clean s20；仍不得跳过 s20 直接启动八卡 s200。

另用该 s2 checkpoint 完成 `1 sample x 1 seed x 1 sampling step` 五路 eval smoke；checkpoint 中的
`amp_dtype=fp16` 被自动恢复，stock/image-only/correct/shuffled/zero 五路均执行并生成 report，退出码为 0。
该 smoke 只验证运行路径，不用于判断两步模型的重建收益。

GPU 0 正式 s20 gate 使用 scale 16384 时，首个 `t=0.9877` batch 出现 gradient NaN；forward 仍 finite，
optimizer step 未执行，`global_step=0`，scaler 正确从 16384 降到 8192，且没有 checkpoint。该结果证明
hard-fail/step accounting 正常，也说明 16384 不能作为跨 GPU 的默认 scale。第 33.2 节已改为从 8192
重新开始；若仍 overflow，再以新目录降到 4096。

### 32.9 正式 33.2 clean s20

GPU 0 当时存在 PID `1935638`、约 4.66 GB 的其他 Python 占用。相同首个 `t=0.9877` batch 在 GPU 0
依次使用 scale 16384、8192、4096、2048、1024，均出现 `forward_finite=true` 但
`gradient_finite=false`；每次 optimizer step 都被正确阻止，global step 保持 0，没有 checkpoint。
因此该序列只能证明 GPU0/并发环境存在异常，不能作为模型普遍不稳定的结论。

随后在完全空闲的 GPU 1 上运行正式配置：

```text
output = pointpose_ss_lora_ssfixed_fp16_guard_s20_gpu1_scale8192
FP16 scale = 8192
lr = 1e-5
indices = 0-7
drop_all/physical_drop = 0
```

结果：

```text
applied updates = 20 / 20
completed/checkpoint step = 20 / 20
nonfinite attempts = 0
history skipped rows = 0
scaler = 8192 -> 8192
flow loss min/mean/max = 0.03980 / 0.23826 / 0.73400
delta RMS step20 = 0.04483
max pre-clip total grad norm = 1.36950
t range = [0.09666, 0.98772]
model/optimizer/scaler finite = PASS
independent audit = PASS
```

stock bridge、scale0 和 zero-init scale1 审计均通过。结论是训练器和模型配置可以 clean 完成 s20，
但暂不直接进入八卡 s200：GPU 0、GPU 7 当前有其他进程，且 GPU 0 的 backward 行为与 GPU 1 不一致。
下一步先在所有计划参与 DDP 的 GPU 上各跑同配置 1-step finite smoke；只有所有 rank 都通过且 0-7
全部空闲，才运行 33.4。否则八卡 hard-fail 会被单张异常 GPU 终止。

### 32.10 两卡 overfit64 s200（2026-07-13）

使用 GPU 1/2、每卡 batch 1、grad accumulation 16，保持 effective batch=32，完成两卡 overfit64：

```text
output = pointpose_ss_lora_ssfixed_overfit64_s200_2gpu_fp16
world size = 2
dataset / unique objects = 64 / 64
updates = 200
lr = 1e-5
FP16 scale = 8192
```

增强版审计全部通过：

```text
applied/completed/checkpoint step = 200/200/200
nonfinite attempts = 0
skipped history rows = 0
model trainable tensors = 262, nonfinite = 0
optimizer state finite = true
scaler state finite = true
audit = PASS
```

训练诊断：

```text
flow loss logged mean:
  step 1-50   = 0.2432
  step 51-100 = 0.1794
  step 101-150= 0.1565
  step 151-200= 0.2237

physical delta RMS:
  step 1   = 0
  step 50  = 0.1409
  step 100 = 0.1895
  step 150 = 0.1841
  step 200 = 0.2065
```

loss 在前 150 步总体下降，physical encoder/cross-attention/output projection 与 flow LoRA 均保持非零
梯度；末段均值回升受随机 t、对象和仅每 5 step 记录一次影响，不能单独解释成发散。更值得注意的是
condition delta 已增长到 0.2065，说明 physical branch 确实改变 condition，但也存在过拟合或过强扰动风险。

当前裁决：数值稳定、DDP、梯度路径和可优化性 PASS；物理条件有效性尚未 PASS。不能仅凭 flow loss 进入
长训，因为 LoRA 可能只是在记忆 image/latent，delta 变大也不等于正确 point/pose 有用。下一步必须先对
s200 做五路 pure-noise 评估：stock、image-only、correct、shuffled、zero。只有 correct 在多数对象和多个
seed 上同时优于另外四路、且 component/coord count 不恶化，才允许从 stock 初始化在 full train 上启动
两卡 s500 pilot。overfit64 checkpoint 不直接续训 full dataset；它只用于机制裁决。

### 32.11 五路 eval condition 复用修复

初版 s200 eval 在首个样本报：

```text
native stock condition differs from physical_scale=0 image condition
```

原因与早期训练审计相同：`pipeline.get_ss_cond()` 和 `model.build_condition()` 分别重算了一次 FP16
bridge，再用 `torch.equal` 比较，测到的是低精度重算差异；同时这会让五路 comparison 混入不同 image
condition。现已改为每个样本只计算一次 native condition，image-only/correct/shuffled/zero 全部从同一个
`cond_base` 进入 physical branch。scale0 仍要求 bit-exact 等于 native。

新增 `run_eval_overfit64_s200_2gpu.sh`，避免复制长命令导致参数串接。真实 s200 checkpoint 的
`1 sample x 1 seed x 1 step` smoke 已成功完成，退出码 0：

```text
threshold IoU:
  stock      = 0.02272
  image-only = 0.02286
  correct    = 0.07587
  shuffled   = 0.07682
  zero       = 0.02388
```

correct 明显优于 stock/image-only/zero，说明 learned physical branch 对 rollout 有实质影响；但该样本中
correct 略差于 shuffled，尚不能证明模型使用了对象对齐的 point/pose。下一步必须运行完整 64 objects、
seeds 42/43/44、30 steps 五路评估，以 `correct-shuffled` 的对象级 win rate 作为核心裁决。

### 32.12 overfit64 s200 完整五路评估（2026-07-13）

修复 condition 单次复用后，完整运行 `64 objects x 3 seeds x 30 steps x 5 sources`，共 192 个配对样本：

```text
report = outputs/pointpose_ss_lora_ssfixed_overfit64_s200_2gpu_fp16/
         eval_overfit64_noise_t1_seeds424344_v2/report.json

stock rollout equivalence:
  latent max abs diff                    = 0
  threshold coord set equal              = true
  pipeline-native threshold set equal    = true
  oracle-count coord set equal           = true
```

这证明评估的 stock 路径与原生 ReconViaGen 路径严格等价，五路差异不是 noise 或重复 bridge forward 引入的。

#### 32.12.1 主指标

native threshold 坐标的对象均衡结果如下。当前每个对象恰好三个 seed，因此 sequence-weighted 与
object-balanced 均值相同：

| source | IoU | recall | precision | coord ratio | components | largest comp. ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stock native | 0.04675 | 0.10204 | 0.08305 | 1.368 | 2.64 | 0.9824 |
| LoRA image-only | 0.04582 | 0.09772 | 0.08304 | 1.319 | 3.28 | 0.9822 |
| PointPose correct | 0.04520 | 0.10306 | 0.07952 | 1.481 | 5.65 | 0.9637 |
| PointPose shuffled | 0.04478 | 0.10238 | 0.07900 | 1.469 | 5.86 | 0.9703 |
| PointPose zero | 0.04652 | 0.09784 | 0.08483 | 1.282 | 3.08 | 0.9846 |

correct 相对各对照的配对 threshold IoU：

| comparison | mean delta | median delta | win rate | degradation rate |
| --- | ---: | ---: | ---: | ---: |
| correct - stock | -0.00155 | -0.00254 | 41.15% | 57.81% |
| correct - image-only | -0.00062 | -0.00147 | 42.19% | 56.25% |
| correct - shuffled | +0.00042 | -0.000009 | 45.31% | 52.08% |
| correct - zero | -0.00132 | -0.00139 | 41.67% | 56.25% |

按对象先聚合三个 seed 后，correct 相对 shuffled 的平均 delta 仍为 `+0.00042`，但只有 50% 对象的
三-seed均值为正，且仅 39.06% 对象达到三个 seed 至少两胜。该微弱均值优势来自少数样本，不能作为
对象对齐 point/pose 有效的证据。

seed 稳定性也未通过：correct-shuffled 的 threshold IoU 均值 delta 在 seed 42/43/44 分别为
`+0.000068/+0.000109/+0.001074`，对应胜率仅 `42.2%/43.8%/50.0%`。方向均值虽为正，但多数样本并未受益。

#### 32.12.2 结构退化和 oracle-count 诊断

correct 相比 stock 平均多生成 529 个坐标，coord ratio 增加 0.112，component count 增加 3.01，precision
下降 0.00354，largest-component ratio 下降 0.0186。recall 仅增加 0.00103。主要行为是扩大并碎片化 sparse
occupancy，而不是更准确地恢复目标结构。

固定预测数量的 `topk_target_oracle_count` 也未扭转结论：

```text
correct - stock:     mean IoU delta = -0.00098, win rate = 47.40%
correct - image-only:mean IoU delta = -0.00085, win rate = 45.31%
correct - shuffled:  mean IoU delta = +0.00023, win rate = 50.52%
correct - zero:      mean IoU delta = -0.00254, win rate = 40.62%
```

因此失败不只是 threshold calibration；在 oracle-count 下，对齐物理场仍未稳定优于 shuffled。

#### 32.12.3 zero-input 控制暴露的架构问题

评估中的 condition 统计为：

```text
correct physical delta RMS: mean 0.1700, range [0.1528, 0.1908]
zero physical delta RMS:    mean 0.0991, range [0.0988, 0.0995]
```

`pointpose_zero` 并不等价于“无 physical residual”。当前 grid encoder 含 bias，zero physical grid 仍可生成
非零 tokens；cross-attention 又以 `cond_base` 为 query，最终产生非零 delta。`image-only` 的实际 applied
scale 为 0，且已 bit-exact 等于 native condition；report 中的 `image_only_delta_rms` 是 scale 前的内部预测
delta，不能解释成已施加的 condition drift。

这说明当前 physical branch 同时学到了一个接近 object-independent 的通用 condition residual。correct 与
shuffled 高度接近，与该机制一致。后续必须让物理分支满足结构性的 zero-input invariance，例如使用
`delta = output(cond, physical) - output(cond, zero_physical)`，或以 occupancy/support gate 显式乘在 delta 上；
不能只靠训练期 zero 样本期待网络自行学会零响应。

#### 32.12.4 裁决与下一步

本轮结论是：

```text
数值稳定性、DDP、LoRA/physical 梯度路径、可优化性：PASS
stock rollout 等价性：PASS
对齐 PointPose 的对象特异收益：FAIL
结构质量与 precision preservation：FAIL
```

因此现在不进入 full-train 长训，也不从 overfit64 s200 checkpoint 续训。增加数据或训练步数会放大当前
“通用 residual + occupancy 膨胀”，无法解决 correct 与 shuffled 不可分的问题。

下一步按以下顺序进行：

1. 用同一完整五路协议评估已保存的 step100 和 step150，确认 s200 是否只是后期 residual 过强；这是低成本
   checkpoint selection，不新增训练变量。若两者仍不能稳定满足 `correct > shuffled` 且优于 stock，则停止当前 v2。
2. 实现 zero-input-invariant physical residual，并在训练 loss 中加入 correct/shuffled 配对辨别约束；仅 flow-matching
   loss 不会强迫网络使用对象对应关系。
3. 在 overfit64 上重新跑短程机制实验，准入条件至少包括：correct-shuffled 对象三-seed多数胜率 > 60%，
   correct-stock win rate > 55%，mean/median delta 同为正，coord ratio 增量受控，component count 不显著增加。
4. 只有上述机制门槛通过，才从 stock 权重 fresh start，在 full train 上做两卡 s500 pilot，并使用 val 做 checkpoint
   selection；holdout 保持未触碰。若门槛仍失败，停止 SS-condition/SS-flow PointPose LoRA 路线，转向直接 sparse
   occupancy/coordinate supervision 或生成后的几何约束模块。

### 34.9 J1a stock-preserving bridge-last1 实验结果（34.3--34.7，2026-07-13）

本轮训练的可训练范围为：

```text
frozen VGGT + frozen DINO
  -> frozen ReconViaGen bridge prefix
  -> zero-centered physical encoder + physical cross-attention adapter
  -> frozen bridge last block
  -> frozen stock SS Flow
```

原 ReconViaGen bridge 和 SS Flow 参数均不更新，只训练约 `7.91M` 的 physical bridge adapter。此阶段未开启
Flow LoRA、occupancy loss 和 SLAT 训练。

#### 34.9.1 执行和数值稳定性

| 阶段 | 结果 | 核心证据 |
| --- | --- | --- |
| 34.3 单卡 s5 | PASS | 5/5 updates，nonfinite=0，model/optimizer/scaler finite |
| 34.4--34.5 两卡 s100 | PASS | 100/100 updates，nonfinite=0，effective batch=32 |
| 34.6 overfit64 teacher-forced | 自动判定 PASS | correct 稳定优于 stock，但 correct/shuffled 效应量极小 |
| 34.7 validation teacher-forced | FAIL | correct 未稳定优于 shuffled |

s100 数值审计全部通过：

```text
applied/completed/checkpoint updates = 100/100/100
nonfinite attempts                  = 0
optimizer state finite              = true
scaler state finite                 = true
history update accounting           = true
```

stock-preserving 硬路由在 overfit 和 validation 上均为：

```text
condition disabled max abs diff     = 0
velocity disabled max abs diff      = 0
stock vs split condition max diff   = 0
condition exact                     = true
velocity exact                      = true
Flow LoRA enabled                   = false
```

因此，数值稳定性、bridge 拆分、frozen-flow 梯度和无 physical 先验时的 stock fallback 都已经 PASS。

#### 34.9.2 训练轨迹暴露的问题

s100 的 condition delta 持续增长：

```text
step 1:   delta/stock RMS ratio = 0
step 20:  delta/stock RMS ratio = 0.02225
step 50:  delta/stock RMS ratio = 0.05363
step 100: delta/stock RMS ratio = 0.09876
```

但 alignment 没有学会区分 correct 和 shuffled：

```text
step 1:
  positive probability = 0.500000
  negative probability = 0.500000
  alignment loss       = 0.693147

step 100:
  positive probability = 0.520996
  negative probability = 0.520996
  alignment loss       = 0.694028
```

两路 gate 同时上升，alignment loss 甚至略差于随机分类的 `log(2)`。这说明 adapter 在学习一个对所有
physical-present 样本都启用的 condition 校正，而不是学习“这个 physical 是否与当前图像/VGGT 对齐”。

#### 34.9.3 overfit64：表面 PASS，对象特异性证据很弱

`64 objects x 3 seeds x 5 t = 960` 条配对记录：

```text
mean flow MSE:
  stock    = 0.19614002
  correct  = 0.19290612
  shuffled = 0.19290718

stock - correct mean               = 0.00323389
stock - correct relative reduction = 1.65%
shuffled - correct mean            = 0.00000105
```

对象均衡结果：

```text
stock - correct:
  mean/median = +0.00323389 / +0.00182267
  object win  = 100.00%

shuffled - correct:
  mean/median = +0.00000105 / +0.00000077
  object win  = 60.94%
  positive t  = 4/5
```

自动 PASS 的 `60.94%` 只比阈值 `60%` 高一个对象左右，而 correct/shuffled 差值只是 stock 改善量的约
`0.033%`。这个量级远小于行级波动：

```text
shuffled - correct row p25/p75 = -5.51e-6 / +6.72e-6
row positive rate             = 53.75%
```

同时 correct 和 shuffled 的 alignment probability 分别为 `0.521834` 和 `0.521799`，平均 gate gap 仅
`+3.44e-5`。所以 34.6 只能说明“该 adapter 学到了能降低 overfit flow MSE 的通用校正”，不能证明
point/pose 对象对齐信息被有效使用。

#### 34.9.4 validation：通用校正泛化，physical specificity 失败

validation 包含 `175 sequences / 128 objects`，共 `2625` 条 teacher-forced 记录：

```text
mean flow MSE:
  stock    = 0.22357467
  correct  = 0.21989360
  shuffled = 0.21989382

stock - correct row mean            = +0.00368108
stock - correct relative reduction  = 1.65%
```

correct 相对 stock 的改善在 validation 上仍然存在：

```text
object-balanced stock - correct:
  mean/median = +0.00347429 / +0.00165648
  object win  = 98.44%
  positive t  = 5/5
```

这说明 adapter 学到的通用 bridge/domain 校正不是单纯记忆 64 个对象。但它对 physical 对齐关系不敏感：

```text
object-balanced shuffled - correct:
  mean       = +0.000000059
  median     = -0.000000339
  object win = 42.97%
  positive t = 2/5

per-t shuffled - correct mean:
  t=0.1  -0.00000009
  t=0.3  +0.00000070
  t=0.5  -0.00000051
  t=0.7  +0.00000103
  t=0.9  approximately 0, negative
```

validation 上 correct/shuffled gate 分别为 `0.521757` 和 `0.521762`，差值变为错误方向的 `-5.33e-6`。
condition delta RMS 也几乎不随样本改变：

```text
mean delta RMS       = 0.15918
delta RMS range      = [0.15621, 0.16116]
mean delta/stock RMS = 9.91%
```

这些数据共同支持：

> J1a 学到了一个 physical-present 触发的、近似 object-independent 的 bridge 校正；该校正能降低
> teacher-forced flow MSE，但改善不来自正确 point/pose 与当前图像的对齐关系。

#### 34.9.5 原因分析

1. 训练的 flow loss 只使用 correct physical，没有要求 correct 必须比 shuffled 更好。一个对所有 physical-present
   样本都生效的常量/domain residual 就能降低主 loss。
2. 现有 alignment head 对 bridge tokens 和 physical tokens 分别做全局平均后再分类，缺少 token-level 空间对齐交互；
   相似占据率和 point count 的 hard negative 在全局统计上本来就很接近。
3. alignment 权重只有 `0.05` 且前 20 步 warmup，对约 `9.9%` 的 condition residual 缺少实质约束。
4. zero-centered physical encoder 只保证 `E(grid)-E(0)=0`。cross-attention 和 output projection 仍含可训练 bias，
   physical-present 分支仍可学到与对象无关的常量 residual。hard-off stock 路由虽然保证了回退能力，却不会
   自动禁止这种 present-only 常量校正。

#### 34.9.6 裁决

```text
数值稳定性与 DDP:                  PASS
stock-preserving hard fallback:        PASS
bridge 内 physical adapter 可优化性:     PASS
通用 teacher-forced domain correction:      PASS
correct-vs-shuffled 对象特异性:         FAIL
validation 上的物理先验收益:             FAIL
```

因此当前不进入 bridge-last2，不实现 J1b Flow LoRA，不在 full train 上长训。Flow LoRA 会提高表达能力，但在
correct/shuffled 未分离时，更可能放大通用 residual，而不是自动产生 physical specificity。

该 checkpoint 可以保留为“stock-preserving generic bridge calibration”对照，但不能作为 PointPose 有效的论文证据。

#### 34.9.7 下一步建议：J1a.1 specificity-first

第一步先增加不训练的 sensitivity audit，对同一 `x_t/t/noise` 报告：

```text
cond(correct) - cond(stock)
cond(shuffled) - cond(stock)
cond(correct) - cond(shuffled)
cond(zero-present) - cond(stock)
v(correct) - v(shuffled)
```

并新增效应量：

```text
physical_specificity_ratio =
  abs(shuffled_loss - correct_loss)
  / (abs(stock_loss - correct_loss) + eps)
```

当前 overfit 该比率约为 `0.033%`，validation 对象均衡比率约为 `0.0017%`，应视为无物理敏感性。

第二步修正 adapter 结构：

```text
physical residual = R(bridge, physical) - R(bridge, zero_physical)
```

这个 residual-level zero centering 应在 cross-attention 和 output projection 之后完成，从结构上抵消 attention/projection bias 导致的
present-only 常量修正。alignment score 改为使用 token-level 交互，例如 bridge query 与 physical-attended tokens 的
bilinear/product/difference 特征，不再只拼接两个全局平均向量。

第三步使用安全的 paired objective，对 correct/shuffled 共用同一 `target/x_t/t/noise`：

```text
L = L_fm_correct
  + lambda_gate * BCE(correct=1, shuffled=0)
  + lambda_shuffled_stock * MSE(v_shuffled, stopgrad(v_stock))
  + lambda_gain * relu(margin + L_correct - stopgrad(L_stock))
  + lambda_delta * relative_delta_norm
```

`shuffled_stock` 保持项很重要：它要求错误 physical 回到 stock，避免普通 ranking loss 通过故意破坏 shuffled 来作弊。

第四步只在 overfit16/64 做 50--100 updates 机制验证，新的硬门槛应至少包括：

```text
stock hard route max diff = 0
correct-shuffled object win rate >= 65%
correct-shuffled mean and median > 0
correct-shuffled positive t >= 4/5
alignment p(correct)-p(shuffled) > 0 with a nontrivial margin
physical_specificity_ratio >= 5%
correct-stock mean/median > 0
```

只有 overfit 和 validation 都通过 specificity 门槛，才进入 J1b physical-gated Flow LoRA。如果 residual-level zero centering
和 paired objective 仍无法让 correct 稳定优于 shuffled，就停止 bridge/condition 注入路线，转向直接 sparse occupancy/
coordinate supervision 或生成后可解释的几何约束。


# J1a.1 local16 空间映射审计、代码修改与下一步

日期：2026-07-13

## 1. 本轮目的

本轮不是继续扩大训练，而是审查以下核心假设：

```text
ReconViaGen bridge condition token 数量为 4096
因此 token index 可以直接解释为 16^3 voxel index
```

`4096 = 16^3` 只能证明 token 数量一致，不能证明：

```text
token 0   <-> (0,0,0)
token 1   <-> (0,0,1)
token 16  <-> (0,1,0)
token 256 <-> (1,0,0)
```

因此在运行 overfit16 前，新增真实 ReconViaGen bridge + frozen SS Flow 空间映射审计，并同步修复 null physical、训练 loss、评估硬门和逐 stage 诊断。

## 2. 修改的代码

### 2.1 `stock_preserving_pointpose_bridge.py`

主要修改：

1. `multistage_local16` checkpoint 语义升级为 `v2`，拒绝把旧全零 null checkpoint 当作新模型加载。
2. 新增 `make_null_physical_grid()`：

```text
channels 0--10：point / visibility / mask / depth 证据归零
channels 11--13：固定 XYZ 坐标保持不变
```

3. `ZeroCenteredPhysicalGridEncoder16` 改为：

```text
E(evidence + xyz) - E(zero_evidence + xyz)
```

因此没有对象证据但保留标准 XYZ 时，physical token 严格为零。

4. 明确实际注入顺序仍是 after-block：

```text
frozen bridge block i
-> physical adapter i
```

5. 增加每个 stage 的：

```text
alignment probability
effective gate
delta RMS / abs max
hidden RMS
effective delta / hidden ratio
```

6. 推荐 stage 改为 `0,1,2`，避免 block 3 后再增加一个没有后续 frozen bridge block 处理的裸 post-bridge residual。

对应位置：

```text
make_null_physical_grid                 lines 18--30
ZeroCenteredPhysicalGridEncoder16       lines 92--132
MultiStage condition_paths              lines 600--711
per-stage diagnostics                   lines 662--670
metadata / null semantics               lines 733--748
```

### 2.2 `train_stock_preserving_pointpose_bridge.py`

主要修改：

1. 架构启动审计使用 preserved-XYZ null，检查：

```text
hard stock route exact
null physical condition exact
null physical token max abs = 0
zero-init fused condition = stock
Flow LoRA disabled
```

2. `shuffled-stock` loss 改为相对尺度：

```text
MSE(v_shuffled, v_stock) / mean(v_stock^2)
```

3. correct-vs-stock gain 改为相对改善：

```text
relative_gain = (L_stock - L_correct) / L_stock
L_gain = relu(gain_margin - relative_gain)
```

默认 `gain_margin=0.005`，表示要求至少 0.5% 相对改善。

4. 日志增加 raw/relative shuffled loss、relative correct gain 和逐 stage correct/shuffled 指标。
5. resume 时检查 `multistage_local16 v2`，拒绝旧 `v1`。

对应位置：

```text
preserved-XYZ architecture audit        lines 350--440
relative paired losses                  lines 725--766
finite diagnostics                      lines 790--809
training report fields                  lines 840--900
```

### 2.3 `eval_stock_preserving_pointpose_teacher_forced.py`

主要修改：

1. 评估 null route 时保留 XYZ，并要求 condition bit-exact stock。
2. 同时报告：

```text
clipped physical specificity ratio
signed physical specificity ratio
```

signed ratio 不会把错误方向裁剪为零。

3. 报告每个 stage 的 correct/shuffled gate、gate gap 和 delta/hidden ratio。
4. 新增 `--fail_on_decision`：报告写完后，decision FAIL 返回退出码 2。
5. eval 加载时同样拒绝旧 `multistage_local16 v1` checkpoint。

### 2.4 `audit_reconbridge_condition_spatial_mapping.py`

新增独立真实模型审计，包含：

1. 同输入重复前向，检查数值可重复性。
2. condition token 随机置换审计。
3. token `0,1,16,256` sentinel 单点扰动。
4. 对每个 sentinel 计算：

```text
预期 xyz
Flow 响应 argmax xyz / 距离
预期位置响应 percentile
响应能量质心 / 距离
半径 1、半径 2 局部能量比例
```

5. 使用 `FP32 + ATTN_BACKEND=sdpa`，避免 FlashAttention 只支持 FP16/BF16 以及低精度归约顺序误差。
6. 新增 `--fail_on_audit`，FAIL 返回退出码 2。

### 2.5 测试和命令

`test_stock_preserving_pointpose_bridge.py` 扩展为 10 个测试，新增：

```text
preserved-XYZ null -> zero physical token
adapter 更新后 null condition仍 bit-exact stock
after-block 注入调用顺序
specificity FAIL exit code = 2
```

`命令说明.txt` 第 35 节已更新：

```text
35.1：静态检查和 10 个单测
35.2：真实 FP32/SDPA 空间映射硬审计
35.3--35.9：仅在空间映射审计 PASS 时允许执行
```

训练参数统一改为：

```text
fusion_stages = 0,1,2
gain_margin = 0.005（相对改善）
eval --fail_on_decision
```

## 3. 真实空间映射审计结果

审计对象：

```text
sample uid = 1d9e94391e524624bc79f00a494e70a5_seq001
condition shape = [1,4096,1024]
SS Flow output shape = [1,8,16,16,16]
checkpoint Flow dtype = FP16
audit Flow dtype = FP32
attention backend = SDPA
t = 0.5
perturb scale = 0.05
```

### 3.1 数值控制

同一 `x_t / t / cond_stock` 重复运行：

```text
max abs diff = 0
RMSE = 0
bit-exact = true
```

因此 sentinel 的空间差异不是重复前向随机噪声。

### 3.2 Condition permutation

随机置换 4096 个 condition rows：

```text
max abs velocity diff = 8.702278e-6
RMSE = 1.256246e-6
permutation RMSE / median sentinel RMSE = 0.044903
```

试运行报告使用了较松阈值，曾记录 `row_order_sensitive=true`。最终代码和正式命令使用：

```text
max abs diff >= 1e-5
response ratio >= 0.1
```

当前结果两项都不满足，因此正式判定为：

```text
condition row-order sensitivity NOT VALIDATED
```

注意：cross-attention 对 KV row permutation 在数学上可以不变，因此 permutation 结果本身不是空间语义的充分证据；下面的 sentinel localization 才是直接局部性测试。

### 3.3 Sentinel localization

| token | 假设 xyz | Flow 响应 argmax xyz | argmax 距离 | 响应质心距离 | 预期位置 percentile | 半径1能量 |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 0 | [0,0,0] | [14,0,15] | 20.5183 | 12.6608 | 0.9695 | 0.7911% |
| 1 | [0,0,1] | [12,15,9] | 20.8087 | 13.5026 | 0.1204 | 0.4961% |
| 16 | [0,1,0] | [9,9,2] | 12.2066 | 10.5179 | 0.9951 | 1.4166% |
| 256 | [1,0,0] | [15,10,1] | 17.2337 | 12.2908 | 0.9160 | 1.0529% |

四个 sentinel 均未满足：

```text
argmax distance <= 2
expected percentile >= 0.95
```

其中 token 0 和 16 虽然预期位置 percentile 较高，但全局 argmax 和响应质心距离很远，且局部能量比例很低，不能解释为空间局部响应。

## 4. 最终结论

```text
4096 condition tokens != 已验证的 16^3 voxel-token 对应关系
```

当前 `multistage_local16` 的核心前提没有成立。继续 overfit16 即使得到训练 loss 下降，也不能证明物理先验被注入到正确空间位置，可能只是：

```text
通用 condition calibration
token-content sensitivity
全局 domain correction
```

因此正式裁决为：

```text
停止 J1a.1 multistage_local16 的 s5 / overfit16 / overfit64
不启用该路线的 Flow LoRA
不通过增加步数尝试挽救错误的 index-alignment 假设
```

preserved-XYZ null、相对 loss、逐 stage 诊断和硬退出仍是有效工程改进，可复用于下一版模型。

## 5. 下一步建议

下一步进入 `J1a.2 content-based physical-to-visual fusion`，取消：

```text
bridge_token[i] <-> physical_voxel[i]
```

推荐结构：

```text
physical tokens p = E(evidence + xyz) - E(null evidence + xyz)

每个选定 bridge stage i：
  visual context c_i = concat(VGGT_i, image features)
  delta_context_i = CrossAttn(Q=c_i, K=p, V=p)
                  - CrossAttn(Q=c_i, K=0, V=0)
  fused_context_i = c_i + gate_i * alpha_i * delta_context_i
  hidden_i = frozen_bridge_block_i(hidden_{i-1}, fused_context_i)
```

这样 Point/Pose 通过内容匹配进入与 VGGT/Image 相同的 bridge block，同时：

```text
不依赖 condition row 与 voxel index 对齐
null physical严格退化到原 visual context
stock bridge权重保持冻结
physical关闭时走硬 stock route
```

为控制 3090 显存，第一版建议：

```text
physical token grid：8^3 = 512
bridge stages：先只做 0,1
physical cross-attention hidden dim：128
Flow：继续完全冻结
训练对象：physical encoder + context fusion adapters + alignment gates
```

训练顺序：

1. 新增结构单测和真实 null/stock 等价审计。
2. 做未训练 correct-vs-shuffled context delta 诊断，确认正确与 shuffled physical 会产生不同匹配响应。
3. 单卡 s5 数值 smoke。
4. overfit16 teacher-forced paired training。
5. 要求 correct 同时优于 stock 和 shuffled，且多个 t、多个 seed、对象级指标一致。
6. 只有 J1a.2 overfit16 通过，才进入 overfit64；只有 overfit64 通过，才讨论 physical-gated Flow LoRA。

J1a.2 的硬准入条件：

```text
physical-off condition/velocity bit-exact stock
null-evidence condition bit-exact stock
correct flow MSE < stock flow MSE
correct flow MSE < shuffled flow MSE
correct-vs-shuffled object win rate >= 65%
至少 4/5 t 区间方向正确
alignment probability correct > shuffled
condition delta幅度受控且无末层裸 residual
```

如果 content-based J1a.2 仍无法在 overfit16 建立 correct-vs-shuffled 优势，应停止 bridge condition 注入路线，转向显式 3D sparse occupancy conditioner 或在 SS Flow 内增加独立的 physical cross-attention，而不是继续修改 `get_ss_cond` 输出。

## 6. 验证状态

```text
Python py_compile：PASS
结构单测：10/10 PASS
真实 FP32/SDPA mapping audit：FAIL
multistage_local16 overfit gate：CLOSED
```

---

# J1a.2-A 36.4/36.5 结果与下一步（2026-07-13）

## 1. 完成状态

36.4 和 36.5 均已完整运行，没有残留训练或评估进程。

- 36.4：seed 42/43/44 的 s50 训练、checkpoint 和 finite audit 全部生成。
- 36.5：三个 checkpoint 的 paired teacher-forced eval 全部生成。
- 多 seed 汇总已生成：`pointpose_j1a2a_content_visual8_overfit16_s50_multiseed_summary/report.json`。
- 36.5 最终判定为 `FAIL`，不是中途被 OOM 或系统杀死。

终端退出的直接原因是最后的汇总命令使用了 `--fail_on_decision`。多 seed 判定失败后，脚本按设计返回退出码 2；外层又使用 `set -euo pipefail`，因此执行 shell随即退出。该行为属于实验门控退出，不代表输出不完整。

## 2. 36.4 训练健康分析

三个 seed 的工程健康均为 `PASS`：

| seed | optimizer updates | nonfinite | 所有梯度组非零且 finite | frozen/architecture audit |
| ---: | ---: | ---: | --- | --- |
| 42 | 50 | 0 | PASS | PASS |
| 43 | 50 | 0 | PASS | PASS |
| 44 | 50 | 0 | PASS | PASS |

这证明以下链路有效：physical encoder、visual query projection、physical cross-attention、output projection 和 alignment gate 都能收到梯度；stock bridge 和 SS Flow 保持冻结；BF16 两卡训练数值稳定。

但训练日志已经显示 seed 敏感性：

| seed | stock-correct last3 | shuffled-correct last3 | final condition delta/stock | final gate gap |
| ---: | ---: | ---: | ---: | ---: |
| 42 | -0.00006297 | -0.00002375 | 0.035212 | 0.002930 |
| 43 | 0.000404 | 0.00004387 | 0.028034 | 0.006836 |
| 44 | 0.000242 | -0.00001688 | 0.017363 | 0.001953 |

训练末期逐 stage 的 `context delta` correct/shuffled cosine 仍很高：seed 42/43 约为 0.998，seed 44 为 0.980--0.990。这说明 content attention 已能感知输入差异，但最终注入 bridge 的修正方向仍有明显的 object-independent 共性。

## 3. 36.5 固定噪声、多 t 评估

所有 seed 均保持严格 stock 回退：physical off 和 null evidence（保留 XYZ）时，condition 与 velocity 的最大绝对差均为 0；Flow LoRA 未启用。

| train seed | decision | stock-correct mean | shuffled-correct mean | shuffled object win | specificity | alignment gap |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | FAIL | 0.00034244 | 0.00000742 | 56.25% | 2.17% | 0.011280 |
| 43 | PASS | 0.00024253 | 0.00001697 | 68.75% | 7.00% | 0.015068 |
| 44 | PASS | 0.00019955 | 0.00001776 | 81.25% | 8.90% | 0.014740 |

seed 42 失败的两个明确条件是：

- correct-vs-shuffled object win rate 为 56.25%，低于 65%。
- physical specificity ratio 为 2.17%，低于 5%。

seed 43/44 通过全部门槛。三个 seed 的 correct 都在均值上优于 stock，且 seed 43/44 在五个 t 上均保持 correct 优于 shuffled。说明 content-based fusion 不是完全无效：它首次在部分独立初始化中建立了可测的 physical specificity。但这种 specificity 尚不稳定，不能视为已经学会可靠的 Point/Pose 与视觉对应。

另一个重要现象是 `t=0.9` 时三个 seed 的 stock-correct 均为负，而较低 t 多数为正。这表明当前修正主要在中低噪声区有益，对高噪声阶段的全局生成方向还不稳。

## 4. 是否进入 36.6

**不进入 36.6。**

36.6 的预注册前提是 36.5 三个训练 seed 全部通过同一硬门；实际只有 2/3 通过。此时扩大到 overfit64 会同时引入对象数和优化稳定性变量，无法判断结果来自架构改善还是 seed/数据平均效应。

也不建议通过补跑 seed 42、选择更好 checkpoint 或增加 J1a.2-A 步数来绕过门槛。这会把 validation 式筛选泄漏到机制判断中。

## 5. 下一步建议

按既定决策树进入 **J1a.2-B1 pose-guided projected patch features**：

1. 使用每个 view 的 K/T，把 point confidence、depth、mask support、visibility 和 outside evidence 投影到与 VGGT/DINO 相同的 patch 索引。
2. 保留 view identity、patch `(u,v)` 和逐 view depth，不再依赖 global content attention 自行发现 2D-3D 对应。
3. 继续冻结 VGGT、DINO、stock bridge、SS Flow、decoder 和 SLAT；先不启用 Flow LoRA。
4. 保留 physical-off 与 null-evidence 的 bit-exact stock 回退。
5. 先做投影 sentinel、view 顺序、patch flatten、K/T 坐标系和 mask round-trip audit，再运行 s5 与 overfit16 三 seed。
6. 使用与 36.5 完全相同的 fixed-noise、multi-t 和 object-balanced 门槛，保证 J1a.2-A/B1 可直接比较。

J1a.2-A 的结论应记录为：**机制有正信号，但跨 seed 不稳定，因此停止扩规模；转向显式 pose-guided 2D-3D 对齐。**

## 6. 后续命令安全约定

- 科学判定 FAIL 只写入 `report.json/report.md` 并打印，不再让交互式命令返回退出码 2。
- 真正的 Python、CUDA、DDP、文件或 finite audit 错误继续由 `set -euo pipefail` 硬失败。
- 长时间训练先进入持久会话：`tmux new-session -A -s pointpose`；使用 `Ctrl-b d` 分离，`tmux attach -t pointpose` 恢复。
- 每个阶段保留独立 `.log`、`report.json` 和 checkpoint audit；终端断开后以文件和进程状态判断完成情况。

---

# J1a.2-B1 37.2--37.5 结果与下一步（2026-07-13）

## 1. 完成状态与工程结论

37.2--37.5 已全部运行完毕。所有训练、审计、单 seed 评估和多 seed 汇总进程的
`.exit_code` 均为 `0`；输出、checkpoint 和报告完整。

- 37.2 真实 K/T 投影审计：`PASS`。
- 37.3 BF16 s5：5/5 optimizer updates，nonfinite=0，finite audit=`PASS`。
- 37.4 seed 42/43/44：均完成 50/50 optimizer updates，nonfinite=0，finite audit=`PASS`。
- 37.5 三个单 seed teacher-forced eval 均完整，但科学 decision 均为 `FAIL`。
- 跨 seed 汇总：预期和实际 seed 均为 `[42,43,44]`，stock equivalence 全部通过，最终
  `passed=false`。

三个 s50 训练中，以下五组参数在全部 11 个记录点均获得 finite/nonzero 梯度：

```text
projected_patch_encoder
visual_query_projection
projected_patch_interaction
physical_output_projection
alignment_gate
```

因此这次失败不是 CUDA、DDP、精度、梯度断路或 checkpoint 污染问题，而是明确的模型效果问题。

## 2. 37.2 投影与严格回退审计

真实样本审计结果：

```text
views = 4
patch grid = 37 x 37
visual/physical feature shape = [1, 5476, 20]
correct/shuffled pose-ray-UV = bit-exact
correct/shuffled point evidence = different
null evidence = 0
null pose-ray-UV = preserved
physical-off/null condition = bit-exact stock
```

审计样本每个 view 只有 37 个可投影 correct point，对应 23--34 个 occupied patch；相对于每个
view 的 1369 个 patch，直接点证据只覆盖约 1.68%--2.48%。shuffled 样本每个 view 为 42 个点、
28--35 个 occupied patch。该统计只来自审计样本，不能代表全数据分布，但证明当前 B1 的对象特异
点证据非常稀疏。

未训练 sensitivity 也通过：correct/shuffled physical-token RMS difference 为 `0.023576`，stage
0/1 的 internal response RMS difference 为 `0.048091/0.052260`。所以投影特征和局部 adapter
确实能区分两种输入；最终失败不能归因于 correct/shuffled 输入完全相同。

## 3. 37.5 固定噪声、多 t 结果

所有 seed 的 physical-off 和 null-present condition/velocity 最大绝对差均为 0，Flow LoRA 未启用。

| train seed | stock-correct mean | stock object win | shuffled-correct mean | shuffled median | shuffled object win | specificity | alignment gap | positive t: stock/shuffled |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.00002700 | 81.25% | 0.00001072 | -0.00000217 | 50.00% | 39.71% | 0.001331 | 4/4 |
| 43 | 0.00003652 | 81.25% | 0.00001003 | 0.00000271 | 56.25% | 27.48% | 0.000943 | 5/3 |
| 44 | 0.00003052 | 75.00% | 0.00000288 | 0.00000018 | 50.00% | 9.44% | 0.001110 | 5/2 |

三个 seed 都满足 correct 在均值和中位数上优于 stock，并有 75%--81.25% 的 object-balanced stock
胜率。这说明 B1 学到了一条可重复的小幅 condition 校正方向，不是完全 no-op。

但该收益非常小：三个 seed 的平均 stock flow MSE 约为 `0.183689`，平均 stock-correct 仅约
`3.13e-5`，相对改善约 `0.017%`。condition delta/stock RMS 已约为 1.39%--1.52%，说明较明显的
condition 改动只换来很小的 flow-loss 收益。

更关键的是对象特异性没有成立：

- correct-vs-shuffled object win rate 只有 50%--56.25%，全部低于 65% 门槛。
- seed 43/44 只有 3/5 和 2/5 个 t 上 correct 优于 shuffled。
- alignment gap 仅为 `0.000943--0.001331`，只达到 `0.01` 门槛的约 9%--13%。
- seed 42 的 shuffled-correct object median 还是负数。
- 三个 seed 的 stage context-delta correct/shuffled cosine 约为 0.776--0.872，最终注入方向仍有
  很强的共性。

因此 `specificity ratio > 5%` 不能单独作为通过依据：它是两个极小均值之比，而对象胜率、跨 t
一致性和 alignment gap 都没有支持可靠的正确点云利用。

## 4. 失败机制判断

当前最符合数据的解释是：B1 将物理信息放到了正确的 view/patch，但网络主要学到了
`physical/mask present -> generic condition correction`，没有稳定学会“这个 point prior 是否属于
当前视觉对象”。

这是一个基于代码与结果的推断，依据包括：

1. raw projected physical features 和内部 response 明确区分 correct/shuffled，说明前端对应不是零信号。
2. 最终 correct 与 shuffled 的 object win 接近随机，说明差异在 learned output/gate 到 flow-loss 的
   链路中被压缩成相似方向。
3. `mask_patch_fraction` 和相机 ray/pose 对 correct/shuffled 是共享的，而审计样本中的点证据只覆盖约
   2% patch；共享 silhouette/mask 证据比对象特异点证据更容易提供通用收益。
4. alignment probability 始终接近 0.5，correct-shuffled gap 小一个数量级，表明辅助分类头也没有学出
   稳定对象匹配。

## 5. 阶段裁决

**不进入 B1 overfit64，不启用 Flow LoRA，也不通过增加 B1 step、选择单一 seed 或选择更好
checkpoint 绕过门槛。**

J1a.2-B1 的正式结论是：

> 显式 K/T patch 对齐解决了输入对应和工程可训练性，并产生了跨 seed 的微小 stock 改善；但正确
> point prior 相对 shuffled prior 的对象特异收益不稳定，因此 bridge/get_cond 路线仍未证明能够可靠
> 利用真实 Point/Pose。

## 6. 下一步建议

建议先做一次不训练的 checkpoint attribution audit，固定现有三个 B1 checkpoint、noise 和 t，比较：

```text
full correct
full shuffled
mask-only（清零对象特异 point/depth evidence）
point-only（去掉共享 mask_patch_fraction）
null pose/ray/UV
```

该审计只用于确认通用收益是否主要来自共享 mask，不用于继续筛选 B1 checkpoint。

下一条主训练路线应转到 **SS Flow latent 16^3 内的 stock-preserving physical cross-attention**：

1. 使用具有明确 16^3 空间语义的 noisy SS latent 作为 query，16^3 point/mask/visual-hull physical grid
   作为 key/value，不再借 bridge context 间接影响空间生成。
2. 保留 frozen stock Flow 路径；physical gate=0 时 adapter 完全关闭，velocity 和 rollout 必须 bit-exact
   stock。
3. 第一阶段冻结原 Flow 权重，只训练 zero-init physical cross-attention/adapter，不启用全 Flow LoRA。
4. 继续使用同一 target/noise/t 的 correct、shuffled、stock 配对，要求 correct 改善 target flow loss，
   shuffled 回到 stock。
5. 先运行 gradient/stock-equivalence/single-object overfit audit，再做 overfit16 三 seed；仍使用 object win、
   多 t 一致性和 absolute gain，而不是只看 ratio。
6. 只有 latent physical adapter 在 overfit16 和 fresh overfit64 上都建立稳定 specificity，才考虑弱
   Flow LoRA 和 differentiable occupancy/precision loss。

不建议直接投入 J1a.2-B2 的大规模 bridge 训练。若论文需要完整说明 bridge 失败原因，可把 ray-depth
bin B2 限定为单 seed、小步数机制消融；它不应优先于空间语义明确的 SS Flow latent 注入。

---

# SS Flow 16^3 Sparse-anchor 38.1--38.7 结果与阶段裁决（2026-07-13）

## 1. 完成状态与工程结论

38.1--38.7 已全部运行完毕。静态编译、5项单元测试、B1 attribution、64对象输入审计、单卡
s5、单对象s50、三组两卡overfit16训练、finite audit、teacher-forced评估和跨seed汇总均生成了
完整报告，所有进程码和audit进程码均为0。

所有训练运行均满足：

```text
completed step = checkpoint step = expected updates
nonfinite attempts = 0
model parameters finite
optimizer state finite
scaler state finite
stock/null velocity bit-exact
```

因此38阶段的最终 `FAIL` 不是Python、CUDA、DDP、BF16、梯度断路或checkpoint污染导致，而是严格
评估指标没有全部通过。

## 2. 38.1结构与梯度测试

5项测试全部PASS：

```text
non-wrapping sparse-prior shift
positive/negative label nonempty and exclusive
point-only/mask-only/null physical views
zero-init and stock fallback
output-first gradient startup followed by encoder gradients
```

s5也验证了真实模型中的预期启动顺序：step 1只有zero-init output projection获得梯度；step 5时
`physical_encoder/state_encoder/time_mlp/fusion/output` 梯度均finite且非零。s5完成5/5更新，
nonfinite=0。

## 3. 38.2 B1 mask/point attribution

该实验固定已完成的三个B1 checkpoint、16个对象、noise和 `t=0.5`，只改变physical输入组成。

| train seed | full correct Flow gain | full shuffled gain | mask/full | point/full | correct condition RMS | shuffled condition RMS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.00005493 | 0.00003960 | 15.95% | 80.94% | 0.022472 | 0.022613 |
| 43 | 0.00003886 | 0.00006682 | 77.30% | 102.02% | 0.024523 | 0.025382 |
| 44 | 0.00005067 | 0.00004761 | 76.25% | 90.01% | 0.023741 | 0.024344 |

这些比例不是可加分解，因为mask-only、point-only与full会经过同一个非线性网络。但结果纠正了一个
过强判断：B1的微小收益并非纯粹来自mask，point-only分支也能产生接近full的Flow gain。

然而它仍没有证明对象特异利用：

- seed 43 的shuffled gain明显大于correct。
- seed 44的correct和shuffled几乎相同。
- correct/shuffled condition delta RMS在每个seed都非常接近。
- 三个seed的point局部positive gain方向不一致。

所以B1更准确的结论是：网络能利用“point evidence存在”产生通用校正，但没有稳定利用point属于
当前视觉对象这一对应关系。

## 4. 38.3输入与标签审计

64对象、64样本审计整体PASS，failed sample为0。

| metric | mean | median | min | max |
| --- | ---: | ---: | ---: | ---: |
| sparse prior cells | 139.56 | 82 | 19 | 619 |
| positive 16^3 cells | 310.30 | 291.5 | 102 | 807 |
| reliable negative 16^3 cells | 1742.16 | 1786 | 242 | 3041 |
| positive 64^3 voxels | 5763.31 | 4942 | 1435 | 18595 |
| reliable negative 64^3 voxels | 112625.31 | 115761 | 15644 | 194833 |
| sparse-prior/GT-cell overlap | 0.84857 | 0.87192 | 0.55056 | 1.0 |

全部样本的positive/negative均非空且互斥。448个controlled shifts全部改变输入；correct相对shift的
物理一致性胜率为72.77%，GT局部重叠胜率为95.54%。50% point dropout的一致性胜率只有59.38%，
说明dropout是较弱反事实，但平移corruption足以提供稳定训练信号。

这证明当前失败不能归因于“稀疏点太少所以输入层完全不可分”。点数分布确实稀疏且长尾，但正确点与
受控错误点在GT局部重叠和现有physical channels上具有明确统计差异。

## 5. 38.5单对象结果

单对象s50数值audit PASS，但scientific decision为FAIL。

```text
correct - stock positive probability       = +0.00387951
correct - corrupted positive probability   = +0.00000344
stock - correct global Flow MSE             = +0.00004362
correct-vs-stock positive t                 = 5/5
correct-vs-corrupted positive t             = 3/5（要求4/5）
neutral velocity MSE                        = 0.00014351（上限0.00001）
velocity delta RMS                          = 0.013214
```

单对象模型能增强anchor附近occupancy，也没有恶化平均Flow MSE；但correct相对corrupted的差异极小，
在 `t=0.7/0.9` 方向不稳定，并且neutral泄漏超过门槛约14倍。因此单对象实验只证明adapter可学习，
没有证明局部修正足够专一。

## 6. 38.6--38.7 overfit16三seed结果

三个seed均完成50/50更新且finite audit PASS。所有seed的physical-off和null velocity最大绝对差均为0。

| seed | correct-stock local | correct-corrupt local | corrupt object win | positive t stock/corrupt | neutral MSE | stock-correct Flow MSE | Flow相对退化 | strict |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 0.00389738 | 0.00021300 | 93.75% | 5/5 | 0.00036791 | -0.00030337 | 0.163% | FAIL |
| 43 | 0.00584524 | 0.00023203 | 75.00% | 5/5 | 0.00090200 | -0.00102044 | 0.549% | FAIL |
| 44 | 0.00539169 | 0.00021759 | 81.25% | 5/5 | 0.00059697 | -0.00070643 | 0.380% | FAIL |

这里的 `correct-stock local` 是anchor positive probability增益；`correct-corrupt local` 是正确点相对
受控平移点的额外增益。三个seed均有：

```text
correct > stock local mean/median
correct > corrupted local mean/median
correct > stock object win = 100%
correct > corrupted object win = 75%--93.75%
correct > stock at 5/5 t
correct > corrupted at 5/5 t
outside degradation小于预注册0.001上限
```

这是迄今Point/Pose实验中最明确的正结果：直接在SS 16^3空间使用局部物理监督，首次跨三个训练seed、
16个对象和全部五个噪声区间稳定建立了correct优于controlled-corrupted的方向。它明显强于bridge
路线接近随机的correct-vs-shuffled object win。

但增益的主体仍是通用anchor校正。correct相对corrupted只占correct相对stock增益约：

```text
seed42: 5.47%
seed43: 3.97%
seed44: 4.04%
```

即corrupted prior也获得了绝大多数局部positive提升。正确空间对应提供了稳定但较弱的额外收益。

## 7. strict FAIL的直接原因

三个seed失败项完全一致：

```text
neutral_preserved = false
global_flow_not_degraded = false
```

### 7.1 Neutral泄漏

neutral velocity MSE为 `3.68e-4--9.02e-4`，是 `1e-5` 上限的约37--90倍。原因不仅是loss权重，
还包括当前mask和inference gate定义不完全一致：

- adapter gate启用 `anchor_region OR reliable_outside`。
- training/eval positive要求anchor同时与GT target重叠。
- negative要求reliable outside同时远离GT target。
- gate-active但未满足GT positive/negative的歧义cell被归入neutral。

因此部分neutral cell在结构上允许非零delta，`neutral_preserve_weight=0.5` 只能软约束它们，不能保证
严格不变。当前 `neutral_preserved` 失败是可解释的空间泄漏，不是评估脚本误差。

### 7.2 全局Flow MSE轻微退化

三个seed的stock平均Flow MSE相同，约为 `0.185727`；correct后相对退化为0.163%、0.549%和0.380%。
这说明局部decoder occupancy目标确实推动了anchor区域，但该方向并不完全等价于原SS latent的全局
flow-matching目标。

退化幅度不大，却跨三个seed和全部主要t稳定存在，因此不能作为随机波动忽略，也不能在结果出来后
删除预注册的 `global_flow_not_degraded` 条件来宣布PASS。

### 7.3 Outside指标

outside probability的绝对均值从stock的约 `0.00044812` 增至：

```text
seed42: 0.00046361
seed43: 0.00047331
seed44: 0.00046472
```

增量只有 `1.55e-5--2.52e-5`，远低于0.001上限，所以outside检查PASS。当前主要问题不是外部区域
失控，而是gate-active歧义neutral和全局Flow目标冲突。

## 8. 阶段裁决

跨seed汇总的正式结果是：

```text
expected seeds present                 PASS
all seed stock/null exact              PASS
all seed correct beats corrupted mean  PASS
all seed strict decisions              FAIL
overall passed                         FAIL
```

因此按预注册门槛：

> **现在不进入fresh overfit64，不启用Flow LoRA，不扩大adapter，也不通过增加训练步数绕过门槛。**

但本阶段不能简单归类为“稀疏点路线无效”。更准确的结论是：

> **稀疏点在SS 16^3空间能够提供跨seed、跨对象、跨t稳定的局部物理修正方向；当前实现尚未把该方向
> 限制在足够小的空间范围内，而且局部occupancy改善以轻微全局Flow退化为代价。**

这为论文保留了一个可解释结果：bridge/get_cond注入缺少可靠对象specificity，而直接SS空间监督能
恢复local specificity，但需要更严格的stock-preserving spatial parameterization。

## 9. 下一步建议

在转入Pose-guided dense visual lifting前，只建议增加一次**不重训、不选checkpoint**的有界
`physical_scale`审计：

```text
scale = 0.1, 0.2, 0.3, 0.5
固定现有三个seed checkpoint、全部16对象、三noise、五个t
```

理由是velocity residual对scale近似线性，而neutral MSE近似按scale平方下降。以当前结果估算，
`scale=0.1` 可能把neutral MSE降到 `3.7e-6--9.0e-6`，同时保留较小的correct-corrupted局部符号。
这项实验只能回答“失败是否主要来自幅度”，不能用于重新挑选训练seed或修改原始scale=1.0结论。

低scale审计继续条件仍应预先固定：

```text
三个seed correct-corrupted mean/median > 0
三个seed correct-corrupted object win >= 65%
每个seed至少4/5 t方向正确
neutral velocity MSE <= 1e-5
outside degradation <= 0.001
报告global Flow相对退化，不事后删除该指标
```

若没有同一个scale跨三个seed满足上述条件，立即停止sparse-anchor训练路线并执行
`ar_ss_flow/pose_lifting主线.txt`。即便低scale通过，也应先做object-disjoint validation和完整rollout，
不能直接启用Flow LoRA或SLAT。

## 10. Physical-scale有界审计结果

按照上一节的预注册建议，固定三个s50 checkpoint、16个对象、三个noise和五个`t`，只在eval-time
测试：

```text
physical_scale = 0.025, 0.05, 0.075, 0.10, 0.20
```

五个scale的评估和multi-seed summary进程码均为0，stock/null在全部运行中保持bit-exact。但没有
任何一个scale跨三个训练seed通过全部strict checks。

| scale | seed42 | seed43 | seed44 | 主要失败项 |
| ---: | --- | --- | --- | --- |
| 0.025 | PASS | FAIL | FAIL | seed43/44 global Flow degraded |
| 0.050 | PASS | FAIL | FAIL | seed43/44 global Flow degraded |
| 0.075 | PASS | FAIL | FAIL | seed43/44 global Flow degraded |
| 0.100 | PASS | FAIL | FAIL | seed43/44 global Flow degraded |
| 0.200 | FAIL | FAIL | FAIL | neutral超限；seed43/44 global Flow degraded |

### 10.1 Neutral响应符合scale平方规律

| scale | seed42 neutral MSE | seed43 neutral MSE | seed44 neutral MSE |
| ---: | ---: | ---: | ---: |
| 0.025 | 2.30e-7 | 5.64e-7 | 3.73e-7 |
| 0.050 | 9.20e-7 | 2.26e-6 | 1.49e-6 |
| 0.075 | 2.07e-6 | 5.07e-6 | 3.36e-6 |
| 0.100 | 3.68e-6 | 9.02e-6 | 5.97e-6 |
| 0.200 | 1.47e-5 | 3.61e-5 | 2.39e-5 |

`scale <= 0.10` 时三个seed均满足 `neutral MSE <= 1e-5`，而 `scale=0.20` 时全部超限。该曲线
基本严格遵循scale平方，证明scale=1.0时的neutral失败主要来自residual幅度，而不是数值异常。

### 10.2 局部Point specificity在低scale仍然存在

全部scale上都满足：

```text
三个seed correct-stock local mean/median > 0
correct-stock object win = 100%
三个seed correct-corrupted local mean/median > 0
correct-corrupted object win = 81.25%--100%
correct-vs-corrupted positive t >= 4/5
```

以neutral余量和local signal较平衡的 `scale=0.075` 为例：

| seed | correct-corrupted local | object win | neutral MSE | stock-correct Flow MSE |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 1.35e-5 | 87.50% | 2.07e-6 | +4.75e-6 |
| 43 | 1.40e-5 | 93.75% | 5.07e-6 | -1.95e-5 |
| 44 | 1.47e-5 | 100.00% | 3.36e-6 | -1.19e-5 |

因此降低scale没有抹掉correct point相对controlled-corrupted point的稳定优势。这进一步确认局部
physical方向是真实存在的，不是scale=1.0下的偶然大扰动。

### 10.3 Global Flow冲突不能通过scale修复

seed42在 `0.025--0.10` 均通过全部检查；seed43/44在每一个非零低scale上唯一失败项都是
`global_flow_not_degraded`。

```text
seed43 stock-correct Flow MSE:
  scale 0.025 = -5.46e-6
  scale 0.050 = -1.19e-5
  scale 0.075 = -1.95e-5
  scale 0.100 = -2.80e-5

seed44 stock-correct Flow MSE:
  scale 0.025 = -3.23e-6
  scale 0.050 = -7.20e-6
  scale 0.075 = -1.19e-5
  scale 0.100 = -1.74e-5
```

减小scale只让退化幅度近似趋近于0，没有改变seed43/44的符号。这说明当前失败不再只是adapter过强，
而是两个独立训练seed学到的局部occupancy修正方向与全局flow-matching目标存在稳定冲突。继续测试
更小scale只会趋近stock/no-op，不能形成有意义的新模型收益。

### 10.4 Scale sweep最终裁决

跨seed summary在五个scale上均为 `passed=false`。因此维持原预注册裁决：

```text
不进入sparse-anchor fresh overfit64
不启用Flow LoRA
不训练SLAT
不继续扫描更小scale
不事后删除global Flow门槛
```

`scale=0.075` 可以保留为论文机制消融点：它证明稀疏点能够产生跨seed、跨对象、跨t稳定且低泄漏的
局部修正，但当前parameterization不能把该修正无损地并入完整SS生成方向。

## 11. 下一步建议：转入Pose-guided Dense Visual Lifting

下一阶段按 `ar_ss_flow/pose_lifting主线.txt` 执行。核心变化是让稀疏点只承担深度/局部锚点职责，
使用相机K/T把稠密VGGT/Image patch feature显式提升到SS `16^3` 空间，提供完整对象的视觉证据。

推荐顺序：

1. **P0坐标审计**：统一world/camera/canonical/SS voxel坐标；完成voxel、view patch、ray/depth的
   sentinel round-trip，确认view顺序和patch flatten顺序。
2. **P1离线dense visual volume**：把每个16^3 voxel投影到所有view，采样VGGT/Image feature，使用
   mask、visibility、depth consistency聚合；同步保存support和depth variance审计通道。
3. **P2未训练可分性**：比较correct pose、small pose perturbation、shuffled pose、depth corruption和
   null input。只有correct pose稳定提高跨视图feature consistency才允许训练。
4. **P3 stock-preserving smoke**：冻结VGGT、Image encoder、bridge、stock SS Flow和decoder，只训练
   zero-init spatial adapter；physical-off必须bit-exact stock。
5. **P4 overfit16三seed**：使用与38阶段相同的fixed noise、多`t`、object-balanced和controlled
   corruption协议；先证明correct pose优于corrupted pose。
6. **P5 fresh overfit64和object-disjoint validation**：只有P4三seed全部通过才扩大，不从overfit16
   checkpoint resume。

第一版不启用普通Flow LoRA、不训练SLAT，也不把sparse-anchor adapter直接叠加到dense lifting上。
先单独验证dense pose-guided volume能否同时改善local geometry和global Flow；通过后再把38阶段已经
证明有效的稀疏局部锚点作为第二个独立分支加入。
