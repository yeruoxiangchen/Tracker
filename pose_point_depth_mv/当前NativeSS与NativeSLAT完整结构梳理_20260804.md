# 当前 Native-SS 与 Native-SLAT 完整结构梳理

更新时间：2026-08-04  
适用范围：当前 `Native SS step2000 EMA + Stock/Full SLat` 主线，不包含早期 Direct-SLAT residual v3/v4/v5。

## 1. 先给出最重要的结论

当前完整系统不是“把 GenReCon 的 SLat 直接搬进 ReconViaGen”，也不是“在 Stock 和 GenReCon 两套 SLat 输出之间训练一个 gate”。准确结构是：

```text
多视图 RGB / mask / K / T
          |
          v
Native-SS Full（已训练并固定为 step2000 EMA）
          |
          v
预测 16^3 sparse structure
          |
          v
转换成实际 active 32^3 SLat 坐标
          |
          +------------------------------+
          |                              |
          v                              v
Stock-SLat                         Native-SLat Full v3
冻结原版 SLat Flow                同一冻结 SLat Flow 基座
原版跨视图均值                    + LoRA
无新 3D condition                 + posed-DINO 3D condition
                                  + 学习式逐点视图融合
          |                              |
          +---------------+--------------+
                          |
                          v
                  同一个冻结 Mesh decoder
                          |
                          v
                         Mesh
```

因此，当前 SLat 对比中的 Stock 和 Full 共享：

- 同一个 Native-SS step2000 EMA 输出坐标；
- 同一组输入图像、相机内外参和 native image context；
- 同一个初始 SLat 噪声、采样器和 CFG 参数；
- 同一个冻结 SLat 基座与 Mesh decoder。

它们的差异只在 SLat Flow 内部：Full 打开 LoRA、posed-DINO every-block 3D condition 和学习式视图融合，Stock 将这些增量全部关闭。

## 2. “Stock”一词必须按阶段区分

当前讨论里出现过两个不同的 Stock：

| 名称 | 含义 | 当前用途 |
|---|---|---|
| SS-stage Stock | 预训练模型原始、冻结的 SS Flow | 用于判断 Native-SS 相对原始 SS 是否改进 |
| 当前端到端 Stock | `Native-SS step2000 EMA + 冻结原版 Stock-SLat + 冻结 decoder` | 当前 SLat 开发的主基线 |

现在说“Full 对 Stock”时，默认指第二行。也就是说，Native-SS 已经成为 Full 和 Stock 的共同上游，不再是两者的变量。

当前冻结基线标识为：

```text
native_ss_step2000_ema_cfg5_plus_frozen_stock_slat.v1
```

Stock-SLat freeze SHA256：

```text
aeef32f5c17c1d039f93f1237b7c012f51d9854ad21a027a8c6645d09ceed91f
```

## 3. Native-SS：当前已经成立的上游结构

### 3.1 输入与 3D lifting

Native-SS 使用多视图 RGB/mask、相机内参 `K` 和相机外参 `T`。DINO 图像特征按照相机几何投影到原生 SS 的 16^3 三维查询位置，形成 pose-aware 的 3D evidence。

这条路径不是仅仅把多视图 token 做无条件平均，而是先利用 pose 建立“某个三维位置由哪些视图支持”的对应关系，再由 `GenreconViewAggregator` 对有效视图证据进行学习式聚合。

### 3.2 注入方式

聚合后的 1024-channel 3D condition 通过 24 个彼此独立、零初始化的线性投影，在每个 SS transformer block 之前注入：

```text
h_b <- h_b + P_b(condition_3d)
```

其中 `P_b` 在初始化时为零，因此初始前向保持 Stock SS；训练后投影自己学习注入方向和幅度。这里没有额外的手工 condition gate。

### 3.3 同时训练的部分

Native-SS 的可训练部分为：

- SS attention LoRA：rank 8、alpha 16，覆盖全部 24 个 block，共 120 个模块；
- posed-DINO 3D aggregator；
- 24 个 every-block condition projection。

冻结部分为：

- Stock SS Flow 基座；
- SS decoder；
- 已缓存的图像编码器特征。

参数统计：

| 部分 | 参数量 |
|---|---:|
| SS LoRA | 2,555,904 |
| 新 3D condition | 33,584,129 |
| 总可训练参数 | 36,140,033 |

### 3.4 当前部署绑定

Native-SS 当前固定使用：

- checkpoint：step 2000；
- weights：EMA；
- sampler steps：25；
- CFG strength：5.0；
- CFG interval：`[0.5, 1.0]`；
- guidance rescale：0.0；
- `rescale_t=3.0`。

这一部署结果为后续 Stock-SLat 和 Full-SLat 提供完全相同的 sparse coordinates。

## 4. Stock-SLat：原版跨视图平均没有动

Stock-SLat 仍是 ReconViaGen/TRELLIS-vggt 预训练 SLat Flow。其 block 内对多视图 context 的处理保持发布代码：

```python
for ctx in context:
    x = x + self.cross_attn(h, ctx) / len(context)
```

写成公式即：

```text
StockCross_b(p) = (1 / V) * sum_v CrossAttn_b(q_b(p), c_v)
```

关键事实：

- Stock 的均值实现没有被删除或重写；
- Stock 权重没有继续训练；
- Stock 不使用新 posed-DINO 3D condition；
- Stock 不使用 Full 的学习式 view fusion；
- Stock 不启用 LoRA；
- Stock decoder 保持冻结。

保留这一分支的目的，是让当前任何新结构都能与同一个稳定、可复现的原版 SLat 基线比较。

## 5. Native-SLat Full v3：在同一 Stock 基座上增加三类能力

### 5.1 增量 A：SLat attention LoRA

Full 在冻结 SLat Flow 的 attention 层上启用 LoRA：

- rank 8；
- alpha 16；
- 24 个 block 全覆盖；
- 120 个 LoRA 模块；
- 参数量 2,555,904。

LoRA 是通用 SLat Flow 修正路径。它可以修改 self-attention/cross-attention 投影，但本身不保证改动来自 pose，也不等同于 support 身份机制。

### 5.2 增量 B：GenReCon 启发的 posed-DINO 3D condition

这一部分继承的是 GenReCon 的设计思想，不是 GenReCon 的 SLat 权重或完整模型：

```text
每视图 DINO patch feature
        + K/T
        + active 32^3 SLat 查询坐标
                   |
                   v
       frustum/visibility-aware lifting
                   |
                   v
       learned mean/variance aggregation
                   |
                   v
          1024-channel condition_3d
                   |
                   v
  24 个独立零初始化 projection，逐 block 注入
```

其查询位置是 Native-SS 实际预测出来的 active 32^3 SLat stem coordinates，而不是另建规则密集网格。

注入公式仍为：

```text
h_b <- h_b + P_b(condition_3d)
```

`P_b` 没有单独的标量 gate。它从零初始化开始，但训练后可以自行产生较大改动。

### 5.3 增量 C：Full-only 的学习式跨视图融合

这是 v3 相对 v2 新增的“去平均”尝试。准确说法不是删除平均，而是：

> Stock 永久保留均匀平均；Full 在每个 block、每个 sparse point 上学习从均匀平均过渡到非均匀视图加权。

对第 `b` 个 block、三维点 `p`、视图 `v`：

```text
o_bpv = CrossAttn_b(q_bp, context_v)

content_score_bpv = MLP_b(o_bpv)
geometry_score_bpv = posed_DINO_view_logit_pv

target_score_bpv = content_score_bpv
                 + gamma_b * geometry_score_bpv

target_weight_bpv = masked_softmax_v(target_score_bpv)
```

其中：

- `content_score` 来自该 block 自己的 per-view cross-attention 输出；
- `geometry_score` 来自 active32 posed-DINO aggregator 的原始逐点/逐视图 logit；
- 无效投影视图在 target softmax 中被 mask；
- 所有视图都无效时回退到均匀权重；
- `gamma_b` 是每个 block 的 geometry-logit scale，初始化为 1，范围 `[0,4]`。

最终实际使用的权重不是直接等于 target weight，而是：

```text
uniform_weight = 1 / V

effective_weight_bpv
  = uniform_weight
  + alpha_b * (target_weight_bpv - uniform_weight)
```

`alpha_b` 是每个 block 一个零初始化、范围 `[0,1]` 的 transition gate。

### 5.4 这个 gate 到底控制什么

`alpha_b` 只控制该 block 的 cross-view 权重：

```text
alpha_b = 0  -> 该 block 精确使用 Stock 的均匀视图平均
alpha_b = 1  -> 该 block 完全使用学习得到的非均匀视图权重
0 < alpha_b < 1 -> 二者插值
```

它不控制：

- Stock SLat 与 GenReCon SLat 两个模型的混合；
- Full 总输出与 Stock 总输出的混合；
- LoRA 的开关或幅度；
- every-block 3D condition 的开关或幅度；
- sampler CFG 的强度。

所以，即使 `alpha_b` 很小，也不能推出整个 Full 约等于 Stock，因为 LoRA 和 `P_b(condition_3d)` 仍然可以使 Full 明显偏离 Stock。

### 5.5 为什么实现里仍然计算 uniform mean

Full v3 同时累积两条量：

```text
uniform_fused = sum_v(o_bpv / V)
target_fused  = sum_v(target_weight_bpv * o_bpv)
fused         = uniform_fused + alpha_b * (target_fused - uniform_fused)
```

这是为零初始化的精确 Stock 起点和安全消融服务，不是又在训练一个“Stock/GenReCon 整体 gate”。代码用 streaming log-sum-exp 实现 target softmax，不会显式保存完整 `V x N x C` 特征张量。

## 6. 四类容易混淆的控制量

| 控制量 | 作用位置 | 控制内容 | 不控制什么 |
|---|---|---|---|
| sampler CFG | 扩散/Flow sampler | positive 与 negative condition 的速度组合 | 不选择 Stock/Full 架构 |
| `alpha_b` view transition gate | Full 的每个 SLat block | 均匀视图权重到学习视图权重 | 不控制 LoRA 和 3D condition |
| `gamma_b` geometry scale | view score 内部 | posed-DINO 几何 logit 相对 content score 的权重 | 不直接缩放最终 SLat residual |
| `P_b` every-block projection | SS/SLat block 输入 | 学习注入聚合后的 3D condition | 没有单独手工 gate |

此外，初始化时的“精确 Stock anchor”只是保证新模块刚创建时前向等于 Stock；训练后并不要求 Full 永远贴近 Stock。

## 7. Full v3 的冻结与训练边界

| 模块 | Stock 分支 | Full v3 分支 | 训练状态 |
|---|---|---|---|
| Native-SS step2000 EMA | 使用 | 使用 | 冻结 |
| SLat Flow base | 使用 | 使用 | 冻结 |
| native `slat_vggt_cond` / cached context | 使用 | 使用 | 冻结 |
| SLat LoRA | 关闭 | 使用 | 可训练 |
| posed-DINO active32 aggregator | 不使用 | 使用 | 可训练 |
| 24 个 block condition projection | 不使用 | 使用 | 可训练 |
| per-block view scorer/gate/geometry scale | 不使用 | 使用 | 可训练 |
| Mesh decoder | 使用 | 使用 | 冻结 |

Full v3 参数统计：

| 部分 | 参数量 |
|---|---:|
| SLat LoRA | 2,555,904 |
| posed-DINO aggregator + every-block condition | 33,584,129 |
| view fusion | 1,576,008 |
| 总可训练参数 | 37,716,041 |

## 8. 训练和部署的数据流仍有一个关键差异

训练 Native-SLat 时，flow matching 使用目标 Mesh 编码得到的真实 target SLat coordinates；部署和 Mesh 评估时，则使用冻结 Native-SS 预测得到的 coordinates。

```text
训练：GT target SLat coords -> Full SLat flow matching target
部署：Native-SS predicted coords -> Full/Stock SLat rollout -> Mesh
```

因此训练 loss 或 teacher-forced gain 改善，并不自动保证在 Native-SS predicted coordinates 上的完整 rollout 和 decoder Mesh 指标改善。这仍是当前训练收益不能稳定迁移到 Mesh 的重要候选原因之一。

## 9. 当前训练配置与结果状态

### 9.1 训练配置

- 数据：868 个训练 object，1417 个 sequence/sample；
- SLat 训练：2000 step；
- 两卡，global effective batch 8；
- 随机 condition view count：1 到 16；
- `p_uncond=0.1`；
- logit-normal `t`：mean 1.0，std 1.0；
- EMA target decay：0.9995；
- 推理使用 25 step、CFG 5.0、interval `[0.5,1.0]`。

### 9.2 v3 gate 实际学到的状态

训练尾部统计显示：

- `fusion_gate_mean` 约 0.0036；
- `effective_view_weight_deviation` 约 0.000354；
- EMA checkpoint 中 24 个 block 只有 16 个 gate 为正，8 个保持 0；
- 最大 gate 约 0.0198。

这说明 v3 的学习式视图融合实际只非常轻微地偏离 Stock 均匀平均。当前结果不能解释为“彻底去掉均值后失败”，因为模型事实上几乎没有离开均值起点。

### 9.3 6-object 开发集 Mesh 结果

同一 Native-SS coordinates、同一初始噪声、同一 sampler 和 decoder 下，Full v3 相对 Stock：

| 指标 | mean | median | win rate |
|---|---:|---:|---:|
| Chamfer-L1 improvement | -0.00062761 | -0.00074915 | 0.3333 |
| F-score@0.02 delta | -0.00671906 | -0.00029742 | 0.1667 |
| Normal consistency delta | -0.00041244 | -0.00181339 | 0.1667 |
| Largest-component-ratio delta | +0.00240991 | +0.00001128 | 0.5000 |

结论：当前 v3 没有超过冻结 Stock；6 例只用于开发诊断，不能形成正式科学结论。

## 10. 当前结构与 GenReCon 的关系

当前命名里的 `genrecon` 表示结构借鉴，不表示基座归属。

| 内容 | 当前系统来源 |
|---|---|
| SS/SLat 预训练 Flow 基座 | ReconViaGen/TRELLIS-vggt |
| native image context / `slat_vggt_cond` | ReconViaGen/TRELLIS-vggt |
| Mesh decoder | ReconViaGen/TRELLIS-vggt |
| pose-aware 2D-to-3D lifting、每层 condition 思路 | GenReCon 启发后的本地实现 |
| LoRA、零初始化 Stock anchor | 本项目适配设计 |
| SLat per-block learned view fusion | 本项目 v3 新设计，非原版 GenReCon 权重 |

因此最准确的描述是：

> 当前 Full 是以 ReconViaGen/TRELLIS-vggt 为冻结生成基座，加入 GenReCon 启发的 pose-aware 3D condition，并增加本项目的 LoRA 与逐 block 学习式视图融合。

## 11. 当前应采用的消融顺序

在继续改架构前，应使用同一 v3 checkpoint 和相同 6-object 协议拆开以下分支：

1. Stock：所有增量关闭；
2. LoRA-only；
3. 3D-condition-only；
4. LoRA + 3D condition，但强制 `alpha=0`；
5. 完整 Full v3，使用 learned `alpha`；
6. 仅用于诊断的 forced-`alpha` sweep，例如 0.05、0.1、0.25、1.0。

这里最先需要回答的是：

```text
同一 checkpoint 下：learned alpha 是否优于 forced alpha=0？
```

如果答案是否定的，说明 v3 view fusion 没有带来收益；如果两者几乎相同，则说明 gate 太小，当前 Mesh 退化主要来自 LoRA/3D condition，而不是视图融合。

在这个消融完成前，不建议直接启动 32 object x 3 seeds 的更大验证，也不建议把 Stock 的原版均值实现删除。

## 12. 代码与证据位置

核心代码：

- Native-SS：`pose_point_depth_mv/native_ss_genrecon.py`
- Native-SLat v3：`pose_point_depth_mv/native_slat_genrecon.py`
- SLat 训练：`pose_point_depth_mv/train_native_slat_genrecon.py`
- Stock 原版均值：`ReconViaGen/trellis/modules/sparse/transformer/modulated.py`

关键产物：

- Native-SS report：`/data/zjr/native3d_condition_ss_mixed1k_20260801_v1/ss868_sourceholdout_seed42_v1/report.json`
- Native-SS 部署证据：`/data/zjr/native3d_condition_ss_mixed1k_20260801_v1/ss868_sourceholdout_seed42_v1/final_sourcebalanced32_seed424344_v1/report.json`
- Stock-SLat freeze：`/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json`
- Native-SLat v3 train report：`/data/zjr/native_slat_viewfusion_v3_mixed1k_20260803_v1/train868_step2000_seed42_2gpu_v1/report.json`
- Native-SLat v3 checkpoint：`/data/zjr/native_slat_viewfusion_v3_mixed1k_20260803_v1/train868_step2000_seed42_2gpu_v1/checkpoints/step_002000.pt`
- v3 六例 Mesh report：`/data/zjr/native_slat_viewfusion_v3_mixed1k_20260803_v1/val6_seed42_step2000_ema_v1/report.json`

## 13. 一句话版本

当前系统是“已验证并冻结的 Native-SS + 原版冻结 Stock-SLat”作为基线；Full 在同一 SLat 基座上同时加入 LoRA、GenReCon 启发的 pose-aware every-block 3D condition 和一个从原版均匀平均零初始化起步的学习式视图融合。Stock 的平均没有动，view gate 也不是 Stock/GenReCon 整体 gate；而现有 v3 gate 几乎没打开，Full 的 Mesh 退化更可能来自 LoRA/3D condition 或训练坐标与部署坐标的差异，必须先做组件消融再决定下一次训练。
