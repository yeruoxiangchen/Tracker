# 历史失败实验索引与 Direct-SLAT 经验复盘

日期：2026-07-28

本文只复盘已经发生过的实验，不把工程成功、teacher-forced loss 改善或单个样本效果写成模型效益结论。分析目标是回答：

1. 当前提出的“增强 support 身份机制”能否吸取历史教训；
2. “多步 rollout 对齐”是否有历史证据支持；
3. 哪些旧结论与当前 Direct-SLAT 的适用条件相同，哪些不能直接搬用。

## 1. 归档文件

| 编号 | 归档副本 | 原始路径 | SHA-256 |
|---:|---|---|---|
| 01 | `01_AR_SS_Flow_实验结果.md` | `ar_ss_flow/实验结果.md` | `55c43d0696cea25fcf8b5888a9d41bd827a84f69f9cb948255372a11748f6fb8` |
| 02 | `02_Pixal3D多视图_当前模型结构演变梳理.md` | `pixal3d_multiview/outputs/五层测试流程报告/当前模型结构演变梳理.md` | `f5927b004f075c8027d1a68e9f49ddc712e35753834289798a28cfdb321c4fd5` |
| 03 | `03_Pixal3D多视图_多视图稀疏结构五层测试报告.md` | `pixal3d_multiview/outputs/五层测试流程报告/多视图稀疏结构五层测试报告.md` | `da635afd0e4ed7b5b1e3424c85ce35b588bb0dac7f5f090094833fc1f7c697de` |
| 04 | `04_PointsTo3D_单图像condition_8viewprior_scaleup路线_20260629.md` | `points_to_3d_strict/单图像condition_8viewprior_scaleup路线_20260629.md` | `056656df651cbfd13b940343ff5df9c939ed8ce8ed01d0f0468a8da005779bd9` |
| 05 | `05_ReconVGGT_AR_Adapter_A_B阶段结果与B结构_20260703.md` | `reconvggt_ar_adapter_a/A_B阶段结果与B结构_20260703.md` | `1968706b9b35c7143a0066eb20518bbb13bad30dc11ae2fc2eb9e6957e25aef2` |
| 06 | `06_TRELLIS点云先验_点云先验PixalV9Smoke对比分析.md` | `trellis_point_prior_mv/阶段报告/点云先验PixalV9Smoke对比分析.md` | `278002bbb2e364441ab7c1e9165db484164cd43ebcf7566b0ff019c6e51c6d4a` |
| 07 | `07_TRELLIS点云先验_latent拼接Sanity结果分析.md` | `trellis_point_prior_mv/阶段报告/latent拼接Sanity结果分析.md` | `327ce0143ac793905f2ce656859b5e9a4d61224e7f7376f9d3b97b289f6b2f1e` |
| 08 | `08_TRELLIS点云先验_latentVal32与MaskedLatentInpaint实现报告.md` | `trellis_point_prior_mv/阶段报告/latentVal32与MaskedLatentInpaint实现报告.md` | `6b14e3ae0ddb09e1b96f9da3d3a3e63bb7537ce88c5e515560340e93f3a73b5e` |

这些文件是原文副本，不对原实验记录做改写。本文件是额外的交叉分析。

## 2. 总结结论

当前改进方向总体合理，但必须收窄为两个彼此独立的待验证问题：

```text
问题 A：正确 support 是否产生对象特异的有效残差，
        且 wrong/no-support 时所有学习残差都回到 Stock？

问题 B：在问题 A 成立后，该残差是否能在真实 25-step
        post-CFG 轨迹和 native decoder 中保留下来？
```

历史文档对这两个问题都有直接经验，并非完全不同的适用条件：

- support 身份问题与旧的 pose head、pair token、point-prior condition、SS Flow sparse-anchor 属于同一类问题：模型容易学到“有某种 evidence 就做通用修正”，却没有学到“该 evidence 必须属于当前对象”。
- teacher 到 rollout 的断裂与旧的 masked latent inpainting、SS Flow teacher probe、Direct-SLAT 旧实验高度同源：GT/teacher 状态上的 loss 改善可以在自由采样中消失，甚至出现 latent 更接近而 decoded sparse/mesh 更差。
- 当前 Direct-SLAT 的条件又不完全相同。它已经有 corrected SS coordinates、通过审计的 target SLAT/decoder、精确 Stock fallback、部署态 post-CFG teacher 目标和 same-noise Mesh 协议。因此旧实验里的坐标错位、预处理错误、无效 sampler wrapper、支持质量未归一化等结论不能机械地当作当前根因。

所以答案不是“旧文档不适用”，也不是“照搬旧方案”。正确做法是：

> 保留当前已经修正的坐标、Stock、post-CFG 和 decoder 绑定；用历史失败淘汰掉标量 gate、通用 LoRA、teacher-only、简单加步数和只看均值等做法，再分别实现 support-specific residual 与 native-schedule 多步状态对齐。

## 3. 历史实验反复出现的五种失败模式

### 3.1 判别信号存在，不等于生成控制有效

多视图五层测试中，pose/alignment/match head 多次能够把 wrong pose、cyclic、reverse 或 cross-sample 排到较低分；部分 scorer 甚至可以达到很高的 correct top-1。可是把这个 score 作为归一化权重、gate 或 condition prior 注入 sparse generation 后，收益通常只有 `1e-4` 到 `1e-3`，且 median、win rate 或 hard-negative rank 不稳定。

AR SS Flow 的 C1/C3 也得到相同结论：pairwise pose-sensitive 信号真实存在，但模型最终主要学到局部 pruning 或通用 anchor correction，没有学到对象身份对应。

对当前 Direct-SLAT 的含义：

```text
support compatibility classifier/gate 自己准确，
不能作为 support 身份机制通过的判据。
```

最终必须同时满足：

```text
Full(correct support) > LoRA-only
Full(correct support) > Stock
Full(wrong support) ≈ Stock
Full(no support) = Stock
```

### 3.2 “evidence 存在”捷径会压过“evidence 属于谁”

历史模型经常学到：

```text
有 mask / physical / point / pair token
    -> 做一个对多数对象略有利的通用修正
```

典型证据包括：

- normalized weighted aggregation 把绝对 confidence 除掉，只留下了“存在某种支持”的统计；
- J1a.2-B1 中 point evidence 只覆盖少量 patch，共享 mask/physical 特征却覆盖更广，最终学习成通用 correction；
- SS Flow sparse-anchor 的大部分收益来自 generic anchor，correct-vs-corrupt 的额外贡献很小；
- point-prior Stage 1 即使使用干净 oracle prior，correct 仍不能稳定胜过 shuffle/jitter；模型依赖 point prior，但不依赖其样本身份；
- 当前 Direct-SLAT v3 的 teacher 归因为 `LoRA-only > Full > Stock`，且 wrong-support Stock reversion 失败，属于同一症状。

因此，当前不能只增加一个 support-presence 标量 gate，也不能保留独立的通用 LoRA 分支，再期待较小的 wrong-support loss 自动把它变成对象特异修正。

### 3.3 同一残差乘不同权重，不是真正的 wrong-support 反事实

AR SS Flow C2 的失败非常直接：correct/wrong 分支只是对同一个正确 residual 使用不同 scalar，没有为 wrong branch 重建自己的 evidence。该实验不能证明模型理解了错误对应。

当前训练中的 hard negative 必须满足：

```text
相同 target
相同 corrected coordinates
相同 x_t / noise / t
相同 image/native condition
只替换并完整重建 support evidence
```

不能把 correct support token 打乱一个索引后仍复用 correct 的聚合结果，也不能只缩放同一 support residual。

### 3.4 teacher/latent 指标改善，可能不在 decoder 有效流形上

`latentVal32与MaskedLatentInpaint实现报告` 中，训练后的 q_pred latent L1 明显变好，但 decoded sparse IoU 反而低于 q_vis，component 数增加。teacher-forced image-only 在若干 `t` 上接近 oracle q_splice，自由 sampling 仍低于 q_vis。

点云先验 Stage 2 也给出更强的后续证据：

- hard known-latent reinjection 使 correct-vs-wrong 变得稳定；
- 但生成的 Stage2 sparse 在 synthetic、real SLAM 和手机 AR 上普遍碎成上千 component；
- frozen stock SLAT/mesh 对这些 OOD sparse coords 产生黑色碎片和拓扑问题；
- raw stock union 能修 artifact，却会稀释 point-prior geometry correction。

因此，当前 Direct-SLAT 不能只看 velocity MSE、latent L1 或 teacher gain。每一个训练候选都必须监控：

```text
final SLAT feature distribution
sparse topology / component / boundary
native decoder proxy 或 decoder endpoint
same-noise Mesh
```

### 3.5 缩小 residual、加训练步数不能修复方向冲突

ReconVGGT adapter 与 SS Flow 多次显示：

- residual norm 过大时会伤害全局结构；
- scale sweep 可以让模型更接近 no-op，却不能把错误方向变成正确方向；
- 单 session 稳定的 teacher residual 在跨 session 后可能不稳定；
- teacher relaxed pass 后继续 s200/长训，不会自然补出 rollout 或身份效益。

这直接支持当前锁住 v3 step1000。`LoRA-only > Full` 和 wrong-support 机制失败时，增加 step 数更可能强化通用 LoRA，而不是形成 support 身份。

## 4. 哪些经验直接适用于当前 Direct-SLAT

| 历史经验 | 适用程度 | 对当前方案的约束 |
|---|---|---|
| scorer/rank 有效不等于生成有效 | 直接适用 | gate 必须用生成分支和 Mesh 共同验收 |
| correct/wrong 必须完整重建反事实 evidence | 直接适用 | 同 target/noise/t/coords，只替换 support |
| 通用 LoRA/anchor 会吞掉身份信号 | 直接适用 | 所有可学习残差必须受 support 身份控制 |
| teacher gain 不等于 rollout gain | 直接适用 | 使用真实 sampler schedule 的多步 visited-state 训练 |
| latent L1 变好可能 decoder 变差 | 直接适用 | 加 decoder-aware endpoint，不只看 flow MSE |
| hard known-latent reinjection 可建立身份约束 | 原理适用，形式不直接适用 | 可借鉴“每步约束”，不能把物理 support 当成精确 GT SLAT latent 硬 clamp |
| sparse coords filter/union 可修 artifact | 仅作诊断 | 不应成为 Direct-SLAT 主训练方案 |
| 输入 crop、坐标 frame、sampler wrapper 曾有 bug | 当前已大幅排除 | 保留回归测试，不把它们继续当默认根因 |
| 低/中 `t` 先稳定，再覆盖高 `t` | 部分适用 | 多步训练采用 native timestep curriculum |
| 真实 AR prior 需要尺度/质量拒绝 | 未来部署适用 | 当前 897 合成训练先不与 support 身份改造混合 |

## 5. support 身份机制应如何增强

### 5.1 先改结构，再调 loss 权重

当前 v3 保留一个可以独立工作的 LoRA-only 通用修正路径，训练再用 support-dropout 把它拉回 Stock。实际结果仍是：

```text
LoRA-only > Full > Stock
```

优先修改为结构上 support-dependent 的 learned residual：

```text
v_full = v_stock + R_support(x_t, t, native_condition, support)

support missing:
  R_support = 0 exactly

support wrong/incompatible:
  R_support -> 0

support correct:
  R_support -> useful object-specific residual
```

实现上可以保留 LoRA 参数形式，但 LoRA delta 不能再是独立的通用路径。每个 LoRA delta 或每个 block 的 learned residual 都应由 support/native-condition/current-state 的兼容性进行空间化调制。兼容性必须看“support 与当前对象/状态是否一致”，而不只是 support 是否存在、support mass 是否大。

推荐第一版：

1. support token 与 native image/SLAT condition 做轻量 cross-attention；
2. support token 与当前 sparse state token 做局部兼容性；
3. 输出 per-token/per-block gate，而不是单个全局 scalar；
4. gate 和残差均 zero-init，保持精确 Stock 起点；
5. gate 控制 adapter 与 LoRA 的全部 learned delta；
6. support 缺失时保留现有 exact bypass。

不推荐第一版就做大型身份网络。历史上高容量 scorer 很容易只学会排序，未必能控制生成。

### 5.2 hard negative 需要覆盖捷径

wrong support 至少分为：

```text
cross-object, same support seed
cross-object, similar category/extent/point count
same-object spatial permutation
support tokens 保留但 identity/coordinate correspondence 打乱
mask/physical presence control
constant/mean support control
```

easy random negative 只能证明模型识别明显异常，不能证明对象身份。训练和验证均应报告每类 negative，而不是把它们混成一个均值。

### 5.3 身份目标要同时防止 no-op 和 generic correction

仅加大 wrong-support Stock loss 有退化为 `Full=Stock` 的风险；仅加 correct-vs-wrong hinge 又可能让 wrong 更差而不是回到 Stock。

需要三项联合约束：

```text
correct utility:
  Full(correct) 优于 Stock，且优于 LoRA-only/generic baseline

wrong reversion:
  Full(wrong) 接近同状态 Stock

off exactness:
  no-support / scale=0 精确等于 Stock
```

margin 应使用注册过的绝对/相对范围，且同时报告 mean、median、object win rate、bootstrap CI。若 correct utility 不成立，不能用更强 reversion loss 把模型压成 no-op 后宣布机制通过。

### 5.4 support 质量与身份是两个 gate

历史 P2 指标曾因 support mass、active view set 和 visibility 不一致而误判。建议拆成：

```text
quality gate:
  support 是否足够、可见、坐标有效

identity gate:
  support 是否属于当前对象/condition/state
```

质量不足时可以回 Stock；质量足够但身份错误时也必须回 Stock。两者不能用一个未经归一化的 mass score 混在一起。

## 6. 多步 rollout 对齐应如何实现

### 6.1 v3 的一个固定 Euler step 不够

当前 v3 已经修复了旧的语义错位：

```text
训练目标是部署态 post-CFG velocity；
Euler 方向与 sampler 一致；
一个 detached step 后再次监督。
```

这是必要修复，但历史和当前 S5 都说明它不充分。训练期 one-step rollout gain 约 `+0.0332`，最终 25-step Mesh Chamfer gain 约 `+0.000097`，且区间跨零。

下一版不应继续使用固定 `dt=0.05` 代表整个 25-step sampler。应直接读取部署 sampler 的：

```text
实际 timestep 序列
实际相邻 dt
CFG strength/interval
guidance rescale/rescale_t
delta clip policy
```

### 6.2 采用 visited-state 短 unroll curriculum

在两张 3090 和 897 数据条件下，推荐按阶段使用：

```text
horizon 0/1 -> 2 -> 4
```

每个 micro-step 只抽一个 horizon，避免同时保留多套大图。轨迹由当前 Full policy 在 native schedule 上生成；步间先 detach，主要目标是让模型看到自身造成的 off-teacher state 分布，而不是第一版就做昂贵的 25-step full-BPTT。

每个 visited state 至少记录：

```text
Full-vs-Stock target/velocity utility
predicted endpoint or x0 proxy error
raw/effective residual RMS
clip activation
state drift from Stock and target
support gate statistics
```

如果 horizon 2 已经出现 gain 翻转，不应直接增加到 4 或 25；先定位残差方向、clip 或 decoder 流形问题。

### 6.3 timestep curriculum 要覆盖真实高噪声起点

Points-to-3D 路线曾出现低/中 `t` 成立、`START_T=1` 纯噪声自由 rollout 失败。当前多步训练应分开报告：

```text
late/low-noise steps
middle steps
early/high-noise steps
```

可以先稳定中后段，再逐步加入高噪声起点，但最终验收必须覆盖完整 25-step 轨迹。不能用只在 GT 邻域有效的 teacher probe替代真实起点。

### 6.4 增加 decoder-aware endpoint，但不要第一版反传完整 Mesh

历史已经证明 latent MSE 和 decoder utility 可能相反。第一版可以增加较便宜、稳定的 endpoint：

```text
final SLAT latent distribution statistics
frozen decoder 中间 feature/logit proxy
decoded topology、boundary、component proxy
少量样本的真实 native Mesh
```

不建议一开始穿过 marching cubes、surface sampling 和完整 Mesh metric 反传。先用 frozen decoder proxy 约束 latent 落在有效流形，再用 same-noise Mesh 做不可替代的最终 gate。

## 7. 推荐的下一版分阶段顺序

### V4-L0：只做定位，不训练

在现有 v3 step100、同 6 个对象和同噪声上导出每个 native timestep：

```text
Stock / LoRA-only / adapter-only / Full
target/velocity loss
latent drift
residual RMS 与 clip
decoder proxy
```

目标是确认 Full 的 support harm 和 teacher gain 分别从第几步出现或消失。

### V4-A：support 身份结构 smoke

只解决：

```text
Full(correct) > LoRA-only/Stock
Full(wrong) ≈ Stock
no-support = Stock
```

先跑 2-object/5-step 代码覆盖，再跑 32-object teacher/mechanism。此阶段不因 teacher gain 好看就直接长训。

### V4-B：native-schedule horizon 2/4

只有 V4-A 身份 gate 通过后，加入 detached visited-state unroll。否则多步训练只会把通用 LoRA 修正扩散到更多状态。

### V4-C：decoder-aware endpoint

当 horizon 2/4 的 latent utility 能保留，再加 frozen decoder proxy，并执行 6-object same-noise Mesh。

### V4-D：扩大训练

只有以下条件同时成立才讨论 step1000 或 10k 数据：

```text
correct-support identity gate 通过
Full > LoRA-only
wrong/no-support 回 Stock
multi-step latent gain 不翻转
same-noise Mesh mean/median/win/CI 不再为零证据
```

扩大数据可以改善泛化，不能替代机制成立。

## 8. 预注册验收建议

### 8.1 身份 gate

```text
exact fallback:
  no-support / scale=0 max_abs = 0

correct utility:
  Full > Stock
  Full > LoRA-only
  mean > 0, median > 0, object win rate 达标

wrong reversion:
  wrong Full 距 Stock 小于 correct Full 距 Stock
  mean/median/win rate/CI 同时报告

hard-negative coverage:
  cross-object、similar-object、spatial-permutation、
  presence-only control 分开通过
```

### 8.2 rollout gate

```text
native schedule:
  timestep/dt/CFG/rescale/clip 与部署完全绑定

per-step:
  target utility curve 与 AUC
  gain 首次翻转 step
  state drift 与 decoder proxy

final:
  Full-vs-Stock latent endpoint
  6-object same-noise Mesh
  mean、median、win rate、bootstrap CI
```

### 8.3 不构成通过的证据

以下任一项单独成立都不能解锁长训：

```text
训练 loss finite
teacher Full > Stock
compatibility classifier top-1 高
错误 support 比正确 support 更差，但没有回 Stock
latent L1 改善
Mesh 均值被单个样本拉正
没有灾难性 collapse
```

## 9. 对当前路线的最终判断

当前“增强 support 身份机制 + 多步 rollout 对齐”的方向比继续训练 v3 更合理，也得到了历史实验的强支持。但应修正为：

```text
先结构性消除 ungated generic LoRA，
建立 correct-specific / wrong-revert / off-exact 三联机制；
再用真实 native schedule 做短多步 visited-state 对齐；
最后加入 decoder-aware endpoint 和 same-noise Mesh gate。
```

不建议的版本是：

```text
在现有 v3 上只提高 wrong-support loss；
增加一个全局 scalar compatibility gate；
继续保留独立 LoRA-only 通用修正；
把一个 detached fixed-dt step 增加训练次数；
teacher 指标转正后直接 step1000。
```

历史过程并没有证明 support/rollout 路线本身无效；它证明的是 soft condition、通用残差、teacher-only 和结果层后处理不够。当前方案只有在吸收这些限制后，才与旧失败方案形成实质区别。
