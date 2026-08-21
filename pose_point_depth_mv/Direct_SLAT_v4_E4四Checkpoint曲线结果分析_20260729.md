# Direct-SLAT V4 E4 四 Checkpoint 曲线结果分析

日期：2026-07-29

## 1. 结论

E4 的汇总产物、四个 checkpoint 绑定、对象数、Mesh 范围标记和 low-`t`
exact-Stock gate 均通过严格检查。但 E4 是**产物完整性门**，不是
“Full > Stock”科学结论门。

四个 checkpoint 中：

```text
step200: 当前唯一值得进入 fresh blind 的探索性候选
step100: 可保留为早期对照，不是主候选
step300: REJECT
step400: REJECT
step1000 / 当前V4的10k模型长训: 不启动
```

step200 在当前相同6对象、seed42协议上取得四项 Mesh 指标最高的 mean，四项 median
也全部为正；其中 largest-component ratio（LCR）为5/6对象获益，bootstrap mean
95% CI 不跨0。这是目前最有价值的候选信号。

但 Chamfer、F-score、normal 都只有3/6对象获益且 CI 跨0。因此目前只能说：

```text
step200 是探索性最优 checkpoint；
不能说 V4 Full 已经稳定优于 Stock；
更不能用这6个已参与选点的对象重复检验后作正式结论。
```

## 2. E4 范围与产物身份

汇总目录：

```text
/data/zjr/direct_slat_v4_checkpoint_curve_step100_200_300_400_seed42_20260729_v1/
aggregate_v1
```

核心产物：

```text
report.json:
  sha256 = 290a291a5e3f970b5f54816965f061d3e77fdd2aab58a00a6620edc3e266ec4a

curve.csv:
  sha256 = 0ae46311c86ff3a2f2664f2551e89fa5bb6003c88f076230816cb8657d68fc0c

summary.txt:
  sha256 = 97a13637134db5d624996f6719a8723982cfe77088057487d666d21654d3fa4a
```

复核结果：

```text
format = pose_point_depth_mv.direct_slat_checkpoint_curve.v1
steps = [100, 200, 300, 400]
teacher objects/checkpoint = 8
Mesh objects/checkpoint = 6
Mesh joint seeds = [42]
same corrected-SS coordinates = true
same coordinate-keyed initial noise = true
all low-t exact-Stock gates = pass
formal = false
automatic checkpoint selection = false
E4 strict recheck = PASS
```

## 3. 四 Checkpoint 最终 Mesh 曲线

下表正数表示 Full 优于 Stock。

| step | Chamfer mean / median / 胜率 | F-score mean / median / 胜率 | Normal mean | LCR mean |
|---:|---:|---:|---:|---:|
| 100 | +0.000032 / -0.000230 / 2/6 | -0.000066 / -0.000298 / 3/6 | -0.002845 | -0.000230 |
| 200 | **+0.000113 / +0.000009 / 3/6** | **+0.000239 / +0.000229 / 3/6** | **+0.002011** | **+0.001734** |
| 300 | +0.000031 / -0.000052 / 2/6 | -0.000884 / -0.000323 / 2/6 | +0.000182 | +0.001323 |
| 400 | +0.000026 / -0.000130 / 3/6 | -0.000295 / +0.000130 / 3/6 | -0.000276 | -0.000260 |

step200 的完整 bootstrap 结果：

| 指标 | mean | median | 胜率 | bootstrap mean 95% CI |
|---|---:|---:|---:|---:|
| Chamfer improvement | +0.00011308 | +0.00000917 | 3/6 | [-0.00028643, +0.00059470] |
| F-score delta | +0.00023866 | +0.00022908 | 3/6 | [-0.00203059, +0.00240508] |
| Normal consistency delta | +0.00201093 | +0.00082114 | 3/6 | [-0.00248495, +0.00810102] |
| LCR delta | +0.00173404 | +0.00126999 | 5/6 | **[+0.00044602, +0.00330236]** |

所以 step200 的信号不是“四项都已经显著”，而是：

```text
四项中心趋势一致偏正；
LCR 的对象覆盖和区间最强；
其他三项样本量不足、对象方向仍不一致。
```

step300/400 没有延续 step200 的 Mesh 峰值。尤其 step300 的 F-score 明显转负，
step400 的 normal 与 LCR 也重新转负。这排除了“训练越久，Mesh 越好”的单调解释。

## 4. Teacher/训练 Proxy 与 Mesh 已明确分叉

Teacher-forced Full gain 随训练严格上升：

```text
step100: +0.096264，8/8对象获益
step200: +0.121652，8/8对象获益
step300: +0.134725，8/8对象获益
step400: +0.144856，8/8对象获益
```

训练窗口的 rollout gain 也总体上升：

```text
step1-100:   +0.09964
step101-200: +0.16160
step201-300: +0.22502
step301-400: +0.23394
```

最终 Mesh 却在 step200 达到当前峰值，随后回落。由此可得：

```text
teacher gain 可证明模型在 teacher state 上学到了修正；
训练 rollout/endpoint proxy 可用于监视优化是否工作；
两者都不能替代最终 decoder 后的 Mesh utility。
```

当前主要瓶颈仍是 learned residual 在部署 rollout 中如何被累计、bound 如何改变它，
以及这些 latent/velocity 改善是否落到 decoder 可利用方向，而不是 teacher
学习完全失败。

## 5. Residual 与 smooth bound 曲线

只统计每个 checkpoint 的114次 support-active 调用：

| step | raw ratio > 0.1 | raw ratio mean / max | effective ratio mean / max |
|---:|---:|---:|---:|
| 100 | 5/114 = 4.4% | 0.0706 / 0.1317 | 0.0567 / 0.0797 |
| 200 | 71/114 = 62.3% | 0.1209 / 0.2097 | 0.0748 / 0.0903 |
| 300 | 108/114 = 94.7% | 0.1725 / 0.3381 | 0.0838 / 0.0959 |
| 400 | 114/114 = 100% | 0.2221 / 0.5090 | 0.0884 / 0.0981 |

必须区分两个概念：

```text
smooth scaling participated:
  smooth_rms_v2 对所有非零 residual 连续缩放，四个 checkpoint 都是114/114。

raw ratio > cap:
  表示缩放前 residual 是否已经越过配置的0.1半径。
```

因此不能把 step100 的5/114称为“bound只激活5次”。正确含义是：step100 只有5次
raw residual 超过0.1，但所有114次非零 residual 都经过 smooth mapping。到 step400，
114次 raw residual 全部超过0.1，effective mean 已从0.0567靠近0.0884。

先前 S7 初次摘要中出现的 `0.05365 -> 0.16881` 和
`0.04310 -> 0.06717`，是把36次 low-`t` exact-Stock 零调用也放进150次总调用
平均后的值；E4 表格使用 support-active 分母。两组数值没有冲突，但 active-only
口径更适合诊断 residual/bound；S7 分析文档现已同步改成 active-only 口径。

这条曲线说明 step200 是一个可能的“偏移量尚可利用”的中间点；继续训练后，raw
修正越来越大，effective 修正越来越接近 trust region 外沿，但 Mesh 不再改善。

## 6. LoRA 与 support adapter 的次级诊断

Teacher 层的 `Full - LoRA-only` gain：

```text
step100: +0.02693
step200: -0.03278
step300: -0.08792
step400: -0.13157
```

step200/300/400 的 bootstrap CI 均位于0以下。这表示更多训练后，LoRA-only 的
teacher gain 高于 Full；support adapter 与通用 LoRA 修正在 Full 组合中没有形成
单调协同。

这不能单独证明“support 没有因果作用”，因为它仍是 teacher 指标，且 Full 相对
Stock 始终为正。但它与以下现象一致：

```text
通用 LoRA correction 随训练增强；
组合后的 Full 被 smooth bound 共同缩放；
最终 Mesh 没有随 teacher gain 增强。
```

后续若重构 V5，应考虑分支级预算或组合约束，避免 adapter 提供的对象特定修正与
LoRA 通用修正相互竞争后再被同一个总 residual bound 压缩。

## 7. 当前裁决

```text
E4 artifact/integrity gate:             PASS
step200 exploratory candidate:          SELECT FOR BLIND TEST
step200 Full > Stock science claim:     NOT ESTABLISHED
step300 checkpoint promotion:           REJECT
step400 checkpoint promotion:           REJECT
continue V4 directly to step1000:       REJECT
start current V4 on 10k model training: REJECT
continue Mixed10k data construction:    ALLOW
```

## 8. 下一步

1. 冻结 fresh blind 协议后，才运行 step200。对象必须排除本次6个 Mesh 对象以及
   8个 teacher 选点对象；当前128-object val cache 足以选出新的对象。
2. 建议正式主门使用至少32个 fresh objects、joint seeds 42/43/44、
   same corrected-SS coordinates、same coordinate-keyed noise 和相同25-step
   sampler。统计应以对象为独立单位，不能把三个 seed 当作三个独立对象扩大样本量。
3. 预先冻结主指标和裁决规则。建议 Chamfer 为主指标，F-score、normal、LCR 为
   不退化/一致性指标；blind 结果出来后不能再按结果调整对象或 seed。
4. 可在 blind 前对 step200 做 LoRA-only、adapter-only、Full 的同6对象 Mesh
   诊断，但该诊断只能帮助设计 V5，不能用于 step200 的正式胜率声明。
5. 若 fresh blind 支持 step200，则将 step200 固定为897-object候选，再决定是否做
   10k 短程 scaling smoke；若 blind 失败，则停止 V4 同配置扩步，转向分支预算和
   decoder-aware/rollout-aware utility 对齐。
