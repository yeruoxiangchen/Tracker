# Native SS + frozen Stock SLAT T1-T4 完整实验结果分析

日期：2026-08-02 UTC

## 1. 结论

本轮实验的工程门和 Mesh transfer 质量门均通过。

在固定 Native SS `step2000 EMA / CFG=5 / 25-step / post-CFG cap=false`、冻结原版
Stock SLAT 和 Mesh decoder 的条件下，仅将 Stock SS support 替换为 Native SS support，
最终 32 个对象、3 个联合 seed 的结果相对 Stock 全面改善：

- Chamfer-L1 从 `0.079316` 降至 `0.034347`，下降约 `56.7%`；
- F-score@0.02 从 `0.229280` 升至 `0.568267`；
- normal consistency 从 `0.619894` 升至 `0.761631`；
- largest-component ratio 从 `0.827953` 升至 `0.952847`；
- Stock 与 Native 两支均为 `96/96` Mesh 成功，无空 Mesh 或非有限输出。

因此，当前证据支持以下机制结论：

> Native SS 的 occupancy/support 改善能够被未经微调的 Stock SLAT 和 Stock Mesh decoder
> 直接消费，并转化为明显的最终 Mesh 几何改善。当前没有必要为了建立这条主结论而立即
> 重训 SLAT。

但 T3 仍标记为 `exploratory=true`。这 32 个 final 对象已经参与 Native SS 的 V4 判断，
而本次 Mesh transfer 协议是在看到 V4 结果后建立的，所以本轮结果是很强的开发/机制证据，
不能直接写成新的 untouched formal blind 结论。正式论文证据仍需新的预注册、未触碰、
source-balanced holdout。

## 2. 实验问题与固定协议

本轮没有训练 SLAT，比较的是：

```text
Control: Stock SS  -> frozen Stock SLAT -> frozen Stock Mesh decoder
Full:    Native SS -> frozen Stock SLAT -> frozen Stock Mesh decoder
```

固定条件如下：

- Native SS checkpoint：`step_002000.pt`，SHA256
  `c87cc3b6581b3ba58786bdbe924883a4611557b4b7f21ab3a062abf85336f94a`；
- Native SS 权重：EMA；
- Native SS：CFG `5`，25 steps，CFG interval `[0.5, 1.0]`，`rescale_t=3`；
- Stock SLAT：`Stable-X/trellis-vggt-v0-2` 原版冻结权重；
- Stock SLAT：CFG `5`，25 steps，CFG interval `[0.5, 1.0]`，`rescale_t=3`；
- 两支共享图像、相机、SLAT condition、SLAT sampler、Mesh decoder 和 GT canonical frame；
- 每个对象/seed 使用同一个 `64^3` coordinate-keyed master noise field；
- 两支公共 support coordinate 上的 SLAT 初始噪声必须 bit-exact 相同；
- T3 使用 32 个 source-balanced final 对象和 seeds `42/43/44`；
- 每个预测 Mesh 采样 20,000 个表面点计算指标；
- 对象是统计单位，先跨三个 seed 求对象均值，再做对象 bootstrap CI。

## 3. T1-T4 阶段结果

| 阶段 | 规模 | 作用 | 结果 |
|---|---:|---|---|
| T1 | 1 对象 x seed42 | 完整 SS -> SLAT -> Mesh 工程 smoke | runtime PASS，transfer checks PASS |
| T2 | 8 对象 x seed42 | 四来源各 2 个、2/4/8-view 为 3/3/2 的诊断 | runtime PASS，transfer checks PASS |
| T3 | 32 对象 x 3 seeds | 完整 exploratory transfer gate | `96/96` paired cases 完整，全部硬检查 PASS |
| T4 | 读取 T3 report | 汇总总体、来源、视角和 seed | 正确复现 T3 PASS；不是独立评估 |

T1 只证明路径可运行，不承担科学判断。T2 已出现一致正向趋势：

- Chamfer improvement mean `+0.025675`，对象胜率 `7/8`；
- F-score@0.02 delta mean `+0.242183`，对象胜率 `7/8`；
- normal consistency delta mean `+0.090787`，对象胜率 `7/8`；
- Chamfer bootstrap CI `[+0.001919, +0.051969]`。

T3 将该趋势扩大到完整 32 对象和三个 seed 后，方向保持稳定且幅度变大，因此 T1/T2
不是偶然的单对象或单 seed 现象。

## 4. T3 总体 Mesh 结果

这里的 Chamfer improvement 定义为 `Stock Chamfer - Native Chamfer`，越大越好；
其余 delta 定义为 `Native - Stock`，越大越好。

| 指标 | Stock 绝对均值 | Native 绝对均值 | 对象级改善 mean | median | 对象胜率 | 95% bootstrap CI |
|---|---:|---:|---:|---:|---:|---:|
| Chamfer-L1 | 0.079316 | 0.034347 | +0.044969 | +0.039656 | 30/32 = 93.75% | [0.031856, 0.059393] |
| F-score@0.02 | 0.229280 | 0.568267 | +0.338987 | +0.307242 | 28/32 = 87.50% | [0.249660, 0.429630] |
| Normal consistency | 0.619894 | 0.761631 | +0.141738 | +0.110819 | 28/32 = 87.50% | [0.095967, 0.190409] |
| Largest-component ratio | 0.827953 | 0.952847 | +0.124894 | +0.059677 | 26/32 = 81.25% | [0.069676, 0.185872] |
| Mesh success | 96/96 | 96/96 | 0 | 0 | 同等 | [0, 0] |

所有预注册在 T3 evaluator 中的检查均为 `true`：记录数、配对数、无无效 pair、
Chamfer mean/median/对象胜率/CI、F-score 非退化、Mesh success 非退化、最大连通分量
非退化，以及至少两个 seed 的方向非负。

`mesh_success_delta=0` 不代表没有改善，而是两支都能生成有效 Mesh；质量差异由 Chamfer、
F-score、法向和连通性指标刻画。

## 5. 三个 seed 的稳定性

| Seed | Chamfer improvement mean | median | 对象胜率 | 95% CI | F-score delta mean |
|---:|---:|---:|---:|---:|---:|
| 42 | +0.045287 | +0.032086 | 87.50% | [0.030088, 0.061630] | +0.343714 |
| 43 | +0.041514 | +0.027452 | 93.75% | [0.028082, 0.055237] | +0.332796 |
| 44 | +0.048105 | +0.045547 | 87.50% | [0.032430, 0.063442] | +0.340452 |

三个 seed 的 Chamfer mean、median 和 CI 下界全部为正，均值范围仅
`0.0415-0.0481`。总体结论不是由单一随机噪声 seed 驱动。

## 6. 按来源分析

每个来源均为 8 个对象。

| 来源 | Chamfer mean | Chamfer 胜率 | F-score delta mean | F-score 胜率 | Normal delta mean | LCR delta mean |
|---|---:|---:|---:|---:|---:|---:|
| legacy_objaverse | +0.017925 | 75% | +0.122973 | 50% | +0.058741 | +0.115783 |
| gap_objaverse | +0.046304 | 100% | +0.300912 | 100% | +0.160056 | +0.068560 |
| pilot_objaverse | +0.061022 | 100% | +0.399938 | 100% | +0.164898 | +0.037713 |
| omni | +0.054623 | 100% | +0.532126 | 100% | +0.183256 | +0.277521 |

结论：

1. 总体优势不是只由 Omni 驱动。Gap、Pilot 和 Omni 的 Chamfer/F-score/normal
   对象胜率均为 100%。
2. Pilot 的 Chamfer 改善最大，Omni 的 F-score 与连通性改善最大。
3. Legacy Objaverse 是明确的剩余薄弱来源。它的 Chamfer 均值仍为正，但 F-score median
   为 `-0.003962`，F-score 和 normal 仅 4/8 对象改善。
4. 仅有的两个对象级 Chamfer 回退都来自 legacy Objaverse 的 4-view 对象：
   `4c906eb3c9354102870b8a5dd9573bfd` 和
   `6b3fe40191804a31a8a0776698b57eb5`。

不应使用这两个对象继续调当前 checkpoint 或 CFG；它们只能作为后续 legacy 分布诊断样本。

## 7. 按视图数分析

| 视图数 | 对象数 | Chamfer mean | Chamfer 胜率 | F-score delta mean | Normal delta mean | LCR delta mean |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 11 | +0.054635 | 100% | +0.287486 | +0.192644 | +0.175082 |
| 4 | 11 | +0.037978 | 81.82% | +0.333947 | +0.107765 | +0.094010 |
| 8 | 10 | +0.042026 | 100% | +0.401183 | +0.123111 | +0.103660 |

2-view、4-view、8-view 的均值都为正，没有出现“只有多视图才有效”。2-view 的
Chamfer 胜率也是 100%，说明 Native SS 的 AR 稀疏视图价值已经传到 Mesh。

4-view 是当前最弱分层，两个 Chamfer 回退对象都位于该组。但 4-view 仍有 9/11
Chamfer 胜率，不能据此判断 4-view 机制整体失败；更可能是 legacy source 与具体对象难度
共同作用。

## 8. V4 count 下界的重新解释

绑定的 Native SS V4 结果为：

| SS 指标 | Mean gain | Median gain | 对象胜率 | 95% CI |
|---|---:|---:|---:|---:|
| Occupancy IoU | +0.162281 | +0.126811 | 87.50% | [0.110815, 0.216465] |
| Precision | +0.217436 | +0.200894 | 87.50% | [0.155405, 0.281337] |
| Recall | +0.218296 | +0.203988 | 81.25% | [0.149761, 0.291073] |
| Latent MSE gain | +0.087999 | +0.086597 | 78.13% | [0.046092, 0.129244] |

唯一失败项是 `count_ratio_lower`：

- Native/Stock occupied-count ratio mean `0.845971`；
- 门下界为 `0.85`，只低 `0.004029`；
- ratio median `0.755378`；
- Native 每对象平均比 Stock 少约 `3706` 个 occupied coordinates。

本次 Mesh transfer 给出了直接下游证据：更少的 support 没有造成 Mesh 退化，反而对应
更低 Chamfer、更高 F-score/normal consistency 和更高 largest-component ratio。派生的
对象级相关性也支持这一解释：

| 关系 | Pearson r | p | Spearman rho | p |
|---|---:|---:|---:|---:|
| SS IoU gain vs Mesh Chamfer improvement | 0.453 | 0.0093 | 0.570 | 0.00067 |
| SS IoU gain vs Mesh F-score delta | 0.944 | <1e-15 | 0.959 | <1e-17 |
| SS Recall gain vs Mesh Chamfer improvement | 0.577 | 0.00054 | 0.670 | 0.000027 |
| Count ratio vs Mesh Chamfer improvement | 0.114 | 0.534 | 0.073 | 0.693 |

因此，在当前对象上，Mesh 改善与 occupancy 的正确性高度相关，与“保留了多少 Stock
occupied count”基本无关。更合理的解释是 Native SS 删除了大量 Stock false-positive
support，而不是发生了有害 occupancy collapse。

这些相关性是同一 final32 上的事后诊断，不能作为独立因果证明；但它们足以否定
“count 低于 0.85 就必然不能传到 Mesh”的解释。历史 V4 的 PASS 字段不能事后篡改，
应保留为 count guard FAIL，同时将其科学解释校准为“count-only guard miss，后续 Mesh
质量门通过”。

## 9. 工程完整性与残余风险

完整性结果：

- SS support：96/96 记录通过，Stock/Native SS 使用同一初始噪声；
- SLAT：96/96 pair 的公共 coordinate 初始噪声 bit-exact 相同；
- SLAT 执行顺序：Stock-first 50 次，Native-first 46 次，基本平衡；
- 共享预处理重放：32/32 condition audit 通过，cached/recomputed Stock condition
  最大绝对差为 `0.0`；
- Mesh：192/192 branch 通过，无 invalid pair；
- 日志中没有 Traceback、RuntimeError、AssertionError 或 command-not-found；
- Stock Mesh watertight rate 为 `88.54%`，Native 为 `100%`；
- Stock winding-consistent rate 为 `98.96%`，Native 为 `100%`；
- Native 的 mean largest-component ratio 更高，但平均 component count 也从
  `10.06` 增至 `13.20`。新增分量占比很小，不能把结果表述为“分量数量全面改善”。

SLAT repeat audit 的 coordinate 完全一致，但 feature 不是 bit-exact：最大绝对差
`0.033177 < 0.1`，按预设 tolerance 通过。这反映 CUDA sparse/attention 路径存在小幅
非确定性。公共初始噪声公平性不受影响，且三个 seed 方向一致，但正式 holdout 应保留该
repeat audit 和交替分支执行顺序，不能声称整个 SLAT 采样过程 bit-exact deterministic。

本次 `render_previews=false`，因此报告只提供几何数值证据，没有完成盲化定性视觉审查。

## 10. 当前决策与下一步

### 当前决策

1. 接受 `Native SS + frozen Stock SLAT + frozen Stock Mesh decoder` 作为当前主候选管线。
2. 暂不训练 Native SLAT。当前目标“Native SS 改善能否传到最终 Mesh”已经得到强正向回答；
   立刻训练 SLAT 会扩大变量数量，并削弱“只替换 coarse SS 即获得收益”的清晰论文叙事。
3. 不再在当前 final32 上回选 SS checkpoint、EMA/raw、CFG、SLAT CFG 或门阈值。

### 下一步优先级

第一优先级是 T5 新 untouched holdout，而不是 SLAT 重训：

- 在看结果前冻结新的 source-balanced、view-balanced 对象列表；
- 与训练 868、dev16、CFG16 和当前 final32 全部 object-disjoint；
- 固定当前 step2000 EMA、Native CFG=5 和 Stock SLAT 参数，不再校准；
- 继续使用 seeds `42/43/44` 和相同 same-noise protocol；
- 预先冻结总体 Chamfer/F-score/normal/LCR 门和来源分层报告方式；
- 建议至少 32 个对象，若数据和算力允许优先 64 个对象、每来源 16 个；
- 增加盲化 Mesh render/contact-sheet 审查，但不得用视觉结果回调当前参数。

只有以下情况才进入 Native SLAT 训练：

- 新 formal holdout 上 Native SS occupancy 仍改善，但 Stock SLAT Mesh transfer 不再改善；或
- 当前冻结 Stock SLAT 已通过 formal transfer，但论文明确需要额外 SLAT 创新，并将其作为
  独立增量/消融，而不是修复当前已经通过的主链。

## 11. 证据文件

- T1 report：
  `/data/zjr/native_ss_stock_slat_mesh_transfer_20260802_v1/smoke1_seed42_v1/report.json`
- T2 report：
  `/data/zjr/native_ss_stock_slat_mesh_transfer_20260802_v1/sourcebalanced8_seed42_v1/report.json`
- T3/T4 report：
  `/data/zjr/native_ss_stock_slat_mesh_transfer_20260802_v1/final32_seed424344_exploratory_v1/report.json`
- T3 summary：
  `/data/zjr/native_ss_stock_slat_mesh_transfer_20260802_v1/final32_seed424344_exploratory_v1/summary.txt`
- T3 SLAT repeat audit：
  `/data/zjr/native_ss_stock_slat_mesh_transfer_20260802_v1/final32_seed424344_exploratory_v1/slat_repeat_audit.json`
- T3 log：
  `/data/zjr/native_ss_stock_slat_mesh_transfer_20260802_v1/logs/T3_final32_seed424344.log`
