# Direct-SLAT V4 step200 Fresh Blind32 正式结果分析

日期：2026-07-29

## 1. 最终结论

本次 V5 执行与产物完整性全部通过：

```text
V5_EXEC_RC = 0
report/protocol canonical SHA256 = PASS
formal report binding = PASS
32 fresh objects × seeds 42/43/44 = 96 paired records
same corrected-SS coordinates = true
same coordinate-keyed initial noise = true
```

但预注册的正式科学门未通过：

```text
V5_SCIENCE_RC = 3
primary_pass = false
secondary_pass = true
formal_pass = false
```

`SCIENCE_RC=3` 不是程序崩溃。它准确表示：

> V4 step200 的 Full 相对 Stock 呈现小幅、方向性正向的 Mesh 信号，但32个 fresh
> 对象上的不确定性仍覆盖0，因此不能正式宣称 `Full > Stock`。

当前应按预注册路线停止晋升这一 V4 配置：

```text
V4 step200 artifact/integrity:               PASS
V4 step200 directional Mesh signal:          WEAK POSITIVE
V4 step200 formal Full > Stock claim:         FAIL / NOT ESTABLISHED
promote V4 step200 as validated checkpoint:   REJECT
continue V4 step400 -> step1000:              REJECT
start current V4 configuration on 10k:        REJECT
continue Mixed10k data construction:          ALLOW
```

## 2. 正式协议与不可变身份

正式协议：

```text
/data/zjr/direct_slat_v4_step200_matched_blind32_seed424344_20260729_v1/
protocol.json

protocol_sha256:
b371b99119ccbc639b7273e5e103e4d5447da27c7ef1c28243eb26dcb1fe811e
```

正式报告：

```text
/data/zjr/direct_slat_v4_step200_matched_blind32_seed424344_20260729_v1/
mesh_formal_v1/report.json

report_sha256:
2b05758b58fb110b5faa454e06f2efc649d21c9c079338e62d7414f71ea66d6a
```

候选 checkpoint：

```text
V4 step_000200.pt
sha256:
2d4a272048b74272acb8ebcf9b735811520ee000a34c185c0b13dc118c3f906d
```

协议排除了11个先前参与 teacher/Mesh 选点的对象，并冻结32个新的
`object_uid`。统计时先对每个对象的三个 seed paired delta 求平均，再以
`object_uid` 为 bootstrap 单位；没有把96次 rollout 当成96个独立对象。

正式门在运行前固定为：

```text
primary Chamfer:
  bootstrap mean 95% CI lower > 0
  median > 0
  object win rate >= 0.50
  positive seed fraction >= 2/3

secondary mean non-degradation:
  F-score delta >= 0
  normal-consistency delta >= 0
  largest-component-ratio delta >= 0
```

## 3. 正式结果

正数均表示 Full 优于 Stock。

| 指标 | mean | median | 对象正向率 | bootstrap mean 95% CI | 正式检查 |
|---|---:|---:|---:|---:|---|
| Chamfer improvement | +0.00013156 | +0.00004072 | 18/32 = 56.25% | **[-0.00003437, +0.00032746]** | 主门失败 |
| F-score@0.02 delta | +0.00044584 | -0.00004222 | 16/32 = 50.00% | [-0.00041082, +0.00142561] | mean不退化通过 |
| Normal consistency delta | +0.00025885 | -0.00016718 | 14/32 = 43.75% | [-0.00105020, +0.00167140] | mean不退化通过 |
| Largest-component ratio delta | +0.00048623 | 0.00000000 | 14正/4平/14负 | [-0.00083916, +0.00216560] | mean不退化通过 |

Chamfer 主门中四个子检查只有一个失败：

```text
bootstrap lower > 0:       false
median > 0:                true
object win rate >= 0.50:   true
positive seed fraction:    1.0，true
```

CI 下界只比0低约 `3.44e-5`，所以这不是“Full 明显退化”的结果；但正式协议要求
下界严格大于0，结果出来后不能放宽阈值或改用单侧描述把失败改成通过。

## 4. 效应大小很小

96个 paired rollout 的 Stock/Full 平均 Chamfer 为：

```text
Stock mean Chamfer = 0.12916303
Full  mean Chamfer = 0.12903147
absolute improvement = 0.00013156
relative reduction ≈ 0.1019%
```

因此即使采用点估计，当前收益也约为 Stock Chamfer 的千分之一。这个量级解释了
为什么 teacher-forced velocity 层可以观察到强优势，而最终 Mesh 的对象级效应仍
容易被 rollout、decoder 阈值、对象差异和 seed 差异淹没。

对象平均 Chamfer improvement 的离散程度为：

```text
mean = +0.00013156
standard deviation = 0.00052978
standard error = 0.00009365
min / max = -0.00097810 / +0.00189139
25% / median / 75% = -0.00013023 / +0.00004072 / +0.00021962
```

效应标准差约为均值的4倍。CI 失败的主要原因是对象异质性相对点估计过大，而不是
均值方向转负。

## 5. 跨 seed 稳定性不足

三个 seed 的 Chamfer mean 都为正，所以 `positive_seed_fraction=1.0`：

| seed | Chamfer mean | 对象正向率 | median |
|---:|---:|---:|---:|
| 42 | +0.00010090 | 17/32 = 53.13% | +0.00006484 |
| 43 | +0.00003343 | 17/32 = 53.13% | +0.00002860 |
| 44 | +0.00026034 | 16/32 = 50.00% | +0.00000571 |

但“每个 seed 的总体均值为正”不等于“同一对象稳定受益”。每个对象三个 seed 中
Chamfer 为正的次数为：

```text
3/3 seeds positive:  5 objects
2/3 seeds positive: 12 objects
1/3 seeds positive: 11 objects
0/3 seeds positive:  4 objects
```

不同 seed 的对象级 Chamfer delta 相关性很低：

```text
corr(seed42, seed43) = +0.087
corr(seed42, seed44) = -0.042
corr(seed43, seed44) = -0.097
```

这说明 Full 并没有形成“哪些对象会被稳定改善”的强结构。当前小幅正均值更像
seed-dependent 的弱偏置，而不是可靠的对象级修正。

## 6. Secondary PASS 不能解释成全面结构稳定

三个 secondary gate 按预注册规则只检查对象平均值是否不小于0，因此均通过。
但其 median 和对象方向并不强：

```text
F-score:
  mean positive，median negative，16/32 objects positive

Normal:
  mean positive，median negative，14/32 objects positive
  seed42 mean = -0.00140526

LCR:
  mean positive，median = 0，14 positive / 4 equal / 14 negative
  seed43 mean = -0.00085453
```

所以正确表述是：

> 未观察到 secondary 指标的平均系统性退化；尚未证明 secondary 指标在多数对象、
> 多数 seed 上稳定改善。

## 7. 与 E4 和四分支诊断的关系

E4 在6个用于 checkpoint 选择的对象上，step200 Chamfer mean 为
`+0.00011308`；fresh Blind32 为 `+0.00013156`。二者的方向和量级相近。这说明
E4 的小幅正信号没有在 fresh 对象上完全消失。

但 fresh blind 同时证明：

```text
信号可重复为“小幅正点估计”；
不能重复为“统计上排除0的稳定优势”。
```

先前四分支6对象诊断也显示 Full 的 Chamfer mean 为约 `+0.000145`，而
LoRA-only、adapter-only 与 Full 在 Chamfer、normal、LCR 上各有不同方向。
结合本次 seed/object 异质性，V4 的主要问题不再是“完全学不到 residual”，而是：

1. adapter 与 LoRA 的组合预算没有形成稳定协同；
2. teacher/endpoint 的优势在25-step部署 rollout 中被重新分配；
3. smooth bound 后的 residual 并不总落在 decoder 能稳定转化为 Mesh 收益的方向；
4. 最终收益量级太小，难以支撑扩大同配置训练。

## 8. Post-hoc 稳健性检查

以下检查没有参与正式裁决，只用于判断失败形态：

```text
去掉两端各1个对象的 Chamfer trimmed mean: +0.00010989
去掉两端各2个对象:                       +0.00008666
任意 leave-one-object-out mean:           全部仍为正
```

因此正均值并非完全由单个最佳对象制造。不过这不能替代预注册 bootstrap 门：
trimmed mean 与 leave-one-out 是看到结果后的诊断，不能据此把 `formal_pass=false`
改写为通过。

按当前均值与对象标准差做的粗略正态近似表明，若真实分布不变，约需60余个对象才
可能把同量级均值的95%区间推到0以上。这只是 post-hoc 精度估计，不构成追加相同
V4 样本来“救活”本次 blind 的授权。

## 9. 路线裁决

应遵守第68节预注册分支：

```text
freeze V4 step200 artifacts as a negative/near-null formal result: YES
change thresholds after seeing results:                          NO
reuse these 32 objects to tune V4 and call them blind again:     NO
continue current V4 to step1000:                                 NO
start current V4 on Mixed10k long training:                      NO
continue Mixed10k rendering/cache construction:                  YES
use V4 step200 for human review / V5 baseline:                   YES
```

V5 应优先解决效应稳定性和 decoder utility，而不是单纯增加 V4 step 数：

1. 对 LoRA-only、adapter-only、Full 设置分支级 residual 预算或组合门，避免通用
   LoRA 修正和对象特定 support 修正竞争后再被同一个总 bound 缩放；
2. 将多步 rollout/decoder-aware proxy 用于训练或 checkpoint 选择，降低
   teacher velocity 与最终 Mesh 的错位；
3. 在897对象上先完成新的 exploratory 与 fresh gate，再考虑10k短程 scaling
   smoke；
4. 若确有必要复核当前 V4，只能另建全新对象、预注册 sequential/replication
   规则的独立重复实验；它不能覆盖本次 `formal_pass=false`。

## 10. 范围限制

本报告只检验：

```text
fixed corrected-SS coordinates
native SLAT Stock vs Direct-SLAT Full
same coordinate-keyed SLAT initial noise
```

它没有检验端到端 Direct-SS 坐标分支。因此即使 formal gate 通过，也只能证明
matched corrected-coordinate Direct-SLAT utility；本次 formal gate 未通过，更
不能推出端到端 Full 已经优于 ReconViaGen stock 或 Pixal3D。
