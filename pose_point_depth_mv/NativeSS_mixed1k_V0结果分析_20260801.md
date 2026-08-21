# Native SS mixed1k V0 checkpoint/raw-EMA结果分析

日期：2026-08-01 UTC

## 1. 结论

V0已正常完成8个候选的评估。与旧SS64在已消费S3对象上表现出的负IoU、负recall和
occupancy收缩不同，本次868-object长训在source-balanced checkpoint-dev16上首次形成了
明确且随训练总体增强的Stock-relative decoded occupancy正优势。

按预注册V1规则，预计严格合格的候选为：

```text
step500  raw
step1000 raw
step1500 EMA
step2000 EMA
```

预注册排序首先比较IoU，因此V1应选择`step2000 EMA`。分析时
`coarse_selection.json`尚不存在，必须实际执行V1生成选择记录，不能用本文代替V1。

这只是16对象、单seed的checkpoint development结果，不是CFG校准，更不是final结论。
下一步应执行V1，然后只用独立indices `16-31`运行V2三seed CFG校准；V2通过前不得消费
final32。

## 2. T0训练合同

```text
train objects:                  868
train sequences:                1417
optimizer updates:              2000
world size:                     2
per-rank grad accumulation:     4
global effective batch:         8
global sample exposures:        16000
elapsed:                        13377.18 s（约3小时42分57秒）
report passed:                  true
```

最后一个optimizer update仍有正常梯度：LoRA gradient norm为`0.03804`，condition分支为
`0.15620`；flow loss相对Stock loss的当步gain为`+0.03526`。训练没有停在zero-init，
也没有出现非有限参数或未完成报告。

## 3. V0协议

```text
objects:                         16
joint seeds:                     [42]
CFG:                             1
sampling steps:                  25
weights:                         raw / EMA
checkpoints:                     500 / 1000 / 1500 / 2000
condition scale policy:          learned_projection_only
post-CFG cap:                    false
Stock empty records:             all candidates 0/16
disabled Stock equivalence:      all candidates latent_max_abs=0
```

dev16按来源严格均衡：Legacy Objaverse、Gap Objaverse、Pilot Objaverse、Omni各4对象；
2/4/8-view对象数为6/5/5。它与后续CFG16、final32以及868个训练对象均object-disjoint。

## 4. 八候选总体结果

`base pass`是评估器原始均值/count门；`V1 strict`还要求IoU和recall对象胜率至少0.5、
count ratio中位数位于`[0.85,1.20]`且Stock全部非空。

| Step | 权重 | IoU gain | IoU win | Recall gain | Recall win | Latent gain | Count mean/median | Base pass | V1 strict |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 500 | EMA | +0.050058 | 0.6875 | +0.065363 | 0.4375 | +0.031785 | 1.0651 / 0.9211 | YES | NO |
| 500 | raw | +0.061712 | 0.8125 | +0.099766 | 0.7500 | +0.008511 | 1.1937 / 1.0023 | YES | YES |
| 1000 | EMA | +0.108859 | 0.8750 | +0.155224 | 0.8125 | +0.020707 | 1.2430 / 1.0149 | NO | NO |
| 1000 | raw | +0.114864 | 0.8750 | +0.153955 | 0.8125 | +0.045915 | 1.1139 / 0.9449 | YES | YES |
| 1500 | EMA | +0.131568 | 0.9375 | +0.180120 | 0.9375 | +0.033086 | 1.0721 / 1.0185 | YES | YES |
| 1500 | raw | +0.140750 | 0.8750 | +0.194061 | 0.8750 | +0.032111 | 1.7357 / 0.9685 | NO | NO |
| 2000 | EMA | +0.149527 | 0.8750 | +0.200053 | 0.8750 | +0.033920 | 1.1636 / 0.9415 | YES | YES |
| 2000 | raw | +0.173497 | 0.9375 | +0.249884 | 0.9375 | +0.020816 | 1.2973 / 1.0668 | NO | NO |

主要趋势：

1. raw与EMA的IoU和recall随step总体上升，不存在旧SS64那种训练后occupancy持续收缩的
   总体签名。
2. raw在四个checkpoint上的IoU都高于同step EMA，但step1500/2000 raw因count ratio
   均值超过1.20而失败。raw的后期优势伴随更强的occupied-count膨胀风险。
3. EMA抑制了后期count波动。step1500/2000 EMA同时通过base门和V1 strict门，其中
   step2000 EMA按预注册IoU优先规则胜出。
4. 不应放宽规则改选step2000 raw。其IoU虽最高，但count mean=`1.2973`已经越过预注册
   上界；该候选只能保留为开发诊断证据。

## 5. 预期选择：step2000 EMA

```text
IoU gain:
  mean/median/win:   +0.14952659 / +0.09898650 / 0.8750
  95% CI:            [+0.07876517, +0.22850967]

Precision gain:
  mean/median/win:   +0.22478204 / +0.14434836 / 0.8750
  95% CI:            [+0.12306909, +0.33886952]

Recall gain:
  mean/median/win:   +0.20005334 / +0.14138442 / 0.8750
  95% CI:            [+0.11215218, +0.30376148]

Latent MSE gain:
  mean/median/win:   +0.03392022 / +0.04345734 / 0.7500
  95% CI:            [-0.00455022, +0.07203586]

Full/Stock count ratio:
  mean/median:        1.16355294 / 0.94146453
  min/max:            0.70483999 / 3.62730627
Full-Stock count:
  mean/median:        +158.69 / -330.00
```

IoU、precision和recall的object-bootstrap CI均完全高于0，且IoU/recall均有14/16对象
优于Stock；因此decoded occupancy优势不是仅由平均值符号或单个对象造成。latent MSE
均值为正，但CI跨0，说明当前最可靠证据是decoded occupancy改善，而不是所有来源上的
连续latent拟合都改善。

count ratio均值为1.164但中位数为0.941，最大值达到3.627，同时绝对count差值中位数仍为
`-330`。这说明少数Stock occupied count较小的对象仍会放大ratio；V2和final必须继续
同时审查均值、中位数、绝对count和对象级异常，不能只看总体均值通过。

## 6. step2000 EMA来源分解

每组只有4对象，以下用于检查方向一致性，不作来源级显著性结论。

| 来源 | n | IoU gain | IoU win | Recall gain | Latent gain | Count ratio | Full-Stock count |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy Objaverse | 4 | +0.009081 | 0.50 | +0.025653 | -0.047471 | 1.222031 | +1724.50 |
| Gap Objaverse | 4 | +0.180643 | 1.00 | +0.222704 | +0.058061 | 0.954399 | -719.50 |
| Pilot Objaverse | 4 | +0.152709 | 1.00 | +0.217912 | +0.059634 | 0.861583 | -654.75 |
| Omni | 4 | +0.255674 | 1.00 | +0.333944 | +0.065457 | 1.616199 | +284.50 |

四个来源的平均IoU和recall方向都为正，因此总体优势不是只由Omni或新Objaverse来源
构成。Legacy改善最弱，只有2/4对象IoU为正且latent均值为负；Omni的count ratio均值偏高，
Pilot接近count下界。这两点是V2三seed独立CFG校准需要重点复核的来源风险。

## 7. step2000 EMA视图数分解

| 条件视图数 | n | IoU gain | IoU win | Recall gain | Latent gain | Count ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 6 | +0.054064 | 0.833 | +0.076845 | +0.020820 | 1.414058 |
| 4 | 5 | +0.223061 | 1.000 | +0.281058 | +0.041976 | 0.961073 |
| 8 | 5 | +0.190547 | 0.800 | +0.266899 | +0.041584 | 1.065426 |

2-view组仍为正，但改善明显弱于4/8-view且count ratio更不稳定。这符合可见几何证据减少
后的预期，也说明后续final报告必须保留view-count分层，不能仅报告全体平均。

## 8. 与旧SS64失败的关系

旧SS64在S3上即使latent MSE改善，decoded IoU/recall仍为负，并伴随明显occupancy收缩。
本次V0在独立source-balanced dev16上已经改变该符号关系：从step500开始，多数候选的
IoU与recall均为正；step1500/2000 EMA还同时满足对象胜率和count稳定性约束。

因此“Native SS机制必然无法越过frozen decoder边界”的判断已被V0开发证据否定。
更合理的当前解释是：旧64-object训练规模和来源覆盖不足，导致模型没有学到可泛化的
decoder-compatible SS修正。这个结论仍需V2和未触碰final32验证，不能提前写成最终成功。

## 9. 执行决定

1. 运行V1，生成不可变`coarse_selection.json`；预期选择step2000 EMA。
2. V1通过后运行V2：只在独立CFG16、seeds 42/43/44上比较CFG=1/3/5。
3. V2必须继续检查count上界、Legacy弱增益、Omni膨胀和Pilot收缩；不得因V0较强而降低
   原有硬门。
4. V2通过后执行V3严格审计，再一次性消费final32；V2失败则停止，不运行final。
5. V0的16对象已经用于checkpoint/raw-EMA选择，今后不得作为final或独立验证证据。

## 10. 证据路径

```text
T0 report:
/data/zjr/native3d_condition_ss_mixed1k_20260801_v1/
  ss868_sourceholdout_seed42_v1/report.json

V0 reports:
/data/zjr/native3d_condition_ss_mixed1k_20260801_v1/
  ss868_sourceholdout_seed42_v1/checkpoint_weight_selection_dev16_v1/

split audit:
/data/zjr/native3d_condition_reviewed1k_inputs_20260731_v4/
  native_ss_sourcebalanced_split_seed20260801_v1/split_audit.json
```
