# Native SS v2 D2 step500 EMA CFG诊断结果分析

日期：2026-08-01 UTC

## 1. 诊断范围

- checkpoint：`step_000500.pt`
- 权重：EMA
- 对象：已消费的S3 `val object 16:64`，共48个对象
- seeds：`42/43/44`
- CFG候选：`1/3/5`
- sampler：25 steps，`cfg_interval=0.5,1.0`，`guidance_rescale=0`，`rescale_t=3`
- condition语义：`learned_projection_only`
- post-CFG cap：未使用
- 评估：每个CFG内部采用same-noise Stock/Native比较，frozen SS occupancy decoder

输入报告：

- CFG 1/3：`/data/zjr/native3d_condition_ss_genrecon_20260731_v2/ss64_seed42_v1/diagnostics_s3_consumed_20260801_v1/step500_ema_cfg1_3/calibration.json`
- CFG 5：`/data/zjr/native3d_condition_ss_genrecon_20260731_v2/ss64_seed42_v1/final_ema_val16_64_seed424344_v1/report.json`
- 汇总：`/data/zjr/native3d_condition_ss_genrecon_20260731_v2/ss64_seed42_v1/diagnostics_s3_consumed_20260801_v1/D2_step500_ema_cfg1_3_5_by_source.json`

这些对象已经在S3中被查看，本次只能作为诊断，不能再作为未触碰final test。

## 2. 判定

`D1_RC=2`是正常完成评估但没有候选通过硬门，不是程序崩溃。`selected=null`是正确结果。

三个CFG均未证明Native SS优于Stock：

| CFG | IoU gain mean | IoU median | IoU胜率 | Recall gain mean | Latent MSE gain | Count ratio mean | Count ratio median |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -0.004361 | -0.008316 | 0.354167 | -0.013164 | +0.012853 | 0.889365 | 0.802624 |
| 3 | -0.007556 | -0.003135 | 0.416667 | -0.028455 | +0.034256 | 0.825447 | 0.789374 |
| 5 | -0.005794 | -0.003016 | 0.458333 | -0.028054 | +0.052710 | 0.806634 | 0.783618 |

主要结论：

1. CFG=1在平均IoU、平均recall和平均count ratio上相对最不差，但仍然低于Stock，不是通过配置。
2. CFG=5的IoU中位数和对象胜率反而最高。因此不能笼统地称CFG=1为所有统计口径下的“最佳CFG”。
3. 没有任何CFG达到IoU对象胜率`0.5`；对应胜出对象数分别约为17/48、20/48和22/48。
4. CFG=1/3/5的IoU bootstrap 95% CI都跨0，没有证据证明任一配置在总体IoU上稳定优于Stock。
5. CFG=3和5的recall 95% CI完全低于0，存在明确的recall退化；CFG=1的recall CI为`[-0.026060,+0.000255]`，退化有所缓解，但仍未证明改善。

因此，D2否定了“只要把S2选出的CFG=5降下来，step500模型就能通过S3”的假设。CFG过高是放大因素，但不是唯一根因。

## 3. CFG放大的核心趋势

CFG从1增加到3和5时：

- latent MSE gain从`+0.012853`增加到`+0.034256`和`+0.052710`；
- count ratio从`0.889365`下降到`0.825447`和`0.806634`；
- recall gain从`-0.013164`恶化到约`-0.028`；
- 每对象平均Full-Stock occupied count从`-1131`恶化到`-2208`和`-2630`。

这给出了比原S3更直接的证据：更强CFG稳定放大连续latent空间中的修正，但该修正越来越倾向于跨过frozen decoder的empty边界，造成occupied count和recall下降。

这不是“decoder被解冻”，而是当前训练得到的flow方向与frozen decoder occupancy边界仍未完全对齐。降低CFG只能减弱这一现象，不能消除它。

需要注意，每个CFG内部的Stock和Native使用相同CFG及相同初始噪声，但不同CFG之间的Stock rollout本身也会变化。因此跨CFG结果代表对应部署协议下的Stock-relative效果，不应解释成只改变Native residual幅度的纯消融。

## 4. CFG=1不能被视为通过

CFG=1通过了latent均值和count ratio均值上下界，但仍有以下失败：

- `iou_gain_mean=false`
- `recall_gain_mean=false`
- `stock_baseline_nonempty=false`
- IoU对象胜率只有`0.354167`
- count ratio中位数只有`0.802624`，仍低于0.85

CFG=1共有4/144个Stock rollout解码为空，占`2.78%`。这些seed的IoU、precision、recall和latent gain仍然有效，但`Full/Stock count ratio`在数学上未定义，因此记录为`null`，不参与ratio均值；48个对象都至少还有一个非空Stock seed，所以对象级ratio仍有定义。

这意味着CFG=1的`count ratio mean=0.889365`是对有效ratio的条件统计，不能单独用它宣称occupancy硬门已经健康。空Stock还说明低CFG下Stock采样本身存在少量退化尾部。

## 5. 四来源结果

### 5.1 CFG=1

| 来源 | n | IoU gain | Recall gain | Latent gain | Count ratio |
|---|---:|---:|---:|---:|---:|
| Legacy Objaverse | 20 | -0.008907 | -0.021273 | +0.015098 | 0.794113 |
| Gap Objaverse | 10 | -0.011757 | -0.027301 | +0.013251 | 0.809729 |
| Pilot Objaverse | 9 | -0.005909 | -0.014312 | -0.001437 | 1.066675 |
| Omni | 9 | +0.015505 | +0.021712 | +0.021709 | 1.012211 |

CFG=1对Omni组最有利：IoU、recall和latent均为正，count接近Stock。总体仍失败，是因为三个Objaverse组均为负，其中Legacy和Gap仍有明显收缩。

### 5.2 CFG=3

| 来源 | n | IoU gain | Recall gain | Latent gain | Count ratio |
|---|---:|---:|---:|---:|---:|
| Legacy Objaverse | 20 | -0.001894 | -0.016335 | +0.035939 | 0.809657 |
| Gap Objaverse | 10 | -0.039794 | -0.106763 | +0.044449 | 0.723608 |
| Pilot Objaverse | 9 | -0.006255 | -0.015718 | +0.007179 | 0.965303 |
| Omni | 9 | +0.014378 | +0.018884 | +0.046266 | 0.833834 |

CFG=3让Legacy的IoU接近0，但Gap组明显恶化；Omni仍有正IoU/recall，却开始出现count收缩。

### 5.3 CFG=5

| 来源 | n | IoU gain | Recall gain | Latent gain | Count ratio |
|---|---:|---:|---:|---:|---:|
| Legacy Objaverse | 20 | -0.001157 | -0.019473 | +0.065047 | 0.771718 |
| Gap Objaverse | 10 | -0.031548 | -0.095241 | +0.046795 | 0.750086 |
| Pilot Objaverse | 9 | -0.001355 | +0.006008 | -0.001210 | 1.023881 |
| Omni | 9 | +0.008076 | -0.006535 | +0.085787 | 0.729810 |

CFG=5对Pilot Objaverse的decoded occupancy相对最好，但其latent gain略负；对Omni则出现“latent大幅改善但recall/count恶化”的典型错配。Gap在三个CFG下都失败，并对总体退化贡献最大。

## 6. 分来源结论

不同来源没有共同的最优CFG：

- Omni偏向CFG=1。
- Pilot Objaverse在当前指标下偏向CFG=5。
- Legacy的IoU在CFG=3/5更接近Stock，但recall和count仍偏向较低CFG。
- Gap在所有CFG下都失败，且对高CFG最敏感。

这解释了为什么用16个Legacy对象选出的单一CFG=5不能泛化到混合S3。它也说明不能通过为每种来源临时选择不同CFG来宣称模型通过：来源特定CFG需要独立development set和新的未触碰holdout，而且当前三个Objaverse组在CFG=1下仍未整体转正。

## 7. 对D3/D4的决策

D2完成后，D3/D4固定CFG=1现在有了数据依据，而不再是预先假设：

- CFG=1具有三者中最好的总体平均IoU、平均recall和count保持率；
- 它最适合用于检查“EMA是否额外造成occupancy收缩”；
- raw和EMA必须使用同一个CFG，才能只隔离权重类型这一变量。

因此D3应运行`step500 raw / CFG=1`，D4比较`step500 raw vs EMA / CFG=1`。这里选择CFG=1是诊断参考，不是部署配置，也不是通过判定。CFG=1的4个空Stock记录会同时出现在raw/EMA对照中，ratio应继续按当前`null + coverage`语义报告。

## 8. 后续顺序

1. 执行D3/D4，判断raw是否比EMA显著缓解recall/count退化。
2. 执行D5/D6/D7，先看step100/300/500在固定EMA/CFG=1下的粗轨迹。
3. 若粗轨迹存在转折，再执行D8/D9/D10补step200/400。
4. 如果raw和中间checkpoint也持续出现“latent gain为正、decoded occupancy为负”，则下一轮重训应直接处理flow目标与frozen decoder边界错配，而不是继续搜索CFG。
5. 所有当前D系列结果只能用于诊断；正式模型选择必须使用重新设计的分层development set和新的未触碰holdout。

## 9. 最终结论

D2证明CFG=5确实放大了occupancy收缩，但降低到CFG=1只能缓解，不能修复。当前step500 EMA在任何全局CFG候选下都没有优于Stock。

最重要的新证据是：随着CFG增强，latent MSE gain持续增加，而count ratio和recall整体下降；这种反向趋势支持“训练目标改善没有稳定穿过frozen occupancy decoder”的判断。与此同时，来源之间的响应方向不同，说明小样本Legacy校准偏置和混合来源泛化不足也是实际问题。
