# Native SS v2 S3 step500 EMA失败结果分析

日期：2026-08-01 UTC

## 1. 审计对象

- 训练：64个对象，500 optimizer steps，EMA权重。
- checkpoint：`/data/zjr/native3d_condition_ss_genrecon_20260731_v2/ss64_seed42_v1/checkpoints/step_000500.pt`
- S2校准：val对象`0:16`，3个joint seeds，候选CFG为`1/3/5`。
- S3最终测试：val对象`16:64`，48个对象，3个joint seeds。
- S3报告：`/data/zjr/native3d_condition_ss_genrecon_20260731_v2/ss64_seed42_v1/final_ema_val16_64_seed424344_v1/report.json`
- 评估语义：same-noise Stock/Native对比，frozen SS occupancy decoder，`condition_scale_policy=learned_projection_only`，没有post-CFG cap。

`S3_RC=2`表示评估正常完成但硬门失败，不是程序崩溃。

## 2. 总结结论

S3必须判定为失败，当前结果不能解锁T0的932对象正式训练，也不能进入SLAT/Mesh验证。

但是本次失败和此前“条件完全无效”的失败不同：Native SS已经学到有效的latent修正和相机位姿依赖，问题集中在这些latent改善没有稳定转化为frozen decoder下的occupancy改善。输出相对Stock明显收缩，precision略升，但recall、occupied count和IoU没有通过最终硬门。

此外，S2校准协议存在明确的渲染域构成偏置：16个校准对象全部来自旧Objaverse裸UUID渲染池，而S3还包含gap Objaverse、pilot Objaverse和Omni。CFG=5是在偏置的16对象小样本上按IoU均值选出的，三个候选的IoU置信区间均跨0且高度重叠，因此CFG=5的选择不够稳健。

## 3. S3核心指标

| 指标 | S3结果 | 解释 | 硬门 |
|---|---:|---|---|
| IoU gain mean | -0.005794 | Native平均低于Stock | 失败 |
| IoU对象胜率 | 0.458333 | 22/48对象优于Stock | 失败，要求至少0.5 |
| Precision gain mean | +0.004424 | 预测更保守后，precision略升 | 非主硬门，CI跨0 |
| Recall gain mean | -0.028054 | 漏掉更多GT occupancy | 失败 |
| Latent MSE gain mean | +0.052710 | Native latent比Stock更接近target latent | 通过 |
| Full/Stock count ratio mean | 0.806634 | Native occupied count只有Stock的80.66% | 失败，要求至少0.85 |
| Full-Stock count mean | -2630.42 | 每个对象平均少约2630个occupied voxels | 失败证据 |
| Correct over pose-control IoU | +0.004044 | 正确位姿优于循环错位姿 | 通过 |
| Disabled wrapper equivalence | max abs=0 | 关闭Native分支时严格等于Stock | 通过 |

关键置信区间：

- `recall_gain` 95% CI为`[-0.05144, -0.00490]`，完全低于0，recall退化具有统计证据。
- `latent_mse_gain` 95% CI为`[+0.03034, +0.08049]`，完全高于0，latent改善具有统计证据。
- `correct_over_pose_control_iou` 95% CI为`[+0.00075, +0.00804]`，完全高于0，投影条件确实使用了相机位姿。
- `iou_gain` 95% CI为`[-0.01800, +0.00622]`，跨0；仅凭IoU均值不能证明稳定改善或稳定退化，但它显然没有达到“优于Stock”的门槛。

## 4. 结果代表什么

### 4.1 3D condition和pose-control已经生效

正确位姿相对循环错位姿的IoU优势均值为`+0.004044`，对象胜率为`0.6875`，置信区间下界大于0。这说明共享crop/pad/resize、更新后的内参以及体素到图像投影不是完全失效状态，模型也不是简单忽略3D condition。

这只是必要条件，不是优于Stock的充分条件。正确位姿分支虽然优于错误位姿控制，但仍可能同时低于Stock。

### 4.2 latent改善没有转化为occupancy改善

`latent_mse_gain=+0.052710`且40/48对象为正，说明Native输出在连续latent空间中更接近训练target。与此同时，decoded occupancy平均减少19.34%，recall显著下降。

这说明当前flow-MSE方向与frozen occupancy decoder的离散决策边界没有完全对齐。模型可以降低全体latent元素的平均误差，同时把一部分靠近occupancy边界的体素推到empty侧，最终造成occupied count和recall下降。

decoder在评估中仍然冻结；本结果不能解释为“decoder被意外解冻”。当前证据也不足以支持立即解冻decoder。

### 4.3 precision上升主要来自更保守的occupancy

Precision均值增加`+0.004424`，但其95% CI跨0。结合count ratio仅`0.8066`和recall下降`-0.0281`，更合理的解释是Native预测变得更稀疏，减少了一部分false positive，同时删除了更多true positive。它不是整体重建质量改善。

### 4.4 不是legacy cap或Stock基线污染

- `post_cfg_cap=false`，本次没有使用旧Direct SLAT式post-CFG residual cap。
- 外部`condition_scale`已经不存在，注入幅度由训练得到的逐block projection决定。
- disabled wrapper与Stock的latent max abs为0，说明Stock对照没有被Native包装器污染。

因此，本次失败不能归因于旧cap仍在生效，也不能通过简单删除cap解决。

## 5. S2校准为何没有泛化到S3

### 5.1 三个CFG候选在S2上都勉强eligible

| CFG | IoU gain mean | IoU 95% CI | Recall gain mean | Latent MSE gain | Count ratio mean |
|---:|---:|---|---:|---:|---:|
| 1 | +0.000987 | [-0.00632, +0.00898] | +0.000297 | +0.013412 | 1.004557 |
| 3 | +0.003247 | [-0.00430, +0.01139] | +0.000494 | +0.029953 | 0.916647 |
| 5 | +0.006560 | [-0.00071, +0.01535] | +0.007154 | +0.040483 | 0.892668 |

当前选择规则在所有候选通过零阈值后，优先最大化IoU均值，因此选择了CFG=5。但三个IoU区间都跨0并高度重叠，没有统计证据表明CFG=5稳定优于CFG=1或3。

随着CFG从1增加到5，S2 count ratio从`1.0046`下降到`0.8927`，已经显示出更强CFG会推动occupancy收缩。当前gate只检查count ratio均值；CFG=5在S2的count ratio中位数实际只有`0.8074`，这个风险被均值掩盖了。

### 5.2 校准集不是分层抽样

val64按object UID排序后直接取`0:16`，导致S2的16个对象全部来自旧Objaverse裸UUID渲染池。其图像路径位于：

```text
/data/zjr/pixal3d_multiview/objaverse_sparse_mv_artraj_pbr_5000_v9_select8
```

S3的48个对象构成为：

- 旧Objaverse裸UUID渲染池：20个。
- `gap_objaverse288_cyclescuda_v1` Objaverse渲染池：10个。
- `pilot_objaverse256_cyclescuda_v2` Objaverse渲染池：9个。
- `pilot_omni256_cyclescuda_v4` Omni渲染池：9个。

也就是说，gap Objaverse、pilot Objaverse和Omni在S2中均为0个，却占S3的28/48。S2选择出的CFG=5对最终混合渲染域缺少代表性。

## 6. S3分组结果

以下分组先按`object_uid`前缀划分，再通过上游`image_paths`和`source_glb`追溯来源。裸UUID同样来自Objaverse，只是旧渲染池没有添加`objaverse_`前缀。

| UID组 | 对象数 | IoU gain mean | Recall gain mean | Latent MSE gain mean | Count ratio mean |
|---|---:|---:|---:|---:|---:|
| 旧Objaverse裸UUID | 20 | -0.001157 | -0.019473 | +0.065047 | 0.771718 |
| Gap Objaverse | 10 | -0.031548 | -0.095241 | +0.046795 | 0.750086 |
| Pilot Objaverse | 9 | -0.001355 | +0.006008 | -0.001210 | 1.023881 |
| Omni pilot | 9 | +0.008076 | -0.006535 | +0.085787 | 0.729810 |

结论：

- occupancy异常存在于四组，不是单一渲染域的孤立故障。
- Gap Objaverse组对总体IoU和recall失败贡献最大；相对旧Objaverse，其IoU差值为`-0.03039`、recall差值为`-0.07577`，两项bootstrap差值CI均完全低于0。
- Pilot Objaverse并没有表现出更差的decoded occupancy：IoU接近旧Objaverse、recall均值略正、count ratio接近1；但其latent MSE gain均值接近0，尚未证明latent改善。
- Omni pilot组虽然IoU均值为正，但count ratio最低，说明IoU小幅改善仍伴随明显欠覆盖。
- 旧Objaverse裸UUID组与S2属于同一渲染池，但在S3仍出现明显收缩，说明问题不只是跨渲染域shift；CFG选择的小样本过拟合同样重要。

## 7. 训练本身是否正常

训练工程合同正常：

- step 0的conditional/unconditional输出与Stock严格一致。
- 24个SS block均有condition projection。
- step500完成4000个micro steps，训练报告`passed=true`。
- step500 batch上的flow loss为`0.10019`，Stock loss为`0.10479`，训练目标上仍有正gain。
- EMA更新500次，最后decay为`0.98235`，不是停留在初始化权重。
- 梯度、参数、optimizer和EMA均通过finite检查。

因此没有证据表明S3失败来自训练未运行、projection未更新或EMA没有更新。当前缺失的是raw权重与EMA权重的同协议对比，以及不同checkpoint的occupancy轨迹。

## 8. 当前决策

1. 不执行T0，不把当前step500/CFG=5标记为通过。
2. 保留S2、S3全部输出；S3是有效的正式失败结果，不应删除或覆盖。
3. 不在同一S3对象上调参后再把它宣称为独立final test。S3已经被查看，只能用于诊断；后续正式结论需要新的未触碰holdout。
4. 暂不解冻decoder，不恢复外部condition scale，也不恢复post-CFG cap。这些都不是当前证据直接支持的首要修改。

## 9. 后续优先级

第一优先级是不重训的诊断：

- 在当前S3对象上仅作为诊断比较EMA的CFG=1/3/5，确认count收缩随CFG单调加重的程度。
- 用相同协议比较step500 raw与EMA，排除EMA平滑造成的occupancy偏差。
- 对step100/200/300/400/500做checkpoint轨迹，定位latent gain出现和count contraction开始的时间。

第二优先级是修正校准协议：

- development/calibration对象按旧Objaverse、gap Objaverse、pilot Objaverse、Omni分层抽样，而不是按排序后的前16个对象切片。
- eligibility除均值外加入recall对象胜率、count ratio中位数或低分位约束。
- CFG选择不再仅最大化小样本IoU均值；至少同时考虑count closeness和置信区间，避免CFG越高、latent越好但occupancy越收缩。

第三优先级才是决定是否重训：

- 如果低CFG或raw权重能显著缓解收缩，优先修正校准/权重选择，不先改模型。
- 如果所有CFG、raw/EMA和中间checkpoint都保持“latent改善、occupancy收缩”，再处理训练目标与frozen decoder边界不对齐的问题。
- 64对象只能用于机制smoke，不能证明混合来源泛化。扩大训练对象数可以作为独立开发实验，但在当前硬门语义下不能把未通过S3的结果直接升级为正式T0路线。

## 10. 最终判定

本次Native SS v2已经证明：共享几何投影有效、pose-control有效、latent优化有效、Stock bypass正确。尚未证明的是最终SS occupancy优于Stock。

当前最可能的直接问题是：偏置的小样本S2把CFG选择到5，放大了模型已有的occupancy收缩倾向；更深层问题是flow latent MSE与frozen decoder occupancy/recall之间仍存在目标错配。两者需要通过CFG/raw/EMA/checkpoint轨迹诊断分开，不能仅凭本次S3直接归因于decoder或投影代码错误。
