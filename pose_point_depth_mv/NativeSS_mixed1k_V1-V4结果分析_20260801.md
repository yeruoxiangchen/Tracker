# Native SS mixed1k V1-V4结果分析

日期：2026-08-01 UTC

## 1. 总结论

本轮V1、V2、V3通过，V4未通过，因此按预注册协议不得进入V5，Native SLAT与Mesh仍保持
锁定。各阶段状态为：

```text
V1  PASS：在dev16锁定step2000 EMA
V2  PASS：在独立CFG16锁定CFG=5
V3  PASS：CFG16严格对象胜率/count-median审计通过
V4  FAIL：final32的Full/Stock count-ratio均值低于0.85
V5  NOT RUN
```

V4的形式失败只有一个：`count_ratio_lower=false`。IoU、precision、recall、latent MSE、
pose-control、Stock非空和disabled-wrapper等检查全部通过，而且前三项occupancy质量指标的
bootstrap 95% CI均完整高于0。

进一步核对target occupancy后，V4不支持“Native SS发生有害occupancy塌缩”这一解释。
Stock平均生成`1.708x target`的occupied voxels，Native SS降到`1.314x target`；32个对象中
有27个Native SS的count比Stock更接近target，只有2个Native SS低于target count。结合
precision gain `+0.2174`和recall gain `+0.2183`同时为正，更符合“模型删除了Stock的大量
false positive并重排了occupied support”，而不是单纯牺牲recall换取更小体积。

因此必须区分两层结论：

1. **协议结论**：V4失败，不能追认通过，也不能继续V5或解锁SLAT/Mesh。
2. **科学结论**：868-object长训得到的Native SS在未用于选择的final32上，对Stock形成了
   强且广泛的occupancy质量优势；失败暴露的是`Full/Stock count ratio >= 0.85`这一
   Stock-relative约束与target-relative质量之间可能存在语义错配。

## 2. 协议与数据隔离

| 阶段 | 对象 | Seeds | 用途 | 结果 |
|---|---:|---|---|---|
| V1 | dev16，四来源各4 | 42 | checkpoint/raw-EMA选择 | PASS |
| V2 | CFG16，四来源各4 | 42/43/44 | CFG=1/3/5选择 | PASS |
| V3 | 同CFG16 | 42/43/44 | 严格审计，不新增推理 | PASS |
| V4 | final32，四来源各8 | 42/43/44 | 一次性最终评估 | FAIL |

三个对象集合互不重叠；V4固定使用V1/V2已经锁定的参数，没有在final32上继续选择
checkpoint、权重或CFG。固定部署组合为：

```text
checkpoint:             step_002000.pt
checkpoint SHA256:      c87cc3b6581b3ba58786bdbe924883a4611557b4b7f21ab3a062abf85336f94a
weights:                EMA
CFG:                    5
condition scale:        learned_projection_only
post-CFG cap:           false
sampling steps:         25
```

## 3. V1：checkpoint与raw/EMA锁定

V1从V0的8个候选中找到4个严格合格候选：

| Step | 权重 | IoU gain | Recall gain | Count mean/median | 严格合格 |
|---:|---|---:|---:|---:|---|
| 500 | raw | +0.061712 | +0.099766 | 1.1937 / 1.0023 | YES |
| 1000 | raw | +0.114864 | +0.153955 | 1.1139 / 0.9449 | YES |
| 1500 | EMA | +0.131568 | +0.180120 | 1.0721 / 1.0185 | YES |
| 2000 | EMA | +0.149527 | +0.200053 | 1.1636 / 0.9415 | YES，选中 |

`step2000 raw`虽有最高IoU和recall，但count mean为`1.2973`，超过预注册上限`1.20`；
`step1500 raw`更高达`1.7357`。V1按“先过严格门，再按IoU/recall/latent/count closeness
/EMA/lower-step字典序排序”的规则正确选择`step2000 EMA`，没有为了追求最高IoU忽略
occupancy膨胀风险。

V1本身不产生新验证证据，其作用是把dev16上的开发选择固化到checkpoint哈希和EMA权重，
避免后续看过CFG16或final32后更换模型。

## 4. V2：独立CFG16校准

三个CFG均通过V2基础门，但随着CFG增强，IoU/recall/latent持续上升，Full/Stock count
ratio则持续下降：

| CFG | IoU gain/win | Recall gain/win | Latent gain/win | Count mean/median | 合格 |
|---:|---:|---:|---:|---:|---|
| 1 | +0.115281 / 0.8125 | +0.168176 / 0.8125 | +0.005640 / 0.5625 | 1.1547 / 1.1762 | YES |
| 3 | +0.139016 / 0.8125 | +0.207859 / 0.7500 | +0.025528 / 0.6875 | 0.9814 / 0.9437 | YES |
| 5 | +0.152249 / 0.8750 | +0.229743 / 0.8125 | +0.037963 / 0.7500 | 0.9324 / 0.8937 | YES，选中 |

V2按预注册的IoU优先字典序选择CFG=5是执行正确，不是运行错误。但数据已经出现明确预警：

- CFG从1升到5时，count mean从`1.1547`降到`0.9324`。
- CFG=5下，Gap Objaverse count mean为`0.8425`，Omni为`0.8325`，均已低于0.85；
  只是四来源总体均值仍通过。
- CFG=5下，4-view count median为`0.8490`，8-view为`0.8438`，都位于或略低于下界。

也就是说，V4的count收缩不是完全突发；V2的小样本全局聚合掩盖了来源级和view-count级的
边界风险。当前V2规则只要求总体count mean通过，随后V3要求总体median通过，没有要求每个
来源或每个view-count分层均通过。

## 5. V3：校准严格审计

V3没有运行新推理，只审计V2锁定的CFG=5报告，以下检查全部通过：

```text
base_calibration_passed        true
source_balanced_object_count   true
three_joint_seeds              true
iou_object_win                 true
recall_object_win              true
count_ratio_median_lower       true
count_ratio_median_upper       true
stock_nonempty                 true
```

V3通过是正确的：CFG16总体count median=`0.8937`，确实高于`0.85`。但其安全余量只有
`0.0437`，且分层风险未进入V3硬门，所以V3不能保证final32的count分布仍在边界内。

## 6. V4：final32总体结果

| 指标 | Mean | Median | Win | Bootstrap mean 95% CI | 检查 |
|---|---:|---:|---:|---:|---|
| IoU gain | +0.162281 | +0.126811 | 0.8750 | [+0.110815, +0.216465] | PASS |
| Precision gain | +0.217436 | +0.200894 | 0.8750 | [+0.155405, +0.281337] | PASS |
| Recall gain | +0.218296 | +0.203988 | 0.8125 | [+0.149761, +0.291073] | PASS |
| Latent MSE gain | +0.087999 | +0.086597 | 0.7813 | [+0.046092, +0.129244] | PASS |
| Correct over pose-control IoU | +0.164228 | +0.135890 | 0.9063 | [+0.112261, +0.219808] | PASS |

V4的count结果为：

```text
Full/Stock count ratio mean:    0.84597115   (< 0.85，FAIL)
Full/Stock count ratio median:  0.75537762
min / max:                      0.42457010 / 1.95585413
Full-Stock count mean:          -3706.23 voxels
Full-Stock count median:        -3399.33 voxels
ratio < 0.85:                   24/32 objects
ratio > 1.20:                    3/32 objects
Stock empty:                     0/96 records
```

均值只比基础门低`0.00403`，但中位数和24/32对象低于下界说明它不是舍入误差，也不是单个
极端对象拉低。即使跳过V4直接运行V5，V5仍会因`base_evaluator_passed=false`以及
`count_ratio_median=false`失败，因此V5不能修复或重新解释本次协议结果。

## 7. V2到V4的泛化变化

固定使用step2000 EMA、CFG=5时：

| 指标 | V2 CFG16 | V4 final32 | V4-V2 |
|---|---:|---:|---:|
| IoU gain | +0.152249 | +0.162281 | +0.010032 |
| Recall gain | +0.229743 | +0.218296 | -0.011447 |
| Latent MSE gain | +0.037963 | +0.087999 | +0.050036 |
| Count ratio mean | 0.932403 | 0.845971 | -0.086431 |
| Count ratio median | 0.893746 | 0.755378 | -0.138368 |

几何质量优势从CFG16泛化到final32，IoU甚至略有提高，latent改善也更强；没有泛化的是
Stock-relative count校准。这个组合说明模型不是整体失效，而是CFG=5在不同对象分布上对
occupied support大小的响应不够稳定。

## 8. V4分来源结果

每个来源固定8对象：

| 来源 | IoU gain/win | Recall gain/win | Latent gain | Count mean/median | ratio<0.85 | Full-Stock count mean |
|---|---:|---:|---:|---:|---:|---:|
| Legacy Objaverse | +0.080220 / 0.50 | +0.085931 / 0.50 | +0.035919 | 1.0104 / 0.8181 | 5/8 | -1284 |
| Gap Objaverse | +0.111093 / 1.00 | +0.158721 / 0.75 | +0.089789 | 0.7349 / 0.7168 | 7/8 | -5300 |
| Pilot Objaverse | +0.195903 / 1.00 | +0.282976 / 1.00 | +0.060479 | 0.9288 / 0.7994 | 5/8 | -3431 |
| Omni | +0.261910 / 1.00 | +0.345553 / 1.00 | +0.165811 | 0.7097 / 0.6789 | 7/8 | -4810 |

四个来源的平均IoU、recall和latent方向全部为正，质量优势并非由单一来源制造。count收缩
则跨来源存在，Gap和Omni最明显。Legacy/Pilot的count mean被少数膨胀对象抬高，其median
仍低于0.85，所以也不能认为这两个来源不存在收缩。

## 9. V4分视图数结果

| 视图数 | n | IoU gain/win | Recall gain/win | Latent gain | Count mean/median | ratio<0.85 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 11 | +0.145906 / 1.000 | +0.222595 / 1.000 | +0.010947 | 1.1064 / 0.9294 | 4/11 |
| 4 | 11 | +0.149520 / 0.818 | +0.187962 / 0.727 | +0.096613 | 0.7151 / 0.6967 | 10/11 |
| 8 | 10 | +0.194332 / 0.800 | +0.246933 / 0.700 | +0.163282 | 0.7034 / 0.7554 | 10/10 |

收缩主要集中在4/8-view：两组共21对象，其中20个ratio低于0.85。2-view总体count较接近
Stock，但latent改善最弱。一个可能机制是更多一致视图带来更强的投影条件，使CFG=5更有力
地删除Stock的宽松occupied support；这与4/8-view的IoU/latent改善更强一致，但当前数据只
能支持相关性，不能单独证明因果。

## 10. 相对target count的复核

`Full/Stock count ratio`衡量的是部署偏移，不直接衡量谁更接近target occupancy。使用V4
记录中的`coord_count / target_coord_count`重新聚合到对象级后：

| 预测 | Target ratio mean | Target ratio median | 对数比例绝对误差均值 |
|---|---:|---:|---:|
| Stock | 1.708174 | 1.733500 | 0.521468 |
| Native SS | 1.314321 | 1.323420 | 0.272504 |

Native SS将target-count对数误差降低约47.7%。此外：

```text
Native SS count比Stock更接近target： 27/32 objects
Native SS低于target count：             2/32 objects
Stock低于target count：                 2/32 objects
```

分来源上，Stock/Native SS的target ratio mean分别为：

| 来源 | Stock/target | Native SS/target |
|---|---:|---:|
| Legacy Objaverse | 1.4824 | 1.3147 |
| Gap Objaverse | 1.9113 | 1.2744 |
| Pilot Objaverse | 1.7113 | 1.4624 |
| Omni | 1.7277 | 1.2057 |

这解释了为什么Full/Stock ratio下降的同时precision和recall仍能共同上升：Stock在final32
上通常显著过占用，Native SS主要删除错误占用并改善support位置，且输出平均仍比target大
约31%。所以“相对Stock少15%以上”在本批数据上不等价于“相对真值过度收缩”。

该复核不能改变预注册V4的PASS/FAIL，因为target-relative规则是在看到final结果后才提出；
它只能用于诊断下一版协议，不能用于追认当前实验通过。

## 11. 当前判断与下一步约束

1. 不运行V5，不解锁Native SLAT和Mesh；本轮正式状态保持V4 FAIL。
2. 保存V1选择、V2校准、V3审计和V4 report，不在final32上回选CFG=1/3或修改count门后
   重报同一final32为confirmatory PASS。
3. 下一版协议应在消费新holdout前预注册target-relative count门，例如同时报告
   `Full/target`、`Stock/target`和对象级target-count误差改善；Full/Stock ratio可保留为
   deployment shift诊断，但不应单独否决precision、recall、IoU和target count均改善的模型。
4. CFG选择需要加入count安全余量或来源/view分层约束。V2已经显示CFG越高、count越低；只按
   IoU优先会稳定偏向最强CFG，并把分层收缩风险推迟到final才暴露。
5. 由于final32已经消费，任何修改门禁、改选CFG或重训后的正式结论都需要新的未触碰
   source-balanced holdout；在原final32上的后续比较只能标记为开发或事后分析。

## 12. 证据路径

```text
V1 coarse selection:
/data/zjr/native3d_condition_ss_mixed1k_20260801_v1/
  ss868_sourceholdout_seed42_v1/checkpoint_weight_selection_dev16_v1/
  coarse_selection.json

V2 calibration:
/data/zjr/native3d_condition_ss_mixed1k_20260801_v1/
  ss868_sourceholdout_seed42_v1/cfg_calibration_sourcebalanced16_seed424344_v1/
  calibration.json

V3 strict audit:
/data/zjr/native3d_condition_ss_mixed1k_20260801_v1/
  ss868_sourceholdout_seed42_v1/cfg_calibration_sourcebalanced16_seed424344_v1/
  strict_audit.json

V4 final report:
/data/zjr/native3d_condition_ss_mixed1k_20260801_v1/
  ss868_sourceholdout_seed42_v1/final_sourcebalanced32_seed424344_v1/report.json

Frozen split audit:
/data/zjr/native3d_condition_reviewed1k_inputs_20260731_v4/
  native_ss_sourcebalanced_split_seed20260801_v1/split_audit.json
```
