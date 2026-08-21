# ProObjaverse 官方 SLat Target：Train16 P7 双臂结果分析

日期：2026-08-13  
性质：训练集重叠的开发诊断，`formal=false`，不是泛化或端到端科学结论。

## 1. 核心结论

P7 正常完成，`P7_RC=0`。本轮首次在官方 ProObjaverse SLat target、GT support 和同坐标 Mesh 评测下确认：

1. posed-DINO condition-only 分支能够在 Train16 上超过冻结 Stock，说明条件注入模块并非完全无效；
2. posed-DINO condition + 小规模 SLat attention LoRA 进一步稳定超过 condition-only，并明显超过 Stock；
3. 因此当前选择应为 `condition_lora`，可以进入扩大训练；
4. 现在仍不能进入 predicted Native-SS support bridge，也不能声称测试集泛化或端到端 Mesh 提升。

这修正了此前“当前 SLat 结构或 loss 连 16 个训练对象都无法拟合”的怀疑。更准确的结论是：在官方 SLat target 与 GT support 合同下，当前训练目标是可学习的；之前长期无收益更可能同时受到 target/data contract、support 误差和仅 condition-only 适配能力不足等因素影响。

## 2. 实验合同

- 对象：16 个训练对象；`evaluation_split=train`，`training_overlap=true`。
- 每个对象使用种子 `42,43,44`，共 48 条采样记录。
- 权重：EMA。
- Support：官方 GT SLat support；本轮没有运行 Native SS，也没有消费预测 support。
- Baseline：相同坐标、相同初始噪声、冻结的 Native-SS + Stock-SLat 路径。
- Decoder：各分支使用相同的冻结 Stock Mesh decoder。
- 采样：25 steps，CFG=5，CFG interval `[0.5,1.0]`，`rescale_t=3`。
- 两个训练分支均为 200 optimizer steps，global effective batch 均为 8。A 使用 2 GPU，B 使用 4 GPU，但有效全局 batch 相同，因此 B 的提升不能简单归因于 batch 加倍。

## 3. 相对 Stock 的结果

正数表示优于 Stock；Chamfer 项使用的是 `chamfer_l1_improvement`，因此正数同样表示误差下降。

| 分支 | Chamfer mean / median / win | F-score@0.02 mean / win | Normal mean / win | LCR mean / win |
|---|---:|---:|---:|---:|
| A：condition-only | +0.00011997 / +0.00007386 / 75.00% | -0.00029590 / 37.50% | +0.00131691 / 62.50% | +0.04801088 / 56.25% |
| B：condition+LoRA | +0.00026358 / +0.00012249 / 93.75% | -0.00000655 / 50.00% | +0.00464768 / 81.25% | +0.03209855 / 56.25% |

关键置信区间：

- A 的 Chamfer 95% CI 为 `[-0.00002779, +0.00031218]`，仍跨 0。它通过预设 Stock gate，但证据偏弱。
- B 的 Chamfer 95% CI 为 `[+0.00010420, +0.00046282]`，完整高于 0。
- B 的 Normal 95% CI 为 `[+0.00135855, +0.00855196]`，完整高于 0。
- B 的 F-score 基本中性，不能说四项指标全部提升。
- LCR 的均值受到少数大幅正负样本影响，置信区间跨 0，尚不能形成连通性改善结论。

## 4. LoRA 相对 condition-only 的独立增量

下表为 B 减 A，而不是 B 相对 Stock：

| 指标 | mean | median | 正向对象率 | Bootstrap mean 95% CI |
|---|---:|---:|---:|---:|
| Chamfer improvement | +0.00014361 | +0.00007596 | 87.50% | `[+0.00004844, +0.00027434]` |
| F-score@0.02 delta | +0.00028935 | +0.00002500 | 56.25% | `[+0.00002447, +0.00074180]` |
| Normal consistency delta | +0.00333078 | +0.00272648 | 75.00% | `[+0.00144156, +0.00569827]` |
| Largest-component ratio delta | -0.01591233 | -0.00042432 | 37.50% | `[-0.06638835, +0.03708314]` |

Chamfer、F-score 和 Normal 的 B-A 均值置信区间都高于 0，说明小规模 SLat attention LoRA 确实提供了 condition-only 之外的增量。尤其 Chamfer 在 14/16 个对象上为正，Normal 在 12/16 个对象上为正。

LCR 则存在明显长尾：个别对象约为 `-0.156`、`-0.258`，也有对象约为 `+0.288`。因此 B 的几何表面拟合更好，但仍需在扩大训练中专门检查碎片、孔洞和最大连通分量，不能只看 Chamfer。

## 5. P7 门控判定

| 门 | 结果 |
|---|---|
| condition-only 相对 Stock | 通过 |
| condition+LoRA 相对 Stock | 通过 |
| condition+LoRA 相对 condition-only 的增量 | 通过 |
| 选中分支 | `condition_lora` |
| 是否建立 Train16 拟合优势 | 是 |
| 是否允许扩大训练 | 是 |
| 是否允许进入 predicted-support bridge | 否 |

需要注意，脚本中的 Stock gate 主要依据 Chamfer mean、median 和对象胜率；它不强制 bootstrap CI 下界大于 0。因此 A 虽然通过门，但统计证据不如 B。B 相对 Stock和 B-A 的 Chamfer CI 都高于 0，结论更可靠。

## 6. 本轮证明了什么

- 官方 SLat target 能被当前训练代码正确消费，并能迁移成冻结 decoder 下可测的 Mesh 改善。
- posed-DINO condition 在 SLat 阶段能发挥作用，不是完全失效的旁路。
- attention LoRA 不是冗余模块；在保持相同有效 batch 和相同评测合同后，它提供了明确增量。
- 当前 velocity-MSE 训练至少具备 Train16 拟合能力，暂时没有证据要求立刻推翻全部损失或 timestep 方案。

## 7. 本轮没有证明什么

- 没有证明 object-disjoint 泛化；所有 16 个对象均参与训练。
- 没有证明 Native SS 预测 support 下仍有收益。本轮使用的是 GT support。
- 没有证明真实采集、AR 世界坐标或端到端 Mesh 重建提升。
- 没有证明拓扑与连通性改善；LCR 仍不稳定。
- 没有证明扩大到约 2k 对象后仍能保持收益，也没有形成正式科学 claim。

## 8. 下一步建议

1. 冻结 B 的结构、loss、timestep、学习率和采样合同，使用 `condition_lora` 在约 2k 官方 target 数据上扩大训练。此时不要同时改低噪声课程、trust region 或 decoder loss，否则无法判断扩大数据本身的作用。
2. condition-only A 保留为固定消融，不再作为主训练路线。
3. 扩大训练后先做 object-disjoint、GT-support 的开发集评测，继续使用相同坐标、相同噪声和冻结 Stock decoder；只有该门通过，才说明存在泛化。
4. 评测除 Chamfer、F-score 和 Normal 外，必须增加碎片数、最大连通分量和失败对象列表，重点追踪本轮 LCR 长尾对象。
5. 只有 object-disjoint GT-support 通过后，才接入 Native SS predicted support，单独量化 GT-support 到 predicted-support 的性能损失；当前 `proceed_to_predicted_support_bridge=false` 应严格保留。
6. 如果扩大训练仍在训练集或开发集失效，再按顺序检查 target/cache 合同、低/高噪声分桶方向误差、Stock trust region 和 decoder-aware 目标，不要在本轮正向证据之前一次性加入这些改动。

## 9. 可追溯产物

- P7 决策报告：`/data/zjr/proobjaverse_official_slat_target2000_20260813_v1/train16_fit_two_arm_decision_v1/report.json`
- P7 report SHA256：`9f6e01306063f2eebeb4cc51f3378e3f660c392c5a4c15f6272b979fd61cdcd7`
- 官方协议 SHA256：`0578c85095e782d78f94b33ca556623e9ecd3edbc124af2bbb6506317823c557`
- A 评测报告：`/data/zjr/proobjaverse_official_slat_target2000_20260813_v1/eval_train16_A_condition_only_step200_seed424344_v1/report.json`
- A checkpoint：`/data/zjr/proobjaverse_official_slat_target2000_20260813_v1/A_condition_only_train16_step200_seed42_2gpu_v1/checkpoints/step_000200.pt`
- B 评测报告：`/data/zjr/proobjaverse_official_slat_target2000_20260813_v1/eval_train16_B_condition_lora_step200_seed424344_v1/report.json`
- B checkpoint：`/data/zjr/proobjaverse_official_slat_target2000_20260813_v1/B_condition_lora_train16_step200_seed42_4gpu_v1/checkpoints/step_000200.pt`

最终阶段判断：`condition_lora` 已通过 Train16 官方 target 拟合门，可以扩大训练；但当前结果必须保持为训练集开发诊断，不能提前表述为泛化、predicted-support 或端到端优势。
