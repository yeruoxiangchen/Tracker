# Latent 拼接 Sanity 结果分析

时间：2026-06-26

本文记录 `trellis_point_prior_mv` 从 coords-level point prior 转向 latent inpainting 前的两步 sanity：

```text
Step 1:
  构建 latent inpaint manifest，保存 q_gt / q_vis / m_s。

Step 1.5:
  q_splice = m_s * q_vis + (1 - m_s) * q_gt
  D_s(q_splice) -> sparse coords
```

运行目录：

```text
trellis_point_prior_mv/outputs/latent_inpaint/latent_inpaint_smoke
trellis_point_prior_mv/outputs/latent_inpaint/latent_inpaint_val8
trellis_point_prior_mv/outputs/latent_splice_sanity/latent_splice_sanity_smoke
trellis_point_prior_mv/outputs/latent_splice_sanity/latent_splice_sanity_val8
```

## 1. Latent 数据构建情况

| split | samples | prior points mean | target points mean | mask cell mean |
|---|---:|---:|---:|---:|
| smoke | 1 | 1500.0 | 23414.0 | 559.0 |
| val8 | 8 | 1500.0 | 9776.4 | 326.5 |

val8 中 `m_s` 平均只覆盖 326.5 个 16^3 latent cell，约 8.0%。这说明当前 point prior 在 latent 空间是一个较小的 observed anchor，而不是大面积覆盖完整物体。

`saved_m_s_vs_recomputed_mask_l1` 结果：

```text
d64=0, d16=0:
  smoke: 0
  val8:  0
```

因此 `build_latent_inpaint_dataset.py` 保存的 `m_s` 与 `eval_latent_splice_sanity.py` 重算的 base mask 完全对齐。当前没有 mask 构造不一致的问题。

## 2. Val8 主要结果

以下表格均看 `topk_target_unique`，即预测点数与 target sparse 点数一致。

| source | IoU | recall | precision | component | largest ratio | small64 ratio | mask cells | saved m_s L1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| q_gt | 0.9999 | 1.0000 | 1.0000 | 1.0 | 1.000 | 0.000 | - | - |
| q_vis | 0.4275 | 0.5965 | 0.5965 | 68.8 | 0.896 | 0.0267 | - | - |
| q_splice d64=0 d16=0 | 0.6336 | 0.7734 | 0.7734 | 28.6 | 0.991 | 0.0055 | 326.5 | 0 |
| q_splice d64=1 d16=0 | 0.6739 | 0.7980 | 0.7980 | 19.3 | 0.988 | 0.0073 | 564.8 | 0.0582 |
| q_splice d64=2 d16=0 | 0.6094 | 0.7494 | 0.7494 | 35.5 | 0.976 | 0.0114 | 705.5 | 0.0925 |
| q_splice d64=0 d16=1 | 0.5131 | 0.6732 | 0.6732 | 65.0 | 0.828 | 0.0220 | 1115.1 | 0.1925 |
| q_splice d64=1 d16=1 | 0.4888 | 0.6521 | 0.6521 | 67.3 | 0.842 | 0.0219 | 1343.0 | 0.2482 |
| q_splice d64=2 d16=1 | 0.4786 | 0.6432 | 0.6432 | 72.9 | 0.853 | 0.0221 | 1464.1 | 0.2777 |

结论：

1. `q_gt` decode 几乎完美，说明 TRELLIS sparse decoder 与保存的 target latent 是一致的。
2. `q_vis` 单独 decode 明显不够：IoU 只有 0.4275，component 约 68.8，说明只把 sparse point prior encode 成 latent，不能直接得到完整结构。
3. `q_splice` 明显好于 `q_vis`：`d64=0,d16=0` 把 IoU 从 0.4275 提到 0.6336，component 从 68.8 降到 28.6，largest ratio 从 0.896 提到 0.991。
4. `d64=1,d16=0` 在 val8 上最好：IoU 0.6739，recall/precision 0.7980，component 19.3。
5. `d16=1` 明显过强。它把 known mask 扩到 27%-36% latent cell，覆盖了太多 unknown region，导致 IoU、largest ratio 和 component 都变差。

## 3. Smoke 与 Val8 的差异

smoke 单样本上，`d64=0,d16=0` 最好：

```text
q_vis IoU:                 0.3342
q_splice d64=0,d16=0 IoU:  0.5640
q_splice d64=1,d16=0 IoU:  0.4902
```

val8 上，`d64=1,d16=0` 最好：

```text
q_vis IoU:                 0.4275
q_splice d64=0,d16=0 IoU:  0.6336
q_splice d64=1,d16=0 IoU:  0.6739
```

这说明 `d64=1` 可能能修补稀疏 point prior 的 latent 空洞，但不是所有样本都稳定更好。因此下一步不应直接固定 `d64=1` 长训，应在 val32 上再确认。

## 4. 关键解释

这轮结果不是最终模型效果，因为 `q_splice` 的 unknown 区域直接用了 `q_gt`：

```text
q_splice = m_s * q_vis + (1 - m_s) * q_gt
```

因此它验证的是：

```text
1. q_vis 是否能作为 observed latent anchor；
2. m_s 边界是否合理；
3. q_vis 和 q_gt 能否在 latent 空间拼接后仍被 decoder 正常解码。
```

它不证明模型已经能从 `q_vis` 生成 unknown latent。但它已经证明一件重要的事：

```text
直接转 latent inpainting 是有意义的。
```

如果 `q_splice` 都碎，下一步训练 G_inp 大概率会失败；但现在 `q_splice` 明显优于 `q_vis`，并且拓扑指标明显改善，说明 observed latent anchor 与 target latent 不是完全不兼容。

## 5. 当前判断

当前结果支持继续推进 latent inpainting，但不支持直接训练 slat flow。

原因：

1. sparse decoder / target latent 没问题，`q_gt` 几乎完美。
2. q_vis 本身不够完整，直接作为完整 sparse latent 不现实。
3. q_splice 显著改善，说明 mask-based latent splice 的边界不是坏的。
4. 但 q_splice 仍明显低于 q_gt，尤其在固定 8192 / 4096 top-k 下还有差距，说明下一步训练需要专门学习 unknown region 的补全，而不是继续改 coords 后处理。

## 6. 下一步建议

### 第一优先级：补 val32 latent splice

先扩大到 32 个样本，只测有价值的 mask：

```text
mask_dilate64 = 0,1,2
mask_dilate16 = 0
```

不要再把 `d16=1` 作为默认候选。它在当前结果里过度扩张 observed region。

命令：

```bash
cd /home/zjr/Tracker

GPU=4 \
MODE=val32 \
RUN_NAME=latent_inpaint_val32 \
POINT_RUN_ROOT=/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_rank_w0005_ws05_s200_seed42 \
bash trellis_point_prior_mv/scripts/run_build_latent_inpaint_dataset.sh
```

```bash
cd /home/zjr/Tracker

GPU=4 \
MODE=val32 \
RUN_NAME=latent_splice_sanity_val32_d16off \
LATENT_RUN_ROOT=/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint/latent_inpaint_val32 \
MASK_DILATE64=0,1,2 \
MASK_DILATE16=0 \
TOPK_SPECS=4096,8192,target_unique \
THRESHOLD=0.0 \
bash trellis_point_prior_mv/scripts/run_latent_splice_sanity.sh
```

判断：

```text
如果 val32 仍是 d64=1,d16=0 最好：
  第一版 latent inpainting 训练用 d64=1,d16=0。

如果 d64=0,d16=0 与 d64=1,d16=0 接近：
  第一版训练用 d64=0,d16=0，更保守，避免 known region 泄漏过多。

如果 d64=2,d16=0 明显更差：
  不再扩大 64^3 mask。
```

### 第二优先级：实现 masked latent inpainting 训练

训练目标应从 coords sparse flow 转为 latent 补全：

```text
输入:
  q_t
  q_vis
  m_s
  image condition

目标:
  预测 q_gt 或 flow velocity

重点:
  unknown region loss 为主；
  known region 只做轻量一致性约束；
  不要再让模型在全局 64^3 coords 空间自由撒点。
```

第一版建议：

```text
KNOWN / OBSERVED:
  使用 m_s，对 known region 提供 q_vis anchor。

LOSS:
  unknown latent flow loss: 主损失
  known latent consistency loss: 小权重
  boundary band loss: 可选，先不加或小权重

MASK:
  先跑 d64=0,d16=0 与 d64=1,d16=0 两个版本。
```

### 第三优先级：训练后 eval

训练后先不要直接看 mesh，先看 sparse latent decode：

```text
1. decoded_vs_target IoU / recall / precision；
2. component count；
3. largest component ratio；
4. small64 ratio；
5. q_pred known/unknown region L1；
6. 再接 frozen stock slat/mesh。
```

只有 sparse latent decode 明显优于当前 q_vis，并接近 q_splice，才值得进入 mesh eval 和后续 slat flow。

## 7. 总结

这轮 sanity 的结论是正向的：

```text
q_vis 单独不够；
q_splice 明显有效；
m_s 构造一致；
d16=1 过度扩张；
latent inpainting 值得继续。
```

下一步应先补 val32，确认 `d64=0` 和 `d64=1` 的稳定性，然后实现真正的 masked latent inpainting 训练。当前不建议继续在 coords 输出层做更多后处理，也不建议现在训练 slat flow。
