# Latent Val32 与 Masked Latent Inpaint 实现报告

时间：2026-06-26

本文记录 `latent_splice_sanity_val32_d16off` 的结果分析，以及本轮新增的 masked latent inpainting flow 训练和评估代码。

## 1. Val32 Latent Splice 结果

运行目录：

```text
trellis_point_prior_mv/outputs/latent_inpaint/latent_inpaint_val32
trellis_point_prior_mv/outputs/latent_splice_sanity/latent_splice_sanity_val32_d16off
```

本轮只测：

```text
MASK_DILATE64=0,1,2
MASK_DILATE16=0
```

这是因为 val8 已经显示 `d16=1` 会过度扩张 known region。

## 2. 关键指标

以下均看 `topk_target_unique`。

| source | IoU | recall | precision | component | largest ratio | small64 ratio | mask cell |
|---|---:|---:|---:|---:|---:|---:|---:|
| q_gt | 0.99996 | 0.99998 | 0.99998 | 2.81 | 0.9985 | 0.0009 | - |
| q_vis | 0.45639 | 0.62059 | 0.62059 | 51.00 | 0.9462 | 0.0179 | - |
| q_splice d64=0,d16=0 | 0.66385 | 0.79293 | 0.79293 | 24.66 | 0.9850 | 0.0058 | 330.69 |
| q_splice d64=1,d16=0 | 0.64274 | 0.77577 | 0.77577 | 18.91 | 0.9884 | 0.0058 | 545.94 |
| q_splice d64=2,d16=0 | 0.58411 | 0.73017 | 0.73017 | 36.72 | 0.9779 | 0.0112 | 728.00 |

结论：

1. `q_gt` decode 基本完美，target latent / decoder 一致。
2. `q_vis` 单独作为完整 latent 仍然不够，IoU 只有 0.456。
3. `q_splice` 大幅优于 `q_vis`，说明 latent splice / masked latent inpainting 方向成立。
4. val32 上 `d64=0,d16=0` 的 IoU 和 recall 最好。
5. `d64=1,d16=0` component 更少、largest ratio 略高，但 IoU / recall 低于 `d64=0`。
6. `d64=2,d16=0` 明显开始过度扩张，IoU、recall 和 small component 都变差。

## 3. 与 Val8 的差异

val8 上 `d64=1,d16=0` 最好；val32 上 `d64=0,d16=0` 最好。

这说明：

```text
d64=1 可以补一部分 latent 空洞，但不是稳定收益；
d64=0 更保守，泛化上更可靠；
d64=2 已经不适合作为第一版训练默认值。
```

因此第一版训练默认：

```text
MASK_DILATE64=0
MASK_DILATE16=0
```

`d64=1,d16=0` 保留为对照实验，用于观察 component / IoU tradeoff。

## 4. 本轮新增代码

新增：

```text
trellis_point_prior_mv/train_latent_inpaint_flow.py
trellis_point_prior_mv/eval_latent_inpaint_flow.py
trellis_point_prior_mv/scripts/run_latent_inpaint_flow.sh
```

修改：

```text
trellis_point_prior_mv/scripts/run_build_latent_inpaint_dataset.sh
```

修改点：

```text
新增 MODE=train64:
  默认读取 POINT_RUN_ROOT/data/train/manifest.json
  输出 latent_inpaint_train64

新增 MODE=val64:
  方便后续扩大验证
```

## 5. 当前训练设计

训练不再在 64^3 coords 空间直接输出 sparse coords，而是在 16^3 sparse latent 上做 masked inpainting。

训练目标：

```text
target = (1 - m_s) * q_gt + m_s * q_vis
```

含义：

```text
known / observed region:
  保持 q_vis anchor。

unknown region:
  学 q_gt。
```

loss：

```text
unknown_flow_loss:
  主损失，默认权重 1.0。

known_flow_loss:
  轻量 known consistency，默认权重 0.25。

unknown_x0_loss:
  辅助 x0 约束，默认权重 0.25。

known_x0_loss:
  轻量 known x0 约束，默认权重 0.10。
```

condition：

```text
cond = SparsePointPriorCond(q_vis * m_s, m_s, confidence=m_s)
```

第一版暂时不混入图像 condition，目的是先验证 latent inpainting 本身是否能从 `q_vis/m_s` 补全 unknown sparse latent。

## 6. Eval 设计

`eval_latent_inpaint_flow.py` 会同时解码：

```text
q_gt
q_vis
q_splice
q_pred
```

重点比较：

```text
q_pred vs q_vis:
  是否真的学到了 unknown 补全。

q_pred vs q_splice:
  是否接近 oracle splice 上限。

q_pred component / largest / small64:
  是否重新引入碎片。

q_pred_unknown_vs_q_gt_l1:
  unknown region 是否接近 q_gt。

q_pred_known_vs_q_vis_l1:
  known region 是否保持 q_vis anchor。
```

当前仍然不建议直接以 mesh 为第一评估标准。只有 sparse latent decode 过关，才接 frozen stock slat/mesh。

## 7. 推荐命令

详细命令已写入：

```text
trellis_point_prior_mv/命令说明.txt
```

章节：

```text
五十、2026-06-26 Masked latent inpainting flow 训练与测试命令
```

第一步先跑：

```bash
cd /home/zjr/Tracker

GPU=4 \
MODE=smoke \
RUN_NAME=latent_inpaint_flow_d64_0_smoke \
RUN_BUILD=0 \
RUN_TRAIN=1 \
RUN_EVAL=1 \
MASK_DILATE64=0 \
MASK_DILATE16=0 \
MAX_STEPS=20 \
SAVE_EVERY=20 \
EVAL_STEPS=12 \
bash trellis_point_prior_mv/scripts/run_latent_inpaint_flow.sh
```

如果 smoke 正常，再跑 s200：

```bash
cd /home/zjr/Tracker

GPU=4 \
MODE=s200 \
RUN_NAME=latent_inpaint_flow_d64_0_s200 \
RUN_BUILD=0 \
RUN_TRAIN=1 \
RUN_EVAL=1 \
TRAIN_LATENT_ROOT=/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint/latent_inpaint_train64 \
VAL_LATENT_ROOT=/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint/latent_inpaint_val32 \
MASK_DILATE64=0 \
MASK_DILATE16=0 \
MAX_STEPS=200 \
SAVE_EVERY=100 \
EVAL_STEPS=12 \
bash trellis_point_prior_mv/scripts/run_latent_inpaint_flow.sh
```

## 8. 下一步判断标准

训练后先看 `q_pred/topk_target_unique`：

```text
最低要求:
  q_pred 明显优于 q_vis。

理想情况:
  q_pred 接近 q_splice。

风险信号:
  q_pred IoU 提升但 component 明显升高；
  q_pred known_vs_q_vis_l1 很大；
  q_pred unknown_vs_q_gt_l1 没有下降；
  q_pred 不如 q_vis。
```

如果 smoke 或 s200 出现：

```text
q_pred <= q_vis
```

优先检查：

```text
1. known clamp 是否过强或过弱；
2. known / unknown loss 权重；
3. cond 是否应该用 full q_vis 而不是 q_vis * m_s；
4. 是否需要降低 LoRA 学习率；
5. 是否需要加入 image condition。
```

如果：

```text
q_pred > q_vis 且接近 q_splice
```

下一步再接：

```text
1. frozen stock slat/mesh eval；
2. 真实 AR session latent prior eval；
3. 再决定是否训练 slat flow。
```

## 9. 当前结论

现在可以进入 masked latent inpainting 的 smoke 训练，但不建议直接长训 slat flow。

理由：

```text
q_splice 已经证明 latent anchor 可用；
val32 支持保守 mask；
训练目标已经从 coords 撒点转向 latent unknown 补全；
但 q_pred 是否能接近 q_splice 还没有验证。
```

因此最合理路线是：

```text
d64=0 smoke -> d64=0 s200 -> sparse latent decode eval -> frozen mesh eval
```

`d64=1` 只作为 ablation，不作为第一版默认。

## 10. 运行结果补充：d64=0 smoke / d64=1 smoke / d64=0 s200

已完成运行：

```text
trellis_point_prior_mv/outputs/latent_inpaint_flow/latent_inpaint_flow_d64_0_smoke
trellis_point_prior_mv/outputs/latent_inpaint_flow/latent_inpaint_flow_d64_1_smoke
trellis_point_prior_mv/outputs/latent_inpaint_flow/latent_inpaint_flow_d64_0_s200
```

### 10.1 Sparse decode 指标

以下均看 `topk_target_unique`。

| run | source | IoU | recall | precision | component | largest ratio | small64 |
|---|---|---:|---:|---:|---:|---:|---:|
| d64=0 smoke | q_vis | 0.3342 | 0.5009 | 0.5009 | 263.0 | 0.7360 | 0.0617 |
| d64=0 smoke | q_splice | 0.5640 | 0.7212 | 0.7212 | 68.0 | 0.9914 | 0.0086 |
| d64=0 smoke | q_pred | 0.1826 | 0.3087 | 0.3087 | 638.0 | 0.9060 | 0.0843 |
| d64=1 smoke | q_vis | 0.3342 | 0.5009 | 0.5009 | 263.0 | 0.7360 | 0.0617 |
| d64=1 smoke | q_splice | 0.4902 | 0.6579 | 0.6579 | 110.0 | 0.9556 | 0.0236 |
| d64=1 smoke | q_pred | 0.3366 | 0.5037 | 0.5037 | 236.0 | 0.8272 | 0.0532 |
| d64=0 s200 | q_vis | 0.4564 | 0.6206 | 0.6206 | 51.0 | 0.9462 | 0.0179 |
| d64=0 s200 | q_splice | 0.6638 | 0.7929 | 0.7929 | 24.7 | 0.9850 | 0.0058 |
| d64=0 s200 | q_pred | 0.3561 | 0.5170 | 0.5170 | 145.3 | 0.8583 | 0.0435 |

结论：

```text
d64=0 s200 的 q_pred 没有达到最低要求。
它低于 q_vis，更远低于 q_splice，因此不能进入 mesh eval。
```

### 10.2 Latent L1 指标

d64=0 s200 的 latent L1 有改善：

```text
q_vis_unknown_vs_q_gt_l1:  0.4221
q_pred_unknown_vs_q_gt_l1: 0.2123
q_splice_unknown_vs_q_gt_l1: 0.0000
```

但 sparse decode 变差。这说明模型并非完全没学，而是：

```text
q_pred 在 L1 上接近了 q_gt；
但 q_pred 没有落到 sparse decoder 期望的可解码 latent manifold 上；
或者 logits / top-k ranking 校准变差。
```

### 10.3 Latent 分布

d64=0 s200 的 latent 幅值明显塌缩：

| source | mean | std | abs mean |
|---|---:|---:|---:|
| q_gt | 0.0107 | 0.4416 | 0.2438 |
| q_vis | 0.0639 | 0.5949 | 0.3972 |
| q_splice | 0.0123 | 0.4335 | 0.2390 |
| q_pred | -0.0113 | 0.2938 | 0.1195 |

这解释了为什么 threshold decode 极差：

```text
q_pred/threshold_0:
  IoU=0.1349
  unique=871.8
  component=188.9
```

q_pred 的 latent std 和 abs mean 都明显低于 q_gt/q_splice，说明 MSE 训练倾向于输出偏平均、低幅值的 latent。该 latent 在 L1 上更接近 q_gt，但解码成 sparse structure 时 ranking/occupancy 不成立。

### 10.4 当前判断

当前 masked latent inpainting 的第一版训练结论是：

```text
LoRA flow + q_vis/m_s condition 可以降低 latent L1；
但不能直接得到更好的 sparse coords；
当前版本不能进入 mesh/slat 评估。
```

这回答了前面的 ablation 目标：

```text
仅靠 q_vis/m_s，不加 image condition，暂时不足以稳定补出可解码的 unknown sparse latent。
```

## 11. 下一步建议

### 第一优先级：eval-time clamp / condition 诊断

先不重训，用 d64=0 s200 checkpoint 做 eval sweep，判断是否是 known clamp 或 condition 形式导致 decoder manifold 破坏。

建议测试：

```text
1. known_latent_clamp_strength = 0.0 / 0.25 / 0.5 / 1.0
2. cond_use_full_q_vis = 1
```

如果弱 clamp 或 full q_vis condition 能让 q_pred sparse decode 超过 q_vis，说明采样/condition 还有救。

如果都不行，说明训练目标本身需要改。

### 第二优先级：加入 latent distribution / decoder-aware 约束

当前最大问题是 q_pred 幅值塌缩。下一版训练应加入至少一种约束：

```text
1. unknown latent std/abs matching loss；
2. decoder logits sparse loss，低频使用，避免每步太重；
3. q_pred threshold/top-k 相关的 occupancy regularization；
4. x0 loss 权重上调，但需防止已知区域过拟合。
```

最直接的 smoke 是：

```text
UNKNOWN_X0_LOSS_WEIGHT 从 0.25 提到 1.0
KNOWN_X0_LOSS_WEIGHT 保持 0.10
```

但这仍可能只改善 L1，不保证 decode ranking。

### 第三优先级：加入 image condition

如果 clamp / full q_vis / distribution regularization 都无法让 q_pred 超过 q_vis，那么应接入 image condition。

原因：

```text
q_vis/m_s 只给稀疏 observed anchor；
unknown region 的完整结构有多解；
不加图像时，flow 容易学平均 latent，而不是样本级几何。
```

此时下一版 condition 应变成：

```text
condition = image condition + q_vis/m_s condition
```

而不是继续只靠点云先验。

## 12. 结论更新

当前不建议继续 d64=0 latent inpaint 长训，也不建议进入 mesh/slat。

优先路线：

```text
1. 对现有 d64=0 s200 做 eval-time clamp/full-q_vis 诊断；
2. 若仍不超过 q_vis，则修改训练目标，加入 distribution / decoder-aware loss；
3. 再考虑接 image condition。
```

当前最重要的判断标准仍是：

```text
q_pred/topk_target_unique 必须先超过 q_vis。
```

## 13. Strict clamp 诊断补充

用户已运行：

```bash
RUN_TRAIN=0 \
RUN_EVAL=1 \
RUN_NAME=latent_inpaint_flow_d64_0_s200 \
MASK_DILATE64=0 \
MASK_DILATE16=0 \
KNOWN_CLAMP_START_T=1.0 \
CLAMP_INITIAL_NOISE=1 \
KNOWN_LATENT_CLAMP_STRENGTH=1.0 \
EVAL_STEPS=25 \
bash trellis_point_prior_mv/scripts/run_latent_inpaint_flow.sh
```

需要注意：这条命令没有显式设置 `MODE=s200`，所以 wrapper 使用默认 `MODE=smoke`。实际评估是：

```text
s200 checkpoint
latent_inpaint_smoke manifest
indices=0
steps=25
known_clamp_start_t=1.0
clamp_initial_noise=True
```

也就是说，这是 sample 0 的 strict clamp 诊断，不是 val32 完整评估。

sample 0 结果：

| source | IoU | recall | precision | component | largest | small64 |
|---|---:|---:|---:|---:|---:|---:|
| q_vis | 0.3342 | 0.5009 | 0.5009 | 263 | 0.7360 | 0.0617 |
| q_splice | 0.5640 | 0.7212 | 0.7212 | 68 | 0.9914 | 0.0086 |
| q_pred strict clamp | 0.2644 | 0.4183 | 0.4183 | 364 | 0.7871 | 0.0474 |

结论：

```text
强制从初始噪声开始 clamp known region，并增加到 25 steps，仍不能让 q_pred 超过 q_vis。
```

latent 分布仍然偏塌缩：

| source | std | abs mean |
|---|---:|---:|
| q_gt | 0.4416 | 0.2438 |
| q_splice | 0.4335 | 0.2390 |
| q_pred strict clamp | 0.2953 | 0.1209 |

因此问题不只是 known region 没 clamp 住；更可能是 unknown latent 的多步采样或训练目标使输出落到低幅值、不可稳定解码的区域。

后续重跑 strict clamp 时应使用独立 `EVAL_DIR`，避免覆盖原 eval：

```bash
cd /home/zjr/Tracker

GPU=4 \
MODE=s200 \
RUN_TRAIN=0 \
RUN_EVAL=1 \
RUN_NAME=latent_inpaint_flow_d64_0_s200 \
EVAL_DIR=/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint_flow/latent_inpaint_flow_d64_0_s200/eval_strict_clamp_s25_val32 \
TRAIN_LATENT_ROOT=/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint/latent_inpaint_train64 \
VAL_LATENT_ROOT=/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint/latent_inpaint_val32 \
MASK_DILATE64=0 \
MASK_DILATE16=0 \
KNOWN_CLAMP_START_T=1.0 \
CLAMP_INITIAL_NOISE=1 \
KNOWN_LATENT_CLAMP_STRENGTH=1.0 \
EVAL_STEPS=25 \
bash trellis_point_prior_mv/scripts/run_latent_inpaint_flow.sh
```

## 14. 新增 Teacher-forced x0 诊断

当前还没分清：

```text
A. 单步预测 pred_v -> pred_x0 是可以的，但多步 sampling 坏了；
B. 单步预测也不行，训练目标 / condition 本身没学对。
```

因此新增：

```text
trellis_point_prior_mv/eval_latent_inpaint_teacher_forced.py
trellis_point_prior_mv/scripts/run_latent_inpaint_teacher_forced.sh
```

诊断逻辑：

```text
给定 q_splice target；
采样 t；
构造 x_t；
模型预测 pred_v；
转成 pred_x0；
hard splice known region；
D_s(pred_x0) -> sparse coords。
```

默认测试：

```text
t = 0.25, 0.5, 0.75
```

命令：

```bash
cd /home/zjr/Tracker

GPU=4 \
RUN_NAME=latent_inpaint_flow_d64_0_s200_teacher_forced_val32 \
LATENT_RUN_ROOT=/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint/latent_inpaint_val32 \
POINT_RUN_ROOT=/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint_flow/latent_inpaint_flow_d64_0_s200 \
INDICES=0-31 \
MASK_DILATE64=0 \
MASK_DILATE16=0 \
T_VALUES=0.25,0.5,0.75 \
bash trellis_point_prior_mv/scripts/run_latent_inpaint_teacher_forced.sh
```

full q_vis condition 对照：

```bash
cd /home/zjr/Tracker

GPU=4 \
RUN_NAME=latent_inpaint_flow_d64_0_s200_teacher_forced_fullqvis_val32 \
LATENT_RUN_ROOT=/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint/latent_inpaint_val32 \
POINT_RUN_ROOT=/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint_flow/latent_inpaint_flow_d64_0_s200 \
INDICES=0-31 \
MASK_DILATE64=0 \
MASK_DILATE16=0 \
T_VALUES=0.25,0.5,0.75 \
COND_USE_FULL_Q_VIS=1 \
bash trellis_point_prior_mv/scripts/run_latent_inpaint_teacher_forced.sh
```

判断：

```text
teacher-forced 好，sampling 差：
  优先修 sampling / clamp / steps。

teacher-forced 也差：
  优先修训练目标 / condition。
```

## 15. Image condition 路线提前

当前 latent inpaint 的 condition 是：

```text
SparsePointPriorCond(q_vis, m_s, confidence)
```

它没有融合 TRELLIS 原始 image condition。这与 Points-to-3D 的理论设定有差距。

更合理的第二阶段路线应是：

```text
condition = image_cond + point_cond(q_vis, m_s)
```

原因：

```text
image condition:
  提供语义、类别、全局形状和不可见区域先验。

q_vis / m_s:
  提供 observed geometry anchor。
```

因此如果 teacher-forced 诊断也不超过 q_vis，下一步优先接 image condition，而不是先大量加入 decoder-aware loss。

更新后的优先级：

```text
1. teacher-forced x0 eval；
2. 若 sampling 问题，修采样；
3. 若单步也差，优先接 image condition；
4. 再考虑 decoder-aware / distribution loss。
```

## 16. 2026-06-26 Teacher-forced 结果与 Image Condition 修改

### 16.1 Teacher-forced 结果

已完成：

```text
latent_inpaint_flow_d64_0_s200_teacher_forced_val32
```

`topk_target_unique` 汇总：

| source | IoU | recall | precision | component | largest |
| --- | ---: | ---: | ---: | ---: | ---: |
| q_gt | 0.999960 | 0.999980 | 0.999980 | 2.81 | 0.998541 |
| q_vis | 0.456389 | 0.620594 | 0.620594 | 51.00 | 0.946188 |
| q_splice | 0.663849 | 0.792927 | 0.792927 | 24.66 | 0.984959 |
| tf_t0.25 | 0.651137 | 0.783926 | 0.783926 | 20.91 | 0.991392 |
| tf_t0.50 | 0.615112 | 0.756463 | 0.756463 | 18.97 | 0.992783 |
| tf_t0.75 | 0.476419 | 0.635829 | 0.635829 | 42.09 | 0.955924 |

结论：

```text
1. t=0.25 / t=0.50 的 teacher-forced x0 已经明显好于 q_vis，并接近 q_splice；
2. 这说明当前 LoRA flow + q_vis/m_s condition 并不是完全没有学到局部 denoise；
3. 但从纯 noise 多步 sampling 时 q_pred 仍差，说明问题主要在 unknown 区域的全局生成依据不足；
4. 用户指出的 image condition 缺失是关键问题：只靠 q_vis/m_s 很难从 mask 内局部可观测 latent 推断不可见完整形体。
```

### 16.2 为什么必须接 Image Condition

当前 point-only latent inpaint 的条件实际是：

```text
SparsePointPriorCond(q_vis, m_s, confidence)
```

它提供的是：

```text
1. 可观测区域 latent anchor；
2. known / unknown mask；
3. 由 point prior 得到的空间覆盖信息。
```

它缺少：

```text
1. 图像语义类别；
2. 多视角外观；
3. TRELLIS 原始 image prior 中的全局形体先验；
4. unknown 区域应如何补全的语义依据。
```

这和 Points-to-3D 的完整设定不一致。更合理的条件应是：

```text
condition = image_cond + point_cond(q_vis, m_s)
```

其中：

```text
image_cond:
  提供语义 / 类别 / 全局形状先验。

point_cond:
  提供 observed geometry anchor，约束生成不要偏离 AR/SLAM prior。
```

### 16.3 本次代码修改

新增：

```text
trellis_point_prior_mv/latent_inpaint_image_condition.py
```

功能：

```text
1. 从 latent manifest 的 source_manifest/source_index 找回原始多视角图像；
2. 默认均匀选择 4 个视角；
3. 使用 TRELLIS pipeline.encode_image() 编码 image condition；
4. 支持 mean / first / concat 三种 image view aggregation；
5. 支持 point_only / image_only / concat 三种 condition fusion。
```

修改：

```text
trellis_point_prior_mv/train_latent_inpaint_flow.py
trellis_point_prior_mv/eval_latent_inpaint_flow.py
trellis_point_prior_mv/eval_latent_inpaint_teacher_forced.py
trellis_point_prior_mv/scripts/run_latent_inpaint_flow.sh
trellis_point_prior_mv/scripts/run_latent_inpaint_teacher_forced.sh
trellis_point_prior_mv/命令说明.txt
```

默认仍然关闭 image condition：

```text
USE_IMAGE_COND=0
```

打开后默认配置：

```text
USE_IMAGE_COND=1
IMAGE_MAX_VIEWS=4
IMAGE_FRAME_SELECT=uniform
IMAGE_COND_AGGREGATION=mean
COND_FUSION=concat
```

第一版采用 `mean(image_cond_4views) concat point_cond`，而不是把所有 view tokens 直接 concat。原因是显存更稳，也保留了 TRELLIS 原始 image condition 的语义信息。

### 16.4 下一步判断标准

第一优先级跑 image condition smoke：

```text
latent_inpaint_flow_img4_d64_0_smoke
```

如果 smoke 正常，再跑：

```text
latent_inpaint_flow_img4_d64_0_s200
latent_inpaint_flow_img4_d64_0_s200_teacher_forced_val32
```

判断标准：

```text
1. teacher-forced tf_t0.25 / tf_t0.50 是否进一步接近或超过 q_splice；
2. sampling q_pred 是否至少超过 q_vis；
3. component 是否不显著高于 q_splice；
4. image+point 是否明显优于 image_only，证明 point anchor 仍然有用；
5. 如果 teacher-forced 改善但 sampling 仍差，再集中修 sampling schedule / known clamp；
6. 如果 teacher-forced 也不改善，再考虑 decoder-aware loss 或更强的 image-point fusion。
```

## 17. 2026-06-26 Image Condition 初测结果与诊断路线

### 17.1 已完成实验

已完成两组 image+point concat：

```text
latent_inpaint_flow_img4_d64_0_smoke
latent_inpaint_flow_img4_d64_0_overfit1_s500
```

共同设置：

```text
USE_IMAGE_COND=1
IMAGE_MAX_VIEWS=4
IMAGE_COND_AGGREGATION=mean
COND_FUSION=concat
CFG_DROP_PROB=0
KNOWN_CLAMP_START_T=1.0
CLAMP_INITIAL_NOISE=1
EVAL_STEPS=25
```

sample0 / `topk_target_unique` 结果：

| run | source | IoU | recall | component | largest | unknown L1 vs q_gt |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | q_vis | 0.334169 | 0.500940 | 263 | 0.735970 | 0.665612 |
| baseline | q_splice | 0.564009 | 0.721235 | 68 | 0.991373 | 0.000000 |
| img4 smoke 20 | q_pred | 0.232965 | 0.377894 | 71 | 0.976125 | 0.708536 |
| img4 overfit s500 | q_pred | 0.261938 | 0.415136 | 289 | 0.833817 | 0.620096 |
| point-only s200 strict clamp | q_pred | 0.264426 | 0.418254 | 364 | 0.787050 | 0.592307 |

结论：

```text
1. img4 mean + concat 没有改善 free sampling；
2. overfit s500 的 unknown L1 有下降，但 sparse decode IoU 仍低于 q_vis；
3. overfit s500 和旧 point-only s200 基本同一水平，没有证明 image condition 已经有效进入 unknown 补全；
4. 当前不能直接扩大训练，必须先做 image condition 诊断。
```

### 17.2 为什么先做 A/B/C 诊断

当前失败有三种主要可能：

```text
A. image tokens 本身可用，但 point_cond concat 干扰了 image condition；
B. 单步 teacher-forced 可用，但 free sampling 从 high-t/noise 崩掉；
C. mean 4 views 把视角特异信息平均掉，导致 image condition 不再像 TRELLIS 原始单图条件。
```

因此下一步顺序是：

```text
1. image_only overfit；
2. concat overfit checkpoint 的 teacher-forced；
3. first-view aggregation overfit；
4. 必要时再跑 image_only teacher-forced / first teacher-forced。
```

### 17.3 本次代码补充

新增：

```text
trellis_point_prior_mv/scripts/run_latent_inpaint_imgcond_diagnostics.sh
```

用途：

```text
1. 统一运行 image_only overfit；
2. 统一运行 concat / image_only / first 的 teacher-forced；
3. 统一运行 first aggregation overfit；
4. 避免手写长命令时路径或 .sh 后缀出错。
```

同时修改：

```text
trellis_point_prior_mv/latent_inpaint_image_condition.py
```

增加保护：

```text
COND_FUSION=image_only 时必须 USE_IMAGE_COND=1；
否则直接报错，不再静默退回 point-only。
```

### 17.4 判断标准

image_only overfit：

```text
如果 image_only 明显优于 concat：
  说明 point_cond tokens 干扰 image tokens，后续应改 fusion，例如 gated fusion / adapter fusion，而不是直接 concat。

如果 image_only 也差：
  说明 image condition 的取图、聚合或训练方式没有对齐 TRELLIS 原始 sparse flow 条件。
```

teacher-forced：

```text
如果 teacher-forced 好，free sampling 差：
  继续修 sampler / high-t schedule / clamp。

如果 teacher-forced 也差：
  先修 condition/fusion/target，不继续加长训练。
```

first vs mean：

```text
如果 first 明显优于 mean：
  当前 mean 4 views 聚合不合适，先使用单视角或 TRELLIS 原始 stochastic multi-image 方式。

如果 first 和 mean 都差：
  说明问题不只是 view aggregation。
```

## 18. 2026-06-26 A/B/C 诊断结果与 image-only 解释

### 18.1 全部诊断结果

已完成：

```text
latent_inpaint_flow_img4_d64_0_overfit1_s500
latent_inpaint_flow_img4_d64_0_imageonly_overfit1_s500
latent_inpaint_flow_imgfirst_d64_0_concat_overfit1_s500
latent_inpaint_flow_img4_d64_0_overfit1_s500_teacher_forced
latent_inpaint_flow_img4_d64_0_imageonly_overfit1_s500_teacher_forced
latent_inpaint_flow_imgfirst_d64_0_concat_overfit1_s500_teacher_forced
```

free sampling / `topk_target_unique`：

| run | fusion | image aggregation | IoU | recall | precision | component | largest | unknown L1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q_vis baseline | - | - | 0.334169 | 0.500940 | 0.500940 | 263 | 0.735970 | 0.665612 |
| q_splice upper sanity | - | - | 0.564009 | 0.721235 | 0.721235 | 68 | 0.991373 | 0.000000 |
| img4 overfit s500 | concat | mean | 0.261938 | 0.415136 | 0.415136 | 289 | 0.833817 | 0.620096 |
| image-only overfit s500 | image_only | mean | 0.252688 | 0.403434 | 0.403434 | 281 | 0.768386 | 0.615732 |
| first-view overfit s500 | concat | first | 0.259935 | 0.412616 | 0.412616 | 292 | 0.787648 | 0.610392 |

teacher-forced / `topk_target_unique`：

| run | t | IoU | recall | precision | component | largest | hard x0 vs q_splice L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| img4 concat | 0.25 | 0.540750 | 0.701930 | 0.701930 | 69 | 0.990262 | 0.167831 |
| img4 concat | 0.50 | 0.527332 | 0.690527 | 0.690527 | 59 | 0.989152 | 0.308882 |
| img4 concat | 0.75 | 0.367680 | 0.537670 | 0.537670 | 108 | 0.971470 | 0.440823 |
| image-only | 0.25 | 0.546244 | 0.706543 | 0.706543 | 68 | 0.989408 | 0.165775 |
| image-only | 0.50 | 0.543085 | 0.703895 | 0.703895 | 54 | 0.991245 | 0.301037 |
| image-only | 0.75 | 0.449065 | 0.619800 | 0.619800 | 119 | 0.972239 | 0.418700 |
| first-view concat | 0.25 | 0.541409 | 0.702486 | 0.702486 | 71 | 0.990134 | 0.167715 |
| first-view concat | 0.50 | 0.525590 | 0.689032 | 0.689032 | 63 | 0.989194 | 0.308970 |
| first-view concat | 0.75 | 0.361556 | 0.531093 | 0.531093 | 136 | 0.963953 | 0.441996 |

### 18.2 结论

第一，`image_only` free sampling 不好：

```text
image_only free sampling IoU=0.252688，
仍然低于 q_vis IoU=0.334169，
也没有超过 concat mean / first。
```

第二，`image_only` teacher-forced 明显最好：

```text
t=0.25: image_only IoU=0.546244，接近 q_splice IoU=0.564009；
t=0.50: image_only IoU=0.543085，也明显高于 q_vis；
t=0.75: image_only IoU=0.449065，高于 concat / first。
```

这说明：

```text
image condition 不是完全无效；
在给定接近 target 的 x_t 时，image-only 单步 x0 预测可以恢复到接近 q_splice；
真正失败的是从 high noise 多步 free sampling 到 q_pred 的过程。
```

第三，`first` 不解决问题：

```text
first-view concat free sampling IoU=0.259935；
mean-view concat free sampling IoU=0.261938；
teacher-forced first 也不优于 mean。
```

所以问题不是简单的“4 视角 mean 把信息平均坏了”。

第四，concat 可能轻微干扰 image token：

```text
teacher-forced t=0.75:
  image_only IoU=0.449065
  concat mean IoU=0.367680
  first concat IoU=0.361556
```

高噪声阶段 image-only 明显更稳，说明 point_cond concat 在 high-t 可能干扰 image token，至少不是最优 fusion。

### 18.3 为什么 image-only 不等于 TRELLIS

`COND_FUSION=image_only` 不是 stock TRELLIS。

当前 image-only 实验仍然包含以下差异：

```text
1. 使用的是 latent-inpaint LoRA checkpoint，不是原始 TRELLIS sparse flow；
2. 训练目标是 q_splice = m_s * q_vis + (1 - m_s) * q_gt，不是 TRELLIS 原始 image -> full sparse latent 分布；
3. sampling 仍然使用 q_vis / m_s 做 known-region clamp；
4. 采样代码是当前 masked latent inpaint sampler，不是 TRELLIS 原始 sample_sparse_structure 路径；
5. CFG 使用 guidance_strength=1.0，即 conditional-only，不是 TRELLIS mesh pipeline 常用的高 CFG 设置；
6. image-only condition 中没有 mask token，模型不知道哪个区域是 observed / unknown，只能通过 sampling clamp 间接处理 known 区域。
```

因此它的含义是：

```text
“当前 latent-inpaint LoRA 在只给 image tokens 时能不能补 unknown latent”
```

而不是：

```text
“原始 TRELLIS image-only sparse generation 能不能工作”
```

真正的 TRELLIS 对照应是：

```text
stock sparse flow + 原始 image condition + 原始 sampler + 无 q_vis clamp + 无 latent-inpaint LoRA
```

这类对照前面在 mesh frozen / stock_sparse 实验里已经间接做过；如果要针对当前 sample 精确判断，需要再补一个 `stock image sparse latent` 诊断，而不是把 `COND_FUSION=image_only` 当成 TRELLIS。

### 18.4 下一步建议

当前结果更支持：

```text
teacher-forced 好，free sampling 差。
```

但这里还不能直接进入“修 inpaint 高噪声生成”。更高优先级的问题是：

```text
我们还没有证明自写 sample_latent()
在 no-LoRA / no-clamp / image-only / native CFG 下
能复现 stock TRELLIS sparse generation。
```

这一步没证明之前，free sampling 差会混入：

```text
1. sampler 实现不等价；
2. CFG 被错误覆盖；
3. native sparse sampler 参数没有被使用；
4. known clamp 破坏轨迹；
5. LoRA checkpoint 破坏原始 image prior。
```

因此下一步优先级改为：

```text
1. stock/no-LoRA/no-clamp/image-only/custom sampler 复现 TRELLIS；
2. 打印并使用 TRELLIS 原始 sparse_structure_sampler_params；
3. CFG sweep，而不是盲目固定 guidance_strength=1.0；
4. stock/no-LoRA + clamp，判断 clamp 是否破坏 trajectory；
5. LoRA + no-clamp，判断 LoRA 是否破坏 image prior；
6. oracle warm-start / stock q_img warm-start refinement；
7. 最后再考虑 gated fusion / adapter fusion。
```

### 18.5 CFG=1.0 是高优先级嫌疑

当前 `eval_latent_inpaint_flow.py` 原逻辑中：

```text
parser 默认 guidance_strength=1.0；
sample_latent() 优先使用 args.guidance_strength；
因此 pipeline.sparse_structure_sampler_params 里的原生 cfg_strength 被覆盖。
```

如果 TRELLIS 原生 sparse generation 依赖更高 CFG，那么：

```text
image-only free sampling 差
可能不是 image condition 差，
而是自定义采样把原生 CFG 降没了。
```

因此已修改 eval 代码：

```text
1. 新增 --use_native_ss_cfg；
2. guidance_strength 默认不再强制覆盖 native cfg；
3. --steps <= 0 时使用 TRELLIS 原生 sparse steps；
4. report 中记录 native/effective cfg、steps、guidance_rescale；
5. 新增 --no_lora；
6. 新增 --add_native_sampler_pred，用同一份 initial noise 同时跑手写 loop 和 sampler API。
```

新增 wrapper：

```text
trellis_point_prior_mv/scripts/run_latent_inpaint_sampler_equivalence.sh
```

### 18.6 新判断标准

第一步看：

```text
q_pred vs q_pred_native_sampler
```

如果两者不同：

```text
说明自写 sample_latent() 与 TRELLIS sampler API 不等价，先修 sampler。
```

如果两者一致但都差：

```text
再看 native cfg / CFG sweep。
```

如果 stock/no-LoRA/no-clamp/image-only/native CFG 已经合理：

```text
再逐个打开 clamp、LoRA、point condition。
```

如果 stock/no-LoRA/no-clamp/image-only/native CFG 都不合理：

```text
说明当前 image preprocessing / condition aggregation / custom eval 路径仍未对齐原始 TRELLIS，
不能把失败归因到 point prior 或 latent inpaint。
```

## 19. 2026-06-26 Stock Image-only 诊断代码错误修正

### 19.1 现象

已运行：

```text
DIAG=stock_imageonly_native
no_lora=1
known_latent_clamp_strength=0
cond_fusion=image_only
use_native_ss_cfg=1
```

结果：

```text
q_pred/topk_target_unique:
  IoU=0.055017
  recall=0.104297
  component=555

q_pred_native_sampler/topk_target_unique:
  IoU=0.055017
  recall=0.104297
  component=555

q_vis/topk_target_unique:
  IoU=0.334169
  recall=0.500940
```

这个结果说明：

```text
1. 自写 sample_latent() 与 native sampler API 在这次配置下完全一致；
2. 但是 stock/no-LoRA/no-clamp/image-only 远低于 q_vis；
3. 不能据此直接判断 TRELLIS stock flow 错，因为 image condition 输入路径后来确认没有对齐。
```

### 19.2 找到的代码问题

原始 `latent_inpaint_image_condition.py` 的 image condition 路径是：

```text
Image.open(image_path).convert("RGB")
pipeline.encode_image(raw_full_frame)
```

这和当前 `mesh_frozen_downstream.py` 中更接近 TRELLIS 的 stock sparse 输入路径不一致。`mesh_frozen_downstream.py` 实际会：

```text
1. 读取 RGB；
2. 读取 mask；
3. 用 mask 写入 alpha；
4. RGB 乘 alpha；
5. 按 mask bbox crop；
6. 方形 padding；
7. resize 到 518；
8. 再送入 pipeline.get_cond()/encode_image。
```

也就是说，之前的 stock image-only 诊断使用的是：

```text
未裁剪、未 alpha mask、未按对象居中的整图 RGB。
```

这不是 TRELLIS 原始使用方式，也不是 mesh_frozen 中 `stock_sparse` 使用的条件图像。

### 19.3 修正

已修改：

```text
trellis_point_prior_mv/latent_inpaint_image_condition.py
trellis_point_prior_mv/train_latent_inpaint_flow.py
trellis_point_prior_mv/eval_latent_inpaint_flow.py
trellis_point_prior_mv/eval_latent_inpaint_teacher_forced.py
trellis_point_prior_mv/scripts/run_latent_inpaint_flow.sh
trellis_point_prior_mv/scripts/run_latent_inpaint_teacher_forced.sh
trellis_point_prior_mv/scripts/run_latent_inpaint_imgcond_diagnostics.sh
trellis_point_prior_mv/scripts/run_latent_inpaint_sampler_equivalence.sh
```

新增默认：

```text
IMAGE_USE_SOURCE_MASK=1
IMAGE_MASK_CROP_RESOLUTION=518
```

并新增：

```text
SourceImageResolver.image_mask_paths()
apply_mask_and_crop()
```

校验结果：

```text
latent_inpaint_image_condition.apply_mask_and_crop
vs
eval_mesh_frozen_downstream.apply_mask_and_crop

maxdiff=0
mode=RGBA
size=(518, 518)
```

说明修正后的 latent image condition 裁剪路径已经和 mesh_frozen stock 输入一致。

### 19.4 新结论

之前这条结果：

```text
stock/no-LoRA/no-clamp/image-only/native CFG IoU=0.055
```

不能作为 “stock TRELLIS flow 错” 的证据。

更准确的结论是：

```text
旧的 latent image condition 实现有问题；
它没有使用 source mask 做 TRELLIS 风格对象裁剪和居中；
因此旧 stock_imageonly_native 诊断无效，需要重跑。
```

当前还不能确定 stock flow 是否错。下一步必须重跑：

```text
stock/no-LoRA/no-clamp/image-only/native CFG + mask-crop image condition
```

如果重跑后仍然显著低于 q_vis，再继续检查：

```text
1. latent target frame 是否和 TRELLIS image sparse 输出同一 canonical frame；
2. synthetic render 图像是否符合 TRELLIS 预训练输入分布；
3. 单图 image-only 本身是否足以恢复该 Objaverse 目标；
4. 是否需要用 mesh_frozen 的 stock_sparse 作为唯一 stock 对照。
```

## 20. 2026-06-26 Mask-crop 后 stock image-only native 复测结论

### 20.1 复测命令

已重新运行：

```bash
GPU=4 \
DIAG=stock_imageonly_native \
LATENT_RUN_ROOT=/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint/latent_inpaint_smoke \
INDICES=0 \
IMAGE_MAX_VIEWS=4 \
IMAGE_FRAME_SELECT=uniform \
IMAGE_COND_AGGREGATION=mean \
EVAL_STEPS=0 \
bash trellis_point_prior_mv/scripts/run_latent_inpaint_sampler_equivalence.sh
```

输出路径：

```text
/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint_flow/latent_inpaint_sampler_stock_imageonly_native/report.json
```

这次配置确认：

```text
no_lora=True
cond_fusion=image_only
image_max_views=4
image_frame_select=uniform
image_cond_aggregation=mean
image_use_source_mask=True
image_mask_crop_resolution=518
image_cond_mask_count=4
native_ss_cfg_strength=5.0
effective_ss_cfg_strength=5.0
native_ss_steps=25
effective_ss_steps=25
```

也就是说，这次已经不是旧版 raw RGB 整图输入，mask-crop 路径确实生效。

### 20.2 主要结果

`topk_target_unique` 下：

```text
q_gt:
  IoU=0.99949
  recall=0.99974
  component=1
  largest=1.0000

q_vis:
  IoU=0.33417
  recall=0.50094
  component=263
  largest=0.7360

q_splice:
  IoU=0.56401
  recall=0.72124
  component=68
  largest=0.9914

q_pred:
  IoU=0.01051
  recall=0.02080
  component=214
  largest=0.8610

q_pred_native_sampler:
  IoU=0.01051
  recall=0.02080
  component=214
  largest=0.8610
```

`q_pred` 与 `q_pred_native_sampler` 完全一致，说明在这次配置下：

```text
自写 sample_latent loop
和
TRELLIS native sparse_structure_sampler.sample()
```

得到的是同一个 latent / sparse decode 结果。当前主要问题不是 sampler wrapper 和 native sampler API 不等价。

### 20.3 不是 top-k 假象

不同 decode 方式下的 `q_pred` 都很差：

```text
q_pred/threshold_0:
  IoU=0.00648
  recall=0.00671
  coord_count=961
  component=2

q_pred/topk_4096:
  IoU=0.01584
  recall=0.01832
  coord_count=4096
  component=57

q_pred/topk_8192:
  IoU=0.01480
  recall=0.01969
  coord_count=8192
  component=722

q_pred/topk_target_unique:
  IoU=0.01051
  recall=0.02080
  coord_count=23414
  component=214
```

因此这不是 `target_unique` top-k 选取造成的假象。

相比之下：

```text
q_vis/topk_target_unique:
  IoU=0.33417
  recall=0.50094

q_splice/topk_target_unique:
  IoU=0.56401
  recall=0.72124
```

说明当前 visible latent / splice latent 与 `q_gt` 至少在当前 voxel frame 内有可解释的重叠，而 image-only stock sparse latent 没有。

### 20.4 不是简单坐标轴翻转

对 `q_pred` sparse coords 做 3D 轴交换和 xyz flip 的快速穷举后，最佳 IoU 仍然只有约：

```text
best symmetry IoU ~= 0.0389
best target recall ~= 0.0748
```

所以低 IoU 不是简单的：

```text
x/y/z 轴顺序错了
或
某个轴需要 63 - coord 翻转
```

这类刚性 voxel 坐标变换不能解释当前差异。

### 20.5 q_pred 形态异常

正确读取 `[batch, x, y, z]` 后，`q_gt` 和 `q_pred` 的 bbox 形态明显不同：

```text
q_gt/threshold_0:
  n=23421
  min=[3,16,3]
  max=[60,47,60]
  extent=[58,32,58]

q_pred/threshold_0:
  n=961
  min=[14,12,31]
  max=[63,51,33]
  extent=[50,40,3]
```

`q_pred/threshold_0` 更像一张很薄的片状结构，z 方向只有 3 个 voxel 厚；它不是 `q_gt` 那种完整物体 occupancy。

这说明：

```text
stock image-only sparse latent 在当前评估 frame 中并不是 q_gt 的直接复现。
```

### 20.6 当前最重要的结论

现在仍然不能说：

```text
stock TRELLIS flow 本身错了。
```

更准确的结论是：

```text
当前 PixalV9 / latent-inpaint 的 q_gt canonical voxel frame，
和 TRELLIS image-only stock sparse 输出 frame，
没有建立可直接 voxel IoU 对齐的关系。
```

因此：

```text
stock image-only native sparse
不能直接作为当前 q_gt voxel frame 的上限；
也不能用 q_pred vs q_gt voxel IoU 直接判断 stock TRELLIS 是否正常。
```

这也解释了为什么：

```text
mesh_frozen_downstream 里的 stock_sparse sparse_iou 经常很低，
但 stock_sparse 仍然能继续进入 slat / mesh decoder 生成 mesh。
```

TRELLIS stock image-to-3D pipeline 的合理性应优先通过：

```text
1. pipeline.sample_sparse_structure() 得到的 coords；
2. pipeline.sample_slat()；
3. decode mesh 后的 surface / visual 指标；
```

而不是通过当前 `q_gt` latent voxel IoU 单独判断。

### 20.7 对 latent inpainting 路线的影响

这一结果对后续路线影响很大：

```text
不能继续假设：
  image-only stock sparse 应该复现当前 q_gt sparse latent。

也不能继续把：
  stock image-only -> q_gt voxel IoU
当作 latent inpaint 的合理上限。
```

当前 latent inpaint 的 `q_gt/q_vis/q_splice` sanity 仍然有价值，因为它们来自同一个 latent/voxel frame：

```text
q_vis -> q_gt
q_splice -> q_gt
teacher-forced pred_x0 -> q_splice/q_gt
```

这些比较仍然可以评估：

```text
visible latent 是否有效；
known/unknown mask 是否合理；
模型是否能在同一 latent frame 里补全 unknown。
```

但是 `stock image-only native` 不能再作为这个同 frame 评价体系的一部分，除非先解决 canonical alignment。

### 20.8 下一步建议

下一步不要继续盲训 latent inpaint。优先做两个诊断：

```text
诊断 A：stock sparse coords -> mesh 的 end-to-end 表现
  继续用 mesh/surface/visual 指标判断 TRELLIS stock 是否正常。

诊断 B：q_gt latent frame 与 stock sparse output frame 的关系
  将 stock_sparse coords 与 q_gt coords 做可视化、bbox、PCA/ICP-like 归一化检查；
  如果两者不是同一个 canonical frame，就不能用 voxel IoU 直接评价。
```

如果诊断 B 证明 frame 不一致，后续 latent inpaint 有两条选择：

```text
路线 1：
  继续在 q_gt/q_vis/q_splice 的同一 latent frame 内训练，
  但 image condition 只作为语义辅助，不要求 image-only stock 复现 q_gt。

路线 2：
  改为 TRELLIS stock sparse frame，
  以 stock sparse / stock slat / stock mesh 为 backbone，
  point prior 只做局部 correction 或 rerank。
```

当前更稳的工程方向仍然是路线 2：

```text
stock/base sparse 负责原生拓扑和 TRELLIS 坐标分布；
point/SLAM prior 负责局部几何修正、candidate filtering、rerank；
不要让 Stage2 在全局 64^3 voxel 空间里自由生成完整 sparse coords。
```
