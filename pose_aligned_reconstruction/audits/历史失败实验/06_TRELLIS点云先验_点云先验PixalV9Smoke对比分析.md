# 点云先验 Pixal V9 Smoke 对比分析

时间：2026-06-19

本文记录 `trellis_point_prior_mv` 中两个 smoke run 的结果分析：

- 默认 smoke：`trellis_point_prior_mv/outputs/pointprior_pixal_v9_smoke`
- easy smoke：`trellis_point_prior_mv/outputs/pointprior_pixal_v9_smoke_easy`

两个 run 都默认使用 front-depth support filtering，且默认关闭 support fallback。因此如果投影、坐标系或 mask support 完全失败，`support_failed` 会暴露出来，不会被退回完整 surface 掩盖。

## 实验配置

### 默认 smoke

- train：64 个样本
- val prior 构建：32 个样本
- eval：0-7，共 8 个样本
- train steps：20
- eval steps：8
- prior 点数候选：`50,100,300,800,1500`
- prior 视角候选：`1,2,4,8`
- dropout 最大值：`0.65`
- outlier ratio：`0.03`
- eval prior modes：`correct,empty,shuffle,random,jitter`
- fixed top-k：`4096,8192,target_unique`

### easy smoke

- train：64 个样本
- val prior 构建：32 个样本
- eval：0-7，共 8 个样本
- train steps：20
- eval steps：12
- prior 点数候选：`300,800,1500`
- prior 视角候选：`4,8`
- dropout 最大值：`0.25`
- outlier ratio：`0.0`
- eval prior modes：`correct,empty,shuffle,random,jitter`
- fixed top-k：`4096,8192,target_unique`

## Prior 构建质量

| run | split | samples | support_failed | fallback_used | prior points mean | view count mean | support ratio mean | visible mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default | train | 64 | 0 | 0 | 258.0 | 3.56 | 0.628 | 2.63 |
| default | val | 32 | 0 | 0 | 338.2 | 3.53 | 0.630 | 2.66 |
| easy | train | 64 | 0 | 0 | 710.0 | 5.94 | 0.713 | 4.43 |
| easy | val | 32 | 0 | 0 | 680.3 | 6.00 | 0.699 | 4.54 |

这里有两个结论：

1. 当前 front-depth support filtering 没有出现整体失败。两个 run 的 `support_failed=0`、`fallback_used=0`，说明至少在这批 Pixal V9 样本上，投影和 mask support 不是完全断掉的。
2. easy prior 明显更强。它有更多视角、更低 dropout、更多 prior points、更高 support ratio，因此如果模型真的能稳定利用 point prior，easy run 应该比默认 smoke 更容易出现 correct 优于 wrong prior。

## Eval 总体结果

| run | correct top1 rate | correct rank mean | correct rank median | correct IoU mean | shuffle IoU mean | random IoU mean | jitter IoU mean | empty IoU mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 0.208 | 2.33 | 2.0 | 0.0222 | 0.0242 | 0.0327 | 0.0174 | 0.0080 |
| easy | 0.458 | 1.71 | 2.0 | 0.0190 | 0.0179 | 0.0060 | 0.0211 | 0.0000 |

default smoke 的结果较差：`random` 的平均 IoU 甚至高于 `correct`，`shuffle` 也略高于 `correct`。这说明在 noisy prior 设置下，20 step 训练出来的条件并没有学到稳定的“正确点云先验优于错误点云先验”。

easy smoke 有改善：`correct_rank top1_rate` 从 `0.208` 提升到 `0.458`，`random` 和 `empty` 被明显压低，说明模型确实在使用 point prior。但它仍然没有稳定地区分 `correct` 和 `shuffle/jitter`：

- `correct` 与 `shuffle` 基本打平；
- `jitter` 在总体 IoU 上还略高于 `correct`；
- `correct top1 rate` 仍不到 50%。

这不是“完全没用”，但也不能说当前结构已经证明有效。

## Fixed Top-k 分析

### 默认 smoke

| top-k | correct IoU | empty IoU | shuffle IoU | random IoU | jitter IoU |
|---|---:|---:|---:|---:|---:|
| 4096 | 0.0216 | 0.0075 | 0.0218 | 0.0276 | 0.0164 |
| 8192 | 0.0228 | 0.0081 | 0.0255 | 0.0364 | 0.0181 |
| target_unique | 0.0222 | 0.0084 | 0.0252 | 0.0341 | 0.0176 |

paired delta 也不理想：

- `correct - empty` 为正，说明有 prior 比空 prior 好；
- `correct - random` 平均为负；
- `correct - shuffle` 基本为 0 或略负。

这说明默认 prior 噪声太重时，当前模型没有形成稳定的几何对应关系。

### easy smoke

| top-k | correct IoU | empty IoU | shuffle IoU | random IoU | jitter IoU |
|---|---:|---:|---:|---:|---:|
| 4096 | 0.0181 | 0.0000 | 0.0181 | 0.0070 | 0.0223 |
| 8192 | 0.0196 | 0.0000 | 0.0181 | 0.0056 | 0.0205 |
| target_unique | 0.0194 | 0.0000 | 0.0175 | 0.0053 | 0.0205 |

easy run 的现象更有价值：

- `empty=0`：模型已经依赖 point prior，空 prior 下基本生成失败；
- `random` 明显低于 `correct`：模型不是完全忽略 prior；
- `shuffle` 接近 `correct`：模型仍然没有足够强的样本级几何绑定；
- `jitter` 略好于 `correct`：说明当前 prior 坐标精确性、latent grid 对齐或 target sparse thick surface 可能存在偏差，轻微扰动反而有正则化效果。

## 当前判断

当前结果不支持“继续盲目加训练步数”。原因是 easy prior 已经降低了任务难度，但 correct 仍未稳定胜过 shuffle/jitter。

更准确的判断是：

1. point prior condition 有信号，模型不是完全不用它；
2. 但当前训练目标主要是 flow denoising，没有显式惩罚 wrong prior；
3. 只要 wrong prior 也是一个“物体形状点云”，它也可能提供有用的 occupancy-like bias；
4. 因此模型可能学到的是“有稀疏点云就生成更像物体”，而不是“这个点云必须与当前样本几何一致”。

这和之前 Pixal3D 多视图 pose-sensitive 分支的趋势相似：单纯把 pose/投影信息塞进 sparse flow 条件里，收益容易被 TRELLIS sparse prior 自身和泛化形状先验吞掉。区别是这里的 point prior 至少让 empty/random 明显下降，说明这条路线比纯 pose token 更有希望，但需要改训练约束。

## 下一步建议

### 第一优先级：先做 oracle prior 上限测试

目的不是追求最终效果，而是判断“如果 prior 足够干净，当前结构能不能稳定让 correct 胜过 shuffle”。

建议配置：

- `NUM_PRIOR_VIEWS_CHOICES=8`
- `POINT_COUNT_CHOICES=1500`
- `DROPOUT_MIN=0`
- `DROPOUT_MAX=0`
- `OUTLIER_RATIO=0`
- `COORD_JITTER=0`
- `EVAL_STEPS=12`

命令：

```bash
cd /home/zjr/Tracker

GPU=1 \
MODE=smoke \
RUN_NAME=pointprior_pixal_v9_oracle_smoke \
POINT_COUNT_CHOICES=1500 \
NUM_PRIOR_VIEWS_CHOICES=8 \
DROPOUT_MIN=0.0 \
DROPOUT_MAX=0.0 \
OUTLIER_RATIO=0.0 \
COORD_JITTER=0 \
EVAL_STEPS=12 \
bash trellis_point_prior_mv/scripts/run_pixal_v9_point_prior_stage1.sh
```

观察标准：

- 如果 oracle prior 下 `correct top1_rate` 仍然低，且 `correct` 仍不能稳定胜过 `shuffle`，优先改模型/训练目标，而不是加步数。
- 如果 oracle prior 下 `correct` 明显胜出，再回头逐步加 dropout、jitter、outlier，确定 prior 噪声容忍边界。

### 第二优先级：加入 wrong-prior ranking loss

当前 flow loss 只告诉模型“给这个 condition 时要去拟合 target”，但没有告诉它“错误 prior 应该更差”。建议增加一个小权重 ranking 项：

- 同一个 target、同一个 `t`、同一个 noise；
- 构造 correct prior 和 wrong prior；
- correct condition 的 denoising loss 应低于 wrong condition；
- 使用 hinge/margin 形式，避免过强破坏原始 TRELLIS sparse flow。

建议先做 smoke，不直接大规模训练。

### 第三优先级：再跑 s200 easy

如果 oracle smoke 证明 correct 有明显优势，再跑 s200 才有意义：

```bash
cd /home/zjr/Tracker

GPU=1 \
MODE=s200 \
RUN_NAME=pointprior_pixal_v9_easy_s200 \
POINT_COUNT_CHOICES=300,800,1500 \
NUM_PRIOR_VIEWS_CHOICES=4,8 \
DROPOUT_MAX=0.25 \
OUTLIER_RATIO=0.0 \
EVAL_STEPS=12 \
bash trellis_point_prior_mv/scripts/run_pixal_v9_point_prior_stage1.sh
```

如果 oracle smoke 都不成立，则 s200 很可能只是把“有 prior 就生成”学得更强，而不会自然学出 pose/样本级几何一致性。

## 当前结论

这轮不能说 point-prior 路线失败，但可以说“当前 Stage 1 结构和训练目标还不够”。它比之前纯 pose-sensitive head 更有信号，因为 empty/random 被压低；但 correct 与 shuffle/jitter 没拉开，说明它还没有学到可靠的样本级几何约束。

因此下一步不建议直接长训。先做 oracle prior 上限测试；如果 oracle 成立，再加 wrong-prior ranking loss 并跑 s200；如果 oracle 不成立，优先查坐标/latent 对齐和 condition 注入方式。

## Oracle Prior Smoke 结果补充

运行目录：

```text
trellis_point_prior_mv/outputs/pointprior_pixal_v9_oracle_smoke
```

本次设置：

- `POINT_COUNT_CHOICES=1500`
- `NUM_PRIOR_VIEWS_CHOICES=8`
- `DROPOUT_MIN=0.0`
- `DROPOUT_MAX=0.0`
- `OUTLIER_RATIO=0.0`
- `COORD_JITTER=0`
- `EVAL_STEPS=12`
- train steps 仍为 smoke 默认的 20

### Prior 构建情况

| split | samples | support_failed | fallback_used | prior points mean | view count mean | dropout mean | support ratio mean | visible mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 64 | 0 | 0 | 1495.1 | 8.0 | 0.0 | 0.715 | 5.965 |
| val | 32 | 0 | 0 | 1500.0 | 8.0 | 0.0 | 0.698 | 6.093 |

这个结果说明 oracle prior 的构建本身没有明显失败：8 视角、1500 点、无 dropout、无 outlier、无 support fallback。也就是说，如果后续 eval 仍然不能让 `correct` 稳定胜过 `shuffle/jitter`，就不能再主要归因于 prior 太稀疏或太脏。

### Eval 总体结果

| prior mode | IoU mean | IoU median | recall mean | precision mean |
|---|---:|---:|---:|---:|
| correct | 0.0245 | 0.0288 | 0.0427 | 0.0630 |
| empty | 0.0014 | 0.0000 | 0.0024 | 0.0033 |
| shuffle | 0.0262 | 0.0313 | 0.0467 | 0.0625 |
| random | 0.0199 | 0.0210 | 0.0356 | 0.0469 |
| jitter | 0.0315 | 0.0323 | 0.0551 | 0.0777 |

Correct rank：

| metric | value |
|---|---:|
| count | 24 |
| top1 | 3 |
| top1 rate | 0.125 |
| rank mean | 2.792 |
| rank median | 3.0 |

### Fixed Top-k 对比

| top-k | correct IoU | empty IoU | shuffle IoU | random IoU | jitter IoU |
|---|---:|---:|---:|---:|---:|
| 4096 | 0.0264 | 0.0016 | 0.0274 | 0.0219 | 0.0318 |
| 8192 | 0.0237 | 0.0012 | 0.0254 | 0.0187 | 0.0316 |
| target_unique | 0.0233 | 0.0012 | 0.0258 | 0.0191 | 0.0312 |

Paired delta：

| top-k | correct-empty | correct-shuffle | correct-random | correct-jitter |
|---|---:|---:|---:|---:|
| 4096 | +0.0247, 8/8 wins | -0.0010, 2/8 wins | +0.0045, 6/8 wins | -0.0054, 3/8 wins |
| 8192 | +0.0225, 8/8 wins | -0.0017, 2/8 wins | +0.0050, 5/8 wins | -0.0079, 2/8 wins |
| target_unique | +0.0221, 8/8 wins | -0.0025, 2/8 wins | +0.0042, 5/8 wins | -0.0079, 2/8 wins |

## Oracle 结果解读

Oracle prior 没有让 correct 变成最优，反而让 `correct_rank top1_rate` 从 easy smoke 的 `0.458` 降到 `0.125`。这说明当前问题不是简单的 prior 稀疏、dropout、outlier 或 support fallback。

更关键的现象是：

1. `empty` 明显最差，说明模型确实依赖 point prior；
2. `random` 大体低于 `correct`，说明模型不是完全忽略点；
3. `shuffle` 和 `jitter` 高于 `correct`，说明模型没有学到“当前样本的点云必须对应当前目标”；
4. `jitter` 最好，说明精确坐标并没有被当成硬约束，点云更像一种软的形状/覆盖提示。

因此当前 `point as condition` 的方式本质上仍然是一个 soft condition。它可以告诉模型“这里有一些物体形状点”，但没有强迫 sparse structure 在这些点附近完成 inpainting，也没有强迫错误点云导致更差结果。

这和 TRELLIS sparse flow 的强先验有关：只把 point prior 经过 encoder 后拼进 condition，模型很容易学到“有一团稀疏 occupancy bias 就行”，而不是学习严格的几何约束。只要 `shuffle` prior 也是来自另一个合理物体，它仍然可能帮助生成一个合理 sparse structure。

## 是否需要更换思路

需要。这里不是放弃点云，而是不能继续把点云仅仅当作普通 condition token。更合理的方向是按 Points-to-3D 式思路，把点云作为 sparse structure inpainting 的已知部分。

换句话说，下一阶段应从：

```text
point prior -> encoder -> condition -> sparse flow
```

改成：

```text
point prior -> known sparse structure / known latent mask
target sparse -> unknown region
sparse flow 学习补全 unknown，而不是自由生成 whole sparse
sampling 过程中对 known region 做 clamp / reinjection
```

这和当前实现的区别很大：

- 当前做法：点云只是影响 denoising 的条件，模型可以不严格服从；
- Points-to-3D 式做法：点云是部分结构本身，采样时 known region 应该被保留或反复注入；
- 当前评测只看 fixed top-k overlap；
- 新评测还要看 known-point consistency、known-region recall、unknown completion quality。

## 下一步建议更新

### 第一优先级：停止直接加长当前 condition 版本训练

不建议直接跑 `pointprior_pixal_v9_oracle_s200` 或 `pointprior_pixal_v9_easy_s200` 作为主线。原因是 oracle smoke 已经说明，点更干净、更密、更完整并不能让 correct 稳定胜出。长训可能只会强化“有点云就生成”的软 bias，而不是得到真正的 pose/点云约束。

### 第二优先级：实现 sparse structure inpainting 版本

建议新开 Stage 2，不覆盖当前 Stage 1：

1. 构建 `known_coords`、`known_conf`、`known_latent_mask`；
2. 将 point prior voxelize 到 64 网格，再下采样到 16 latent grid；
3. 训练时区分 known / unknown latent cells；
4. flow loss 主体仍对 target sparse latent 做 denoising，但 known 区域要加入一致性约束；
5. sampling 时每一步或每若干步把 known latent/mask 重新注入；
6. final logits/top-k 阶段保证 known coords 不被完全丢掉。

关键指标：

- known point recall；
- known latent cell recall；
- unknown region IoU；
- correct-vs-shuffle rank；
- correct-vs-jitter rank；
- final sparse 是否覆盖输入点云。

### 第三优先级：wrong-prior ranking 只作为辅助项

wrong-prior ranking 仍然有价值，但它不应该是下一步主线。因为当前 oracle 结果说明，仅靠 condition-level ranking 可能还不够。更合理的是：

- 先把点云改成 hard/semi-hard inpainting substrate；
- 再加入 wrong prior ranking，要求 correct inpainting loss 低于 shuffle/random prior；
- ranking 的对象应包含 known consistency 和 unknown completion，而不只是全局 IoU 或 flow MSE。

## 更新后的结论

当前 Stage 1 证明了一件事：point prior 有信号，但“简单作为 condition”不够。即便 oracle prior 很干净，`correct` 也没有稳定优于 `shuffle/jitter`。这已经接近当前路线的上限判断。

因此后续应该切换到 Points-to-3D 式 sparse structure inpainting：让 AR SLAM 点云或多视图 pose/mask 反投影点云成为已知稀疏结构的一部分，而不是普通条件向量。这样才有机会把准确 pose 和稀疏点云转化为真正的几何约束。

## Stage 2 Strict Masked Inpainting Smoke

运行目录：

```text
trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_strictmask_smoke
```

本次设置：

- `POINT_COUNT_CHOICES=1500`
- `NUM_PRIOR_VIEWS_CHOICES=8`
- `DROPOUT_MIN=0.0`
- `DROPOUT_MAX=0.0`
- `OUTLIER_RATIO=0.0`
- `COORD_JITTER=0`
- `KNOWN_USE_CONFIDENCE=0`
- `KNOWN_LATENT_CLAMP_STRENGTH=1.0`
- `KNOWN_LOGIT_BOOST=0.0`
- `CLAMP_INITIAL_NOISE=1`
- train steps：20
- eval steps：12
- eval indices：0-7

这次是严格 masked inpainting 版本：known 区域 target 使用 `raw_partial_latent`，unknown 区域才使用 full `target_x0`；采样时每一步用同一份 known noise 将 `raw_partial_latent` 加噪到当前 timestep 后 clamp 回 `x_t`。最终 logits 没有做 point boost，因此结果不是后处理插点得到的。

### Prior 构建情况

| split | samples | support_failed | fallback_used | prior points mean | view count mean | dropout mean | support ratio mean | visible mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 64 | 0 | 0 | 1495.1 | 8.0 | 0.0 | 0.715 | 5.965 |
| val | 32 | 0 | 0 | 1500.0 | 8.0 | 0.0 | 0.698 | 6.093 |

构建侧仍然是干净 oracle prior：无 fallback、无 dropout、无 outlier、无 jitter。

### Eval 总体结果

| prior mode | IoU mean | IoU median | recall mean | precision mean | prior recall mean |
|---|---:|---:|---:|---:|---:|
| correct | 0.1070 | 0.1076 | 0.1819 | 0.2309 | 0.4400 |
| empty | 0.0469 | 0.0401 | 0.0831 | 0.1085 | 0.0000 |
| shuffle | 0.0332 | 0.0315 | 0.0596 | 0.0789 | 0.4862 |
| random | 0.0282 | 0.0255 | 0.0505 | 0.0682 | 0.4443 |
| jitter | 0.0316 | 0.0291 | 0.0568 | 0.0759 | 0.3609 |

Correct rank：

| metric | value |
|---|---:|
| count | 24 |
| top1 | 24 |
| top1 rate | 1.000 |
| rank mean | 1.000 |
| rank median | 1.000 |

### Fixed Top-k 对比

| top-k | correct IoU | empty IoU | shuffle IoU | random IoU | jitter IoU |
|---|---:|---:|---:|---:|---:|
| 4096 | 0.0904 | 0.0319 | 0.0223 | 0.0195 | 0.0236 |
| 8192 | 0.1114 | 0.0507 | 0.0358 | 0.0300 | 0.0336 |
| target_unique | 0.1192 | 0.0581 | 0.0416 | 0.0352 | 0.0377 |

Paired delta：

| top-k | correct-empty | correct-shuffle | correct-random | correct-jitter |
|---|---:|---:|---:|---:|
| 4096 | +0.0584, 8/8 wins | +0.0680, 8/8 wins | +0.0709, 8/8 wins | +0.0668, 8/8 wins |
| 8192 | +0.0607, 8/8 wins | +0.0756, 8/8 wins | +0.0814, 8/8 wins | +0.0778, 8/8 wins |
| target_unique | +0.0611, 8/8 wins | +0.0776, 8/8 wins | +0.0840, 8/8 wins | +0.0815, 8/8 wins |

### 对比 Stage 1 Oracle

| run | correct top1 rate | correct rank mean | correct IoU mean | shuffle IoU mean | jitter IoU mean |
|---|---:|---:|---:|---:|---:|
| Stage 1 oracle condition | 0.125 | 2.792 | 0.0245 | 0.0262 | 0.0315 |
| Stage 2 strict masked inpainting | 1.000 | 1.000 | 0.1070 | 0.0332 | 0.0316 |

这是目前最强的正结果。Stage 1 oracle 中，点云很干净但 correct 仍输给 shuffle/jitter；Stage 2 strict masked inpainting 中，correct 在所有 sample 和所有 top-k 上都排第一。

这里尤其重要的是：`KNOWN_LOGIT_BOOST=0.0`，所以不是把 known points 在 final logits 阶段硬塞进 top-k 得到的结果。收益来自训练/采样路径中 known latent 进入 `x_t`，并按 timestep 被 clamp 回去。

### 结果解读

这次结果验证了路线判断：

1. `point as condition` 不够，soft condition 很容易被 TRELLIS sparse prior 吞掉；
2. `point as known sparse structure` 明显有效；
3. wrong prior 被 hard clamp 后也会覆盖自己的 prior 点，所以 `shuffle/random` 的 `prior_recall` 不低；
4. 但 wrong prior 会伤害 target IoU，而 correct prior 显著提高 target IoU；
5. 因此当前关键不是“模型是否会保留点”，而是“保留的点是否对应当前目标”。Strict masked inpainting 已经能通过 target IoU 把 correct 和 wrong prior 拉开。

`prior_recall` 不是越高越好，也不能跨 prior mode 直接比较。比如 `shuffle` 的 prior recall 高，表示模型确实被强制保留了错误 prior；它的 target IoU 低，说明错误 known structure 会伤害目标重建。这正是 inpainting 约束开始生效的证据。

### 当前限制

这仍然只是 oracle smoke，不能直接推出真实 AR/SLAM 点云也会稳定有效。原因：

- prior 来自 target sparse surface 筛选，不是真实 SLAM 稀疏点；
- 点云无 dropout/outlier/jitter；
- eval 只有 8 个样本；
- 目前指标主要是全局 IoU/recall，尚未拆分 known region 与 unknown completion region；
- 真实重建最终还要看 mesh，而不只是 sparse structure。

### 下一步建议

第一优先级：跑 Stage 2 strict oracle s200，扩大到更多样本，确认不是 20-step smoke 偶然结果。

```bash
cd /home/zjr/Tracker

GPU=1 \
MODE=s200 \
RUN_NAME=pointprior_pixal_v9_stage2_strictmask_s200 \
POINT_COUNT_CHOICES=1500 \
NUM_PRIOR_VIEWS_CHOICES=8 \
DROPOUT_MIN=0.0 \
DROPOUT_MAX=0.0 \
OUTLIER_RATIO=0.0 \
COORD_JITTER=0 \
KNOWN_FLOW_LOSS_WEIGHT=2.0 \
KNOWN_X0_LOSS_WEIGHT=1.0 \
KNOWN_USE_CONFIDENCE=0 \
KNOWN_LATENT_CLAMP_STRENGTH=1.0 \
KNOWN_LOGIT_BOOST=0.0 \
CLAMP_INITIAL_NOISE=1 \
MAX_STEPS=200 \
SAVE_EVERY=100 \
EVAL_STEPS=12 \
bash trellis_point_prior_mv/scripts/run_pixal_v9_point_prior_stage2.sh
```

结果查看：

```bash
cat /home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_strictmask_s200/eval/report.json
```

第二优先级：跑噪声 prior 版本，模拟真实 AR/SLAM 点云，先不打开 confidence。

```bash
cd /home/zjr/Tracker

GPU=1 \
MODE=smoke \
RUN_NAME=pointprior_pixal_v9_stage2_noisy_smoke \
POINT_COUNT_CHOICES=300,800,1500 \
NUM_PRIOR_VIEWS_CHOICES=4,8 \
DROPOUT_MAX=0.25 \
OUTLIER_RATIO=0.03 \
COORD_JITTER=1 \
KNOWN_FLOW_LOSS_WEIGHT=2.0 \
KNOWN_X0_LOSS_WEIGHT=1.0 \
KNOWN_USE_CONFIDENCE=0 \
KNOWN_LATENT_CLAMP_STRENGTH=1.0 \
KNOWN_LOGIT_BOOST=0.0 \
CLAMP_INITIAL_NOISE=1 \
EVAL_STEPS=12 \
bash trellis_point_prior_mv/scripts/run_pixal_v9_point_prior_stage2.sh
```

第三优先级：同样噪声 prior 下打开 confidence，对比 hard known 与 confidence-soft known。

```bash
cd /home/zjr/Tracker

GPU=1 \
MODE=smoke \
RUN_NAME=pointprior_pixal_v9_stage2_noisy_confidence_smoke \
POINT_COUNT_CHOICES=300,800,1500 \
NUM_PRIOR_VIEWS_CHOICES=4,8 \
DROPOUT_MAX=0.25 \
OUTLIER_RATIO=0.03 \
COORD_JITTER=1 \
KNOWN_FLOW_LOSS_WEIGHT=2.0 \
KNOWN_X0_LOSS_WEIGHT=1.0 \
KNOWN_USE_CONFIDENCE=1 \
KNOWN_LATENT_CLAMP_STRENGTH=1.0 \
KNOWN_LOGIT_BOOST=0.0 \
CLAMP_INITIAL_NOISE=1 \
EVAL_STEPS=12 \
bash trellis_point_prior_mv/scripts/run_pixal_v9_point_prior_stage2.sh
```

第四优先级：补 unknown-only 指标。当前全局 IoU 会包含 known region，下一步应该在 `eval_sparse_inpaint_stage2.py` 里增加：

- known-region recall；
- unknown-region IoU；
- unknown-region recall；
- wrong-prior 下 unknown completion 是否崩坏。

这样才能判断模型是真的补全 unknown，还是主要靠已知点覆盖拉高全局指标。

## Stage 2 当前结论

截至这轮结果，最合理的主线已经从“pose/point 作为 condition”切换为“point/pose 生成 known sparse structure，再做 masked sparse inpainting”。这条路线现在有明确正信号，值得继续扩大训练和接入真实 AR/SLAM 点云。

## Stage 2 Strict Masked Inpainting S200

运行目录：

```text
trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_strictmask_s200
```

本次设置：

- train samples：512
- val prior samples：64
- eval samples：0-31，共 32 个样本
- train steps：200
- eval steps：12
- `POINT_COUNT_CHOICES=1500`
- `NUM_PRIOR_VIEWS_CHOICES=8`
- `DROPOUT_MIN=0.0`
- `DROPOUT_MAX=0.0`
- `OUTLIER_RATIO=0.0`
- `COORD_JITTER=0`
- `KNOWN_USE_CONFIDENCE=0`
- `KNOWN_LATENT_CLAMP_STRENGTH=1.0`
- `KNOWN_LOGIT_BOOST=0.0`
- `CLAMP_INITIAL_NOISE=1`

### Prior 构建情况

| split | samples | support_failed | fallback_used | prior points mean | view count mean | dropout mean | support ratio mean | visible mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 512 | 0 | 0 | 1492.7 | 8.0 | 0.0 | 0.727 | 6.174 |
| val | 64 | 0 | 0 | 1500.0 | 8.0 | 0.0 | 0.718 | 6.235 |

构建侧仍然正常：没有 support failure，也没有 fallback。train 中少量样本 prior point 少于 1500，是因为 supported surface 本身不足 1500，例如最小为 847。

### Eval 总体结果

| prior mode | IoU mean | IoU median | recall mean | precision mean | prior recall mean |
|---|---:|---:|---:|---:|---:|
| correct | 0.1324 | 0.1291 | 0.2364 | 0.2616 | 0.5260 |
| empty | 0.0292 | 0.0263 | 0.0549 | 0.0669 | 0.0000 |
| shuffle | 0.0287 | 0.0242 | 0.0538 | 0.0656 | 0.5080 |
| random | 0.0178 | 0.0158 | 0.0331 | 0.0426 | 0.4960 |
| jitter | 0.0334 | 0.0309 | 0.0636 | 0.0754 | 0.4151 |

Correct rank：

| metric | value |
|---|---:|
| count | 96 |
| top1 | 96 |
| top1 rate | 1.000 |
| rank mean | 1.000 |
| rank median | 1.000 |

### Fixed Top-k 对比

| top-k | correct IoU | empty IoU | shuffle IoU | random IoU | jitter IoU |
|---|---:|---:|---:|---:|---:|
| 4096 | 0.1121 | 0.0207 | 0.0202 | 0.0125 | 0.0236 |
| 8192 | 0.1364 | 0.0319 | 0.0314 | 0.0192 | 0.0372 |
| target_unique | 0.1487 | 0.0351 | 0.0345 | 0.0216 | 0.0394 |

Paired delta：

| top-k | correct-empty | correct-shuffle | correct-random | correct-jitter |
|---|---:|---:|---:|---:|
| 4096 | +0.0915, 32/32 wins | +0.0919, 32/32 wins | +0.0996, 32/32 wins | +0.0886, 32/32 wins |
| 8192 | +0.1045, 32/32 wins | +0.1050, 32/32 wins | +0.1173, 32/32 wins | +0.0992, 32/32 wins |
| target_unique | +0.1136, 32/32 wins | +0.1142, 32/32 wins | +0.1271, 32/32 wins | +0.1093, 32/32 wins |

### 与 Smoke 对比

| run | eval samples | correct top1 rate | correct IoU mean | empty IoU mean | shuffle IoU mean | jitter IoU mean |
|---|---:|---:|---:|---:|---:|---:|
| Stage 2 strict smoke | 8 | 1.000 | 0.1070 | 0.0469 | 0.0332 | 0.0316 |
| Stage 2 strict s200 | 32 | 1.000 | 0.1324 | 0.0292 | 0.0287 | 0.0334 |

s200 没有破坏 smoke 的结论，反而更稳。扩大到 32 个 eval 样本、96 个 rank case 后，correct 仍然全部 top1，并且 correct 与 wrong prior 的差距进一步拉大。

需要注意：wrong prior 的 `prior_recall` 也不低，例如 shuffle 为 0.508。这不是坏事，而是说明 hard clamp 的确会保留输入 known structure。关键区别在于：错误 known structure 虽然被保留，但会显著降低 target IoU；正确 known structure 则显著提升 target IoU。这正是 masked inpainting 路线想验证的行为。

### 当前判断

Stage 2 strict masked inpainting 已经通过了 oracle 级别验证。它和 Stage 1 的差异非常明确：

- Stage 1：point 只是 condition，oracle prior 仍无法压过 shuffle/jitter；
- Stage 2：point 是 known structure，correct 在所有 top-k 和所有样本上稳定 top1；
- s200：正结果在更大训练/评测规模下仍成立。

因此后续不应该再把主要精力放在“pose/point condition 怎么设计”上，而应该转向“如何从真实 AR/SLAM 或多视图 pose/mask 得到足够好的 known structure”。

### 下一步建议

第一优先级：跑 noisy prior，测试从 oracle 走向真实点云的鲁棒性。先不要打开 confidence。

```bash
cd /home/zjr/Tracker

GPU=1 \
MODE=smoke \
RUN_NAME=pointprior_pixal_v9_stage2_noisy_smoke \
POINT_COUNT_CHOICES=300,800,1500 \
NUM_PRIOR_VIEWS_CHOICES=4,8 \
DROPOUT_MAX=0.25 \
OUTLIER_RATIO=0.03 \
COORD_JITTER=1 \
KNOWN_FLOW_LOSS_WEIGHT=2.0 \
KNOWN_X0_LOSS_WEIGHT=1.0 \
KNOWN_USE_CONFIDENCE=0 \
KNOWN_LATENT_CLAMP_STRENGTH=1.0 \
KNOWN_LOGIT_BOOST=0.0 \
CLAMP_INITIAL_NOISE=1 \
EVAL_STEPS=12 \
bash trellis_point_prior_mv/scripts/run_pixal_v9_point_prior_stage2.sh
```

结果查看：

```bash
cat /home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_noisy_smoke/eval/report.json
```

第二优先级：同样 noisy prior，打开 confidence，判断 confidence 是否能缓解 outlier/jitter。

```bash
cd /home/zjr/Tracker

GPU=1 \
MODE=smoke \
RUN_NAME=pointprior_pixal_v9_stage2_noisy_confidence_smoke \
POINT_COUNT_CHOICES=300,800,1500 \
NUM_PRIOR_VIEWS_CHOICES=4,8 \
DROPOUT_MAX=0.25 \
OUTLIER_RATIO=0.03 \
COORD_JITTER=1 \
KNOWN_FLOW_LOSS_WEIGHT=2.0 \
KNOWN_X0_LOSS_WEIGHT=1.0 \
KNOWN_USE_CONFIDENCE=1 \
KNOWN_LATENT_CLAMP_STRENGTH=1.0 \
KNOWN_LOGIT_BOOST=0.0 \
CLAMP_INITIAL_NOISE=1 \
EVAL_STEPS=12 \
bash trellis_point_prior_mv/scripts/run_pixal_v9_point_prior_stage2.sh
```

第三优先级：补 unknown-only 指标，再解释 noisy prior 的结果。当前全局 IoU 已经证明 strict oracle 有效，但真实任务更关心 unknown completion，而不只是 known region 被保留。建议在 `eval_sparse_inpaint_stage2.py` 增加：

- known prior voxel recall；
- target-minus-known unknown IoU；
- target-minus-known unknown recall；
- wrong-prior 下 unknown region 的损伤程度；
- per-sample correct-vs-shuffle unknown delta。

第四优先级：接真实 AR/SLAM 点云或 COLMAP 点云。当前 oracle prior 来自 target sparse surface，不是真实 SLAM 点。下一阶段应该把手机 AR 稀疏点云或 COLMAP 稀疏点云投到 TRELLIS 64 网格，先用同一套 Stage 2 strict inpainting eval 做 sanity check。

## S200 后结论

Stage 2 strict masked inpainting 已经不是“可能有希望”，而是当前最有希望的主线。接下来决定成败的不是 flow 是否能用点云，而是 known structure 的质量：真实 AR/SLAM 点云能不能经过 pose/mask/尺度对齐筛成足够干净的 object sparse prior。

## Stage 2 Noisy Prior Smoke

运行目录：

```text
trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_noisy_smoke
trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_noisy_confidence_smoke
```

两个 run 使用同一类 noisy prior：

- `POINT_COUNT_CHOICES=300,800,1500`
- `NUM_PRIOR_VIEWS_CHOICES=4,8`
- `DROPOUT_MAX=0.25`
- `OUTLIER_RATIO=0.03`
- `COORD_JITTER=1`
- `KNOWN_LATENT_CLAMP_STRENGTH=1.0`
- `KNOWN_LOGIT_BOOST=0.0`
- `CLAMP_INITIAL_NOISE=1`
- eval samples：0-7，共 8 个样本
- train steps：20

区别：

- `pointprior_pixal_v9_stage2_noisy_smoke`：`KNOWN_USE_CONFIDENCE=0`
- `pointprior_pixal_v9_stage2_noisy_confidence_smoke`：`KNOWN_USE_CONFIDENCE=1`

### Prior 构建情况

| run | split | samples | support_failed | fallback_used | prior points mean | view count mean | dropout mean | support ratio mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| noisy | train | 64 | 0 | 0 | 732.3 | 5.94 | 0.136 | 0.713 |
| noisy | val | 32 | 0 | 0 | 701.5 | 6.00 | 0.129 | 0.699 |
| noisy+confidence | train | 64 | 0 | 0 | 732.3 | 5.94 | 0.136 | 0.713 |
| noisy+confidence | val | 32 | 0 | 0 | 701.5 | 6.00 | 0.129 | 0.699 |

构建侧正常，两个 run 的 prior 数据相同；差异来自 inpainting target / timestep clamp 是否使用 confidence 软化 known region。

### Eval 总体结果

| run | correct top1 rate | correct IoU | empty IoU | shuffle IoU | random IoU | jitter IoU | correct prior recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| noisy, no confidence | 1.000 | 0.0676 | 0.0276 | 0.0292 | 0.0240 | 0.0389 | 0.6001 |
| noisy + confidence | 1.000 | 0.0718 | 0.0485 | 0.0423 | 0.0413 | 0.0439 | 0.4887 |

两个 noisy run 都保持了 correct 100% top1，说明 Stage 2 strict masked inpainting 对中等 dropout、jitter、outlier 仍有鲁棒性。这个结果很关键：路线没有只在完全 oracle prior 下成立。

confidence 的效果不是简单更好：

- correct IoU 从 `0.0676` 略升到 `0.0718`；
- correct prior recall 从 `0.6001` 降到 `0.4887`；
- empty / shuffle / random / jitter 的 IoU 也整体升高；
- correct 对 wrong prior 的 margin 变小。

因此当前 confidence 更像是“软化 known clamp，让生成更接近无约束 TRELLIS prior”，它可能减少 noisy point 的过约束，但也削弱了 known structure 的区分力。

### Fixed Top-k 对比

No confidence：

| top-k | correct IoU | empty IoU | shuffle IoU | random IoU | jitter IoU |
|---|---:|---:|---:|---:|---:|
| 4096 | 0.0541 | 0.0193 | 0.0197 | 0.0159 | 0.0293 |
| 8192 | 0.0720 | 0.0298 | 0.0314 | 0.0257 | 0.0412 |
| target_unique | 0.0768 | 0.0338 | 0.0366 | 0.0304 | 0.0463 |

No confidence paired delta：

| top-k | correct-empty | correct-shuffle | correct-random | correct-jitter |
|---|---:|---:|---:|---:|
| 4096 | +0.0348, 8/8 wins | +0.0343, 8/8 wins | +0.0382, 8/8 wins | +0.0248, 8/8 wins |
| 8192 | +0.0421, 8/8 wins | +0.0406, 8/8 wins | +0.0463, 8/8 wins | +0.0307, 8/8 wins |
| target_unique | +0.0431, 8/8 wins | +0.0402, 8/8 wins | +0.0464, 8/8 wins | +0.0305, 8/8 wins |

With confidence：

| top-k | correct IoU | empty IoU | shuffle IoU | random IoU | jitter IoU |
|---|---:|---:|---:|---:|---:|
| 4096 | 0.0573 | 0.0333 | 0.0286 | 0.0275 | 0.0323 |
| 8192 | 0.0751 | 0.0523 | 0.0456 | 0.0444 | 0.0457 |
| target_unique | 0.0831 | 0.0598 | 0.0527 | 0.0521 | 0.0537 |

With confidence paired delta：

| top-k | correct-empty | correct-shuffle | correct-random | correct-jitter |
|---|---:|---:|---:|---:|
| 4096 | +0.0240, 8/8 wins | +0.0287, 8/8 wins | +0.0298, 8/8 wins | +0.0250, 8/8 wins |
| 8192 | +0.0228, 8/8 wins | +0.0294, 8/8 wins | +0.0307, 8/8 wins | +0.0294, 8/8 wins |
| target_unique | +0.0233, 8/8 wins | +0.0305, 8/8 wins | +0.0310, 8/8 wins | +0.0294, 8/8 wins |

### 与 Oracle S200 对比

| run | eval samples | correct top1 rate | correct IoU | shuffle IoU | jitter IoU | correct prior recall |
|---|---:|---:|---:|---:|---:|---:|
| strict oracle s200 | 32 | 1.000 | 0.1324 | 0.0287 | 0.0334 | 0.5260 |
| noisy no confidence smoke | 8 | 1.000 | 0.0676 | 0.0292 | 0.0389 | 0.6001 |
| noisy confidence smoke | 8 | 1.000 | 0.0718 | 0.0423 | 0.0439 | 0.4887 |

Noisy prior 会显著降低 correct IoU，这是预期内的：点更少、有 dropout、有 jitter、有 outlier。但 correct 仍然稳定 top1，说明当前 masked inpainting 的方向对真实点云噪声有一定余量。

### 当前判断

1. Stage 2 不只在 oracle 下成立，noisy prior 下也成立。
2. 默认仍建议 `KNOWN_USE_CONFIDENCE=0` 作为主线，因为它保留了更硬的 known structure 区分力，correct-vs-wrong margin 更大。
3. `KNOWN_USE_CONFIDENCE=1` 可以作为真实 AR/SLAM 点云的备选设置：它可能让结果更保守、更像原 TRELLIS prior，但会降低 known point recall 和判别 margin。
4. 下一步不应该回到 condition 设计，而应该推进两个方向：
   - 扩大 noisy prior 到 s200；
   - 补 unknown-only 指标，避免全局 IoU 被 known region 覆盖掩盖。

### 下一步建议

第一优先级：跑 noisy no-confidence s200。这个是最接近“真实点云但仍保持硬 known structure”的主线。

```bash
cd /home/zjr/Tracker

GPU=1 \
MODE=s200 \
RUN_NAME=pointprior_pixal_v9_stage2_noisy_s200 \
POINT_COUNT_CHOICES=300,800,1500 \
NUM_PRIOR_VIEWS_CHOICES=4,8 \
DROPOUT_MAX=0.25 \
OUTLIER_RATIO=0.03 \
COORD_JITTER=1 \
KNOWN_FLOW_LOSS_WEIGHT=2.0 \
KNOWN_X0_LOSS_WEIGHT=1.0 \
KNOWN_USE_CONFIDENCE=0 \
KNOWN_LATENT_CLAMP_STRENGTH=1.0 \
KNOWN_LOGIT_BOOST=0.0 \
CLAMP_INITIAL_NOISE=1 \
MAX_STEPS=200 \
SAVE_EVERY=100 \
EVAL_STEPS=12 \
bash trellis_point_prior_mv/scripts/run_pixal_v9_point_prior_stage2.sh
```

第二优先级：补 unknown-only eval。建议在 `eval_sparse_inpaint_stage2.py` 中新增：

- `known_target_iou`
- `unknown_target_iou`
- `unknown_target_recall`
- `known_prior_recall`
- `unknown correct-vs-shuffle delta`

尤其要看 noisy prior 下 correct 的 unknown completion 是否仍优于 wrong prior。如果只是 known region 拉高全局 IoU，后续 mesh 阶段未必能受益；如果 unknown region 也提升，说明它真的在补全。

第三优先级：开始接真实 COLMAP/AR sparse points。当前 noisy prior 仍然来自 target surface 的模拟噪声，不是真实 SLAM/COLMAP 分布。下一步应该构建：

```text
rgb + mask + camera pose + sparse points
-> object point filtering
-> TRELLIS 64^3 voxel known_coords
-> Stage 2 masked inpainting eval
```

可以先从 `CoarseModel/datasets/heimei` 的 COLMAP 点云开始，因为它比手机 AR SLAM 点云更容易离线复现和调试。
关掉 latent clamp 后，correct prior 优势基本消失，prior_recall 从 0.526 降到 0.054，correct_rank top1_rate 从 1.0 降到 0.0417。

## Stage 2 Noisy Prior S200

运行目录：

```text
trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_noisy_s200
```

本次设置：

- train samples：512
- val prior samples：64
- eval samples：0-31，共 32 个样本
- train steps：200
- eval steps：12
- `POINT_COUNT_CHOICES=300,800,1500`
- `NUM_PRIOR_VIEWS_CHOICES=4,8`
- `DROPOUT_MAX=0.25`
- `OUTLIER_RATIO=0.03`
- `COORD_JITTER=1`
- `KNOWN_USE_CONFIDENCE=0`
- `KNOWN_LATENT_CLAMP_STRENGTH=1.0`
- `KNOWN_LOGIT_BOOST=0.0`
- `CLAMP_INITIAL_NOISE=1`

### Prior 构建情况

| split | samples | support_failed | fallback_used | prior points mean | view count mean | dropout mean | support ratio mean | visible mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 512 | 0 | 0 | 756.7 | 6.10 | 0.125 | 0.724 | 4.707 |
| val | 64 | 0 | 0 | 734.9 | 5.94 | 0.136 | 0.718 | 4.643 |

构建侧正常，没有 support failure / fallback。相比 oracle s200，这个 run 的 prior 点数约减半，并引入了 dropout、jitter、outlier，因此目标本来就更难。

### Eval 总体结果

| prior mode | IoU mean | IoU median | recall mean | precision mean | prior recall mean |
|---|---:|---:|---:|---:|---:|
| correct | 0.0385 | 0.0345 | 0.0709 | 0.0895 | 0.4280 |
| empty | 0.0237 | 0.0212 | 0.0442 | 0.0558 | 0.0000 |
| shuffle | 0.0175 | 0.0144 | 0.0327 | 0.0414 | 0.4159 |
| random | 0.0157 | 0.0134 | 0.0291 | 0.0381 | 0.5962 |
| jitter | 0.0248 | 0.0216 | 0.0457 | 0.0587 | 0.4922 |

Correct rank：

| metric | value |
|---|---:|
| count | 96 |
| top1 | 91 |
| top1 rate | 0.948 |
| rank mean | 1.073 |
| rank median | 1.000 |

### Fixed Top-k 对比

| top-k | correct IoU | empty IoU | shuffle IoU | random IoU | jitter IoU |
|---|---:|---:|---:|---:|---:|
| 4096 | 0.0274 | 0.0170 | 0.0112 | 0.0108 | 0.0176 |
| 8192 | 0.0426 | 0.0255 | 0.0191 | 0.0167 | 0.0267 |
| target_unique | 0.0455 | 0.0286 | 0.0224 | 0.0196 | 0.0299 |

Paired delta：

| top-k | correct-empty | correct-shuffle | correct-random | correct-jitter |
|---|---:|---:|---:|---:|
| 4096 | +0.0105, 29/32 wins | +0.0162, 32/32 wins | +0.0167, 32/32 wins | +0.0098, 31/32 wins |
| 8192 | +0.0171, 32/32 wins | +0.0236, 32/32 wins | +0.0259, 32/32 wins | +0.0159, 32/32 wins |
| target_unique | +0.0169, 30/32 wins | +0.0231, 32/32 wins | +0.0259, 31/32 wins | +0.0155, 32/32 wins |

### 与 Noisy Smoke 的同索引对比

为了排除 eval 样本范围差异，我把 noisy s200 report 也切到 0-7，与 noisy smoke 的 0-7 同口径比较。

| run | eval samples | correct top1 rate | correct IoU | empty IoU | shuffle IoU | jitter IoU | correct prior recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| noisy smoke | 0-7 | 1.000 | 0.0676 | 0.0276 | 0.0292 | 0.0389 | 0.6001 |
| noisy s200 subset | 0-7 | 1.000 | 0.0409 | 0.0253 | 0.0183 | 0.0251 | 0.4319 |

这个对比说明：下降不只是因为 s200 评了更多样本。同样 0-7 上，s200 的 correct IoU 和 prior recall 也明显低于 20-step smoke。

### 当前判断

你的直觉是对的：noisy hard clamp 继续长训后，效果确实下降。

但它不是彻底失败：

- correct rank 仍然很强：`top1_rate=0.948`；
- correct 仍然稳定压过 shuffle/random/jitter；
- 下降主要体现在 target IoU、target recall、prior recall 和 margin 缩小。

这更像是 hard noisy clamp 长训带来的过约束/过拟合问题：模型在训练中长期被迫把 noisy partial latent 当作硬 known target，导致整体补全质量下降。相比 smoke，s200 可能学会了更严格服从 noisy known structure，但 unknown completion 被牺牲了。

另一个佐证是 `prior_recall`：noisy smoke correct prior recall 为 `0.6001`，noisy s200 降到 `0.4280`。这说明不是“长训保留点更强”，反而可能在 noisy prior 下让 sparse logits 分布变得更保守或更分散。

### 下一步建议

第一优先级：先评估 step100 checkpoint，确认是否存在训练步数最佳点。当前目录里有：

```text
trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_noisy_s200/checkpoints/ss-pointprior-stage2-epoch=00-step=100.ckpt
trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_noisy_s200/checkpoints/ss-pointprior-stage2-epoch=00-step=200.ckpt
```

命令：

```bash
cd /home/zjr/Tracker

CUDA_VISIBLE_DEVICES=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/zjr/anaconda3/envs/reconviagen/bin/python -u trellis_point_prior_mv/eval_sparse_inpaint_stage2.py \
  --manifest /home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_noisy_s200/data/val/manifest.json \
  --checkpoint /home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_noisy_s200/checkpoints/ss-pointprior-stage2-epoch=00-step=100.ckpt \
  --output_dir /home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_noisy_s200/eval_step100 \
  --weights microsoft/TRELLIS-image-large \
  --indices 0-31 \
  --prior_modes correct,empty,shuffle,random,jitter \
  --fixed_topk 4096,8192,target_unique \
  --steps 12 \
  --guidance_strength 1.0 \
  --known_latent_clamp_strength 1.0 \
  --known_logit_boost 0.0 \
  --no_clamp_initial_noise
```

注意：上面临时用了 `--no_clamp_initial_noise` 作为诊断，目的是减少初始噪声阶段的硬注入。如果要完全复现当前设置，把最后一行改成：

```bash
  --clamp_initial_noise
```

第二优先级：做 eval-time clamp strength sweep。先不用重训，用当前 noisy_s200 checkpoint 看 `0.25/0.5/0.75/1.0` 哪个更稳。

```bash
cd /home/zjr/Tracker

for CLAMP in 0.25 0.5 0.75 1.0; do
  CUDA_VISIBLE_DEVICES=1 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn \
  SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib \
  NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/zjr/anaconda3/envs/reconviagen/bin/python -u trellis_point_prior_mv/eval_sparse_inpaint_stage2.py \
    --manifest /home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_noisy_s200/data/val/manifest.json \
    --checkpoint /home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_noisy_s200/checkpoints/last.ckpt \
    --output_dir /home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_noisy_s200/eval_clamp${CLAMP//./} \
    --weights microsoft/TRELLIS-image-large \
    --indices 0-31 \
    --prior_modes correct,empty,shuffle,random,jitter \
    --fixed_topk 4096,8192,target_unique \
    --steps 12 \
    --guidance_strength 1.0 \
    --known_latent_clamp_strength "$CLAMP" \
    --known_logit_boost 0.0 \
    --clamp_initial_noise
done
```

第三优先级：不要继续简单把 noisy hard clamp 训到更长。更合理的是改训练策略：

1. 从 oracle strict s200 checkpoint 出发，低学习率 noisy fine-tune 20-50 step；
2. 或者做 curriculum：先 easy/noiseless，再逐步增加 dropout/jitter/outlier；
3. 或者训练时 `KNOWN_USE_CONFIDENCE=1`，但 eval 时同时比较 hard/soft clamp；
4. 或者降低 `KNOWN_X0_LOSS_WEIGHT`，避免 known noisy latent 对 x0 target 约束过强。

第四优先级：补 unknown-only 指标后再判断 mesh 价值。现在 noisy_s200 的全局 IoU 下降，可能是 known noisy region 伤害 unknown completion，也可能是 top-k 分布变化。需要把 target 拆成：

- known target region；
- unknown target region；
- wrong prior injected region。

没有这个拆分，继续调参容易误判。

## Stage 2 Noisy Prior Step100 与 Clamp Sweep 复查

时间：2026-06-20

本轮补充读取了以下结果：

- `/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_noisy_s200/eval_step100/report.json`
- `/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_noisy_s200/eval_step100_clampinit/report.json`
- `/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_noisy_s200/eval_clamp025/report.json`
- `/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_noisy_s200/eval_clamp05/report.json`
- `/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_noisy_s200/eval_clamp075/report.json`
- `/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_noisy_s200/eval_clamp10/report.json`
- 对照：`/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_noisy_s200/eval/report.json`

### 主结果

| run | checkpoint | clamp | init clamp | topk | correct IoU | recall | prior recall | top1 rate | rank mean |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| s200 原始 eval | `last.ckpt` | 1.0 | true | 4096 | 0.0274 | 0.0413 | 0.3190 | 0.906 | 1.12 |
| s200 原始 eval | `last.ckpt` | 1.0 | true | 8192 | 0.0426 | 0.0853 | 0.4873 | 1.000 | 1.00 |
| s200 原始 eval | `last.ckpt` | 1.0 | true | target_unique | 0.0454 | 0.0860 | 0.4777 | 0.938 | 1.09 |
| step100 no init | `step=100.ckpt` | 1.0 | false | 4096 | 0.0218 | 0.0323 | 0.2541 | 0.688 | 1.38 |
| step100 no init | `step=100.ckpt` | 1.0 | false | 8192 | 0.0336 | 0.0656 | 0.3897 | 0.906 | 1.09 |
| step100 no init | `step=100.ckpt` | 1.0 | false | target_unique | 0.0366 | 0.0696 | 0.3985 | 0.688 | 1.34 |
| step100 init | `step=100.ckpt` | 1.0 | true | 4096 | 0.0218 | 0.0323 | 0.2541 | 0.688 | 1.38 |
| step100 init | `step=100.ckpt` | 1.0 | true | 8192 | 0.0336 | 0.0656 | 0.3897 | 0.906 | 1.09 |
| step100 init | `step=100.ckpt` | 1.0 | true | target_unique | 0.0366 | 0.0696 | 0.3985 | 0.688 | 1.34 |
| s200 clamp 0.25 | `last.ckpt` | 0.25 | true | 4096 | 0.0137 | 0.0205 | 0.1217 | 0.188 | 2.22 |
| s200 clamp 0.25 | `last.ckpt` | 0.25 | true | 8192 | 0.0239 | 0.0470 | 0.2237 | 0.469 | 1.59 |
| s200 clamp 0.25 | `last.ckpt` | 0.25 | true | target_unique | 0.0275 | 0.0527 | 0.2387 | 0.375 | 1.78 |
| s200 clamp 0.5 | `last.ckpt` | 0.5 | true | 4096 | 0.0199 | 0.0297 | 0.2331 | 0.656 | 1.44 |
| s200 clamp 0.5 | `last.ckpt` | 0.5 | true | 8192 | 0.0326 | 0.0646 | 0.3743 | 0.875 | 1.12 |
| s200 clamp 0.5 | `last.ckpt` | 0.5 | true | target_unique | 0.0356 | 0.0678 | 0.3797 | 0.719 | 1.34 |
| s200 clamp 0.75 | `last.ckpt` | 0.75 | true | 4096 | 0.0246 | 0.0370 | 0.2909 | 0.844 | 1.19 |
| s200 clamp 0.75 | `last.ckpt` | 0.75 | true | 8192 | 0.0385 | 0.0768 | 0.4531 | 1.000 | 1.00 |
| s200 clamp 0.75 | `last.ckpt` | 0.75 | true | target_unique | 0.0414 | 0.0787 | 0.4466 | 0.875 | 1.16 |
| s200 clamp 1.0 | `last.ckpt` | 1.0 | true | 4096 | 0.0274 | 0.0413 | 0.3190 | 0.906 | 1.12 |
| s200 clamp 1.0 | `last.ckpt` | 1.0 | true | 8192 | 0.0426 | 0.0853 | 0.4873 | 1.000 | 1.00 |
| s200 clamp 1.0 | `last.ckpt` | 1.0 | true | target_unique | 0.0454 | 0.0860 | 0.4777 | 0.938 | 1.09 |

### Paired Delta

`target_unique` 下 correct 相对 wrong pose / wrong prior 的 IoU 差值：

| run | wrong | delta IoU | correct wins |
|---|---|---:|---:|
| s200 原始 eval | empty | 0.0169 | 30/32 |
| s200 原始 eval | shuffle | 0.0231 | 32/32 |
| s200 原始 eval | random | 0.0259 | 31/32 |
| s200 原始 eval | jitter | 0.0155 | 32/32 |
| step100 no init | empty | 0.0103 | 23/32 |
| step100 no init | shuffle | 0.0154 | 32/32 |
| step100 no init | random | 0.0162 | 31/32 |
| step100 no init | jitter | 0.0098 | 31/32 |
| step100 init | empty | 0.0103 | 23/32 |
| step100 init | shuffle | 0.0154 | 32/32 |
| step100 init | random | 0.0162 | 31/32 |
| step100 init | jitter | 0.0098 | 31/32 |
| s200 clamp 0.25 | empty | -0.0011 | 12/32 |
| s200 clamp 0.25 | shuffle | 0.0075 | 29/32 |
| s200 clamp 0.25 | random | 0.0081 | 30/32 |
| s200 clamp 0.25 | jitter | 0.0061 | 32/32 |
| s200 clamp 0.5 | empty | 0.0070 | 22/32 |
| s200 clamp 0.5 | shuffle | 0.0147 | 32/32 |
| s200 clamp 0.5 | random | 0.0164 | 31/32 |
| s200 clamp 0.5 | jitter | 0.0101 | 31/32 |
| s200 clamp 0.75 | empty | 0.0129 | 28/32 |
| s200 clamp 0.75 | shuffle | 0.0197 | 32/32 |
| s200 clamp 0.75 | random | 0.0221 | 31/32 |
| s200 clamp 0.75 | jitter | 0.0134 | 32/32 |
| s200 clamp 1.0 | empty | 0.0169 | 30/32 |
| s200 clamp 1.0 | shuffle | 0.0231 | 32/32 |
| s200 clamp 1.0 | random | 0.0259 | 31/32 |
| s200 clamp 1.0 | jitter | 0.0155 | 32/32 |

### 结论

这轮结果修正了上一轮对 `s200` 的一个判断：`s200` 的确比 `smoke` 的全局 IoU 低，但在同一套 noisy 数据和 eval 配置里，`step200` 并不比 `step100` 差，反而明显更好。也就是说，问题不是简单的“100 到 200 训练过头”，而是 noisy hard clamp 这一训练分布本身上限偏低；在这个分布内继续训练到 200 还能提升排序和 IoU。

`step100_clampinit` 和 `step100_no_clampinit` 的指标完全一致，说明当前采样流程里初始 noise 是否先 clamp 对最终 fixed top-k 几乎没有影响。真正起作用的是每个 reverse step 后的 known latent reinjection，以及训练得到的条件响应。

Clamp sweep 的趋势很明确：`0.25 < 0.5 < 0.75 < 1.0`。减弱 eval-time clamp 会同时降低 correct IoU、prior recall 和 correct top1。`clamp=0.25` 甚至在 `target_unique` 下输给 empty，说明当前模型还没有学到“只靠条件 token 自己把 prior 传播进结构”的能力，仍然强依赖 hard latent clamp。

因此，当前不是应该把 clamp 减弱，而是应该承认这条路线目前更像“partial sparse latent inpainting with hard known constraints”，还不是完整可泛化的 Points-to-3D。它能稳定区分 correct / wrong prior，但 noisy prior 下生成质量上限明显低于 strict oracle。

### 下一步建议

第一优先级：补 known / unknown 分区指标。现在必须回答下降来自哪里：

1. correct prior 已知区域是否被保住；
2. unknown target 区域是否被补出来；
3. wrong/random/jitter prior 注入的错误区域是否污染输出。

建议新增 eval 输出：

- `known_target_recall`
- `unknown_target_recall`
- `known_prior_recall`
- `wrong_prior_leak_rate`
- `unknown_iou`

第二优先级：不要继续跑 `clamp < 1.0` 的推理配置。当前结果说明弱 clamp 没有收益，后续默认保留：

```bash
--known_latent_clamp_strength 1.0
--clamp_initial_noise
--known_logit_boost 0.0
```

第三优先级：下一轮训练不要再直接 noisy hard clamp 长训。更合适的是 curriculum：

1. 用 strict/oracle 或 easy prior 训练到稳定；
2. 再用 noisy prior 小步 fine-tune；
3. fine-tune 时降低 `KNOWN_X0_LOSS_WEIGHT`，例如从 `1.0` 降到 `0.25/0.5`；
4. 保留 hard clamp，但只把 known 区域当作边界条件，不要让 noisy known 区域过强主导全局 x0。

第四优先级：如果目标是接近论文式 Points-to-3D，需要进一步把 partial latent 作为扩散过程的 known x_t 边界，而不仅是 condition + post-step clamp。当前实验已经证明：只把 point prior 做成 condition token 不够，弱 clamp 也不够；有效信号主要来自硬约束区域。

## Stage 2 Noisy Prior Unknown 指标复查与公平切分修正

时间：2026-06-20

读取结果：

- `/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_noisy_s200/eval_unknown_metrics/report.json`

### 旧 unknown 指标结果

旧版 `unknown_*` 使用的是 condition split，也就是每个 mode 都用自己的输入 prior 切 known/unknown：

- correct mode 用 correct prior；
- shuffle mode 用 shuffle prior；
- random mode 用 random prior；
- jitter mode 用 jitter prior；
- empty mode 没有 known region，因此 unknown 基本等于全体 target/pred。

这套指标适合分析“当前输入 prior 自己的 known clamp 造成了什么影响”，但不适合直接比较 `correct.unknown_iou` 和 `shuffle.unknown_iou`，也不适合直接解释 `correct_rank_unknown_iou`。

旧版结果如下：

| topk | mode | IoU | known IoU | condition unknown IoU | condition unknown recall | known prior recall | wrong leak |
|---|---|---:|---:|---:|---:|---:|---:|
| 4096 | correct | 0.0274 | 0.0500 | 0.0119 | 0.0251 | 0.3190 | 0.3398 |
| 4096 | empty | 0.0170 | 0.0000 | 0.0170 | 0.0260 | 0.0000 | 0.0000 |
| 4096 | shuffle | 0.0112 | 0.0242 | 0.0082 | 0.0124 | 0.3045 | 0.3086 |
| 4096 | random | 0.0108 | 0.0250 | 0.0066 | 0.0087 | 0.5152 | 0.5123 |
| 4096 | jitter | 0.0176 | 0.0306 | 0.0104 | 0.0202 | 0.3877 | 0.4022 |
| 8192 | correct | 0.0426 | 0.0896 | 0.0172 | 0.0529 | 0.4873 | 0.4993 |
| 8192 | empty | 0.0255 | 0.0000 | 0.0255 | 0.0513 | 0.0000 | 0.0000 |
| 8192 | shuffle | 0.0191 | 0.0358 | 0.0149 | 0.0296 | 0.4718 | 0.4760 |
| 8192 | random | 0.0167 | 0.0290 | 0.0127 | 0.0214 | 0.6416 | 0.6386 |
| 8192 | jitter | 0.0267 | 0.0480 | 0.0164 | 0.0455 | 0.5520 | 0.5665 |
| target_unique | correct | 0.0454 | 0.0913 | 0.0198 | 0.0551 | 0.4777 | 0.4959 |
| target_unique | empty | 0.0286 | 0.0000 | 0.0286 | 0.0551 | 0.0000 | 0.0000 |
| target_unique | shuffle | 0.0223 | 0.0388 | 0.0182 | 0.0350 | 0.4715 | 0.4749 |
| target_unique | random | 0.0196 | 0.0305 | 0.0158 | 0.0266 | 0.6319 | 0.6289 |
| target_unique | jitter | 0.0299 | 0.0518 | 0.0194 | 0.0475 | 0.5368 | 0.5491 |

整体全局 IoU rank 仍然强：

- `correct_rank.top1_rate = 0.9479`
- `correct_rank.rank_mean = 1.073`

但旧 condition split 的 unknown rank 很差：

- `correct_rank_unknown_iou.top1_rate = 0.0729`
- `correct_rank_unknown_iou.rank_mean = 2.823`
- `correct_rank_unknown_target_recall.top1_rate = 0.4375`
- `correct_rank_unknown_target_recall.rank_mean = 1.781`

这个不能简单解读为“correct 对 unknown 没用”。因为 empty 的 unknown 是全体 target/pred，而 correct 的 unknown 是剔除了 correct observed known cell 后的剩余区域；shuffle/random/jitter 也各自剔除了不同区域。因此这些 unknown region 不是同一个集合。

### 当前可得结论

旧指标仍然说明两件事：

1. hard clamp 主要提升 known/condition 区域。比如 target_unique 下 correct `known_iou=0.0913`，明显高于 condition unknown IoU `0.0198`。
2. wrong prior 也会被 hard clamp 保留。比如 shuffle/random/jitter 的 `known_prior_recall` 都不低，说明输入错误 prior 会真实进入输出，并可能污染结构。

但旧指标不能回答真正关键的问题：

> 在同一个 observed known region 之外，correct prior 是否比 shuffle/random/jitter 更能补全 unknown？

### 代码修正

已修改：

- `/home/zjr/Tracker/trellis_point_prior_mv/eval_sparse_inpaint_stage2.py`
- `/home/zjr/Tracker/trellis_point_prior_mv/命令说明.txt`

保留旧字段：

- `known_*`
- `unknown_*`
- `exact_unknown_*`

同时新增公平 common observed split：

- `obs_known_*`
- `obs_unknown_*`

定义：

```text
obs_known region = correct_prior 映射到 16^3 latent cell 后覆盖的 64^3 区域
obs_unknown region = target/pred 中不落入 obs_known 的部分
```

所有 mode 都用同一个 `correct_prior` 切 `obs_known/obs_unknown`。因此以下指标才适合做 correct-vs-wrong 的公平比较：

- `obs_unknown_iou`
- `obs_unknown_target_recall`
- `correct_rank_obs_unknown_iou`
- `correct_rank_obs_unknown_target_recall`
- `paired_delta[*].obs_unknown_iou`
- `paired_delta[*].obs_unknown_target_recall`

### 下一步命令

不需要重训，直接重跑 eval：

```bash
cd /home/zjr/Tracker

CUDA_VISIBLE_DEVICES=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/zjr/anaconda3/envs/reconviagen/bin/python -u trellis_point_prior_mv/eval_sparse_inpaint_stage2.py \
  --manifest /home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_noisy_s200/data/val/manifest.json \
  --checkpoint /home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_noisy_s200/checkpoints/last.ckpt \
  --output_dir /home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_noisy_s200/eval_obs_unknown_metrics \
  --weights microsoft/TRELLIS-image-large \
  --indices 0-31 \
  --prior_modes correct,empty,shuffle,random,jitter \
  --fixed_topk 4096,8192,target_unique \
  --steps 12 \
  --guidance_strength 1.0 \
  --known_latent_clamp_strength 1.0 \
  --known_logit_boost 0.0 \
  --known_conf_power 1.0 \
  --lora_rank 64 \
  --lora_alpha 128 \
  --clamp_initial_noise
```

结果查看：

```bash
cat /home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_noisy_s200/eval_obs_unknown_metrics/report.json
```

快速抽取公平指标：

```bash
/home/zjr/anaconda3/envs/reconviagen/bin/python -c "import json; p='/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_noisy_s200/eval_obs_unknown_metrics/report.json'; d=json.load(open(p)); s=d['summary']; print('correct obs_unknown_iou=', s['correct']['obs_unknown_iou_mean']); print('correct obs_unknown_recall=', s['correct']['obs_unknown_target_recall_mean']); print('obs unknown rank=', s['correct_rank_obs_unknown_iou']); print('target_unique/shuffle obs_unknown delta=', s['paired_delta']['target_unique/shuffle']['obs_unknown_iou']); print('target_unique/empty obs_unknown delta=', s['paired_delta']['target_unique/empty']['obs_unknown_iou'])"
```

后续判断标准改为：

1. 全局 IoU / correct rank 判断 prior 是否整体有效；
2. `obs_unknown_iou` / `obs_unknown_target_recall` 判断 correct prior 是否帮助补 observed 之外的区域；
3. 旧的 `unknown_*` 只用于分析当前输入 prior 自身的 hard clamp 影响，不再作为 correct-vs-wrong 的公平 rank 依据。

## Stage 2 Noisy Prior 公平 Obs-Unknown 结果

时间：2026-06-20

读取结果：

- `/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_noisy_s200/eval_obs_unknown_metrics/report.json`

评测设置：

- checkpoint: `pointprior_pixal_v9_stage2_noisy_s200/checkpoints/last.ckpt`
- indices: `0-31`
- topk: `4096,8192,target_unique`
- prior modes: `correct,empty,shuffle,random,jitter`
- clamp: `known_latent_clamp_strength=1.0`
- logit boost: `0.0`

### 主表

| topk | mode | IoU | obs known IoU | obs unknown IoU | obs unknown recall | obs known prior recall | obs wrong leak |
|---|---|---:|---:|---:|---:|---:|---:|
| 4096 | correct | 0.0274 | 0.0500 | 0.0119 | 0.0251 | 0.3190 | 0.3398 |
| 4096 | empty | 0.0170 | 0.0255 | 0.0108 | 0.0253 | 0.0268 | 0.0262 |
| 4096 | shuffle | 0.0112 | 0.0161 | 0.0078 | 0.0183 | 0.0171 | 0.0180 |
| 4096 | random | 0.0108 | 0.0154 | 0.0075 | 0.0171 | 0.0158 | 0.0178 |
| 4096 | jitter | 0.0176 | 0.0239 | 0.0136 | 0.0311 | 0.0251 | 0.0266 |
| 8192 | correct | 0.0426 | 0.0896 | 0.0172 | 0.0529 | 0.4873 | 0.4993 |
| 8192 | empty | 0.0255 | 0.0468 | 0.0150 | 0.0491 | 0.0544 | 0.0529 |
| 8192 | shuffle | 0.0191 | 0.0345 | 0.0117 | 0.0379 | 0.0416 | 0.0425 |
| 8192 | random | 0.0167 | 0.0300 | 0.0105 | 0.0356 | 0.0323 | 0.0352 |
| 8192 | jitter | 0.0267 | 0.0452 | 0.0178 | 0.0587 | 0.0535 | 0.0570 |
| target_unique | correct | 0.0454 | 0.0913 | 0.0198 | 0.0551 | 0.4777 | 0.4959 |
| target_unique | empty | 0.0286 | 0.0501 | 0.0171 | 0.0516 | 0.0586 | 0.0573 |
| target_unique | shuffle | 0.0223 | 0.0385 | 0.0141 | 0.0429 | 0.0454 | 0.0455 |
| target_unique | random | 0.0196 | 0.0340 | 0.0124 | 0.0392 | 0.0379 | 0.0417 |
| target_unique | jitter | 0.0299 | 0.0495 | 0.0201 | 0.0607 | 0.0568 | 0.0587 |

### Rank

| metric | top1 | top1 rate | rank mean | rank median |
|---|---:|---:|---:|---:|
| global IoU | 91/96 | 0.9479 | 1.073 | 1.0 |
| condition unknown IoU | 7/96 | 0.0729 | 2.823 | 3.0 |
| condition unknown recall | 42/96 | 0.4375 | 1.781 | 2.0 |
| obs unknown IoU | 31/96 | 0.3229 | 2.115 | 2.0 |
| obs unknown recall | 23/96 | 0.2396 | 2.323 | 2.0 |

### Target-Unique Paired Delta

| wrong | global delta IoU | global wins | obs unknown delta IoU | wins | obs unknown recall delta | wins | obs known delta IoU | wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| empty | 0.0169 | 30/32 | 0.0027 | 23/32 | 0.0035 | 19/32 | 0.0412 | 31/32 |
| shuffle | 0.0231 | 32/32 | 0.0057 | 29/32 | 0.0122 | 28/32 | 0.0528 | 32/32 |
| random | 0.0259 | 31/32 | 0.0074 | 30/32 | 0.0158 | 28/32 | 0.0573 | 31/32 |
| jitter | 0.0155 | 32/32 | -0.0003 | 13/32 | -0.0056 | 10/32 | 0.0418 | 32/32 |

### 结论

公平 `obs_unknown_*` 指标确认了一个更细的结论：

1. `correct` 的全局 sparse 结构排序仍然很强，`global IoU top1_rate=0.9479`。
2. `correct` 对 observed known 区域非常有效，target_unique 下相对 shuffle 的 `obs_known_iou +0.0528`，32/32 胜；相对 random 的 `obs_known_iou +0.0573`，31/32 胜。
3. `correct` 对 observed unknown 区域有收益，但收益很小。target_unique 下：
   - 相对 empty: `obs_unknown_iou +0.0027`，23/32 胜；
   - 相对 shuffle: `+0.0057`，29/32 胜；
   - 相对 random: `+0.0074`，30/32 胜。
4. `correct` 在 obs unknown 上不能稳定压过 jitter。target_unique 下相对 jitter：
   - `obs_unknown_iou -0.0003`，13/32 胜；
   - `obs_unknown_recall -0.0056`，10/32 胜。

这说明当前模型确实使用了 point prior，但主要收益来自 observed/known 区域的硬约束。它对 unknown completion 的帮助存在，但还不够强；尤其 jitter 这种“近似正确但有局部偏移”的 prior 在 obs_unknown 上甚至略强，说明模型还没有学到可靠的由正确 observed sparse 推断未观测结构的能力。

需要注意，jitter 不应该简单当成完全错误 prior。它更接近真实 AR/SLAM 的 noisy correct prior：点云大体来自同一个物体，但局部存在偏移、量化、跟踪噪声。因此 `correct` 不压过 jitter 并不等同于完全失败；它说明当前路线对真实 noisy prior 可能是鲁棒的，但不会显著提升未观测部分。

### 判断

当前结果支持以下判断：

- 如果目标是“把 AR/SLAM 稀疏点云作为硬边界，让 TRELLIS sparse 结果更贴合已观测几何”，这条路线有价值。
- 如果目标是“用稀疏点云显著提升未观测面补全质量”，当前 Stage 2 noisy hard clamp 还不够。
- 继续单纯加训练步数或减小 eval clamp 不太可能解决问题；前面 clamp sweep 已显示弱 clamp 会变差。

### 下一步建议

第一优先级：在 strict/oracle checkpoint 上重跑同样的 `obs_unknown` eval，确认上限。

如果 strict/oracle 的 `obs_unknown_iou` 也只比 empty 略高，说明 TRELLIS sparse latent 的点云边界主要只能保 known，无法强力改善 unknown 补全；后续就应该把 point prior 用作 sanity/geometry anchor，而不是期待它大幅提升完整 mesh。

如果 strict/oracle 的 `obs_unknown_iou` 明显高于 empty/shuffle/random/jitter，则说明 noisy prior 构建和训练策略损失了上限，下一步再做 curriculum。

命令：

```bash
cd /home/zjr/Tracker

CUDA_VISIBLE_DEVICES=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/zjr/anaconda3/envs/reconviagen/bin/python -u trellis_point_prior_mv/eval_sparse_inpaint_stage2.py \
  --manifest /home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_strictmask_s200/data/val/manifest.json \
  --checkpoint /home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_strictmask_s200/checkpoints/last.ckpt \
  --output_dir /home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_strictmask_s200/eval_obs_unknown_metrics \
  --weights microsoft/TRELLIS-image-large \
  --indices 0-31 \
  --prior_modes correct,empty,shuffle,random,jitter \
  --fixed_topk 4096,8192,target_unique \
  --steps 12 \
  --guidance_strength 1.0 \
  --known_latent_clamp_strength 1.0 \
  --known_logit_boost 0.0 \
  --known_conf_power 1.0 \
  --lora_rank 64 \
  --lora_alpha 128 \
  --clamp_initial_noise
```

快速查看：

```bash
/home/zjr/anaconda3/envs/reconviagen/bin/python -c "import json; p='/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_strictmask_s200/eval_obs_unknown_metrics/report.json'; d=json.load(open(p)); s=d['summary']; print('correct global=', s['correct']['iou_mean']); print('correct obs_unknown_iou=', s['correct']['obs_unknown_iou_mean']); print('correct obs_unknown_recall=', s['correct']['obs_unknown_target_recall_mean']); print('obs unknown rank=', s['correct_rank_obs_unknown_iou']); print('target_unique/empty obs_unknown delta=', s['paired_delta']['target_unique/empty']['obs_unknown_iou']); print('target_unique/shuffle obs_unknown delta=', s['paired_delta']['target_unique/shuffle']['obs_unknown_iou'])"
```

第二优先级：把 jitter 从 wrong-prior ranking 里分离出来。建议分成：

- hard wrong: `empty/shuffle/random`
- noisy correct: `jitter`

训练目标应当是：

- correct/noisy-correct 都应优于 empty/shuffle/random；
- 不强行要求 clean correct 在 unknown 上压过 jitter；
- 对 jitter 重点控制 known leak 和 global IoU，不把它当负样本打死。

第三优先级：如果继续训练，建议做 curriculum 而不是直接 noisy hard clamp 长训：

1. strict/easy prior 训练到稳定；
2. 加入 jitter/noisy-correct 做鲁棒性 fine-tune；
3. 再加入 shuffle/random 作为 ranking negatives；
4. 降低 `KNOWN_X0_LOSS_WEIGHT` 到 `0.25/0.5`，避免 noisy known region 主导全局 x0；
5. ranking loss 只对 hard wrong 起作用，不对 jitter 起作用。

第四优先级：如果要服务 AR 重建系统，短期更实际的用途是把 point prior 用作 TRELLIS 输出后的 geometry anchor / filter / sanity check，而不是直接指望它补全 unseen geometry。

## Stage 2 Strict/Oracle Obs-Unknown 上限复查

时间：2026-06-20

读取结果：

- `/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_strictmask_s200/eval_obs_unknown_metrics/report.json`

评测设置：

- checkpoint: `pointprior_pixal_v9_stage2_strictmask_s200/checkpoints/last.ckpt`
- indices: `0-31`
- topk: `4096,8192,target_unique`
- prior modes: `correct,empty,shuffle,random,jitter`
- clamp: `known_latent_clamp_strength=1.0`
- logit boost: `0.0`

### Strict/Oracle 主表

| topk | mode | IoU | obs known IoU | obs unknown IoU | obs unknown recall | obs known prior recall |
|---|---|---:|---:|---:|---:|---:|
| 4096 | correct | 0.1121 | 0.1883 | 0.0156 | 0.0424 | 0.3983 |
| 4096 | empty | 0.0207 | 0.0315 | 0.0089 | 0.0265 | 0.0355 |
| 4096 | shuffle | 0.0202 | 0.0293 | 0.0099 | 0.0294 | 0.0334 |
| 4096 | random | 0.0125 | 0.0183 | 0.0065 | 0.0183 | 0.0187 |
| 4096 | jitter | 0.0236 | 0.0326 | 0.0139 | 0.0410 | 0.0342 |
| 8192 | correct | 0.1364 | 0.2792 | 0.0193 | 0.0799 | 0.5951 |
| 8192 | empty | 0.0319 | 0.0599 | 0.0117 | 0.0548 | 0.0718 |
| 8192 | shuffle | 0.0314 | 0.0566 | 0.0129 | 0.0604 | 0.0683 |
| 8192 | random | 0.0192 | 0.0352 | 0.0086 | 0.0392 | 0.0374 |
| 8192 | jitter | 0.0372 | 0.0666 | 0.0159 | 0.0722 | 0.0799 |
| target_unique | correct | 0.1487 | 0.2704 | 0.0214 | 0.0783 | 0.5844 |
| target_unique | empty | 0.0351 | 0.0617 | 0.0131 | 0.0554 | 0.0741 |
| target_unique | shuffle | 0.0345 | 0.0584 | 0.0143 | 0.0599 | 0.0703 |
| target_unique | random | 0.0216 | 0.0379 | 0.0096 | 0.0420 | 0.0410 |
| target_unique | jitter | 0.0394 | 0.0647 | 0.0173 | 0.0738 | 0.0750 |

### Rank 对比

| run | metric | top1 | top1 rate | rank mean | rank median |
|---|---|---:|---:|---:|---:|
| noisy s200 | global IoU | 91/96 | 0.9479 | 1.073 | 1.0 |
| noisy s200 | obs unknown IoU | 31/96 | 0.3229 | 2.115 | 2.0 |
| noisy s200 | obs unknown recall | 23/96 | 0.2396 | 2.323 | 2.0 |
| strict/oracle s200 | global IoU | 96/96 | 1.0000 | 1.000 | 1.0 |
| strict/oracle s200 | obs unknown IoU | 66/96 | 0.6875 | 1.375 | 1.0 |
| strict/oracle s200 | obs unknown recall | 48/96 | 0.5000 | 1.688 | 1.5 |

### Correct 均值对比

| metric | noisy s200 | strict/oracle s200 | delta |
|---|---:|---:|---:|
| IoU mean | 0.0385 | 0.1324 | +0.0939 |
| target recall mean | 0.0709 | 0.2364 | +0.1655 |
| obs known IoU mean | 0.0770 | 0.2460 | +0.1690 |
| obs unknown IoU mean | 0.0163 | 0.0188 | +0.0025 |
| obs unknown recall mean | 0.0444 | 0.0669 | +0.0225 |
| obs known prior recall mean | 0.4280 | 0.5260 | +0.0980 |

### Target-Unique Correct 对比

| metric | noisy s200 | strict/oracle s200 | delta |
|---|---:|---:|---:|
| IoU | 0.0454 | 0.1487 | +0.1032 |
| target recall | 0.0860 | 0.2573 | +0.1713 |
| obs known IoU | 0.0913 | 0.2704 | +0.1791 |
| obs unknown IoU | 0.0198 | 0.0214 | +0.0016 |
| obs unknown recall | 0.0551 | 0.0783 | +0.0232 |
| obs known prior recall | 0.4777 | 0.5844 | +0.1067 |

### Target-Unique Paired Delta

| wrong | global delta IoU | global wins | obs unknown delta IoU | wins | obs unknown recall delta | wins | obs known delta IoU | wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| empty | 0.1136 | 32/32 | 0.0083 | 31/32 | 0.0228 | 28/32 | 0.2087 | 32/32 |
| shuffle | 0.1142 | 32/32 | 0.0071 | 29/32 | 0.0183 | 28/32 | 0.2120 | 32/32 |
| random | 0.1270 | 32/32 | 0.0119 | 32/32 | 0.0363 | 31/32 | 0.2325 | 32/32 |
| jitter | 0.1093 | 32/32 | 0.0041 | 26/32 | 0.0044 | 21/32 | 0.2057 | 32/32 |

### 结论

Strict/oracle 的结论非常明确：point prior 作为 known sparse boundary 很有效，但对 observed 之外的 unknown 补全提升仍然有限。

最关键的对比是：

- 全局 IoU：`0.0385 -> 0.1324`，提升 `+0.0939`；
- obs known IoU：`0.0770 -> 0.2460`，提升 `+0.1690`；
- obs unknown IoU：`0.0163 -> 0.0188`，只提升 `+0.0025`；
- target_unique 下 obs unknown IoU：`0.0198 -> 0.0214`，只提升 `+0.0016`。

这说明即使给模型近乎 oracle 的点云先验，主要收益仍然来自已观测/已知区域被保住，而不是 TRELLIS sparse flow 学会从点云强力推断 unseen geometry。

Strict/oracle 在 `obs_unknown` rank 上确实比 noisy 好：

- noisy `obs_unknown_iou top1_rate=0.3229`
- strict `obs_unknown_iou top1_rate=0.6875`

但绝对 IoU 提升很小，这意味着排序信号存在，幅度却不足以支撑“显著改善 mesh 未观测部分”的预期。

### 当前判断

1. Points-to-3D 式 point prior 路线没有失败，但它目前更适合做 sparse geometry anchor，而不是完整形状补全核心。
2. 继续在 sparse flow 里追求 unknown completion 大幅提升，收益预期不高。
3. 如果目标是 AR 系统可用 mesh，短期更合理的工程路线是：
   - 先用 ReconViaGen/TRELLIS 生成合理 mesh；
   - 再用 AR/SLAM sparse points、mask projection、相机位姿做后验筛选、对齐、过滤和局部约束；
   - 不把 sparse point prior 训练当作主提升来源。

### 下一步建议

第一优先级：停止继续单纯训练 Stage 2 point-prior sparse flow。当前上限已经说明，known 区域收益大，unknown 区域收益小；继续加步数、调 clamp 或加 noisy curriculum 可能提升排序，但很难显著提升完整 mesh。

第二优先级：把 point prior 转成后处理/优化约束：

1. 生成多个 TRELLIS/ReconViaGen mesh candidates；
2. 用 AR/SLAM sparse points + mask projection 计算候选 mesh 与观测点云/多视图 mask 的一致性；
3. 选择最匹配的 mesh；
4. 对 mesh 做轻量非刚性或局部表面拉回，而不是改 sparse flow。

第三优先级：如果还要保留训练方向，只做一个更小的验证：

- strict/easy checkpoint 作为初始化；
- noisy correct 包括 jitter，不当负样本；
- hard wrong 只用 empty/shuffle/random；
- ranking loss 只比较 global IoU proxy / observed-region consistency，不再强推 unknown completion；
- 目标从“补 unknown”改成“防止错误 prior 污染 known/observed geometry”。

第四优先级：回到 AR 系统目标。用 pose/SLAM 点云提升 mesh 精度，更应该放在生成后的 candidate rerank、geometry anchor、mask consistency refinement，而不是直接指望 TRELLIS sparse flow 根据 sparse points 补全 unseen sides。

## Frozen Stock Slat/Mesh Downstream Smoke

时间：2026-06-20

本节补充端到端 mesh 兼容性评测。目的不是训练新的 slat/texture flow，而是固定 stock TRELLIS slat/mesh downstream，只替换 sparse coords，回答两个问题：

1. 如果 sparse 真的变好，stock slat/mesh 是否能吃进去并生成更好的 mesh；
2. 当前 Stage 2 point-prior sparse 是否已经足够接近这个上限。

运行目录：

```text
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/strict_val0_with_stage2
```

说明：`strict_val0_stock_target_prior` 目录只有空样本子目录，没有 `report.json`，因此该 smoke 没有完整写出报告；后续 `strict_val0_with_stage2` 已包含 `stock_sparse,target_sparse,prior_sparse,stage2_correct` 四种模式，可以覆盖同一组对比。

### 单样本结果

样本：

```text
index=0
uid=d0c4672e5fd14e21a6e244ce39246f57_seq000
target_unique=23414
```

| mode | sparse coords | sparse IoU | sparse recall | sparse precision | vertices | faces | target->mesh mean | mesh->target mean | chamfer L2 | extent ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock_sparse | 6893 | 0.0323 | 0.0404 | 0.1374 | 176604 | 351520 | 0.1639 | 0.0946 | 0.0557 | 0.2433 |
| target_sparse | 23414 | 1.0000 | 1.0000 | 1.0000 | 441914 | 884608 | 0.0170 | 0.0127 | 0.00066 | 0.5332 |
| prior_sparse | 1500 | 0.0641 | 0.0641 | 1.0000 | 1308 | 2484 | 0.1618 | 0.0117 | 0.0320 | 0.3077 |
| stage2_correct | 23414 | 0.1640 | 0.2818 | 0.2818 | 529622 | 1053572 | 0.0264 | 0.0510 | 0.00737 | 0.9931 |

对应 mesh 路径：

```text
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/strict_val0_with_stage2/0000_d0c4672e5fd1/stock_sparse/mesh.obj
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/strict_val0_with_stage2/0000_d0c4672e5fd1/target_sparse/mesh.obj
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/strict_val0_with_stage2/0000_d0c4672e5fd1/prior_sparse/mesh.obj
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/strict_val0_with_stage2/0000_d0c4672e5fd1/stage2_correct/mesh.obj
```

### 结果解读

这次 smoke 的结论比 sparse-only eval 更积极。

第一，`target_sparse -> stock slat/mesh` 明显有效。`target_sparse` 的 Chamfer L2 为 `0.00066`，远好于 `stock_sparse` 的 `0.0557`。这说明 stock TRELLIS 的 slat/mesh downstream 并不是完全无法使用更好的 sparse structure；如果 sparse structure 真的接近目标，后续 mesh 可以明显变好。因此，不能再简单说“改 sparse flow 对最终 mesh 没意义”。

第二，`stage2_correct` 已经显著优于 `stock_sparse`，但仍远没有达到 `target_sparse` 上限。`stage2_correct` 的 Chamfer L2 为 `0.00737`，比 `stock_sparse` 好很多，但比 `target_sparse` 的 `0.00066` 差一个数量级左右。它的 sparse IoU 只有 `0.1640`，recall/precision 都是 `0.2818`，说明当前 Stage 2 sparse 只恢复了部分目标结构。

第三，`stage2_correct` 的 mesh 有过满风险。它的 extent ratio 为 `0.9931`，接近完整立方体范围；顶点数和面数也最高。这意味着它虽然让目标点到 mesh 的距离大幅下降，但 mesh 可能生成了偏厚、偏满或过扩张的结构。`target_to_mesh` 好不等于最终形状一定干净，还必须结合 `mesh_to_target`、extent、可视化和多样本统计。

第四，`prior_sparse` 单独作为 sparse coords 不够。它只有 1500 个观测点，precision 为 1.0，但 recall 只有 0.0641；生成 mesh 很小，`target_to_mesh` 仍接近 stock sparse。因此，直接把观测点当 sparse structure 送给 stock slat/mesh 不是正确用法，必须经过 inpainting/completion sparse flow。

### 对前面判断的修正

前面 sparse-only obs-unknown 评测显示，point-prior 的主要收益集中在 observed/known 区域，unknown 区域提升很小。这个判断仍然成立，但它不能直接推出“最终 mesh 没有收益”。

这次 frozen downstream smoke 说明：

1. stock slat/mesh 下游对 sparse structure 的质量很敏感；
2. GT target sparse 可以显著改善 mesh；
3. 当前 Stage 2 point-prior sparse 已经能带来端到端 mesh 指标改善；
4. 主要瓶颈不是下游完全不兼容，而是 Stage 2 sparse 还不够准，并且有过满/过厚趋势。

因此当前更准确的结论是：

```text
point-prior 路线没有被证伪；相反，GT sparse 上限证明它值得继续做。
但当前 Stage 2 还不足以作为最终 AR 系统模块，需要先解决 sparse precision、过满形状和多样本稳定性。
```

### 下一步建议

第一优先级：把 frozen downstream 从单样本扩到小批量。先不要训练新 slat flow，先确认这个现象在 `0-7` 或 `0-15` 是否稳定。

建议先跑：

```bash
cd /home/zjr/Tracker

CUDA_VISIBLE_DEVICES=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/zjr/anaconda3/envs/reconviagen/bin/python -u trellis_point_prior_mv/eval_mesh_frozen_downstream.py \
  --manifest /home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_strictmask_s200/data/val/manifest.json \
  --output_dir /home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/strict_val0_7_with_stage2 \
  --weights microsoft/TRELLIS-image-large \
  --indices 0-7 \
  --modes stock_sparse,target_sparse,stage2_correct \
  --stage2_checkpoint /home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_strictmask_s200/checkpoints/last.ckpt \
  --stage2_topk target_unique \
  --max_frames 8 \
  --cond_mode multi_stochastic \
  --ss_steps 12 \
  --slat_steps 12 \
  --ss_guidance_strength 7.5 \
  --slat_guidance_strength 7.5 \
  --slat_guidance_rescale 0.5 \
  --slat_rescale_t 3.0 \
  --steps 12 \
  --guidance_strength 1.0 \
  --known_latent_clamp_strength 1.0 \
  --known_logit_boost 0.0 \
  --known_conf_power 1.0 \
  --mesh_eval_samples 4000
```

第二优先级：加入 mesh 质量的形状惩罚/诊断，不只看 Chamfer。当前 `stage2_correct` 的 Chamfer 明显改善，但 extent ratio 接近 1，可能是过满结构。后续报告需要同时统计：

- `target_to_mesh_mean`
- `mesh_to_target_mean`
- `chamfer_l2_mean`
- `extent_ratio`
- `vertex_count/face_count`
- 是否出现巨大闭合块、薄片或满格 blob

第三优先级：Stage 2 sparse 的下一轮训练目标应从“提高 observed IoU”转向“提高 precision 并抑制过满”。可考虑：

- `stage2_topk` 不直接用 `target_unique`，增加 `4096/8192/12000` mesh eval；
- 给 known/inpaint logits 加 occupancy sparsity prior；
- 训练中加入 hard negative prior ranking，但只约束 global/observed consistency；
- 降低 hard clamp 或增加 late-step reinjection，避免从初始噪声阶段把 latent 推成满格结构。

第四优先级：暂时不急着训练新的 slat flow。因为 `target_sparse -> stock slat/mesh` 已经证明 stock downstream 可以工作。只有在多样本评测中发现 `target_sparse` 仍然经常失败，才需要优先做 slat/shape adapter。

## Frozen Downstream 0-7 多样本复查

时间：2026-06-20

运行目录：

```text
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/strict_val0_7_with_stage2
```

评测设置：

- indices: `0-7`
- modes: `stock_sparse,target_sparse,stage2_correct`
- checkpoint: `pointprior_pixal_v9_stage2_strictmask_s200/checkpoints/last.ckpt`
- `stage2_topk=target_unique`
- `cond_mode=multi_stochastic`
- `ss_steps=12`
- `slat_steps=12`
- `mesh_eval_samples=4000`

所有 8 个样本都成功生成 mesh，报告文件：

```text
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/strict_val0_7_with_stage2/report.json
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/strict_val0_7_with_stage2/report.csv
```

### 均值结果

| mode | sparse IoU | sparse recall | sparse precision | Chamfer L2 | target->mesh mean | mesh->target mean | extent ratio | vertices | faces |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock_sparse | 0.0490 | 0.0928 | 0.1046 | 0.0518 | 0.1345 | 0.1050 | 0.3995 | 189678 | 379068 |
| stage2_correct | 0.1463 | 0.2542 | 0.2542 | 0.0198 | 0.0373 | 0.0863 | 0.9228 | 151630 | 297572 |
| target_sparse | 1.0000 | 1.0000 | 1.0000 | 0.00040 | 0.0133 | 0.0113 | 0.4828 | 227774 | 455703 |

### Paired 胜负

| metric | stage2_correct 优于 stock_sparse | target_sparse 优于 stage2_correct | stage2-stock 平均差值 | stage2-target 平均差距 |
|---|---:|---:|---:|---:|
| Chamfer L2 越低越好 | 6/8 | 8/8 | +0.0320 | +0.0194 |
| target->mesh mean 越低越好 | 8/8 | 8/8 | +0.0972 | +0.0240 |
| mesh->target mean 越低越好 | 5/8 | 8/8 | +0.0187 | +0.0750 |

这里的 `stage2-stock 平均差值` 表示 `stock - stage2`，为正代表 stage2 更好；`stage2-target 平均差距` 表示 `stage2 - target`。

### 每样本简表

| idx | Chamfer stock/stage2/target | target->mesh stock/stage2/target | mesh->target stock/stage2/target | sparse IoU stock/stage2/target | extent stock/stage2/target |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0566 / 0.0040 / 0.0010 | 0.166 / 0.031 / 0.021 | 0.097 / 0.037 / 0.017 | 0.032 / 0.202 / 1.000 | 0.24 / 0.90 / 0.53 |
| 1 | 0.0162 / 0.0239 / 0.0004 | 0.094 / 0.034 / 0.013 | 0.049 / 0.105 / 0.011 | 0.047 / 0.132 / 1.000 | 0.48 / 0.90 / 0.92 |
| 2 | 0.0309 / 0.0071 / 0.0003 | 0.124 / 0.033 / 0.013 | 0.071 / 0.045 / 0.012 | 0.048 / 0.145 / 1.000 | 0.73 / 0.93 / 0.71 |
| 3 | 0.0310 / 0.0210 / 0.0004 | 0.101 / 0.039 / 0.013 | 0.095 / 0.099 / 0.012 | 0.048 / 0.139 / 1.000 | 0.49 / 0.91 / 0.48 |
| 4 | 0.0418 / 0.0217 / 0.0002 | 0.099 / 0.057 / 0.009 | 0.118 / 0.077 / 0.007 | 0.086 / 0.162 / 1.000 | 0.13 / 0.96 / 0.31 |
| 5 | 0.0749 / 0.0292 / 0.0003 | 0.123 / 0.029 / 0.013 | 0.176 / 0.117 / 0.012 | 0.034 / 0.114 / 1.000 | 0.51 / 0.96 / 0.30 |
| 6 | 0.0236 / 0.0273 / 0.0004 | 0.113 / 0.043 / 0.014 | 0.052 / 0.118 / 0.011 | 0.081 / 0.113 / 1.000 | 0.12 / 0.85 / 0.13 |
| 7 | 0.1393 / 0.0239 / 0.0002 | 0.256 / 0.032 / 0.011 | 0.182 / 0.091 / 0.010 | 0.015 / 0.162 / 1.000 | 0.49 / 0.98 / 0.48 |

### 多样本结论

这次 `0-7` 结果确认了单样本 smoke 的主结论，但也暴露了更清楚的风险。

第一，`target_sparse -> stock slat/mesh` 的上限稳定成立。8 个样本里 target sparse 的 Chamfer L2 均值只有 `0.00040`，显著低于 stock sparse 的 `0.0518`。这说明 stock slat/mesh downstream 对 sparse structure 的质量非常敏感，而且能稳定利用 GT sparse。当前不需要优先训练新的 slat flow 来证明兼容性。

第二，`stage2_correct` 对 mesh 有真实收益。相比 stock sparse，`stage2_correct` 的 Chamfer L2 从 `0.0518` 降到 `0.0198`，`target->mesh mean` 从 `0.1345` 降到 `0.0373`；paired 统计里 target->mesh 是 `8/8` 胜出，Chamfer 是 `6/8` 胜出。这说明 point-prior sparse 不是只在 sparse metric 上好看，它已经能进入 downstream 并改善最终 mesh。

第三，`stage2_correct` 的问题不是“不起作用”，而是“过覆盖/过满”。它的 `target->mesh` 很好，说明目标表面附近总能找到 mesh；但 `mesh->target` 只从 `0.1050` 降到 `0.0863`，改善较小，且只 `5/8` 胜出。与此同时 extent ratio 从 stock 的 `0.3995` 提到 `0.9228`，远高于 target sparse mesh 的 `0.4828`。这表示 stage2 mesh 往往填得太满，可能生成接近包围盒或厚壳结构。也就是说，当前模型提高了覆盖，但 precision 不够。

第四，`stage2_topk=target_unique` 是一个偏 oracle 的设置。它保证 stage2 输出点数和 GT target sparse 一样，但真实 AR/SLAM 场景并不知道 target_unique。因此后续必须做 fixed top-k / adaptive top-k 的 mesh eval，否则会高估实际可用性。

### 当前判断

现在不能说 point-prior 改模型没用。更合理的判断是：

```text
point-prior sparse 已经能让 frozen stock slat/mesh 的端到端 mesh 指标变好；
GT target sparse 上限很高，说明路线有继续价值；
当前最大瓶颈是 Stage 2 sparse precision 和过满结构，而不是 downstream 完全不兼容。
```

这也修正了之前偏保守的判断：sparse-only 的 obs-unknown 提升小，并不等价于最终 mesh 没收益。mesh downstream 会把更完整的 sparse structure 放大成几何收益；但如果 sparse 太满，也会把错误同样放大。

### 下一步建议

第一优先级：做 `stage2_topk` sweep，不重训，只评估。建议固定：

- `4096`
- `8192`
- `12000`
- `target_unique`

目标是找出是否存在一个比 `target_unique` 更稳的点数，能保留 `target->mesh` 改善，同时降低 `mesh->target` 和 extent ratio。

第二优先级：加入 sparse precision / anti-overfill 诊断。后续训练或 eval 不应只看 recall、target->mesh 和 Chamfer，还要同时看：

- sparse precision；
- mesh->target mean；
- extent ratio；
- mesh 顶点/面数；
- fixed top-k 下的 Chamfer。

第三优先级：如果 top-k sweep 证明降低点数能显著缓解过满，那么下一轮训练不用先碰 slat flow，而是改 Stage 2 sparse：

- 训练/采样时加入 occupancy sparsity regularization；
- 减弱 early hard clamp，改成 late-step known reinjection；
- 把 `known_logit_boost` 继续保持 `0`，避免进一步扩张；
- 引入 wrong-prior ranking 时，把目标设为提高 precision 和 observed consistency，而不是单纯提高 recall。

第四优先级：只有在 `target_sparse` 多样本上也出现 mesh 失败时，才需要优先训练 slat/shape flow adapter。当前 `target_sparse` 上限很稳定，所以短期还是优先修 Stage 2 sparse。

## Stage2 Frozen Downstream 相对 Top-k Sweep

时间：2026-06-21

运行目录：

```text
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/strict_val0_stage2_relcap_sweep
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/strict_val0_7_stage2_relcap_sweep
```

本轮测试的目的不是训练模型，而是回答一个更窄的问题：

```text
Stage2 mesh 过满问题，是否主要由 target_unique top-k 过大导致？
如果不用 oracle target_unique，而用相对 top-k + cap，能否保留 mesh 覆盖收益，同时降低 mesh->target 和 extent 过满？
```

### 为什么这个测试能达到目的

这个测试能隔离出 top-k 的影响，原因有四点。

第一，所有模式都使用 frozen stock TRELLIS slat/mesh downstream。也就是说 downstream 不训练、不改权重，mesh 差异主要来自 sparse coords，而不是 slat/mesh flow 本身变化。

第二，`stock_sparse` 和 `target_sparse` 分别提供下限和上限：

- `stock_sparse`：原版 TRELLIS sparse 生成的 mesh baseline；
- `target_sparse`：GT sparse 直接进入 stock slat/mesh 的 oracle 上限。

如果 `target_sparse` 稳定好，说明 downstream 能吃更好的 sparse；如果 Stage2 改善有限，问题就在 Stage2 sparse 质量或取点策略。

第三，同一个样本的多个 `stage2_correct_*` mode 共享同一次 Stage2 logits，只从同一份 logits 中取不同 top-k。这样避免了“不同随机噪声导致结果不同”，让对比主要反映 top-k / density 选择。

第四，top-k 规格改成相对/cap，而不是固定绝对值：

```text
r0.35_cap4096
r0.50_cap8192
r0.75_cap12000
r1.00_cap12000
target_unique
```

这能适应不同样本 `target_unique` 差异，避免固定 12000 对小物体过满、对大物体欠采样。

评测同时看四类指标：

- sparse：`sparse_iou / precision / recall`
- coverage：`target_to_mesh_mean`
- overfill/precision：`mesh_to_target_mean`
- shape sanity：`extent_ratio / vertex_count / face_count`

因此它不是只看 Chamfer，也能暴露“目标点都被覆盖了，但 mesh 自己长出去很多”的问题。

### Sample 0 Smoke

sample 0 的 `target_unique=23414`，所以带 cap 后实际 top-k 差异很明显。

| mode | coords | ratio to target | sparse IoU | precision | recall | Chamfer L2 | target->mesh | mesh->target | extent ratio | vertices |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock_sparse | 6893 | - | 0.0323 | 0.1374 | 0.0404 | 0.0574 | 0.1659 | 0.0999 | 0.2434 | 176604 |
| r0.35_cap4096 | 4096 | 0.1749 | 0.0618 | 0.3909 | 0.0684 | 0.0114 | 0.0716 | 0.0465 | 0.8596 | 34390 |
| r0.50_cap8192 | 8192 | 0.3499 | 0.1043 | 0.3643 | 0.1274 | 0.0102 | 0.0563 | 0.0491 | 0.9188 | 74424 |
| r0.75_cap12000 | 12000 | 0.5125 | 0.1323 | 0.3449 | 0.1768 | 0.0084 | 0.0467 | 0.0470 | 0.8937 | 141692 |
| r1.00_cap12000 | 12000 | 0.5125 | 0.1323 | 0.3449 | 0.1768 | 0.0084 | 0.0470 | 0.0472 | 0.8933 | 141578 |
| target_unique | 23414 | 1.0000 | 0.1855 | 0.3129 | 0.3129 | 0.0059 | 0.0296 | 0.0479 | 0.9228 | 624818 |
| target_sparse | 23414 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0009 | 0.0205 | 0.0168 | 0.5332 | 441772 |

sample 0 的现象：

- `target_unique` 的 Chamfer 和 `target->mesh` 最好，但顶点数暴涨到 `624818`，extent ratio 仍很高；
- `r0.75_cap12000` 和 `r1.00_cap12000` 在 coverage 上接近 target_unique，但顶点数下降很多；
- `r0.35_cap4096` 的 `mesh->target` 最好，但 coverage 下降；
- 所有 Stage2 top-k 都明显优于 stock sparse，但都离 target sparse 上限很远。

### 0-7 多样本均值

| mode | coords mean | ratio mean | sparse IoU | precision | recall | Chamfer L2 | target->mesh | mesh->target | extent ratio | vertices |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock_sparse | 8385 | - | 0.0490 | 0.1046 | 0.0928 | 0.0520 | 0.1343 | 0.1060 | 0.3996 | 189663 |
| r0.35_cap4096 | 2909 | 0.3281 | 0.0810 | 0.3094 | 0.0997 | 0.0179 | 0.0575 | 0.0713 | 0.8547 | 29505 |
| r0.50_cap8192 | 4449 | 0.4812 | 0.1056 | 0.2963 | 0.1413 | 0.0169 | 0.0475 | 0.0717 | 0.8799 | 46164 |
| r0.75_cap12000 | 6637 | 0.7203 | 0.1308 | 0.2784 | 0.1985 | 0.0171 | 0.0465 | 0.0742 | 0.8911 | 71551 |
| r1.00_cap12000 | 8350 | 0.9391 | 0.1438 | 0.2648 | 0.2438 | 0.0168 | 0.0422 | 0.0747 | 0.9082 | 101430 |
| target_unique | 9776 | 1.0000 | 0.1505 | 0.2608 | 0.2608 | 0.0163 | 0.0395 | 0.0747 | 0.9117 | 161814 |
| target_sparse | 9776 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0004 | 0.0133 | 0.0113 | 0.4828 | 227782 |

### Paired 统计

| mode | Chamfer 胜 stock | target->mesh 胜 stock | mesh->target 胜 stock | extent 更接近 target |
|---|---:|---:|---:|---:|
| r0.35_cap4096 | 6/8 | 8/8 | 5/8 | 1/8 |
| r0.50_cap8192 | 6/8 | 8/8 | 5/8 | 1/8 |
| r0.75_cap12000 | 7/8 | 8/8 | 4/8 | 1/8 |
| r1.00_cap12000 | 6/8 | 8/8 | 4/8 | 1/8 |
| target_unique | 6/8 | 8/8 | 4/8 | 1/8 |

每样本最佳 mode：

| metric | 最佳分布 |
|---|---|
| Chamfer L2 | target_unique 3/8, r0.35 2/8, r0.50 1/8, r0.75 2/8 |
| target->mesh | target_unique 4/8, r0.50 1/8, r0.75 1/8, r1.00_cap12000 2/8 |
| mesh->target | r0.35 3/8, target_unique 2/8, r0.50 1/8, r0.75 2/8 |

相对 stock-target 区间的平均 closure：

| mode | Chamfer closure | target->mesh closure | mesh->target closure |
|---|---:|---:|---:|
| r0.35_cap4096 | 0.3728 | 0.5938 | -0.0247 |
| r0.50_cap8192 | 0.4164 | 0.6828 | -0.0036 |
| r0.75_cap12000 | 0.4264 | 0.6857 | -0.0018 |
| r1.00_cap12000 | 0.4371 | 0.7301 | -0.0278 |
| target_unique | 0.4518 | 0.7515 | -0.0225 |

`mesh->target closure` 接近 0 或为负，说明 Stage2 相比 stock 对 mesh 自身外扩的改善很弱；它主要改善的是目标表面是否被 mesh 覆盖。

### 结果解读

第一，Stage2 sparse 的端到端收益稳定存在。所有相对 top-k 都让 `target->mesh` 在 `8/8` 样本上优于 stock；Chamfer 也大多是 `6/8` 或 `7/8` 胜出。这说明 point-prior sparse 的确进入了 stock downstream 并改善了 mesh 覆盖。

第二，降低 top-k 能减少复杂度，但不能根治过满。比如从 `target_unique` 降到 `r0.50_cap8192`：

- coords: `9776 -> 4449`
- vertices: `161814 -> 46164`
- extent ratio: `0.9117 -> 0.8799`
- target->mesh: `0.0395 -> 0.0475`
- Chamfer: `0.0163 -> 0.0169`

也就是说，`r0.50_cap8192` 用更少点数保住了大部分 Chamfer 收益，但 extent ratio 仍远高于 target_sparse 的 `0.4828`。过满不是单纯 top-k 数量问题，而是 Stage2 logits 的空间分布本身偏向大范围填充。

第三，`target_unique` 不是实际系统可用设置。它的确给最好的 coverage，但真实 AR/SLAM 场景没有 GT target sparse 数量，而且它会显著增加 mesh 复杂度和过满风险。因此它只能作为 oracle 上限，不应作为实际默认。

第四，`r0.50_cap8192` 是当前更合理的工程默认。它的 Chamfer `0.0169` 接近 target_unique 的 `0.0163`，target->mesh 仍显著优于 stock，同时顶点数只有 target_unique 的约 28.5%。如果更重视覆盖，可以考虑 `r0.75_cap12000`；如果更重视少点和更轻 mesh，可以考虑 `r0.35_cap4096`。

### 当前结论

这轮测试达到目的：它证明了 top-k 数量会影响 coverage/复杂度 tradeoff，但也证明了当前 Stage2 的过满不是只靠降低 top-k 就能解决。

更准确的判断是：

```text
Stage2 point-prior sparse 已经能稳定提升 mesh coverage；
target_unique 会放大 coverage，但也放大复杂度和过满；
相对 top-k + cap 能让评测更公平，并给出可用默认；
但 Stage2 logits 本身仍偏扩张，后续需要训练或采样层面的 precision/anti-overfill 约束。
```

### 下一步建议

第一优先级：后续 mesh eval 默认不要再使用裸 `target_unique`，而使用：

```text
r0.50_cap8192
```

作为工程默认，同时保留 `target_unique` 作为 oracle 上限。

第二优先级：下一轮 Stage2 改法应针对 precision/anti-overfill，而不是继续盲目提高 recall：

- 训练中加入 occupancy sparsity regularization；
- 加 hard negative prior ranking，但目标设为压制 overfill 和错误 support；
- 调整 known reinjection，从全程 hard clamp 改成 late-step 或分段 clamp；
- 对 logits 加 outside/unknown penalty，避免大范围满格。

第三优先级：继续做端到端 mesh eval 时建议固定三档：

```text
r0.35_cap4096
r0.50_cap8192
target_unique
```

分别代表轻量高 precision、默认平衡、oracle coverage 上限。这样比固定 `4096/8192/12000/target_unique` 更适合多样本比较。

## Stage2 Late Clamp 与 Anti-overfill 结果

时间：2026-06-21

本轮运行了四组：

```text
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/strict_val0_7_stage2_relcap_lateclamp05
/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_smoke
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/antioverfill_smoke_mesh_relcap
/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_s200
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/antioverfill_s200_mesh_relcap_val8
```

### 测试目的

这轮测试回答两个问题：

1. 过满是否主要来自采样时 early hard clamp；
2. decoder-based anti-overfill loss 是否能提高 sparse precision，并减少 mesh 外扩。

late clamp 只改采样，不重训；anti-overfill 则改训练目标：

```text
anti_overfill_loss = mean softplus(decoder(pred_x0) + margin) on voxels outside GT target sparse
```

这个 loss 直接惩罚 GT target sparse 外部的正 occupancy logits，目标不是继续提高 recall，而是压制 outside occupancy。

### Late Clamp 诊断

对比 baseline strictmask relcap 与 `KNOWN_CLAMP_START_T=0.5`。

| mode | run | sparse IoU | precision | recall | Chamfer L2 | target->mesh | mesh->target | extent ratio | vertices |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r0.35_cap4096 | baseline | 0.0810 | 0.3094 | 0.0997 | 0.0179 | 0.0575 | 0.0713 | 0.8547 | 29505 |
| r0.35_cap4096 | late clamp 0.5 | 0.0811 | 0.3095 | 0.0998 | 0.0193 | 0.0600 | 0.0730 | 0.8562 | 30508 |
| r0.50_cap8192 | baseline | 0.1056 | 0.2963 | 0.1413 | 0.0169 | 0.0475 | 0.0717 | 0.8799 | 46164 |
| r0.50_cap8192 | late clamp 0.5 | 0.1061 | 0.2974 | 0.1419 | 0.0177 | 0.0523 | 0.0718 | 0.8883 | 41716 |
| r0.75_cap12000 | baseline | 0.1308 | 0.2784 | 0.1985 | 0.0171 | 0.0465 | 0.0742 | 0.8911 | 71551 |
| r0.75_cap12000 | late clamp 0.5 | 0.1310 | 0.2787 | 0.1988 | 0.0168 | 0.0459 | 0.0731 | 0.8963 | 75645 |
| target_unique | baseline | 0.1505 | 0.2608 | 0.2608 | 0.0163 | 0.0395 | 0.0747 | 0.9117 | 161814 |
| target_unique | late clamp 0.5 | 0.1502 | 0.2604 | 0.2604 | 0.0177 | 0.0407 | 0.0790 | 0.9101 | 161123 |

结论：late clamp 单独不是主要解法。`r0.50_cap8192` 下：

- `mesh_to_target`: `0.0717 -> 0.0718`，基本不变；
- `extent_ratio`: `0.8799 -> 0.8883`，略差；
- `target_to_mesh`: `0.0475 -> 0.0523`，coverage 变差。

因此过满不是主要由 early hard clamp 造成，而是 Stage2 logits 训练后的空间分布本身偏扩张。

### Anti-overfill Smoke

smoke 只跑 sample 0，不能代表多样本结论，但方向是正的。

| mode | sparse IoU | precision | recall | Chamfer L2 | target->mesh | mesh->target | extent ratio | vertices |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stock_sparse | 0.0323 | 0.1374 | 0.0404 | 0.0565 | 0.1659 | 0.0972 | 0.2434 | 176614 |
| r0.35_cap4096 | 0.0709 | 0.4446 | 0.0778 | 0.0078 | 0.0707 | 0.0282 | 0.7089 | 26702 |
| r0.50_cap8192 | 0.1048 | 0.3658 | 0.1280 | 0.0076 | 0.0685 | 0.0257 | 0.7585 | 51814 |
| target_unique | 0.1664 | 0.2853 | 0.2853 | 0.0040 | 0.0341 | 0.0387 | 0.8119 | 518068 |
| target_sparse | 1.0000 | 1.0000 | 1.0000 | 0.0009 | 0.0205 | 0.0168 | 0.5332 | 441746 |

相比之前 sample 0 baseline，anti-overfill smoke 明显降低 `mesh_to_target` 和 `extent_ratio`，说明 loss 没有方向性错误。

### Anti-overfill s200 Sparse Eval

对比 strictmask s200 与 anti-overfill s200 的 sparse eval。

| run | correct IoU | precision | recall | obs known IoU | obs unknown IoU | obs unknown recall | obs known prior recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| strictmask s200 | 0.1324 | 0.2616 | 0.2364 | 0.2460 | 0.0188 | 0.0669 | 0.5260 |
| anti-overfill s200 | 0.1463 | 0.2864 | 0.2566 | 0.2699 | 0.0179 | 0.0626 | 0.5655 |

ranking：

| run | correct rank global top1 | obs unknown IoU top1 | obs unknown recall top1 |
|---|---:|---:|---:|
| strictmask s200 | 96/96 | 66/96 | 48/96 |
| anti-overfill s200 | 96/96 | 52/96 | 33/96 |

解释：

- anti-overfill s200 提高了全局 sparse IoU、precision、recall 和 observed/known 区域；
- obs unknown 指标略下降，说明这个 loss 会牺牲一点 unknown completion；
- 这符合预期，因为它把目标从“尽量覆盖更多”拉回到“不要在 GT 外部过度激活”。

### Anti-overfill s200 Mesh Eval

核心比较是 `r0.50_cap8192`，因为这是前面建议的工程默认。

| mode | run | sparse IoU | precision | recall | Chamfer L2 | target->mesh | mesh->target | extent ratio | vertices |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r0.35_cap4096 | baseline | 0.0810 | 0.3094 | 0.0997 | 0.0179 | 0.0575 | 0.0713 | 0.8547 | 29505 |
| r0.35_cap4096 | anti-overfill s200 | 0.1019 | 0.3798 | 0.1229 | 0.0127 | 0.0560 | 0.0515 | 0.6496 | 29021 |
| r0.50_cap8192 | baseline | 0.1056 | 0.2963 | 0.1413 | 0.0169 | 0.0475 | 0.0717 | 0.8799 | 46164 |
| r0.50_cap8192 | anti-overfill s200 | 0.1318 | 0.3600 | 0.1722 | 0.0121 | 0.0491 | 0.0527 | 0.6768 | 43988 |
| r0.75_cap12000 | baseline | 0.1308 | 0.2784 | 0.1985 | 0.0171 | 0.0465 | 0.0742 | 0.8911 | 71551 |
| r0.75_cap12000 | anti-overfill s200 | 0.1604 | 0.3320 | 0.2372 | 0.0120 | 0.0437 | 0.0564 | 0.6753 | 88965 |
| target_unique | baseline | 0.1505 | 0.2608 | 0.2608 | 0.0163 | 0.0395 | 0.0747 | 0.9117 | 161814 |
| target_unique | anti-overfill s200 | 0.1832 | 0.3085 | 0.3085 | 0.0109 | 0.0372 | 0.0561 | 0.6876 | 182536 |
| target_sparse | reference | 1.0000 | 1.0000 | 1.0000 | 0.0004 | 0.0133 | 0.0113 | 0.4828 | 227789 |

paired 胜负：

| mode | Chamfer 胜 stock | target->mesh 胜 stock | mesh->target 胜 stock |
|---|---:|---:|---:|
| r0.35_cap4096 | 8/8 | 8/8 | 5/8 |
| r0.50_cap8192 | 8/8 | 8/8 | 5/8 |
| r0.75_cap12000 | 8/8 | 8/8 | 5/8 |
| target_unique | 8/8 | 8/8 | 5/8 |

### 结论

anti-overfill s200 是目前为止最有价值的一轮 Stage2 修改。

对 `r0.50_cap8192`：

- `sparse precision`: `0.2963 -> 0.3600`
- `sparse recall`: `0.1413 -> 0.1722`
- `sparse IoU`: `0.1056 -> 0.1318`
- `Chamfer L2`: `0.0169 -> 0.0121`
- `mesh_to_target`: `0.0717 -> 0.0527`
- `extent_ratio`: `0.8799 -> 0.6768`
- `target_to_mesh`: `0.0475 -> 0.0491`

也就是说，anti-overfill 明显降低了外扩和过满，且只轻微牺牲 coverage。它不是简单把 mesh 变小，因为 sparse recall 和 IoU 也提升了。

但它还没有到达 target sparse 上限：

- `mesh_to_target`: anti-overfill `0.0527` vs target sparse `0.0113`
- `extent_ratio`: anti-overfill `0.6768` vs target sparse `0.4828`
- `Chamfer`: anti-overfill `0.0121` vs target sparse `0.0004`

所以当前结论是：

```text
point-prior Stage2 继续改模型有收益；
收益方向应该是 anti-overfill / precision，而不是继续推高 recall；
r0.50_cap8192 可以作为当前默认工程评测点；
target_unique 仍只作为 oracle coverage 上限。
```

### 下一步建议

第一优先级：不要继续盲目长训当前配置。先做小范围 loss weight sweep，找 precision/coverage 平衡点：

```text
ANTI_OVERFILL_LOSS_WEIGHT = 0.005 / 0.01 / 0.02 / 0.04
KNOWN_X0_LOSS_WEIGHT = 0.5
KNOWN_CLAMP_START_T = 0.5
CLAMP_INITIAL_NOISE = 0
```

评测仍以 `r0.50_cap8192` 为主，保留 `r0.35_cap4096` 和 `target_unique`。

第二优先级：加入 hard-negative ranking，但不要直接优化 unknown completion。ranking 目标应该是压制 wrong prior 下的 overfill 和 observed inconsistency：

- correct prior 应该比 shuffle/random prior 有更低 outside occupancy；
- correct prior 应该有更高 observed prior recall；
- jitter 不作为 hard negative，只作为 noisy-correct 鲁棒性。

第三优先级：准备 AR 场景迁移时，实际默认建议是：

```text
top-k: r0.50_cap8192
checkpoint: anti-overfill s200 或其 loss sweep 最优版
use target_unique: 只做 synthetic oracle，不进真实系统
```

第四优先级：端到端 mesh 结果已经说明 point-prior 有继续价值，但仍应和 ReconViaGen candidate rerank 并行推进。短期工程上，不应只赌一个 Stage2 sparse，而应保留多候选 mesh，并用 AR/SLAM points + mask projection 做 sanity check。

## 2026-06-21 补充：自动化 sweep 脚本

已补充：

```text
/home/zjr/Tracker/trellis_point_prior_mv/scripts/run_stage2_antioverfill_weight_sweep.sh
/home/zjr/Tracker/trellis_point_prior_mv/summarize_mesh_frozen_reports.py
```

作用：

- `run_stage2_antioverfill_weight_sweep.sh`：自动扫 `ANTI_OVERFILL_LOSS_WEIGHT`，每个权重会依次跑 Stage2 训练、sparse eval 和 frozen downstream mesh eval；
- `summarize_mesh_frozen_reports.py`：把多个 frozen downstream `report.json` 汇总成表格，便于横向比较不同权重。

下一轮建议顺序：

```text
1. 先跑 smoke sweep: 0.005 / 0.01 / 0.02 / 0.04
2. 从 smoke 中选 1-2 个最稳权重跑 s200
3. 核心指标仍看 r0.50_cap8192
4. 优先选择 mesh_to_target_mean 和 extent_ratio 下降，同时 target_to_mesh_mean 不明显劣化的配置
5. anti-overfill 权重稳定后，再考虑 hard-negative ranking smoke
```

完整命令已写入：

```text
/home/zjr/Tracker/trellis_point_prior_mv/命令说明.txt
```

## 2026-06-21 补充：Anti-overfill smoke weight sweep 结果

本轮运行：

```text
MODE=smoke
SWEEP_WEIGHTS=0.005,0.01,0.02,0.04
TOPK_SPECS=r0.35_cap4096,r0.50_cap8192,target_unique
```

结果路径：

```text
/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_weight_sweep_smoke_w0p005/eval/report.json
/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_weight_sweep_smoke_w0p01/eval/report.json
/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_weight_sweep_smoke_w0p02/eval/report.json
/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_weight_sweep_smoke_w0p04/eval/report.json
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/pointprior_pixal_v9_stage2_antioverfill_weight_sweep_smoke_w0p005_mesh_relcap/report.json
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/pointprior_pixal_v9_stage2_antioverfill_weight_sweep_smoke_w0p01_mesh_relcap/report.json
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/pointprior_pixal_v9_stage2_antioverfill_weight_sweep_smoke_w0p02_mesh_relcap/report.json
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/pointprior_pixal_v9_stage2_antioverfill_weight_sweep_smoke_w0p04_mesh_relcap/report.json
```

### Sparse eval

`correct` prior 的 sparse 指标：

| anti-overfill weight | IoU | precision | recall | obs-known IoU | obs-unknown IoU | obs-unknown recall |
|---:|---:|---:|---:|---:|---:|---:|
| 0.005 | 0.1104 | 0.2377 | 0.1870 | 0.2073 | 0.0227 | 0.0700 |
| 0.01 | 0.1110 | 0.2391 | 0.1879 | 0.2088 | 0.0224 | 0.0689 |
| 0.02 | 0.1117 | 0.2405 | 0.1889 | 0.2098 | 0.0226 | 0.0692 |
| 0.04 | 0.1075 | 0.2318 | 0.1829 | 0.2039 | 0.0215 | 0.0663 |

`correct_rank` 全部是 `24/24 top1`，说明在 smoke 设置下，correct prior 与 wrong prior 的全局 sparse 排序已经能分开。更细的 observed-unknown 排序仍不稳定：

| anti-overfill weight | obs-unknown IoU top1 | obs-unknown recall top1 |
|---:|---:|---:|
| 0.005 | 12/24 | 8/24 |
| 0.01 | 12/24 | 7/24 |
| 0.02 | 13/24 | 9/24 |
| 0.04 | 13/24 | 6/24 |

解释：anti-overfill 对 observed/known 区域和整体 sparse 有帮助，但 unknown completion 仍不是它的直接强项。`0.04` 已经开始损伤 sparse 指标。

### Frozen downstream mesh eval

核心仍看 `r0.50_cap8192`：

| anti-overfill weight | sparse IoU | precision | recall | Chamfer L2 | target->mesh | mesh->target | extent ratio | vertices |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.005 | 0.0926 | 0.2640 | 0.1252 | 0.0099 | 0.0546 | 0.0369 | 0.8176 | 34296 |
| 0.01 | 0.0932 | 0.2657 | 0.1261 | 0.0122 | 0.0588 | 0.0410 | 0.8625 | 37264 |
| 0.02 | 0.0949 | 0.2701 | 0.1281 | 0.0102 | 0.0526 | 0.0406 | 0.8124 | 37901 |
| 0.04 | 0.0902 | 0.2580 | 0.1222 | 0.0109 | 0.0520 | 0.0446 | 0.8467 | 37312 |

### 结论

`0.02` 是当前 smoke sweep 里最平衡的 anti-overfill 权重：

- sparse IoU / precision / recall 都是四组里最高；
- `target->mesh` 接近 `0.04`，但没有 `0.04` 的 sparse 退化；
- `mesh->target` 虽不如 `0.005`，但 `0.005` 的 sparse 指标更弱，且 extent 没有优势；
- `0.01` 在 mesh 和 sparse 上都没有明显优势；
- `0.04` 过强，开始压坏 sparse 表达。

因此不建议再单独推进 `0.005` 的 s200。下一步应该固定：

```text
ANTI_OVERFILL_LOSS_WEIGHT=0.02
KNOWN_X0_LOSS_WEIGHT=0.5
KNOWN_CLAMP_START_T=0.5
CLAMP_INITIAL_NOISE=0
```

然后做 hard-negative ranking smoke。

### Ranking 代码调整

现有 ranking 不能只写：

```text
loss = relu(margin + outside_correct - outside_wrong)
```

因为这会把问题简化成 outside calibration，不能真正约束 wrong prior 对 observed 区域和 wrong support 的影响。本轮已把训练代码改成 decoder-level 多项 hard-negative 目标：

1. `rank_observed`：correct condition 在 correct prior observed support 上的 decoder logit 应高于 wrong condition；
2. `wrong_support`：wrong prior 落在 GT target 外的 support 区域，应被 negative condition 输出为空；
3. `target_support`：correct prior 落在 GT target 内的 support 区域，应被 positive condition 输出为实；
4. `rank_outside`：只作为弱 calibration 项，负样本 outside 分支 detach，避免训练主动鼓励 wrong condition 生成更坏结果。

对应代码：

```text
/home/zjr/Tracker/trellis_point_prior_mv/train_sparse_inpaint_stage2.py
/home/zjr/Tracker/trellis_point_prior_mv/scripts/run_pixal_v9_point_prior_stage2.sh
```

下一步 ranking smoke 命令已更新到：

```text
/home/zjr/Tracker/trellis_point_prior_mv/命令说明.txt
```

## 2026-06-21 补充：Hard-negative ranking smoke 对比

本轮运行了两组 ranking smoke：

```text
pointprior_pixal_v9_stage2_antioverfill_rank_w001_support_smoke
pointprior_pixal_v9_stage2_antioverfill_rank_smoke_x0p5
```

对照基线：

```text
pointprior_pixal_v9_stage2_antioverfill_weight_sweep_smoke_w0p02
```

结果路径：

```text
/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_weight_sweep_smoke_w0p02/eval/report.json
/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_rank_w001_support_smoke/eval/report.json
/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_rank_smoke_x0p5/eval/report.json
```

### Correct sparse 指标

| run | IoU | precision | recall | known IoU | unknown IoU | obs-known IoU | obs-unknown IoU | obs-unknown recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| anti-overfill 0.02 base | 0.1117 | 0.2405 | 0.1889 | 0.2098 | 0.0226 | 0.2098 | 0.0226 | 0.0692 |
| rank w0.01 support | 0.1277 | 0.2702 | 0.2127 | 0.2393 | 0.0223 | 0.2393 | 0.0223 | 0.0655 |
| rank x0p5 | 0.0553 | 0.1280 | 0.0974 | 0.1029 | 0.0193 | 0.1029 | 0.0193 | 0.0617 |

`rank_w001_support` 相比 anti-overfill 0.02 base：

```text
IoU:        0.1117 -> 0.1277
precision:  0.2405 -> 0.2702
recall:     0.1889 -> 0.2127
known IoU:  0.2098 -> 0.2393
```

它的主要收益来自 observed/known 区域，unknown 几乎没有改善，甚至 `obs_unknown_target_recall` 略降：

```text
obs_unknown_target_recall: 0.0692 -> 0.0655
```

这符合当前 ranking 设计：它主要是在压 wrong support 和增强 correct observed support，而不是做 unknown completion。

`rank_x0p5` 明显失败：

```text
IoU:        0.1117 -> 0.0553
precision:  0.2405 -> 0.1280
recall:     0.1889 -> 0.0974
known IoU:  0.2098 -> 0.1029
```

原因判断：`RANKING_LOSS_WEIGHT=0.05` 加上 `RANKING_TARGET_SUPPORT_WEIGHT=0.5` 对 20-step smoke 来说过强，且 target-support positive 项会强行改变 decoder logit calibration，压过了原来的 flow/inpainting 学习信号。这个配置不应继续 s200。

### Correct rank

| run | correct top1 | rank mean | obs-unknown IoU top1 | obs-unknown recall top1 |
|---|---:|---:|---:|---:|
| anti-overfill 0.02 base | 24/24 | 1.000 | 13/24 | 9/24 |
| rank w0.01 support | 24/24 | 1.000 | 11/24 | 6/24 |
| rank x0p5 | 18/24 | 1.500 | 7/24 | 7/24 |

`rank_w001_support` 没有破坏全局 correct top1，但也没有改善 unknown rank；`rank_x0p5` 连全局 top1 都破坏了。

### Paired delta 观察

`rank_w001_support` 在 8192/top-k 和 target_unique 下，对 shuffle/random 的整体 IoU delta 比 base 更大，说明它确实增强了 correct prior 的整体区分度：

```text
8192/random IoU delta:  0.0834 -> 0.0974
8192/shuffle IoU delta: 0.0778 -> 0.0927
target_unique/random:   0.0864 -> 0.1006
target_unique/shuffle:  0.0799 -> 0.0951
```

但 observed-unknown 排序没有提升，说明这轮 ranking 不是 unknown completion 的解法。

### 当前结论

```text
1. hard-negative ranking 不是没用；
2. 有效配置是轻量 support ranking，而不是强 target-support ranking；
3. rank_w001_support 可以进入 frozen downstream mesh eval；
4. rank_x0p5 不应继续训练；
5. 暂时不建议把 RANKING_TARGET_SUPPORT_WEIGHT 打开到 0.5。
```

下一步优先级：

```text
第一步：对 rank_w001_support 做 frozen downstream mesh eval；
第二步：如果 mesh 没有反弹，再跑 rank_w001_support s200；
第三步：s200 后再考虑更小的 target-support，例如 0.05/0.1，而不是 0.5。
```

## 2026-06-21 补充：rank_w001_support frozen downstream mesh eval

本轮补跑：

```text
antioverfill_rank_w001_support_smoke_mesh_relcap
antioverfill_rank_w001_support_val8_mesh_relcap
```

结果路径：

```text
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/antioverfill_rank_w001_support_smoke_mesh_relcap/report.json
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/antioverfill_rank_w001_support_val8_mesh_relcap/report.json
```

`smoke` 是 sample 0 sanity check；主要判断依据是 `val8`。

### val8 汇总

与 anti-overfill 0.02 smoke 对比：

| run | mode | sparse IoU | precision | recall | Chamfer L2 | target->mesh | mesh->target | extent ratio | vertices |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| anti-overfill 0.02 smoke | r0.35_cap4096 | 0.0768 | 0.2984 | 0.0947 | 0.0094 | 0.0603 | 0.0317 | 0.7839 | 23452 |
| rank_w001_support val8 | r0.35_cap4096 | 0.0839 | 0.3230 | 0.1030 | 0.0115 | 0.0670 | 0.0359 | 0.6485 | 23738 |
| anti-overfill 0.02 smoke | r0.50_cap8192 | 0.0949 | 0.2701 | 0.1281 | 0.0102 | 0.0526 | 0.0406 | 0.8124 | 37901 |
| rank_w001_support val8 | r0.50_cap8192 | 0.1052 | 0.2965 | 0.1408 | 0.0092 | 0.0511 | 0.0392 | 0.6689 | 35709 |
| anti-overfill 0.02 smoke | target_unique | 0.1222 | 0.2172 | 0.2172 | 0.0136 | 0.0377 | 0.0657 | 0.8142 | 146812 |
| rank_w001_support val8 | target_unique | 0.1401 | 0.2451 | 0.2451 | 0.0152 | 0.0472 | 0.0691 | 0.7014 | 148518 |

核心工程点仍是 `r0.50_cap8192`。它相比 anti-overfill 0.02 smoke：

```text
sparse IoU:       0.0949 -> 0.1052
precision:        0.2701 -> 0.2965
recall:           0.1281 -> 0.1408
Chamfer L2:       0.0102 -> 0.0092
target->mesh:     0.0526 -> 0.0511
mesh->target:     0.0406 -> 0.0392
extent ratio:     0.8124 -> 0.6689
vertices:         37901 -> 35709
```

这说明 `rank_w001_support` 的 sparse 提升没有在 frozen downstream mesh 里反弹，反而减少了外扩，并略微降低 mesh 复杂度。

与 anti-overfill s200 对比，`rank_w001_support` 只是 20-step smoke，但 `r0.50_cap8192` 的 mesh 指标已经有竞争力：

```text
anti-overfill s200:
  sparse IoU 0.1318, Chamfer 0.0121, target->mesh 0.0491, mesh->target 0.0527, extent 0.6768

rank_w001_support smoke:
  sparse IoU 0.1052, Chamfer 0.0092, target->mesh 0.0511, mesh->target 0.0392, extent 0.6689
```

解释：s200 的 sparse 更强，但 smoke ranking 的 mesh 外扩更少。这个信号足够支持进入 `rank_w001_support_s200`，但 s200 后必须同时看 step100 与 last/step200，避免长训后重新走向过填充。

### 结论

现在可以跑 `rank_w001_support_s200`。

推荐保留当前配置，不要加回强 target-support：

```text
RANKING_LOSS_WEIGHT=0.01
RANKING_OUTSIDE_WEIGHT=0.0
RANKING_OBSERVED_WEIGHT=0.25
RANKING_WRONG_SUPPORT_WEIGHT=1.0
RANKING_TARGET_SUPPORT_WEIGHT=0.0
ANTI_OVERFILL_LOSS_WEIGHT=0.02
KNOWN_X0_LOSS_WEIGHT=0.5
KNOWN_CLAMP_START_T=0.5
CLAMP_INITIAL_NOISE=0
```

s200 后的判断标准：

```text
1. r0.50_cap8192 sparse IoU / precision 应继续高于 anti-overfill smoke；
2. Chamfer 不应高于 anti-overfill s200 的 0.0121；
3. mesh->target 不应回到 0.0527 以上；
4. extent ratio 应维持在 0.67 左右或更低；
5. target->mesh 不应明显高于 0.051。
```

## 2026-06-21 补充：rank_w001_support s200 结果

本轮运行：

```text
pointprior_pixal_v9_stage2_antioverfill_rank_w001_support_s200
antioverfill_rank_w001_support_s200_mesh_relcap_val8
antioverfill_rank_w001_support_s200_step100_mesh_relcap_val8
```

结果路径：

```text
/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_rank_w001_support_s200/eval/report.json
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/antioverfill_rank_w001_support_s200_mesh_relcap_val8/report.json
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/antioverfill_rank_w001_support_s200_step100_mesh_relcap_val8/report.json
```

### Sparse eval

`rank_w001_support_s200` 对比 `anti_overfill_s200`：

| run | count | IoU | precision | recall | known IoU | unknown IoU | obs-known IoU | obs-unknown IoU | obs-unknown recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| anti-overfill s200 | 96 | 0.1463 | 0.2864 | 0.2566 | 0.2699 | 0.0179 | 0.2699 | 0.0179 | 0.0626 |
| rank_w001_support s200 | 96 | 0.1634 | 0.3148 | 0.2821 | 0.2918 | 0.0197 | 0.2918 | 0.0197 | 0.0672 |

rank 结果：

| run | correct top1 | obs-unknown IoU top1 | obs-unknown recall top1 |
|---|---:|---:|---:|
| anti-overfill s200 | 96/96 | 52/96 | 33/96 |
| rank_w001_support s200 | 96/96 | 57/96 | 35/96 |

结论：`rank_w001_support_s200` 不仅提升 observed/known 区域，也轻微改善了 common observed split 下的 unknown 指标；它没有破坏 correct 全局排序。

### Frozen downstream mesh eval

主要看 `r0.50_cap8192`：

| run | sparse IoU | precision | recall | Chamfer L2 | target->mesh | mesh->target | extent ratio | vertices |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rank_w001_support smoke | 0.1052 | 0.2965 | 0.1408 | 0.0092 | 0.0511 | 0.0392 | 0.6689 | 35709 |
| anti-overfill s200 | 0.1318 | 0.3600 | 0.1722 | 0.0121 | 0.0491 | 0.0527 | 0.6768 | 43988 |
| rank_w001_support s200 step100 | 0.0889 | 0.2531 | 0.1208 | 0.0296 | 0.0636 | 0.0951 | 0.9562 | 39095 |
| rank_w001_support s200 last | 0.1381 | 0.3751 | 0.1797 | 0.0100 | 0.0489 | 0.0448 | 0.6875 | 47990 |

`step100` 很差：

```text
Chamfer:      0.0296
mesh->target: 0.0951
extent:       0.9562
```

它不是更好的早停点，反而说明训练到 100 step 时 mesh distribution 处于不稳定中间态。当前应使用 `last.ckpt` / `step200`，不是 `step100`。

`last.ckpt` / `step200` 相比 anti-overfill s200：

```text
sparse IoU:    0.1318 -> 0.1381
precision:     0.3600 -> 0.3751
recall:        0.1722 -> 0.1797
Chamfer:       0.0121 -> 0.0100
target->mesh:  0.0491 -> 0.0489
mesh->target:  0.0527 -> 0.0448
extent ratio:  0.6768 -> 0.6875
```

除了 `extent_ratio` 从 `0.6768` 小幅升到 `0.6875`，其他关键指标都更好，尤其 `mesh->target` 明显降低，说明外扩点到目标表面的距离减少。这个结果可以认为是当前 Stage2 最优版本。

### 不同 top-k 的选择

`rank_w001_support_s200 last`：

| top-k | sparse IoU | precision | recall | Chamfer L2 | target->mesh | mesh->target | extent ratio | vertices |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r0.35_cap4096 | 0.1083 | 0.4002 | 0.1301 | 0.0092 | 0.0532 | 0.0379 | 0.6544 | 29688 |
| r0.50_cap8192 | 0.1381 | 0.3751 | 0.1797 | 0.0100 | 0.0489 | 0.0448 | 0.6875 | 47990 |
| r0.75_cap12000 | 0.1686 | 0.3465 | 0.2480 | 0.0095 | 0.0417 | 0.0471 | 0.7061 | 86312 |
| target_unique | 0.1920 | 0.3214 | 0.3214 | 0.0093 | 0.0353 | 0.0492 | 0.7335 | 180876 |

建议：

```text
默认工程点：r0.50_cap8192
覆盖优先候选：r0.75_cap12000
轻量/低外扩候选：r0.35_cap4096
target_unique：只作为 synthetic oracle / 上限参考，不建议真实系统默认使用
```

### 当前结论

`rank_w001_support_s200 last.ckpt` 是目前最好的 Stage2 sparse checkpoint。

推荐当前 best：

```text
checkpoint:
/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_rank_w001_support_s200/checkpoints/last.ckpt

default top-k:
r0.50_cap8192

candidate top-k:
r0.35_cap4096, r0.50_cap8192, r0.75_cap12000
```

### 下一步建议

第一优先级：扩大 mesh eval，而不是立刻改模型。

当前 mesh 只评了 `0-7`，应补一个 `0-15` 或 `0-31`。考虑显存和耗时，建议先 `0-15`：

```bash
cd /home/zjr/Tracker

GPU=1 \
MODE=val8 \
INDICES=0-15 \
RUN_NAME=antioverfill_rank_w001_support_s200_val16_mesh_relcap \
POINT_RUN_ROOT=/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_rank_w001_support_s200 \
KNOWN_CLAMP_START_T=0.5 \
TOPK_SPECS=r0.35_cap4096,r0.50_cap8192,r0.75_cap12000,target_unique \
bash trellis_point_prior_mv/scripts/run_mesh_frozen_topk_sweep.sh
```

第二优先级：如果 `val16` 仍稳定，再做 `val32`，但可以只测两个 top-k，节省时间：

```bash
cd /home/zjr/Tracker

GPU=1 \
MODE=val8 \
INDICES=0-31 \
RUN_NAME=antioverfill_rank_w001_support_s200_val32_mesh_r05_r075 \
POINT_RUN_ROOT=/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_rank_w001_support_s200 \
KNOWN_CLAMP_START_T=0.5 \
TOPK_SPECS=r0.50_cap8192,r0.75_cap12000 \
bash trellis_point_prior_mv/scripts/run_mesh_frozen_topk_sweep.sh
```

第三优先级：暂时不建议马上加 target-support positive。已有 `rank_x0p5` 证明强 target-support 会压坏 sparse flow。如果后面还要试，只能在当前 best 基础上小范围 smoke：

```text
RANKING_TARGET_SUPPORT_WEIGHT=0.05 或 0.1
RANKING_LOSS_WEIGHT=0.01
```

第四优先级：进入真实 AR/SLAM 前，应实现候选 top-k mesh rerank，而不是固定单一 top-k。候选集合建议：

```text
r0.35_cap4096
r0.50_cap8192
r0.75_cap12000
```

rerank 指标优先使用 AR/SLAM 点云到 mesh 的距离、mask projection consistency、mesh extent sanity check。

## 2026-06-22 补充：rank_w001_support s200 val16 / val32 扩展 mesh eval

本轮运行：

```text
antioverfill_rank_w001_support_s200_val16_mesh_relcap
antioverfill_rank_w001_support_s200_val32_mesh_r05_r075
```

结果路径：

```text
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/antioverfill_rank_w001_support_s200_val16_mesh_relcap/report.json
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/antioverfill_rank_w001_support_s200_val32_mesh_r05_r075/report.json
```

### 汇总结果

| eval | top-k | count | sparse IoU | precision | recall | Chamfer L2 | target->mesh | mesh->target | extent ratio | vertices |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val8 | r0.50_cap8192 | 8 | 0.1381 | 0.3751 | 0.1797 | 0.0100 | 0.0489 | 0.0448 | 0.6875 | 47990 |
| val8 | r0.75_cap12000 | 8 | 0.1686 | 0.3465 | 0.2480 | 0.0095 | 0.0417 | 0.0471 | 0.7061 | 86312 |
| val16 | r0.35_cap4096 | 16 | 0.1136 | 0.4140 | 0.1352 | 0.0094 | 0.0524 | 0.0379 | 0.7093 | 28774 |
| val16 | r0.50_cap8192 | 16 | 0.1443 | 0.3822 | 0.1872 | 0.0113 | 0.0490 | 0.0489 | 0.7357 | 46035 |
| val16 | r0.75_cap12000 | 16 | 0.1726 | 0.3472 | 0.2545 | 0.0103 | 0.0406 | 0.0497 | 0.7494 | 86309 |
| val16 | target_unique | 16 | 0.1919 | 0.3208 | 0.3208 | 0.0114 | 0.0367 | 0.0551 | 0.7692 | 165910 |
| val32 | r0.50_cap8192 | 32 | 0.1415 | 0.3760 | 0.1843 | 0.0119 | 0.0508 | 0.0489 | 0.7240 | 43707 |
| val32 | r0.75_cap12000 | 32 | 0.1690 | 0.3408 | 0.2502 | 0.0117 | 0.0449 | 0.0506 | 0.7390 | 83985 |

### 相对 stock_sparse 胜率

| eval | top-k | Chamfer win | target->mesh win | mesh->target win | mesh success |
|---|---|---:|---:|---:|---:|
| val8 | r0.50_cap8192 | 8/8 | 8/8 | 7/8 | 8/8 |
| val8 | r0.75_cap12000 | 8/8 | 8/8 | 5/8 | 8/8 |
| val16 | r0.50_cap8192 | 15/16 | 15/16 | 10/16 | 16/16 |
| val16 | r0.75_cap12000 | 15/16 | 16/16 | 8/16 | 16/16 |
| val32 | r0.50_cap8192 | 29/32 | 30/32 | 24/32 | 32/32 |
| val32 | r0.75_cap12000 | 29/32 | 31/32 | 21/32 | 32/32 |

### 稳定性观察

`r0.50_cap8192` 从 val8 到 val32：

```text
Chamfer:      0.0100 -> 0.0119
target->mesh: 0.0489 -> 0.0508
mesh->target: 0.0448 -> 0.0489
extent:       0.6875 -> 0.7240
precision:    0.3751 -> 0.3760
recall:       0.1797 -> 0.1843
```

指标有轻微回落，但没有崩。`r0.50` 仍是默认工程点，因为它在 precision、mesh->target 和 extent 上更稳。

`r0.75_cap12000` 的特点：

```text
更高 recall:       val32 0.2502 vs r0.50 0.1843
更好 target->mesh: val32 0.0449 vs r0.50 0.0508
更高 mesh->target: val32 0.0506 vs r0.50 0.0489
更高 extent:       val32 0.7390 vs r0.50 0.7240
```

所以 `r0.75` 可以作为覆盖优先候选，但不适合作为唯一默认。真实系统里应该保留 `r0.50` 和 `r0.75` 两个候选，用 AR/SLAM 点云和 mask projection 做 rerank。

### Worst case

val32 的最差 Chamfer 样本：

```text
r0.50:
  idx18  chamfer=0.0333  extent=0.6751
  idx23  chamfer=0.0251  extent=0.7993
  idx13  chamfer=0.0202  extent=0.7611

r0.75:
  idx18  chamfer=0.0426  extent=0.7857
  idx23  chamfer=0.0204  extent=0.8174
  idx26  chamfer=0.0193  extent=0.9808
```

这说明当前方法不是所有样本都稳定收益。尤其 `r0.75` 在个别样本上会显著过扩张，因此后续必须有 candidate rerank 或 sanity check，不能只固定 top-k。

## 当前方向判断

方向是正确的，而且已经不是“只在 sparse 指标上看起来好”的阶段。

已有证据：

```text
1. sparse eval：rank_w001_support_s200 明显优于 anti-overfill_s200；
2. frozen downstream mesh eval：val8 / val16 / val32 都保持高胜率；
3. r0.50 在 val32 上 mesh success 32/32；
4. r0.50 对 stock_sparse 的 Chamfer 胜率 29/32，target->mesh 胜率 30/32；
5. r0.75 提供更高覆盖，但会带来更多外扩风险。
```

合理预期收益：

```text
短期：提高 sparse precision / recall，同时降低明显过扩张；
中期：通过候选 top-k rerank，把 r0.50 的稳和 r0.75 的覆盖优势结合起来；
长期：如果训练 slat/mesh downstream，可能进一步消化 point-prior sparse 的分布变化。
```

但收益边界也很清楚：

```text
1. 它不是完整 unknown completion 解法；
2. 它主要提升 observed/known 支撑和 sparse precision；
3. frozen stock slat/mesh 仍不是专门适配这个 sparse distribution；
4. 个别样本仍会失败，特别是高 top-k 的过扩张。
```

## 完整训练前还需要的测试

现在还不建议直接投入 full-scale / all-data 长训。进入完整训练前至少补以下测试。

第一，做同一 val32 上的强基线对比。

目前 `rank_w001_support_s200` 已经有 val32，但 anti-overfill_s200 只有 val8。需要补一个公平的 anti-overfill_s200 val32 对照：

```bash
cd /home/zjr/Tracker

GPU=1 \
MODE=val8 \
INDICES=0-31 \
RUN_NAME=antioverfill_s200_val32_mesh_r05_r075 \
POINT_RUN_ROOT=/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_s200 \
KNOWN_CLAMP_START_T=0.5 \
TOPK_SPECS=r0.50_cap8192,r0.75_cap12000 \
bash trellis_point_prior_mv/scripts/run_mesh_frozen_topk_sweep.sh
```

第二，做 val64 轻量 mesh eval。

不要全 top-k，先只跑 `r0.50`：

```bash
cd /home/zjr/Tracker

GPU=1 \
MODE=val8 \
INDICES=0-63 \
RUN_NAME=antioverfill_rank_w001_support_s200_val64_mesh_r05 \
POINT_RUN_ROOT=/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_rank_w001_support_s200 \
KNOWN_CLAMP_START_T=0.5 \
TOPK_SPECS=r0.50_cap8192 \
bash trellis_point_prior_mv/scripts/run_mesh_frozen_topk_sweep.sh
```

第三，做训练随机性检查。

当前只有一个训练 seed。完整训练前建议至少补一个 `s200_seed123`，看 ranking 是否稳定复现。若脚本还没有 seed 参数，需要先接入 seed。

第四，做 noisy prior 泛化。

现在 best 是较干净 prior setting。真实 AR/SLAM 点云会有 dropout、jitter、outlier。需要补：

```text
DROPOUT_MAX=0.25
OUTLIER_RATIO=0.03
COORD_JITTER=1
```

至少跑 smoke 和 sparse eval，确认不会把模型训练成只吃 oracle-like prior。

第五，做真实 AR/SLAM candidate rerank 原型。

完整训练前，必须证明 `r0.50/r0.75` 候选能被真实可用信号区分。建议 rerank 指标：

```text
1. AR/SLAM object points -> mesh distance
2. mesh rendered mask -> input mask IoU
3. mesh extent / aspect sanity
4. visible view support ratio
```

第六，再考虑 slat/mesh downstream 训练。

当前所有 mesh eval 都是 frozen stock downstream。只有当 sparse best 在更大 val 和 noisy prior 下稳定，才值得训练 slat/mesh flow；否则 downstream 会学习 sparse 阶段的偶发外扩错误。

## 是否可以进入完整训练

现在的结论：

```text
可以进入“扩大验证 + 准完整训练准备”；
还不建议直接进入最终 full-scale 长训。
```

最小门槛：

```text
1. rank_w001_support_s200 在 val64 r0.50 上仍稳定；
2. anti-overfill_s200 val32 对照确认 rank 不是样本选择优势；
3. noisy prior smoke 不崩；
4. 至少一个第二 seed 没有明显退化；
5. candidate rerank 在真实或半真实 AR 数据上能选对 r0.50/r0.75。
```

## 2026-06-22 补充：anti-overfill val32 对照与 rank best val64

本轮运行：

```text
antioverfill_s200_val32_mesh_r05_r075
antioverfill_rank_w001_support_s200_val64_mesh_r05
```

结果路径：

```text
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/antioverfill_s200_val32_mesh_r05_r075/report.json
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/antioverfill_rank_w001_support_s200_val64_mesh_r05/report.json
```

### 同样 val32 上的公平对照

| run | top-k | count | sparse IoU | precision | recall | Chamfer L2 | target->mesh | mesh->target | extent ratio | vertices |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| anti-overfill s200 | r0.50_cap8192 | 32 | 0.1183 | 0.3215 | 0.1572 | 0.0185 | 0.0553 | 0.0702 | 0.7715 | 39183 |
| rank_w001_support s200 | r0.50_cap8192 | 32 | 0.1415 | 0.3760 | 0.1843 | 0.0119 | 0.0508 | 0.0489 | 0.7240 | 43707 |
| anti-overfill s200 | r0.75_cap12000 | 32 | 0.1447 | 0.2987 | 0.2187 | 0.0180 | 0.0451 | 0.0733 | 0.7798 | 79839 |
| rank_w001_support s200 | r0.75_cap12000 | 32 | 0.1690 | 0.3408 | 0.2502 | 0.0117 | 0.0449 | 0.0506 | 0.7390 | 83985 |

paired delta，rank - anti：

| top-k | metric | rank mean | anti mean | delta | rank better |
|---|---|---:|---:|---:|---:|
| r0.50 | sparse IoU | 0.1415 | 0.1183 | +0.0233 | 28/32 |
| r0.50 | precision | 0.3760 | 0.3215 | +0.0545 | 28/32 |
| r0.50 | recall | 0.1843 | 0.1572 | +0.0271 | 28/32 |
| r0.50 | Chamfer | 0.0119 | 0.0185 | -0.0067 | 27/32 |
| r0.50 | target->mesh | 0.0508 | 0.0553 | -0.0045 | 22/32 |
| r0.50 | mesh->target | 0.0489 | 0.0702 | -0.0212 | 30/32 |
| r0.50 | extent ratio | 0.7240 | 0.7715 | -0.0475 | 20/32 |
| r0.75 | sparse IoU | 0.1690 | 0.1447 | +0.0243 | 28/32 |
| r0.75 | precision | 0.3408 | 0.2987 | +0.0422 | 28/32 |
| r0.75 | recall | 0.2502 | 0.2187 | +0.0315 | 28/32 |
| r0.75 | Chamfer | 0.0117 | 0.0180 | -0.0062 | 26/32 |
| r0.75 | target->mesh | 0.0449 | 0.0451 | -0.0002 | 17/32 |
| r0.75 | mesh->target | 0.0506 | 0.0733 | -0.0227 | 29/32 |
| r0.75 | extent ratio | 0.7390 | 0.7798 | -0.0408 | 18/32 |

结论：`rank_w001_support_s200` 的收益不是样本选择造成的。同样 val32 上，它在 sparse、Chamfer、mesh->target 上显著优于纯 anti-overfill；target->mesh 的提升较小，尤其 `r0.75` 基本持平，但没有变差。

### val64 r0.50 稳定性

`rank_w001_support_s200` 的 val64 r0.50：

| metric | mean | median | p90 | p95 | max |
|---|---:|---:|---:|---:|---:|
| sparse IoU | 0.1412 | 0.1331 | 0.1764 | 0.1995 | 0.2776 |
| precision | 0.3726 | 0.3618 | 0.4499 | 0.4990 | 0.6517 |
| recall | 0.1833 | 0.1762 | 0.2250 | 0.2494 | 0.3260 |
| Chamfer L2 | 0.0131 | 0.0103 | 0.0224 | 0.0253 | 0.0540 |
| target->mesh | 0.0500 | 0.0474 | 0.0700 | 0.0815 | 0.1213 |
| mesh->target | 0.0506 | 0.0477 | 0.0740 | 0.0928 | 0.1498 |
| extent ratio | 0.7298 | 0.7433 | 0.8860 | 0.9185 | 0.9500 |

相对 stock_sparse 胜率：

```text
Chamfer:      57/64
target->mesh: 61/64
mesh->target: 49/64
mesh success: 64/64
```

val64 比 val32 略回落：

```text
Chamfer:      0.0119 -> 0.0131
target->mesh: 0.0508 -> 0.0500
mesh->target: 0.0489 -> 0.0506
extent:       0.7240 -> 0.7298
```

但整体仍稳定，尤其 target->mesh 没有退化，mesh success 全部成功。

### Worst cases

val64 最差 Chamfer：

```text
idx61  bbbb22f40d6a4914  chamfer=0.0540  target->mesh=0.1213  mesh->target=0.1498  extent=0.9403
idx58  a80bd429f8854440  chamfer=0.0366  target->mesh=0.0946  mesh->target=0.1095  extent=0.8537
idx36  c24f4cdb196a4367  chamfer=0.0363  target->mesh=0.0424  mesh->target=0.1368  extent=0.6743
idx18  0acc5836ade54aae  chamfer=0.0337  target->mesh=0.0743  mesh->target=0.1198  extent=0.6752
idx23  6ca57851490f4bab  chamfer=0.0253  target->mesh=0.0534  mesh->target=0.0871  extent=0.7994
```

这些 worst cases 说明问题仍集中在少数样本的 mesh-to-target 和 Chamfer；这类失败通常不是简单调 top-k 能完全解决，需要候选 rerank 或输入 prior 质量诊断。

## 更新后的判断

当前方向可以认为已经通过了“扩大验证”的第一关。

更具体地说：

```text
1. ranking 确实优于 anti-overfill，不只是 val8 偶然；
2. val64 r0.50 没崩；
3. 当前 best checkpoint 可以作为后续候选 pipeline 的 sparse checkpoint；
4. 但还不能直接宣称可以最终 full-scale，因为还缺 seed 和 noisy prior 泛化。
```

当前推荐版本保持不变：

```text
checkpoint:
/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_rank_w001_support_s200/checkpoints/last.ckpt

default sparse top-k:
r0.50_cap8192

candidate top-k:
r0.50_cap8192, r0.75_cap12000
```

## 下一步建议

第一优先级：补第二 seed。

只有一个 seed 时，不能排除 ranking loss 的偶然性。建议接入/使用 seed 参数，跑：

```text
pointprior_pixal_v9_stage2_antioverfill_rank_w001_support_s200_seed123
```

已接入：

```text
BUILD_SEED
TRAIN_SEED
```

如果只想检查训练随机性，建议固定 `BUILD_SEED=42`，只改 `TRAIN_SEED=123`：

```bash
cd /home/zjr/Tracker

GPU=1 \
MODE=s200 \
RUN_NAME=pointprior_pixal_v9_stage2_antioverfill_rank_w001_support_s200_seed123 \
BUILD_SEED=42 \
TRAIN_SEED=123 \
ANTI_OVERFILL_LOSS_WEIGHT=0.02 \
KNOWN_FLOW_LOSS_WEIGHT=2.0 \
KNOWN_X0_LOSS_WEIGHT=0.5 \
KNOWN_USE_CONFIDENCE=0 \
RANKING_LOSS_WEIGHT=0.01 \
RANKING_OUTSIDE_WEIGHT=0.0 \
RANKING_OBSERVED_WEIGHT=0.25 \
RANKING_WRONG_SUPPORT_WEIGHT=1.0 \
RANKING_TARGET_SUPPORT_WEIGHT=0.0 \
RANKING_NEGATIVE_MODES=shuffle,random \
KNOWN_CLAMP_START_T=0.5 \
CLAMP_INITIAL_NOISE=0 \
MAX_STEPS=200 \
SAVE_EVERY=100 \
EVAL_STEPS=12 \
bash trellis_point_prior_mv/scripts/run_pixal_v9_point_prior_stage2.sh
```

第二优先级：补 noisy prior smoke。

真实 AR/SLAM 点云会有噪声，目前 best 更接近 clean prior。建议先做 smoke：

```text
DROPOUT_MAX=0.25
OUTLIER_RATIO=0.03
COORD_JITTER=1
MAX_STEPS=20
```

第三优先级：实现候选 top-k rerank 原型。

现在已经证明 `r0.50` 稳、`r0.75` 覆盖更强。下一阶段不要固定单一 top-k，而应该生成两个候选：

```text
r0.50_cap8192
r0.75_cap12000
```

用下面信号排序：

```text
1. AR/SLAM object points -> mesh distance
2. render mask -> input mask IoU
3. mesh extent sanity
4. multi-view visible support
```

第四优先级：在上述通过后，再进入更大规模训练。

进入完整训练的最低条件现在变成：

```text
1. seed123 s200 不明显差于当前 seed；
2. noisy prior smoke 不崩；
3. candidate rerank 能减少 val64 worst cases 或至少不选明显外扩的候选；
4. 再决定是否训练 slat/mesh downstream。
```

## 2026-06-22 补充：seed123 复验结果

本轮运行：

```text
pointprior_pixal_v9_stage2_antioverfill_rank_w001_support_s200_seed123
antioverfill_rank_w001_support_s200_seed123_val32_mesh_r05
```

结果路径：

```text
/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_rank_w001_support_s200_seed123/eval/report.json
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/antioverfill_rank_w001_support_s200_seed123_val32_mesh_r05/report.json
```

时间戳检查显示 mesh eval 在 seed123 训练和 sparse eval 之后完成，因此该 report 有效。

### Sparse eval 对比

| run | count | IoU | precision | recall | known IoU | unknown IoU | obs-known IoU | obs-unknown IoU | obs-unknown recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rank seed42 | 96 | 0.1634 | 0.3148 | 0.2821 | 0.2918 | 0.0197 | 0.2918 | 0.0197 | 0.0672 |
| rank seed123 | 96 | 0.1461 | 0.2881 | 0.2556 | 0.2651 | 0.0191 | 0.2651 | 0.0191 | 0.0659 |
| anti-overfill s200 | 96 | 0.1463 | 0.2864 | 0.2566 | 0.2699 | 0.0179 | 0.2699 | 0.0179 | 0.0626 |

rank 指标：

| run | correct top1 | obs-unknown IoU top1 | obs-unknown recall top1 |
|---|---:|---:|---:|
| rank seed42 | 96/96 | 57/96 | 35/96 |
| rank seed123 | 96/96 | 61/96 | 37/96 |

解释：

```text
seed123 没有破坏 correct top1；
obs-unknown rank 甚至略好；
但全局 sparse IoU / precision / recall 明显弱于 seed42；
seed123 的全局 sparse 指标基本回到 anti-overfill s200 附近。
```

### Mesh eval 对比

同样 val32、`r0.50_cap8192`：

| run | sparse IoU | precision | recall | Chamfer L2 | target->mesh | mesh->target | extent ratio | vertices |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| anti-overfill s200 | 0.1183 | 0.3215 | 0.1572 | 0.0185 | 0.0553 | 0.0702 | 0.7715 | 39183 |
| rank seed42 | 0.1415 | 0.3760 | 0.1843 | 0.0119 | 0.0508 | 0.0489 | 0.7240 | 43707 |
| rank seed123 | 0.1235 | 0.3335 | 0.1631 | 0.0138 | 0.0548 | 0.0535 | 0.8598 | 35808 |

相对 stock_sparse 胜率：

| run | Chamfer win | target->mesh win | mesh->target win | mesh success |
|---|---:|---:|---:|---:|
| anti-overfill s200 | 28/32 | 30/32 | 20/32 | 32/32 |
| rank seed42 | 29/32 | 30/32 | 24/32 | 32/32 |
| rank seed123 | 29/32 | 31/32 | 24/32 | 32/32 |

paired delta：

| compare | metric | delta | better count |
|---|---|---:|---:|
| seed123 - anti | sparse IoU | +0.0052 | 19/32 |
| seed123 - anti | precision | +0.0120 | 19/32 |
| seed123 - anti | recall | +0.0060 | 19/32 |
| seed123 - anti | Chamfer | -0.0048 | 23/32 |
| seed123 - anti | target->mesh | -0.0004 | 18/32 |
| seed123 - anti | mesh->target | -0.0167 | 27/32 |
| seed123 - anti | extent ratio | +0.0883 | 10/32 |
| seed123 - seed42 | sparse IoU | -0.0180 | 4/32 |
| seed123 - seed42 | precision | -0.0424 | 4/32 |
| seed123 - seed42 | recall | -0.0212 | 4/32 |
| seed123 - seed42 | Chamfer | +0.0019 | 11/32 |
| seed123 - seed42 | target->mesh | +0.0041 | 13/32 |
| seed123 - seed42 | mesh->target | +0.0045 | 12/32 |
| seed123 - seed42 | extent ratio | +0.1358 | 1/32 |

### Worst cases

seed123 最差 Chamfer：

```text
idx28  d08a2722451d499a  chamfer=0.0376  target->mesh=0.0405  mesh->target=0.1055  extent=0.9171
idx18  0acc5836ade54aae  chamfer=0.0336  target->mesh=0.0740  mesh->target=0.1250  extent=0.9008
idx30  f030894a090d4f3c  chamfer=0.0301  target->mesh=0.0911  mesh->target=0.1067  extent=0.8871
```

seed123 最差 extent：

```text
idx26  extent=0.9889
idx11  extent=0.9884
idx24  extent=0.9833
idx15  extent=0.9745
idx16  extent=0.9538
```

seed123 的主要问题不是 mesh 生成失败，而是更容易外扩。它的 `extent_ratio mean=0.8598`，显著差于 seed42 的 `0.7240`，也差于 anti-overfill 的 `0.7715`。

## seed123 后的结论

方向仍然成立，但 seed 稳定性不足以支撑直接进入最终 full-scale。

更准确地说：

```text
1. ranking 不是偶然完全无效：seed123 仍优于 anti-overfill 的 Chamfer 和 mesh->target；
2. seed123 没有复现 seed42 的强 sparse / mesh 收益；
3. seed123 的 extent 明显变差，说明当前 ranking loss 对训练随机性敏感；
4. 当前 best 仍是 seed42 last.ckpt，但不能只靠这一条 run 宣称稳定。
```

当前状态应从：

```text
可以准备完整训练
```

调整为：

```text
需要先提升/验证训练稳定性，再进入完整训练。
```

## 下一步建议更新

第一优先级：不要直接 full-scale。先做“小改稳定性”的 smoke。

建议把 ranking 再降一点，减少 seed sensitivity：

```text
RANKING_LOSS_WEIGHT=0.005
RANKING_OBSERVED_WEIGHT=0.25
RANKING_WRONG_SUPPORT_WEIGHT=0.5 或 0.75
RANKING_OUTSIDE_WEIGHT=0.0
RANKING_TARGET_SUPPORT_WEIGHT=0.0
ANTI_OVERFILL_LOSS_WEIGHT=0.02
```

目标不是追 seed42 的最高收益，而是让 seed42/seed123 都稳定优于 anti-overfill，并且 extent 不反弹。

第二优先级：补 `seed123` 的 r0.75 mesh eval 不急。

因为 seed123 的 `r0.50` extent 已经明显偏大，`r0.75` 大概率会进一步外扩。除非要分析候选 rerank，否则优先级低于稳定训练。

第三优先级：做 noisy prior smoke 仍然需要，但应放在稳定性 smoke 之后。

如果 clean seed 都不稳定，noisy prior 的结论会混杂训练随机性和输入噪声。

第四优先级：如果要继续用当前 best 做 AR pipeline 原型，可以暂时使用 seed42：

```text
/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_rank_w001_support_s200/checkpoints/last.ckpt
```

但它应标注为：

```text
best experimental checkpoint, not yet robust final checkpoint
```

## 2026-06-22 补充：w0.005 ranking 稳定性实验

本轮运行：

```text
pointprior_pixal_v9_stage2_antioverfill_rank_w0005_ws05_smoke
antioverfill_rank_w0005_ws05_smoke_val16_mesh_r05
pointprior_pixal_v9_stage2_antioverfill_rank_w0005_ws075_smoke
antioverfill_rank_w0005_ws075_smoke_val16_mesh_r05
pointprior_pixal_v9_stage2_antioverfill_rank_w0005_ws05_s200_seed123
antioverfill_rank_w0005_ws05_s200_seed123_val32_mesh_r05
```

其中 `pointprior_pixal_v9_stage2_antioverfill_rank_w0005_ws075_s200_seed123` 目前只看到 data build 文件，没有 checkpoint / eval report，因此不纳入本轮结论。

结果路径：

```text
/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_rank_w0005_ws05_smoke/eval/report.json
/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_rank_w0005_ws075_smoke/eval/report.json
/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_rank_w0005_ws05_s200_seed123/eval/report.json
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/antioverfill_rank_w0005_ws05_smoke_val16_mesh_r05/report.json
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/antioverfill_rank_w0005_ws075_smoke_val16_mesh_r05/report.json
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/antioverfill_rank_w0005_ws05_s200_seed123_val32_mesh_r05/report.json
```

### Sparse eval

| run | count | IoU | precision | recall | known IoU | unknown IoU | obs-known IoU | obs-unknown IoU | obs-unknown recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w0.005/ws0.5 smoke | 24 | 0.1318 | 0.2775 | 0.2184 | 0.2440 | 0.0233 | 0.2440 | 0.0233 | 0.0672 |
| w0.005/ws0.75 smoke | 24 | 0.1321 | 0.2777 | 0.2188 | 0.2444 | 0.0229 | 0.2444 | 0.0229 | 0.0664 |
| w0.005/ws0.5 s200 seed123 | 96 | 0.1509 | 0.2972 | 0.2628 | 0.2735 | 0.0192 | 0.2735 | 0.0192 | 0.0652 |
| w0.01/ws1.0 s200 seed123 | 96 | 0.1461 | 0.2881 | 0.2556 | 0.2651 | 0.0191 | 0.2651 | 0.0191 | 0.0659 |
| w0.01/ws1.0 s200 seed42 | 96 | 0.1634 | 0.3148 | 0.2821 | 0.2918 | 0.0197 | 0.2918 | 0.0197 | 0.0672 |
| anti-overfill s200 | 96 | 0.1463 | 0.2864 | 0.2566 | 0.2699 | 0.0179 | 0.2699 | 0.0179 | 0.0626 |

结论：

```text
w0.005/ws0.5 seed123 比 w0.01/ws1.0 seed123 略好；
w0.005/ws0.5 seed123 也略好于 anti-overfill s200；
但它仍明显弱于 w0.01/ws1.0 seed42。
```

### Mesh eval

| run | count | sparse IoU | precision | recall | Chamfer L2 | target->mesh | mesh->target | extent ratio | vertices |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w0.005/ws0.5 smoke val16 | 16 | 0.1163 | 0.3150 | 0.1537 | 0.0110 | 0.0519 | 0.0513 | 0.7101 | 41090 |
| w0.005/ws0.75 smoke val16 | 16 | 0.1169 | 0.3163 | 0.1544 | 0.0116 | 0.0529 | 0.0499 | 0.7121 | 42764 |
| w0.005/ws0.5 s200 seed123 val32 | 32 | 0.1264 | 0.3407 | 0.1667 | 0.0123 | 0.0524 | 0.0505 | 0.8385 | 39733 |
| w0.01/ws1.0 s200 seed123 val32 | 32 | 0.1235 | 0.3335 | 0.1631 | 0.0138 | 0.0548 | 0.0535 | 0.8598 | 35808 |
| w0.01/ws1.0 s200 seed42 val32 | 32 | 0.1415 | 0.3760 | 0.1843 | 0.0119 | 0.0508 | 0.0489 | 0.7240 | 43707 |
| anti-overfill s200 val32 | 32 | 0.1183 | 0.3215 | 0.1572 | 0.0185 | 0.0553 | 0.0702 | 0.7715 | 39183 |

相对 stock_sparse 胜率：

| run | Chamfer win | target->mesh win | mesh->target win | success |
|---|---:|---:|---:|---:|
| w0.005/ws0.5 smoke val16 | 14/16 | 15/16 | 9/16 | 16/16 |
| w0.005/ws0.75 smoke val16 | 14/16 | 15/16 | 11/16 | 16/16 |
| w0.005/ws0.5 s200 seed123 val32 | 30/32 | 30/32 | 25/32 | 32/32 |
| w0.01/ws1.0 s200 seed123 val32 | 29/32 | 31/32 | 24/32 | 32/32 |
| w0.01/ws1.0 s200 seed42 val32 | 29/32 | 30/32 | 24/32 | 32/32 |
| anti-overfill s200 val32 | 28/32 | 30/32 | 20/32 | 32/32 |

paired delta：

| compare | metric | delta | better count |
|---|---|---:|---:|
| w0.005/ws0.5 seed123 - anti | sparse IoU | +0.0081 | 22/32 |
| w0.005/ws0.5 seed123 - anti | precision | +0.0192 | 22/32 |
| w0.005/ws0.5 seed123 - anti | recall | +0.0095 | 22/32 |
| w0.005/ws0.5 seed123 - anti | Chamfer | -0.0062 | 25/32 |
| w0.005/ws0.5 seed123 - anti | target->mesh | -0.0029 | 23/32 |
| w0.005/ws0.5 seed123 - anti | mesh->target | -0.0197 | 27/32 |
| w0.005/ws0.5 seed123 - anti | extent ratio | +0.0671 | 14/32 |
| w0.005/ws0.5 seed123 - w0.01/ws1.0 seed123 | sparse IoU | +0.0029 | 24/32 |
| w0.005/ws0.5 seed123 - w0.01/ws1.0 seed123 | Chamfer | -0.0014 | 21/32 |
| w0.005/ws0.5 seed123 - w0.01/ws1.0 seed123 | mesh->target | -0.0030 | 20/32 |
| w0.005/ws0.5 seed123 - w0.01/ws1.0 seed123 | extent ratio | -0.0212 | 23/32 |
| w0.005/ws0.5 seed123 - w0.01/ws1.0 seed42 | sparse IoU | -0.0152 | 4/32 |
| w0.005/ws0.5 seed123 - w0.01/ws1.0 seed42 | Chamfer | +0.0005 | 19/32 |
| w0.005/ws0.5 seed123 - w0.01/ws1.0 seed42 | extent ratio | +0.1146 | 6/32 |

### Worst cases

`w0.005/ws0.5 s200 seed123` 最差 Chamfer：

```text
idx26  chamfer=0.0322  target->mesh=0.0779  mesh->target=0.0922  extent=0.9956
idx28  chamfer=0.0262  target->mesh=0.0444  mesh->target=0.0960  extent=0.9172
idx18  chamfer=0.0260  target->mesh=0.0729  mesh->target=0.0980  extent=0.9449
idx30  chamfer=0.0234  target->mesh=0.0818  mesh->target=0.0848  extent=0.7646
idx23  chamfer=0.0198  target->mesh=0.0784  mesh->target=0.0615  extent=0.8222
```

主要失败仍是外扩。`extent_ratio mean=0.8385`，虽然比 `w0.01/ws1.0 seed123` 的 `0.8598` 好，但仍显著差于：

```text
anti-overfill s200: 0.7715
w0.01/ws1.0 seed42: 0.7240
```

### 本轮结论

`w0.005/ws0.5` 是对 `seed123` 更稳的方向，但只解决了一部分问题。

可以确认：

```text
1. 降低 ranking loss 从 0.01 到 0.005 后，seed123 比原 w0.01/ws1.0 seed123 稳；
2. w0.005/ws0.5 seed123 相比 anti-overfill 在 Chamfer、mesh->target、sparse 上有收益；
3. 但 extent 仍偏大，说明 ranking 仍会在长训后诱发外扩；
4. smoke 的 extent 很好，但 s200 后 extent 变差，说明问题不是初始方向，而是长训阶段的 calibration 漂移。
```

当前最优实验 checkpoint 仍是：

```text
/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_rank_w001_support_s200/checkpoints/last.ckpt
```

但更稳的候选方向是：

```text
RANKING_LOSS_WEIGHT=0.005
RANKING_WRONG_SUPPORT_WEIGHT=0.5
```

它需要进一步控制 s200 后的 extent。

## 下一步建议更新

第一优先级：不要继续增大 ranking，也不要直接 full-scale。

当前问题不是 ranking 没效果，而是长训后 extent calibration 不稳。下一步应加一个 eval-time 或 train-time 的 extent / overfill 控制，而不是继续调大 ranking。

第二优先级：先评估 `w0.005/ws0.5 s200 seed123` 的 step100。

这组 smoke extent 好、s200 extent 差，需要确认是不是训练中后期漂移。如果 step100 extent 明显更好，就说明早停比继续加 loss 更直接。

建议命令：

```bash
cd /home/zjr/Tracker

GPU=4 \
MODE=val8 \
INDICES=0-31 \
RUN_NAME=antioverfill_rank_w0005_ws05_s200_seed123_step100_val32_mesh_r05 \
POINT_RUN_ROOT=/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_rank_w0005_ws05_s200_seed123 \
STAGE2_CHECKPOINT=/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_rank_w0005_ws05_s200_seed123/checkpoints/ss-pointprior-stage2-epoch=00-step=100.ckpt \
KNOWN_CLAMP_START_T=0.5 \
TOPK_SPECS=r0.50_cap8192 \
bash trellis_point_prior_mv/scripts/run_mesh_frozen_topk_sweep.sh
```

第三优先级：试更强 anti-overfill、弱 ranking 的 smoke。

目标是保持 ranking 的 sparse 收益，同时压住 extent：

```text
ANTI_OVERFILL_LOSS_WEIGHT=0.03
RANKING_LOSS_WEIGHT=0.005
RANKING_WRONG_SUPPORT_WEIGHT=0.5
```

只做 smoke，不直接 s200。

第四优先级：候选 rerank 仍然需要。

现在已经看到不同 seed 的外扩样本不同，单一 checkpoint / top-k 不可能完全稳定。进入真实 AR pipeline 前，需要用：

```text
AR/SLAM points -> mesh distance
render mask IoU
extent sanity
visible support
```

筛掉外扩候选。

## 三十、step100 与 anti-overfill=0.03 补测结果

### 输入结果路径

```text
step100 mesh:
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/antioverfill_rank_w0005_ws05_s200_seed123_step100_val32_mesh_r05/report.json

anti-overfill=0.03 smoke sparse:
/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill003_rank_w0005_ws05_smoke/eval/report.json

anti-overfill=0.03 smoke mesh:
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/antioverfill003_rank_w0005_ws05_smoke_val16_mesh_r05/report.json

anti-overfill=0.03 s200 sparse:
/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill003_rank_w0005_ws05_s200_seed123/eval/report.json
```

注意：本次没有在预期位置找到 `anti-overfill=0.03 s200` 的 mesh eval report：

```text
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/antioverfill003_rank_w0005_ws05_s200_seed123_val32_mesh_r05/report.json
```

同时该训练目录下存在两套 checkpoint：

```text
last.ckpt
last-v1.ckpt
ss-pointprior-stage2-epoch=00-step=100.ckpt
ss-pointprior-stage2-epoch=00-step=100-v1.ckpt
ss-pointprior-stage2-epoch=00-step=200.ckpt
ss-pointprior-stage2-epoch=00-step=200-v1.ckpt
```

因此如果后续补跑这组 mesh eval，必须显式传入 `STAGE2_CHECKPOINT`，不要依赖脚本默认 `last.ckpt`。

### Sparse 结果

| run | count | IoU | precision | recall | known IoU | unknown IoU | obs unknown recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| anti003 smoke | 24 | 0.1320 | 0.2779 | 0.2187 | 0.2442 | 0.0232 | 0.0670 |
| anti003 s200 seed123 | 96 | 0.1235 | 0.2487 | 0.2213 | 0.2325 | 0.0178 | 0.0637 |
| w0.005/ws0.5 s200 seed123 | 96 | 0.1509 | 0.2972 | 0.2628 | 0.2735 | 0.0192 | 0.0652 |
| w0.01/ws1.0 s200 seed123 | 96 | 0.1461 | 0.2881 | 0.2556 | 0.2651 | 0.0191 | 0.0659 |
| w0.01/ws1.0 s200 seed42 | 96 | 0.1633 | 0.3148 | 0.2821 | 0.2918 | 0.0197 | 0.0672 |
| anti-overfill s200 | 96 | 0.1463 | 0.2864 | 0.2566 | 0.2699 | 0.0179 | 0.0626 |

解释：

```text
anti-overfill=0.03 在 smoke 上没有异常；
但 s200 后 sparse IoU/precision/known IoU 明显低于 0.02 的 w0.005/ws0.5 s200；
这说明 0.03 不是当前更稳的长训候选。
```

### Mesh 结果

| run | count | sparse IoU | precision | recall | Chamfer | target->mesh | mesh->target | extent | vertices |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w0.005/ws0.5 step100 seed123 val32 | 32 | 0.1023 | 0.2815 | 0.1377 | 0.0183 | 0.0581 | 0.0689 | 0.9174 | 33949 |
| anti003 w0.005/ws0.5 smoke val16 | 16 | 0.1168 | 0.3161 | 0.1543 | 0.0110 | 0.0505 | 0.0506 | 0.6998 | 43710 |
| w0.005/ws0.5 s200 seed123 val32 | 32 | 0.1264 | 0.3407 | 0.1667 | 0.0123 | 0.0524 | 0.0505 | 0.8385 | 39733 |
| w0.01/ws1.0 s200 seed123 val32 | 32 | 0.1235 | 0.3335 | 0.1631 | 0.0138 | 0.0548 | 0.0535 | 0.8598 | 35808 |
| w0.01/ws1.0 s200 seed42 val32 | 32 | 0.1415 | 0.3760 | 0.1843 | 0.0119 | 0.0508 | 0.0489 | 0.7240 | 43707 |
| anti-overfill s200 val32 | 32 | 0.1183 | 0.3215 | 0.1572 | 0.0185 | 0.0553 | 0.0702 | 0.7715 | 39183 |

### Paired delta

`anti003 smoke - w0.005/ws0.5 smoke`：

| metric | delta | better count |
|---|---:|---:|
| sparse IoU | +0.0005 | 10/16 |
| precision | +0.0011 | 10/16 |
| recall | +0.0005 | 10/16 |
| Chamfer | -0.0001 | 12/16 |
| target->mesh | -0.0014 | 12/16 |
| mesh->target | -0.0007 | 10/16 |
| extent ratio | -0.0103 | 11/16 |

`w0.005/ws0.5 step100 - w0.005/ws0.5 last`：

| metric | delta | better count |
|---|---:|---:|
| sparse IoU | -0.0240 | 1/32 |
| precision | -0.0592 | 1/32 |
| recall | -0.0290 | 1/32 |
| Chamfer | +0.0060 | 7/32 |
| target->mesh | +0.0057 | 8/32 |
| mesh->target | +0.0184 | 7/32 |
| extent ratio | +0.0789 | 5/32 |

### 本轮结论

1. `w0.005/ws0.5 step100` 不是早停解。它在 sparse、Chamfer、target->mesh、mesh->target、extent 上几乎全面差于 last。
2. `anti-overfill=0.03` 的 smoke mesh 略好于 `0.02` smoke，但提升非常小，属于弱信号。
3. `anti-overfill=0.03 s200` 的 sparse 结果明显变差，低于 `0.02 + w0.005/ws0.5`、低于 `0.02 + w0.01/ws1.0`，也低于纯 anti-overfill s200。
4. 当前主要问题仍不是“训练步数不够”，而是长训后 sparse calibration / outside suppression 不稳定。
5. 目前不建议把 `anti-overfill=0.03` 作为主线继续长训；也不建议继续盲目提高 anti-overfill。

### 下一步建议

第一优先级：固定当前较稳候选，补 `val64`，不要再只看 val32。

当前更值得扩大的候选仍是：

```text
ANTI_OVERFILL_LOSS_WEIGHT=0.02
RANKING_LOSS_WEIGHT=0.005
RANKING_WRONG_SUPPORT_WEIGHT=0.5
TRAIN_SEED=123
```

原因是它比 `w0.01/ws1.0 seed123` 稳，比纯 anti-overfill 的 mesh->target 明显好，虽然 extent 仍偏大。

第二优先级：加 eval-time extent/rerank，而不是继续改 loss。

当前结果已经说明 sparse/mesh 质量不是单调由 loss 权重决定。进入真实 AR 或 ReconViaGen/TRELLIS pipeline 前，应该对多个候选做 rerank：

```text
1. mesh 与 AR/SLAM point cloud 的双向距离；
2. 多视角 render mask IoU；
3. mesh extent / bbox sanity；
4. visible support coverage；
5. 若有 CoarseModel 位姿优化，则加入最终 mask alignment score。
```

第三优先级：如果仍要改训练，优先做 schedule，而不是加大权重：

```text
前 50-100 step: ranking 正常；
后 100 step: 降低 ranking 或冻结 ranking，只保留 anti-overfill；
或者对 outside logits 加 target-count calibration loss。
```

这比继续把 `ANTI_OVERFILL_LOSS_WEIGHT` 从 `0.03` 往上加更合理，因为 `0.03 s200` 已经出现 sparse 退化。

## 三十一、w0.005/ws0.5 seed123 val64 mesh 结果

### 输入结果路径

```text
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/antioverfill_rank_w0005_ws05_s200_seed123_val64_mesh_r05/report.json
```

对比参考：

```text
w0.01/ws1.0 seed42 val64:
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/antioverfill_rank_w001_support_s200_val64_mesh_r05/report.json

w0.005/ws0.5 seed123 val32:
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/antioverfill_rank_w0005_ws05_s200_seed123_val32_mesh_r05/report.json
```

### 汇总结果

| run | mode | count | sparse IoU | precision | recall | Chamfer | target->mesh | mesh->target | extent | vertices |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w0.005/ws0.5 seed123 val64 | stage2 | 64 | 0.1301 | 0.3474 | 0.1707 | 0.0125 | 0.0514 | 0.0500 | 0.8297 | 38767 |
| w0.005/ws0.5 seed123 val64 | stock | 64 | 0.0460 | 0.1082 | 0.0779 | 0.0455 | 0.1338 | 0.0977 | 0.3796 | 132248 |
| w0.005/ws0.5 seed123 val64 | target sparse | 64 | 1.0000 | 1.0000 | 1.0000 | 0.0005 | 0.0129 | 0.0105 | 0.4900 | 222481 |
| w0.01/ws1.0 seed42 val64 | stage2 | 64 | 0.1413 | 0.3726 | 0.1833 | 0.0131 | 0.0500 | 0.0506 | 0.7298 | 42728 |

### 与 stock sparse 的 paired delta

`w0.005/ws0.5 seed123 val64 - stock sparse`：

| metric | delta | better count |
|---|---:|---:|
| sparse IoU | +0.0841 | 61/64 |
| precision | +0.2392 | 63/64 |
| recall | +0.0928 | 59/64 |
| Chamfer | -0.0330 | 58/64 |
| target->mesh | -0.0825 | 61/64 |
| mesh->target | -0.0477 | 48/64 |

这说明 point-prior Stage2 对 frozen downstream mesh 是有真实收益的，不是只改善 sparse 指标。

但 stock sparse 的 `extent_ratio=0.3796` 更小并不表示 stock 更好。stock 这里明显偏扁/欠重建，低 extent 不能作为单独优点看。

### 与当前 seed42 best 的 paired delta

`w0.005/ws0.5 seed123 val64 - w0.01/ws1.0 seed42 val64`：

| metric | delta | better count |
|---|---:|---:|
| sparse IoU | -0.0111 | 14/64 |
| precision | -0.0252 | 14/64 |
| recall | -0.0125 | 14/64 |
| Chamfer | -0.0006 | 39/64 |
| target->mesh | +0.0013 | 32/64 |
| mesh->target | -0.0005 | 34/64 |
| extent ratio | +0.0999 | 14/64 |

解释：

```text
1. w0.005/ws0.5 seed123 的 mesh Chamfer 与 seed42 best 接近，甚至略低；
2. 但 sparse IoU/precision/recall 明显落后；
3. extent 明显更大，说明外扩问题仍然存在；
4. 它不是新的最好 checkpoint，只能说明弱 ranking 更稳于 seed123，但还没有超过 seed42 best。
```

### 分布与 worst cases

`w0.005/ws0.5 seed123 val64`：

| metric | mean | median | q75 | q90 |
|---|---:|---:|---:|---:|
| sparse IoU | 0.1301 | 0.1260 | 0.1435 | 0.1617 |
| precision | 0.3474 | 0.3425 | 0.3785 | 0.4176 |
| recall | 0.1707 | 0.1678 | 0.1882 | 0.2088 |
| Chamfer | 0.0125 | 0.0103 | 0.0171 | 0.0240 |
| target->mesh | 0.0514 | 0.0470 | 0.0608 | 0.0794 |
| mesh->target | 0.0500 | 0.0414 | 0.0696 | 0.0983 |
| extent | 0.8297 | 0.8294 | 0.8984 | 0.9448 |

最差 Chamfer 样本：

```text
idx26  chamfer=0.0363  target->mesh=0.0848  mesh->target=0.0984  extent=0.9956
idx61  chamfer=0.0321  target->mesh=0.0862  mesh->target=0.1125  extent=0.9684
idx58  chamfer=0.0287  target->mesh=0.0933  mesh->target=0.0983  extent=0.9543
idx28  chamfer=0.0271  target->mesh=0.0449  mesh->target=0.0985  extent=0.9171
idx52  chamfer=0.0269  target->mesh=0.0689  mesh->target=0.1005  extent=0.6240
idx18  chamfer=0.0263  target->mesh=0.0694  mesh->target=0.1029  extent=0.9448
```

最差样本仍然集中在外扩和 mesh->target 变差：`idx26/61/58/18` 的 extent 都接近 1，说明失效模式没有消失。

### 本轮结论

1. `w0.005/ws0.5 seed123` 在 val64 上通过了“相对 stock 是否有效”的验证：Chamfer 58/64 胜、target->mesh 61/64 胜、mesh->target 48/64 胜。
2. 但它没有超过当前 seed42 best：sparse 只赢 14/64，extent 只赢 14/64。
3. 这说明 point-prior Stage2 方向是正确的，但当前训练配置还没有稳定到可以直接 full-scale。
4. 目前最可信的判断是：`w0.005/ws0.5` 是更温和的候选，但它需要 seed42 复验；否则无法判断它的弱排名权重是否本身更稳，还是只是 seed123 下的折中。

### 下一步建议

第一优先级：补 `w0.005/ws0.5 seed42 s200`。

原因：

```text
已有 w0.01/ws1.0 seed42 很强；
已有 w0.005/ws0.5 seed123 比 w0.01/ws1.0 seed123 稳；
但缺少 w0.005/ws0.5 seed42。
```

只有补齐这个格子，才能判断：

```text
弱 ranking 是否真的优于强 ranking；
还是 seed42 偶然让强 ranking 表现更好。
```

建议配置：

```text
ANTI_OVERFILL_LOSS_WEIGHT=0.02
RANKING_LOSS_WEIGHT=0.005
RANKING_WRONG_SUPPORT_WEIGHT=0.5
TRAIN_SEED=42
MAX_STEPS=200
```

第二优先级：同时准备 candidate rerank。

从 val64 看，单个 sparse checkpoint 无法稳定避免外扩；但不同配置的失败样本不完全相同。因此真实 AR pipeline 更合理的落地方式不是单样本一次生成，而是：

```text
1. 生成 2-4 个 sparse/mesh 候选；
2. 用 AR/SLAM sparse points、render mask IoU、bbox/extent sanity 做 rerank；
3. 再把胜出的 mesh 交给 CoarseModel 位姿优化。
```

第三优先级：暂时不建议完整训练。

完整训练前至少需要：

```text
1. w0.005/ws0.5 seed42 s200 val64；
2. 当前 best 与弱 ranking 的 val64/val128 对照；
3. candidate rerank 能否稳定筛掉 idx26/61/58 这类外扩样本。
```

## 三十二、w0.005/ws0.5 seed42 s200 val64 结果

### 输入结果路径

```text
sparse eval:
/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_rank_w0005_ws05_s200_seed42/eval/report.json

mesh eval:
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/antioverfill_rank_w0005_ws05_s200_seed42_val64_mesh_r05/report.json

checkpoint:
/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_rank_w0005_ws05_s200_seed42/checkpoints/last.ckpt
```

训练目录中 checkpoint：

```text
last.ckpt
ss-pointprior-stage2-epoch=00-step=100.ckpt
ss-pointprior-stage2-epoch=00-step=200.ckpt
```

### Sparse eval

| run | count | IoU | precision | recall | known IoU | unknown IoU | obs unknown recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| w0.005/ws0.5 seed42 | 96 | 0.1929 | 0.3631 | 0.3232 | 0.3314 | 0.0207 | 0.0644 |
| w0.005/ws0.5 seed123 | 96 | 0.1509 | 0.2972 | 0.2628 | 0.2735 | 0.0192 | 0.0652 |
| w0.01/ws1.0 seed42 | 96 | 0.1633 | 0.3148 | 0.2821 | 0.2918 | 0.0197 | 0.0672 |
| w0.01/ws1.0 seed123 | 96 | 0.1461 | 0.2881 | 0.2556 | 0.2651 | 0.0191 | 0.0659 |
| anti-overfill s200 | 96 | 0.1463 | 0.2864 | 0.2566 | 0.2699 | 0.0179 | 0.0626 |

这组 sparse 指标非常关键：`w0.005/ws0.5 seed42` 同时超过了同 seed 的强 ranking 和 seed123 的弱 ranking，说明之前 `w0.005/ws0.5 seed123` 表现一般主要是 seed 敏感，而不是弱 ranking 配置本身无效。

### Mesh eval

| run | mode | count | sparse IoU | precision | recall | Chamfer | target->mesh | mesh->target | extent | vertices |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w0.005/ws0.5 seed42 val64 | stage2 | 64 | 0.1707 | 0.4373 | 0.2155 | 0.0097 | 0.0484 | 0.0374 | 0.6876 | 45194 |
| w0.01/ws1.0 seed42 val64 | stage2 | 64 | 0.1413 | 0.3726 | 0.1833 | 0.0131 | 0.0500 | 0.0506 | 0.7298 | 42728 |
| w0.005/ws0.5 seed123 val64 | stage2 | 64 | 0.1301 | 0.3474 | 0.1707 | 0.0125 | 0.0514 | 0.0500 | 0.8297 | 38767 |
| stock sparse val64 | stock | 64 | 0.0460 | 0.1082 | 0.0779 | 0.0455 | 0.1338 | 0.0977 | 0.3796 | 132245 |
| target sparse val64 | target | 64 | 1.0000 | 1.0000 | 1.0000 | 0.0005 | 0.0129 | 0.0105 | 0.4900 | 222488 |

### Paired delta

`w0.005/ws0.5 seed42 - w0.01/ws1.0 seed42`：

| metric | delta | better count |
|---|---:|---:|
| sparse IoU | +0.0294 | 63/64 |
| precision | +0.0647 | 63/64 |
| recall | +0.0322 | 63/64 |
| Chamfer | -0.0034 | 51/64 |
| target->mesh | -0.0016 | 43/64 |
| mesh->target | -0.0132 | 56/64 |
| extent ratio | -0.0422 | 41/64 |

`w0.005/ws0.5 seed42 - w0.005/ws0.5 seed123`：

| metric | delta | better count |
|---|---:|---:|
| sparse IoU | +0.0406 | 63/64 |
| precision | +0.0899 | 63/64 |
| recall | +0.0448 | 63/64 |
| Chamfer | -0.0027 | 50/64 |
| target->mesh | -0.0029 | 44/64 |
| mesh->target | -0.0126 | 55/64 |
| extent ratio | -0.1421 | 52/64 |

`w0.005/ws0.5 seed42 - stock sparse`：

| metric | delta | better count |
|---|---:|---:|
| sparse IoU | +0.1246 | 64/64 |
| precision | +0.3291 | 63/64 |
| recall | +0.1376 | 62/64 |
| Chamfer | -0.0358 | 58/64 |
| target->mesh | -0.0854 | 61/64 |
| mesh->target | -0.0603 | 54/64 |

### 分布

`w0.005/ws0.5 seed42 val64`：

| metric | mean | median | q75 | q90 |
|---|---:|---:|---:|---:|
| sparse IoU | 0.1707 | 0.1546 | 0.1846 | 0.2389 |
| precision | 0.4373 | 0.4085 | 0.4675 | 0.5786 |
| recall | 0.2155 | 0.2008 | 0.2337 | 0.2892 |
| Chamfer | 0.0097 | 0.0067 | 0.0111 | 0.0198 |
| target->mesh | 0.0484 | 0.0430 | 0.0543 | 0.0766 |
| mesh->target | 0.0374 | 0.0280 | 0.0526 | 0.0737 |
| extent | 0.6876 | 0.7190 | 0.8543 | 0.9025 |

对比旧 best `w0.01/ws1.0 seed42`：

```text
Chamfer:      0.0131 -> 0.0097
mesh->target: 0.0506 -> 0.0374
extent:       0.7298 -> 0.6876
sparse IoU:   0.1413 -> 0.1707
```

这是目前最强的一组 Stage2 sparse checkpoint。

### Worst cases

最差 Chamfer：

```text
idx58  chamfer=0.0530  target->mesh=0.1587  mesh->target=0.1037  extent=0.5702
idx53  chamfer=0.0322  target->mesh=0.1010  mesh->target=0.1014  extent=0.7846
idx18  chamfer=0.0296  target->mesh=0.0764  mesh->target=0.1152  extent=0.4778
idx52  chamfer=0.0254  target->mesh=0.0763  mesh->target=0.0854  extent=0.6017
idx61  chamfer=0.0206  target->mesh=0.0793  mesh->target=0.0737  extent=0.9054
```

这次的失败模式已经不是单纯外扩。`idx58/53/18/52` 更像局部欠覆盖或形状偏移，extent 不一定大；`idx61/13/30` 仍有外扩倾向。

对旧 worst cases 的改善：

```text
idx26: seed123 chamfer=0.0363 extent=0.9956 -> seed42 chamfer=0.0078 extent=0.6224
idx61: seed123 chamfer=0.0321 extent=0.9684 -> seed42 chamfer=0.0206 extent=0.9054
idx28: seed123 chamfer=0.0271 extent=0.9171 -> seed42 chamfer=0.0055 extent=0.5245
```

`idx58/18/52` 仍然是难例，需要后续用候选 rerank 或更多 top-k 策略处理。

### 本轮结论

1. `w0.005/ws0.5 seed42` 是当前新的 best Stage2 sparse checkpoint。
2. 弱 ranking 配置不是退化方向；在 `seed42` 下，它同时优于强 ranking、seed123 弱 ranking 和纯 anti-overfill。
3. 这组结果把“继续做 point-prior Stage2”从实验探索推进到了可进入扩大验证的状态。
4. 但仍不建议直接 full-scale 训练，因为当前只验证了 `val64 + r0.50_cap8192`，还缺少 top-k sensitivity 和更大验证集。

### 下一步建议

第一优先级：对当前 best 做 top-k sensitivity。

建议跑：

```text
r0.35_cap4096
r0.50_cap8192
r0.75_cap12000
target_unique
```

目标是判断当前 best 的收益是否依赖单一 `r0.50_cap8192`。如果 `r0.50` 和 `r0.75` 都稳定，说明 sparse logits 排序质量真实改善；如果只有 `r0.50` 好，则需要后处理固定 top-k 或 rerank。

第二优先级：补 `val128` 主候选。

如果 val128 仍保持：

```text
Chamfer < 0.011
mesh->target < 0.045
extent < 0.75
sparse IoU > 0.16
```

可以把这组作为后续 AR pipeline 的默认 sparse checkpoint。

第三优先级：开始做 candidate rerank。

当前 best 已经足够强，下一步不是继续微调 loss，而是把 2-4 个候选 mesh 用：

```text
AR/SLAM point distance
multi-view render mask IoU
extent sanity
visible support
```

做自动选择。这样可以专门处理 `idx58/53/18/52` 这类局部失败样本。

## 三十三、w0.005/ws0.5 seed42 s200 val128 结果

### 输入结果路径

```text
val128 manifest:
/home/zjr/Tracker/trellis_point_prior_mv/outputs/pointprior_pixal_v9_stage2_antioverfill_rank_w0005_ws05_s200_seed42/data/val128/manifest.json

mesh eval:
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/antioverfill_rank_w0005_ws05_s200_seed42_val128_mesh_r05/report.json
```

`val128` manifest 样本数确认：

```text
samples=128
prior_point_mean=1498.46
target_count_mean=8159.47
```

### 汇总结果

| run | mode | count | sparse IoU | precision | recall | Chamfer | target->mesh | mesh->target | extent | vertices |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w0.005/ws0.5 seed42 val128 | stage2 | 128 | 0.1719 | 0.4393 | 0.2170 | 0.0090 | 0.0471 | 0.0346 | 0.7015 | 45733 |
| w0.005/ws0.5 seed42 val128 | stock | 128 | 0.0438 | 0.1018 | 0.0763 | 0.0467 | 0.1320 | 0.1022 | 0.3876 | 138536 |
| w0.005/ws0.5 seed42 val128 | target sparse | 128 | 1.0000 | 1.0000 | 1.0000 | 0.0004 | 0.0126 | 0.0104 | 0.4794 | 223157 |
| w0.005/ws0.5 seed42 val64 | stage2 | 64 | 0.1707 | 0.4373 | 0.2155 | 0.0097 | 0.0484 | 0.0374 | 0.6876 | 45194 |

此前设置的 val128 通过线：

```text
Chamfer < 0.011
mesh->target < 0.045
extent < 0.75
sparse IoU > 0.16
```

本次结果全部通过：

```text
Chamfer      0.0090
mesh->target 0.0346
extent       0.7015
sparse IoU   0.1719
```

### 前后 64 分桶

| subset | sparse IoU | precision | recall | Chamfer | target->mesh | mesh->target | extent |
|---|---:|---:|---:|---:|---:|---:|---:|
| all 0-127 | 0.1719 | 0.4393 | 0.2170 | 0.0090 | 0.0471 | 0.0346 | 0.7015 |
| first 0-63 | 0.1707 | 0.4373 | 0.2155 | 0.0097 | 0.0485 | 0.0375 | 0.6876 |
| second 64-127 | 0.1732 | 0.4412 | 0.2184 | 0.0082 | 0.0457 | 0.0317 | 0.7155 |

这点很重要：新增的 `64-127` 没有拖垮结果，反而在 Chamfer / mesh->target 上略好。因此 val128 不是只靠前 64 个样本撑起来的。

### 分布

| metric | mean | median | q75 | q90 |
|---|---:|---:|---:|---:|
| sparse IoU | 0.1719 | 0.1556 | 0.1902 | 0.2420 |
| precision | 0.4393 | 0.4095 | 0.4794 | 0.5846 |
| recall | 0.2170 | 0.2020 | 0.2398 | 0.2923 |
| Chamfer | 0.0090 | 0.0063 | 0.0109 | 0.0176 |
| target->mesh | 0.0471 | 0.0436 | 0.0532 | 0.0713 |
| mesh->target | 0.0346 | 0.0269 | 0.0444 | 0.0633 |
| extent | 0.7015 | 0.7217 | 0.8504 | 0.9062 |

尾部统计：

```text
Chamfer > 0.02:      8/128
Chamfer > 0.03:      4/128
Chamfer > 0.04:      3/128
Chamfer > 0.05:      1/128
mesh->target > 0.08: 7/128
mesh->target > 0.10: 3/128
extent > 0.90:       20/128
extent > 0.95:       3/128
sparse IoU < 0.12:   12/128
```

### Worst cases

最差 Chamfer：

```text
idx058  chamfer=0.0543  target->mesh=0.1595  mesh->target=0.1067  extent=0.5702
idx093  chamfer=0.0484  target->mesh=0.0600  mesh->target=0.1235  extent=0.8163
idx120  chamfer=0.0430  target->mesh=0.0896  mesh->target=0.0962  extent=0.6886
idx053  chamfer=0.0314  target->mesh=0.1000  mesh->target=0.0988  extent=0.7846
idx018  chamfer=0.0293  target->mesh=0.0747  mesh->target=0.1158  extent=0.4778
```

最差 mesh->target：

```text
idx093  mesh->target=0.1235  chamfer=0.0484  sparse IoU=0.1114
idx018  mesh->target=0.1158  chamfer=0.0293  sparse IoU=0.1352
idx058  mesh->target=0.1067  chamfer=0.0543  sparse IoU=0.1223
idx053  mesh->target=0.0988  chamfer=0.0314  sparse IoU=0.1577
idx120  mesh->target=0.0962  chamfer=0.0430  sparse IoU=0.1069
```

最差 extent：

```text
idx112  extent=0.9815  chamfer=0.0043  mesh->target=0.0316
idx051  extent=0.9598  chamfer=0.0026  mesh->target=0.0217
idx099  extent=0.9572  chamfer=0.0047  mesh->target=0.0237
idx121  extent=0.9500  chamfer=0.0100  mesh->target=0.0234
```

注意：extent 高的样本不一定 mesh 差。现在主要风险已经从“全局外扩导致 mesh 失控”转为少数样本的局部偏移、欠覆盖或 target->mesh 变差。

### 本轮结论

1. `w0.005/ws0.5 seed42 s200` 通过了 val128 主验证。
2. 它可以作为当前 AR pipeline / frozen downstream mesh eval 的默认 sparse checkpoint 候选。
3. 相比之前 pixal3d / ar_pose_trellis 的多视图 condition 结果，这条 point-prior sparse inpainting 路线目前是最有希望的模型改进方向。
4. 仍不能说完整系统已经可用，因为还没有测 top-k sensitivity，也没有接真实 AR/SLAM prior 和 candidate rerank。

### 下一步建议

第一优先级：补当前 best 的 top-k sweep。

必须确认 `r0.50_cap8192` 不是唯一好点。建议继续跑：

```text
r0.35_cap4096
r0.50_cap8192
r0.75_cap12000
target_unique
```

如果 `r0.50` 和 `r0.75` 都稳定，就可以认为 sparse logits ranking 真实变好；如果只有 `r0.50` 好，就需要在 AR pipeline 中固定 top-k 或使用候选 rerank。

第二优先级：开始实现 candidate rerank。

现在模型端已经有足够强的候选，下一步应该针对 worst cases 做自动选择：

```text
1. 生成多个 sparse/mesh 候选；
2. 用 AR/SLAM point-to-mesh distance；
3. 用多视角 render mask IoU；
4. 用 extent / bbox sanity；
5. 选择最可信 mesh 进入 CoarseModel。
```

第三优先级：暂时不建议再调 loss。

当前 loss 组合已经给出稳定收益：

```text
ANTI_OVERFILL_LOSS_WEIGHT=0.02
RANKING_LOSS_WEIGHT=0.005
RANKING_WRONG_SUPPORT_WEIGHT=0.5
TRAIN_SEED=42
```

继续微调 loss 的边际收益大概率低于 top-k sweep 和 candidate rerank。

## 三十四、当前 best top-k sweep 结果

### 输入结果路径

```text
val64 top-k:
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/antioverfill_rank_w0005_ws05_s200_seed42_val64_mesh_topk/report.json

val128 top-k:
/home/zjr/Tracker/trellis_point_prior_mv/outputs/mesh_frozen_downstream/antioverfill_rank_w0005_ws05_s200_seed42_val128_mesh_topk/report.json
```

本轮测试的 top-k：

```text
r0.35_cap4096
r0.50_cap8192
r0.75_cap12000
target_unique
```

### Val128 汇总

| top-k | sparse IoU | precision | recall | Chamfer | target->mesh | mesh->target | extent | vertices |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r0.35_cap4096 | 0.1371 | 0.4787 | 0.1596 | 0.0095 | 0.0555 | 0.0320 | 0.6893 | 24804 |
| r0.50_cap8192 | 0.1719 | 0.4393 | 0.2170 | 0.0089 | 0.0468 | 0.0346 | 0.7015 | 45728 |
| r0.75_cap12000 | 0.2039 | 0.3943 | 0.2917 | 0.0082 | 0.0376 | 0.0367 | 0.7178 | 88462 |
| target_unique | 0.2204 | 0.3580 | 0.3580 | 0.0088 | 0.0330 | 0.0419 | 0.7253 | 152699 |
| stock sparse | 0.0438 | 0.1018 | 0.0763 | 0.0467 | 0.1320 | 0.1022 | 0.3877 | 138532 |
| target sparse | 1.0000 | 1.0000 | 1.0000 | 0.0004 | 0.0126 | 0.0104 | 0.4794 | 223157 |

### Val64/Val128 一致性

| top-k | val64 Chamfer | val128 Chamfer | val64 target->mesh | val128 target->mesh | val64 mesh->target | val128 mesh->target |
|---|---:|---:|---:|---:|---:|---:|
| r0.35 | 0.0099 | 0.0095 | 0.0564 | 0.0555 | 0.0334 | 0.0320 |
| r0.50 | 0.0097 | 0.0089 | 0.0483 | 0.0468 | 0.0374 | 0.0346 |
| r0.75 | 0.0083 | 0.0082 | 0.0385 | 0.0376 | 0.0376 | 0.0367 |
| target_unique | 0.0087 | 0.0088 | 0.0336 | 0.0330 | 0.0426 | 0.0419 |

结果很稳定，val128 没有推翻 val64。

### Paired delta

`r0.75_cap12000 - r0.50_cap8192`：

| set | metric | delta | better count |
|---|---|---:|---:|
| val64 | sparse IoU | +0.0335 | 64/64 |
| val64 | recall | +0.0762 | 64/64 |
| val64 | Chamfer | -0.0014 | 35/64 |
| val64 | target->mesh | -0.0098 | 61/64 |
| val64 | mesh->target | +0.0003 | 20/64 |
| val64 | extent | +0.0120 | 27/64 |
| val128 | sparse IoU | +0.0319 | 128/128 |
| val128 | recall | +0.0748 | 128/128 |
| val128 | Chamfer | -0.0007 | 70/128 |
| val128 | target->mesh | -0.0092 | 122/128 |
| val128 | mesh->target | +0.0022 | 41/128 |
| val128 | extent | +0.0163 | 46/128 |

解释：

```text
r0.75 的优势很明确：更高 recall、更高 sparse IoU、更低 target->mesh、略低 Chamfer。
代价是 precision 降低、顶点数约翻倍、mesh->target 略差、extent 略大。
```

`target_unique - r0.75_cap12000`：

| set | metric | delta | better count |
|---|---|---:|---:|
| val128 | sparse IoU | +0.0166 | 121/128 |
| val128 | recall | +0.0663 | 128/128 |
| val128 | Chamfer | +0.0007 | 60/128 |
| val128 | target->mesh | -0.0047 | 116/128 |
| val128 | mesh->target | +0.0052 | 32/128 |
| val128 | extent | +0.0075 | 62/128 |
| val128 | vertices | +64236 | 1/128 |

解释：

```text
target_unique 对 target->mesh 最好，但 mesh->target、顶点数和尾部风险更差。
它不适合作默认推理 top-k，更适合作候选之一参与 rerank。
```

`r0.35_cap4096 - r0.50_cap8192`：

```text
r0.35 precision 更高、mesh->target 略低，但 recall/target->mesh 明显变差。
它适合作低顶点/保守候选，不适合作主默认。
```

### 每样本最佳 top-k 分布

Val128：

| criterion | r0.35 | r0.50 | r0.75 | target_unique |
|---|---:|---:|---:|---:|
| best Chamfer | 27 | 28 | 30 | 43 |
| best target->mesh | 0 | 0 | 12 | 116 |
| best mesh->target | 65 | 34 | 18 | 11 |
| balanced rank-sum | 28 | 41 | 31 | 28 |

这说明不存在单一 top-k 支配所有指标。`r0.75` 是最好的单默认折中；但候选 rerank 仍然有价值，因为不同样本的最佳 top-k 不同。

### Tail 风险

Val128 尾部：

| top-k | Chamfer > 0.03 | mesh->target > 0.10 | extent > 0.95 |
|---|---:|---:|---:|
| r0.50 | 4/128 | 4/128 | 4/128 |
| r0.75 | 2/128 | 1/128 | 6/128 |
| target_unique | 4/128 | 4/128 | 6/128 |

`r0.75` 的 Chamfer 和 mesh->target 尾部最好，但 extent 高尾部略多。这个 tradeoff 可以接受，因为很多高 extent 样本并不对应差 Chamfer。

### 难例对比

| idx | r0.50 Chamfer | r0.75 Chamfer | target_unique Chamfer | 结论 |
|---:|---:|---:|---:|---|
| 58 | 0.0508 | 0.0209 | 0.0171 | 更大 top-k 明显修复欠覆盖 |
| 93 | 0.0482 | 0.0549 | 0.0660 | 更大 top-k 反而变差 |
| 120 | 0.0425 | 0.0463 | 0.0562 | 更大 top-k 变差 |
| 53 | 0.0321 | 0.0249 | 0.0225 | 更大 top-k 改善 |
| 18 | 0.0295 | 0.0179 | 0.0338 | r0.75 最好，target_unique 过填充 |
| 52 | 0.0253 | 0.0207 | 0.0263 | r0.75 最好 |

这进一步说明：固定 `target_unique` 不安全；`r0.75` 更适合作主默认，但 `r0.50/r0.75/target_unique` 的候选选择仍然能处理个体差异。

### 本轮结论

1. 当前 checkpoint 的收益不依赖单一 `r0.50_cap8192`，top-k sweep 通过。
2. `r0.75_cap12000` 是当前最合理的默认 mesh top-k：它在 val64/val128 上都给出最低 Chamfer 和明显更好的 target coverage。
3. `r0.50_cap8192` 是保守低顶点候选：mesh->target 稍好、顶点更少，但欠覆盖更明显。
4. `target_unique` 不应作为默认：target->mesh 最好，但 mesh->target、顶点数和部分难例尾部更差。
5. 模型侧目前已经足够进入系统集成验证，继续微调 loss 的优先级低于 candidate rerank 和真实 AR/SLAM prior 测试。

### 下一步建议

第一优先级：把默认 mesh eval / AR pipeline sparse top-k 改为：

```text
default: r0.75_cap12000
fallback/fast: r0.50_cap8192
candidate set: r0.50_cap8192, r0.75_cap12000, target_unique
```

第二优先级：实现 candidate rerank。

每个输入生成 2-3 个候选 mesh：

```text
r0.50_cap8192
r0.75_cap12000
target_unique
```

然后用：

```text
1. AR/SLAM point-to-mesh distance
2. 多视角 render mask IoU
3. extent / bbox sanity
4. visible support coverage
```

选择最终 mesh。这样能处理 `idx58` 这类更大 top-k 修复样本，也能避免 `idx93/120` 这类更大 top-k 变差样本。

第三优先级：真实 AR/SLAM prior 测试。

当前评测用的是 synthetic clean/oracle-like prior。进入系统前必须测试：

```text
真实 AR sparse point prior
真实 mask
真实 SLAM/AR pose
```

否则不能保证合成 prior 上的收益完全迁移到手机采集流程。

## 2026-06-23 真实 SLAM-like prior 初步测试

### 本轮运行状态

用户运行了两组真实数据测试：

1. `real_slam_prior_four_triangulated_mesh_eval`：计划对四个真实/半真实数据集做 `stock_sparse, prior_sparse, stage2_correct` mesh 对比。
2. `real_slam_prior_triangulated_four_strictmask_countcheck`：只做 strict mask 特征点三角化点数检查。

当前在默认输出根目录：

```text
/home/zjr/Tracker/trellis_point_prior_mv/outputs/real_slam_prior
```

没有找到 `real_slam_prior_four_triangulated_mesh_eval` 目录，也没有对应 `mesh_eval/report.json`。因此目前不能把“四数据集 mesh eval”当作已完成结果来下结论；能分析的是 strict-mask countcheck，以及前面已有的 `GOOD_MESH_TEST` 单样本 mesh smoke。

### strict mask countcheck 结果

结果文件：

```text
trellis_point_prior_mv/outputs/real_slam_prior/real_slam_prior_triangulated_four_strictmask_countcheck/slam_like_points_report.json
```

| dataset | raw triangulated | support filtered | final points | support mean | support median |
|---|---:|---:|---:|---:|---:|
| GOOD_MESH_TEST | 836 | 835 | 764 | 13.25 | 14 |
| reconviagen_20260520_021556 | 52 | 52 | 49 | 4.79 | 5 |
| reconviagen_20260617_073549 | 16 | 16 | 16 | 6.75 | 6 |
| reconviagen_20260617_075506 | 109 | 109 | 101 | 3.83 | 3 |

对比 relaxed countcheck：

```text
trellis_point_prior_mv/outputs/real_slam_prior/real_slam_prior_triangulated_four_countcheck_relaxed/slam_like_points_report.json
```

| dataset | relaxed final | strict-mask final | 判断 |
|---|---:|---:|---|
| GOOD_MESH_TEST | 865 | 764 | strict 点数略少，但仍很充足，且 support 更干净 |
| reconviagen_20260520_021556 | 71 | 49 | strict 接近 `MIN_PRIOR_POINTS=40` 下限，可用但较弱 |
| reconviagen_20260617_073549 | 50 | 16 | strict 明显不足，说明该序列 mask 内可三角化纹理很弱 |
| reconviagen_20260617_075506 | 94 | 101 | strict 不差，说明该序列 mask 内特征质量可以 |

结论：strict-mask 不是不可行；它在 `GOOD_MESH_TEST` 和 `075506` 上甚至更像干净物体点云。但真实数据的点数差异非常大，`073549` 这种样本只靠 mask 内特征点会过稀疏。后续系统里不能只用一个策略，应保留：

```text
strict-mask 作为高置信默认；
relaxed 全图匹配 + mask/pose 后筛作为弱纹理 fallback。
```

### GOOD_MESH_TEST 单样本 mesh smoke

已有结果：

```text
trellis_point_prior_mv/outputs/real_slam_prior/real_slam_prior_goodmesh_triangulated_mesh_smoke/mesh_eval/report.json
```

该测试只包含 `GOOD_MESH_TEST` 一个样本，`max_frames=8`，真实 COLMAP/SLAM-like prior 经过 `prior_bbox` 归一化后得到 `prior_point_count=116`。

| mode | coord count | vertex count | extent ratio | ref norm Chamfer L2 | 结论 |
|---|---:|---:|---:|---:|---|
| stock_sparse | 23819 | 771765 | 0.9998 | 0.1125 | 原版 sparse 明显过填充，接近整 cube |
| prior_sparse | 116 | 520 | 0.4011 | 0.0759 | 只用 prior 太稀疏，得到的是局部/骨架式 mesh |
| stage2_correct | 12000 | 222624 | 0.9398 | 0.0168 | 明显优于 stock 和 prior_sparse，能把 sparse prior 扩展成较完整结构 |

这一点很关键：真实 SLAM-like 点只有 116 个时，`stage2_correct` 并不是简单复制 prior，而是把点先验扩展成完整得多的 sparse structure，并且参考 mesh Chamfer 明显更低。

### 当前判断

1. 真实 SLAM-like prior 路线仍然值得继续。`GOOD_MESH_TEST` 单样本已经显示：点先验可以把 stock TRELLIS 的过填充问题显著压下去。
2. 不能只看单样本成功。四数据集 mesh report 当前缺失，必须补跑后才能判断是否能泛化到 `reconviagen_2026*` 这些更接近手机采集的数据。
3. strict-mask 与 relaxed 不是二选一。strict-mask 更干净，但在弱纹理样本上会点数不足；relaxed 更容易凑够点，但背景/边缘误点风险更高。
4. `NORMALIZATION_SOURCE=prior_bbox` 是严格真实设定，但会放大局部点云尺度风险。如果 SLAM 点只覆盖物体局部，prior_bbox 会把局部撑满 canonical cube，mesh 可能被误导。

### 下一步建议

第一优先级：补跑真正的四数据集 mesh eval。当前没有找到：

```text
trellis_point_prior_mv/outputs/real_slam_prior/real_slam_prior_four_triangulated_mesh_eval/mesh_eval/report.json
```

建议用新的 run name 重新跑，避免与旧目录混淆，并先把 `MESH_EVAL_SAMPLES` 降到 2000 控制显存和耗时。

第二优先级：四数据集 mesh eval 后，对每个样本检查：

```text
stock_sparse 是否过填充
prior_sparse 是否过稀疏
stage2_correct 是否同时降低 ref_norm_chamfer_l2_mean 和 prior_chamfer_l2_mean
projection_mask_hit_over_inside_mean 是否足够高
prior_point_count 是否低于 40-50 的可靠阈值
```

第三优先级：补 strict-mask mesh eval，但不要强行 `MIN_PRIOR_POINTS=40` 覆盖所有四个样本。因为 `073549` strict-mask 只有 16 点，如果强行 min40 会失败。可分两组：

```text
strict-mask reliable: 排除 073549，MIN_PRIOR_POINTS=40
strict-mask weak-prior smoke: 包含 073549，MIN_PRIOR_POINTS=10/15
```

如果 strict-mask reliable 组也能让 `stage2_correct` 优于 `stock_sparse`，说明真实 AR/SLAM prior 对系统集成有实际价值。

## 2026-06-23 四数据集真实 SLAM-like prior mesh eval

### 运行配置

本轮重新运行：

```text
RUN_NAME=real_slam_prior_four_triangulated_mesh_eval_v2
RUN_TRIANGULATE=1
RUN_BUILD=1
RUN_EVAL=1
PRIOR_SOURCE=colmap_points
SPARSE_SUBDIR=sparse_slam_eval_four_v2/0
NORMALIZATION_SOURCE=prior_bbox
MIN_PRIOR_POINTS=40
MAX_FRAMES=18
TRI_ALLOW_PAIR_OUTSIDE_MASK=1
TRI_MIN_PAIR_MATCHES=8
TRI_MIN_OUTPUT_POINTS=40
MODES=stock_sparse,prior_sparse,stage2_correct
TOPK_SPECS=12000
MESH_EVAL_SAMPLES=2000
```

结果路径：

```text
trellis_point_prior_mv/outputs/real_slam_prior/real_slam_prior_four_triangulated_mesh_eval_v2/mesh_eval/report.json
```

### SLAM-like prior 构建质量

| dataset | final triangulated | prior coords | support mean | mask-hit inside mean | normalization scale |
|---|---:|---:|---:|---:|---:|
| GOOD_MESH_TEST | 865 | 559 | 10.37 | 0.539 | 0.998 |
| reconviagen_20260520_021556 | 71 | 66 | 3.65 | 0.291 | 2.094 |
| reconviagen_20260617_073549 | 50 | 45 | 4.42 | 0.247 | 0.692 |
| reconviagen_20260617_075506 | 94 | 62 | 3.84 | 0.278 | 2.674 |

这说明本轮不是 oracle prior：除 `GOOD_MESH_TEST` 外，三个 `reconviagen_2026*` 数据集的 prior 都很稀疏，mask-hit inside 只有约 `0.25-0.29`，而且 `prior_bbox` normalization scale 差异很大。这更接近真实 SLAM/AR 点云会遇到的问题：点少、局部、尺度不稳定、mask 内命中率不高。

### Mesh 指标对比

| dataset | mode | prior pts | coords | vertices | extent ratio | ref Chamfer | ref A->B | ref B->A |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GOOD_MESH_TEST | stock_sparse | 559 | 23819 | 752897 | 0.999 | 0.1162 | 0.2572 | 0.1779 |
| GOOD_MESH_TEST | prior_sparse | 559 | 559 | 3802 | 0.477 | 0.1121 | 0.1510 | 0.2638 |
| GOOD_MESH_TEST | stage2_correct | 559 | 12000 | 283986 | 0.908 | 0.0163 | 0.0969 | 0.0512 |
| reconviagen_20260520_021556 | stock_sparse | 66 | 4204 | 99326 | 0.827 | 0.0363 | 0.1252 | 0.0968 |
| reconviagen_20260520_021556 | prior_sparse | 66 | 66 | 288 | 0.622 | 0.1588 | 0.1903 | 0.3113 |
| reconviagen_20260520_021556 | stage2_correct | 66 | 12000 | 216034 | 0.926 | 0.0219 | 0.1152 | 0.0488 |
| reconviagen_20260617_073549 | stock_sparse | 45 | 15809 | 299386 | 0.817 | 0.0345 | 0.1099 | 0.1085 |
| reconviagen_20260617_073549 | prior_sparse | 45 | 45 | 452 | 0.449 | 0.0741 | 0.0620 | 0.2374 |
| reconviagen_20260617_073549 | stage2_correct | 45 | 12000 | 233930 | 0.892 | 0.0218 | 0.1096 | 0.0591 |
| reconviagen_20260617_075506 | stock_sparse | 62 | 4802 | 117039 | 0.089 | 0.0751 | 0.0896 | 0.2132 |
| reconviagen_20260617_075506 | prior_sparse | 62 | 62 | 1708 | 0.566 | 0.0703 | 0.1033 | 0.2096 |
| reconviagen_20260617_075506 | stage2_correct | 62 | 12000 | 165470 | 0.952 | 0.0205 | 0.1032 | 0.0582 |

### 结果解读

`stage2_correct` 在 4/4 样本上取得最低 `ref_norm_chamfer_l2_mean`：

| dataset | stock Chamfer | prior Chamfer | stage2 Chamfer | stage2 vs stock |
|---|---:|---:|---:|---:|
| GOOD_MESH_TEST | 0.1162 | 0.1121 | 0.0163 | -0.0999 |
| reconviagen_20260520_021556 | 0.0363 | 0.1588 | 0.0219 | -0.0144 |
| reconviagen_20260617_073549 | 0.0345 | 0.0741 | 0.0218 | -0.0127 |
| reconviagen_20260617_075506 | 0.0751 | 0.0703 | 0.0205 | -0.0546 |

这比单样本 smoke 更有说服力：即使 prior 只有 `45-66` 个 voxel 点，Stage2 仍能在 reference mesh 指标上稳定优于 stock sparse 和直接 prior sparse。

`prior_sparse` 本身不能直接用作 mesh：

```text
prior_sparse 顶点数通常只有几百到几千；
ref B->A 很差，表示 reference surface 大量区域没有被覆盖；
它更像局部点云/骨架，不是完整 mesh。
```

`stage2_correct` 的主要收益是补全 coverage：

```text
ref B->A 明显下降：
GOOD_MESH_TEST: 0.1779 -> 0.0512
20260520:       0.0968 -> 0.0488
073549:         0.1085 -> 0.0591
075506:         0.2132 -> 0.0582
```

这说明 point-prior Stage2 在真实 SLAM-like prior 上不是只贴近输入点，而是把稀疏点扩展成更完整的 sparse structure。

### 风险和限制

1. 这组 reference mesh 来自数据集 `models/*_norm.obj`，其中 `reconviagen_2026*` 本身可能是 ReconViaGen 生成/整理出的 mesh，不等价于真实扫描 GT。因此指标可以比较相对趋势，但不能当绝对几何真值。
2. `TOPK_SPECS=12000` 是固定绝对 top-k。它在这 4 个样本上表现好，但 mesh 顶点仍偏多，后续需要 real prior top-k sweep。
3. prior 的 `mask_hit_inside_mean` 偏低，说明 relaxed 全图匹配会混入不少边界/背景风险。当前结果证明 Stage2 对这种噪声有一定鲁棒性，但不能跳过 strict-mask ablation。
4. `prior_bbox` normalization 仍有尺度风险。尤其 `20260520/075506` scale 超过 2，说明 prior bbox 与 reference canonical 尺度并不稳定。

### 当前结论

这一轮结果支持继续推进真实 AR/SLAM prior 接入。更具体地说：

```text
真实 SLAM-like sparse prior 对 stock TRELLIS sparse/mesh 有可测收益；
Stage2 可以把很稀疏的 object points 扩展成更完整 mesh；
该方向不再只是 PixalV9 synthetic prior 上的现象。
```

但这还不是可以直接上线的最终结论。现在证明的是“方向有效”，下一步要证明“系统可控”：

```text
top-k 是否可稳定选择；
strict/relaxed prior 如何自动切换；
尺度异常 prior 如何拒绝或重归一化；
真实手机 AR session 中是否也能得到类似点数和 mask-hit 质量。
```

### 下一步建议

第一优先级：对同一四数据集做 real prior top-k sweep：

```text
TOPK_SPECS=6000,8192,12000,16000
MODES=stock_sparse,stage2_correct
```

观察：

```text
ref_norm_chamfer_l2_mean
ref_norm_a_to_b_mean
ref_norm_b_to_a_mean
vertex_count
extent_ratio
```

目标是确认 `12000` 是否真是当前真实 prior 的合理默认，而不是偶然偏大。

第二优先级：补 strict-mask reliable mesh eval：

```text
排除 reconviagen_20260617_073549
MIN_PRIOR_POINTS=40
TRI_FEATURE_MASK_MODE=mask
TRI_ALLOW_PAIR_OUTSIDE_MASK=0
```

如果 strict-mask reliable 组也能保持 `stage2_correct > stock_sparse`，说明高置信 object-SLAM prior 本身足够有价值。

第三优先级：开始做真实手机 session 接入 smoke。当前这 4 个数据集仍是离线 COLMAP/固定数据目录测试，和 Unity AR SLAM 返回的在线 sparse map 仍有差异。

## 2026-06-23 四数据集真实 SLAM-like prior top-k sweep

### 运行配置

本轮在同一 manifest 上只重跑 mesh eval：

```text
RUN_NAME=real_slam_prior_four_triangulated_mesh_topk_v1
MANIFEST=real_slam_prior_four_triangulated_mesh_eval_v2/manifest/manifest.json
MODES=stock_sparse,stage2_correct
TOPK_SPECS=6000,8192,12000,16000
MESH_EVAL_SAMPLES=2000
```

结果路径：

```text
trellis_point_prior_mv/outputs/real_slam_prior/real_slam_prior_four_triangulated_mesh_topk_v1/mesh_eval/report.json
```

### 平均指标

| mode | ref Chamfer mean | ref Chamfer median | ref A->B mean | ref B->A mean | prior Chamfer mean | vertices mean | extent ratio mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| stock_sparse | 0.0655 | 0.0552 | 0.1443 | 0.1492 | 0.1018 | 317228 | 0.683 |
| stage2_6000 | 0.0286 | 0.0240 | 0.1150 | 0.0781 | 0.0460 | 64853 | 0.945 |
| stage2_8192 | 0.0274 | 0.0238 | 0.1190 | 0.0653 | 0.0557 | 113146 | 0.958 |
| stage2_12000 | 0.0248 | 0.0235 | 0.1170 | 0.0542 | 0.0616 | 210543 | 0.962 |
| stage2_16000 | 0.0236 | 0.0219 | 0.1146 | 0.0510 | 0.0603 | 310144 | 0.958 |

### 分样本结论

| dataset | stock Chamfer | 6000 | 8192 | 12000 | 16000 | Chamfer 最佳 |
|---|---:|---:|---:|---:|---:|---|
| GOOD_MESH_TEST | 0.1173 | 0.0206 | 0.0198 | 0.0186 | 0.0182 | 16000 |
| reconviagen_20260520_021556 | 0.0343 | 0.0462 | 0.0420 | 0.0340 | 0.0333 | 16000 |
| reconviagen_20260617_073549 | 0.0342 | 0.0202 | 0.0200 | 0.0183 | 0.0172 | 16000 |
| reconviagen_20260617_075506 | 0.0762 | 0.0274 | 0.0277 | 0.0284 | 0.0255 | 16000 |

`16000` 在 4/4 样本上都是 Chamfer 最低，同时 `ref B->A` 也最低，说明更大的 top-k 在真实 SLAM-like prior 上主要改善 coverage。

但是顶点数代价明显：

```text
6000:   约  6.5 万 vertices
8192:   约 11.3 万 vertices
12000:  约 21.1 万 vertices
16000:  约 31.0 万 vertices
```

`16000` 相比 `12000`：

```text
ref Chamfer: 0.0248 -> 0.0236，改善约 0.0013
ref B->A:    0.0542 -> 0.0510，改善约 0.0032
vertices:    21.1 万 -> 31.0 万，增加约 47%
```

因此 `16000` 是当前四样本上最好的 quality 候选，但不一定是系统默认值。

### 重要现象

1. `6000/8192` 在弱 prior 样本上会欠覆盖。  
   `reconviagen_20260520_021556` 中，`6000/8192` 的 Chamfer 反而比 stock 更差，只有 `12000/16000` 才略好于 stock。

2. `12000/16000` 能稳定压低 reference-to-mesh 距离。  
   这说明真实 prior 的 Stage2 不是只贴近 SLAM 点，而是需要足够 top-k 才能把物体补全。

3. `prior Chamfer` 不是唯一选择标准。  
   `6000` 的 prior Chamfer 最低，但 reference coverage 较弱；这符合预期，因为点云 prior 本身是稀疏局部观测，过度贴近 prior 会牺牲完整性。

4. `stock_sparse` 不稳定。  
   `075506` 的 stock extent ratio 只有 `0.0898`，明显几何尺度异常；Stage2 top-k 候选都把 extent 修回 `0.97-0.99` 附近。

### 当前默认建议

当前不建议把 `16000` 直接作为唯一默认。更合理的是：

```text
fast/default: 12000
quality candidate: 16000
low-cost candidate: 8192 或 6000
```

如果只能出一个 mesh 给 CoarseModel：

```text
优先用 12000
```

理由：`12000` 相比 `16000` Chamfer 只差很小，但顶点数少约三分之一，更适合后续 CoarseModel 位姿估计和工程运行。

如果允许候选选择：

```text
候选集用 6000,12000,16000
```

其中：

```text
6000: 低顶点、贴近 prior，用于防止过填充
12000: 当前默认折中
16000: coverage 最强，用于弱 prior/欠覆盖样本
```

### 下一步建议

第一优先级：做 candidate rerank，而不是继续盲目改 Stage2 loss。

候选：

```text
stage2_6000
stage2_12000
stage2_16000
```

rerank 指标：

```text
1. render mask IoU / silhouette consistency
2. AR/SLAM prior point-to-mesh distance
3. mesh extent sanity
4. vertex/face complexity penalty
```

目标是自动区分：

```text
需要 16000 补 coverage 的样本；
6000/12000 已足够、16000 只是增加复杂度的样本。
```

第二优先级：补 strict-mask reliable mesh eval。  
本轮 relaxed prior 已经证明方向有效，但 relaxed 全图匹配可能混入背景/边缘点。下一轮应该排除 strict-mask 只有 16 点的 `073549`，先测 3 个 reliable 样本。

推荐命令：

```bash
cd /home/zjr/Tracker

DATASETS=/home/zjr/Tracker/CoarseModel/datasets/GOOD_MESH_TEST:/home/zjr/Tracker/CoarseModel/datasets/reconviagen_20260520_021556:/home/zjr/Tracker/CoarseModel/datasets/reconviagen_20260617_075506 \
GPU=4 \
RUN_NAME=real_slam_prior_three_strictmask_mesh_eval_v1 \
RUN_TRIANGULATE=1 \
RUN_BUILD=1 \
RUN_EVAL=1 \
PRIOR_SOURCE=colmap_points \
SPARSE_SUBDIR=sparse_slam_strictmask_mesh_v1/0 \
TRI_INPUT_SPARSE_SUBDIR=sparse/0 \
NORMALIZATION_SOURCE=prior_bbox \
MIN_PRIOR_POINTS=40 \
MAX_FRAMES=18 \
TRI_FEATURE_MASK_MODE=mask \
TRI_MIN_PAIR_MATCHES=8 \
TRI_MIN_OUTPUT_POINTS=40 \
TRI_MIN_SUPPORT_VIEWS=2 \
TRI_MIN_SUPPORT_RATIO=0.10 \
MODES=stock_sparse,stage2_correct \
TOPK_SPECS=6000,12000,16000 \
MESH_EVAL_SAMPLES=2000 \
bash trellis_point_prior_mv/scripts/run_real_slam_prior_eval.sh
```

第三优先级：真实手机 AR session smoke。  
离线 COLMAP 数据已经说明方法可行，下一步必须验证 Unity/ARKit/ARCore 返回的在线 sparse point map 是否能达到类似：

```text
prior coords >= 40-60
support median >= 3
projection_any_mask_hit_ratio 接近 1
mask_hit_inside_mean 不低于 0.25-0.30
```

如果真实手机 session 达不到这些阈值，就优先改前端采集/点云筛选；如果能达到，再接 candidate rerank 到系统 pipeline。

---

## 真实内部数据 existing sparse direct/streaming 结果

时间：2026-06-23 07:08:06 UTC

相关输出：

```text
heimei direct countcheck:
/home/zjr/Tracker/trellis_point_prior_mv/outputs/ar_like_streaming_slam/ar_like_colmap_direct_heimei_existing_sparse_countcheck_v1/manifest/build_report.json

heimei streaming countcheck:
/home/zjr/Tracker/trellis_point_prior_mv/outputs/ar_like_streaming_slam/ar_like_streaming_heimei_existing_sparse_countcheck_v1/manifest/build_report.json

internal7 streaming countcheck:
/home/zjr/Tracker/trellis_point_prior_mv/outputs/ar_like_streaming_slam/ar_like_streaming_existing_internal7_countcheck_v1/manifest/build_report.json

internal7 streaming mesh eval:
/home/zjr/Tracker/trellis_point_prior_mv/outputs/ar_like_streaming_slam/ar_like_streaming_existing_internal7_mesh_eval_v1/mesh_eval/report.json
```

### Countcheck 结果

| run | uid | frames | source pts | prior pts | mask-any | hit/inside | support mean | norm scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| heimei_direct | heimei | 32 | 3059 | 970 | 1.000 | 0.692 | 21.52 | 10.153 |
| heimei_stream | heimei | 32 | 322 | 140 | 1.000 | 0.903 | 28.58 | 4.836 |
| internal7_stream | GOOD_MESH_TEST | 18 | 460 | 363 | 1.000 | 0.547 | 9.61 | 0.783 |
| internal7_stream | reconviagen_20260520_021556 | 18 | 35 | 32 | 1.000 | 0.321 | 4.00 | 1.221 |
| internal7_stream | reconviagen_20260617_073549 | 18 | 37 | 33 | 1.000 | 0.251 | 4.15 | 0.547 |
| internal7_stream | reconviagen_20260617_075506 | 18 | 77 | 53 | 0.962 | 0.287 | 3.45 | 2.674 |
| internal7_stream | heimei | 32 | 322 | 140 | 1.000 | 0.903 | 28.58 | 4.836 |
| internal7_stream | maliao | 32 | 92 | 85 | 1.000 | 0.914 | 29.08 | 3.555 |
| internal7_stream | snoopy | 32 | 907 | 334 | 1.000 | 0.777 | 24.12 | 11.169 |

补充现象：

```text
1. heimei direct 使用已有 sparse/0 的 COLMAP points，object prior 很强：3059 source points -> 970 voxel prior。
2. heimei streaming 更稀疏，但点更集中在 mask 内：322 source points -> 140 prior，hit/inside 从 0.692 提升到 0.903。
3. internal7 direct 使用已有 sparse/0 时在 GOOD_MESH_TEST 上失败：supported points = 0 < min_prior_points 20。
   这说明已有 sparse/0 的 points3D 不一定是可用 object prior，不能把 existing sparse direct 当作所有真实数据的默认路线。
4. internal7 streaming 分支 7/7 通过 countcheck，prior 点数从 32 到 363 不等；弱样本集中在 2026 几个 reconviagen 序列。
```

当前解释：

```text
direct COLMAP points:
  点多，覆盖可能更强，但更依赖原始 COLMAP points 是否真的落在物体上。
  heimei 上很好，GOOD_MESH_TEST existing sparse/0 上为 0，说明 direct 需要逐数据集检查。

streaming triangulation:
  点少，但经过 mask/pose support 后更像未来 AR-like object points。
  在 internal7 上鲁棒性更好，至少能为所有样本生成可用 prior。
```

### Mesh Eval 汇总

internal7 streaming mesh eval 使用：

```text
MODES=stock_sparse,stage2_correct
TOPK_SPECS=8192,12000,16000
MESH_EVAL_SAMPLES=2000
```

| mode | count | ref chamfer mean | ref chamfer median | prior chamfer mean | extent ratio mean | vertex mean |
|---|---:|---:|---:|---:|---:|---:|
| stock_sparse | 7 | 0.0609 | 0.0502 | 0.1046 | 0.6358 | 271110 |
| stage2_correct_8192 | 7 | 0.0253 | 0.0277 | 0.0537 | 0.9369 | 111966 |
| stage2_correct_12000 | 7 | 0.0252 | 0.0267 | 0.0580 | 0.9383 | 206115 |
| stage2_correct_16000 | 7 | 0.0246 | 0.0262 | 0.0582 | 0.9405 | 310333 |

逐样本关键结果：

| uid | mode | coord | extent | ref chamfer | prior chamfer | vertices |
|---|---|---:|---:|---:|---:|---:|
| GOOD_MESH_TEST | stock_sparse | 23819 | 0.999 | 0.1143 | 0.1817 | 753348 |
| GOOD_MESH_TEST | stage2_correct_8192 | 8192 | 0.968 | 0.0162 | 0.0227 | 168164 |
| GOOD_MESH_TEST | stage2_correct_12000 | 12000 | 0.948 | 0.0160 | 0.0280 | 289410 |
| GOOD_MESH_TEST | stage2_correct_16000 | 16000 | 0.947 | 0.0161 | 0.0261 | 380860 |
| heimei | stock_sparse | 3715 | 0.515 | 0.0664 | 0.0532 | 106498 |
| heimei | stage2_correct_8192 | 8192 | 0.881 | 0.0344 | 0.0422 | 133558 |
| heimei | stage2_correct_12000 | 12000 | 0.881 | 0.0296 | 0.0441 | 244518 |
| heimei | stage2_correct_16000 | 16000 | 0.907 | 0.0279 | 0.0465 | 353768 |
| maliao | stock_sparse | 2802 | 0.657 | 0.0502 | 0.0659 | 70912 |
| maliao | stage2_correct_8192 | 8192 | 0.915 | 0.0222 | 0.0451 | 108148 |
| maliao | stage2_correct_12000 | 12000 | 0.917 | 0.0260 | 0.0498 | 234346 |
| maliao | stage2_correct_16000 | 16000 | 0.943 | 0.0280 | 0.0468 | 376676 |
| reconviagen_20260520_021556 | stock_sparse | 4204 | 0.827 | 0.0355 | 0.1312 | 99316 |
| reconviagen_20260520_021556 | stage2_correct_8192 | 8192 | 0.986 | 0.0281 | 0.0625 | 86898 |
| reconviagen_20260520_021556 | stage2_correct_12000 | 12000 | 0.975 | 0.0267 | 0.0630 | 118688 |
| reconviagen_20260520_021556 | stage2_correct_16000 | 16000 | 0.981 | 0.0252 | 0.0695 | 216764 |
| reconviagen_20260617_073549 | stock_sparse | 15809 | 0.817 | 0.0346 | 0.0506 | 299398 |
| reconviagen_20260617_073549 | stage2_correct_8192 | 8192 | 0.979 | 0.0169 | 0.0572 | 107460 |
| reconviagen_20260617_073549 | stage2_correct_12000 | 12000 | 0.981 | 0.0179 | 0.0580 | 227256 |
| reconviagen_20260617_073549 | stage2_correct_16000 | 16000 | 0.953 | 0.0186 | 0.0530 | 326948 |
| reconviagen_20260617_075506 | stock_sparse | 4802 | 0.090 | 0.0788 | 0.1723 | 117043 |
| reconviagen_20260617_075506 | stage2_correct_8192 | 8192 | 0.979 | 0.0318 | 0.0919 | 60326 |
| reconviagen_20260617_075506 | stage2_correct_12000 | 12000 | 0.996 | 0.0308 | 0.1099 | 139302 |
| reconviagen_20260617_075506 | stage2_correct_16000 | 16000 | 1.000 | 0.0262 | 0.1144 | 242638 |
| snoopy | stock_sparse | 19648 | 0.546 | 0.0468 | 0.0770 | 451258 |
| snoopy | stage2_correct_8192 | 8192 | 0.851 | 0.0277 | 0.0541 | 119206 |
| snoopy | stage2_correct_12000 | 12000 | 0.871 | 0.0291 | 0.0532 | 189288 |
| snoopy | stage2_correct_16000 | 16000 | 0.852 | 0.0300 | 0.0508 | 274678 |

### 结论

这一轮结果支持继续推进真实 AR/SLAM point prior 路线。

关键依据：

```text
1. internal7 streaming prior 全部样本通过 countcheck，没有出现 prior 完全不可用的问题。
2. Stage2_correct 在 7 个样本上都明显降低 ref_norm_chamfer_l2_mean：
   stock mean 0.0609 -> stage2 约 0.0246-0.0253。
3. extent_ratio 从 stock mean 0.636 提升到 stage2 约 0.937-0.940；
   这说明 Stage2 不只是贴 prior 点，而是在补完整物体尺度。
4. 8192 top-k 的 reference chamfer 与 12000/16000 接近，但顶点数显著更低：
   8192 mean vertices 111,966；
   12000 mean vertices 206,115；
   16000 mean vertices 310,333。
```

因此，现在不是“模型没用”的趋势。更准确的判断是：

```text
1. 对真实 sparse/SLAM-like prior，Stage2 已经能显著改善 stock sparse/mesh 下游。
2. 目前收益主要体现在 sparse->mesh 的粗几何覆盖和尺度恢复；
3. 还不能直接宣称最终系统可用，因为缺少真实手机 AR point map、candidate rerank、以及 direct COLMAP/streaming 的正式成对比较。
```

### 下一步建议

第一优先级：补 direct vs streaming 成对 mesh eval。

```text
heimei direct existing sparse 已经有 970 prior coords；
heimei streaming 有 140 prior coords。
```

应对同一个 heimei 分别跑 mesh eval：

```text
branch A: existing sparse direct
branch B: existing sparse streaming
```

观察：

```text
1. direct 是否因为点更多而 mesh 更好；
2. streaming 是否因为点更干净而更稳；
3. 8192/12000/16000 哪个 top-k 更适合真实 prior。
```

第二优先级：补 `RUN_RECONSTRUCT_POSE=1` 的 COLMAP direct/streaming 分支。

当前系统已经可用：

```text
/home/zjr/anaconda3/envs/foundpose/bin/colmap
```

小 smoke 显示 heimei 8 帧 COLMAP 可跑通：

```text
registered_image_count=3 / selected_count=8
points3D_count=83
mean_reprojection_error=0.614
intrinsics_source=existing_sparse
```

但 8 帧注册比例偏低，不足以作为正式结论。建议用 64 帧跑：

```text
branch A: RGB -> COLMAP points -> direct prior
branch B: RGB -> COLMAP pose -> streaming triangulation prior
```

第三优先级：candidate rerank。

当前 8192、12000、16000 都有合理场景：

```text
8192:
  mesh 更轻，prior chamfer 通常更好，适合默认第一候选；

12000:
  ref chamfer 与 16000 接近，复杂度适中，适合作主候选；

16000:
  ref chamfer mean 最低，但复杂度最高，适合补 coverage 或弱 prior 样本。
```

下一步不建议只固定一个 top-k，而是输出 3 个 mesh candidate，再用：

```text
1. render mask IoU / silhouette consistency
2. AR/SLAM prior point-to-mesh distance
3. extent ratio sanity
4. mesh complexity penalty
```

做 rerank。

第四优先级：真实手机 AR session smoke。

离线 existing sparse 和 COLMAP proxy 已经说明路线有潜力，但最终系统还是要看手机端返回的 sparse point map 是否达到：

```text
prior coords >= 40-60
projection_any_mask_hit_ratio 接近 1
projection_mask_hit_over_inside_mean >= 0.25-0.30
support_mean >= 3
```

如果真实手机点云达不到这个阈值，优先改前端点云采集/筛选；如果达标，再把 candidate rerank 接入实际 AR pipeline。

---

## heimei direct/streaming 与 RGB->COLMAP 64 帧评测结果

时间：2026-06-23 UTC

相关输出：

```text
existing sparse direct mesh:
/home/zjr/Tracker/trellis_point_prior_mv/outputs/ar_like_streaming_slam/ar_like_direct_heimei_existing_sparse_mesh_eval_v1/mesh_eval/report.json

existing sparse streaming mesh:
/home/zjr/Tracker/trellis_point_prior_mv/outputs/ar_like_streaming_slam/ar_like_streaming_heimei_existing_sparse_mesh_eval_v1/mesh_eval/report.json

RGB->COLMAP direct countcheck:
/home/zjr/Tracker/trellis_point_prior_mv/outputs/ar_like_streaming_slam/ar_like_colmap_direct_heimei_rgb64_countcheck_v1/colmap_reconstruction_report.json
/home/zjr/Tracker/trellis_point_prior_mv/outputs/ar_like_streaming_slam/ar_like_colmap_direct_heimei_rgb64_countcheck_v1/manifest/build_report.json

RGB->COLMAP streaming countcheck:
/home/zjr/Tracker/trellis_point_prior_mv/outputs/ar_like_streaming_slam/ar_like_streaming_heimei_rgb64_countcheck_v1/slam_like_points_report.json
/home/zjr/Tracker/trellis_point_prior_mv/outputs/ar_like_streaming_slam/ar_like_streaming_heimei_rgb64_countcheck_v1/manifest/build_report.json

RGB->COLMAP direct mesh:
/home/zjr/Tracker/trellis_point_prior_mv/outputs/ar_like_streaming_slam/ar_like_colmap_direct_heimei_rgb64_mesh_eval_v1/mesh_eval/report.json

RGB->COLMAP streaming mesh:
/home/zjr/Tracker/trellis_point_prior_mv/outputs/ar_like_streaming_slam/ar_like_streaming_heimei_rgb64_mesh_eval_v1/mesh_eval/report.json
```

### Countcheck 对比

| prior source | frames | source pts | prior pts | mask-any | hit/inside | support mean | norm scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| existing direct | 32 | 3059 | 970 | 1.000 | 0.692 | 21.52 | 10.153 |
| existing streaming | 32 | 322 | 140 | 1.000 | 0.903 | 28.58 | 4.836 |
| RGB64 direct | 29 | 434 | 182 | 0.989 | 0.649 | 18.73 | 18.075 |
| RGB64 streaming | 29 | 69 | 55 | 1.000 | 0.915 | 26.55 | 7.359 |

RGB->COLMAP 64 帧重建质量：

```text
selected_count = 64
registered_image_count = 29
registered_ratio = 0.453
points3D_count = 1656
mean_reprojection_error = 0.549
intrinsics_source = existing_sparse
fix_intrinsics = 1
```

解释：

```text
1. existing direct 是 heimei 上最强的 point prior：已有 sparse/0 中 object support 后还有 3059 点，voxel 后 970 点。
2. existing streaming 点更少，但更“干净”：hit/inside 达到 0.903，高于 direct 的 0.692。
3. RGB64 direct 的 COLMAP 重建可用，但注册率只有 29/64=45.3%，导致 object prior 比 existing direct 明显变弱。
4. RGB64 streaming 点最少，仅 55 prior coords，但 mask 命中质量最好，说明 streaming filter 更严格。
```

### Mesh Eval 对比

| prior | mode | coord | ref chamfer | prior chamfer | extent | vertices |
|---|---|---:|---:|---:|---:|---:|
| existing direct | stock_sparse | 7714 | 0.0533 | 0.0712 | 0.645 | 163068 |
| existing direct | stage2_8192 | 8192 | 0.0186 | 0.0242 | 0.960 | 143620 |
| existing direct | stage2_12000 | 12000 | 0.0197 | 0.0301 | 0.964 | 248826 |
| existing direct | stage2_16000 | 16000 | 0.0196 | 0.0334 | 0.979 | 337523 |
| existing streaming | stock_sparse | 7714 | 0.0534 | 0.0731 | 0.645 | 163082 |
| existing streaming | stage2_8192 | 8192 | 0.0189 | 0.0438 | 0.943 | 151004 |
| existing streaming | stage2_12000 | 12000 | 0.0206 | 0.0437 | 0.952 | 258234 |
| existing streaming | stage2_16000 | 16000 | 0.0181 | 0.0450 | 0.939 | 371172 |
| RGB64 direct | stock_sparse | 6780 | 0.0545 | 0.0328 | 0.557 | 122324 |
| RGB64 direct | stage2_8192 | 8192 | 0.0210 | 0.0521 | 0.948 | 165072 |
| RGB64 direct | stage2_12000 | 12000 | 0.0208 | 0.0449 | 0.939 | 280299 |
| RGB64 direct | stage2_16000 | 16000 | 0.0204 | 0.0447 | 0.955 | 391754 |
| RGB64 streaming | stock_sparse | 6780 | 0.0554 | 0.0636 | 0.557 | 122362 |
| RGB64 streaming | stage2_8192 | 8192 | 0.0207 | 0.0590 | 0.961 | 174424 |
| RGB64 streaming | stage2_12000 | 12000 | 0.0203 | 0.0592 | 0.953 | 273444 |
| RGB64 streaming | stage2_16000 | 16000 | 0.0210 | 0.0590 | 0.952 | 395704 |

### 结论

这轮 heimei 结果有三个明确结论。

第一，Stage2 对真实 prior 是有效的，不是只在合成数据上有效：

```text
existing direct:
  stock ref chamfer 0.0533 -> stage2 0.0186-0.0197

existing streaming:
  stock ref chamfer 0.0534 -> stage2 0.0181-0.0206

RGB64 direct:
  stock ref chamfer 0.0545 -> stage2 0.0204-0.0210

RGB64 streaming:
  stock ref chamfer 0.0554 -> stage2 0.0203-0.0210
```

第二，existing sparse direct 在 heimei 上最好：

```text
existing direct + stage2_8192:
  ref chamfer = 0.0186
  prior chamfer = 0.0242
  extent = 0.960
  vertices = 143620
```

它同时有最强 prior 点数和最低 prior chamfer。说明如果真实系统能提供高质量 sparse map points，直接使用 map points 是有价值的。

第三，RGB->COLMAP proxy 目前能跑通，但还不是最优 proxy：

```text
registered_ratio = 0.453
RGB64 direct prior = 182 coords
RGB64 streaming prior = 55 coords
```

这说明当前 RGB->COLMAP 64 帧在 heimei 上只注册了不到一半帧。Stage2 仍然能补出合理几何，但 prior 本身明显弱于已有 `sparse/0`。因此如果后面要用 RGB->COLMAP 作为真实数据集 proxy，应该先优化 COLMAP/SLAM 重建质量，而不是直接把它当作最终训练数据来源。

### top-k 判断

当前 heimei 上：

```text
8192:
  通常已经足够，顶点数最低；
  existing direct 中 ref/prior chamfer 都最好。

12000:
  更稳的中间值，但复杂度明显上升；
  RGB64 direct/streaming 中与 8192/16000 差距不大。

16000:
  有时 ref chamfer 略好，但顶点/面数最高；
  不适合作默认唯一输出，适合作候选之一。
```

因此默认 candidate 仍建议保留：

```text
8192 / 12000 / 16000
```

但 rerank 时应该偏好：

```text
1. ref/silhouette 合理；
2. prior distance 不爆；
3. extent_ratio 在合理区间；
4. vertex_count 不过高。
```

### 下一步建议

第一优先级：补 `COLMAP_USE_MASKS=1` ablation。

目的不是让 mask-only 必然更好，而是判断：

```text
full-image COLMAP:
  pose 更稳，但背景点更多；

mask-only COLMAP:
  object points 更干净，但低纹理物体可能注册失败。
```

建议只先跑 heimei：

```text
RGB64 direct full-image vs RGB64 direct mask-only
RGB64 streaming full-image vs RGB64 streaming mask-only
```

第二优先级：优化 RGB->COLMAP 注册率。

当前 `registered_ratio=0.453` 偏低。建议尝试：

```text
1. COLMAP_MAX_FRAMES=96 或 128；
2. COLMAP_MATCHER=sequential, COLMAP_SEQUENTIAL_OVERLAP=12；
3. COLMAP_FRAME_SELECT=random_uniform 保持；
4. 保持 fixed existing intrinsics。
```

目标是：

```text
registered_ratio >= 0.65
points3D_count 明显高于 1656
object prior coords >= 300
```

第三优先级：实现 candidate rerank。

现在 Stage2 候选都有效，真正需要的是自动选：

```text
8192 / 12000 / 16000 哪个 mesh 更适合当前样本。
```

rerank 指标建议：

```text
1. multi-view silhouette IoU；
2. prior point-to-mesh distance；
3. mesh extent sanity；
4. vertex/face complexity penalty。
```

第四优先级：真实手机 AR session smoke。

heimei 说明：如果有高质量 sparse points，direct route 很强。
所以手机端如果能上传 AR/SLAM map points，应优先测试：

```text
AR map points direct prior
AR pose + local feature streaming triangulation prior
```

并比较二者是否复现 heimei 的趋势。

## 四十四、2026-06-24 手机 AR session mesh smoke 与黑色碎片分析

### 测试对象

本轮使用 `trellis_point_prior_mv/scripts/run_ar_session_server.sh` 采集的真实手机 AR session，其中四组完成 capture-only finalize，具备 RGB、pose、mask、`slam_points.jsonl` 和 `trellis_point_prior_capture_report.json`：

```text
20260624_010530_849
20260624_011149_207
20260624_011450_447
20260624_011933_843
```

其中 `20260624_011933_843` 点云最密，`ar_direct` 构建后达到 `prior=1500` 上限；`20260624_011450_447` 只有 `prior=100`，用于低点数压力测试。

可视化和统计输出路径：

```text
trellis_point_prior_mv/outputs/ar_session_smoke/mesh_visual_analysis_20260624/
```

### AR direct countcheck 结论

| session | frames | source points | prior coords | mask any | hit inside | support mean |
|---|---:|---:|---:|---:|---:|---:|
| 20260624_010530_849 | 18 | 2810 | 565 | 0.986 | 0.326 | 3.61 |
| 20260624_011149_207 | 18 | 2025 | 302 | 0.997 | 0.270 | 3.99 |
| 20260624_011450_447 | 18 | 278 | 100 | 1.000 | 0.430 | 7.45 |
| 20260624_011933_843 | 18 | 9811 | 1500 | 0.996 | 0.374 | 5.26 |

结论：

1. 真实手机 session 的格式是可用的，AR direct point prior 能进入当前 Stage2 构建和 mesh eval 流程。
2. 手机直接回传的 AR 点云明显比 pose-streaming SIFT 三角化更密。`pose_streaming` 在同样 session 上通常只有几十到一百多个 prior coords，因此当前真实手机接入优先测试 `ar_direct`。
3. `20260624_011933_843` 是当前最适合做 mesh smoke 的样本；`20260624_011450_447` 是低点数失败边界样本。

### Dense session mesh 结果

session：`20260624_011933_843`

输入图像显示目标是紫红色透明/半透明抽屉类物体，背景主要是地砖和椅子，不存在大面积黑色目标表面。

| mode | prior coords | sparse coords | vertices | faces | extent ratio | prior chamfer | black vertex ratio | components |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stock_sparse | 1500 | 3097 | 93978 | 187940 | 0.201 | 0.0587 | 0.64% | 2 |
| stage2_correct_8192 | 1500 | 8192 | 140036 | 276360 | 0.789 | 0.0178 | 54.9% | 1004 |
| stage2_correct_12000 | 1500 | 12000 | 246140 | 488766 | 0.824 | 0.0246 | 60.9% | 957 |

判断：

1. `stage2_correct_8192` 明显比 `stock_sparse` 更贴近 AR point prior，`prior chamfer` 从 `0.0587` 降到 `0.0178`。
2. 但 Stage2 输出接冻结 stock slat/mesh 后产生大量黑色碎片。`8192` 已经有 `1004` 个连通块，最大连通块比例只有约 `46.8%`。
3. `12000` 比 `8192` 更过填充，顶点/面数大幅增加，黑色顶点比例和 prior chamfer 都更差，不应作为当前主候选。

### Sparse session mesh 结果

session：`20260624_011450_447`

输入图像显示目标是橙红色玩具，颜色干净，目标本身也不能解释 Stage2 mesh 的大面积黑色。

| mode | prior coords | sparse coords | vertices | faces | extent ratio | prior chamfer | black vertex ratio | largest component ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stock_sparse | 100 | 4383 | 77874 | 155736 | 0.448 | 0.2155 | 0.61% | 86.1% |
| stage2_correct_8192 | 100 | 8192 | 141134 | 274924 | 0.946 | 0.1743 | 82.8% | 3.7% |
| stage2_correct_12000 | 100 | 12000 | 248234 | 489604 | 0.941 | 0.1684 | 67.9% | 2.4% |

判断：

1. 低点数 AR prior 下，Stage2 仍会把预测 sparse 拉向 prior 附近，但 mesh 基本碎成大量黑色小片。
2. 该样本只能说明“100 个 voxel prior 太弱”，不能作为当前模型可用于 CoarseModel 的证据。
3. 真实 AR session 进入 mesh 生成时，至少需要 prior coords 达到几百以上；`prior=100` 这种场景应触发重采集或降级到 stock/reconviagen，而不是直接使用 Stage2 mesh。

### 黑色微影原因

本轮检查了 OBJ 顶点颜色。黑色不是 viewer 阴影，也不是输入物体真实颜色，而是写入 OBJ 的顶点颜色本身已经大面积接近黑色。

主要原因判断：

1. 当前 Stage2 只训练/替换 sparse flow，后续 slat flow、mesh decoder、texture/vertex color 仍是冻结 stock TRELLIS/ReconViaGen 下游。
2. Stage2 输出的 sparse coords 对冻结 slat flow 来说存在分布偏移，尤其是 AR prior bbox 归一化后可能产生 unsupported voxel 和大量孤立小连通块。
3. 这些孤立 voxel 进入冻结 slat/mesh 后没有可靠图像支持，texture/vertex color 容易被生成为黑色或近黑色。
4. top-k 越高，unsupported coords 越多，黑色碎片越明显。`12000` 在本轮已经比 `8192` 更差。
5. 低点数 prior 下，模型为了补全会更容易发散到大范围碎片，因此黑色微影更严重。

因此，这不是简单的“颜色后处理”问题。重染色只能遮住黑色，不能修复 geometry 碎片。真正需要先处理 Stage2 输出 sparse 的 unsupported / isolated component。

### 当前阶段结论

1. 真实手机 AR direct point prior 已经打通，数据格式和构建流程可用。
2. Dense AR point prior 上，Stage2 sparse 的确有正向信号：它能显著提升 mesh 对 point prior 的贴合程度。
3. 但当前 Stage2 mesh 不能直接作为最终 mesh 进入 CoarseModel，因为黑色碎片和连通块爆炸说明 sparse 输出与冻结下游 slat/mesh 不匹配。
4. `stage2_correct_8192` 是当前比 `12000` 更合理的候选；`12000/16000` 不应作为默认主候选。
5. 在解决 post-filter / rerank / slat 适配之前，不建议直接做大规模长训并期望 mesh 质量自然变好。

### 下一步建议

第一优先级：补低 top-k sweep，只测 dense session。

建议：

```text
topk = 4096 / 6144 / 8192
```

目标是确认能否在保留 prior 贴合收益的同时减少黑色碎片和连通块数量。

第二优先级：在 Stage2 sparse 输出后、进入 slat 前增加过滤。

建议过滤项：

```text
1. mask/pose projection support filter；
2. visible/support view count filter；
3. remove tiny connected components；
4. optionally keep largest/high-support components；
5. 对过低 prior_point_count 的 session 直接拒绝或降级。
```

第三优先级：实现 candidate rerank。

候选不应该只按 prior distance 排序，还需要加入：

```text
1. dark_vertex_ratio；
2. component_count；
3. largest_component_ratio；
4. mesh extent sanity；
5. prior point-to-mesh distance；
6. multi-view mask silhouette consistency。
```

第四优先级：再考虑 slat flow 训练。

如果过滤和 rerank 后 Stage2 sparse 仍然稳定优于 stock 但颜色/细节差，才进入 slat flow 训练。当前直接训练 slat flow 的风险是：sparse 输入本身还没有被约束干净，slat 会被迫学习处理大量 unsupported/fragmented geometry。

## 四十五、2026-06-24 四组手机 AR session top-k 诊断结果

### 测试设置

本轮对四组真实手机 AR session 统一运行：

```text
PRIOR_BRANCH=ar_direct
MODES=stock_sparse,stage2_correct
TOPK_SPECS=4096,6000,8192,12000
MESH_EVAL_SAMPLES=2000
```

对应输出：

```text
trellis_point_prior_mv/outputs/ar_session_smoke/ar_session_20260624_*_ar_direct_mesh_smoke_topk_diag_v2/
```

额外 OBJ 级黑色顶点和连通块统计：

```text
trellis_point_prior_mv/outputs/ar_session_smoke/topk_diag_v2_mesh_artifact_summary.json
```

### 按 top-k 聚合结果

| mode | count | chamfer mean | black mean | component mean | largest comp mean | vertex mean | extent mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| stock_sparse | 4 | 0.1099 | 0.019 | 4.2 | 0.962 | 108276 | 0.373 |
| stage2_correct_4096 | 4 | 0.0464 | 0.532 | 738.2 | 0.258 | 44348 | 0.908 |
| stage2_correct_6000 | 4 | 0.0702 | 0.649 | 1360.2 | 0.182 | 87518 | 0.914 |
| stage2_correct_8192 | 4 | 0.0794 | 0.705 | 1662.8 | 0.156 | 144989 | 0.910 |
| stage2_correct_12000 | 4 | 0.0789 | 0.709 | 1635.8 | 0.169 | 262682 | 0.913 |

结论：

1. `4096` 是当前最稳的 Stage2 top-k。四个 session 中，`stage2_correct_4096` 都是 prior chamfer 最低的 Stage2 候选。
2. top-k 增大后，vertex/face 数显著增加，但 prior chamfer 反而变差，说明 `6000/8192/12000` 主要是在补 unsupported / low-confidence 区域，而不是稳定补全物体。
3. 降到 `4096` 能减少碎片和黑色顶点，但没有根治。`stage2_correct_4096` 的平均黑色顶点比例仍有 `53.2%`，平均连通块仍有 `738` 个。
4. stock mesh 的 black ratio 和连通性非常正常，说明黑色碎片不是输入图像或 OBJ viewer 的普遍问题，而是 Stage2 sparse 输出接冻结 slat/mesh 后产生的分布问题。

### 每组最佳 Stage2 候选

| session | prior | stock chamfer | stock black | stock comp | stock largest | best stage2 | stage2 chamfer | stage2 black | stage2 comp | stage2 largest |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 20260624_010530_849 | 565 | 0.0602 | 0.064 | 1 | 1.000 | stage2_correct_4096 | 0.0103 | 0.515 | 682 | 0.166 |
| 20260624_011149_207 | 302 | 0.1007 | 0.001 | 5 | 0.999 | stage2_correct_4096 | 0.0557 | 0.537 | 914 | 0.156 |
| 20260624_011450_447 | 100 | 0.2190 | 0.006 | 9 | 0.861 | stage2_correct_4096 | 0.1140 | 0.816 | 963 | 0.127 |
| 20260624_011933_843 | 1500 | 0.0596 | 0.006 | 2 | 0.988 | stage2_correct_4096 | 0.0056 | 0.260 | 394 | 0.583 |

解读：

1. `20260624_011933_843` 是唯一一个 Stage2 结果相对可分析的 session：`prior=1500`，`stage2_correct_4096` 的 chamfer 降到 `0.0056`，黑色比例降到 `26.0%`，最大连通块比例达到 `58.3%`。
2. 其余三组即使使用 `4096`，仍然表现为大量黑色碎片，最大连通块比例只有 `12.7%-16.6%`。这说明当前 Stage2 对低质量/低密度 AR prior 非常敏感。
3. `prior=100` 的 `20260624_011450_447` 明确不适合作为可用生成样本，应视为重采集或降级触发条件。

### 对三个可能原因的判断

本轮 top-k 诊断进一步支持“三个因素叠加”的解释：

```text
1. Stage2 sparse coords 本身已经有碎片化问题；
2. unsupported surface 缺少可靠图像条件，导致 texture/vertex color 生成黑色；
3. 即使部分 sparse coords 贴近 prior，冻结 slat/mesh decoder 对这种 Stage2 sparse 分布仍然 OOD。
```

其中最直接的证据是：

```text
stock_sparse:
  black mean = 1.9%
  component mean = 4.2
  largest comp mean = 96.2%

stage2_correct_4096:
  black mean = 53.2%
  component mean = 738.2
  largest comp mean = 25.8%
```

如果只是 texture 问题，连通块不应该从个位数暴增到几百上千。因此 Stage2 sparse 本身的 unsupported / isolated coords 必须优先处理。

### 当前阶段结论

1. 手机 AR direct point prior 方向仍然有信号：Stage2 可以显著降低 mesh 到 prior 的距离。
2. 但当前 mesh 质量瓶颈不是 top-k 取值本身，而是 Stage2 sparse 输出缺少几何连通性和可见性约束。
3. `4096` 应作为后续默认诊断 top-k；`8192/12000/16000` 暂时不适合作为 AR session 默认候选。
4. 现在还不能把 Stage2 mesh 直接接入 CoarseModel，也不应该立刻进入 slat 长训。

### 下一步建议

第一优先级：新增 sparse-level 诊断，不经过 slat/mesh。

需要直接统计 Stage2 输出 sparse coords：

```text
1. sparse coord connected components；
2. largest sparse component ratio；
3. tiny component count；
4. prior-supported coord ratio；
5. mask-projection-supported coord ratio；
6. outside visual hull ratio。
```

如果 sparse coords 已经碎，问题在 sparse flow / top-k / prior normalization；如果 sparse coords 连通但 mesh 碎，问题更偏 slat/mesh OOD。

第二优先级：实现 Stage2 sparse 输出过滤后再进 slat。

建议先做 eval-time 过滤，不重训：

```text
1. keep largest sparse component；
2. keep components near prior coords；
3. mask/pose support filter；
4. prior distance filter；
5. top-k=4096。
```

只要过滤后黑色顶点比例和连通块数显著下降，就证明当前主要问题是 Stage2 输出中的 unsupported/fragment coords。

第三优先级：candidate rerank 加入 artifact 指标。

当前 rerank 不应只看 prior chamfer，应加入：

```text
dark_vertex_ratio
component_count
largest_component_ratio
extent_ratio
prior_norm_chamfer
```

第四优先级：设置真实 AR session 使用门槛。

建议暂定：

```text
prior coords < 300:
  不跑 Stage2，触发重采集或降级；

300 <= prior coords < 800:
  只做 diagnostic，不作为最终 mesh；

prior coords >= 1000:
  可以进入 Stage2 + filter + rerank smoke。
```

第五优先级：在 sparse 输出过滤验证有效后，再考虑 slat flow 训练。

如果不先过滤 sparse，slat 训练会被迫学习大量 unsupported/fragmented sparse 输入，训练信号会混乱。当前更合理的顺序是：

```text
sparse diagnostics
-> sparse filtering
-> candidate rerank
-> 再决定是否训练 slat flow
```

## 四十六、2026-06-24 Stage2 sparse filter 诊断结果与 visual-hull 指标修正

### 本轮运行内容

本轮对四组手机 AR session 运行了 Stage2 sparse-level 诊断：

```text
topk = 2048 / 4096 / 6144 / 8192
filter_specs =
  none
  min_component_size
  prior_radius,projection_support
  min_component_size,prior_radius,projection_support
  largest_component,prior_radius,projection_support
```

输出路径：

```text
trellis_point_prior_mv/outputs/ar_session_smoke/ar_session_20260624_*_stage2_sparse_diag_v1/
```

随后对最密的 session `20260624_011933_843` 跑了两组 filtered mesh：

```text
min_component_size,prior_radius,projection_support
largest_component,prior_radius,projection_support
```

输出路径：

```text
trellis_point_prior_mv/outputs/ar_session_smoke/ar_session_20260624_011933_843_stage2_filter_*_4096_mesh_v1/
```

### 代码修正

本轮确认了一个问题：之前的 `projection_support` 只表示“任意一帧投影进 mask”，不是严格 visual hull。

因此补充了以下指标：

```text
visible_outside_mask_event_ratio
visible_outside_mask_point_mean
visible_support_ratio_mean
visible_support_ratio_median
visual_hull_keep_ratio
visual_hull_kept_count
```

并新增可配置阈值：

```text
--visual_hull_min_visible_views
--visual_hull_min_support_ratio
```

修改文件：

```text
trellis_point_prior_mv/sparse_coord_tools.py
trellis_point_prior_mv/eval_real_slam_stage2_sparse_diagnostics.py
trellis_point_prior_mv/run_real_slam_prior_mesh.py
trellis_point_prior_mv/scripts/run_ar_session_smoke.sh
trellis_point_prior_mv/命令说明.txt
```

注意：

```text
--filter_min_component_size 64
```

只有在 filter spec 包含 `min_component_size` 时生效；对 `largest_component` 不生效。

### Sparse 诊断结论

#### 1. raw Stage2 sparse 本身已经碎

以 `topk=4096` 为例：

| session | prior coords | raw coord count | raw components | raw largest ratio | raw projection keep |
|---|---:|---:|---:|---:|---:|
| 20260624_010530_849 | 565 | 4096 | 1616 | 0.078 | 0.590 |
| 20260624_011149_207 | 302 | 4096 | 1846 | 0.075 | 0.563 |
| 20260624_011450_447 | 100 | 4096 | 2124 | 0.039 | 0.506 |
| 20260624_011933_843 | 1500 | 4096 | 1214 | 0.392 | 0.737 |

结论：

1. 前三组 raw Stage2 sparse 已经非常碎，最大连通块比例低于 `0.08`。
2. 最密的 `20260624_011933_843` 明显更好，但仍有 `1214` 个连通块。
3. 因此黑色碎片不能只归因于 texture/vertex color，Stage2 sparse coords 本身确实存在 unsupported / isolated component。

#### 2. filter 能显著清理 sparse，但不同策略强度不同

`20260624_011933_843, topk=4096`：

| filter | final coords | final components | final largest ratio | filter keep |
|---|---:|---:|---:|---:|
| none | 4096 | 1214 | 0.392 | 1.000 |
| min_component_size | 1794 | 3 | 0.895 | 0.438 |
| prior_radius,projection_support | 2280 | 147 | 0.689 | 0.557 |
| min_component_size,prior_radius,projection_support | 1727 | 5 | 0.910 | 0.422 |
| largest_component,prior_radius,projection_support | 1575 | 2 | 0.997 | 0.385 |

结论：

1. `min_component_size` 是更稳的基础过滤。它能把 component 从 `1214` 降到 `3`，同时保留 `1794` 个 coords。
2. `largest_component` 最强，能得到接近单一连通块，但会更激进地删除结构。
3. `prior_radius,projection_support` 单独使用不够，仍保留 `147` 个 component。

#### 3. 当前 projection_support 太宽松

在重新计算 stricter visual-hull 指标后，发现很多 coords 虽然能在至少一帧命中 mask，但在所有可见视角中仍大量落在 mask 外。

`20260624_011933_843, topk=4096`：

| filter | coord count | component | largest | projection keep | visual hull keep | outside mask event | visible support mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 4096 | 1214 | 0.392 | 0.737 | 0.698 | 0.699 | 0.273 |
| min_component_size | 1794 | 3 | 0.895 | 0.963 | 0.944 | 0.643 | 0.354 |
| prior_radius,projection_support | 2280 | 147 | 0.689 | 1.000 | 0.975 | 0.637 | 0.367 |
| min_component_size,prior_radius,projection_support | 1727 | 5 | 0.910 | 1.000 | 0.981 | 0.634 | 0.368 |
| largest_component,prior_radius,projection_support | 1575 | 2 | 0.997 | 1.000 | 0.979 | 0.633 | 0.368 |

解读：

1. `projection_keep=1.0` 不代表严格 visual hull 正确，只代表至少满足当前支持阈值。
2. `outside mask event` 仍在 `0.63` 左右，说明在可见视角里仍有大量投影落在 mask 外。
3. 当前 `visual_hull_min_support_ratio=0.10` 仍较宽松；下一轮应提高到 `0.25` 或更高，并要求至少 2 个可见视角。

### Filtered mesh 结果

对 `20260624_011933_843, topk=4096`，两组 filtered mesh 的 OBJ 统计如下：

| mode | coords | prior chamfer | black ratio | dark ratio | mesh components | largest mesh ratio | small comp ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| unfiltered stage2_4096 | 4096 | 0.0056 | 0.260 | 0.340 | 394 | 0.583 | 0.208 |
| largest_component,prior_radius,projection_support | 1815 | 0.0089 | 0.267 | 0.341 | 45 | 0.832 | 0.020 |
| min_component_size,prior_radius,projection_support | 1956 | 0.0043 | 0.349 | 0.424 | 72 | 0.811 | 0.049 |

结论：

1. sparse filter 明显改善了 mesh 连通性：mesh components 从 `394` 降到 `45/72`，最大 mesh component 从 `0.583` 提升到 `0.811-0.832`。
2. 但黑色顶点比例没有下降，甚至 `min_component_size` 方案从 `0.260` 升到 `0.349`。
3. 这说明过滤能修几何碎片，但黑色问题不完全由小连通块导致；冻结 slat/mesh/texture 对 Stage2 sparse 分布仍然 OOD，或者剩余表面仍缺少可靠图像支持。
4. `largest_component,prior_radius,projection_support` 在 mesh 连通性上更好，`min_component_size,prior_radius,projection_support` 在 prior chamfer 上更好。二者体现了“几何干净”和“贴 prior”之间的 tradeoff。

### 当前判断

本轮结果把三个原因拆得更清楚：

```text
1. Stage2 sparse coords 本身碎：成立。
   raw component 数量很高，largest ratio 很低。

2. unsupported surface 缺少可靠图像条件：成立。
   projection_support 过宽松，visible outside mask event 仍很高。

3. frozen slat/mesh decoder 对 Stage2 coords OOD：也成立。
   sparse filter 后 mesh 连通性改善，但黑色比例没有同步下降。
```

因此，下一步不能只做单一方案。应先继续把 sparse filter 做严格，再判断 slat 是否必须训练。

### 下一步建议

第一优先级：跑 stricter visual-hull diagnostics。

建议配置：

```text
topk = 2048,4096
filter_min_support_views = 2
visual_hull_min_visible_views = 2
visual_hull_min_support_ratio = 0.25
```

只先跑 `20260624_011933_843`。

第二优先级：如果严格 visual-hull 后仍保留几百到一千多个 coords，再跑 filtered mesh。

优先测试：

```text
min_component_size,prior_radius,projection_support
```

原因是它比 `largest_component` 更不容易误删真实多部件结构。

第三优先级：补 mesh 后 artifact rerank。

现在需要把以下指标自动算入 report：

```text
black_vertex_ratio
dark_vertex_ratio
mesh_component_count
mesh_largest_component_ratio
small_component_ratio
```

否则只能人工额外脚本统计，无法自动 rerank。

第四优先级：暂不直接 slat 长训。

本轮已经证明 sparse filter 能减少 mesh component，但不能解决黑色。下一步应先确认 stricter visual-hull 是否能同时减少黑色；如果不能，再考虑：

```text
1. slat flow 适配 Stage2 sparse；
2. texture/vertex color 约束；
3. 或 Stage2 sparse 后处理 + stock/reconviagen candidate rerank。
```

## 四十七、2026-06-24 Base sparse vs Stage2 sparse 直接对比

### 测试目的

上一轮已经从 mesh 结果看到：

```text
stock mesh 连通性正常；
Stage2 mesh 黑色碎片和连通块很多。
```

但 mesh 结果仍然是经过 slat/mesh decoder 后的间接证据。本轮直接比较：

```text
base / stock sparse sampler 输出的 sparse coords
vs
Stage2 topk=4096 输出的 sparse coords
```

目的是回答：

```text
正常 TRELLIS sparse 输出本来是否就会像 Stage2 一样碎？
```

### 结果

统计文件：

```text
trellis_point_prior_mv/outputs/ar_session_smoke/base_vs_stage2_sparse_diag_v1.json
```

| session | prior | base coords | base comp | base largest | base proj keep | stage2 coords | stage2 comp | stage2 largest | stage2 proj keep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260624_010530_849 | 565 | 5530 | 1 | 1.000 | 0.533 | 4096 | 1616 | 0.078 | 0.590 |
| 20260624_011149_207 | 302 | 5427 | 1 | 1.000 | 0.371 | 4096 | 1846 | 0.075 | 0.563 |
| 20260624_011450_447 | 100 | 4383 | 2 | 0.901 | 0.752 | 4096 | 2124 | 0.039 | 0.506 |
| 20260624_011933_843 | 1500 | 3097 | 1 | 1.000 | 0.812 | 4096 | 1214 | 0.392 | 0.737 |

### 结论

1. 可以直接证明：base sparse 的输出本来不应该像 Stage2 这样散。
2. 四组手机 AR session 中，base sparse 基本都是 `1-2` 个连通块，最大连通块比例 `0.901-1.000`。
3. Stage2 topk=4096 则是 `1214-2124` 个连通块，前三组最大连通块比例只有 `0.039-0.078`。
4. 因此，Stage2 当前的碎片不是 TRELLIS sparse decoder 的天然现象，而是 Stage2 point-prior sparse 输出分布的问题。
5. base sparse 的 projection keep 未必比 Stage2 高，但其几何连通性远强。这说明 Stage2 虽然更贴 prior，但打破了 stock sparse 的自然几何连通分布。

### 对后续路线的影响

这一步把问题定位进一步前移：

```text
Stage2 sparse flow 本身需要连通性 / topology regularization / component-aware filtering。
```

如果只训练 slat flow，slat 会被迫适配碎片化 sparse 输入，风险较高。更合理的顺序仍然是：

```text
1. 先修 Stage2 sparse 输出连通性；
2. 再做 filtered sparse -> frozen slat/mesh；
3. 若几何干净但颜色仍黑，再训练 slat / texture。
```

下一步应优先测试：

```text
1. stricter visual-hull filter；
2. component-aware sparse post-filter；
3. 训练阶段加入 sparse connectivity / anti-island regularization；
4. 最后再考虑 slat flow。
```

## 四十八、2026-06-24 既有真实/合成数据的 Base vs Stage2 sparse 连通性复查

### 测试目的

上一节只比较了四组手机 AR session。为了确认“Stage2 sparse 碎片化”是否只发生在手机 AR session，本轮复查了先前已经跑过的既有输出：

```text
1. PixalV9 synthetic val64 top-k sweep
2. PixalV9 synthetic val128 top-k sweep
3. real_slam_prior_four top-k sweep
```

这一步不需要重新跑 GPU，只读取已有 `sparse_coords.npz` 并统计连通块。

统计文件：

```text
trellis_point_prior_mv/outputs/sparse_component_compare_base_stage2_existing_runs_v1.json
```

### PixalV9 synthetic val64

输出来源：

```text
trellis_point_prior_mv/outputs/mesh_frozen_downstream/antioverfill_rank_w0005_ws05_s200_seed42_val64_mesh_topk/
```

| mode | count | coords mean | comp mean | comp median | largest mean | largest median | small64 coord mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_sparse | 64 | 8363 | 2.8 | 1.0 | 0.958 | 1.000 | 0.001 |
| stock_sparse | 64 | 5755 | 1.9 | 1.0 | 0.941 | 1.000 | 0.002 |
| stage2_correct_r0_35_cap4096 | 64 | 2585 | 984.8 | 1070.0 | 0.050 | 0.025 | 0.934 |
| stage2_correct_r0_50_cap8192 | 64 | 3989 | 1101.1 | 1190.5 | 0.081 | 0.043 | 0.806 |
| stage2_correct_r0_75_cap12000 | 64 | 5965 | 1040.2 | 1082.5 | 0.238 | 0.177 | 0.555 |
| stage2_correct_target_unique | 64 | 8363 | 887.0 | 895.0 | 0.482 | 0.491 | 0.404 |

结论：

1. GT `target_sparse` 和 stock/base sparse 都是正常连通结构，median component 都是 `1`。
2. Stage2 在所有 top-k 下都有大量碎片。即使 top-k 较低，`r0.35_cap4096` 的平均 component 也接近 `985`。
3. Stage2 top-k 越高，largest component ratio 会提升，但 small component 占比仍然很高。

### PixalV9 synthetic val128

输出来源：

```text
trellis_point_prior_mv/outputs/mesh_frozen_downstream/antioverfill_rank_w0005_ws05_s200_seed42_val128_mesh_topk/
```

| mode | count | coords mean | comp mean | comp median | largest mean | largest median | small64 coord mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_sparse | 128 | 8159 | 2.3 | 1.0 | 0.974 | 1.000 | 0.001 |
| stock_sparse | 128 | 5901 | 1.9 | 1.0 | 0.962 | 1.000 | 0.002 |
| stage2_correct_r0_35_cap4096 | 128 | 2532 | 969.1 | 1053.0 | 0.061 | 0.026 | 0.923 |
| stage2_correct_r0_50_cap8192 | 128 | 3923 | 1089.6 | 1177.5 | 0.091 | 0.043 | 0.798 |
| stage2_correct_r0_75_cap12000 | 128 | 5868 | 1045.9 | 1049.5 | 0.244 | 0.164 | 0.567 |
| stage2_correct_target_unique | 128 | 8159 | 919.7 | 908.5 | 0.463 | 0.476 | 0.423 |

结论：

1. val128 复现了 val64 的现象，说明碎片化不是小样本偶然。
2. `target_sparse` 和 `stock_sparse` 仍然高度连通，而 Stage2 输出是上千个小岛。
3. 这进一步证明：Stage2 sparse 输出分布偏离了 TRELLIS 正常 sparse 分布。

### Real SLAM four top-k

输出来源：

```text
trellis_point_prior_mv/outputs/real_slam_prior/real_slam_prior_four_triangulated_mesh_topk_v1/mesh_eval/
```

| mode | count | coords mean | comp mean | comp median | largest mean | largest median | small64 coord mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| stock_sparse | 4 | 12158 | 1.0 | 1.0 | 1.000 | 1.000 | 0.000 |
| stage2_correct_6000 | 4 | 6000 | 2174.5 | 2189.0 | 0.042 | 0.026 | 0.881 |
| stage2_correct_8192 | 4 | 8192 | 2281.2 | 2374.5 | 0.060 | 0.025 | 0.823 |
| stage2_correct_12000 | 4 | 12000 | 1958.5 | 2060.0 | 0.145 | 0.085 | 0.663 |
| stage2_correct_16000 | 4 | 16000 | 1502.8 | 1550.5 | 0.360 | 0.373 | 0.440 |

结论：

1. 真实 SLAM prior 数据上，stock sparse 也是单一主连通结构。
2. Stage2 仍然碎成上千个 component，和手机 AR session、PixalV9 synthetic 一致。
3. 因此，Stage2 sparse 碎片化不是手机 AR session 独有，也不是真实数据或合成数据某一类数据独有，而是当前 Stage2 sparse flow 输出的普遍问题。

### 总结判断

现在可以更明确地说：

```text
base / stock sparse 输出本来不应该像 Stage2 这样散。
```

证据来自三类数据：

```text
1. 手机 AR session：
   base sparse 基本 1-2 个 component，Stage2 上千 component。

2. PixalV9 synthetic：
   target_sparse 和 stock_sparse median component = 1，Stage2 上千 component。

3. real SLAM prior：
   stock_sparse component = 1，Stage2 上千 component。
```

因此，当前问题不应优先归咎于 slat/mesh decoder。slat/mesh 确实可能对 Stage2 coords OOD，但更前面的事实是：

```text
Stage2 sparse coords 已经明显偏离正常 TRELLIS sparse topology。
```

### 下一步建议

第一优先级不再是继续 top-k sweep，而是修改 Stage2 sparse 训练目标或采样后处理，让 sparse 输出接近 stock/target sparse 的连通拓扑。

建议方向：

```text
1. 训练阶段加入 connectivity / anti-island loss；
2. 对 logits 加邻域平滑或 3D connected-component prior；
3. top-k 后强制 component-aware selection，而不是全局 top-k；
4. 把 point-prior 作为 anchor，但不要让模型在全局 voxel 中产生大量孤立高 logit；
5. 继续保留 eval-time min_component_size / visual-hull filter 作为安全阀。
```

第二优先级才是 slat flow：

```text
只有当 Stage2 sparse 连通性接近 stock/target 后，仍然出现黑色或 texture 错误，才说明必须训练 slat/texture。
```

## 四十九、2026-06-24 手机 AR dense session strict visual-hull filter 结果

### 本轮测试

样本：

```text
20260624_011933_843
```

命令使用：

```text
topk = 2048 / 4096
filter = none / min_component_size / prior_radius,projection_support / min_component_size,prior_radius,projection_support / largest_component,prior_radius,projection_support
filter_min_support_views = 2
visual_hull_min_visible_views = 2
visual_hull_min_support_ratio = 0.25
```

### sparse 诊断结论

`topk=4096` 的 raw Stage2 sparse 仍然非常碎：

| 指标 | 数值 |
|---|---:|
| raw coord count | 4096 |
| raw component count | 1214 |
| raw largest component ratio | 0.392 |
| raw small component coord ratio <64 | 0.562 |
| raw projection keep ratio | 0.629 |
| raw visual hull keep ratio | 0.480 |
| raw visible outside mask event ratio | 0.699 |
| raw visible support ratio mean | 0.273 |

加入 `min_component_size,prior_radius,projection_support` 后：

| 指标 | 数值 |
|---|---:|
| final coord count | 1163 |
| final component count | 23 |
| final largest component ratio | 0.709 |
| final small component coord ratio <64 | 0.194 |
| final within prior radius ratio | 1.000 |
| final projection keep ratio | 1.000 |
| final visual hull keep ratio | 1.000 |
| final visible outside mask event ratio | 0.525 |
| final visible support ratio mean | 0.466 |

这说明 strict filter 是有效的：它把上千个 sparse 小岛压到几十个 component，主连通块占比也明显提高。但它不是根治，因为最终仍有约一半的可见投影事件落在 mask 外，这意味着 sparse 仍存在明显 unsupported / 边界外扩问题。

### filtered mesh 结果

filtered mesh 输出：

```text
trellis_point_prior_mv/outputs/ar_session_smoke/ar_session_20260624_011933_843_stage2_filter_min_component_size__prior_radius__projection_support_4096_strict_vh_mesh_v1/
```

关键指标：

| 指标 | 数值 |
|---|---:|
| stage2 final coords | 1163 |
| vertex count | 24334 |
| face count | 48504 |
| extent ratio | 0.225 |
| prior_norm_chamfer_l2_mean | 0.00665 |
| black vertex ratio, luminance < 0.03 | 0.078 |
| dark vertex ratio, luminance < 0.08 | 0.147 |
| mesh component count | 56 |
| mesh largest component ratio | 0.733 |
| small mesh component vertex ratio <100 | 0.040 |

相比未过滤的 Stage2 mesh，strict filter 后黑色顶点和 mesh 小碎片都有明显下降。这个结果支持一个判断：

```text
黑色碎片很大一部分来自 Stage2 sparse coords 本身的碎片化和 unsupported surface。
```

但剩余问题仍然存在：

```text
1. mesh 仍有 56 个 component，主连通块只有 73.3%；
2. extent ratio 只有 0.225，几何偏薄；
3. final visible outside mask event ratio 仍有 0.525；
4. strict filter 只保留 28.4% raw coords，说明 raw Stage2 输出本身离正常 sparse 拓扑还很远。
```

### 本轮代码改动

已加入自动 mesh artifact 统计：

```text
trellis_point_prior_mv/eval_mesh_frozen_downstream.py
trellis_point_prior_mv/run_real_slam_prior_mesh.py
```

后续 mesh eval 的 `report.json / report.csv` 会直接包含：

```text
artifact_black_vertex_ratio_lum_lt_003
artifact_dark_vertex_ratio_lum_lt_008
artifact_mesh_component_count
artifact_mesh_largest_component_ratio
artifact_mesh_small_component_vertex_ratio_lt100
artifact_mesh_top10_component_counts
```

同时已在 Stage2 训练中加入可关闭的 anti-island / connectivity regularization：

```text
trellis_point_prior_mv/train_sparse_inpaint_stage2.py
trellis_point_prior_mv/scripts/run_pixal_v9_point_prior_stage2.sh
```

新增参数：

```text
CONNECTIVITY_LOSS_WEIGHT
CONNECTIVITY_NEIGHBOR_THRESHOLD
```

默认 `CONNECTIVITY_LOSS_WEIGHT=0.0`，不影响旧实验复现。开启后，它会惩罚目标 sparse 外部、邻域平均概率过低但自身概率偏高的孤立 high-logit 点，目标是减少 Stage2 raw sparse 的小岛。

### 下一步建议

当前最合理的下一步不是训练 slat flow，而是先验证 connectivity regularization 是否能让 Stage2 sparse 接近 stock/target sparse 的拓扑：

```text
1. 先跑 connectivity smoke；
2. 看 synthetic val16/val64 的 Stage2 component count 是否从上千下降；
3. 看 phone AR strict VH 后的 black/dark ratio 是否继续下降；
4. 如果 sparse topology 明显改善，再做 s200；
5. 只有 sparse 已经连通但 mesh 仍黑，才进入 slat flow / texture 训练。
```

如果 connectivity loss 让 mesh 过薄或 recall 明显下降，应优先降低：

```text
CONNECTIVITY_LOSS_WEIGHT: 0.01 -> 0.005
```

而不是继续加强 filter。filter 适合作为安全阀，不应该成为主要建模能力。

## 五十、2026-06-24 connectivity smoke 结果与 base-guided selection 判断

### 本轮新结果

本轮完成了：

```text
1. connectivity_w001 smoke 训练；
2. synthetic val16 mesh eval；
3. 手机 AR dense session 20260624_011933_843 strict visual-hull mesh eval。
```

### synthetic val16：同为 20-step smoke 的公平对比

| run | sparse comp | sparse largest | sparse small64 | black | dark | mesh comp | mesh largest | mesh->target | target->mesh | sparse IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline smoke w0005/ws05 | 1569.4 | 0.0287 | 0.921 | 0.270 | 0.352 | 689.3 | 0.0556 | 0.0513 | 0.0518 | 0.116 |
| connectivity w0.01 smoke | 1550.6 | 0.0305 | 0.914 | 0.256 | 0.333 | 694.6 | 0.0536 | 0.0482 | 0.0524 | 0.118 |

判断：

```text
connectivity loss 在 20-step smoke 上有轻微正向信号：
black / dark / mesh_to_target / sparse_iou 略好。
```

但改善幅度很小，mesh component 没有下降，largest component ratio 也没有实质提升。因此它不能单独解决 Stage2 sparse 碎片化。

再与 s200 旧基线比较时，connectivity smoke 明显不能直接说明更好：

```text
s200 baseline first16:
  sparse comp 1164.5
  sparse largest 0.0907
  sparse small64 0.749
  mesh comp 638.8
  mesh largest 0.106
  chamfer 0.00898

connectivity smoke:
  sparse comp 1550.6
  sparse largest 0.0305
  sparse small64 0.914
  mesh comp 694.6
  mesh largest 0.0536
  chamfer 0.01054
```

所以目前只能说：

```text
connectivity loss 可以保留，但不能作为主解决方案。
```

### 手机 AR strict VH：connectivity smoke 的效果

样本：

```text
20260624_011933_843
```

| run | final coords | final sparse comp | final largest | black | dark | mesh comp | mesh largest | prior_norm_chamfer | extent ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old s200 strict VH | 1163 | 23 | 0.709 | 0.0784 | 0.1466 | 56 | 0.733 | 0.00665 | 0.225 |
| connectivity smoke strict VH | 764 | 20 | 0.819 | 0.0039 | 0.0131 | 20 | 0.780 | 0.01345 | 0.343 |

判断：

```text
connectivity smoke + strict VH 在手机 AR 上显著减少黑色碎片和 mesh 小组件。
```

但代价也明显：

```text
1. coords 从 1163 降到 764；
2. prior_norm_chamfer 从 0.00665 变差到 0.01345；
3. raw sparse component 仍然很高：1334；
4. final visible outside mask event ratio 仍约 0.512。
```

也就是说，这更像是“后处理选出了更干净的一小部分”，而不是 Stage2 raw sparse 本身学到了正常拓扑。

### 是否需要更温和、更有拓扑先验的过滤机制？

结论：需要。

原因是 strict visual-hull / component filter 的性质是事后删除：

```text
Stage2 已经在全局 64^3 voxel 空间里撒点；
filter 再把明显不支持的点删掉。
```

这种方法能减少黑点，但容易带来两个问题：

```text
1. 删得太多，mesh 变薄或缺块；
2. 它只知道 mask/projection/component，不知道 TRELLIS 原生 sparse 拓扑是什么。
```

相比之下，stock/base-guided selection 更符合当前证据：

```text
stock/base sparse 负责形体连通性 / TRELLIS 原生拓扑；
Stage2 sparse 只在 base sparse 附近做 point-prior correction。
```

这不是简单把 Stage2 后处理成 largest component，而是在 top-k 选择前限制候选空间：

```text
1. 先用 stock TRELLIS 生成 base sparse；
2. 对 base sparse 做半径膨胀，得到正常拓扑附近的候选体素；
3. Stage2 logits 只在这些候选体素里排序；
4. 再用轻量 projection/prior filter 做安全检查。
```

这样可以避免 Stage2 在全局 64^3 空间里自由产生孤立高 logit。

### 本轮代码更新

已加入 eval-time base-guided selection：

```text
trellis_point_prior_mv/eval_mesh_frozen_downstream.py
trellis_point_prior_mv/run_real_slam_prior_mesh.py
trellis_point_prior_mv/scripts/run_mesh_frozen_topk_sweep.sh
trellis_point_prior_mv/scripts/run_ar_session_smoke.sh
```

新增参数：

```text
--stage2_base_guidance stock_sparse
--stage2_base_radius 2/4/6
--stage2_base_min_candidates 512
```

对应 wrapper 环境变量：

```text
STAGE2_BASE_GUIDANCE=stock_sparse
STAGE2_BASE_RADIUS=4.0
STAGE2_BASE_MIN_CANDIDATES=512
```

报告会记录：

```text
stage2_base_guidance_enabled
stage2_base_guidance_radius
stage2_base_guidance_base_coord_count
stage2_base_guidance_candidate_count
stage2_base_guidance_candidate_ratio
stage2_base_guidance_fallback_unmasked
stage2_topk_requested
stage2_topk_effective
```

同时 synthetic frozen downstream eval 已补充 sparse topology 字段，不再需要额外脚本间接统计：

```text
sparse_component_count
sparse_largest_component_ratio
sparse_small_component_coord_ratio_lt64
stage2_pre_filter_sparse_component_count
stage2_pre_filter_sparse_largest_component_ratio
stage2_pre_filter_sparse_small_component_coord_ratio_lt64
stage2_final_sparse_component_count
stage2_final_sparse_largest_component_ratio
stage2_final_sparse_small_component_coord_ratio_lt64
sparse_filter_output_count
sparse_filter_total_keep_ratio
```

因此 base-guided sweep 可以直接回答：

```text
Stage2 sparse 本身是否变得更连通；
而不是只从 mesh component 间接判断。
```

### 下一步建议

优先级应调整为：

```text
1. 先跑 base-guided radius sweep；
2. 再跑 base-guided + min_component_size=32；
3. 如果 base-guided 显著减少 sparse/mesh component 且 chamfer 不劣化，再考虑和 connectivity loss 结合；
4. connectivity loss 暂时只作为辅助项，不应直接 s200 放大；
5. slat flow 仍然放后面，等 sparse topology 稳定后再判断是否需要训练。
```

最关键的判断标准：

```text
base-guided 后：
  sparse_component_count 应明显下降；
  stage2_final_sparse_largest_component_ratio 应上升；
  sparse_small_component_coord_ratio_lt64 应下降；
  mesh component 应明显下降；
  artifact_mesh_largest_component_ratio 应上升；
  artifact_black/dark ratio 应下降；
  mesh_to_target / target_to_mesh 不应明显变差；
  phone AR 上 prior_norm_chamfer 不能因为过度贴 base 而明显劣化。
```

radius 的选择不能只看黑色比例。预期情况是：

```text
R=2.0:
  可能最干净，但修正空间太小，容易过度贴 stock/base。

R=4.0:
  最可能是拓扑稳定和 point-prior 修正之间的平衡点。

R=6.0:
  修正空间更大，但可能重新放入 island。
```

因此优先判断 `R=4.0` 是否在 component 与 chamfer 之间最平衡，而不是简单选择 black/dark 最低的半径。

## 五十一、2026-06-24 base-guided sweep 与手机 AR 分支结果

### 本轮运行内容

本轮完成：

```text
1. synthetic val16 base-guided radius sweep: R=2/4/6
2. synthetic val16 base-guided R=4 + min_component_size=32
3. synthetic val64 base-guided R=4 + min_component_size=32
4. phone AR 20260624_011933_843:
   - base-guided only
   - base-guided + min_component_size=32
   - base-guided + light projection/prior filter
   - base-guided + min_component_size=32 + light projection/prior filter
```

### synthetic val16 结果

对照 baseline 是旧 `antioverfill_rank_w0005_ws05_s200_seed42` 的 val64 前 16 个样本。

| run | sparse comp | sparse largest | small64 | black | mesh comp | mesh largest | mesh->target | target->mesh | chamfer | sparse IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no-base baseline | 1164.5 | 0.091 | 0.749 | 0.249 | 638.8 | 0.106 | 0.036 | 0.049 | 0.00898 | 0.173 |
| base R=2 | 74.4 | 0.825 | 0.103 | 0.094 | 132.0 | 0.756 | 0.0456 | 0.1066 | 0.0261 | 0.128 |
| base R=4 | 191.1 | 0.691 | 0.202 | 0.100 | 253.4 | 0.637 | 0.0369 | 0.0905 | 0.0203 | 0.156 |
| base R=6 | 348.2 | 0.584 | 0.308 | 0.141 | 385.9 | 0.485 | 0.0324 | 0.0774 | 0.0157 | 0.167 |
| base R=4 + min32 | 4.9 | 0.832 | 0.029 | 0.032 | 102.8 | 0.689 | 0.0292 | 0.0950 | 0.0204 | 0.160 |

结论：

```text
base-guided selection 明显有效。
```

它把 Stage2 sparse 从上千个 component 压到几十或几百个 component；加入 `min_component_size=32` 后，sparse component 均值降到 `4.9`，small64 ratio 降到 `0.029`，mesh component 降到 `102.8`。

但代价也很清楚：

```text
base-guided 越强，target 覆盖越差。
```

`R=2` 最干净，但 target->mesh 明显变差到 `0.1066`，chamfer 也最差。`R=6` 拓扑更碎，但几何覆盖更接近 baseline。`R=4 + min32` 是“最干净”的方案，但仍有 target->mesh 偏差，说明它可能删掉了部分真实结构。

### synthetic val64 结果

| run | sparse comp | sparse largest | small64 | black | mesh comp | mesh largest | mesh->target | target->mesh | chamfer | sparse IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no-base baseline val64 | 1101.1 | 0.081 | 0.806 | 0.193 | 523.1 | 0.103 | 0.0374 | 0.0484 | 0.00971 | 0.171 |
| base R=4 + min32 val64 | 5.0 | 0.787 | 0.038 | 0.0735 | 79.6 | 0.646 | 0.0406 | 0.1027 | 0.0244 | 0.139 |

val64 复现了 val16 的结论：

```text
base-guided + min32 在拓扑和黑色碎片上非常有效，但覆盖不足。
```

它把 sparse component 从 `1101.1` 降到 `5.0`，mesh component 从 `523.1` 降到 `79.6`，黑色顶点从 `0.193` 降到 `0.0735`。但 target->mesh 从 `0.0484` 变差到 `0.1027`，chamfer 从 `0.00971` 变差到 `0.0244`。

因此它还不能作为默认最终方案，只能作为“强拓扑安全阀”。

### 手机 AR 20260624_011933_843 结果

| run | coords | final comp | largest | black | dark | mesh comp | mesh largest | prior norm chamfer | extent ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old strict VH | 1163 | 23 | 0.709 | 0.078 | 0.147 | 56 | 0.733 | 0.00665 | 0.225 |
| connectivity smoke strict VH | 764 | 20 | 0.819 | 0.0039 | 0.013 | 20 | 0.780 | 0.01345 | 0.343 |
| base R=4 only | 4096 | 126 | 0.723 | 0.550 | 0.594 | 228 | 0.534 | 0.03865 | 0.344 |
| base R=4 + min32 | 3547 | 6 | 0.835 | 0.477 | 0.521 | 107 | 0.649 | 0.03902 | 0.371 |
| base R=4 + lightproj | 2253 | 23 | 0.973 | 0.038 | 0.077 | 43 | 0.942 | 0.01423 | 0.624 |
| base R=4 + min32 + lightproj | 2236 | 14 | 0.980 | 0.041 | 0.084 | 53 | 0.933 | 0.01668 | 0.656 |

结论：

```text
phone AR 上，base-guided only 不能用。
```

它虽然把 sparse 限在 stock 邻域，但没有 mask/projection 约束时，会生成大量颜色很差的 unsupported surface，black ratio 达到 `0.55`。

真正有效的是：

```text
base-guided + light projection/prior filter
```

该方案相比 old strict VH：

```text
1. black 从 0.078 降到 0.038；
2. mesh component 从 56 降到 43；
3. mesh largest 从 0.733 升到 0.942；
4. extent ratio 从 0.225 升到 0.624，几何不再那么薄。
```

但它的 prior norm chamfer 从 `0.00665` 变差到 `0.01423`，说明它更像是得到一个拓扑更合理、但不如 strict VH 贴近 AR prior 的 mesh。

`min_component_size=32` 在 phone AR lightproj 后没有带来收益：

```text
base R=4 + lightproj:
  black 0.038, mesh comp 43, prior_norm_chamfer 0.01423

base R=4 + min32 + lightproj:
  black 0.041, mesh comp 53, prior_norm_chamfer 0.01668
```

因此 phone AR 当前不建议默认叠加 min32。

### 当前判断

现在可以把路线分成两类：

```text
1. synthetic / 有 GT 的评测：
   base-guided + min_component_size=32 是很强的拓扑安全阀，
   但会牺牲 target coverage。

2. phone AR real session：
   base-guided + light projection/prior filter 是当前更平衡的方案，
   比 strict VH 更不容易删薄，黑色碎片也明显减少。
```

这说明最合理的方向不是继续强化 strict VH，也不是单纯训练 connectivity loss，而是：

```text
stock/base sparse 提供拓扑候选空间；
Stage2 logits 在候选空间里排序；
AR point prior + mask projection 做轻量 sanity check。
```

### 下一步建议

第一优先级：补 `R=6 + light projection/prior filter`，因为 synthetic 里 R=6 的 coverage 明显更好，而 phone AR 里 lightproj 可以抑制 R=6 可能带来的碎片。

第二优先级：补 `R=4/R=6 + lightproj` 的四组手机 AR session，而不是只看 dense session。

第三优先级：若 R=6 lightproj 在手机 AR 上能降低 prior_norm_chamfer，同时保持 black/mesh component 不爆炸，则把它作为下一阶段默认 eval-time sparse selection。

第四优先级：暂时不要基于 base-guided + min32 做长训练。它太像强安全阀，容易把模型推向低覆盖。

## 五十二、2026-06-24 R4/R6 projection-only 0.15 与 stock union 判断

### 本轮测试内容

本轮补了两组手机 AR dense session `20260624_011933_843` 的 projection-only 过滤：

```text
1. base-guided R=4 + projection_support, support_ratio=0.15
2. base-guided R=6 + projection_support, support_ratio=0.15
```

这里刻意没有使用 `prior_radius`，目的是判断：

```text
只靠 mask projection support 是否足够约束 Stage2 sparse；
R=6 是否能在不明显增加碎片的情况下获得更好覆盖；
是否应该继续做 stock backbone / union。
```

### 关键结果

| run | coords | final comp | largest | outside event | support ratio | black | dark | mesh comp | mesh largest | prior norm chamfer | extent ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock_sparse | 3097 | - | - | - | - | 0.007 | 0.018 | 2 | 0.988 | 0.0629 | 0.201 |
| old strict VH | 1163 | 23 | 0.709 | 0.525 | 0.466 | 0.078 | 0.147 | 56 | 0.733 | 0.00665 | 0.225 |
| base R=4 + lightproj | 2253 | 23 | 0.973 | 0.624 | 0.389 | 0.038 | 0.077 | 43 | 0.942 | 0.01423 | 0.624 |
| base R=4 + proj015 | 1980 | 32 | 0.956 | 0.571 | 0.443 | 0.044 | 0.074 | 34 | 0.919 | 0.06048 | 0.446 |
| base R=6 + proj015 | 1946 | 48 | 0.906 | 0.575 | 0.440 | 0.203 | 0.249 | 61 | 0.924 | 0.06642 | 0.542 |

补充：`proj015` 的候选空间来自 stock sparse 邻域。

| run | base radius | base coords | candidate coords | candidate ratio | filter keep |
|---|---:|---:|---:|---:|---:|
| R=4 + proj015 | 4.0 | 3097 | 22969 | 0.0876 | 0.483 |
| R=6 + proj015 | 6.0 | 3097 | 33529 | 0.1279 | 0.475 |
| R=4 + lightproj | 4.0 | 3097 | 22969 | 0.0876 | 0.550 |

### 分析

第一，`projection_support=0.15` 确实让 sparse 更符合 mask projection。
相比 `base R=4 + lightproj`，`R=4 + proj015` 的 `outside event` 从 `0.624` 降到 `0.571`，visible support ratio 从 `0.389` 升到 `0.443`。

但这个改善没有转化为更好的最终 mesh。`R=4 + proj015` 的 `prior norm chamfer` 是 `0.06048`，几乎退回到 stock_sparse 的 `0.0629`；而 `R=4 + lightproj` 是 `0.01423`。这说明只靠 projection support 不够，必须保留 `prior_radius` 这一类“靠近 AR point prior”的约束，否则 Stage2 会在 stock 邻域里选出 projection 上看起来合理、但和真实 AR 点云距离很远的结构。

第二，`R=6 + proj015` 当前不值得作为默认。
R=6 把候选空间从 `22969` 扩到 `33529`，理论上给了 Stage2 更多修正空间；但结果是 black ratio 从 R4 的 `0.044` 升到 `0.203`，mesh component 从 `34` 升到 `61`，prior norm chamfer 也变差到 `0.06642`。也就是说，在没有 `prior_radius` 的情况下，R=6 更容易重新放入 OOD / unsupported surface。

第三，当前最平衡的手机 AR 分支仍然是：

```text
base R=4 + prior_radius + light projection_support
```

它的 prior norm chamfer 明显优于 projection-only，black/dark 又明显低于 base-guided only，同时 extent ratio 比 strict VH 更合理。

### 是否需要 stock backbone / union

需要，但不建议直接把 raw union 当最终默认。

现在 stock_sparse 和 Stage2 的优缺点非常互补：

```text
stock_sparse:
  拓扑、颜色、连通性很好；
  black ratio 只有约 0.007；
  mesh component 只有 2；
  但 prior norm chamfer 很差，extent ratio 也偏薄。

Stage2 lightproj:
  明显更贴近 AR point prior；
  extent ratio 更合理；
  但仍有一定黑色区域和小组件风险。
```

因此后续值得做 stock backbone / union，目标不是简单“多加点”，而是：

```text
stock_sparse 提供 TRELLIS 原生拓扑骨架；
Stage2 filtered sparse 提供 AR point prior correction；
union 后再做轻量 component / projection sanity check。
```

两个 union 分支的优先级如下：

```text
第一优先级：
  stock_sparse ∪ R4 lightproj

原因：
  R4 lightproj 已经证明能兼顾 prior alignment 和低黑色碎片；
  union 有机会补回 stock 的稳定拓扑和颜色区域。

第二优先级：
  stock_sparse ∪ R4 projection-only

原因：
  projection-only 的黑色比例还可以，但 prior chamfer 几乎退回 stock；
  这组更适合作诊断“stock backbone 能否单独拯救颜色/连通性”，不适合作默认候选。
```

暂时不建议优先做：

```text
stock_sparse ∪ R6 projection-only
```

因为 R6 projection-only 已经出现明显黑色和碎片回升。

### 下一步建议

第一优先级：实现 eval-time union sparse mode，不重训。建议先支持：

```text
stage2_union_stock:
  final_coords = unique(stock_sparse ∪ filtered_stage2_coords)
```

并记录：

```text
union_coord_count
stock_only_count
stage2_only_count
intersection_count
artifact_black_vertex_ratio
artifact_mesh_component_count
prior_norm_chamfer
extent_ratio
```

第二优先级：只在 dense phone AR session 上先跑两个 smoke：

```text
1. stock_sparse ∪ R4 lightproj
2. stock_sparse ∪ R4 projection-only
```

如果 `stock ∪ R4 lightproj` 能保持 `prior_norm_chamfer < 0.02`，同时把 black ratio 压到接近 stock 或低于 `0.03`，它就可以成为真实 AR session 的默认 eval-time sparse selection。

第三优先级：不要现在训练 slat flow。
当前问题仍然主要是 sparse selection/topology 与 unsupported surface 控制。只有当 union / base-guided / projection sanity check 后，sparse topology 已经稳定但颜色仍失败，才需要训练 slat/texture downstream。

## 五十三、2026-06-26 eval-time stock union 实测结果

### 本轮测试内容

本轮完成三组 eval-time union：

```text
1. 手机 AR dense session:
   stock_sparse ∪ R4 lightproj

2. 手机 AR dense session:
   stock_sparse ∪ R4 projection-only 0.15

3. synthetic val16:
   stock_sparse ∪ R4 min_component_size=32
```

这里的 union 是最直接的：

```text
final_coords = unique(stock_sparse ∪ filtered_stage2_coords)
```

### 手机 AR dense session 结果

样本：`20260624_011933_843`

| run | coords | black | dark | mesh comp | mesh largest | prior norm chamfer | extent ratio | sparse comp | sparse largest | outside event | support ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock_sparse | 3097 | 0.007 | 0.018 | 2 | 0.988 | 0.0615 | 0.201 | - | - | - | - |
| R4 lightproj | 2253 | 0.038 | 0.077 | 43 | 0.942 | 0.01423 | 0.624 | 23 | 0.973 | 0.624 | 0.389 |
| stock ∪ R4 lightproj | 4961 | 0.0011 | 0.0129 | 9 | 0.988 | 0.04730 | 0.260 | 9 | 0.998 | 0.716 | 0.268 |
| R4 proj015 | 1980 | 0.044 | 0.074 | 34 | 0.919 | 0.06048 | 0.446 | 32 | 0.956 | 0.571 | 0.443 |
| stock ∪ R4 proj015 | 4759 | 0.00013 | 0.0032 | 12 | 0.994 | 0.04783 | 0.272 | 15 | 0.995 | 0.701 | 0.280 |

union 统计：

| run | stock count | stage2 count | intersection | stock-only | stage2-only | union count | intersection / stage2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| stock ∪ R4 lightproj | 3097 | 2253 | 389 | 2708 | 1864 | 4961 | 0.173 |
| stock ∪ R4 proj015 | 3097 | 1980 | 318 | 2779 | 1662 | 4759 | 0.161 |

### 手机 AR 结论

raw union 成功解决了黑色碎片和拓扑破碎：

```text
stock ∪ R4 lightproj:
  black: 0.038 -> 0.0011
  mesh comp: 43 -> 9
  mesh largest: 0.942 -> 0.988

stock ∪ R4 proj015:
  black: 0.044 -> 0.00013
  mesh comp: 34 -> 12
  mesh largest: 0.919 -> 0.994
```

但它没有保持 Stage2 对 AR point prior 的贴合：

```text
R4 lightproj prior_norm_chamfer:
  0.01423 -> 0.04730

R4 proj015 prior_norm_chamfer:
  0.06048 -> 0.04783
```

也就是说，`stock ∪ R4 lightproj` 虽然颜色和连通性很好，但几何又明显被 stock backbone 拉回去了。它比 stock 稍微更贴 prior，但远不如单独的 R4 lightproj。

原因可以从 union 统计看出来：

```text
R4 lightproj 只有 389 / 2253 个点与 stock 重合；
union 里 stock-only 有 2708 个点，stage2-only 有 1864 个点。
```

这说明 stock sparse 和 Stage2 filtered sparse 的空间重合很低。直接 union 后，slat/mesh decoder 很可能更偏向 stock 的原生拓扑和颜色分布，因此 artifact 少了，但 AR correction 被稀释了。

### synthetic val16 结果

| run | sparse IoU | recall | precision | sparse comp | sparse largest | small64 | black | dark | mesh comp | mesh largest | mesh->target | target->mesh | chamfer | extent |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock_sparse | 0.0466 | 0.0779 | 0.1226 | 1.69 | 0.973 | 0.0019 | 0.0498 | 0.0715 | 5.75 | 0.911 | 0.0856 | 0.1250 | 0.0397 | 0.382 |
| R4 min32 | 0.1596 | 0.1893 | 0.4864 | 4.94 | 0.832 | 0.0286 | 0.0324 | 0.1220 | 102.8 | 0.689 | 0.0292 | 0.0950 | 0.0204 | 0.473 |
| stock ∪ R4 min32 | 0.1264 | 0.2254 | 0.2294 | 1.56 | 0.998 | 0.0018 | 0.1112 | 0.1856 | 103.4 | 0.901 | 0.0772 | 0.0982 | 0.0308 | 0.459 |
| target_sparse | 1.000 | 1.000 | 1.000 | 2.19 | 0.9998 | 0.0002 | 0.1162 | 0.2284 | 17.5 | 0.942 | 0.0110 | 0.0147 | 0.00079 | 0.486 |

synthetic union 统计：

```text
stock count mean: 6525
stage2 count mean: 3860
intersection mean: 893
union count mean: 9492
intersection / stage2: 0.228
```

### synthetic 结论

synthetic 上 raw union 也不是更优方案：

```text
R4 min32 chamfer: 0.0204
stock ∪ R4 min32 chamfer: 0.0308

R4 min32 mesh->target: 0.0292
stock ∪ R4 min32 mesh->target: 0.0772

R4 min32 precision: 0.486
stock ∪ R4 min32 precision: 0.229
```

union 的 sparse topology 很干净，`largest=0.998`，但 precision 明显下降，mesh 指标也退化。它更像是把 Stage2 的 correction 放进 stock 的完整拓扑里，结果变得更连通，但不再足够贴近 target。

### 总结判断

当前可以确认：

```text
raw stock union 是强 artifact repair，不是好的最终 sparse selection。
```

它适合证明：

```text
1. 黑色碎片和 mesh 小组件确实主要来自 Stage2 sparse 的 OOD / unsupported islands；
2. stock sparse 的原生拓扑能显著稳定 frozen slat/mesh；
3. 但直接 union 会稀释 AR point prior correction。
```

因此，下一步不应该继续扩大 raw union，也不应该立刻训练 slat flow。

### 下一步建议

第一优先级：实现 `union candidate rerank`，而不是 raw union。

建议形式：

```text
candidate_set = stock_sparse ∪ filtered_stage2_coords
final_coords = topk(candidate_set, score)
```

score 不应只用是否属于 stock，而应组合：

```text
score =
  Stage2 logit
  + stock topology bonus
  + prior_radius bonus
  + projection_support bonus
  - outside / unsupported penalty
```

这样可以让 stock 提供候选空间和拓扑先验，但最终点数、几何位置仍由 Stage2 / AR prior 排序决定。

第二优先级：实现 `stock anchor + Stage2 correction`。

不要把所有 stock 点都并入 final sparse，而是保留一部分高置信 stock anchor：

```text
final = selected_stock_anchor ∪ selected_stage2_correction
```

例如：

```text
stock anchor: stock 中靠近 prior 或 projection support 较高的点
stage2 correction: R4 lightproj 中 Stage2 logit 高、prior/projection support 高的点
```

第三优先级：补一个更小的 union cap。

当前 union count 接近 5000，手机 AR 中明显被 stock 拖回。可以测试：

```text
stock ∪ R4 lightproj 后再 topk 3000 / 3500 / 4096
```

但这个 top-k 必须基于 score rerank，不能随机裁剪。

第四优先级：暂时不要基于 raw union 做长训练，也不要直接训练 slat flow。

raw union 的 artifact 很少，但它的几何目标不对。如果拿它训练下游，很可能让模型学到“更像 stock、但不贴 AR prior”的方向。

## 五十四、2026-06-26 停止 coords 后处理，转向 latent inpainting 的路线

### 当前阶段判断

经过 fixed top-k、strict visual-hull、base-guided、min component、projection support、stock union 等多轮测试，结论已经足够明确：

```text
coords 层面的后处理可以修补 artifact，
但不能从根本上解决 point-prior sparse 的结构生成问题。
```

具体表现是：

```text
1. Stage2 raw coords 更贴近 AR prior，但碎片多、黑色 mesh 多；
2. strict VH / projection filter 能减少 unsupported coords，但容易删薄；
3. stock union 能强力修复黑色和连通性，但几何又被 stock 拉回；
4. synthetic 上 union 也会降低 precision 和 Chamfer 表现。
```

因此下一步主线不再继续修改最终 sparse coords：

```text
不再把 filter / union / top-k rerank 作为主要研究方向。
```

这些工具后续只保留为：

```text
1. 诊断 sparse 输出是否碎；
2. 分析 unsupported surface；
3. 做必要的安全阀 ablation。
```

### 为什么要转 latent

从 Points-to-3D 的角度，正确范式不是：

```text
point prior -> condition -> decoder logits -> top-k coords 修修补补
```

而是：

```text
point prior -> voxelize -> SS encoder -> q_vis
mask -> m_s
q_comb = m_s * q_vis + (1 - m_s) * noise
x_inp = concat(q_comb, m_s)
inpainting flow -> q_pred
decoder(q_pred) -> sparse structure
```

也就是说，point prior 应该进入 SS latent 初始化和 inpainting dynamics，而不是只在最终 coords 层做约束。

当前碎片的根本原因也在这里：

```text
我们让模型在 64^3 coords/logits 空间里排序，
但没有让 unknown latent 在 16^3 SS latent 空间里学会从 known latent 连续补全。
```

coords 后处理只能处理结果，不能让模型学会结构连续性。

### 新主线：Stage2-Latent Inpainting

下一阶段建议定义一个新的主线版本：

```text
pointprior_stage2_latent_v1
```

目标：

```text
直接训练和采样 SS latent q，
让 observed latent 成为生成起点和硬约束，
unknown region 由 flow inpaint，
最终不再对 coords 做 union/filter/rerank。
```

### 数据构建

保留当前 PixalV9 visible prior 构建，但输出重点从 coords 迁移到 latent：

```text
input:
  prior_coords / prior_conf
  target sparse latent q_gt
  target_coords 仅作为评测 GT

build:
  M_vis = voxelize(prior_coords)
  q_vis = SparseStructureEncoder(M_vis)
  m_s = downsample / project prior support 到 16^3 latent grid
  q_comb = m_s * q_vis + (1 - m_s) * noise

output npz:
  q_vis
  m_s
  q_gt
  prior_coords
  target_coords
```

注意：

```text
target_coords 后续只用于 metric；
不参与最终 sparse coords 的后处理。
```

### 模型输入

第一版不要再只把 point prior 当 condition token。建议直接改成：

```text
x_t_known = q_comb at timestep t
model input = concat(x_t_known, m_s, optional confidence)
condition = image condition
target = q_gt
```

如果不想大改 TRELLIS flow 主干，可以先做最小改法：

```text
1. 保留 LoRA / adapter；
2. 增加 latent input adapter，把 [q_comb, mask, confidence] 投影到 flow 输入通道；
3. 每一步 sampling 对 known latent 做 re-injection；
4. unknown latent 按 flow 正常更新。
```

关键点：

```text
known region 的 q_vis 不能再被 confidence 缩小两次；
strict oracle 下 known_weight 应优先使用 mask，而不是 density confidence。
```

### 训练目标

第一阶段训练目标保持简单：

```text
L = full_latent_flow_loss
  + lambda_known * known_latent_consistency
  + lambda_unknown * unknown_completion_loss
  + lambda_boundary * boundary_smoothness_loss
```

这里的 `boundary_smoothness_loss` 不再是 coords 连通块后处理，而是在 latent mask 边界上约束：

```text
q_pred 在 known / unknown 边界不要出现突变；
unknown latent 应该从 q_vis 周围连续扩展。
```

这比 coords 的 `min_component_size` 更接近论文里的 boundary refinement 思路。

### 采样策略

按 Points-to-3D 的思路改成两阶段：

```text
Stage A: structural inpainting
  - known region hard reinjection
  - unknown region flow sampling

Stage B: boundary refinement
  - 只在 boundary band / unknown-near-known 区域加少量 noise
  - known core 继续固定
  - 目标是修复 known/unknown 交界 holes 和断裂
```

暂定配置：

```text
total steps: 12 或 20
inpaint steps: 8 / 14
refine steps: 4 / 6
known clamp: always on for known core
boundary clamp: softer
```

### 评测方式

下一阶段 eval 不再以 coords 后处理为主，而是看 latent-inpainting 是否自然产生正常 sparse：

```text
1. decoded sparse component count
2. largest component ratio
3. small64 ratio
4. known-region recall / precision
5. unknown-region completion
6. target Chamfer / mesh eval
7. phone AR black/dark ratio
```

允许统计 coords topology，但只作为诊断：

```text
不再用统计结果反过来修改 coords。
```

### 是否还需要 stock sparse

stock sparse 后续不建议再 raw union。更合理的使用方式是 latent 级别的 soft prior：

```text
stock_sparse -> encode -> q_stock
q_init = m_vis * q_vis
       + m_stock_soft * alpha * q_stock
       + remaining noise
```

也可以只把 stock latent 作为 condition，不进入最终 coords：

```text
condition = image condition + point latent condition + optional stock latent condition
```

这样 stock 提供全局 TRELLIS 拓扑倾向，但不会像 raw union 那样把最终几何硬拉回 stock。

### 最小实验顺序

第一步：实现 latent manifest / latent loader。

```text
build_point_prior_dataset.py 增加 q_vis / m_s / q_gt 保存；
或新增 build_latent_inpaint_dataset.py，避免污染旧流程。
```

第二步：实现 `train_sparse_latent_inpaint_stage3.py`。

```text
不再训练 coords ranking；
直接训练 q_comb -> q_gt 的 latent inpainting。
```

第三步：实现 `eval_sparse_latent_inpaint_stage3.py`。

```text
输出 decoded sparse coords；
不做 filter/union/rerank；
只记录 topology / mesh / prior alignment。
```

第四步：先跑 PixalV9 strict oracle。

```text
point_count=1500
views=8
dropout=0
outlier=0
jitter=0
known_use_confidence=0
```

目标：

```text
证明 latent inpainting 在 oracle visible prior 下不会碎。
```

第五步：再跑 noisy prior。

```text
dropout <= 0.25
outlier=0.03
jitter=1
```

目标：

```text
验证真实 AR/SLAM prior 的鲁棒性。
```

第六步：最后接手机 AR session smoke。

```text
不做 coords filter；
只看 latent path 是否能自然减少 black fragments。
```

### 是否现在训练 slat flow

暂时不建议。

原因：

```text
当前问题还在 sparse / SS latent 阶段。
如果 q_pred 自然 decoded 后仍碎，训练 slat 只是在适配错误 sparse；
如果 latent inpainting 后 sparse 已经连通，再评估 frozen slat 是否仍有颜色问题。
```

只有当：

```text
1. latent path 的 sparse topology 已经接近 stock/target；
2. known-region / unknown-region sparse 指标正常；
3. mesh 仍然黑色或 texture 错；
```

才进入 slat flow / texture adapter。

### 当前推荐结论

下一步主线应改为：

```text
Point prior -> SS latent inpainting -> decoded sparse -> frozen slat/mesh eval
```

而不是：

```text
Point prior -> Stage2 logits -> coords post-process -> frozen slat/mesh
```

当前 coords 层实验的价值已经完成：

```text
它证明了 point prior 有用；
也证明了碎片主要来自 sparse topology / unsupported regions；
同时证明 raw stock topology 可以修 artifact，但会牺牲 AR prior alignment。
```

因此继续在 coords 层调 filter / union 的边际收益已经很低，应该转向 latent formulation。
