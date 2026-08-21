# AR SS Flow：局部多视图 Correspondence 分支实验记录

> 更新时间：2026-07-14  
> 仓库：`/home/zjr/Tracker`  
> 代码目录：`/home/zjr/Tracker/ar_ss_flow`  
> Lifting cache：`/data/ar_ss_flow_pose_lifting_overfit64_v1_20260714/manifest.json`  
> 训练对象：indices `0-15`  
> Fresh-object 审计：indices `16-63`  
> 当前结论：**C1 correspondence detector 通过；C1 learned weighted aggregator 未通过；C2 尚未正式开启。**

---

## 0. 执行摘要

本分支的原始目标是先完全脱离 ReconViaGen / SS Flow，验证 Pixal3D 多视图视觉特征是否包含可泛化的局部 image-pose correspondence 信号；只有输入端 C1 通过，才允许把 correspondence confidence 接入 C2 的 SS velocity residual gate。

截至当前，经历了三次主要实现迭代：

| 版本 | 核心实现 | 主要结果 | 阶段结论 |
|---|---|---|---|
| C1 v1 | same-voxel mean aggregation + held-out reconstruction + score heads | 三 seed 主门失败；后续发现 fixed-target、采样调度和 view-count 协议存在严重问题 | **实验结果不可作为原假设的干净检验** |
| C1 v2 | fixed held-out target、source-only pose corruption、对象/负例解耦、paired source-count 审计 | 3-source 在训练对象和 fresh 对象上均失败，且增加 source 后系统性恶化 | **干净否定 early mean aggregation** |
| C1 v3 | pairwise-before-aggregation；先计算 view-pair confidence，再聚合 | 三 seed fresh-object 重建与 pairwise confidence 均稳定为正；visual-zero/shuffle 后崩溃；geometry-pair-off 几乎不变 | **视觉 correspondence detector 成立** |
| C1 v3 ablation | full vs visual-zero / visual-shuffle / geometry-off / uniform-pairwise | learned pairwise 与 uniform aggregation 的重建差异极小 | **learned confidence 适合外部 gate，不足以证明内部加权聚合有收益** |

当前最重要的科学结论是：

1. **Pixal3D per-view visual patch features 中确实存在可泛化、依赖正确 image-pose binding 的局部对应信号。**
2. 该信号必须在过早的跨视图平均之前被提取；v2 的 mean aggregation 会将错误 pose 带来的差异洗平。
3. v3 学到的 **absolute pairwise confidence** 能稳定区分 correct/wrong pose，但作为归一化 source 权重时，其整体尺度大多被分母抵消，因此没有显著改善当前 reconstruction probe。
4. 下一步不应继续优化 C1 reconstruction head，而应把 visual-only correspondence confidence 校准成 C2 可直接使用的 residual gate。

---

## 1. 原始研究问题与阶段设计

### 1.1 核心研究问题

输入为多视图 Pixal3D / VGGT + DINO patch feature、mask、depth、confidence、相机内外参。目标是判断：

```text
同一对象的多张图像
  + 正确 image-pose 绑定
是否能够在同一 16^3 voxel 上形成
比错误 pose binding 更一致的视觉证据？
```

不使用 mesh / occupancy GT，不加载或训练 ReconViaGen Flow，先做纯输入端机制审计。

### 1.2 C1：输入端 correspondence 审计

每个 `16^3` voxel 保留 per-view：

- VGGT + DINO patch feature；
- valid / mask support；
- depth confidence / depth consistency / normalized residual；
- camera ray 与 normalized camera depth。

主负例：

```text
pose_cyclic1
pose_cyclic2
pose_reverse
```

辅助 easy negative：

```text
cross_sample
```

主判据是在 common support voxel 上：

```text
held-out reconstruction error(correct)
<
held-out reconstruction error(wrong)
```

### 1.3 C2：correspondence-gated SS residual

最初计划在 C1 checkpoint 冻结后构建：

```text
correspondence confidence
  -> visual / physical evidence gate
  -> same-voxel zero-init SS velocity residual
```

C2 必须保持：

- stock Flow 冻结；
- adapter same-voxel；
- zero-init；
- physical-off / null-present 路径 bit-exact stock；
- correct gate 优于 stock 与 hard wrong；
- wrong gate 回到 stock，而不是破坏 stock。

截至本记录，**C2 代码原型已经存在，但尚未用 v3 correspondence detector 正式执行机制门。**

---

## 2. 代码与实验版本时间线

### 2.1 初始 C1/C2 分支（v1）

主要文件：

```text
ar_ss_flow/correspondence_lifting.py
ar_ss_flow/train_heldout_correspondence.py
ar_ss_flow/eval_heldout_correspondence.py
ar_ss_flow/summarize_heldout_correspondence.py
ar_ss_flow/correspondence_gated_flow.py
ar_ss_flow/train_correspondence_gated_ss_flow.py
ar_ss_flow/eval_correspondence_gated_ss_flow.py
ar_ss_flow/summarize_correspondence_gated_multiseed.py
ar_ss_flow/test_correspondence_lifting.py
```

原始交付物：

```text
ar_ss_flow_correspondence_v1.zip
ar_ss_flow_correspondence_v1.patch
ar_ss_flow_correspondence_v1_fixed.zip
ar_ss_flow_correspondence_import_fix.patch
correspondence分支说明.md
heldout_correspondence_命令说明.txt
```

### 2.2 C1 v2：fixed-target / protocol repair

主要改动规模：

```text
correspondence_lifting.py         +52 / -7
train_heldout_correspondence.py   +80 / -4
eval_heldout_correspondence.py   +126 / -76
```

交付物：

```text
ar_ss_flow_c1_fixed_target_v2_source_only.zip
c1_fixed_target_v2_git.patch
```

注意：v2 的 checkpoint `format` 字符串仍沿用 `ar_ss_flow.local_voxel_correspondence.v1`；这里的“v2”指实验协议与实现修复版，不是 checkpoint format 名称。

### 2.3 C1 v3：pairwise-before-aggregation

主要改动规模：

```text
correspondence_lifting.py         +235 / -103
train_heldout_correspondence.py    +41 / -7
eval_heldout_correspondence.py     +95 / -31
```

checkpoint format 改为：

```text
ar_ss_flow.local_voxel_correspondence.pairwise_v3
```

与 v1/v2 checkpoint 不兼容，必须重新训练。

交付物：

```text
ar_ss_flow_c1_pairwise_v3_source_only.zip
c1_pairwise_v3_next_tools.zip
C1_pairwise_v3_后续完整命令.txt
```

新增审计工具：

```text
eval_pairwise_ablation.py
summarize_pairwise_ablation.py
summarize_pairwise_v3_multiseed.py
```

---

## 3. C1 v1：初始 mean-aggregation 实现

### 3.1 结构

v1 的核心流程近似为：

```text
per-view visual embedding
+ per-view geometry embedding
        ↓
physical-weighted source mean
        ↓
linear reconstruction
        ↓
held-out 3x3 patch neighborhood matching
```

另有两个诊断/分类 head：

```text
score_head
geometry_only_head
```

### 3.2 初始训练与评估协议

- train：indices `0-15`；
- fresh eval：indices `16-63`；
- train seeds：42 / 43 / 44；
- 300 optimizer steps；
- negative modes：三类 pose hard negative + cross-sample；
- 主门：4-view；2-view 做趋势诊断；
- 每类 hard negative 要求 object-balanced win rate ≥ 65%。

### 3.3 初始三 seed 结果

4-view 主结果：

| Seed | Mode | Object win | Advantage mean | Advantage median | 结论 |
|---:|---|---:|---:|---:|---|
| 42 | cyclic1 | 51.85% | +0.00690 | +0.00115 | fail |
| 42 | cyclic2 | 53.85% | +0.00526 | +0.00624 | fail |
| 42 | reverse | 51.85% | +0.00655 | +0.00099 | fail |
| 43 | cyclic1 | 48.15% | -0.00058 | -0.00031 | fail |
| 43 | cyclic2 | 57.69% | +0.00452 | +0.00274 | fail |
| 43 | reverse | 55.56% | +0.00059 | +0.00298 | fail |
| 44 | cyclic1 | 40.74% | +0.00517 | -0.00336 | fail |
| 44 | cyclic2 | 69.23% | +0.00532 | +0.00495 | 单项通过 |
| 44 | reverse | 48.15% | +0.00510 | -0.00215 | fail |

三 seed 中仅 seed44 / cyclic2 单项通过，整体 C1 fail。

同时存在明显现象：

- visual score 与 geometry score 对 wrong pose 的分类胜率很高；
- 但实际 held-out reconstruction 优势很弱；
- cross-sample 远比同对象 pose negative 容易；
- 2-view apparent advantage 明显高于 4-view。

### 3.4 v1 代码/协议审计发现的问题

#### 问题 1：wrong 分支改变了 held-out target

wrong evidence 使用错误 extrinsics 重新构建全部 projection grid，而 `evaluate_heldout()` 又从 wrong evidence 的 held-out `patch_grid` 采样 target。

因此比较实际上是：

```text
correct source -> correct target patch
vs
wrong source -> wrong target patch
```

而不是固定同一个 held-out target。

#### 问题 2：sample 与 negative mode 确定性绑定

训练循环同时使用：

```python
sample_index = (attempt - 1) % len(dataset)
mode = negative_modes[(attempt - 1) % len(negative_modes)]
```

16 个对象、4 种负例导致每个对象长期只看到固定一种负例，形成 object/mode confounding。

#### 问题 3：2-view 实际只有一个 source

leave-one-out 后：

```text
2 total views = 1 source + 1 held-out
```

因此并不是 multi-source consensus 任务。

#### 问题 4：2-view 下三种 pose negative 完全相同

两个 source/view 的 cyclic1、cyclic2、reverse 都退化成同一个 swap，因此三行结果重复，不能视为三种独立 hard negative。

#### 问题 5：view-count trend 非等价

原协议比较的是不同总 view 数、不同对象子集、不同 source 数和不同扰动强度的聚合均值，并非同对象、同 held-out、nested source set 的配对趋势。

#### 问题 6：所谓 visual score 并非 visual-only

`score_head` 同时接收 reconstruction、target、visual interactions 和包含 support / geometry 的 summary。`geometry_only_head` 又是另一套独立网络，因此两者 raw logit 相减没有严格因果含义。

#### 问题 7：cross-sample 过强且过量

cross-sample 替换另一个对象的视觉特征，但保留当前对象 geometry/depth/mask，形成很容易识别的语义不匹配；在 300 step 中占 100 step，容易主导学习。

### 3.5 v1 阶段结论

```text
运行与数值稳定性：PASS
初始主门：FAIL
实验协议有效性：FAIL
原始 correspondence 假设：未被干净检验
C2：保持关闭
```

v1 的价值主要是暴露了协议错误，而不是给出可靠的科学否定。

---

## 4. C1 v2：fixed-target 与协议修复

### 4.1 主要代码改动

v2 修复了：

1. **固定 held-out target**：correct 与 wrong 共享同一 held-out visual target、K/T 和 target grid；
2. **只扰动 source pose**：held-out extrinsic 永远保持 correct；
3. **对象与负例解耦**：构造 balanced object × mode 调度；
4. **cross-sample 默认关闭或严格限额**；
5. **source-count 语义重定义**：
   - 3 total views = 2 source + 1 held-out；
   - 4 total views = 3 source + 1 held-out；
6. **paired view trend**：同对象、同 mode、nested source count 比较；
7. 新增 fixed-target 单测，单测总数从 4 增至 5。

### 4.2 训练配置

```text
run: corr_heldout_fixedtarget_v2_s150_seed42
seed: 42
train indices: 0-15
steps: 150
negative modes: cyclic1 / cyclic2 / reverse
cross_sample_fraction: 0
```

训练完成：

```text
finite = true
completed_steps = 150
attempts = 200
skipped = 50
mode counts = 49 / 50 / 51
```

说明负例调度已经基本均衡。

### 4.3 训练对象评估

3-source 主门（12 个 eligible 对象）：

| Mode | Object win | Advantage mean | Median |
|---|---:|---:|---:|
| cyclic1 | 33.33% | -0.00123 | -0.00582 |
| cyclic2 | 33.33% | +0.00473 | -0.00486 |
| reverse | 33.33% | +0.00287 | -0.00395 |

2-source 诊断：三种 mode 因两个 source swap 而相同：

```text
object win = 58.33%
mean advantage = +0.03960
median = +0.01704
```

从 2-source 增加到 3-source 后，paired advantage 平均下降：

```text
cyclic1: -0.04083
cyclic2: -0.03487
reverse: -0.03673
```

### 4.4 Fresh-object 评估

3-source 主门：

| Mode | Eligible objects | Object win | Advantage mean | Median |
|---|---:|---:|---:|---:|
| cyclic1 | 26 | 23.08% | -0.00098 | -0.00312 |
| cyclic2 | 25 | 24.00% | -0.00526 | -0.00612 |
| reverse | 26 | 26.92% | -0.00518 | -0.00295 |

2-source 诊断：

```text
object win = 57.69%
mean advantage ≈ +0.0190
median ≈ +0.0010
```

从 2-source 增加到 3-source 后，paired advantage 再次系统下降约 `-0.020 ~ -0.025`。

### 4.5 v2 结果解释

v2 已经排除了 fixed-target 和调度错误，因此失败是结构性的：

```text
per-view feature
  -> 过早均值聚合
  -> reconstruction
```

wrong pose 下，不同表面位置的同对象特征被平均后，可能形成稳定的 object/local semantic mean；source 越多，这种错误平均越平滑，导致 wrong aggregation 有时反而更容易接近 held-out embedding。

训练日志同时显示：

```text
correct_error ≈ wrong_error
但 score advantage 逐渐为正
```

说明 score head 能利用 geometry/support 差异识别 wrong pose，但实际聚合内容没有变得更正确。

### 4.6 v2 阶段结论

```text
fixed target：PASS
source-only corruption：PASS
负例均衡：PASS
训练数值稳定：PASS
mean aggregation reconstruction：FAIL
训练对象泛化：FAIL
fresh-object 泛化：FAIL
更多 source 带来收益：FAIL
C2：继续关闭
```

v2 干净否定的是：

> **“先平均 source，再判断 correspondence”这一结构。**

它没有否定“聚合前 pairwise correspondence”这一更强假设。

---

## 5. C1 v3：pairwise-before-aggregation

### 5.1 核心架构改动

v3 将流程改为：

```text
per-view visual embedding z_i
        ↓
对每一对 source views 计算 pairwise logit s_ij
        ↓
按其他 views 对每个 source 求支持度 c_i
        ↓
physical weight × pairwise confidence
        ↓
聚合 source visual value
        ↓
重建固定 held-out target
```

近似形式：

\[
s_{ij}=\operatorname{MLP}\left[z_i,z_j,z_i\odot z_j,|z_i-z_j|,g_i,g_j\right]
\]

\[
c_i=\operatorname{mean}_{j\neq i}\sigma(s_{ij})
\]

\[
\tilde w_i=w_i^{physical}\,c_i
\]

\[
z_{consensus}=\frac{\sum_i \tilde w_i z_i}{\sum_i\tilde w_i}
\]

重要约束：

- pairwise confidence 在 aggregation **之前**计算；
- geometry 不再加入 reconstruction value；
- geometry 只可参与 pairwise 诊断/匹配；
- fixed held-out target 与 source-only corruption 保留；
- 旧 score/geometry heads 默认不训练，仅保留诊断。

### 5.2 训练目标

主要 loss：

```text
correct reprojection
+ normalized reprojection ranking
+ pairwise-confidence ranking
+ pairwise BCE calibration
+ embedding variance regularization
```

关闭旧 shortcut heads：

```text
score_rank_weight = 0
score_bce_weight = 0
geometry_bce_weight = 0
anti_shortcut_weight = 0
```

### 5.3 Seed42 训练

```text
run: corr_pairwise_v3_s200_seed42
steps: 200
attempts: 266
skipped: 66
mode counts: 67 / 65 / 68
finite: true
```

最后一步：

```text
normalized reprojection advantage = +0.15039
pairwise confidence advantage = +0.09641
```

### 5.4 Seed42 训练对象结果

3-source，12 个 eligible 训练对象：

| Mode | Reprojection win | Reproj mean | Reproj median | Pairwise win | Pairwise mean |
|---|---:|---:|---:|---:|---:|
| cyclic1 | 100% | +0.02047 | +0.01045 | 100% | +0.05660 |
| cyclic2 | 100% | +0.01183 | +0.00934 | 100% | +0.06004 |
| reverse | 100% | +0.01620 | +0.01058 | 100% | +0.05755 |

这证明 v3 至少在训练对象上完整学通了 correspondence → aggregation → reconstruction 链路。

### 5.5 Seed42 fresh-object 结果

3-source 主门：

| Mode | Objects | Reprojection win | Reproj mean | Reproj median | Pairwise win | Pairwise mean | Pairwise median |
|---|---:|---:|---:|---:|---:|---:|---:|
| cyclic1 | 26 | 65.38% | +0.00732 | +0.00619 | 96.15% | +0.03147 | +0.02871 |
| cyclic2 | 25 | 76.00% | +0.00841 | +0.00433 | 96.00% | +0.02882 | +0.02260 |
| reverse | 26 | 69.23% | +0.00735 | +0.00292 | 92.31% | +0.02485 | +0.01644 |

三类 hard negative 全部通过 seed42 主门。

### 5.6 三 seed 训练稳定性

| Seed | Steps | Attempts | Skipped | Mode counts | Finite | Fresh 平均 reproj win |
|---:|---:|---:|---:|---|---|---:|
| 42 | 200 | 266 | 66 | 67 / 65 / 68 | true | 70.21% |
| 43 | 200 | 263 | 63 | 68 / 66 / 66 | true | 78.00% |
| 44 | 200 | 268 | 68 | 66 / 68 / 66 | true | 88.31% |

三 seed mode 汇总：

| Mode | Mean reproj win | Mean reproj advantage | Mean reproj median | Mean pairwise win | Mean pairwise advantage | Mean pairwise median |
|---|---:|---:|---:|---:|---:|---:|
| cyclic1 | 74.36% | +0.00836 | +0.00524 | 94.87% | +0.03336 | +0.02961 |
| cyclic2 | 82.67% | +0.00915 | +0.00591 | 94.67% | +0.03030 | +0.02211 |
| reverse | 79.49% | +0.00835 | +0.00378 | 93.59% | +0.02641 | +0.01634 |

多 seed 汇总：

```text
passed = true
```

没有任何 seed 整体退回随机水平，pairwise confidence 比最终 reconstruction 更稳定。

### 5.7 Seed43/44 顶层 `evaluation_passed=false` 的解释

seed43/44 的每个 mode 在新的主指标上都通过：

- reprojection win / mean / median；
- pairwise confidence win / mean / median；
- paired view trend。

但旧评估器仍要求：

```text
visual_score_advantage >= 0
```

而 v3 训练中旧 visual score head 权重为 0，未被训练。seed43/44 该 head 的均值仅约 `-0.0002`，因此旧顶层 gate 将报告标成 false。

这是评估门遗留问题，不是 v3 主机制失败。后续应从 v3 required gate 中删除旧 visual/geometry score head 条件。

---

## 6. C1 v3 消融实验

### 6.1 消融设置

以 seed42 fresh 3-source 为基准：

```text
full
visual_zero
visual_shuffle
geometry_pair_off
uniform_pairwise
```

含义：

- `visual_zero`：source visual feature 置零，held-out target 保持原视觉；
- `visual_shuffle`：source view 间循环打乱 visual feature，geometry / pose 保持；
- `geometry_pair_off`：关闭 pairwise head 中的 geometry pair term，只保留 visual pair term；
- `uniform_pairwise`：关闭 learned pairwise gate，退化为 physical-weight-only / uniform pairwise aggregation。

### 6.2 平均结果

| Ablation | Reproj win | Reproj mean | Reproj median | Pairwise win | Pairwise mean | Pairwise median |
|---|---:|---:|---:|---:|---:|---:|
| full | 70.21% | +0.007692 | +0.004483 | 94.82% | +0.028380 | +0.022586 |
| visual_zero | 47.90% | ≈0 | ≈0 | 11.64% | ≈0（略负） | ≈0（略负） |
| visual_shuffle | 46.82% | +0.000764 | -0.000480 | 26.26% | -0.015789 | -0.013213 |
| geometry_pair_off | 70.21% | +0.007690 | +0.004484 | 94.82% | +0.028150 | +0.022365 |
| uniform_pairwise | 68.92% | +0.007654 | +0.004599 | 0%* | 0* | 0* |

`*` uniform 模式人为关闭 pairwise 输出，因此 pairwise 指标为 0，不代表分类性能。

### 6.3 消融硬门

```text
full_gate_passed = true
visual_zero_causes_meaningful_drop = true
visual_shuffle_causes_meaningful_drop = true
visual_only_pairwise_signal_survives_geometry_pair_off = true
learned_pairwise_gate_beats_uniform_pairwise = false
```

因此消融总报告：

```text
passed = false
```

这个 false 只来自最后一项：learned pairwise weighting 未显著优于 uniform aggregation。

### 6.4 消融的科学解释

#### 视觉信号是真实且依赖 image-pose binding 的

`visual_zero` 后 reconstruction 回到随机附近，pairwise 信号几乎消失。

`visual_shuffle` 更关键：视觉仍存在，但被绑定到错误 source view 后：

```text
reprojection win: 70.2% -> 46.8%
pairwise win: 94.8% -> 26.3%
pairwise advantage 由正变负
```

因此模型依赖的是：

```text
正确视觉内容
+
正确 camera pose / source identity
```

而不是单纯“有视觉特征即可”。

#### geometry shortcut 基本被排除

`geometry_pair_off` 与 full 几乎逐项相同，证明 pairwise confidence 的主要来源是视觉内容一致性，而不是 geometry/support shortcut。

后续部署可以优先使用 visual-only pairwise confidence，避免不必要的 geometry pair 分支。

#### learned pairwise weighting 没有显著改善 reconstruction

full 与 uniform：

```text
reprojection win: 70.21% vs 68.92%
mean advantage: +0.007692 vs +0.007654
```

差异极小，未达到预设：

```text
win gap >= 0.05
或 mean gap >= 0.003
```

原因可能是 correct/wrong branch 的 absolute confidence 整体不同，但 branch 内各 source 的相对 confidence 很接近。归一化加权时，整体尺度被分母抵消：

```text
correct c_i ≈ [0.83, 0.81, 0.82]
wrong   c_i ≈ [0.67, 0.65, 0.66]
```

两者归一化后都近似 `[1/3, 1/3, 1/3]`。

因此：

- absolute confidence 很适合作为整套多视图条件的可信度 gate；
- per-view normalized confidence 不足以证明能选择性改善 source aggregation。

---

## 7. 2-source 与 3-source 趋势

v3 成功阻止了 v2 的 3-source 灾难性反转，但并未达到“更多 source 更好”。

seed42 fresh：

```text
2-source reprojection win ≈ 80.77%
3-source reprojection win = 65.38% / 76.00% / 69.23%
```

三 seed 平均：

```text
2-source mean reprojection win ≈ 88.46%
3-source = 74.36% / 82.67% / 79.49%
```

因此当前只能声称：

> 模型能在多 source 下稳定检测 image-pose 一致性，且不再发生 v2 的系统性错误方向。

不能声称：

> learned pairwise aggregation 能随着视图增加获得更好 reconstruction。

可能原因：

- 第三个 source 与 held-out / 其他 source 的可见表面重叠更差；
- pairwise confidence 是软权重，未真正剔除 outlier；
- source 数增加后 pair 数增多，噪声累积；
- confidence 未按 source count 校准；
- 单一 weighted mean 无法表达多模态表面观察。

---

## 8. 当前正式科学结论

### 8.1 已确认

1. **Pixal3D per-view visual feature 含有可泛化的局部 image-pose correspondence 信号。**
2. 该信号在 train objects 和 fresh objects 上均可检测，且三 seed 稳定。
3. visual zero 会使重建和 pairwise 信号崩溃。
4. visual shuffle 会使 pairwise advantage 反向，证明依赖正确 image-view / pose binding。
5. geometry pair term 基本不是主要信息来源。
6. pairwise-before-aggregation 训练显著优于 v2 的 early mean-pooling 机制表现。
7. v3 pairwise confidence 是一个比 reconstruction probe 更稳定的 correspondence detector。

### 8.2 已否定或未获支持

1. **early mean aggregation 足以提取 correspondence：否定。**
2. **learned pairwise confidence 作为 normalized source weight 显著改善 reconstruction：未获支持。**
3. **增加 source view 数能够提升当前重建优势：未获支持。**
4. 旧 visual score / geometry score heads 可作为 v3 主 gate：否定，应从 required checks 移除。

### 8.3 尚未解决

1. 4-source、无 held-out 的实际部署 confidence 如何校准；
2. voxel-level confidence 与真实“可安全注入 SS residual”的关系；
3. confidence threshold 是否跨 seed / object 稳定；
4. high-confidence voxel 是否确实具有更高 reconstruction advantage；
5. correspondence gate 是否能在 frozen Flow 中产生可归因的生成收益；
6. learned gate 是否在 C2 中优于 uniform gate。

---

## 9. 当前阶段状态

| 项目 | 状态 | 说明 |
|---|---|---|
| fixed held-out target | PASS | correct/wrong 共享 target |
| source-only pose corruption | PASS | held-out pose 不被扰动 |
| balanced object × mode schedule | PASS | 三类 mode 计数均衡 |
| C1 v2 mean aggregation | FAIL | train/fresh 3-source 均失败 |
| C1 v3 pairwise detector | PASS | 三 seed fresh 指标稳定为正 |
| visual dependency | PASS | visual-zero/shuffle 均崩溃 |
| geometry shortcut exclusion | PASS | geometry-pair-off 几乎不变 |
| learned weighted aggregator | FAIL | 与 uniform reconstruction 几乎相同 |
| 3-source 优于 2-source | FAIL | 3-source 仍更弱 |
| C1 multi-seed detector gate | PASS | 汇总报告 passed=true |
| C2 correspondence-gated Flow | NOT RUN | 应先完成 C1.5 部署校准 |

建议正式命名当前成果：

```text
C1 visual correspondence detector：PASS
C1 learned pairwise weighted aggregator：FAIL
```

不要把二者合并成单一“C1 全面通过”或“C1 全面失败”。

---

## 10. 下一步建议

### 10.1 第一步：修正 v3 evaluator 的旧 gate

从 v3 required checks 删除：

```text
min_visual_score_advantage
旧 visual_score_head
旧 geometry_score_head
```

v3 主门只保留：

```text
object count
reprojection win / mean / median
pairwise confidence win / mean / median
paired source-count trend
```

不需要重新训练，只需重新生成或修正 seed43/44 顶层 report。

### 10.2 第二步：C1.5 部署形态校准审计

目的：将 held-out research probe 转换成 C2 实际可用的 confidence map。

部署输入应改成：

```text
所有可用 views 都作为 source
不保留 held-out
4-source visual-only pairwise confidence
```

需要输出每个 voxel：

```text
correct confidence
wrong-pose confidence
visual-shuffle confidence
physical support
pairwise disagreement
有效 source count
```

必须验证：

1. 4-source correct confidence > wrong；
2. visual shuffle 后优势消失或反向；
3. confidence 分布不是全开或全关；
4. high-confidence voxel 的 held-out reconstruction advantage 显著高于 low-confidence voxel；
5. 阈值在 seed42/43/44 间稳定。

建议门：

```text
对象级 correct > wrong 胜率 >= 80%
voxel-level ROC-AUC >= 0.70
top 30% confidence voxel 的 reconstruction advantage
  至少比全体 voxel 高 25%
visual-shuffle AUC 接近 0.5
有效 gate coverage 位于 20%--80%
```

阈值选择纪律：

```text
只在 train objects 0-15 上选 threshold
在 fresh objects 16-63 上冻结验证
不得用 seed44 / fresh 指标挑 threshold
```

### 10.3 第三步：C2 v3 correspondence gate

冻结 seed42 v3 checkpoint，不挑 seed44 最优结果。

推荐 gate：

\[
g(x)=\operatorname{clamp}\left(\frac{c(x)-\tau_{low}}{\tau_{high}-\tau_{low}},0,1\right)
\]

\[
\Delta v_{final}(x)=g(x)\,\Delta v_{SS}(x)
\]

部署上使用：

```text
visual-only pairwise confidence
```

不使用：

```text
旧 visual score head
旧 geometry score head
geometry pair term（除非后续证明必要）
normalized source weighting 作为主要收益点
```

C2 第一轮必须包含五路：

```text
1. stock
2. correct learned-confidence gate
3. wrong-pose learned-confidence gate
4. uniform gate
5. visual-shuffled confidence gate
```

关键对照：

```text
learned confidence gate vs uniform gate
```

因为 C1 已经证明 learned weighting 在 reconstruction probe 中几乎不比 uniform 好；C2 必须重新证明 learned confidence 对 SS residual 有因果价值。

建议 C2 机制门：

```text
null/off bit-exact stock
correct learned gate > stock
correct learned gate > wrong gate
correct learned gate > uniform gate
correct learned gate > shuffled-confidence gate
wrong gate接近 stock，而不是显著破坏 stock
至少 4/5 个 t 为正
对象胜率 >= 65%
mean / median 均为正
```

### 10.4 第四步：C2 单 seed 通过后再做多 seed

顺序：

```text
seed42 C2 s5 smoke
  -> seed42 小规模机制训练
  -> 固定噪声、多 t 五路评估
  -> 通过后 seed43/44
  -> 三 seed 汇总
```

暂不提前启用：

```text
Flow LoRA
DDP 大规模训练
mesh / SLAT 长流程
```

### 10.5 可选：论文级因果 baseline

如果需要严格归因 v2 → v3 提升来自哪里，应从头训练一个 baseline：

```text
v3 visual encoder / value / reconstruction
uniform aggregation
关闭 pairwise rank / pairwise BCE
其余训练协议完全相同
```

这样才能区分：

- pairwise auxiliary supervision 改善表示；
- learned pairwise weighting 改善聚合；
- geometry-from-value removal 的作用。

该实验对论文归因有价值，但不是进入 C2 的必要前置条件。

---

## 11. 当前不建议继续做的事情

```text
不要把 C1 从 200 step 延长到 300/500 step
不要继续增加 C1 seed 数
不要根据 seed44 最优表现挑 checkpoint
不要把旧 visual score head 当 correspondence gate
不要声称 learned pairwise weighting 已改善 reconstruction
不要声称 3-source 优于 2-source
不要跳过 C1.5 直接长跑 C2
不要在 C2 只比较 correct vs wrong，而缺少 uniform / shuffle 对照
```

当前瓶颈不是 C1 训练不足，而是：

```text
如何把稳定的 absolute visual confidence
校准成真正对 SS residual 有用的 gate
```

---

## 12. 关键实验路径与可复现产物

### 12.1 v1 原始三 seed 报告

本地归档文件：

```text
train_report.json / report.json
train_report43 / report43
train_report44 / report44
```

原始命令约定的服务器 run：

```text
ar_ss_flow/outputs/corr_heldout_overfit16_s300_seed42
ar_ss_flow/outputs/corr_heldout_overfit16_s300_seed43
ar_ss_flow/outputs/corr_heldout_overfit16_s300_seed44
```

### 12.2 v2 seed42

```text
ar_ss_flow/outputs/corr_heldout_fixedtarget_v2_s150_seed42/train_report.json
ar_ss_flow/outputs/corr_heldout_fixedtarget_v2_s150_seed42/eval_train16_sources2_3_fixedtarget/report.json
ar_ss_flow/outputs/corr_heldout_fixedtarget_v2_s150_seed42/eval_fresh48_sources2_3_fixedtarget/report.json
```

### 12.3 v3 三 seed

```text
ar_ss_flow/outputs/corr_pairwise_v3_s200_seed42/
ar_ss_flow/outputs/corr_pairwise_v3_s200_seed43/
ar_ss_flow/outputs/corr_pairwise_v3_s200_seed44/
```

每个 run 包含：

```text
train_report.json
eval_train16_sources2_3/report.json
eval_fresh48_sources2_3/report.json
checkpoints/last.pt
```

### 12.4 v3 消融

```text
ar_ss_flow/outputs/corr_pairwise_v3_s200_seed42/
  ablation_fresh48_sources2_3/
    visual_zero/report.json
    visual_shuffle/report.json
    geometry_pair_off/report.json
    uniform_pairwise/report.json
  ablation_fresh48_sources2_3_summary/report.json
```

### 12.5 v3 多 seed 汇总

```text
ar_ss_flow/outputs/corr_pairwise_v3_s200_multiseed_summary/report.json
```

### 12.6 当前完整结果归档

```text
c1_pairwise_v3_ablation_multiseed_results.tar.gz
```

归档内包含：

- seed42/43/44 train reports；
- train-object / fresh-object eval reports；
- seed42 四路 ablation；
- ablation summary；
- multi-seed summary。

---

## 13. 最终阶段性判断

截至 2026-07-14，本分支已经完成了从“是否存在 correspondence”到“如何部署 correspondence”的问题转移。

早期问题：

```text
Pixal3D feature 中是否有 image-pose-sensitive visual signal？
```

当前答案：

```text
有，而且 visual zero / visual shuffle / geometry-off / 三 seed
均给出较强证据。
```

新的核心问题：

```text
如何把 absolute visual correspondence confidence
转换成对 frozen SS Flow 有因果增益的 residual gate？
```

推荐项目路线：

```text
修正 v3 evaluator 旧 score gate
        ↓
4-source visual-only confidence calibration（C1.5）
        ↓
冻结 seed42 correspondence checkpoint
        ↓
zero-init SS residual gate
        ↓
stock / correct / wrong / uniform / visual-shuffle 五路机制门
        ↓
通过后才做 C2 多 seed
```

当前正式状态：

```text
C1 visual correspondence detector：PASS
C1 learned pairwise weighted aggregator：FAIL
C1.5 deployment calibration：NEXT
C2 correspondence-gated SS Flow：NOT RUN
```

# C1.5 Visual-only Pairwise Deployment Calibration（2026-07-14）

### 1. 实验目的

此前C1 pairwise-v3实验已经证明：

1. pairwise confidence能够稳定区分correct pose与wrong pose；
2. visual zero和visual shuffle会显著破坏该信号；
3. geometry pair分支关闭后结果基本不变；
4. 三个训练seed上的对象级correspondence判断具有较好的稳定性；
5. learned pairwise confidence作为归一化source aggregation权重，并没有明显优于uniform aggregation。

因此，本阶段不再把pairwise confidence视为source feature的归一化权重，而是尝试将其校准为后续C2可直接使用的visual-only residual gate。

本次C1.5审计采用：

```text
confidence source = visual-only pairwise confidence
geometry_pair_scale = 0
threshold selection objects = indices 0-15
frozen validation objects = indices 16-63
```

Gate定义为：

```text
gate = clamp(
    (confidence - tau_low) /
    (tau_high - tau_low),
    0,
    1
)
```

训练对象仅用于选择阈值；fresh对象上的所有指标均使用冻结阈值计算。

---

### 2. 阈值校准结果

在训练对象`indices 0-15`上选择得到：

```text
tau_low  = 0.643568
tau_high = 0.707083
```

三种hard negative分别选择出的阈值非常接近：

```text
pose_cyclic1 threshold = 0.643340
pose_cyclic2 threshold = 0.642616
pose_reverse threshold = 0.642966

per-mode threshold spread = 0.000725
```

该spread远小于预设上限`0.15`，说明不同pose corruption模式对阈值的要求高度一致，confidence数值尺度本身比较稳定。

---

### 3. Fresh对象级部署结果

在`indices 16-63`上，4-view全source部署形式的对象级correct-vs-wrong判断结果为：

| Hard negative | Correct > wrong | Correct > visual shuffle | Mean confidence advantage |
| ------------- | --------------: | -----------------------: | ------------------------: |
| pose_cyclic1  |          96.43% |                   92.86% |                  +0.03147 |
| pose_cyclic2  |          88.89% |                   92.59% |                  +0.02298 |
| pose_reverse  |          92.59% |                   96.30% |                  +0.02847 |

三类hard negative全部显著超过预设的80%对象级胜率门槛。
这说明在真实部署形式下，即不依赖held-out target时，visual-only pairwise confidence仍能稳定判断：

```text
当前多视图image-pose binding是否自洽
```

因此，C1阶段关于“是否存在可泛化的视觉对应信号”的结论继续成立。

---

### 4. Held-out probe结果

在4-source + 1 fixed held-out target的probe形式下：

| Hard negative | Correct > wrong | Reprojection win rate | Mean reprojection advantage |
| ------------- | --------------: | --------------------: | --------------------------: |
| pose_cyclic1  |          92.31% |                92.31% |                    +0.01231 |
| pose_cyclic2  |          75.00% |                75.00% |                    +0.01374 |
| pose_reverse  |          75.00% |                83.33% |                    +0.01108 |

虽然probe有效对象数只有12至13个，但三类hard negative的平均reprojection advantage均为正。

## 这说明较高的visual-only pairwise confidence通常对应更可靠的held-out visual reconstruction，而不是一个与实际内容质量无关的分类分数。

### 5. Voxel级AUC结果

Fresh对象上的单voxel correct-vs-wrong AUC为：

```text
pose_cyclic1: 0.6606
pose_cyclic2: 0.6340
pose_reverse: 0.6502
```

Correct-vs-visual-shuffle AUC为：

```text
pose_cyclic1: 0.7039
pose_cyclic2: 0.7002
pose_reverse: 0.6981
```

预设科学门要求：

```text
AUC >= 0.70
```

因此：

1. 三类correct-vs-wrong voxel AUC均未通过；
2. reverse的correct-vs-shuffle AUC也略低于0.70；
3. 单voxel confidence不足以作为高精度的独立二分类器。
   但该结果不能解释为“confidence没有意义”。

对象级胜率超过88%，而voxel级AUC只有0.63至0.66，说明信号更接近：

```text
对象级或局部区域级的多视图一致性
```

而不是：

```text
每个单独voxel都能被严格判断为正确或错误
```

---

### 6. Gate coverage和selectivity

使用训练对象冻结得到的`tau_low`后，fresh对象correct gate coverage为：

```text
pose_cyclic1: 0.8070
pose_cyclic2: 0.8035
pose_reverse: 0.8033
```

预设上限为：

```text
max_gate_coverage = 0.80
```

因此三种模式均轻微超过上限，超出范围约为0.003至0.007。

该失败程度很小，本身不构成严重问题，但意味着当前阈值会让约80%的有效voxel获得非零gate，gate仍偏宽松。

相比之下，gate selectivity明显为正：

| Hard negative | Correct coverage | Wrong coverage | Correct − wrong |
| ------------- | ---------------: | -------------: | --------------: |
| pose_cyclic1  |           0.8070 |         0.5993 |         +0.2077 |
| pose_cyclic2  |           0.8035 |         0.6077 |         +0.1958 |
| pose_reverse  |           0.8033 |         0.6039 |         +0.1994 |

Correct与visual-shuffle之间的coverage差约为：

```text
+0.251 ～ +0.260
```

## 因此，虽然当前gate开启范围偏大，但correct输入比wrong和shuffle输入获得了明显更多的有效gate覆盖。

### 7. Confidence与重建质量的关系

对confidence最高的top 30% voxel进行统计后，其reprojection advantage相对全部voxel的提升为：

```text
pose_cyclic1: +51.8%
pose_cyclic2: +61.3%
pose_reverse: +68.5%
```

绝对提升分别为：

```text
pose_cyclic1: +0.00577
pose_cyclic2: +0.00873
pose_reverse: +0.00736
```

三种模式均明显超过预设的：

```text
relative uplift >= 25%
absolute uplift >= 0.002
```

这说明confidence虽然不适合做精确的单voxel correct/wrong二分类，但非常适合做排序：

```text
高confidence voxel
通常比低confidence voxel
具有更高的视觉重建可靠性
```

这正是后续soft residual gate需要的性质。

---

### 8. C1.5严格科学门结果

本次脚本最终输出：

```text
passed = False
exit_code = 2
```

程序本身正常完成，并成功写出了：

```text
deployment_calibration_v1/report.json
deployment_calibration_v1/calibration.json
```

退出码2来自`--fail_on_error`，代表科学门失败，不是程序异常。

失败项主要为：

```text
fresh voxel AUC correct-vs-wrong < 0.70
correct gate coverage略高于0.80
```

通过项包括：

```text
对象级correct > wrong胜率
对象级correct > shuffle胜率
gate selectivity
shuffle selectivity
top-30% reprojection uplift
不同pose mode阈值稳定性
```

因此本阶段应记录为：

```text
C1.5对象级deployment confidence：PASS
C1.5 confidence质量排序能力：PASS
C1.5 gate selectivity：PASS
C1.5阈值稳定性：PASS

C1.5单voxel分类AUC门：FAIL
C1.5严格总门：FAIL
```

---

### 9. 当前阶段科学结论

结合此前的v2、v3、多seed和消融实验，可以进一步明确：

1. Pixal3D per-view visual feature中存在可泛化的image-pose correspondence信号；
2. 该信号不是geometry/support shortcut；
3. 该信号在对象级和多voxel总体统计上非常稳定；
4. 该信号可以有效对voxel重建质量进行排序；
5. 当前confidence不适合被解释为精确的单voxel正确概率；
6. 当前confidence更适合作为局部区域级soft gate，而不是hard voxel classifier；
7. 当前生成的`calibration.json`尚不能作为最终C2冻结配置。

当前阶段不应直接宣称：

```text
每个voxel都能够被可靠地区分为correct或wrong correspondence
```

可以宣称：

```text
模型能够稳定判断多视图image-pose binding的整体和局部区域一致性，
并且高confidence区域具有显著更高的held-out视觉重建可靠性。
```

---

## 下一步建议

### 10. 不重新训练C1 pairwise checkpoint

当前问题不是pairwise模型没有学到视觉对应，而是原始单voxel confidence存在局部噪声。

因此下一步不建议：

```text
增加训练到300或500步
调低AUC门槛直接放行
重新增加geometry pair分支
继续训练旧visual score head
```

应保留现有三个pairwise-v3 checkpoint不变。

---

### 11. 新增局部区域confidence校准

下一阶段建议定义为：

```text
C1.5-v2 local confidence calibration
```

对原始voxel confidence进行局部、support-aware聚合，再重新执行同一套训练集阈值选择和fresh对象冻结验证。

优先测试以下三个版本：

#### Baseline

```text
raw voxel confidence
```

即本轮结果，作为固定基线。

#### Local mean

```text
在3×3×3局部邻域内，
只对valid且具有physical support的voxel
进行加权均值平滑
```

示意：

```text
c_local(x) =
sum_y K(x,y) * support(y) * c(y)
/
sum_y K(x,y) * support(y)
```

#### Local top-k mean

```text
在3×3×3邻域内，
选择confidence最高的若干有效voxel求均值
```

该方式可以减少背景、遮挡边界和低support voxel对局部confidence的污染。

第一版不建议使用可学习网络，先验证无参数局部聚合是否能够改善AUC。

---

### 12. 局部校准的新科学门

训练对象仍只用于选择阈值，fresh对象继续冻结验证。

建议保留以下硬门：

```text
对象级correct > wrong胜率 >= 80%
对象级correct > shuffle胜率 >= 80%

correct-vs-wrong voxel/region AUC >= 0.70
correct-vs-shuffle voxel/region AUC >= 0.70

correct gate coverage位于20%～80%
correct - wrong coverage >= 10%
correct - shuffle coverage >= 10%

top 30% confidence区域的reprojection：
relative uplift >= 25%
absolute uplift >= 0.002
```

同时新增：

```text
local confidence AUC
必须显著高于raw confidence AUC

建议平均AUC提升 >= 0.03
且三种hard negative均不得下降
```

如果局部聚合后仍只有约0.63至0.66 AUC，则不应继续针对voxel classifier调参。

---

### 13. 进入C2的两种可能路径

#### 路径A：局部区域校准通过

如果3×3×3局部confidence达到：

```text
fresh AUC >= 0.70
对象级胜率 >= 80%
gate coverage稳定
top-30% uplift保持为正
```

则冻结：

```text
seed42 pairwise-v3 checkpoint
local aggregation规则
tau_low
tau_high
```

随后进入C2五路机制门：

```text
stock
correct learned-confidence gate
wrong-pose learned-confidence gate
uniform gate
visual-shuffled-confidence gate
```

#### 路径B：局部区域AUC仍未通过

如果局部聚合不能把AUC提高到0.70，但对象级胜率和top-30% uplift仍稳定，则不再把confidence作为voxel-level gate。

可以改为更保守的：

```text
object-level / volume-level scalar gate
```

或者：

```text
只选择top 20%～30% confidence voxel启用SS residual
其余区域完全关闭
```

此时C2应被定义为探索性机制实验，而不是正式进入完整Flow训练。

---

### 14. 当前执行纪律

```text
保留现有pairwise-v3三个seed checkpoint
保留本轮C1.5报告作为raw-confidence baseline
暂不使用当前calibration.json进入正式C2
下一步只实现无参数局部confidence聚合审计
```

局部审计完成前，不进行：

```text
完整C2训练
多seed C2
mesh/SLAT长流程
进一步扩大数据或训练步数
```

当前最合理的下一步是：

```text
raw confidence
→ 3×3×3 support-aware local confidence
→ train-object threshold selection
→ fresh-object frozen evaluation
→ 与raw baseline逐项配对比较
```


## C1.5-v2 / v3：局部平滑与Percentile Gate更新（2026-07-14）

### 1. 本阶段目标

C1.5-v1实验表明，visual-only pairwise confidence具有以下性质：

```text
对象级correct > wrong判断稳定；
高confidence voxel通常具有更高reprojection advantage；
但单voxel correct-vs-wrong AUC只有约0.63～0.66。
```

因此，后续进行了两次不重新训练模型的部署后处理实验：

```text
C1.5-v2：
对raw voxel confidence进行3×3×3局部聚合，
尝试把噪声较大的单voxel信号转化为稳定的局部区域信号。

C1.5-v3：
放弃绝对阈值和单voxel分类，
改为在每个volume内部按raw confidence排序，
选择top 20%、30%或40%的高confidence区域。
```

两次实验均保持：

```text
checkpoint固定；
geometry_pair_scale = 0；
confidence来源为visual-only pairwise confidence；
训练对象indices 0-15只用于方法或比例选择；
fresh对象indices 16-63用于冻结验证。
```

---

# C1.5-v2 Local Confidence Calibration

## 2. 方法

并行比较三种confidence：

```text
raw：
原始pairwise voxel confidence。

local_mean：
在3×3×3邻域中，对具有共同source support的有效voxel求均值。

local_topk：
在3×3×3邻域中，选择confidence最高的8个有效voxel求均值。
```

Correct、wrong和visual-shuffle三条分支使用完全相同的binary support mask，避免局部聚合重新引入geometry shortcut。

方法选择只基于训练对象。

---

## 3. 训练对象上的方法选择

训练集结果为：

| 方法         | Selection score |
| ---------- | --------------: |
| raw        |          0.7514 |
| local_mean |          0.7628 |
| local_topk |          0.7501 |

因此训练集选择：

```text
selected_method = local_mean
train_selection_gain_vs_raw = +0.0114
```

对应冻结阈值：

```text
tau_low  = 0.640435
tau_high = 0.701184
```

训练对象上，local mean相对raw存在轻微提升。

---

## 4. Fresh对象上的泛化结果

Fresh correct-vs-wrong AUC为：

| Hard negative | Local mean AUC | Raw AUC |    Gain |
| ------------- | -------------: | ------: | ------: |
| pose_cyclic1  |         0.6472 |  0.6491 | -0.0019 |
| pose_cyclic2  |         0.6210 |  0.6242 | -0.0032 |
| pose_reverse  |         0.6278 |  0.6379 | -0.0101 |

平均变化：

```text
fresh mean AUC gain vs raw = -0.0051
```

Correct-vs-visual-shuffle AUC也全部下降：

```text
pose_cyclic1: -0.0232
pose_cyclic2: -0.0235
pose_reverse: -0.0218
```

因此，local mean在训练对象上的轻微收益没有泛化到fresh对象，并且在三种hard negative上均产生负面影响。

---

## 5. Local mean仍保留的正面结果

对象级correct-vs-wrong胜率仍然很高：

```text
pose_cyclic1: 96.43%
pose_cyclic2: 88.89%
pose_reverse: 92.59%
```

Correct gate coverage为：

```text
0.675 ～ 0.683
```

相对于v1中约0.803～0.807的coverage，local mean与新阈值使gate宽度进入预设的20%～80%区间。

Correct-vs-wrong gate selectivity为：

```text
pose_cyclic1: +0.2090
pose_cyclic2: +0.1805
pose_reverse: +0.1754
```

Top-30%区域相对整体reprojection advantage的提升为：

```text
pose_cyclic1: +23.2%
pose_cyclic2: +46.7%
pose_reverse: +48.1%
```

其中cyclic2和reverse仍然具有明显的质量排序能力。

---

## 6. C1.5-v2科学结论

本轮全局检查为：

```text
selected_method_is_local: PASS
train_selection_gain: PASS

fresh_mean_auc_gain_vs_raw: FAIL
no_fresh_wrong_auc_mode_drop: FAIL
no_fresh_shuffle_auc_mode_drop: FAIL
all_selected_method_modes_passed: FAIL
```

最终：

```text
passed = false
exit_code = 2
```

该结果说明：

> Raw confidence中的误差不是简单、独立、可通过局部平均消除的空间噪声。

如果错误主要是孤立噪声，3×3×3均值应当提高fresh AUC。但实际三种hard negative的AUC全部下降，尤其correct-vs-shuffle下降约0.022，说明局部平均反而模糊了有用的视觉对应差异。

当前confidence较低或错误的区域很可能来自具有空间连续性的因素：

```text
遮挡边界；
错误pose投影形成的连续区域；
相邻但不同的物体表面；
重复纹理和对称结构；
深度或可见性误差；
大块低纹理区域。
```

简单空间平滑会同时发生：

```text
高confidence向邻近错误voxel扩散；
低confidence向邻近正确voxel扩散；
correct和wrong confidence分布进一步重叠。
```

因此：

```text
C1.5-v2 local mean训练集收益：PASS
C1.5-v2 local mean fresh泛化：FAIL
C1.5-v2局部空间平滑路线：FAIL
```

后续不应继续盲目尝试更大的5×5×5或7×7×7均值核。

---

# C1.5-v3 Raw Percentile Gate

## 7. 方法

C1.5-v3不再把confidence解释为经过校准的绝对概率，也不再要求单voxel AUC达到0.70。

新的假设是：

> 即使confidence不能完整区分所有correct和wrong voxel，它仍可能在每个volume内部可靠地找出相对最可信的一部分区域。

因此，对每个held-out volume的有效voxel独立排序，并比较：

```text
top 20%
top 30%
top 40%
```

训练对象仅用于选择top fraction；fresh对象使用冻结比例。

每个比例均比较四种区域选择方式：

```text
correct-confidence selector；
同覆盖率random selector；
wrong-pose-confidence selector；
visual-shuffle-confidence selector。
```

该实验直接检验：

```text
由correct confidence选择的高排名区域，
是否真的比随机区域以及wrong/shuffle confidence选择的区域
具有更高reprojection advantage。
```

---

## 8. 训练对象上的Fraction选择

训练结果为：

| Top fraction | Minimum mode score | Mean mode score | Selection score |
| ------------ | -----------------: | --------------: | --------------: |
| 20%          |           +0.04554 |        +0.05282 |        +0.05875 |
| 30%          |           +0.04119 |        +0.04936 |        +0.05353 |
| 40%          |           +0.03505 |        +0.04249 |        +0.04567 |

训练集选择：

```text
selected_fraction = 0.20
```

这说明confidence最高的20%区域在训练对象上具有最强、最稳定的质量优势。

---

## 9. Fresh对象上相对整体和随机区域的结果

Top-20% correct-confidence区域的reprojection advantage为：

| Hard negative | Selected advantage | 相对整体绝对提升 |   相对提升 |
| ------------- | -----------------: | -------: | -----: |
| pose_cyclic1  |           +0.02133 | +0.00902 | +73.3% |
| pose_cyclic2  |           +0.02492 | +0.01119 | +81.4% |
| pose_reverse  |           +0.02205 | +0.01097 | +99.0% |

Top-20% correct selector相对同覆盖率random selector的平均增益为：

```text
pose_cyclic1: +0.00906
pose_cyclic2: +0.01121
pose_reverse: +0.01116
```

对象级correct selector优于random selector的胜率为：

```text
pose_cyclic1: 76.92%
pose_cyclic2: 83.33%
pose_reverse: 91.67%
```

该结果稳定支持：

> Raw confidence的对象内部排序能够选出比随机区域明显更可靠的高质量区域。

因此，C1.5-v1中观察到的top-confidence质量排序能力在更严格的等覆盖率随机对照下仍然成立。

---

## 10. 相对wrong-confidence selector的结果

Correct-confidence selector相对wrong-confidence selector的平均增益均为正：

```text
pose_cyclic1: +0.01176
pose_cyclic2: +0.00859
pose_reverse: +0.01074
```

但对象级胜率为：

```text
pose_cyclic1: 76.92%
pose_cyclic2: 58.33%
pose_reverse: 58.33%
```

其中cyclic2和reverse未通过对象级稳定性门。

这形成了一个重要现象：

```text
平均增益为正；
但多数对象上不稳定获胜。
```

这说明correct selector的收益集中在部分对象或部分volume上，而不是在绝大多数对象中都一致优于wrong selector。

换言之：

> Wrong-pose confidence虽然整体数值低于correct confidence，但它在空间上选择出的高排名区域，仍经常与correct confidence选择出的高质量区域重叠。

因此，当前raw confidence较强地编码了：

```text
该区域是否容易重建；
是否具有稳定视觉内容；
是否属于高纹理或多视图一致区域；
是否具有较好可见性和support。
```

但它较弱地编码了：

```text
该区域只有在correct pose下才应该被选择，
而在wrong pose下必须被排除。
```

这是对象级confidence判断很强、但voxel/region级pose-specific gate不稳定的根本原因。

---

## 11. Visual-shuffle对照

Fresh结果中：

```text
correct selector vs shuffle selector的平均增益：
pose_cyclic1: +0.01122
pose_cyclic2: +0.01155
pose_reverse: +0.01030
```

说明正确视觉对应产生的排名仍然明显优于visual-shuffle confidence产生的排名。

但shuffle-confidence selector相对random selector本身仍有正增益：

```text
pose_cyclic1: +0.00596
pose_cyclic2: +0.00508
pose_reverse: +0.00492
```

这表示即使打乱image-view binding，confidence中仍残留部分通用的区域质量信息，例如：

```text
视觉特征强度；
纹理丰富程度；
物体表面可见性；
基础physical support；
稳定的语义区域。
```

所以当前confidence不是纯粹的pose correctness分数，而是：

```text
pose-sensitive correspondence信号
+
pose-insensitive reconstructability / saliency信号
```

二者的混合。

---

## 12. C1.5-v3科学门结果

三种hard negative结果为：

```text
pose_cyclic1: PASS
pose_cyclic2: FAIL
pose_reverse: FAIL
```

失败项均为：

```text
object_win_vs_wrong_confidence_selector
```

全局检查：

```text
train_selection_score: PASS
all_fresh_modes_passed: FAIL
```

最终：

```text
passed = false
exit_code = 2
```

因此：

```text
C1.5-v3 top-percentile相对整体区域：PASS
C1.5-v3 top-percentile相对随机区域：PASS
C1.5-v3 top-percentile相对shuffle selector：平均PASS
C1.5-v3相对wrong-pose selector对象级稳定性：FAIL
C1.5-v3严格总门：FAIL
```

---

# 13. 两次更新后的综合结论

C1.5-v2与v3共同排除了两种假设。

## 被否定的假设一：单voxel噪声可以通过局部平滑解决

Local mean在训练对象上略有提升，但在fresh对象上三种hard negative全部下降。

因此，当前voxel confidence的空间误差不是简单独立噪声，无法通过无参数局部平滑修复。

## 被否定的假设二：对象内部最高confidence区域具有稳定的pose-specific选择能力

Top-20%区域确实明显优于整体和随机区域，但wrong-pose confidence经常也能选择出相似的高质量区域。

因此，当前confidence能够找到“好重建区域”，但不能稳定保证这些区域只有correct pose才能找到。

---

## 14. 当前对Pairwise Confidence最准确的功能定义

经过visual-zero、visual-shuffle、geometry-off、多seed、absolute threshold、local smoothing和percentile gate等实验后，当前confidence应被定义为：

> 一个具有明显image-pose敏感性的多视图视觉一致性分数，同时混合了较强的通用区域可重建性和视觉显著性信息。

它已经可靠地支持：

```text
对象级correct-vs-wrong判断；
多视图整体一致性检测；
高质量区域排序；
排除geometry-only shortcut；
visual shuffle敏感性；
跨seed泛化。
```

它尚未可靠地支持：

```text
单voxel correct/wrong分类；
局部空间精确定位错误pose区域；
稳定的pose-specific top-region选择；
逐voxel或逐region的正式Flow residual gate。
```

---

# 15. C1阶段最终拆分裁决

当前C1分支应正式拆成三个结论：

```text
C1 object-level correspondence detector：
PASS

C1 region quality / reconstructability ranker：
PASS

C1 voxel-level or region-level pose-specific gate：
FAIL
```

同时保留此前结论：

```text
C1 learned normalized source weighting：
FAIL
```

因此，不应继续将当前confidence描述为精确的voxel correspondence probability。

---

# 下一步建议

## 16. 停止继续开发无参数voxel/region后处理

不建议继续尝试：

```text
更大local mean kernel；
更多local top-k组合；
Gaussian smoothing；
形态学膨胀或腐蚀；
更多top fraction；
降低科学门槛；
在fresh对象上重新选择fraction或threshold。
```

原因是：

1. Local mean在三个hard negative上方向一致地降低fresh AUC；
2. Percentile gate已经证明区域质量排序有效；
3. 真正失败的是pose specificity，而不是比例或平滑强度没有调好；
4. 继续试更多后处理容易演化成针对16-63 fresh对象的调参和过拟合。

---

## 17. 将下一阶段改为Object/Volume-level Scalar Gate

对象级correct-vs-wrong胜率始终稳定在：

```text
约89%～96%
```

这是当前confidence最可靠的能力。

因此，下一步应放弃：

```text
g(x)：每个voxel一个gate
```

改为：

```text
g_volume：每个对象或每个多视图volume一个scalar gate
```

Scalar confidence可以由raw voxel confidence的鲁棒统计得到，例如：

```text
全体有效voxel confidence均值；
中位数；
top-20% confidence均值；
correct confidence分布的高低分位数差；
有效source coverage；
多视图pairwise confidence均值。
```

第一版不应训练复杂网络，可在训练对象上比较少量预定义统计方法。

---

## 18. 建议新增C1.6 Object-level Calibration

C1.6只回答：

> 在实际无held-out部署形式下，能否用一个volume-level scalar confidence稳定判断当前多视图输入是否可信？

训练对象`0-15`用于选择scalar统计方法和阈值，fresh对象`16-63`冻结验证。

建议候选：

```text
mean confidence；
median confidence；
top-20% mean；
trimmed mean；
mean + confidence dispersion。
```

建议科学门：

```text
correct > wrong对象胜率 >= 85%
correct > shuffle对象胜率 >= 85%

对象级correct-vs-wrong ROC-AUC >= 0.85
对象级correct-vs-shuffle ROC-AUC >= 0.85

三种hard negative均通过
不同mode阈值spread稳定
seed42、43、44均通过
```

如果对象数量导致单mode AUC方差较大，可同时报告bootstrap confidence interval，但不能在fresh对象上重新选方法。

---

## 19. C1.6通过后的小规模C2机制门

如果object-level scalar gate通过，C2可以使用：

```text
同一个scalar gate乘到整个SS residual volume上
```

而不是逐voxelgate：

```text
Delta_v_final = g_volume * Delta_v_SS
```

第一轮C2必须保留五路对照：

```text
stock；
correct-pose scalar gate；
wrong-pose scalar gate；
visual-shuffle scalar gate；
uniform scalar gate。
```

Uniform scalar gate必须匹配correct gate的平均幅度，防止收益仅来自整体residual scale变化。

建议机制成功条件：

```text
null/off bit-exact stock；

correct scalar gate > stock；
correct scalar gate > wrong gate；
correct scalar gate > shuffle gate；
correct scalar gate > matched uniform gate；

至少4/5个t为正；
对象胜率 >= 65%；
mean和median advantage均为正。
```

此阶段只使用seed42进行小规模机制审计；通过后才做C2多seed。

---

## 20. 如果Object-level Scalar Gate也失败

如果C1.6无法在fresh对象和多seed上稳定通过，则应停止把correspondence分支接入Flow gate。

此时保留该分支的用途为：

```text
数据质量诊断；
多视图pose一致性检测；
错误camera binding报警；
训练样本筛选；
离线confidence报告。
```

不再将其作为生成模型的可学习控制分支。

---

## 21. 若未来必须实现Voxel-level Gate，需要改变训练监督

当前失败不能继续依靠后处理解决。

若项目后续仍要求精确的voxel-level correspondence gate，需要重新设计训练数据和监督：

```text
保存真实render depth；
保存surface normal；
保存每个view的真实surface visibility；
构建真实2D-to-3D surface correspondence；
对每个voxel建立明确的positive / negative / neutral标签；
加入局部错误pose和遮挡hard negatives；
直接训练voxel-local pose-specific classifier。
```

当前held-out reconstruction与整体pairwise ranking主要训练的是全局或区域一致性，不能自然保证精确空间定位。

这是一个新的训练阶段，不应被视为C1.5后处理的继续微调。

---

# 22. 当前执行纪律

```text
保留pairwise-v3三个seed checkpoint；
保留C1.5-v1、v2、v3全部报告；
不使用任何v1/v2/v3 calibration.json进入正式voxel-level C2；
停止局部平滑与percentile voxel/region gate路线；
下一步只实现C1.6 object/volume-level scalar calibration。
```

当前项目路线调整为：

```text
C1 pairwise correspondence detector
        ↓
对象级检测与区域质量排序已验证
        ↓
voxel / region pose-specific gate失败
        ↓
C1.6 volume-level scalar confidence calibration
        ↓
若通过：小规模scalar-gated C2五路机制门
        ↓
若失败：correspondence仅保留为诊断分支
```


# C1.6 Multi-scale Region Gate诊断结果

### 实验目的

C1.6用于寻找从object级到voxel级之间可能存在的可靠空间尺度。

本轮比较：

```text
object：
  1个16×16×16区域

octant8：
  2×2×2分区，共8个8×8×8区域

grid64：
  4×4×4分区，共64个4×4×4区域

octant8/grid64 shrinkage：
  区域有效voxel不足时向object gate收缩
```

所有候选仅在训练对象`indices 0–15`上选择，随后冻结到fresh对象`indices 16–63`进行验证。

Region gate只基于visual-only raw pairwise confidence，不使用geometry pair score。

---

### 训练集候选选择

训练集结果：

```text
object:
  selection_score = +0.000000

octant8:
  selection_score = -0.000457

octant8_shrink_k32:
  selection_score = -0.000861

octant8_shrink_k64:
  selection_score = -0.000999

grid64:
  selection_score = +0.004097

grid64_shrink_k32:
  selection_score = +0.002210

grid64_shrink_k64:
  selection_score = +0.001547
```

训练集选择结果：

```text
selected_candidate = grid64
divisions = 4
region_count = 64
region_size = 4×4×4 voxels
shrinkage = none
```

说明：

```text
8-region尺度未能优于object基线；
64-region尺度出现稳定正训练收益；
shrinkage版本低于原始grid64。
```

因此，训练集支持`grid64`作为object与voxel之间的候选临界尺度。

---

### Fresh严格门结果

三种hard negative mode均未通过完整科学门：

```text
pose_cyclic1: FAIL
pose_cyclic2: FAIL
pose_reverse: FAIL
```

三种mode唯一失败项均为：

```text
shuffle_spatial_rank_correlation
```

全局结果：

```text
selected_candidate_is_region: PASS
train_selection_score: PASS
all_fresh_modes_passed: FAIL

overall: FAIL
exit_code: 2
```

因此：

```text
不运行seed43/44；
不进入正式region-gated C2；
不将当前grid64 calibration作为正式部署gate。
```

---

### Grid64对wrong pose的结果

#### Region confidence判别

Fresh对象中，correct confidence高于wrong confidence的区域比例为：

```text
pose_cyclic1: 75.60%
pose_cyclic2: 72.71%
pose_reverse: 74.17%
```

对象级correct大于wrong胜率为：

```text
pose_cyclic1: 96.43%
pose_cyclic2: 88.89%
pose_reverse: 92.59%
```

说明object级image-pose一致性检测仍然稳定，同时64区尺度保留了部分区域级pose信息。

#### Correct region gate相对object gate

```text
pose_cyclic1:
  mean gain = +0.002453
  object win rate = 69.23%

pose_cyclic2:
  mean gain = +0.003073
  object win rate = 75.00%

pose_reverse:
  mean gain = +0.003331
  object win rate = 91.67%
```

三种mode均为正，说明`grid64`不是简单复制object scalar，其空间划分对wrong-pose reconstruction advantage具有额外收益。

#### Correct gate相对wrong-pose gate

```text
pose_cyclic1:
  mean gain = +0.003804
  object win rate = 84.62%

pose_cyclic2:
  mean gain = +0.003952
  object win rate = 66.67%

pose_reverse:
  mean gain = +0.004229
  object win rate = 66.67%
```

说明使用correct-pose confidence形成的区域gate，平均优于使用wrong-pose confidence形成的区域gate。

---

### Wrong-pose空间相关性诊断

对每个对象计算：

```text
grid64 correct confidence
vs
wrong-pose局部reprojection advantage
```

所得Spearman/rank correlation结果：

```text
pose_cyclic1:
  mean = +0.180570
  median = +0.175634
  95% bootstrap CI = [+0.050359, +0.320202]
  positive-object rate = 69.23%
  correlation ≥ 0.05 rate = 69.23%
  one-sided sign-flip p = 0.011949

pose_cyclic2:
  mean = +0.173505
  median = +0.174591
  95% bootstrap CI = [+0.085357, +0.261518]
  positive-object rate = 91.67%
  correlation ≥ 0.05 rate = 83.33%
  one-sided sign-flip p = 0.002500

pose_reverse:
  mean = +0.193192
  median = +0.198335
  95% bootstrap CI = [+0.108420, +0.271830]
  positive-object rate = 91.67%
  correlation ≥ 0.05 rate = 83.33%
  one-sided sign-flip p = 0.000650
```

三种mode均满足：

```text
mean correlation > 0.05；
bootstrap置信区间下界 > 0；
单侧符号置换检验p < 0.05。
```

结论：

```text
grid64对wrong pose造成的区域级局部损害具有稳定空间信息。
```

也就是说，correct confidence较高的4³区域，平均更倾向于对应wrong pose下reprojection损害较大的区域。

因此，从object到voxel的尺度探索并非完全失败：

```text
4×4×4分区，即每个region包含4³个voxel，
是当前观察到的wrong-pose空间临界点候选。
```

---

### Visual-shuffle结果

#### Correct gate相对object gate

```text
pose_cyclic1:
  mean gain = +0.001598
  object win rate = 69.23%
  95% bootstrap CI = [+0.000140, +0.003059]
  p = 0.032098

pose_cyclic2:
  mean gain = +0.001151
  object win rate = 66.67%
  95% bootstrap CI = [-0.000319, +0.002620]
  p = 0.086196

pose_reverse:
  mean gain = +0.001161
  object win rate = 66.67%
  95% bootstrap CI = [-0.000315, +0.002669]
  p = 0.082896
```

只有`pose_cyclic1`达到统计显著；另外两种mode均值为正，但置信区间跨0。

#### Correct gate相对shuffle gate

```text
pose_cyclic1:
  mean gain = +0.003820
  object win rate = 61.54%
  95% bootstrap CI = [+0.000959, +0.007245]
  p = 0.013549

pose_cyclic2:
  mean gain = +0.003582
  object win rate = 75.00%
  95% bootstrap CI = [+0.000623, +0.007199]
  p = 0.026349

pose_reverse:
  mean gain = +0.003588
  object win rate = 66.67%
  95% bootstrap CI = [+0.000629, +0.007277]
  p = 0.025599
```

说明：

```text
correct confidence形成的整张64区gate，
平均优于shuffle confidence形成的64区gate。
```

但这只证明整张gate模式的平均加权效果不同，不证明每个region的confidence能够稳定预测shuffle局部损害。

---

### Shuffle空间相关性诊断

对每个对象计算：

```text
grid64 correct confidence
vs
visual-shuffle局部reprojection advantage
```

结果：

```text
pose_cyclic1:
  mean = +0.043464
  median = +0.079255
  95% bootstrap CI = [-0.023059, +0.106360]
  positive-object rate = 69.23%
  correlation ≥ 0.05 rate = 61.54%
  one-sided sign-flip p = 0.114094

pose_cyclic2:
  mean = +0.045265
  median = +0.082797
  95% bootstrap CI = [-0.038977, +0.126087]
  positive-object rate = 66.67%
  correlation ≥ 0.05 rate = 66.67%
  one-sided sign-flip p = 0.161192

pose_reverse:
  mean = +0.048421
  median = +0.079796
  95% bootstrap CI = [-0.025262, +0.119789]
  positive-object rate = 66.67%
  correlation ≥ 0.05 rate = 58.33%
  one-sided sign-flip p = 0.111694
```

三种mode均表现为：

```text
平均相关性略为正；
均值接近预设阈值0.05；
但bootstrap置信区间全部跨0；
单侧置换检验全部不显著。
```

因此该结果不能解释为“门槛0.05略高”。

失败的不只是效应量阈值，还包括统计稳定性：

```text
当前数据与shuffle空间相关性为0的假设仍然相容。
```

降低阈值或删除该条件不能科学地使实验通过。

---

### 系统性失败对象

三种mode中反复出现负shuffle correlation的对象包括：

```text
object 20
object 45
object 27
object 1
```

其中：

```text
object 45和object 27通常具有约35–40个有效region；
object 20和object 1通常具有约20–23个有效region。
```

因此失败不能完全归因于有效region数量不足。

这些对象同时经常出现：

```text
shuffle_spatial_rank_correlation < 0；
shuffle_gain_vs_object < 0；
shuffle_gain_vs_shuffle_gate < 0或接近0。
```

说明visual-shuffle失败具有对象特异性的系统模式，而不是negative mode之间的随机波动。

可能原因包括：

```text
视觉对称或重复纹理；
shuffle后仍保留区域重建难度排序；
confidence主要反映visibility、support或纹理质量；
shuffle损害与confidence之间不是单调关系；
视图打乱造成的误差具有非局部传播。
```

本轮不根据fresh失败对象进行删除、重选或调参。

---

### 科学结论

本轮结果支持：

```text
1. Object-level image-pose consistency detector:
   PASS

2. Grid64 correct-vs-wrong region discrimination:
   PASS

3. Grid64相对object gate的wrong-pose区域增益:
   PASS

4. Correct region gate相对wrong-pose region gate:
   PASS

5. Wrong-pose spatial rank correlation:
   PASS

6. Correct gate相对shuffle gate的平均加权收益:
   PASS

7. Visual-shuffle spatial rank correlation:
   FAIL

8. Grid64作为通用pose/visual correspondence region gate:
   FAIL

9. C1.6严格总门:
   FAIL
```

最准确的解释是：

```text
grid64已经能够粗略定位错误pose造成的局部损害，
但不能稳定定位视觉对应被打乱后造成的局部损害。
```

因此，`grid64`包含一定的pose-specific区域信息，但尚不能作为“局部附加condition是否正确”的通用可靠性估计。

---

### 当前尺度裁决

```text
Object / volume scalar:
  当前唯一严格可靠的通用部署尺度。

Grid64:
  wrong-pose-specific空间诊断尺度；
  可保留为错误pose敏感性热图或探索性特征；
  不可作为正式通用region gate。

Octant8:
  训练阶段未优于object，不继续。

Voxel / local smoothing:
  既有实验已失败，不继续。
```

因此当前可靠尺度结论为：

```text
通用部署临界点仍为object级；
wrong-pose专用空间信号的候选临界点为grid64。
```

---

### 后续执行决定

本轮之后：

```text
不运行seed43/44的region calibration；
不降低shuffle correlation阈值；
不删除失败对象；
不继续搜索更多区域尺度；
不进入正式region-gated C2。
```

正式路线回到：

```text
C1.6 object-level scalar calibration
→ seed42/43/44冻结验证
→ object-gated C2 SS Flow机制实验
```

Region路线保留为诊断结论：

```text
grid64对wrong pose具有稳定区域级敏感性，
但不具备通用visual correspondence空间可靠性。
```

只有未来重新设计具有真实surface visibility、depth、normal或局部positive/negative/neutral监督的模型后，才重新考虑region或voxel级正式gate。


# C1.6 Self-Referenced Object-level Scalar Gate 多 Seed 结果与阶段裁决

## 1. 实验目的

C1.6 的目标是把 visual-only pairwise confidence 从不稳定的跨对象绝对分数，改造成可部署的 object-level scalar gate。

此前 raw object mean 虽然具有较高的同对象 correct > wrong 胜率，但跨对象绝对 AUC 只有约 0.77–0.85，说明不同对象之间存在明显的 confidence baseline 偏移。随后采用 self-reference 形式：

```text
self_reference_score
=
observed_object_score
-
mean(
  pose_cyclic1_reference_score,
  pose_cyclic2_reference_score,
  pose_reverse_reference_score
)
```

并用单调 sigmoid 映射为 object gate：

```text
g_object = sigmoid(self_reference_score / temperature)
```

这一设计具有两个作用：

1. 通过同对象扰动参考抵消纹理、可见性、support 和对象难度造成的 baseline 差异；
2. 避免旧的 `tau_low/tau_high + clip` 把大量 correct 和 wrong 同时截成 0，导致原始排序信息丢失。

本轮先在 seed42 的 train objects 上选择统计配置，然后将配置冻结到 seed43/44；每个训练 seed 只允许用自己的 train16 标定 sigmoid temperature，fresh48 不重新选配置或调整参数。

---

## 2. 冻结配置

seed42 训练选择出的配置为：

```text
selected_config = mean__ref_mean

object statistic:
  mean

reference reducer:
  mean

gate:
  sigmoid_selfref
```

即：

```text
1. 对每个 observed hypothesis 的有效 voxel confidence 求均值；
2. 对 cyclic1 / cyclic2 / reverse 三个 reference 均值再求平均；
3. observed mean 减去 reference mean，得到 self-reference score；
4. 通过 sigmoid(score / temperature) 得到 object gate。
```

seed43 和 seed44 均显示：

```text
selection_mode = config_frozen_from_reference
selected_config = mean__ref_mean
```

因此跨 seed 验证没有重新选择统计量或 reference reducer。

---

## 3. 各 Seed temperature

三个 checkpoint 分别使用自己的 train16 标定 temperature：

| Train seed | Temperature |
|---:|---:|
| 42 | 0.04033607 |
| 43 | 0.03448869 |
| 44 | 0.04910712 |

汇总：

```text
mean   = 0.04131063
median = 0.04033607
min    = 0.03448869
max    = 0.04910712
```

temperature 存在一定 checkpoint-scale 差异，但全部 finite 且为正。这里不要求三个 temperature 完全相同，因为它只负责把不同 checkpoint 的 self-reference score scale 映射到相近 gate range，不是跨模型共享的绝对阈值。

Fresh correct gate mean 为：

```text
seed42: 0.6444
seed43: 0.6598
seed44: 0.6400
```

虽然低于 train calibration target `0.75`，但三 seed 非常接近，说明 sigmoid gate 在 fresh objects 上具有较稳定的输出尺度。

---

## 4. Seed 42 结果

### pose_cyclic1

```text
AUC correct-vs-wrong      = 0.9534
AUC correct-vs-shuffle    = 0.9561
score win vs wrong        = 1.0000
score win vs shuffle      = 0.8889
gate win vs wrong         = 1.0000
gate win vs shuffle       = 0.8889
gate gap vs wrong         = +0.1772
gate gap vs shuffle       = +0.2202
correct gate mean         = 0.6444
PASS
```

### pose_cyclic2

```text
AUC correct-vs-wrong      = 0.9588
AUC correct-vs-shuffle    = 0.9561
score win vs wrong        = 0.8889
score win vs shuffle      = 0.8889
gate win vs wrong         = 0.8889
gate win vs shuffle       = 0.8889
gate gap vs wrong         = +0.1958
gate gap vs shuffle       = +0.2202
correct gate mean         = 0.6444
PASS
```

### pose_reverse

```text
AUC correct-vs-wrong      = 0.9588
AUC correct-vs-shuffle    = 0.9561
score win vs wrong        = 0.9630
score win vs shuffle      = 0.8889
gate win vs wrong         = 0.9630
gate win vs shuffle       = 0.8889
gate gap vs wrong         = +0.2333
gate gap vs shuffle       = +0.2202
correct gate mean         = 0.6444
PASS
```

seed42 的三种 fresh mode 全部通过。

---

## 5. Seed 43 结果

seed43 冻结使用 `mean__ref_mean`，temperature 为：

```text
0.03448869
```

### pose_cyclic1

```text
AUC correct-vs-wrong      = 0.9547
AUC correct-vs-shuffle    = 0.9492
score/gate win vs wrong   = 0.9259
score/gate win vs shuffle = 0.8889
gate gap vs wrong         = +0.2009
gate gap vs shuffle       = +0.2449
correct gate mean         = 0.6598
PASS
```

### pose_cyclic2

```text
AUC correct-vs-wrong      = 0.9479
AUC correct-vs-shuffle    = 0.9492
score/gate win vs wrong   = 0.8519
score/gate win vs shuffle = 0.8889
gate gap vs wrong         = +0.2168
gate gap vs shuffle       = +0.2449
correct gate mean         = 0.6598
PASS
```

### pose_reverse

```text
AUC correct-vs-wrong      = 0.9602
AUC correct-vs-shuffle    = 0.9492
score/gate win vs wrong   = 0.9630
score/gate win vs shuffle = 0.8889
gate gap vs wrong         = +0.2600
gate gap vs shuffle       = +0.2449
correct gate mean         = 0.6598
PASS
```

seed43 的三种 fresh mode 全部通过。最弱项是 `pose_cyclic2` 的对象胜率 `0.8519`，但仍高于预设门槛。

---

## 6. Seed 44 结果

seed44 冻结使用 `mean__ref_mean`，temperature 为：

```text
0.04910712
```

### pose_cyclic1

```text
AUC correct-vs-wrong      = 0.9095
AUC correct-vs-shuffle    = 0.9438
score/gate win vs wrong   = 0.9630
score/gate win vs shuffle = 0.8889
gate gap vs wrong         = +0.1644
gate gap vs shuffle       = +0.2018
correct gate mean         = 0.6400
PASS
```

### pose_cyclic2

```text
AUC correct-vs-wrong      = 0.9342
AUC correct-vs-shuffle    = 0.9438
score/gate win vs wrong   = 0.9259
score/gate win vs shuffle = 0.8889
gate gap vs wrong         = +0.1925
gate gap vs shuffle       = +0.2018
correct gate mean         = 0.6400
PASS
```

### pose_reverse

```text
AUC correct-vs-wrong      = 0.9424
AUC correct-vs-shuffle    = 0.9438
score/gate win vs wrong   = 0.9259
score/gate win vs shuffle = 0.8889
gate gap vs wrong         = +0.2226
gate gap vs shuffle       = +0.2018
correct gate mean         = 0.6400
PASS
```

seed44 的三种 fresh mode 全部通过。其 `pose_cyclic1` AUC 为三 seed 最低值 `0.9095`，但仍明显高于 `0.85` 门槛，而且对象胜率与 gate gap 保持稳定。

---

## 7. 三 Seed 汇总

### 7.1 pose_cyclic1

```text
mean AUC correct-vs-wrong   = 0.9392
mean AUC correct-vs-shuffle = 0.9497
minimum gate win vs wrong   = 0.9259
minimum gate win vs shuffle = 0.8889
mean gate gap vs wrong      = +0.1808
mean gate gap vs shuffle    = +0.2223
PASS
```

### 7.2 pose_cyclic2

```text
mean AUC correct-vs-wrong   = 0.9470
mean AUC correct-vs-shuffle = 0.9497
minimum gate win vs wrong   = 0.8519
minimum gate win vs shuffle = 0.8889
mean gate gap vs wrong      = +0.2017
mean gate gap vs shuffle    = +0.2223
PASS
```

### 7.3 pose_reverse

```text
mean AUC correct-vs-wrong   = 0.9538
mean AUC correct-vs-shuffle = 0.9497
minimum gate win vs wrong   = 0.9259
minimum gate win vs shuffle = 0.8889
mean gate gap vs wrong      = +0.2386
mean gate gap vs shuffle    = +0.2223
PASS
```

全局检查：

```text
same_frozen_config                 = PASS
all_gate_types_sigmoid_selfref     = PASS
all_temperatures_finite_positive   = PASS
all_seed_reports_passed            = PASS
all_modes_passed                   = PASS

overall                            = PASS
exit_code                          = 0
```

---

## 8. 与此前失败版本的对比

### 8.1 Raw absolute object scalar

旧版 raw object mean 的 fresh AUC 约为：

```text
correct-vs-wrong:
  0.765–0.811

correct-vs-shuffle:
  0.824–0.847
```

同对象 correct > wrong 胜率较高，但不同对象的 confidence baseline 差异过大，无法使用统一绝对分数进行跨对象部署。

结论：

```text
对象内相对一致性：有信号
跨对象绝对标定：失败
```

### 8.2 Self-reference + clipped threshold

self-reference score 将 AUC 提升到约 `0.95`，证明对象 baseline 已被大幅抵消。

但旧的：

```text
clip((score - tau_low) / (tau_high - tau_low), 0, 1)
```

在 fresh 上把大量 correct 和 wrong 同时截为 0，导致 gate win 降到 `0.4074`。

结论：

```text
self-reference score：成功
hard threshold transfer：失败
```

### 8.3 Self-reference + sigmoid gate

sigmoid 是严格单调映射，因此保留了 score ordering：

```text
gate win rate == score win rate
```

三 seed 中不再出现因硬裁剪制造的大量平局，同时 gate gap 均稳定高于 `0.10`。

最终证明：

```text
self-reference score：
  能跨对象区分 correct、wrong 和 shuffle

sigmoid gate：
  能把该判别稳定映射到连续 object scalar

跨 checkpoint seed：
  可复现
```

---

## 9. 科学结论

本轮支持以下结论：

```text
1. Visual-only pairwise confidence 含有稳定的 object-level image-pose consistency 信号：
   PASS

2. Raw absolute object confidence 可直接跨对象标定：
   FAIL

3. 同对象三种 pose perturbation 可以作为有效 self-reference：
   PASS

4. mean voxel statistic 优于 median、trimmed mean 和 top20 mean：
   PASS

5. reference mean 优于 reference median，并比 reference max 更稳定：
   PASS

6. Self-reference score 的跨对象 AUC：
   PASS，约 0.94–0.95

7. Self-reference score 的对象级 correct > wrong/shuffle：
   PASS

8. Sigmoid object gate 的 gate gap：
   PASS，约 +0.18–+0.24

9. 三训练 seed、三种 negative mode 的冻结复现：
   PASS

10. C1.6 Object-level Scalar Gate 总门：
    PASS
```

最准确的阶段性表述是：

> visual-only pairwise confidence 的 raw absolute level 受对象自身可重建性影响，不能直接作为统一部署阈值；但当前观测相对于同对象 cyclic1、cyclic2、reverse 扰动版本的 self-reference margin，能够稳定消除对象 baseline，并通过 sigmoid 映射为跨对象、跨训练 seed 可复现的 object-level scalar gate。

---

## 10. 当前可部署配置

第一轮 C2 应固定使用 seed42 配置：

```text
correspondence checkpoint:
  ar_ss_flow/outputs/corr_pairwise_v3_s200_seed42/checkpoints/last.pt

self-reference config:
  mean__ref_mean

gate type:
  sigmoid_selfref

temperature:
  0.04033606723705675
```

计算流程：

```text
1. 对当前 observed binding 计算 object mean confidence；
2. 分别施加 cyclic1、cyclic2、reverse 扰动；
3. 计算三个 perturbation object mean confidence；
4. reference = 三个 perturbation mean 的平均值；
5. score = observed - reference；
6. g_object = sigmoid(score / 0.04033606723705675)；
7. cond_final = cond_base + g_object * delta_cond。
```

`g_object`应广播为：

```text
[B, 1, 1]
```

同一对象的所有 SS tokens 使用同一个 gate。

---

## 11. 下一阶段：Object-gated C2

C1.6 已通过，可以进入小规模 C2 机制实验，但第一轮仍不应直接扩大训练或跑大规模 mesh。

### 11.1 必须包含的路径

```text
1. stock
2. residual-off / gate=0
3. ungated residual
4. correct self-reference object gate
5. wrong-pose self-reference object gate
6. visual-shuffle self-reference object gate
7. matched constant gate
```

### 11.2 严格工程条件

```text
residual-off 或 gate=0：
  condition、velocity 和 rollout 必须 bit-exact 等于 stock

correct / wrong / shuffle：
  使用同一 target、noise、t 和基础 condition

gate：
  只能乘附加 residual
  不能覆盖或重新缩放 cond_base
```

固定形式：

```text
cond_final = cond_base + g_object * delta_cond
```

### 11.3 C2 核心比较

Object gate 已经证明能检测 binding quality，但还没有证明它能改善 SS Flow。因此 C2 必须验证：

```text
correct object gate
>
matched constant gate
>
wrong-pose object gate / shuffle object gate
```

其中 `matched constant gate` 应匹配 correct gate 的平均强度，用于排除“仅仅因为 residual scale 更合适”这一解释。

### 11.4 C2 准入判断

只有同时看到：

```text
1. correct gate 相对 stock 有稳定正 Flow gain；
2. correct gate 优于 matched constant gate；
3. correct gate 优于 wrong/shuffle gate；
4. gate=0 bit-exact stock；
5. sparse occupancy、component 和 precision 不明显退化；
```

才能认为 object-level correspondence gate 已经从 detector 转化为有效的 SS Flow 控制机制。

---

## 12. 阶段裁决

```text
C1 pairwise correspondence detector:
  PASS

C1.5 raw/local/percentile calibration:
  部分信号存在，但不能形成可靠 voxel/region gate

C1.6 raw absolute object scalar:
  FAIL

C1.6 self-reference score:
  PASS

C1.6 sigmoid self-reference object gate:
  PASS

三 seed frozen validation:
  PASS

是否允许进入 object-gated C2:
  YES

是否允许宣称 SS Flow / mesh 已改善:
  NO，仍需 C2 因果机制实验
```

最终结论：

> 当前最可靠的空间尺度是 object level。`mean__ref_mean + sigmoid` 已经通过三 seed、三种 pose corruption 和 visual shuffle 的冻结验证，可以作为下一阶段 SS Flow residual 的 object-level scalar gate。它已证明是稳定的 binding-quality detector，但是否能改善 Flow 和最终几何，必须由 C2 的 stock、ungated、matched-constant、wrong-gate 和 shuffle-gate 对照进一步验证。


# C2 scale=0.5 新噪声复验与阶段性结论

### 实验目的

上一轮C2在：

```text
residual_scale = 1.0
noise seeds = 42,43,44
```

条件下，correct object gate的平均Flow gain略为负，且存在少数对象的灾难性退化。Postmortem显示当前residual更可能是幅度过大，而非方向完全无效。

因此，本轮固定：

```text
checkpoint:
  ar_ss_flow/outputs/c2_object_gate_pose_lifting_train16_s50_seed42/checkpoints/last.pt

residual_scale:
  0.5

new noise seeds:
  45,46,47

t values:
  0.1,0.3,0.5,0.7,0.9

fresh objects:
  indices 16-63
```

不重新训练adapter，不修改C1.6 gate，不根据本轮结果继续调整scale。

---

### 基本结果

```text
eligible objects = 27
records          = 405
residual-off max abs difference = 0.0
positive t count = 4 / 5
```

`residual_off`仍然bit-exact等于stock，说明C2 residual路径没有破坏基础SS Flow旁路。

---

### 各时间点correct-gated收益

|   t |   Mean gain | Median gain |     Minimum |     Maximum |
| --: | ----------: | ----------: | ----------: | ----------: |
| 0.1 | +0.00002078 | +0.00000653 | -0.00076726 | +0.00239500 |
| 0.3 | -0.00000039 | +0.00009085 | -0.00438314 | +0.00357038 |
| 0.5 | +0.00019797 | +0.00015776 | -0.00078397 | +0.00301724 |
| 0.7 | +0.00021119 | +0.00016902 | -0.00124748 | +0.00226683 |
| 0.9 | +0.00020838 | +0.00020736 | -0.00118996 | +0.00214804 |

结果表明：

```text
t=0.1:
  接近零但略正

t=0.3:
  平均值近似为零
  仍存在较明显的对象级负长尾

t=0.5 / 0.7 / 0.9:
  平均值和中位数均稳定为正
```

因此，当前residual的主要有效区间位于：

```text
t >= 0.5
```

---

### 各分支汇总

| 分支                  |       Mean gain |     Median gain |            最差对象 |    正收益对象比例 | Mean delta RMS |
| ------------------- | --------------: | --------------: | --------------: | ---------: | -------------: |
| residual-off        |     +0.00000000 |     +0.00000000 |     +0.00000000 |      0.00% |       0.000000 |
| ungated             |     -0.00000559 |     +0.00015937 |     -0.00391503 |     59.26% |       0.003509 |
| matched constant    |     +0.00011078 |     +0.00008980 |     -0.00036893 |     66.67% |       0.002262 |
| permuted correct    |     +0.00008915 |     +0.00008830 |     -0.00067585 |     66.67% |       0.002275 |
| correct gate        | **+0.00012759** | **+0.00012922** | **-0.00028217** | **74.07%** |       0.002238 |
| cyclic1 gate        |     +0.00009573 |     +0.00007875 |     -0.00029063 |     74.07% |       0.001636 |
| cyclic2 gate        |     +0.00009179 |     +0.00007999 |     -0.00030706 |     74.07% |       0.001589 |
| reverse gate        |     +0.00008561 |     +0.00005796 |     -0.00028217 |     74.07% |       0.001455 |
| visual-shuffle gate |     +0.00007625 |     +0.00005328 |     -0.00034388 |     74.07% |       0.001510 |

与上一轮`scale=1.0`相比，correct gate发生了以下变化：

```text
mean gain:
  -0.00005393
  ->
  +0.00012759

median gain:
  +0.00005389
  ->
  +0.00012922

positive object rate:
  55.56%
  ->
  74.07%

positive t:
  2 / 5
  ->
  4 / 5

minimum object gain:
  -0.00375260
  ->
  -0.00028217

mean delta RMS:
  0.00447532
  ->
  0.00223753
```

这说明上一轮的主要失败原因确实是residual幅度过大，而不是residual方向完全没有泛化能力。

---

### 晚时间段诊断

对：

```text
t >= 0.5
```

单独汇总得到：

```text
objects              = 27
records              = 243
mean gain            = +0.00020585
median gain          = +0.00021311
minimum object gain  = -0.00028727
maximum object gain  = +0.00090536
positive object rate = 77.78%

object-bootstrap 95% CI:
  [+0.00010709, +0.00030829]
```

Bootstrap置信区间完整位于零以上，说明晚时间段的正收益不只是少数对象或单个noise seed造成的偶然结果。

三个晚时间点的平均收益也非常接近：

```text
t=0.5  +0.00019797
t=0.7  +0.00021119
t=0.9  +0.00020838
```

因此可以认为：

> 当前local pose-lifting residual在Flow后半程具有稳定、可复现但幅度较小的正收益。

---

### Correct gate与各对照的比较

| 对照                  | Correct-minus-control mean |      Median |   对象胜率 | 结果        |
| ------------------- | -------------------------: | ----------: | -----: | --------- |
| matched constant    |                +0.00001680 | -0.00000073 | 44.44% | FAIL      |
| permuted correct    |                +0.00003844 | +0.00000210 | 55.56% | PASS      |
| ungated             |                +0.00013318 | -0.00002501 | 40.74% | 不作为对象特异主门 |
| cyclic1 gate        |                +0.00003186 | +0.00001688 | 74.07% | PASS      |
| cyclic2 gate        |                +0.00003579 | +0.00002489 | 62.96% | PASS      |
| reverse gate        |                +0.00004198 | +0.00001995 | 70.37% | PASS      |
| visual-shuffle gate |                +0.00005134 | +0.00003926 | 74.07% | PASS      |

---

### Correct gate优于permuted gate

`permuted_correct`保留了correct gate的完整数值分布，只打乱gate与对象之间的对应关系。

本轮结果：

```text
mean difference   = +0.00003844
median difference = +0.00000210
object win rate   = 55.56%
```

达到预设门槛。

这说明：

> correct gate与具体对象之间的绑定关系开始包含真实的residual utility信息，而不仅仅是gate总体均值或分布产生的缩放效果。

这一结果相比上一轮：

```text
object win rate:
  44.44%
  ->
  55.56%
```

发生了方向性改善。

不过当前优势仍然较小，属于边缘通过，不能视为强对象特异控制证据。

---

### Correct gate优于错误绑定gate

Correct gate稳定优于全部错误绑定或视觉打乱gate：

```text
correct > cyclic1:
  mean difference = +0.00003186
  median          = +0.00001688
  object win      = 74.07%

correct > cyclic2:
  mean difference = +0.00003579
  median          = +0.00002489
  object win      = 62.96%

correct > reverse:
  mean difference = +0.00004198
  median          = +0.00001995
  object win      = 70.37%

correct > visual shuffle:
  mean difference = +0.00005134
  median          = +0.00003926
  object win      = 74.07%
```

四组比较均满足：

```text
mean > 0
median > 0
object win rate >= 60%
```

因此可以认为：

> C1.6 visual binding gate不再只是普通的全局shrinkage，它能够区分正确image-pose binding与错误或shuffle binding对SS residual的控制效果。

---

### Correct gate没有优于matched constant

当前唯一失败的科学门是：

```text
correct gate > matched constant
```

结果：

```text
correct gate mean gain:
  +0.00012759

matched constant mean gain:
  +0.00011078

mean difference:
  +0.00001680

median difference:
  -0.00000073

object win rate:
  44.44%
```

虽然correct gate平均收益略高，但：

```text
median difference < 0
object win rate < 50%
```

说明额外收益主要集中在少数对象，不能证明对多数对象使用连续的逐对象gate优于统一固定gate。

因此当前不能宣称：

> 连续object-specific scalar是比固定residual scale更优的部署形式。

更准确的判断是：

```text
固定scale=0.5：
  已经解决了大部分residual幅度问题

object gate：
  能识别错误binding并进行保护性衰减

object gate：
  尚不能精确预测每个正常对象的最优residual幅度
```

---

### 为什么formal pass仍为False

本轮检查结果：

```text
residual_off_bit_exact_stock:
  PASS

correct_gate_mean_gain_positive:
  PASS

correct_gate_median_gain_positive:
  PASS

correct_gate_positive_t:
  PASS

correct_beats_permuted_gate:
  PASS

correct_beats_wrong_and_shuffle_gates:
  PASS

correct_beats_matched_constant:
  FAIL
```

因此：

```text
formal_passed = False
```

这里的`False`不再意味着整个C2 residual路线失败，而是只意味着最严格假设：

```text
逐对象连续gate优于固定常数gate
```

尚未成立。

---

### 阶段拆分

当前C2应拆分为三个子结论：

```text
C2a residual viability:
  PASS

C2b correct binding beats corrupted binding:
  PASS

C2c continuous object-specific gate beats fixed constant:
  FAIL
```

也可以概括为：

```text
residual是否有用：
  是，scale=0.5且尤其t>=0.5时有稳定小幅正收益

binding gate是否包含控制信息：
  是，correct优于permuted和所有wrong/shuffle gate

连续gate是否优于固定scale：
  暂无证据
```

---

### 当前最合理的部署解释

本轮形成了以下结构：

```text
correct gate > wrong/shuffle gate
correct gate > permuted gate
correct gate ≈ matched constant
```

这说明C1.6 gate当前更适合作为：

```text
低confidence safety cap
```

而不是：

```text
全范围连续residual比例旋钮
```

推荐的机制不再是简单地：

```python
effective_scale = 0.5 * g_object
```

而应考虑：

```python
effective_scale = base_scale * safety_cap(g_object)
```

其中：

```text
base_scale = 0.5

正常或高confidence对象：
  使用接近固定base scale的residual

低confidence、wrong或shuffle-like对象：
  只允许向下衰减

禁止：
  因高gate而把residual放大到base scale以上
```

一种可测试的形式是：

```python
base_scale = 0.5
gate_floor = 0.5

normalized_gate = (
    (g_object - gate_floor)
    / max(1.0 - gate_floor, 1.0e-6)
)

safety_cap = normalized_gate.clamp(0.0, 1.0)

effective_scale = base_scale * safety_cap
```

但该形式仍需使用独立validation预先确定`gate_floor`，不能直接在当前fresh结果上继续调参。

---

### 收益规模判断

全时间段correct gate收益：

```text
+0.00012759
```

晚时间段收益：

```text
+0.00020585
```

因此当前收益具有以下性质：

```text
方向：
  正

新noise复现：
  是

晚时间段bootstrap显著：
  是

绝对幅度：
  小

是否足以保证明显mesh提升：
  否

是否值得保留并继续低成本验证：
  是
```

---

### 当前阶段裁决

```text
C1.6 self-reference binding detector:
  PASS

C2 residual-off stock equivalence:
  PASS

C2 residual scale=1.0:
  FAIL，幅度过大

C2 residual scale=0.5:
  PASS

C2 full-t positive residual generalization:
  PASS

C2 late-time t>=0.5 residual generalization:
  PASS

C2 correct gate > permuted gate:
  PASS，边缘

C2 correct gate > wrong/shuffle gate:
  PASS

C2 correct gate > matched constant:
  FAIL

C2 overall:
  PARTIAL PASS
```

最终结论：

> 将residual scale从1.0降低到0.5后，当前local pose-lifting residual在新noise seed上恢复为稳定的小幅正收益，且晚时间段`t>=0.5`的object-bootstrap置信区间完整位于零以上。Correct binding gate也首次稳定优于permuted gate以及全部wrong-pose和visual-shuffle gate，说明C1.6 gate确实包含对象绑定相关的控制信息。但correct gate仍未稳定优于matched constant，表明其当前价值更接近错误binding下的单向安全保护，而不是精确预测每个正常对象的最优residual幅度。下一步应优先测试“固定base scale + 低confidence safety cap”，而不是继续对连续gate做小数点级调参。
