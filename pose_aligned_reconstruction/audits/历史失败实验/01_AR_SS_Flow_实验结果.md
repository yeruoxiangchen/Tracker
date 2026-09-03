# AR SS Flow实验结果

> 本文件是`ar_ss_flow`唯一实验结果记录。后续结果统一追加到本文件末尾。

## 2026-07-14：P1完成后协议修正与smoke

## 结论

P1缓存可以保留，不需要重建，也不需要修复`stock_condition`。正式训练暂不启动，下一硬门是full-64方向性P2。

## Stock condition审计

独立加载未经VGGT替换的原生ReconViaGen condition路径，对P1缓存中的2/4/8 view代表样本进行比较。

结果：

```text
2-view: torch.equal(fp16)=true, max_abs=0, RMS=0
4-view: torch.equal(fp16)=true, max_abs=0, RMS=0
8-view: torch.equal(fp16)=true, max_abs=0, RMS=0
```

因此builder为depth head重新加载完整VGGT没有改变缓存的stock condition；缓存仍是原生ReconViaGen基线。

报告：

```text
ar_ss_flow/outputs/p1_stock_condition_native_audit_20260714_retry/report.json
```

## 代码修正

1. adapter时间通道从原始`0..1000`改为`t / 1000`，模型版本升级到`local_pose_lifting_velocity.v2`。旧smoke checkpoint作废。
2. P2保留normalized-MSE non-identity，同时增加跨view weighted variance/cosine方向和GT局部支持辅助指标。
3. `pose_shuffle`、`depth_corrupt`定义为hard invalid，训练回到stock。
4. `pose_perturb`定义为部署鲁棒性输入，训练到target并约束接近correct、优于stock。
5. P4 hard corruption near-stock改为`mean(abs(gain))`，不允许正负抵消。
6. P4新增correct-vs-stock对象胜率、hard corruption逐t方向、pose perturb鲁棒门和neutral max-abs硬门。
7. 多seed汇总新增完整协议一致性检查。

## Directional P2 smoke

单个4-view样本的non-identity结果保持不变：

```text
pose_perturb normalized MSE = 0.179792
pose_shuffle normalized MSE = 0.339185
depth_corrupt normalized MSE = 0.00323648
```

correct相对pose perturb和pose shuffle具有正的跨view方向；该样本的depth corruption没有跨view方向优势。三类GT局部支持方向均未通过，所以GT指标当前只作辅助诊断，不能凭单样本设成硬门。

## v2训练与评估smoke

新版单卡BF16 s2：

```text
applied updates = 2/2
nonfinite attempts = 0
model/optimizer/scaler finite = PASS
physical-off/null = bit-exact stock
time normalization = t_div_1000
```

第1步hard depth corruption使用`hard_invalid_to_stock`；第2步pose perturb使用`robust_perturbation_to_target`。第2步state、visual、metadata、fusion和output梯度均已启动。

新版P4单记录smoke成功生成完整协议报告，neutral velocity max-abs为0并进入decision。s2不满足pose perturb收益门是预期现象，不构成工程失败。

## 下一步

1. 运行full-64方向性P2，分别查看三类corruption的non-identity、跨view方向和GT辅助方向分布。
2. full P2通过后运行新版BF16 s5和2-GPU DDP s1，并要求finite audit通过。
3. 三项工程门都通过后，才重新从scratch运行三个训练seed的s50。
4. P4必须同时通过correct-vs-stock、hard corruption、pose perturb鲁棒性、neutral preservation和协议一致性。

所有正式命令已追加到`ar_ss_flow/命令说明.txt`第10节；每个阶段会写独立exit-code文件。
## 2026-07-14：P2方向审计与v2工程门结果
## 最终裁决

```text
10.2 full-64 directional P2: FAIL，exit code = 2
10.3 single-GPU BF16 s5: PASS，train/audit exit code = 0/0
10.4 2-GPU DDP s1: PASS，train/audit exit code = 0/0
```

当前不能进入三个seed的正式s50。10.3/10.4只证明新版模型的数值、梯度、checkpoint和DDP链路正确，不能覆盖10.2的几何准入失败。

但10.2也不能直接解释为pose-lifting路线失败。结果显示：corruption全部生效，correct在GT target附近具有稳定优势；失败主要来自当前cross-view方向指标和未归一化far-support条件。

## 10.2 Non-identity

三类corruption与correct volume具有明确差异：

| mode | normalized MSE mean | median | non-identity pass |
|---|---:|---:|---:|
| pose perturb | 0.297443 | 0.269515 | 100% |
| pose shuffle | 0.362566 | 0.305679 | 100% |
| depth corrupt | 0.025318 | 0.005275 | 96.875% |

因此corruption实现有效，不存在“输入实际上没有变化”的问题。

## 10.2 Cross-view方向

当前硬阈值要求每类corruption的方向通过率不低于60%。实际为：

| mode | cross-view direction pass | variance direction mean | cosine direction mean |
|---|---:|---:|---:|
| pose perturb | 75.0% | +0.013448 | +0.001248 |
| pose shuffle | 57.8125% | +0.004647 | +0.000433 |
| depth corrupt | 17.1875% | -0.033818 | -0.003086 |

pose perturb通过；pose shuffle略低于预注册阈值；depth corruption明显反向。

按view数量拆分后：

| mode | 2-view | 4-view | 8-view |
|---|---:|---:|---:|
| pose perturb | 66.7% | 66.7% | 94.7% |
| pose shuffle | 33.3% | 61.9% | 84.2% |
| depth corrupt | 8.3% | 19.0% | 26.3% |

pose shuffle随view数增加明显改善，说明多视图数量确实提高了几何可辨识性。

depth corruption反向不能直接解释为corrupt几何更正确。当前depth corruption会改变depth权重并减少有效view；更少的view天然可能降低weighted variance、提高表面cosine，从而通过“丢弃冲突观察”获得更好的表观一致性。当前cross-view指标没有控制有效view集合和总权重，因而不适合作为depth corruption的单独硬门。

此外，缓存的VGGT/DINO 3072维特征并未被训练成严格的跨视图同点不变表示。正确pose下同一3D位置的不同视角特征仍可因遮挡、纹理、光照和视角而变化。

## GT局部支持

原联合条件要求：

```text
target support: correct > corrupt
far support: correct <= corrupt
```

联合通过率较低：

| mode | joint pass | target direction mean | far direction mean |
|---|---:|---:|---:|
| pose perturb | 46.875% | +0.070816 | -0.000932 |
| pose shuffle | 46.875% | +0.066465 | -0.003396 |
| depth corrupt | 15.625% | +0.155703 | -0.030047 |

但拆开后，correct在target区域的优势非常稳定：

| mode | target support correct > corrupt | target-vs-far contrast positive |
|---|---:|---:|
| pose perturb | 93.75% | 89.0625% |
| pose shuffle | 89.0625% | 89.0625% |
| depth corrupt | 95.3125% | 92.1875% |

这里的`target-vs-far contrast`为：

```text
(correct_target - correct_far)
-
(corrupt_target - corrupt_far)
```

该指标三类均有接近90%或更高的对象正方向，说明correct lifting确实更集中于GT几何附近。

absolute far条件失败的主要原因是correct通常保留更多总support，target和far都可能同步增加。未对总support mass归一化时，直接要求correct far绝对值更小，会把“总体覆盖更多但target浓度更高”判成失败。

## 10.3 BF16 s5

```text
applied optimizer updates = 5/5
nonfinite attempts = 0
model parameters finite = true
optimizer state finite = true
scaler state finite = true
```

5步覆盖：

```text
depth corrupt: 2次
pose perturb: 2次
pose shuffle: 1次
```

第1步zero-init时只有output projection获得梯度；第2步起state、visual、metadata、fusion和output均有finite nonzero gradient，符合zero-init adapter的预期启动顺序。

5步内relative gain约为`1e-5`到`1e-4`且正负波动，不能作为收益结论。

## 10.4 2-GPU DDP s1

```text
applied optimizer updates = 1/1
nonfinite attempts = 0
step accounting = PASS
checkpoint/optimizer/scaler finite = PASS
```

DDP工程链路成立。由于只有一步且adapter为zero-init，本次只验证了output projection梯度，不能验证模型学习效果。

## 下一步：P2.1受控方向审计

不直接放宽现有60%阈值，也不启动s50。先在同一P1缓存上增加一次P2.1，不需要重建缓存。

### 1. Support concentration作为主方向

对每种volume先按总support mass归一化，再比较：

```text
target_mass_fraction = support_mass(target dilation) / total_support_mass
far_mass_fraction = support_mass(far region) / total_support_mass
target_far_contrast = target_mass_fraction - far_mass_fraction
```

hard corruption要求：

```text
correct target_mass_fraction > corrupt
correct target_far_contrast > corrupt
object win rate >= 65%
mean > 0
median > 0
```

pose perturb作为鲁棒输入，要求其concentration接近correct，而不是回到stock语义。

### 2. Cross-view指标改为受控诊断

只在correct和corrupt具有相同active-view集合、且至少两个view有效的voxel上比较。每个voxel把两种模式的总权重归一到一致，并分别对VGGT与DINO通道做L2 normalization。

在该修正完成前：

```text
cross-view consistency只作诊断；
不得用depth corruption通过减少有效view得到的低variance作为正确方向。
```

### 3. Depth corruption增加专用方向

depth corruption应额外比较：

```text
sparse-point calibrated depth residual
target-region depth support concentration
effective view count
```

不能继续与pose shuffle共用唯一的raw-feature variance门。

### 4. P2.1准入后才训练

```text
P2.1 PASS
  -> 三个from-scratch overfit16 s50
  -> 新P4 teacher-forced评估

P2.1 FAIL
  -> 不训练当前adapter
  -> 停止该lifting定义，转向显式ray/voxel监督或第二主线
```

## 当前判断

当前结果不是“可以长训”，也不是“pose lifting已经失败”。更准确的结论是：

> 输入corruption明确生效，correct volume在GT目标附近具有稳定且很强的相对浓度优势；但现有cross-view和absolute far-support硬门没有控制有效view数量与总support mass，因此P2正式协议未通过。先修正方向审计，再决定是否投入三seed训练。

## 2026-07-14：P0.5 Object-frame Sim(3)实现与smoke

坐标审查确认当前16³网格位于TRELLIS canonical cube `[-0.5,0.5]³`，只经过固定`pixal3d_rotation`。当前合成相机本身处于物体中心化世界坐标，因此原数据隐含已知object pose；真实AR相机位姿不提供该对齐。

新增显式`object_to_world` Sim(3)投影路径，并完成三路audit：

```text
transformed cameras + no object recovery
transformed cameras + oracle object_to_world
transformed cameras + point-only PCA estimate
```

1-sample GPU smoke：

```text
oracle normalized visual MSE = 9.40803e-13
no recovery normalized visual MSE = 1.09092
point-PCA normalized visual MSE = 0.644671
```

这证明完整visual/depth lifting在提供oracle Sim(3)时可恢复原volume；遗漏object frame会产生数量级明显的错误。point-PCA在该样本上优于完全不恢复，但仍远未达到oracle，不能据此进入训练。

下一步先运行8个对象、三个transform seed的P0.5 pilot；pilot全部通过后再扩到64样本。若oracle不稳定，先修投影；若oracle通过而point-PCA失败，则增加gravity/reference-camera与visual-hull约束，不启动s50。

## 2026-07-14：`pixal3d_multiview`对照复核

### 总体判断

`pixal3d_multiview`与当前`ar_ss_flow`的几何起点高度相似：都将canonical 3D位置经K/T投影到多视图patch，采样图像特征并在3D位置上聚合。因此，当前代码并没有自动解决Pixal3D路线已经暴露的image-pose弱约束。

两者也不相同：`pixal3d_multiview`修改`cond['proj']`并可微调Pixal3D sparse flow；`ar_ss_flow`保留ReconViaGen stock condition/flow，只在16³同体素位置加velocity residual，并使用correct/corrupt配对目标与neutral stock preservation。后者更可控，但聚合前没有显式保留view-pair correspondence。

### Pixal3D代码与实验证据

基础路径为：

```text
canonical grid
  -> 按K/T投影到每个view
  -> 采样DINO patch feature
  -> mask/front-depth support
  -> 按voxel跨view聚合
  -> cond['proj']
  -> Pixal3D sparse flow
```

`ViewGatedAggregator`使用DINO特征和11维几何量为每个view/voxel打分，但不直接要求不同view的对应表面特征一致。后续`PoseConsistencyHead`才显式使用`|f_i-f_j|`、`f_i*f_j`、几何差、cosine和support统计，以correct-vs-wrong ranking单独训练。

32个val对象的独立head score：

```text
correct       = 0.5439
cyclic1/2     = 0.4225 / 0.4304, correct win 32/32
reverse       = 0.4199, correct win 32/32
cross_sample  = 0.4229, correct win 32/32
noise/large   = 0.2168 / 0.1082
identity      = 0.5813, correct win 12/32
```

这证明投影后的DINO特征中存在pose-sensitive signal，但identity反而高于correct，说明head仍在使用coverage/front-depth/support等shortcut。

把head prior接回sparse聚合后，这种可分性没有稳定传到生成端。Baseline sparse IoU中correct为`0.035393`，reverse为`0.034884`，correct只在`6/16`对象上胜出。加入head prior并扫描alpha后，reverse/cyclic的correct win rate仍大致处于随机水平。128对象的visible-match prior扫描也没有随alpha增大而系统提高reverse/cyclic/cross-sample胜率。

因此：独立head会识别wrong pose，不等于将head作为view权重就能改善生成。

### 与`ar_ss_flow`的关键差异

| 项目 | `pixal3d_multiview` | `ar_ss_flow` |
|---|---|---|
| image feature | 主要是DINO patch | VGGT + DINO patch，3072维 |
| object frame | mask+pose visual hull估计临时volume | 合成cache隐含canonical frame；刚加Sim(3) audit |
| 注入位置 | Pixal3D `cond['proj']` | ReconViaGen SS velocity residual |
| 交互 | 按voxel跨view聚合，后期有pairwise head | 聚合后的同voxel pointwise MLP |
| flow | sparse flow可微调 | stock flow冻结 |
| stock保护 | 非严格结构保证 | off/null/neutral显式保留stock |

`ar_ss_flow`的空间位置更明确，也不会用全局attention打散3D对应。但各view特征在进入adapter前已被平均成单个volume，view identity和pairwise disagreement被丢失。对某个voxel，adapter无法分辨聚合特征来自多视图同一表面点的一致支持，还是多个错位patch的平均。

### 下一步修正

P0.5只解决object/world/canonical坐标gauge，不解决image-pose specificity。在启动当前adapter长训前，应增加保留view身份的局部对应层：

1. 对每个16³ voxel保留per-view `[V,C]`特征、深度残差、mask和visibility，不在监督前平均。
2. 只在至少两个view可靠可见的voxel构造pairwise match，correct最小化跨view discrepancy，shuffle/cross-sample使用margin ranking。
3. 增加held-out reprojection cycle：用其他view lift得到的3D feature重投影回held-out view，correct pose应比shuffle更好重建该view的VGGT/DINO patch。这个目标不需要mesh GT，且直接约束image-pose对应。
4. correspondence confidence应作为visual residual的必要gate，而不只是加到view softmax logit上。
5. 生成目标改为local masked loss：match/support voxel优化correct-to-target，hard-corrupt/unmatched voxel优化stock preservation，neutral仍为bit-exact stock。
6. 先达到correct对shuffle/reverse/cross-sample的held-out reprojection对象胜率至少65%，且差值应随view数增加；未通过时不启动flow长训。

最终裁决：

> `ar_ss_flow`比`pixal3d_multiview`更严格地保护stock并限制局部修正，但它尚未解决image-pose弱约束；聚合前甚至丢失了Pixal3D后期pairwise head所保留的view身份信息。下一步不应只长训当前pointwise adapter，而应先建立per-view local correspondence / held-out reprojection任务，再将其置信度接到SS Flow的局部监督上。

## 2026-07-15：跨视图 Correspondence 分支复核与后续路线

### 总体裁决

基于最新 C1、C1.5、C1.6 和 C2 结果，类似 Pixal3D 的跨视图方法仍有可改进空间，但继续方向必须从：

```text
投影特征 -> 跨视图均值/加权均值 -> SS residual
```

改成：

```text
投影特征
  -> 显式 view-pair correspondence 表示
  -> SS 当前状态参与的同体素局部融合
  -> actual correct/corrupt volume 配对训练
  -> stock-exact residual
```

当前结果已经否定了两件事：

1. 只靠均值聚合或 learned normalized view weights，不能稳定把 pose-sensitive signal 转成生成收益。
2. 只用 object scalar gate 调节同一个 correct residual，不能证明模型利用了错误位姿下的局部视觉几何差异。

但结果也确认了两件值得保留的事实：

1. 聚合前的 visual pairwise correspondence 信号真实存在，并能跨对象、跨 checkpoint seed 区分 correct、cyclic、reverse 和 visual shuffle。
2. 局部 SS velocity residual 在 `scale=0.5`、`t>=0.5` 时出现了小幅、可复现的正 teacher-forced gain，因此注入位置和残差方向没有被完全否定。

所以不应停止整个跨视图路线，但应停止继续微调当前 C2 标量门版本。

### C1 已经证明了什么

C1 v3 在聚合前计算每个 voxel 的 view-pair visual similarity，并保留 per-view confidence。三 seed fresh-object 结果中：

```text
pose_cyclic1 pairwise win 约 94.9%
pose_cyclic2 pairwise win 约 94.7%
pose_reverse pairwise win 约 93.6%
```

`visual_zero` 和 `visual_shuffle` 会使该信号崩溃或反向，而关闭额外 geometry-pair branch 基本不改变结果。这里的正确解释是：

```text
K/T 已经决定从哪个图像 patch 采样；
采样后的视觉内容之间确实包含 image-pose binding 信号；
额外 geometry-pair MLP 没有提供更多判别力。
```

这不表示 pose 没有作用。pose 已经通过投影采样路径发挥作用，只是后续显式 geometry similarity 项没有新增收益。

### 为什么 learned weighted aggregation 没有收益

当前 pairwise 聚合先计算：

```text
final_weight = physical_weight * per_view_confidence
```

再做归一化平均：

```text
sum(feature * final_weight) / sum(final_weight)
```

如果同一 voxel 各 view 的 confidence 一起升高或降低，其绝对幅度会被分母抵消。这样 detector 最稳定的 object/voxel confidence scale 没有进入最终 visual representation，只剩相对 view reweighting。

实验也与此一致：learned pairwise weighting 与 uniform pairwise aggregation 的重建差异很小。因此下一版不能继续只把 correspondence 当 softmax/normalized weight，而应把下列量作为显式输入特征保留下来：

```text
pair embedding / pair similarity
absolute confidence logit
pairwise disagreement
effective view count
support/depth residual
corrected visibility
```

### C1 voxel/region gate 失败的含义

C1.5 和 C1.6 的结果表明：

```text
raw voxel AUC 约 0.63-0.66
3x3x3 local mean 没有改善 fresh AUC
top-percentile 区域比随机区域更可重建
但 wrong-pose confidence 经常选择同一批区域
```

因此 confidence 同时包含：

```text
pose-sensitive correspondence
+ pose-insensitive reconstructability / texture / visibility
```

object self-reference 能通过同对象 cyclic/reverse 基线消除第二部分，所以 object-level AUC 可达到约 `0.94-0.95`。但这只证明它是 binding-quality detector，不证明它能定位每个 voxel 的正确 residual，也不证明它能预测 residual 最优幅度。

### C2 当前代码实际验证的内容

C2 训练始终构造：

```text
volume_from_sample(sample, mode="correct")
```

C2 评估也只构造一次 correct volume 和一次 `raw_delta`，然后对同一个 residual 分别乘：

```text
correct gate
cyclic gate
reverse gate
visual-shuffle gate
matched constant gate
```

因此当前 C2 比较的是：

```text
同一个 correct-pose residual 的不同标量幅度
```

而不是：

```text
correct-pose visual volume 产生的 residual
vs
wrong-pose / shuffled visual volume 产生的 residual
```

这解释了为什么 correct gate 可以优于 wrong/shuffle gate，却仍不能稳定优于 matched constant。它证明 detector scalar 与 residual utility 有弱相关性，但还没有证明跨视图 correspondence 表示真正进入并改善了 SS Flow。

### C2 数值结果的合理定位

`scale=0.5`、新 noise seeds 45/46/47、27 个 eligible fresh objects 的结果为：

```text
correct gate mean gain          = +0.00012759
correct gate positive objects   = 74.07%
t>=0.5 mean gain                = +0.00020585
t>=0.5 bootstrap 95% CI         = [+0.00010709, +0.00030829]
residual-off max abs diff       = 0
```

正面部分是晚时间段 gain 具有稳定方向，stock bypass 也严格成立。限制是：

```text
correct - matched constant mean = +0.00001680
median                          = -0.00000073
object win rate                 = 44.44%
formal pass                     = false
```

因此当前 object gate 更适合定位为低置信度 safety cap，而不是 object-specific amplitude controller。现阶段不能据此宣称 sampling、occupancy 或 mesh 已改善，因为评估仍是单步 teacher-forced Flow MSE。

### 运行新训练前的硬审计

先增加真正的端到端 paired branch 评估，不训练新模型：

```text
correct volume + correct correspondence features
cyclic/reverse volume + 对应 correspondence features
visual-shuffle volume + 对应 correspondence features
correct volume + matched constant gate
stock
```

所有分支固定：

```text
同一 target
同一 x_t / noise / t
同一 stock condition
```

当前只替换 scalar gate 的 C2 结果作为旧基线保留，但不能替代这个审计。

同时只预注册一次时间策略：

```text
t < 0.5: residual off 或平滑 ramp
t >= 0.5: 固定 base scale 0.5
object detector: 只允许降低 scale，不允许放大
```

不再做连续小数 scale sweep。若上述真实 corrupt-volume 审计中 wrong/shuffle volume 也稳定改善，应停止当前 adapter，因为它仍在学习 generic visual-volume correction。

### 推荐的新模型：C3 Pair-feature-conditioned Local SS Adapter

第一版保持 stock Flow、decoder 和 stock condition 冻结，并保留 exact-off 路径。对每个 16^3 voxel，不先压成单个均值特征，而是保留最多 `V(V-1)/2` 个 view-pair token：

```text
pair token = [
  projected visual feature_i,
  projected visual feature_j,
  abs(f_i - f_j),
  f_i * f_j,
  pair confidence logit,
  depth residual_i/j,
  visibility/support_i/j,
  ray/view geometry
]
```

SS 当前状态作为 query：

```text
q_i = projection(x_t[i], stock_velocity[i], normalized_t)
```

只查询同一个 voxel 的 pair tokens，或者最多查询 `3x3x3` 局部邻域。禁止 4096x4096 全局 attention。输出形式为：

```text
delta_v[i] = zero_init_output(local_pair_fusion(q_i, pair_tokens_i))
```

这与失败的 Pixal3D 聚合有三个实质区别：

1. correspondence 不再只是归一化 view weight，而是 Flow adapter 的显式内容输入。
2. SS 当前 noisy state 决定哪些 pair evidence 对当前去噪阶段有用。
3. correct、wrong 和 shuffle 使用各自真实构造的 visual evidence，而不是只替换标量 gate。

### 监督需要同步修正

当前 C1 把所有 correct common voxels 标成正样本，把所有 wrong common voxels 标成负样本。这会产生标签噪声：

```text
correct voxel 可能并非真实可见表面；
wrong pose 可能因对称、重复纹理或邻域容差仍采到相似内容；
best-of-3x3 held-out match 会弱化精确 pose 敏感性。
```

合成数据阶段应利用 mesh/render buffer 只提供训练标签，不进入推理输入。推荐生成：

```text
per-view exact depth
triangle / surface id
barycentric 或 world-space hit point
occlusion visibility
```

pair positive 定义为两个 patch 命中同一表面位置，pair negative 定义为不同表面、遮挡或明确错位。然后将该 surface correspondence teacher 蒸馏到部署时可用的 RGB、K/T、predicted depth、mask 和 sparse points。

held-out probe 应同时报告：

```text
exact projected patch reconstruction
3x3 tolerant reconstruction
```

模型选择以 exact 指标为主，tolerant 指标只作鲁棒性诊断。

### 真实 AR 还缺少的前置条件

当前合成数据隐含已知 canonical object frame。P0.5 已证明遗漏 object-to-world Sim(3) 会导致 lifting volume 数量级错误，point-PCA 只能部分恢复。因此跨视图方法在真实 AR 上继续之前还必须解决：

```text
canonical object frame / object-to-world gauge
真实 pose 小误差而非只用 cyclic/reverse
intrinsics drift
depth scale/shift error
view dropout 和动态遮挡
```

self-reference gate 只说明 observed pose 比人工 perturbation 更一致，不是绝对正确概率。若 observed pose 本身错误，或某个 perturbation 偶然更好，gate 仍可能失效。

### 建议实验顺序

```text
E0  修正 C2 评估：每个 branch 使用自己的真实 volume/evidence
E1  固定 t>=0.5、scale=0.5，做 teacher-forced actual-corruption 审计
E2  同一 checkpoint 做 SS sampler rollout 和 decoded coords/occupancy 指标
E3  建立 surface-supervised pair correspondence 数据与 exact held-out probe
E4  训练 C3 local pair-feature adapter，先 overfit16 三 seed
E5  fresh object-disjoint overfit64/validation
E6  通过后才讨论小规模 Flow LoRA 或更长训练
```

E4 的准入标准至少包括：

```text
correct < stock Flow MSE
correct < actual cyclic/reverse/shuffle Flow MSE
correct > matched constant residual
object-balanced win rate >= 65%
至少 4/5 t 为正，且 t>=0.5 稳定
off/null/unsupported voxel 严格 stock-preserving
sampler rollout 的 precision/IoU/component 不退化
```

### 停止条件

如果加入真实 surface correspondence 监督、actual-corruption branches 和 state-conditioned local pair features 后，仍不能在三 seed、object-disjoint fresh objects 上同时优于 uniform/matched-constant，并且 rollout 没有正收益，则停止跨视图特征注入 SS Flow。

此时保留的成果应是：

```text
C1 object-level image-pose consistency detector
用于 pose/data quality screening 或 residual safety cap
```

而不再把它作为生成增强模块继续扩大训练。

### 最终建议

跨视图 Pixal3D 类路线仍值得进行最后一轮结构性验证，但可改进点不是更大的 view aggregator、更多训练步数或更细的 scale sweep。最有解释力的下一步是：

> 将已验证的聚合前 view-pair correspondence 从“权重/标量 gate”升级为 SS Flow 同体素的显式条件特征，并用真实 correct/corrupt volume 与 surface-level correspondence 标签训练；object self-reference gate仅保留为不放大 residual 的安全上限。

## 2026-07-15：C3 Same-voxel Pair-feature SS Adapter 实现与工程验证

### 实现内容

本轮已将上一节建议落实为 C3，新增或修改：

```text
correspondence_lifting.py
  导出每个16^3 voxel的未聚合view-pair features、valid mask和pair index。

pair_feature_ss_flow.py
  用(x_t, stock velocity, normalized t)查询同voxel view pairs；
  结合visual volume和physical metadata输出zero-init局部velocity residual。

train_pair_feature_ss_flow.py
  用actual correct/cyclic/reverse/cross-object evidence做paired训练；
  frozen C1、stock Flow、bridge、decoder、SLAT，不启用Flow LoRA。

eval_pair_feature_ss_flow.py
  teacher-forced actual-corruption评估；
  constant-pair和spatial-permuted-pair controls；
  optional same-noise sampler rollout及decoded occupancy/component指标；
  分离benchmark_relaxed与mechanism_strict。

audit_pair_feature_training_run.py
  审计step、参数、optimizer/scaler、冻结范围和stock exact fallback。

summarize_pair_feature_multiseed.py
  检查三seed的cache hash、sample、noise、t、corruption、模型和loss协议一致性。
```

C3 adapter 约 `513,144` 个可训练参数。没有全局 `4096x4096` attention，pair attention 只发生在同一个 16^3 voxel 内。输出仍为：

```text
v = v_stock + support_gate * time_gate * scale * delta_v
```

### 已完成的验证

CPU：

```text
py_compile：PASS
unit tests：9/9 PASS
```

真实 CUDA BF16 s2 smoke：

```text
2/2 optimizer updates finite
trainable state / optimizer / scaler finite
stock Flow trainable parameters = 0
C1 frozen
physical-off max diff = 0
null max diff = 0
zero-init enabled max diff = 0
```

梯度启动符合 zero-init 预期：step 1 先只有 output projection 获得梯度；step 2 后 state、visual、metadata、pair Q/K/V、fusion 和 output 都获得 finite nonzero gradient。

真实 teacher-forced smoke 使用 actual `pose_reverse` 与 `cross_sample` evidence，完成并返回 0。真实 2-step sampler smoke 也完成：

```text
stock / correct / constant-pair / spatial-permuted-pair / pose-reverse
均完成采样和decoder occupancy统计；
每个adapted branch positive CFG calls = 2；
negative CFG calls = 2；
进程exit code = 0。
```

这些是 s2、单对象工程 smoke，delta 约 `1e-5` 到 `1e-4`，不能作为性能收益结论。正式收益必须等待 overfit16 三训练 seed、fresh48 object-disjoint、三 eval seed 和五 active t。

修正 active-t 协议后又完成了一次单对象回归：`inactive_t_max_abs_diff=0`，teacher-forced、sampler、decoder、object-balanced rollout delta 和 `rollout_benchmark` 均能正常写入报告，进程 exit code 为 0。该回归仍不计入正式收益统计。

### 评估协议修正

旧候选时间点：

```text
0.1, 0.3, 0.5, 0.7, 0.9
```

与 C3 的 `residual_t_min=0.5` 冲突。`t<=0.5` 时结构性 residual 为零，因此最多只有 2 个时间点能产生正 gain，却要求 3/5 或 4/5，原门不可达。

正式协议改为：

```text
active decision t：0.55,0.65,0.75,0.85,0.95
inactive exact-stock probe：0.4
```

inactive probe 必须 `max_abs_diff=0`，active t 再分别执行 3/5 relaxed 和 4/5 strict 判定。

### 放宽标准的裁决

可以把“是否继续跑 rollout”的门放宽为：三 seed、fresh objects 上 correct 相对 frozen ReconViaGen stock 的 mean/median 为正、object win rate 至少 55%、active t 至少 3/5 为正。该门只证明 teacher-forced objective 有可重复收益。

最终声称优于 ReconViaGen，必须再通过 same-noise decoded rollout：object-balanced IoU mean/median 和 win rate 为正，precision 与连通性不发生预注册阈值以上的退化。

不能放宽 pose/correspondence 因果归因门。若 correct 不稳定优于 actual corruption、constant-pair 和 spatial-permuted-pair，即使最终优于 stock，也只能解释为 learned local regularizer 或 generic correction，不能声称模型使用了正确 pose 对齐。

与原 TRELLIS、Pixal3D 的跨模型比较不能使用 SS Flow MSE。需要统一 test objects、view/image/pose 输入预算、sampling compute 和 canonical normalization，再比较最终 occupancy/mesh、Chamfer/F-score、multi-view reprojection 和 object-balanced win rate。

### 下一步

严格按命令文档第 12 节执行：

```text
s5 finite smoke
-> overfit16 三独立训练 seed s50
-> fresh48 × 三eval seed × 五active t
-> 协议一致性汇总
-> 仅在三seed benchmark_relaxed 全通过时跑30-step decoded rollout
```

当前没有依据直接增加到 s200 或启用 Flow LoRA。

## 2026-07-15：P0/P1 Independent Cache Audit 阈值复核

对 `/data/ar_ss_flow_pose_lifting_overfit64_v1_20260714` 的 64 个样本运行独立审计，首次命令使用：

```text
max_cached_geometry_diff = 1e-5
max_roundtrip_error = 1e-4
```

输出为 `passed=false`，但这不表示深度、pose 或 projection 全部失败。汇总项为：

```text
samples_loaded = true
depth_calibration_enabled_ratio = 1.0
depth_fallback_count = 0
no_nonfinite_or_missing_inputs = true
sample_audits_passed = false
```

64/64 的失败均来自 cached-vs-fresh feature-volume 最大绝对差超过 `1e-5`。分布为：

```text
volume max-abs diff：
  min    4.23e-5
  median 2.14e-4
  P90    9.22e-4
  P95    1.20e-3
  max    3.24e-3

metadata max-abs diff：
  median 8.52e-6
  P95    2.18e-5
  max    6.79e-5

projection round-trip max error：
  median 5.64e-7
  max    7.63e-7
```

`visual_patch_features` 缓存为 FP16，局部通道最大幅值可超过 100。审计中的 `cached_geometry_volume_max_abs_diff` 实际比较 sampling 后的高维 feature volume，而不是直接比较 K/T 或 projection-grid 坐标；微小 grid 数值差异在高梯度 FP16 feature 上产生 `1e-4–1e-3` 最大差是合理的。

阈值覆盖统计：

```text
1e-5：  0/64 volume PASS
1e-4： 13/64
5e-4： 52/64
1e-3： 58/64
2e-3： 62/64
4e-3： 64/64
```

两个超过脚本旧默认 `2e-3` 的样本分别为 `3.24e-3` 和 `2.51e-3`，但其 metadata 与 round-trip 均正常。因此当前裁决是：

> 本次 FAIL 是不适合 FP16 feature-volume 的过严绝对阈值造成的数值审计假失败，没有证据表明 cache 的深度标定、相机投影或输入完整性损坏。

命令文档已把该数值等价门修正为 `4e-3`，同时保留独立的 projection round-trip `1e-4` 硬门并加入 `--fail_on_error`。外层 `|| true` 仍只用于保护终端，真实状态必须读取 exit-code 文件和 `report.json.passed`。

### 2026-07-20：D3 v2 同设备几何审计取代放宽阈值

后续 train256 复核发现，旧 v1 审计在 CPU 生成 cached geometry，却在 CUDA
重建fresh geometry。约 `2.38e-7` 的 projection-grid 设备数值差异可以跨过
lifting 中的 `weight.sum() > 1e-6` 硬门，使单个 voxel 从零跳到高幅特征；
最严重样本因此产生 `85.1033` 的 max-abs 假离群值。所以单纯放宽到
`4e-3` 不是可靠修复。

D3 v2 改为：

- 在 CPU 上重建fresh geometry，直接审计三个 FP32 geometry tensor 和
  bit-exact `valid`；
- cached/fresh 两边都使用 CPU 生成的 geometry，再在 CUDA 上重放
  volume/metadata；
- projection round-trip 固定在 CPU 执行，与 runtime device 解耦；
- geometry 缺失、schema/dtype/shape 错误、non-finite 或 `valid` bit 变化都会
  明确失败。

2026-07-20 完整重跑结果：

```text
train256: passed=true, failures=0, direct/volume/metadata max diff=0,
          valid mismatch=0, roundtrip max=7.20e-7
val64:   passed=true, failures=0, direct/volume/metadata max diff=0,
          valid mismatch=0, roundtrip max=8.34e-7
```

因此 `4e-3` 仅作旧 v1 历史分析；当前准入以
`ar_ss_flow.pose_lifting_cache_audit.v2` 报告为准。

### P2再次运行的状态确认

修正P0/P1数值阈值后再次运行旧P2，结果与本文件前述7月14日full-64审计完全一致：

```text
paired_same_visual_features = true
null_is_exact_zero = true
all_corruptions_separable = true
correct_geometry_has_directional_advantage = false
```

因此这不是P0/P1 cache阈值调整引入的新问题。旧P2的唯一失败来源仍是raw cross-view方向：`pose_shuffle=57.8125% < 60%`，`depth_corrupt=17.1875%`。后者会通过减少有效view和support weight获得更低variance，当前指标没有固定active-view集合和总support mass，不能继续作为C3硬前置。

旧命令第4至第9节停止执行。后续直接使用第12节C3的actual-corruption、same-voxel pair controls、teacher-forced和decoded rollout双层判定；该放行仅适用于已知canonical frame的合成机制实验，不代表真实AR object-frame问题已解决。

## 2026-07-15：C3 第12.2–12.7节完整结果

### 最终裁决

```text
工程与数值链路：PASS
三训练seed协议一致性：PASS
teacher-forced benchmark_relaxed：3/3 PASS
teacher-forced mechanism_strict：3/3 FAIL
30-step decoded rollout benchmark：FAIL
```

因此：

> C3 在 fresh48 上学到了可重复但极小的 stock-relative Flow-MSE 修正，并对空间置乱具有一定敏感性；但它没有稳定利用正确 view-pair/object correspondence，最终采样又表现为轻微 precision 上升、recall/IoU 下降。按预注册停止条件，不进入 s200、Flow LoRA 或正式长训。

### 12.2–12.4 工程与训练状态

所有 14 个 train/audit/eval/summary/rollout exit-code 文件均为 `0`。三个 s50 run：

```text
completed steps = 50/50
train_report.finite = true
finite_run_audit.passed = true
trainable state / optimizer / scaler finite = true
stock Flow trainable parameters = 0
physical-off / null / inactive-t = bit-exact stock
```

三个 seed 的末步各参数组梯度均为 finite nonzero，包括 state、visual、metadata、pair attention、fusion 和 output。数值和梯度链路没有阻塞。

末步 residual 幅度存在 seed 差异：

| train seed | correct delta RMS | wrong delta RMS | relative correct gain | correct-vs-wrong |
|---:|---:|---:|---:|---:|
| 42 | 2.068e-3 | 1.962e-3 | +4.444e-4 | +1.285e-4 |
| 43 | 2.572e-4 | 2.466e-4 | +2.406e-5 | +1.359e-5 |
| 44 | 1.512e-3 | 1.448e-3 | +3.504e-4 | +5.963e-5 |

这说明优化器可以学习 residual，但不同 seed 收敛到的幅度并不一致。

### 12.5 fresh48 teacher-forced relaxed 结果

协议：

```text
48 fresh objects
3 fixed noise seeds
active t = 0.55,0.65,0.75,0.85,0.95
inactive t = 0.4
actual cyclic1/cyclic2/reverse/cross-sample evidence
constant-pair / spatial-permuted-pair controls
```

三 seed 均满足 exact stock fallback、mean/median gain 为正、object win >=55%、5/5 active t 为正：

| train seed | correct gain mean | median | correct>stock object win | positive t |
|---:|---:|---:|---:|---:|
| 42 | +1.743e-4 | +1.744e-4 | 66.67% | 5/5 |
| 43 | +1.323e-4 | +1.176e-4 | 72.92% | 5/5 |
| 44 | +2.081e-4 | +1.261e-4 | 77.08% | 5/5 |

三 seed 平均 relative Flow-MSE gain 为 `1.715e-4`，即约 `0.0172%`。方向稳定，但绝对量级很小，只足以触发 rollout，不足以单独构成生成收益。

### 12.5 mechanism_strict 失败原因

三 seed 全部 strict FAIL。最稳定的失败是 constant-pair：

| train seed | correct gain | constant-pair gain | correct>constant object win |
|---:|---:|---:|---:|
| 42 | 1.743e-4 | 1.747e-4 | 39.58% |
| 43 | 1.323e-4 | 1.329e-4 | 33.33% |
| 44 | 2.081e-4 | 2.103e-4 | 35.42% |

constant-pair 略优于完整 correct pair features，说明 view-pair 内容和 pair identity 并不是收益来源。

空间置乱 control 明显较弱：

| train seed | spatial-permuted gain | correct>permuted object win |
|---:|---:|---:|
| 42 | 7.493e-5 | 62.50% |
| 43 | 5.031e-5 | 70.83% |
| 44 | 7.368e-5 | 72.92% |

因此模型确实依赖 16^3 support/feature 的空间位置，但没有证明依赖正确的跨视图 pair 内容。更准确的解释是：

```text
spatially aligned physical evidence present
  -> generic local Flow correction
```

而不是：

```text
correct image-pose-view correspondence
  -> object-specific correction
```

actual pose corruptions 上 correct 的平均优势多数为正，但对象胜率不稳定：seed42/43 的 cyclic/reverse 约 `54%–60%`，只有 seed44 的 cyclic2/reverse 达到 `66.67%`。cross-sample 最弱，seed42 中错误对象甚至比 correct gain 更高；这进一步否定了稳定 object-specific correspondence。

### 12.6 多 seed 汇总

```text
protocol_passed = true
benchmark_relaxed_passed_all_seeds = true
mechanism_strict_passed_all_seeds = false
correct gain mean across seeds = +1.715e-4
correct gain median across seeds = +1.394e-4
correct>stock object win mean = 72.22%
positive t mean = 5/5
```

汇总文件中的顶层 `passed=true` 只表示 `decision_profile=report_only` 下协议一致；不能解释成科学结论通过。

### 12.7 30-step decoded rollout

12.7 按 relaxed 规则自动选择 teacher-forced gain 最大的 seed44。协议为：

```text
8 fresh objects
3 rollout noise seeds
30 Euler steps
stock / correct / two controls / four actual corruptions
```

stock 与 correct 的平均最终指标：

| branch | IoU | precision | recall | coord-count ratio | components | largest-component ratio |
|---|---:|---:|---:|---:|---:|---:|
| stock | 0.045808 | 0.079417 | 0.103324 | 1.388871 | 2.208 | 0.940263 |
| correct | 0.045749 | 0.079646 | 0.102832 | 1.378738 | 2.208 | 0.940049 |

object-balanced correct-minus-stock：

| metric | mean | median | object win |
|---|---:|---:|---:|
| IoU | -5.914e-5 | -3.317e-5 | 50.0% |
| precision | +2.285e-4 | +3.626e-4 | 62.5% |
| recall | -4.921e-4 | -4.358e-4 | 25.0% |
| largest-component ratio | -2.146e-4 | 0 | 37.5% |

模型减少了预测体素数量，带来很小的 precision 上升，但丢失更多 GT occupancy，导致 recall 和 IoU 下降。三个 rollout noise seed 中只有 seed43 的 IoU mean 为正；seed42 和 seed44 均为负，方向不稳定。

正式 rollout gate：

```text
correct_iou_mean_positive = false
correct_iou_median_positive = false
correct_iou_object_win = false
precision_not_degraded = true
largest_component_not_degraded = true
rollout_benchmark.passed = false
teacher_forced_and_rollout_passed = false
```

correct 在 rollout 中通常优于 spatial-permuted 和部分 pose-corruption branches，说明空间对齐信号并非完全无效；但 stock 仍然更好。生成任务的目标是超过 stock，不是只比错误 corruption 少退化，因此该结果不能作为正收益。

### 失败机制解释

当前损失可以通过学习一个非常小、空间受 support 控制的通用 velocity calibration 来降低 teacher-forced MSE。Flow-MSE 的 `1e-4` 相对改善不足以稳定跨越 decoder 的离散 occupancy 阈值；当它产生可见变化时，主要表现为删减体素，从而提高 precision、降低 recall。

constant-pair 不弱于 correct 说明 pair attention 没有学到额外 correspondence 内容；cross-sample 也没有稳定退化，说明当前稀疏点和跨视图特征不足以建立对象身份约束。继续增加步数或开启 Flow LoRA 更可能放大 generic pruning，而不是自动产生正确 pose specificity。

### 下一步建议

按第12节预注册停止条件：

```text
benchmark_relaxed PASS
但 mechanism_strict FAIL
且 decoded rollout FAIL
  -> 停止 C3 residual 长训
```

明确不建议：

```text
不跑 C3 s200/s1000
不启用 Flow LoRA
不解冻 stock SS Flow / get_ss_cond
不在这8个 rollout objects上做scale选择后再把它们当test
```

保留成果：

1. C1 pairwise correspondence detector 可作为 pose/data-quality screening 工具。
2. C3 证明 same-voxel spatial layout 比 spatial permutation 更有效，可作为后续显式几何模块的结构依据。
3. relaxed 与 strict/rollout 分层协议成功阻止了把微小 teacher-forced gain误判为生成收益。

下一条学习主线应改变监督和模块职责，而不是继续调 residual：

```text
合成阶段使用 exact render depth / surface ID / visibility
  -> 先预训练可验证的 pose-aligned 3D feature representation
  -> 要求 same-surface correct pair 明显优于 shuffled/cross-object
  -> 再在 SS latent 16^3 内使用局部 ControlNet/adapter
  -> 训练直接 occupancy/coordinate 或 decoder-logit supervision
```

如果近期目标是先形成可发表的稳定系统结果，优先转向：

```text
frozen ReconViaGen/TRELLIS generation
+ 可解释的 pose/point visual-hull filtering 或 post-generation optimization
+ 统一 object-disjoint 最终 mesh/occupancy benchmark
```

真实 AR 路线仍必须单独解决 object-to-world/canonical Sim(3)；本次 canonical synthetic 结果不能外推为真实 AR pose 已可直接注入。
