# 原版单视图 Pixal3D vs 当前多视图 Full：Smoke 结果分析

日期：2026-07-27  
状态：`EXPLORATORY / FORMAL=false`  
样本数：6 个 Objaverse case（current 2/4/8 views 各 2 个）

## 1. 结论摘要

本次 6-object shape-only smoke 对当前系统给出了明确的不利信号：原版预训练
Pixal3D（单视图）在总体 Chamfer-L1、F-score@0.02 和 normal consistency 三项均值上
均优于当前 multi-view full。

| 指标 | current multi-view full | Pixal3D stock | current - Pixal3D | 方向 |
|---|---:|---:|---:|---|
| Chamfer-L1（越低越好） | 0.08868309 | 0.06362408 | +0.02505901 | current 较差；比 Pixal3D 高 39.4% |
| F-score@0.02（越高越好） | 0.28221063 | 0.33895142 | -0.05674079 | current 较差；比 Pixal3D 低 16.7% |
| Normal consistency（越高越好） | 0.54205371 | 0.57292936 | -0.03087565 | current 较差；比 Pixal3D 低 5.4% |

逐对象符号也支持这个方向，而不只是均值被一个离群对象拉动：

- current 的 Chamfer 胜率为 `1/6 = 0.167`；
- current 的 F-score 胜率为 `2/6 = 0.333`；
- current 的 normal consistency 胜率为 `2/6 = 0.333`；
- 8-view 组同时出现一个 current 明显成功对象和一个明显失败对象，显示 current 的
  对象间稳定性仍然很差。

因此，在这批已查看的 6 个对象上，不能声称当前 multi-view full 已优于原版单视图
Pixal3D。更准确的裁决是：

> 当前 full 在个别对象上可以显著超过 Pixal3D，但总体成功率和稳定性不足；原版
> Pixal3D stock weights 在本次 similarity-aligned shape smoke 中占优。这是需要继续
> 定位的工程红灯，不是正式模型优劣结论。

`report.json` 中的 `passed=true` 只表示评估流程完整执行并成功产出报告，不表示
current 通过了效益门，也不表示 current 优于 Pixal3D。

## 2. 比较对象与口径

### 2.1 Current multi-view full

- 冻结复用已有 current Mesh；
- Direct SS checkpoint：step 900；
- Direct SLAT checkpoint：step 800；
- 分支：`full`（训练后的 adapter + LoRA 路径）；
- current 输入预算：2、4、8 views；
- joint seed：42；
- Mesh 文件名：`mesh_pairs/<pair_id>/full/mesh_canonical.obj`。

### 2.2 Pixal3D stock

本文中的“Pixal3D stock”专指 `TencentARC/Pixal3D` 原版预训练权重，没有使用本项目
fine-tune。它不等于 S6 中的 native Direct-SLAT `stock` 分支，两者是完全不同的模型
和对照概念。

本次 Pixal3D 运行还固定了以下有利且可复现的推理条件：

- 每个对象只输入 current 实际输入帧中前景 mask 面积最大的一帧；
- 使用冻结 GT alpha，跳过 RMBG；
- 使用已知 crop FOV，不引入 MoGe 相机估计误差；
- resolution 1024、sampling steps 12、seed 42；
- 官方预训练 NAF，固定本地源码和 checkpoint；
- dense/sparse attention 使用 PyTorch SDPA；
- 保存 geometry-only OBJ，不执行纹理 remesh/4K GLB 导出。

所以这里的“stock”是 stock model weights，不是未经任何运行适配的默认 CLI。GT alpha、
已知 FOV 和最佳可见帧都让该单视图 baseline 获得了相对有利的条件。

### 2.3 主评估坐标系

两种方法都使用相同评估流程：

1. 24 个 proper cube rotations 初始化；
2. isotropic similarity ICP；
3. 禁止 reflection；
4. alignment samples 4000；
5. surface samples 20000；
6. 对同一 canonical GT 计算 surface metrics。

该口径去除了全局旋转、平移和各向同性尺度，只回答 shape quality。它不评价世界坐标
姿态、绝对尺度或 AR 对齐能力。即使 current 在原始 canonical frame 中有优势，本报告
也不会保留这类优势；反过来，current 的坐标错误也可能被 similarity ICP 掩盖。

## 3. 分视角结果

| Current views | Cases | Chamfer current | Chamfer Pixal3D | Current Chamfer wins | F-score current | F-score Pixal3D | Current F-score wins |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 0.12264801 | 0.09082195 | 0/2 | 0.11351243 | 0.22209349 | 0/2 |
| 4 | 2 | 0.07732618 | 0.05696394 | 0/2 | 0.27312784 | 0.32302389 | 1/2 |
| 8 | 2 | 0.06607507 | 0.04308636 | 1/2 | 0.45999164 | 0.47173688 | 1/2 |

表面上 current 随 views 增加而均值改善，F-score 差距也从 2-view 的 `-0.10858` 缩小
到 8-view 的 `-0.01175`。但三个档位使用的是不同对象，每档只有两个 case，因此不能把
这个趋势解释为“增加 views 导致改善”。要回答 view scaling，必须对同一批对象分别跑
2/4/8-view counterfactual。

8-view 平均值尤其不稳定：一个对象 current 显著胜出，另一个对象 current 显著失败，
两者平均后会掩盖真实的双峰行为。

## 4. 逐对象结果

下表中的 Chamfer delta 定义为 `Pixal3D - current`，正值表示 current 更好；F-score
delta 定义为 `current - Pixal3D`，正值同样表示 current 更好。

| Case | Views | Chamfer current | Chamfer Pixal3D | Chamfer delta | F-score current | F-score Pixal3D | F-score delta | Normal current | Normal Pixal3D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v02_00_0d18146b4fdd_seq000` | 2 | 0.143010 | 0.110561 | -0.032448 | 0.079004 | 0.185005 | -0.106002 | 0.424897 | 0.530258 |
| `v02_01_0cd491abba61_seq000` | 2 | 0.102286 | 0.071083 | -0.031204 | 0.148021 | 0.259182 | -0.111161 | 0.516777 | 0.646175 |
| `v04_02_29ddf51d78ae_seq001` | 4 | 0.081896 | 0.054008 | -0.027888 | 0.125505 | 0.319048 | -0.193543 | 0.645320 | 0.610864 |
| `v04_03_092bb8034641_seq000` | 4 | 0.072756 | 0.059920 | -0.012836 | 0.420751 | 0.326999 | +0.093751 | 0.430330 | 0.528343 |
| `v08_04_04d712a0dae0_seq001` | 8 | 0.010673 | 0.030084 | +0.019411 | 0.878950 | 0.580885 | +0.298064 | 0.812921 | 0.609793 |
| `v08_05_262afaaf596a_seq000` | 8 | 0.121477 | 0.056089 | -0.065388 | 0.041034 | 0.362588 | -0.321555 | 0.422078 | 0.512144 |

逐对象诊断：

- `v08_04...` 是 current 的明确成功例：三项指标全部显著优于 Pixal3D；
- `v08_05...` 是 current 的明确失败例，也是总体最强负向对象；
- `v04_03...` 的 F-score 优于 Pixal3D，但 Chamfer 和 normal consistency 仍较差；
- `v04_02...` 的 normal consistency 略优，但覆盖相关指标显著较差；
- 两个 2-view case 中 current 三项指标均失败，低视角输入下风险最明显。

## 5. 主要误差形态：覆盖不足

双向距离均值揭示了比单一 Chamfer 更具体的问题：

| 双向距离 | current | Pixal3D | 解释 |
|---|---:|---:|---|
| prediction-to-GT mean | 0.01694440 | 0.01833733 | current 生成出来的局部表面略贴近 GT |
| GT-to-prediction mean | 0.16042178 | 0.10891083 | current 对 GT 的覆盖明显更差 |

current 的 prediction-to-GT 略好，但 GT-to-prediction 比 Pixal3D 高约 47%。这说明
current 的主要问题更像“已有表面局部较准，但缺失主体区域或部件”，而不是所有表面都
整体偏离。`v02_00...` 是最极端的例子：current 在 0.02 阈值下 precision 为
`0.9584`，recall 只有 `0.0412`；Pixal3D 的 recall 虽然也低，但达到 `0.10425`。

这个诊断与后续人工检查重点一致：优先查看主体缺失、背面/遮挡面恢复、细长部件缺失和
多组件覆盖，而不应只优化已有表面的局部平滑度。

### 5.1 Mesh 复杂度并不匹配

Pixal3D 的 geometry-only raw Mesh 显著更密，当前比较不是等顶点数或等面数比较：

| Case | Current faces | Pixal3D faces | Pixal3D/current face ratio |
|---|---:|---:|---:|
| `v02_00...` | 425,872 | 12,961,774 | 30.4x |
| `v02_01...` | 573,924 | 5,016,408 | 8.7x |
| `v04_02...` | 115,232 | 2,675,642 | 23.2x |
| `v04_03...` | 662,404 | 7,292,826 | 11.0x |
| `v08_04...` | 600,918 | 11,138,268 | 18.5x |
| `v08_05...` | 332,452 | 7,071,652 | 21.3x |

Current OBJ 为约 6.2--37.7 MB，Pixal3D raw OBJ 为约 109.8--529.0 MB。固定 20,000
surface samples 让两个 Mesh 使用相同测量预算，但没有惩罚 Pixal3D 的表示复杂度、存储、
加载和后处理成本。更密的三角化本身不保证更低 Chamfer，但允许保留更多局部几何；正式
系统比较应同时报告 surface quality 与 complexity/latency，并增加统一简化预算后的
敏感性分析。

## 6. Mesh 文件位置

### 6.1 根目录

Current S6 根目录：

```text
/home/zjr/Tracker/pose_point_depth_mv/outputs/direct_slat_step900support_trainall_s1000_seed42_2gpu_bf16_v3/mesh_slat_support_val32_step000800_seed424344_exploratory_v1
```

本次 Pixal3D smoke 根目录：

```text
/home/zjr/Tracker/pose_point_depth_mv/outputs/direct_slat_step900support_trainall_s1000_seed42_2gpu_bf16_v3/mesh_slat_support_val32_step000800_seed424344_exploratory_v1/pixal3d_singleview_smoke6_seed42_skiprembg_localnaf_v3
```

下表路径分别相对于上述 Current S6 根目录和 Pixal3D smoke 根目录。

| Case | Current full Mesh（相对 Current S6 根） | Pixal3D stock raw Mesh（相对 smoke 根） | S3 对齐后 Pixal3D Mesh（相对 smoke 根） | GT（相对 smoke 根） |
|---|---|---|---|---|
| `v02_00_0d18146b4fdd_seq000` | `mesh_pairs/obj_0008_seed_42/full/mesh_canonical.obj` | `pixal3d/v02_00_0d18146b4fdd_seq000/mesh.obj` | `aligned_pixal3d/v02_00_0d18146b4fdd_seq000/mesh.obj` | `targets/v02_00_0d18146b4fdd_seq000.obj` |
| `v02_01_0cd491abba61_seq000` | `mesh_pairs/obj_0007_seed_42/full/mesh_canonical.obj` | `pixal3d/v02_01_0cd491abba61_seq000/mesh.obj` | `aligned_pixal3d/v02_01_0cd491abba61_seq000/mesh.obj` | `targets/v02_01_0cd491abba61_seq000.obj` |
| `v04_02_29ddf51d78ae_seq001` | `mesh_pairs/obj_0024_seed_42/full/mesh_canonical.obj` | `pixal3d/v04_02_29ddf51d78ae_seq001/mesh.obj` | `aligned_pixal3d/v04_02_29ddf51d78ae_seq001/mesh.obj` | `targets/v04_02_29ddf51d78ae_seq001.obj` |
| `v04_03_092bb8034641_seq000` | `mesh_pairs/obj_0004_seed_42/full/mesh_canonical.obj` | `pixal3d/v04_03_092bb8034641_seq000/mesh.obj` | `aligned_pixal3d/v04_03_092bb8034641_seq000/mesh.obj` | `targets/v04_03_092bb8034641_seq000.obj` |
| `v08_04_04d712a0dae0_seq001` | `mesh_pairs/obj_0002_seed_42/full/mesh_canonical.obj` | `pixal3d/v08_04_04d712a0dae0_seq001/mesh.obj` | `aligned_pixal3d/v08_04_04d712a0dae0_seq001/mesh.obj` | `targets/v08_04_04d712a0dae0_seq001.obj` |
| `v08_05_262afaaf596a_seq000` | `mesh_pairs/obj_0022_seed_42/full/mesh_canonical.obj` | `pixal3d/v08_05_262afaaf596a_seq000/mesh.obj` | `aligned_pixal3d/v08_05_262afaaf596a_seq000/mesh.obj` | `targets/v08_05_262afaaf596a_seq000.obj` |

路径语义：

- `current .../full/mesh_canonical.obj`：S6 保存的 current full 原始 canonical Mesh；
- `pixal3d/<case>/mesh.obj`：原版 Pixal3D stock weights 的 geometry-only 输出，已应用
  官方 `inference.py` 输出 rotation，但尚未执行 S3 similarity ICP；
- `aligned_pixal3d/<case>/mesh.obj`：S3 保存的 Pixal3D metric-frame Mesh，适合与 GT
  在对齐坐标中查看；
- S3 对 current full 也执行了 similarity ICP，但当前实现只在内存中使用
  `current_aligned` 计算指标，没有单独导出 aligned-current Mesh；current 的完整
  4x4 alignment matrix 保存在 `report.json -> records[*].current.alignment.matrix`。

若要做严格的 metric-frame 可视化，不能直接把 raw current full 与
`aligned_pixal3d` 叠加；应先按 `report.json` 中的 current alignment matrix 变换 current
Mesh，或扩展 evaluator 对称导出 `aligned_current`。

## 7. 证据与哈希

主产物：

```text
report.json SHA-256:
b4a292fbb8a997d25f043c2c606273d874e7030ca8ce089aeaef26a49b676383

metrics.csv SHA-256:
3450e9911c15446b76fca8eb191b7cc177d93740902b6e44ab8180f764021674

summary.txt SHA-256:
0456a6e442aa00ea8063007e76fdf3e47d0c5a40f095b58431fb097df196a854

protocol canonical SHA-256:
64cc2622f504d024dae1cca08970e1a99bc4117b5121846173d0cde9899cd672
```

每个 current full、Pixal3D raw、Pixal3D aligned 和 GT Mesh 的完整 SHA-256 已绑定在：

```text
report.json -> records[*].current.mesh.sha256
report.json -> records[*].pixal3d.mesh.sha256
report.json -> records[*].pixal3d.aligned_mesh.sha256
report.json -> records[*].target_mesh.sha256
```

## 8. 这份结果不能回答什么

1. `n=6`，没有冻结的置信区间或正式非劣/优效阈值，不能外推总体性能；
2. 这是已看 validation 输出上的 exploratory smoke，不是 unseen blind holdout；
3. current 2/4/8-view 三组不是同一对象，不能据此估计 views 的因果作用；
4. Pixal3D 训练数据与这 6 个 Objaverse 对象是否重叠尚未审计，正式比较前必须检查
   pretrained-data leakage/dedup；
5. similarity ICP 移除了 pose 和 scale，本报告不能证明 AR/world-coordinate 质量；
6. 本报告没有比较 watertight、boundary、non-manifold、component count、Mesh 面数、
   推理时延或显存；
7. current full 来自此前已记录存在 runtime repeatability 问题的 sparse 推理栈，因而
   不能把小幅差异升级为精确模型效应；
8. current full vs Pixal3D stock 是跨架构系统比较，不能否定 S6 中 full 相对
   native Direct-SLAT stock 的局部增益。两项实验回答的是不同问题。

## 9. 下一步建议

1. 先对全部 6 个 case 做成对人工检查，重点看 `v08_04...` 成功例和 `v08_05...`
   失败例，记录主体结构、缺失部件、漂浮片、薄片/毛刺、孔洞和开放边界；
2. 在相同对象上构造 current 2/4/8-view matched counterfactual，单独回答多视角收益；
3. 同时报告 raw canonical/world-frame 指标和 similarity-aligned shape 指标，避免把
   pose/scale 失败隐藏在 ICP 后；
4. 增加对称 topology、component、复杂度、运行时和显存指标；Pixal3D Mesh 面数很高，
   单纯 surface sample 指标没有计入模型复杂度成本；
5. 扩大 exploratory 样本前先冻结对象列表和选帧规则，不根据本次成败对象做替换；
6. 只有在 current sparse runtime repeatability 修复、Pixal3D 预训练重叠完成审计后，
   才设计新的 unseen blind holdout。正式协议应预注册主指标、非劣界限、topology 安全门
   和人工盲评规则。

当前最合理的研发优先级不是继续用这 6 个对象调 checkpoint，而是解释 current 的
coverage failure 和对象间双峰行为，并建立同对象 view-scaling 与 topology 诊断。
