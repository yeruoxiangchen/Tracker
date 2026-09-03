# 2026-06-29 单图像 condition + 8-view prior strict scale-up 路线结论

## 结论

当前不建议直接进入正式多视角训练。更合理的主线是先用现有 PixalV9 / Objaverse synthetic manifest 做 single-image condition + 8-view visible prior 的 strict scale-up，确认 `G_inp` 是否能在更大训练集上学会 high-t / pure-noise rollout。

原因：

```text
1. 当前已确认 q_vis/m_s/q_splice 链路不是根本问题；
2. q_splice_t 在 low/mid-t 可以接近 q_splice；
3. START_T=1 pure-noise rollout 仍然失败；
4. 如果直接多视角训练，结果会混合“视角数量提升”和“high-t 是否学会”两个变量。
```

因此下一步优先回答一个更基础的问题：

```text
1488 train / 128 val 的 single-image condition + 8-view prior strict full fine-tune，
能否让 q_pred 在 val128 上超过 q_vis？
```

## 当前数据是否够用

当前可用合成数据：

```text
/data/pixal3d_multiview/objaverse_sparse_mv_artraj_pbr_5000_v9_select8

train: 1488 samples
val:   128 samples

每个样本包含：
  8 selected views
  24 candidate views
  RGB / mask
  intrinsic / extrinsic
  source_glb
  complete ss_latent
  target sparse coords
```

这批数据够做 pilot 和 scale-up 验证，但还不是最终论文级训练集。当前 `q_vis` 来自 `build_point_prior_dataset.py` 的 sparse coord support prior，不是严格论文式的 rendered depth visible surface prior。

同时要注意：当前 point-prior manifest 默认来自：

```text
MAX_FRAMES=8
NUM_PRIOR_VIEWS_CHOICES=8
```

因此当前实验不是严格 single-view prior。它是：

```text
single-image condition
+
8-view oracle / AR-like visible point prior
```

如果目标是严格 single-view Points-to-3D baseline，需要另建 `MAX_FRAMES=1, NUM_PRIOR_VIEWS_CHOICES=1` 的 point-prior / latent manifest，并让 condition image 与 prior view 对齐。

所以当前定位是：

```text
够用：
  train512 / val128
  train1488 / val128
  single-image condition + 8-view prior strict scale-up
  high-t curriculum 验证

不够：
  直接证明论文级 full fine-tune
  直接代表真实 AR/SLAM domain
```

## 推荐实验顺序

第一步：构建 latent_inpaint_train512 / latent_inpaint_val128。

第二步：single-image condition + 8-view prior low/mid-t curriculum：

```text
TRAIN_T_MIN=0.0
TRAIN_T_MAX=0.75
IMAGE_MAX_VIEWS=1
IMAGE_FRAME_SELECT=random
BOUNDARY_REFINE_STEPS=0
CFG_STRENGTH=1.0
KNOWN_FLOW_LOSS_WEIGHT=0.02
```

目的：先让模型在已证明可行的 low/mid-t 区间稳定学习。phase1 没有训练 `t>0.75`，所以 `START_T=1` 只是 OOD hard probe，不能作为 phase1 生死标准。

phase1 跑完后必须评估：

```text
q_splice_t START_T=0.75 / 0.50 / 0.25
teacher-forced t=0.75 / 0.50 / 0.25
```

`START_T=0.90` 可以保留为 OOD probe，但解释时不能和 phase1 训练区间内结果等价。

第三步：用 phase1 checkpoint 继续 full-t 训练：

```text
TRAIN_T_MIN=0.0
TRAIN_T_MAX=1.0
```

目的：验证数据规模和 curriculum 是否能改善 START_T=1。

第四步：如果 train512 有改善，再扩展到 train1488 / val128。

第五步：只有当 sparse 指标超过 q_vis 后，再跑 mesh eval。

## 暂不建议

```text
1. 不直接开多视角 full training；
2. 不急着开 boundary refinement；
3. 不急着大规模 mesh eval；
4. 不继续只调单视图小数据；
5. 不把 stock warm-start 作为当前主线。
```

## 判断标准

主要看 val128 sparse：

```text
q_pred/topk_target_unique IoU 是否超过 q_vis；
q_pred unknown L1 是否低于 q_vis；
threshold_0 是否不过度 overfill；
component/largest 是否接近 q_splice；
phase1: q_splice_t / teacher-forced t=0.75,0.50,0.25 是否改善；
phase2: START_T=1 是否比 train64 明显改善。
```

如果 train512 / train1488 仍不能超过 q_vis，则下一步应优先检查：

```text
1. visible prior 构造是否需要改成严格论文式 rendered-depth visibility；
2. full fine-tune 是否需要分层解冻，而不是全量或 input-only 两极；
3. high-t loss 是否需要重加权；
4. image condition / preprocessing 是否与 TRELLIS 原始路径完全一致。
```
