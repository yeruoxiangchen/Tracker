# Direct-SLAT v3 S7 step400 停止结果分析与后续方案

日期：2026-07-28  
状态：S7 已停止；R1 Mixed-10k source freeze 正在运行  
结论级别：训练过程诊断，尚不是 final-Mesh 科学结论

## 1. S7 停止现场

S7 是897对象 cache 上的 `rollout_aligned_v3` 续训：

```text
output:
  /data/zjr/direct_slat_v3_rollout_aligned_20260728/
  train897_step1000_from_step100_seed42_2gpu_v1

resume:
  train897_step100_seed42_2gpu_v1/checkpoints/step_000100.pt

training:
  post_cfg_v2
  cfg_strength=5.0
  cfg_interval=[0.5, 1.0]
  slat_delta_rms_ratio_cap=0.10
  raw_delta_excess_weight=0.01
  one-step rollout probability=0.25
  rollout_step_size=0.05
```

进程已经完全停止。日志最后到 step425，但没有 step425 checkpoint，所以 step405–425
不能作为可复现候选。已冻结的候选为：

| checkpoint | step | micro-step | 状态 |
|---|---:|---:|---|
| 原 step100 | 100 | 800 | 已有 v3 评估 |
| `step_000200.pt` | 200 | 1600 | 完整 |
| `step_000300.pt` | 300 | 2400 | 完整 |
| `step_000400.pt` | 400 | 3200 | 完整，当前主候选 |

`last.pt` 和 `step_000400.pt` 都声明 step400、micro-step3200、相同 epoch/data cursor 和
400条 history，但序列化 SHA-256 不同。后续实验应显式绑定
`step_000400.pt`，不要用可变语义的 `last.pt`。

## 2. 完整400步 history 的分段统计

以下统计直接来自 `step_000400.pt["history"]`，不是每5步日志抽样。

### 2.1 主指标

| step窗口 | post-CFG gain mean | post-CFG gain median | post-CFG win | post-CFG clip | non-post gain mean | rollout gain mean | rollout gain median | rollout win |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1–100 | +0.186365 | +0.072441 | 98.1% | 76.9% | -0.003103 | +0.041786 | +0.028388 | 73.7% |
| 101–200 | +0.223305 | +0.153369 | 100% | 100% | -0.007074 | +0.049864 | -0.005527 | 45.8% |
| 201–300 | +0.244997 | +0.147388 | 100% | 100% | -0.002916 | +0.055310 | -0.003227 | 42.4% |
| 301–400 | +0.246109 | +0.137536 | 100% | 100% | -0.002746 | +0.093861 | +0.042079 | 65.4% |

精确解释：

- post-CFG teacher gain 从 step100 到 step200 有增加；
- step200 后 mean 基本停在 `+0.245` 左右，median 反而从 `+0.153` 降到 `+0.138`；
- step301–400 的 one-step rollout 比此前更好，是当前最值得验证的正向信号；
- 但只有 one-step local rollout，仍不能推出25-step Mesh endpoint 改善；
- 非 post-CFG 分支持续平均弱于 Stock，说明 late/low-`t` 修正没有学成安全增益。

### 2.2 残差饱和

| step窗口 | post-CFG raw delta/stock mean | post-CFG raw ratio max | raw-excess loss mean | support token RMS mean |
|---|---:|---:|---:|---:|
| 1–100 | 0.2552 | 0.8983 | 0.02924 | 0.09113 |
| 101–200 | 0.6399 | 1.4459 | 0.17982 | 0.25883 |
| 201–300 | 0.7022 | 2.2162 | 0.25008 | 0.36426 |
| 301–400 | 0.7858 | 2.7773 | 0.39512 | 0.43222 |

训练部署的有效 delta 被固定限制为 Stock RMS 的10%。从 step101 开始，全部 post-CFG
event 都撞上 cap；同时 raw ratio、raw-excess loss 和 support token RMS 持续增长。

当前 hard cap 实现使用：

```text
clip_scale = min(1, cap * stock_rms / raw_delta_rms)
effective_delta = raw_delta * stop_gradient(clip_scale)
```

这样做保证 forward 严格有界，并保留饱和时的梯度；但当
`raw_delta_excess_weight=0.01` 太弱时，模型可以继续放大 raw residual，实际 forward
始终只看10%边界上的方向。结果是：

```text
有效修正幅度:
  被cap完全决定

raw幅度:
  持续增长

优化重点:
  逐渐退化为“在10%球面边界上找方向”

teacher gain:
  可以保持正向

endpoint稳定性:
  没有因此得到保证
```

这不是 NaN、梯度爆炸或 checkpoint 损坏，而是功能性的 trust-region saturation。

### 2.3 support-dropout 与 support identity

| step窗口 | support-dropout loss mean | wrong-support correct advantage mean | wrong-support win |
|---|---:|---:|---:|
| 1–100 | 0.003217 | +0.007083 | 50.0% |
| 101–200 | 0.045886 | +0.000469 | 59.1% |
| 201–300 | 0.125797 | -0.000731 | 70.0% |
| 301–400 | 0.313092 | -0.000475 | 55.6% |

support-dropout loss 随训练显著上升，说明 LoRA-only 分支越来越偏离 Stock。部署时
support 缺失仍通过 hard bypass 精确回到 Stock，所以工程 fallback 仍安全；但模型内部
没有自然学成“无 support 时接近 Stock”。

wrong-support advantage 仍围绕零，未形成稳定身份敏感性。这不自动否决最终 utility，
但说明目前的 post-CFG teacher gain 不能主要归因于正确 support 的因果使用，更可能混有
LoRA/support adapter 的通用方向修正。

## 3. 当前 S7 的总裁决

### 已经成立

```text
训练数值稳定:
  PASS

step400可复现:
  PASS

post-CFG local teacher gain:
  PASS，且step200后进入平台

301–400 one-step rollout局部趋势:
  比此前更好

exact Stock hard fallback:
  设计上仍保留
```

### 尚未成立

```text
25-step final Mesh Full > Stock:
  未测试step200/300/400

更多训练步数继续带来teacher收益:
  不支持；step200后近似平台

残差幅度得到良好校准:
  FAIL；post-CFG从step101起100%撞cap

non-post/low-t修正安全:
  FAIL；均值持续为负

support-dropout自然回到Stock:
  FAIL；loss随训练增长

correct support身份作用:
  未建立
```

因此停止 S7 是正确的。当前证据不支持原样恢复到 step1000。step400 值得做 endpoint
测试，但不值得仅凭训练 loss 直接解锁1k/3k/10k模型训练。

## 4. 立即执行的 checkpoint 裁决

先不改训练代码，按相同对象、相同初始噪声、相同25-step schedule 依次比较：

```text
Stock
step100 Full
step200 Full
step300 Full
step400 Full
```

资源策略：

```text
第一轮:
  1张GPU
  6对象 × seed42
  用于快速淘汰

第二轮:
  1张GPU
  16–32对象 × seeds42,43,44
  只测试第一轮最好的1–2个checkpoint
```

第一轮至少报告：

```text
Chamfer mean / median / win rate
F-score delta
normal delta
largest-component ratio
catastrophic Mesh count
每个timestep的cap激活率
```

step400 的保留门：

```text
Chamfer mean > 0
Chamfer median > 0
win rate >= 0.60
F-score或normal至少一项同步正向
无新增灾难性坏Mesh
```

若 step400 没有超过 step100，说明 S7 的追加300步主要增加了 cap 饱和和内部幅度，而没有
增加 endpoint utility，应停止 v3。

## 5. 利用现有 step400 的低成本推理消融

在重新训练前，建议对同一 step400 做三项顺序消融：

1. `cap=0.05` 与训练值 `cap=0.10`。

   目的不是挑一个最好看的超参数，而是判断 Mesh 对修正幅度是否过敏。如果0.05明显更稳，
   说明10%边界修正过强；如果两者都无收益，问题更可能在方向/rollout。

2. `support/Full only when t in [0.5,1.0]`。

   当前训练统计显示 non-post 分支平均比 Stock 差。现有 wrapper 在 CFG interval 外仍使用
   strength=1 的 Full 修正。增加探索性 `support_interval_policy=cfg_active_only`，在
   `t<0.5` 精确返回 Stock，可直接检验 late-step harm 是否吞掉早期 teacher gain。

3. RVC off/on 四分支：

   ```text
   Stock, RVC off
   Stock, RVC on
   Full,  RVC off
   Full,  RVC on
   ```

   用于判断跨步 input consistency 是否能保存 Full 的局部方向收益。RVC结果必须单独记录
   计算时间，不能混写成纯 learned gain。

只有上述消融至少一个在 final Mesh 上给出可重复正收益，才值得围绕 v3 checkpoint
继续工程化。

## 6. 若 step400 endpoint 仍失败：v4 解决方案

### 6.1 先修 residual parameterization

不要继续使用“raw residual 任意增长 + detached hard clip”作为主要训练参数化。改为平滑
有界 residual，例如：

```text
raw_ratio = raw_delta_rms / stock_rms
soft_scale = cap / sqrt(raw_ratio^2 + cap^2)
effective_delta = raw_delta * soft_scale
```

并增加：

```text
cap utilization
raw/effective ratio
每个t区间的clip/saturation统计
```

训练短 pilot 中把 `raw_delta_excess_weight` 从0.01比较到0.1；目标不是让 delta 消失，
而是避免 raw/effective 比例随步数单调发散。

更完整的结构是分离方向和幅度：

```text
direction = normalize(raw_direction)
amplitude = cap * sigmoid(gate(t, support_summary))
effective_delta = amplitude * stock_rms * direction
```

这样幅度由可审计 gate 决定，不再由 hard clip 隐式固定。

### 6.2 修复 CFG 与 low-t 分支不平衡

当前 CFG active 时 strength=5 会放大 positive residual，随后几乎总被 post-CFG cap
压回10%；CFG inactive 时同一 Full residual 又略微伤害 Stock。

v4 应：

- 明确以最终 post-CFG residual 为训练对象；
- 让 amplitude/gate 感知 `t` 和 applied CFG strength；
- 初版在 `t<0.5` 使用 exact Stock，之后只有在 endpoint ablation 支持时再开放 low-t
  residual；
- 分别报告 CFG-active 与 non-CFG loss，禁止只用混合平均值。

### 6.3 从一步 rollout 改为真实 schedule 的1/2/4步 curriculum

```text
阶段1:
  1-step，验证新残差参数化

阶段2:
  随机2-step truncated rollout

阶段3:
  随机4-step truncated rollout

共同要求:
  使用正式25-step中的真实 timestep
  使用正式post-CFG composition
  随机起始t
  scheduled mixing teacher state / self-generated state
  每个horizon独立报告endpoint proxy
```

不建议直接25步全反传；2/4步 truncated rollout 更适合2卡3090。

### 6.4 加入轻量 endpoint-oriented 目标

velocity MSE 继续保留，但增加未来 `x0/target-SLAT` 预测损失。若显存允许，再在低-`t`
小比例事件中加入 decoder feature 或 differentiable rendering proxy。该损失必须单独
记录，避免再次出现 local MSE 很好而 Mesh 不动。

## 7. 10k 数据构建和 GPU 预算

2026-07-28 首次检查时，R1 正在复用审计 Omni category，约推进到130/215；本报告完成前
R1 已结束并通过 R2A 硬审计：

```text
service:
  tracker-mixed10k-source-freeze-obj11546-omni215-v2.service

GPU:
  0张

最终状态:
  source freeze PASS
  formal mixed source plan = true
  Objaverse = 11546，coverage = 1.0
  Omni archives = 215
  Omni discovered = 5896
  Omni valid = 5889
  Omni rejected = 7（0.1187%）
  Objaverse shards = 16
  Omni shards = 8
  全部hard guards = true
```

7个拒绝对象均有明确 category/object/asset 记录，包括原失败对象
`watermelon_015/Scan/Scan.obj`。这验证了 bounded rejection 修复按预期工作，没有静默
放宽 source gate。

R1 已完成，不再占用 CPU/IO worker。下一步数据侧是 R3 两来源单卡 smoke，而不是直接正式
全量渲染。后续项目 GPU 使用冻结为：

```text
硬上限:
  数据构建 + Direct-SLAT训练/评估 <= 4张GPU

推荐常态:
  模型训练2卡 + render worker 2卡 = 4卡

checkpoint评估:
  模型评估1卡 + render worker最多2卡 = 3卡

没有模型任务:
  render最多4卡

禁止:
  render 4卡 + train 2卡
  为追赶进度临时超过4卡
  未检查他人占卡就按固定GPU编号启动
```

`Mixed10k_安全重启与正式构建命令_20260728.txt` 已增加
`MODEL_GPU_COUNT + render workers <= 4` 的启动门。默认正式渲染仍为2卡。

## 8. 最终执行顺序

```text
现在:
  保留S7停止状态
  继续R1 source freeze（0 GPU）

R1通过后:
  完成R2A硬审计
  单卡做两来源smoke
  最多2卡做Omni256 pilot

模型侧:
  单卡完成step100/200/300/400 endpoint sweep

若step400 Mesh正向:
  做cap、active-only、RVC消融
  再决定是否从最佳checkpoint续训

若step400仍近零:
  不恢复v3 S7
  实现v4平滑有界残差 + CFG/low-t门控 + 1/2/4-step rollout

数据侧:
  10k基础数据继续构建
  Direct-SLAT模型仍按897 -> 1k -> 3k -> 10k门控扩量
```

核心结论：

> S7 没有数值崩溃，但已经从“学习可校准 residual”转为“持续增大 raw residual 并由
> 10% cap 裁切”的饱和解。step400 的 one-step rollout 信号比 step100 更好，值得一次
> final-Mesh 裁决；在裁决前不恢复到 step1000。若 endpoint 仍失败，下一步重点不是继续
> 增加步数或数据，而是先修 residual 参数化、low-t 分支和多步 exposure bias。
