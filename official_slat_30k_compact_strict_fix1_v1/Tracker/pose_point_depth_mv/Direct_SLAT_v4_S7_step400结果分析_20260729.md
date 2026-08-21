# Direct-SLAT V4 S7 step400 结果分析

日期：2026-07-29

## 1. 结论

S7 的命令执行、checkpoint 载入、same-coordinate/same-noise Mesh 导出均成功，但
step400 **没有建立 Full 相对 Stock 的稳定 Mesh 优势**。因此：

```text
S7 command execution:                    PASS
checkpoint / rollout semantic binding:   PASS
low-t exact-Stock gate:                  PASS
step400 Full-vs-Stock Mesh superiority:  NOT ESTABLISHED
step400 checkpoint promotion:            REJECT
continue directly to step1000:           REJECT
start 10k model long training with V4:   REJECT
```

10k 数据集的下载、清洗、渲染和 finalization 可以继续；这里阻塞的是把当前 V4
配置迁移到 10k 上进行模型长训。

## 2. 运行身份与范围

训练与评估路径：

```text
train:
  /data/zjr/direct_slat_v4_rollout_endpoint_20260728/
  train897_step400_from_step100_seed42_2gpu_v1

checkpoint:
  checkpoints/step_000400.pt

Mesh report:
  mesh6_v4_seed42_v1/report.json
```

冻结哈希：

```text
step_000400.pt:
  09f8b3f35154767d63c11a7f9465db03e9edf7620cfe42b238aee3e719fd05e2

train report.json:
  1b8d1234329e823df8d7d57f146f221075481a5c0897a4799bf1ddf200504e76

Mesh report.json:
  01d7d2344f1ad1f1fe5289a8f2badd1838beb2467fd1d1e52fe872f310bd593f
```

训练使用3777个 sequence samples、897个对象，step100 续训到 step400。训练和 Mesh
评估均为：

```text
training_semantics = rollout_endpoint_v4
guided_delta_policy = post_cfg_v2
bound_mode = smooth_rms_v2
support_interval_policy = cfg_active_only_v1
CFG = 5.0, interval [0.5, 1.0]
Mesh steps = 25, rescale_t = 3.0
```

S7 Mesh 仍是6对象、seed42的 exploratory checkpoint-selection protocol，不是 fresh
blind confirmatory protocol。

## 3. step400 Full-vs-Stock Mesh 结果

正数表示 Full 优于 Stock：

| 指标 | mean | median | 胜率 | bootstrap mean 95% CI |
|---|---:|---:|---:|---:|
| Chamfer improvement | +0.00002629 | -0.00012975 | 3/6 | [-0.00051616, +0.00060001] |
| F-score delta | -0.00029524 | +0.00012973 | 3/6 | [-0.00293185, +0.00212676] |
| Normal consistency delta | -0.00027574 | -0.00001929 | 3/6 | [-0.00170437, +0.00092112] |
| Largest-component ratio delta | -0.00026007 | -0.00021193 | 1/6 | [-0.00057799, +0.00003433] |

四项置信区间全部跨0。Chamfer 仅均值微正，median 为负；F-score、normal 和
largest-component ratio 的均值均为负。尤其 largest-component ratio 只有1/6对象获益，
说明 step400 没有改善 Mesh 连通性主门。

逐对象也不存在一致共赢：

```text
04d712...: Chamfer / F-score / normal 正，LCR 负
092bb8...: Chamfer / normal 正，F-score / LCR 负
0cd491...: Chamfer / F-score / LCR 正，normal 负
0d1814...: 仅 F-score 正，其他负
262afa...: 仅 normal 正；Chamfer=-0.00092013，F-score=-0.00576911
29ddf5...: 四个主指标均负
```

这不是被单个统计量掩盖的稳定改善，而是对象间、指标间方向不一致。

## 4. 与 step100 的同协议配对比较

| 指标 | step100 mean / median / 胜率 | step400 mean / median / 胜率 | 裁决 |
|---|---|---|---|
| Chamfer | +0.00003247 / -0.00023026 / 2/6 | +0.00002629 / -0.00012975 / 3/6 | median和胜率改善，但mean略降，仍不稳定 |
| F-score | -0.00006587 / -0.00029765 / 3/6 | -0.00029524 / +0.00012973 / 3/6 | median改善，但mean变差 |
| Normal | -0.00284503 / +0.00008225 / 3/6 | -0.00027574 / -0.00001929 / 3/6 | mean改善主要来自旧负向离群对象减轻，median略降 |
| LCR | -0.00023004 / -0.00017037 / 2/6 | -0.00026007 / -0.00021193 / 1/6 | 全面变差 |

step400 相对 step100 的 mean 变化：

```text
Chamfer improvement:  -0.00000618
F-score delta:        -0.00022937
Normal delta:         +0.00256929
LCR delta:            -0.00003003
```

所以 step100 到 step400 的额外训练没有形成多指标共同改善，不能据此认为“训练量还不
够，再继续到 step1000 就会自然变好”。

## 5. 最关键的机制信号：residual 已全面贴近 bound

同一6对象、同一rollout协议下：

```text
                                      step100              step400
support-active calls:                 114                  114
raw ratio > 0.1 cap:                  5 / 114              114 / 114
smooth scaling participated:          114 / 114            114 / 114
raw delta / Stock RMS mean (active):  0.07059              0.22211
raw delta / Stock RMS max (active):   0.13174              0.50901
effective ratio mean (active):        0.05671              0.08838
effective ratio max (active):         0.07965              0.09812
```

这里修正一个统计口径：`smooth_rms_v2` 对所有非零 residual 连续缩放，因此
“smooth scaling participated”在 step100 已是114/114；`raw ratio > cap` 才表示
未约束 residual 是否越过配置的0.1半径。旧的 `0.05365/0.16881` 与
`0.04310/0.06717` 是把36个 low-`t` exact-Stock 零调用也纳入150次调用平均后的值；
上表改为只在114次 support-active 调用上统计，更适合诊断 bound。

step400 的每一个 active-CFG 调用的 raw ratio 都已超过0.1；有效比例被限制在约0.1
以内，但 raw correction 已明显越过允许半径。这说明更多训练主要在增加未约束修正
幅度，然后由 bound 把它压回，而不是学习更精确的小幅修正。

训练审计也呈现相同分离。step1-100 与 resume 的 step101-400 分段统计分别为：

```text
rollout gain vs Stock:       +0.09049 -> +0.21912
endpoint x0 loss:             1.13706 -> 0.97119
raw-delta excess loss:        0.00357 -> 0.04825
support-dropout loss:         0.00421 -> 0.14397
wrong-support Stock loss:     0.02142 -> 0.04279
rank pass rate:               0.48930 -> 0.47231
```

训练空间中的 rollout/endpoint proxy 在改善，但 raw residual 越界、dropout reversion
和最终 Mesh 没有同步改善。这进一步支持“训练目标到最终 decoder/Mesh utility 仍有
错配”，而不是单纯训练不足。

## 6. 已排除的问题

这次失败不能归因于以下旧问题：

```text
训练未完成: false，step400 completed=true，训练和S7 exit code均为0
Full/Stock初始噪声不同: false，same_initial_noise=true
坐标不同: false，same_coordinates=true
低t仍注入Full: false，36个 t<0.5 调用的最大有效delta为0
support缺失时Stock回退损坏: false，post-training equivalence passed
评估静默改写V4策略: false，checkpoint与export策略字段绑定一致
```

所以本轮结论指向 V4 的优化目标/残差利用方式，而不是评估基础设施错误。

## 7. 下一步

1. 不执行 step1000，不用 step400 作为10k长训起点。
2. 现有 `step_000200.pt`、`step_000300.pt` 尚未跑同一 Mesh protocol。先以完全相同
   的6对象、seed42、same-noise命令评估 step200/300，建立 step100/200/300/400
   checkpoint 曲线。
3. 只有中间 checkpoint 出现方向一致的多指标正向趋势，才将其送入 fresh blind
   protocol；当前 step400 本身不值得直接送 blind confirmatory test。
4. 若 step200/300 同样中性，则停止对 V4 做同配置扩步。下一版应优先解决：

   ```text
   raw residual长期贴近bound
   support/dropout reversion随训练恶化
   endpoint latent proxy与最终decoder/Mesh指标不一致
   ```

5. 10k 数据构建继续，但模型长训仍等待897-object阶段出现可重复的最终 Mesh utility。
