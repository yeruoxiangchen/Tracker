# A72 strict_perf_v1 / fix1 CUDA验证与性能裁决

日期：2026-08-15。硬件为8×NVIDIA RTX PRO 6000 Blackwell Server Edition 96GB；
PyTorch 2.10.0+cu130、CUDA 13.0、bf16、gradient checkpointing、spconv native、
flash-attn。固定输入为step10000 checkpoint：

```text
SHA256 e6bee573a320ce348bf3c19d41c81b9c8e3b592d49a020e445037e281b0a4f49
world_size=8, global effective batch=8
```

## v1失败与fix1根因闭环

旧包`Tracker_A72_strict_perf_v1.tar.zst`（SHA256
`8fc0bd9d2922105b3f4956e6e4a7dcbb9e84e5fff106667e1296a3c9331541e8`）在第一次
10000→10005八卡真实CUDA smoke的首个conditional projection forward失败。八个rank均出现
CPU indices与本rank CUDA visual的`index_select` device mismatch，且没有完成optimizer update；
原step10000 checkpoint未受影响。失败output永久保留于：

```text
/root/autodl-fs/data/zjr/proobjaverse_official_slat_train2000_20260813_v1/
  perf_smoke_step10000_10005_strict_w2_t2_v1
```

真实根因是`DDP(device_ids=[local_rank])`在forward前递归迁移kwargs，使完整
`lifting_sample`提前进入GPU。fix1使用`DDP(device_ids=None)`，由caller显式放置真正的GPU
inputs；projection入口验证完整lifting tensor tree仍在CPU，再只迁移selected DINO/K/T。
不能用“把indices改成CUDA”替代这一修复，因为那会保留全payload H2D。

## PyTorch 2.10 test-only兼容

fix1初次在A72运行36项测试时，唯一错误来自测试probe绕过真实DDP构造、未初始化PyTorch
2.10新增读取的`_use_python_reducer`。正式训练代码没有失败。测试probe补充：

```python
self._use_python_reducer = False
```

后为`Ran 36 tests / OK / RC=0`。这是test harness兼容，不改变训练runtime。

## 真实CUDA合同结果

双GPU真实DDP smoke结果：

```json
{
  "conditional_lifting_device": "cpu",
  "ddp_device_ids": null,
  "gradient_finite_nonzero": true,
  "no_sync_exercised": true,
  "optimizer_updates": 1,
  "passed": true,
  "unconditional_projection_executed": false,
  "world_size": 2
}
```

随后八GPU从原始step10000独立运行到10005，完整产生10001--10005；
`stage_complete=true`、`performance_summary.passed=true`、RC=0。报告记录
`ddp_device_ids=null`、完整lifting保持CPU和selected DINO/K/T在projection内迁移。由此fix1
的CUDA/DDP/device合同正式通过。

## 公平性能A/B

所有运行使用相同step10000、8 GPU、global batch 8、RNG/sample stream、`log_every=10`、
bf16和gradient checkpointing；稳定窗口统一为10010→10050（40 optimizer steps）。

| Arm | 关键host配置 | 稳定s/step | 相对legacy t2吞吐 |
|---|---|---:|---:|
| Legacy t2 | OMP/MKL=2 | 0.792473 | 1.000× |
| Legacy t4 | OMP/MKL=4 | ≈0.747500 | 辅助参考，不作同线程分母 |
| strict fix1 t2 | workers2/persistent/prefetch2/intra2/interop1/finite checks on | 0.653624 | 1.212× |
| audited-cache fix1 t2 | 同上，仅finite checks off | 0.635384 | 1.247× |

strict相对同thread legacy将step time降低约17.5%，吞吐提高约21.2%；audited-cache分别约
19.8%和24.7%。audited相对strict只有约2.9%额外吞吐，因此正式长训选择strict并保留
per-sample cache finite checks。

## 历史8.16秒异常的归因边界

历史正式8002→10000报告记录1998步、16312.956177秒，即8.164643 s/step、约0.979835
global samples/s。当前相同legacy代码重跑约0.75--0.79 s/step，二者约有一个数量级差异。

已知历史现场曾出现run queue约250--500、CPU user约95%、单rank CPU接近2000%，GPU呈
burst与长SM=0空窗；旧路径又包含workers0、同步约90MB cache load、全tensor finite扫描、
全views/branches和lifting payload H2D、逐tensor `.item()`同步及非foreach EMA。这些现象与
host starvation一致，但历史tmux父环境已不存在，且当前unset thread probe会由torchrun把
OMP设为1，不能证明历史线程配置。故其余差异正式标记为：

```text
UNRESOLVED HISTORICAL RUNTIME / HOST-RESOURCE ANOMALY
```

严禁写“fix1实现12×提速”。可审计结论仅为：同checkpoint、同8 GPU/global batch/RNG、
同thread=2的公平A/B中，legacy 0.7925 s/step降至strict fix1 0.6536 s/step，吞吐提高约21%。

## 固化决策

- 正式固化strict fix1：world8、global batch8、workers2、persistent workers、prefetch2、
  intra-op2、interop1、cache finite checks开启。
- audited-cache不作为默认。
- 暂停repeated stem merge、关闭gradient checkpointing、batch-layout变化、Stock诊断频率
  变化等第二波激进优化。
- 旧v1包、失败output和原step10000 checkpoint继续作为不可变证据保留。
