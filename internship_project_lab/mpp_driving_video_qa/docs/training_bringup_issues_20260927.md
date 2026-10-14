# 训练启动问题记录（截至 2026-09-27 UTC）

本文记录正式 LingoQA SFT 开始前及首个优化步骤附近暴露的资源和模型加载问题。
它不代表模型效果实验结果。本文记录故障发生当时的训练前状态；全量 SFT 已于
2026-09-27 随后完成，最终训练与效果结论以 `project_timeline.md` 为准。

## 当前固定配置

- 基座：MPP-Qwen-Next，Qwen-7B Chat。
- 数据：LingoQA train 30,000 条；视觉缓存共 17,869 个 segment。
- 输入缓存：每个 segment 为 `[5, 32, 4096]`，约 22 GB。
- SFT：LoRA `r=16`、`alpha=32`、target `c_attn`、bf16、batch size 1、
  gradient accumulation 4、3 epochs。
- 四卡计划：4-way DDP，每卡一份完整 7B 模型；计划 optimizer steps 5,625。
- 暂存资产：`/tmp/zjr_mpp_fast`，包含约 15 GB Qwen、15 GB MPP sharded
  checkpoint 和约 22 GB 视觉缓存。训练/评测完成前不能删除视觉缓存。

## 1. `/home` 机械盘拥堵导致模型加载极慢

### 现象

`mpp-lingoqa-sft-v4` 在 2026-09-26 16:37:07 开始加载 MPP shard 1，
16:53:36 才进入 shard 2。rank 0 处于 `D` 状态，内核等待点为
`wait_on_page_bit_common`。

### 证据与原因

- 权重和原始缓存位于 `/home` 的 `/dev/sdb1`（机械盘）。
- 实测 `/dev/sdb` 利用率约 98%--100%，读取等待约 46--72 ms。
- 同盘还有多个其他用户的 HOT3D 训练和缓存任务处于 I/O 等待。
- 本项目当时只获得约 1.6--2 MB/s 的读取速度。
- 四卡采用串行 CPU 加载以避免同时读取 15 GB checkpoint；因此其余 rank
  会等待当前 rank 完成。

### 处理

将 Qwen、MPP sharded checkpoint 和视觉缓存通过 `rsync` 暂存到
`/tmp/zjr_mpp_fast`。`/tmp` 位于 Samsung SSD `/dev/sdc2`。
移动后 MPP shard 通常约 5 秒进入下一个 shard，证明原慢速主要来自源盘，
不是 checkpoint 损坏或 Python 死锁。

### 状态

已规避。原始资产仍在项目目录，`/tmp` 仅为可删除的运行副本。

## 2. 慢加载触发默认分布式超时

### 现象

`mpp-lingoqa-sft-v4` 的 rank 1--3 在 `torch.distributed.barrier()` 等待 rank 0
超过约 30 分钟后报 `Socket Timeout`；torchrun 随后终止仍在加载 shard 2 的
rank 0。该运行未进入训练。

### 原因

`init_process_group("nccl")` 使用默认 timeout，而串行权重加载时间在机械盘拥堵时
超过默认限制。这不是 NCCL 拓扑或网卡故障。

### 处理与状态

`train_sft.py` 已将 process-group timeout 设置为 12 小时。SSD 解决主要耗时，
延长 timeout 作为共享服务器负载波动时的容错。已解决。

## 3. LoRA 包装后的 Qwen core 解析错误

### 现象

`mpp-lingoqa-sft-v5` 四个 rank 均成功加载 MPP 权重，但首个 forward 报：

```text
RuntimeError: could not resolve Qwen core
```

没有完成 optimizer step。

### 原因

旧 `qwen_core()` 沿 `.base_model` 持续解包。Qwen 的
`base_model_prefix="transformer"`，所以遍历从完整 `QWenLMHeadModel` 继续进入了
不含 `lm_head` 的 `QWenModel`。

### 处理与验证

- PEFT 包装通过 `get_base_model()` 解开，并在同时拥有实际 `transformer` 和
  `lm_head` 子模块的完整 causal LM 处停止。
- 小型 Qwen + 真实 PEFT 回归通过：forward、backward、非零 LoRA 梯度、参数更新、
  adapter 保存和重载均成功。
- 单卡真实 MPP 7B + LingoQA 缓存短训练通过：8 micro-batches、2 optimizer
  steps，`tracked_lora_max_abs_delta=4.002688729087822e-05`。
- 结果：`outputs/qwen_core_fix_smoke_20260927/summary.json`。

状态：模型访问错误已解决。详细记录见 `docs/qwen_core_fix_20260927.md`。

## 4. 四卡首个 backward 的 cuBLAS 分配失败

### 现象

`mpp-lingoqa-sft-v6` 使用 GPU `0,5,6,7`：

- rank 0/1/2/3 分别于 10:51:09、10:53:50、10:56:09、11:00:41 报告
  `model_loaded`；说明 SSD 权重、MPP checkpoint 和模型解析均已通过。
- 11:02:13，rank 0 在首个反向传播阶段失败：

```text
RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)
```

- 失败点为 `(loss / divisor).backward()`；没有生成 `progress.json`，即没有完成
  optimizer step。输出目录 `outputs/mpp_lingoqa_lora_v2` 只包含
  `run_config.json` 和 `failure_rank_0.json`，不可作为续训 checkpoint。

### 当前判断

该错误最符合 cuBLAS 创建 handle 时缺少可分配 GPU 内存/资源。当前能确认：

- DDP 每个 rank 保留完整 7B 模型，不会把模型参数分片到四张卡。
- LoRA 只减少可训练参数和优化器状态；反向仍需保留完整语言模型的激活。
- 当前代码没有对语言模型启用 activation/gradient checkpointing。
- GPU 为共享资源。事后采样发现 GPU 0 约有其他进程占用 2.2 GB，GPU 5/6/7
  各约有其他进程占用 3.6 GB。该采样发生在 v6 退出后，不能当作失败瞬间的
  精确显存快照，但证明可用显存会受外部任务影响。
- 单卡短训练通过，证明模型、数据、loss 和 LoRA 反向路径本身可运行；它没有覆盖
  DDP 通信额外显存、四卡同时训练及真实共享显存竞争。

当前没有证据支持把它归因于数据错误、SSD 缓存错误或 Qwen core 修复回归。

### 2026-09-27 修复

已完成以下代码修改：

1. 默认启用 Qwen gradient checkpointing，并显式关闭训练时 KV cache。
2. 在冻结 token embedding 和缓存视觉特征的条件下，对拼接后的
   `inputs_embeds` 开启梯度。Qwen 使用 re-entrant checkpointing；缺少这一步会
   让 checkpoint 误判没有需要求导的输入，从而丢失 decoder 内 LoRA 梯度。
3. DDP 使用 `gradient_as_bucket_view=True` 和 10 MB bucket，降低梯度 bucket
   的额外峰值。
4. 每个 rank 在 `model_to_device`、`ddp_wrapped`、首次 forward、首次 backward、
   首次 optimizer step 后写入 `memory_rank_N.jsonl`；异常文件也携带当时的 CUDA
   allocator/device memory 状态。
5. `run_pipeline.sh train` 支持 `SFT_MAX_STEPS`、`SFT_MAX_DEV_RECORDS`、
   `SFT_EPOCHS`、`SFT_GRAD_ACCUMULATION` 和 `SFT_GRADIENT_CHECKPOINTING`，因此
   smoke 与全量运行共用同一入口，不再另外维护简化训练脚本。

### 单卡真实验证

`outputs/gc_single_smoke_20260927` 使用完整 MPP-Qwen-Next、真实 LingoQA 缓存和
正式 LoRA 配置完成 8 micro steps / 2 optimizer steps：

- `tracked_lora_max_abs_delta=4.002688729087822e-05`，确认 LoRA 梯度和更新没有
  因 checkpointing 丢失；
- peak allocated 约 14.82 GiB，peak reserved 约 15.35 GiB；
- 训练阶段本身约 11.63 秒；
- loss 与此前未启用 checkpointing 的真实单卡 smoke 对应步骤一致
  （1.171875、2.140625），说明本次资源修复没有改变监督目标。

### 四卡正式配置验证

`outputs/gc_4gpu_smoke_20260927` 使用与全量相同的 GPU `0,5,6,7`、batch size 1、
gradient accumulation 4、30,000 条训练索引和 DDP 入口，仅通过
`SFT_MAX_STEPS=2` 截断运行。结果：

- 完成 8 micro steps / 2 optimizer steps，旧版失败的首次 backward 已通过；
- `tracked_lora_max_abs_delta=4.002681453130208e-05`，与单卡数值一致；
- 四个 rank 都写出了 `model_to_device`、`ddp_wrapped`、首次 forward、首次
  backward 和首次 optimizer step 记录；
- 每 rank 模型上卡后 allocated 约 14.42 GiB，DDP 构造阶段观察到的
  max allocated 约 16.04 GiB；
- rank 1--3 在首个 optimizer step 后 device free 约 3.88 GiB；rank 0 因同卡
  另有约 2.25 GiB 外部占用，仍保留约 4.70 GiB free 并成功完成；
- 训练阶段耗时约 56.93 秒，最终 adapter 和 summary 均已保存。

状态：四卡显存故障已在同配置短训练中修复。全量运行仍应在启动前检查外部 GPU
占用，因为 DDP 构造期间 allocator 的历史 max reserved 接近卡容量；这属于共享卡
资源条件，不是当前训练逻辑错误。

## 5. 监视与失败可见性

`train_sft.py` 已新增：

- `--max-steps`，用于使用正式训练入口限制 smoke optimizer steps；默认 0 不限制。
- 每个 rank 的 `loading_model` / `model_loaded` 日志。
- Python 异常写入 `failure_rank_N.json`。

`report_progress.py` 会优先显示 summary、progress 或 failure。此前只有
`run_config.json` 时显示 `running_or_interrupted`，不能证明服务仍存活；应同时检查：

```bash
systemctl --user show UNIT.service -p ActiveState -p SubState -p Result
```

## 当前结论

数据准备和视觉缓存已经完成，SSD 暂存解决了主要加载瓶颈，模型访问、LoRA 反向
路径以及 gradient-checkpointing 下的参数更新已在真实 7B 单卡短训练中验证。
正式四卡短训练现已验收通过。下一步可直接使用相同入口、换一个全量输出目录并
移除 `SFT_MAX_STEPS`，启动 5,625-step 全量训练。`memory_rank_N.jsonl` 会在全量
运行中继续保留关键阶段证据；当前没有必要为了本轮训练改成 FSDP/ZeRO。
