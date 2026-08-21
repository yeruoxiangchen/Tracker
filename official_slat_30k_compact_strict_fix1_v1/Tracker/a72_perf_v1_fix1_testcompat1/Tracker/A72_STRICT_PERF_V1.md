# A72 Native-SLat 隔离性能版 v1

这是一份独立运行树，目标是从 official ProObjaverse no-VGGT Native-SLat v2
checkpoint 继续训练，同时不覆盖 A72 当前 `/root/Tracker`。部署目录必须使用新的
`/root/Tracker_perf_v1`；当前长训结束前不要改变正在运行进程的 cwd、`PYTHONPATH` 或文件。

## 不变的训练合同

本版本不改变：模型结构、LoRA 参数、loss、AdamW、LR、EMA 公式、global effective batch、
采样器、样本顺序、`t` 分布、`p_uncond`、条件视图的随机抽样、Stock context、数据 identity、
resume identity 或 relocation 白名单。新增参数均从 checkpoint args 中排除，并记录在
`model_summary.runtime_performance`。

8 GPU 继续训练仍固定为：每 rank batch 1、`grad_accum=1`、global batch 8。不能为了提高
GPU 利用率直接把每卡 batch 改成 2；那会把 global batch 改成 16，属于新的训练实验。

## 已实现的热路径优化

1. 将梯度、参数和 optimizer-state 的逐张量 `.item()` 有限性检查合并成每设备一次
   host synchronization；真实 step8000 checkpoint 对应约 298 个 trainable tensor、298 个
   EMA tensor 和 894 个 optimizer-state tensor，旧实现会产生大量小同步。
2. 三组梯度范数合并为一次 device-to-host 读取；八个每-step诊断标量也合并为一次读取。
3. EMA 更新按 device/dtype 分组，使用 `torch._foreach_mul_` 与
   `torch._foreach_add_`，公式不变。
4. 先在 CPU 选择实际的正/负条件分支与 1--8 个随机视图，再做异步 H2D；posed-DINO
   projection 同样先选视图再传 GPU。抽样仍在原 CUDA RNG 流上发生，抽样顺序不变。
5. DDP 启用 `gradient_as_bucket_view=True`，减少 gradient bucket 额外内存拷贝。
6. DataLoader 支持 persistent workers、prefetch factor、pin memory；并可显式限制每 rank
   PyTorch intra/inter-op 线程，避免 8 ranks 各自展开整机线程池。
7. 每个 optimizer boundary 记录 `optimizer_step_wall_seconds`，可直接统计稳定窗口，不再
   从短窗口或启动时间猜速度。

源端真实单样本中，旧路径每步可能向 GPU 搬运约 67 MB 的正/负 context 与 posed-DINO；
新路径按平均 4.5 个视图、单一实际分支约为 25 MB。实际端到端收益仍必须在 A72 smoke
中测量，不能把字节比例当作训练 speedup。

## 两种运行 profile

- `PERF_PROFILE=strict`：保留每次读取后的全张量 `isfinite` 扫描。先用它做 10--20 step
  等价 smoke。
- `PERF_PROFILE=audited-cache`：显式跳过重复的 per-sample 全张量 finite 扫描，但仍保留
  shape、UID、config、deployment、checkpoint、梯度、optimizer、EMA 和 save-time finite
  检查。该选项被限制为 `--resume`，只应用于已通过源审计、tar SHA256 与解压验证且之后
  保持不可变的 168 GB cache。

源端同一个已缓存样本的只读测量为：strict 0.231 s，audited-cache 0.060 s，返回 payload
逐张量完全相同；这是 CPU loader 微基准，不是 A72 的端到端 speedup。

## A72 安全部署

先把归档解压到新目录，不要覆盖 `/root/Tracker`：

```bash
mkdir -p /root/Tracker_perf_v1
tar --zstd -xf /root/autodl-fs/transfer/Tracker_A72_strict_perf_v1.tar.zst \
  -C /root/Tracker_perf_v1

cd /root/Tracker_perf_v1
export PYTHONPATH=/root/Tracker_perf_v1:/root/Tracker_perf_v1/ReconViaGen:/root/Tracker_perf_v1/ReconViaGen/wheels/vggt
python -m unittest -v \
  pose_point_depth_mv.test_a72_strict_perf_runtime \
  pose_point_depth_mv.test_native_slat_resume_data_identity \
  pose_point_depth_mv.test_native_ss_deployment_relocation
```

检查当前长训 PID 的 cwd 与命令仍指向旧树：

```bash
for PID in $(pgrep -f 'train_proobjaverse_official_slat_condition_lora'); do
  printf 'PID=%s cwd=' "$PID"
  readlink -f "/proc/$PID/cwd"
  tr '\0' ' ' < "/proc/$PID/cmdline"
  echo
done
```

## step10000 后的推荐 benchmark

先完成并冻结 step10000 的 Train64/Dev64 与新 Native-SS 端到端评估；性能代码不应替代
这个科学决策点。随后所有候选配置都从同一个只读 step10000 checkpoint 启动到10020，
使用不同 output 目录，比较去掉前2步后的稳定时间。

严格 profile：

```bash
cd /root/Tracker_perf_v1
export RESUME_CHECKPOINT=/data/zjr/proobjaverse_official_slat_train2000_20260813_v1/B_condition_lora_train2000_step10000_seed42_8gpu_v1/checkpoints/step_010000.pt
export START_STEP=10000
export MAX_STEPS=20000
export RUN_UNTIL_STEP=10020
export OUTPUT_DIR=/data/zjr/proobjaverse_official_slat_train2000_20260813_v1/perf_smoke_step10000_10020_strict_w2_t2_v1
export PERF_PROFILE=strict
export NUM_WORKERS=2
export PREFETCH_FACTOR=2
export TORCH_NUM_THREADS=2
export TORCH_NUM_INTEROP_THREADS=1
bash run_a72_slat_perf_resume.sh
```

确认 strict 的 loss、finite 检查、checkpoint、EMA 和 preflight 全部正常后，从同一个
step10000 运行 audited-cache profile：

```bash
export OUTPUT_DIR=/data/zjr/proobjaverse_official_slat_train2000_20260813_v1/perf_smoke_step10000_10020_audited_w2_t2_v1
export PERF_PROFILE=audited-cache
bash run_a72_slat_perf_resume.sh
```

每个目录会生成 `performance_summary.json`，包含 mean/median/p90 s/step、step/hour 与
global samples/s。不要串接两个 benchmark checkpoint；选定 profile 后，应从原始
step10000 重新启动正式延长轨迹。

## 调参顺序

保持模型参数完全不变，只比较以下小矩阵：

1. strict：workers=0、threads=2；
2. strict：workers=2、threads=2；
3. audited-cache：workers=2、threads=2；
4. 若第3项仍明显 data wait，再测 workers=4、threads=2。

每项至少20步，丢弃头2步，比较 median 与 p90。不要同时改 workers、线程数和训练参数，
否则无法解释收益。`num_workers` 改变不会改变当前 dataset 的随机性，因为 `__getitem__`
没有随机抽样；训练随机抽样仍发生在 rank 主进程。

RTX PRO 6000 的优势不仅是96 GB显存，但当前 dynamic sparse、每GPU batch1、频繁 CPU load
和小同步无法持续填满 GPU。显存容量只有在提高单卡工作量或容纳更大模型时才直接转化为
吞吐；本项目受 global-batch=8 合同与 batch-one sparse实现约束，不能仅靠换卡获得理论峰值。
