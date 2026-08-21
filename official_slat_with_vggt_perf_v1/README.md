# Official Train2000 with‑VGGT SLat：strict 性能训练旁路

## 结论

这是一个与 `pose_point_depth_mv/` 同级的独立训练入口。它不修改原 no‑VGGT
代码，也不覆盖任何既有 cache、checkpoint、report 或 freeze。实现方式以复用为主：

- 科学合同复用已冻结的 official with‑VGGT sidecar/model/trainer；
- 运行时复用 A72 已做过真实 CUDA 验证的 strict-fix1 trainer 和 projection；
- 新增的唯一数据路径代码是 lean dataset：保留旧 condition 文件的 manifest/hash
  身份，但每个 sample 不再读取约 42.8 MiB 的 no‑VGGT `condition.pt`，只读取新的
  native `slat_vggt_cond` sidecar；
- 所有被动态复用的关键源码都按 SHA256 锁定，任一漂移会在模型或输出目录创建前
  拒绝启动。

正式入口：

```bash
python -m official_slat_with_vggt_perf_v1.train_proobjaverse_official ...
```

## 性能路径

该入口保留 strict-fix1 已验证的以下行为：

- `DistributedDataParallel(device_ids=None)`，不让 DDP 递归搬运完整 CPU payload；
- lifting 在 CPU 先选 view，只将选中的 DINO/K/T nonblocking H2D；
- native Stock context 在 CPU 先选 positive/negative 分支与 views，再 selected-only H2D；
- `num_workers=2`、`persistent_workers=true`、`prefetch_factor=2`、`pin_memory=true`；
- 每 rank PyTorch intra-op=2、interop=1；
- batched finite reduction、foreach EMA 和 gradient bucket view；
- 保留 per-sample cache finite checks，禁止 audited-cache 跳过模式。

训练期不会运行 VGGT。VGGT 只在 sidecar builder 中运行一次；训练读取缓存后的
`slat_vggt_cond`。1374-token native context 相比 no‑VGGT 1369-token context只多5个
前缀 token，因此在消除冗余 I/O/H2D 后，额外 GPU 计算量应很小，但真实 3090
吞吐仍必须通过 CUDA smoke 实测，不能从 A72 的 0.65 s/step 直接外推。

## 科学合同保持不变

- fresh Stock-equivalent initialization；
- official GT SLat target；
- official known K/T 驱动 posed-DINO；
- VGGT camera/depth 不消费；
- VGGT 不进入 Native-SS；
- every-block zero-init projection、LoRA rank8/alpha16；
- p_uncond=0.1、uniform timestep、随机1–8 views；
- bf16、gradient checkpointing、global effective batch=8；
- loss、LR、optimizer、EMA、decoder、RNG调用顺序不改。

step0 的参照是 V0：

```text
V(step0, native slat_vggt_cond)
== V0 Stock SLat(native slat_vggt_cond)
```

不是 N0。

## 运行依赖

默认复用：

```text
/home/zjr/Tracker/a72_perf_v1_fix1_testcompat1/Tracker
```

迁移到其他主机时，可通过只读环境变量指定同一 strict-fix1 树：

```bash
export WITH_VGGT_STRICT_FIX1_ROOT=/path/to/exact/strict-fix1/Tracker
```

入口会验证 trainer/projection 的固定 SHA；不能用“相似版本”替代。

## 当前验证边界

静态 source identity、import、lean loader、DDP policy 和正负回归可以在无 GPU 环境
完成。8-object sidecar、step0、1GPU、2GPU DDP、8GPU 性能 smoke 必须在真实 CUDA
环境依次通过后，状态才能从 `READY FOR SOURCE 3090 CUDA RETEST` 升级。

完整执行命令见：

[源服务器3090八卡构建训练与性能Smoke命令_20260816.txt](./源服务器3090八卡构建训练与性能Smoke命令_20260816.txt)
