# A72 strict_perf_v1 fix1：DDP/CPU lifting 合同修复

本文件记录生成原fix1包时的预部署状态：`READY FOR A72 CUDA RETEST`。该包随后已在A72
通过双GPU DDP合同测试和八GPU 10000→10005真实训练smoke；最终CUDA与性能裁决见
`A72_STRICT_PERF_V1_FIX1_A72_RESULTS.md`。

## 保留的失败证据

- 旧归档：`Tracker_A72_strict_perf_v1.tar.zst`
- 旧归档 SHA256：`8fc0bd9d2922105b3f4956e6e4a7dcbb9e84e5fff106667e1296a3c9331541e8`
- A72 失败输出：
  `perf_smoke_step10000_10005_strict_w2_t2_v1`
- 失败日志：上述目录的 `train.log`
- 失败点：`project_sparse_frustum_dino()` 在八个 rank 均以 CPU index 对 CUDA
  `visual_patch_features` 执行 `index_select`。

旧归档和旧失败输出不得删除、覆盖、续跑或作为 fix1 输出目录复用。

## 已确认根因

旧 v1 以：

```python
DistributedDataParallel(
    training_forward,
    device_ids=[local_rank],
    output_device=local_rank,
    ...
)
```

包装 forward。当前源端 PyTorch 2.4.0 的真实 `DDP._pre_forward()` 显示：只要
`self.device_ids` 非空，就会调用 `_to_kwargs()`；最小运行实验进一步确认
`_to_kwargs()` 会递归迁移嵌套 dict/list/tuple 中的 Tensor。A72 的八 rank CUDA
trace 与此行为完全一致：完整 `lifting_sample` 在进入模型前已被搬到本 rank GPU。

因此旧 v1 同时存在两个问题：

1. CPU `view_indices` 与 CUDA visual/K/T 设备不一致，直接报错；
2. 更重要的是，DDP 已经全量迁移 lifting payload，令“CPU 选 view，再仅传 selected
   visual/K/T”的性能优化失效。

## fix1 的最小修改

1. DDP 改为 `device_ids=None`，不再指定 `output_device`。每个 rank 的模型早已显式位于
   `cuda:local_rank`；`x_t`、`t`、condition 和 stock velocity 也都在调用 wrapped forward
   前显式置于该 GPU。DDP 仍负责 gradient all-reduce、`no_sync()` 和 gradient bucket
   view，但不再替调用者迁移 inputs/kwargs。
2. `project_sparse_frustum_dino()` 的 strict 路径会递归检查完整 lifting sample 中每个
   Tensor leaf 都在 CPU。该检查覆盖实际使用的 visual/K/T，也覆盖 depth、confidence、
   masks、prior、stock_condition、target 等本次 SLat projection 不数值使用的大 Tensor。
3. 在 CPU 上对 visual/K/T 做同一组 `index_select`，随后只把 selected visual/K/T 以
   `non_blocking=True` 传到本 rank GPU。`predicted_depth` 只读取 Python shape `(H, W)`；
   grid transform、extrinsics type 和 camera sign 只读取元数据。
4. 训练 runtime summary 标记为 `a72_strict_perf_v1_fix1`。DDP device policy 只进入
   `model_summary.runtime_performance`，没有新增 CLI 参数，也没有进入 checkpoint args 或
   data identity。
5. fix1 launcher 拒绝任何已存在的 `OUTPUT_DIR`，避免误用旧失败目录。

未修改随机调用及顺序：CUDA timestep sampling、view-count `randint`、`randperm`、Python
unconditional draw 均保持原样。未修改模型、loss、LR、optimizer、EMA、batch、gradient
checkpointing、Stock forward 或 cache finite profile。

NCCL 的 device-id warning 未在本修复中处理；它与本次输入迁移根因独立，先保留以避免
扩大修改范围。

源端已重新读取真实 step10000，SHA256 仍为
`e6bee573a320ce348bf3c19d41c81b9c8e3b592d49a020e445037e281b0a4f49`；其
10000→20000、world8、grad-accum1 合同验证通过，optimizer/EMA 继承且 per-rank RNG
仍 exact。旧 v1 与 fix1 对同一 CLI 生成的 canonical checkpoint args 均为 903 bytes，
SHA256 同为 `1b3ccb0dc8da082364754d0d5b5cc821440c304f7b77fe8185f038d448f20cda`。

完整 preflight 不能在源服务器对 step10000 重放：该 checkpoint 按严格合同冻结了 A72
的 `/dev/shm/...` 与 `/autodl-fs/...` resolved paths，而这些物理路径在源机不存在；
validator 正确地拒绝伪造 relocation。用户提供的 A72 preflight 已在这些物理对象存在时
完整通过。fix1 没有修改 preflight 或任何 identity/deployment validator。

## 新增测试

- 用真实 `torch.distributed.utils._to_kwargs` 验证嵌套 Tensor 的递归目标设备迁移；
- 直接运行真实 `DDP._pre_forward()` 两个分支，验证 `device_ids=[0]` 会迁移、
  `device_ids=None` 会保留调用者设备；
- strict contract 检查全部 Tensor leaves，未知嵌套 CUDA/meta leaf 也拒绝；
- 2/4/8 view CPU selection 与旧顺序相同；
- 2/4/8 view projection 数值一致；
- CUDA 可用时验证“CPU select→H2D”和“全量 H2D→GPU select”selected inputs 完全一致；
- conditional 与 unconditional 路由、finite/nonzero gradients；
- 独立多 rank CUDA smoke：`DDP(device_ids=None)`、mixed CPU/CUDA kwargs、`no_sync()`、
  gradient all-reduce 和一次 optimizer update。

## A72 fix1 复测顺序

### 1. 校验并解压到新代码目录

```bash
set -euo pipefail

ARCHIVE=/root/autodl-fs/transfer/Tracker_A72_strict_perf_v1_fix1.tar.zst
sha256sum -c "${ARCHIVE}.sha256"

test ! -e /root/Tracker_perf_v1_fix1
mkdir -p /root/Tracker_perf_v1_fix1
tar --zstd -xf "${ARCHIVE}" -C /root/Tracker_perf_v1_fix1
```

归档内容以 Tracker runtime 文件为根；解压完成后应存在：

```text
/root/Tracker_perf_v1_fix1/pose_point_depth_mv/train_native_slat_genrecon.py
```

### 2. CPU/static 回归

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate reconviagen-bw

cd /root/Tracker_perf_v1_fix1
export PYTHONPATH="$PWD:$PWD/ReconViaGen:$PWD/ReconViaGen/wheels/vggt"
export ATTN_BACKEND=flash_attn
export SPCONV_ALGO=native

python -m unittest -v \
  pose_point_depth_mv.test_a72_ddp_cpu_lifting_contract \
  pose_point_depth_mv.test_a72_strict_perf_runtime \
  pose_point_depth_mv.test_native_slat_resume_data_identity \
  pose_point_depth_mv.test_native_ss_deployment_relocation
```

### 3. 先跑独立双 rank CUDA/DDP 合同 smoke

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m pose_point_depth_mv.a72_ddp_cpu_lifting_cuda_smoke \
  | tee /root/autodl-tmp/a72_ddp_cpu_lifting_cuda_fix1.log
```

必须输出 `"passed": true`、`"ddp_device_ids": null`、
`"conditional_lifting_device": "cpu"` 和 `"optimizer_updates": 1`。

### 4. 从原始 step10000 做新的八卡 10000→10005 strict smoke

```bash
set -euo pipefail

cd /root/Tracker_perf_v1_fix1

export RESUME_CHECKPOINT=/root/autodl-fs/data/zjr/proobjaverse_official_slat_train2000_20260813_v1/B_condition_lora_train2000_step10000_seed42_8gpu_a72_v1/checkpoints/step_010000.pt
export OUTPUT_DIR=/root/autodl-fs/data/zjr/proobjaverse_official_slat_train2000_20260813_v1/perf_smoke_step10000_10005_strict_w2_t2_fix1_v1
export START_STEP=10000
export MAX_STEPS=20000
export RUN_UNTIL_STEP=10005
export PERF_PROFILE=strict
export NUM_WORKERS=2
export PREFETCH_FACTOR=2
export TORCH_NUM_THREADS=2
export TORCH_NUM_INTEROP_THREADS=1
export LOG_EVERY=1
export SAVE_EVERY=1000

test ! -e "${OUTPUT_DIR}"
bash run_a72_slat_perf_resume.sh
```

必须仍从原始 checkpoint：

```text
step_010000.pt
SHA256 e6bee573a320ce348bf3c19d41c81b9c8e3b592d49a020e445037e281b0a4f49
```

启动。不要从旧失败 output resume。A72 完成上述真实 CUDA 复测之前，本构建的最高状态仅为
`READY FOR A72 CUDA RETEST`。
