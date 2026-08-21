# Official Train2000 with‑VGGT SLat 最小修改设计（2026‑08‑16）

## 结论

本分支不重写现有 official no‑VGGT cache，也不修改 SLat v2 数学结构。现有
`NativeSLatGenreconV2Flow` 的 Stock cross-attention 本来就接受外部 context；本次
唯一科学变量是把 no‑VGGT 的 per-view DINO patch context（N0）替换为 ReconViaGen
原生 `slat_vggt_cond`（V0），posed-DINO、known K/T、every-block zero-init projection
和 rank8/alpha16 LoRA 全部保持原实现。

## 资产边界

- 当前没有可复用的 historical official with‑VGGT cache。
- 继续只读复用现有 official no‑VGGT：
  - `slat_manifest.json`；
  - `lifting_manifest.json`；
  - official lh-slat target；
  - frozen ordered `view_ids`；
  - raw DINO、K/T、mask 和 target 引用。
- 新增的唯一大对象是每个 object 的 positive native `slat_vggt_cond` sidecar。
- native negative context 本来就是 `zeros_like(cond)`，不重复落盘，dataset 读取时
  精确重建。
- sidecar 不保存、不消费 VGGT camera/depth，也不会替换 official known K/T。
- Native‑SS 不加入 VGGT。GT-support 训练的 identity 原样保留历史
  no‑VGGT step2000 EMA / CFG3 binding（不执行 SS forward）；predicted-support
  部署评测另行冻结新 official step2000 EMA / CFG5。

这里必须区分两份 SS 证据：为了让 N/V 训练只差 SLat context，新 paired manifest
原样继承 no‑VGGT Train2000 cache 当时冻结的 training Native‑SS deployment（旧
`native_no_vggt_mixed.../ss_eval_synthetic_dev32...` report）；它在 GT-support 训练中
本来就不执行。后续 predicted-support Dev48 则固定使用 2026-08-15 新 official
Train2000 Native‑SS step2000 EMA/CFG5 report。不能把新 report 直接传给 trainer，
否则会同时改变 checkpoint data identity，并在 A/B full-deployment guard 处正确拒绝。

### 源服务器 P0 实际资产结果

2026-08-16 已在真实 Train2000 上完成只读 `--mode preflight --max_objects 8`：

- protocol SHA：`36c2147c9d3d37b5dc867ff3d277a4af8f7ad9f2f5099edcc032719a0fe5c241`；
- base SLat manifest SHA：`ead078ec423475ddbd2e4272e990404f74cddcc39b2332d42b4e2da266aab737`；
- base lifting manifest SHA：`4a275be6d4b013a378d4aab325e48617cf1b5b7b77aeee30964558260be1d20b`；
- Trellis snapshot：`647659a5ad5fbf67e22793e7b5e2cee4b30c5d13`；
- `slat_vggt_cond` weights SHA：`18945872ec1a5fc13b2bfe86f279777c895c015ddff5cabc8f5a6dcd245d862a`；
- VGGT snapshot：`aac569c280b52ccda1c84a81cd0e27e947bc2cf5`；
- VGGT weights SHA：`65c4803ef021a4143bbe499faebaef1d1a522d1437b5742b89c85d557db4bba4`；
- DINOv2 weights SHA：`36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51`。
- 最终 builder source SHA：`bfda60739794995ce0777a7bd2eb1ceba176ae3c0e4049b98aaf76671d709991`。

P0 没有执行 VGGT forward，也没有创建 `/tmp/...preflight...` cache 目录。
最终静态回归为 35/35 PASS，同时通过 `py_compile`、shell `bash -n`和
正式入口 `--help` import。

### 容量规划

sidecar 不主动降精度。`slat_vggt_cond` 原生输出若为 FP32，则每对象约
`8×1374×1024×4 = 42.94 MiB`，2000 对象张量净载荷约 `83.86 GiB`；若真实
CUDA smoke 证明输出为 FP16/BF16，则下界约 `41.93 GiB`。正式容量按 FP32
上界规划，不能提前按 FP16 宣称。

源 official no‑VGGT cache 当前约 168 GiB，因此 F39 若同时放 base cache、完整
sidecar、模型和 checkpoint，数据盘最低约 300 GiB，建议 400 GiB；若还要保留 raw
render staging/解包副本，建议 500 GiB。默认 50 GiB 数据盘不够。

## 原生 context 的真实调用链

固定同一组 official `view_ids` 后：

1. 复用 `shared_object_preprocessing.v1` 的 foreground crop/resize；
2. `TrellisVGGTTo3DPipeline.vggt_feat(images)`；
3. `TrellisVGGTTo3DPipeline.encode_image(image_tensor)`；
4. `normalize_image_cond` 恢复 `[B,V,T,1024]`；
5. `TrellisVGGTTo3DPipeline.get_slat_cond(...)`；
6. 保存 positive `[V,T,1024]`，训练时按冻结 view 顺序恢复 list。

同一次 `encode_image` 结果的 patch 部分还会转换到旧 cache 的 FP16 表示，并与
`visual_patch_features` 做预注册容差检查（max abs ≤ 0.01、mean abs ≤ 0.0005）。
每个 sidecar row 另外冻结源 render tar SHA、解码 RGBA SHA 和处理后 RGB SHA。

原生 context 使用完整 DINO cls/register/patch 序列。当前模型资产下，base
no‑VGGT patch 数为 1369，原生 native context 为 1374 tokens，即 5 个前缀 token。
这是 V0/N0 的预期差异，不是 view 或相机变化。

## 新 identity

- Sidecar object：
  `pose_point_depth_mv.proobjaverse_official_slat_vggt_sidecar_object.v1`
- Paired SLat manifest：
  `pose_point_depth_mv.proobjaverse_official_slat_with_vggt_slat_manifest.v1`
- Paired lifting manifest：
  `pose_point_depth_mv.proobjaverse_official_slat_with_vggt_lifting_manifest.v1`
- Checkpoint：
  `pose_point_depth_mv.native_slat_genrecon_official_with_vggt.v1`

pair identity 同时冻结：base manifest SHA、base config/normalization hash、ordered UID、
每个 sidecar SHA、encoder 权重/配置 SHA、Trellis/VGGT/DINO/preprocessing 源码 SHA。
v1 的 pair/base/sidecar/encoder/sample schema 都是精确 key 集合；新增或删除未知
semantic 字段直接拒绝，不做 subset compare。

训练构造阶段沿用当前 no‑VGGT 的 `TrellisImageTo3DPipeline` 路径，不会在每个 DDP
rank 再加载约 5 GB VGGT；该 legacy loader 仍可能短暂物化其他 Stock pipeline 模块和
DINO，不能表述为“只加载 SLat”。VGGT forward 与 `slat_vggt_cond` 只在 sidecar builder
中执行，训练 forward 使用已缓存 context。这只是启动内存优化，不改变 Stock SLat 数学。

## Step‑0 合同

新模型仍由 Stock flow + zero-init every-block projection + zero-init LoRA-B 构造。
训练器启动前沿用正式 `initial_stock_audit()`，但 reference 明确标记为 V0：

`V(step0, native slat_vggt_cond) == V0 Stock SLat(native slat_vggt_cond)`。

它不应回到 N0。conditional/unconditional 最大绝对差都必须为严格 0，否则拒绝启动。

## 新文件

- `proobjaverse_official_slat_with_vggt_cache.py`：strict paired dataset；
- `prepare_proobjaverse_official_slat_with_vggt_sidecar.py`：可恢复、多 worker builder；
- `audit_proobjaverse_official_slat_with_vggt_cache.py`：只读 cache audit；
- `native_slat_genrecon_with_vggt_official.py`：独立 model/checkpoint identity；
- `train_native_slat_genrecon_with_vggt_official.py`：冻结科学参数和 no‑VGGT SS；
- `train_proobjaverse_official_slat_condition_lora_with_vggt.py`：official 入口；
- `background_jobs/run_proobjaverse_official_slat_with_vggt_cache.sh`：多 GPU builder；
- `test_proobjaverse_official_slat_with_vggt.py`：正负回归测试。

## 当前验证边界

源端真实 Train2000 read-only preflight 已通过，确认了 base manifest SHA、protocol、
Stable‑X snapshot、VGGT snapshot、DINO 权重和代码闭包。本 Codex 运行环境当前不可见
NVIDIA driver，因此还不能把 8-object materialization、真实 step‑0 或 1/2 GPU smoke
写成 CUDA PASS。当前状态只能是 `READY FOR CUDA CACHE/STEP0 RETEST`。

正式 F39 8 GPU launcher 在 8/64 cache、1GPU step‑0 和 2GPU DDP/resume 全部通过前
不解锁。strict-fix1 的 CPU-select/DDP(device_ids=None) 性能路径也需要针对新增
1374-token native context 单独验证，不能直接假定兼容。
