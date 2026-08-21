# Tracker A72 训练运行技术档案

生成日期：2026-08-15 UTC
源项目：/home/zjr/Tracker
目标入口：python -m pose_point_depth_mv.train_proobjaverse_official_slat_condition_lora

## 0. 结论、适用范围与证据强度

这三份档案足以从当前源码反推 A72 的 preflight、8000→8002 migration smoke 和 8002→10000 continuation 命令，也足以区分训练合同、受控 resume 迁移、性能参数和环境兼容项。真实 step8000 artifact 已在源服务器用当前正式 preflight 只读验证通过：

- max_steps：8000 → 10000，显式 extension；
- world_size / grad_accum：4×2 → 8×1；
- global effective batch：8 → 8；
- optimizer 与 EMA：继承；
- per-rank RNG stream：不宣称 exact；
- artifact、Native-SS deployment、data identity、stock freeze、official decoder audit：均通过。

有两个边界必须先说明：

1. 当前工作树很脏。HEAD 不是完整运行版本；A72 实际依赖的是“HEAD + 当前 working tree + runtime increment v2 + 既有 ReconViaGen/runtime/model cache”。不能只 checkout HEAD 后认为得到相同训练器。
2. 源服务器未找到真实 step_008002.pt 或 stage_report_step_008002.json。因此本文对 8002 checkpoint 的 schema 是由当前 save_checkpoint 源码推导的；A72 生成 8002 后，必须先对那个真实文件重新运行 preflight，再开始长训。

本次没有修改训练代码、checkpoint、manifest、report、freeze 或数据。只生成用户要求的三份审计输出。

配套文件：

- A72_RUNTIME_CONTRACT.json：机器可读的字段、路径、默认值、命令和源码位置。
- A72_RUNTIME_SOURCE_EXCERPTS.txt：完整 Git 状态快照、runtime-v2 文件表和带行号源码上下文。

## 1. 代码版本与运行闭包

### 1.1 四个必须分开的“版本”

| 层 | 真实含义 | 可否单独复现 |
|---|---|---|
| Git HEAD | 722296bc754509ce92149b89c908d7231bc603d8，branch=main | 否；训练关键文件有 dirty/untracked 内容 |
| 当前 working tree | 本次审计实际读取的源码 | 是本档案的源码真值；以文件 SHA256 绑定 |
| step8000 历史合同 | checkpoint.args、model_summary、data_identity 中冻结的事实 | 可验证合同；checkpoint 未嵌入当时 Git commit/源码 hash |
| A72 runtime increment v2 | 43 个文件的增量白名单 | 不是完整项目；要求 A72 已有 Tracker、ReconViaGen 和模型缓存 |

审计前、排除本次三个 A72_RUNTIME_* 输出后的 git status --short 共 590 行；完整逐行快照在 SOURCE_EXCERPTS 的第一个区块。训练相关的重点状态如下。

Tracked dirty：

- ar_ss_flow/local_pose_lifting_flow.py
- ar_ss_flow/shared_object_preprocessing.py
- pose_point_depth_mv/export_direct_flow_mesh_pairs.py
- pose_point_depth_mv/native_slat_genrecon.py
- pose_point_depth_mv/native_ss_genrecon.py
- pose_point_depth_mv/train_native_slat_genrecon.py
- reconvggt_ar_adapter_a/train_pointpose_ss_lora.py
- ReconViaGen gitlink 显示 modified；其内部 app.py 与 trellis/representations/mesh/flexicube.py 有修改。

与正式路径直接相关的 untracked：

- pose_point_depth_mv/audit_proobjaverse_official_slat_decoder.py
- pose_point_depth_mv/dino_only_condition.py
- pose_point_depth_mv/native_slat_genrecon_no_vggt.py
- pose_point_depth_mv/native_slat_genrecon_v2.py
- pose_point_depth_mv/native_ss_genrecon_no_vggt.py
- pose_point_depth_mv/no_vggt_ss_evidence.py
- pose_point_depth_mv/omni_real_benchmark_common.py
- pose_point_depth_mv/preflight_proobjaverse_official_slat_resume.py
- pose_point_depth_mv/proobjaverse_official_slat_protocol.py
- pose_point_depth_mv/proobjaverse_official_slat_training.py
- pose_point_depth_mv/test_native_slat_resume_data_identity.py
- pose_point_depth_mv/test_native_ss_deployment_relocation.py
- pose_point_depth_mv/train_native_slat_genrecon_no_vggt.py
- pose_point_depth_mv/train_proobjaverse_official_slat_condition_lora.py

ReconViaGen gitlink commit：

- f6720923260927333e22761fa722144d214733d9

### 1.2 runtime increment v2

- archive：/data/zjr/Tracker_A72_runtime_increment_v2.tar.zst
- SHA256：08eba092723e97c9d877c35d44be95147ca75877660a6c9aa3f40865450dbf05
- compressed bytes：217207
- apparent uncompressed size：约 1.2 MiB
- whitelist：/data/zjr/a72_runtime_sync_filelist_v2.txt
- 文件数：43

43 项完整 whitelist 同时记录在 CONTRACT.json 和 SOURCE_EXCERPTS。它补齐了 trellis_point_prior_mv/__init__.py 与 common.py，但刻意不包含其约 139 GB outputs。它也没有重复打包 A72 已存在的整个 ReconViaGen，因此“增量包完整”不等于“裸机器可独立运行”。

### 1.3 关键源码 SHA256

| 文件 | SHA256 |
|---|---|
| train_proobjaverse_official_slat_condition_lora.py | 6250f08133aebadbf24f7dc0a31944620e6aa0cd7ec8832204e0c71b265930e2 |
| train_native_slat_genrecon_no_vggt.py | 418df0d34b126290ef0ce7cf756afed8cc390eabfe7f8bc6bdcc8a7c3ebe5a4f |
| train_native_slat_genrecon.py | b1d36eaad08a08e9334da22bed9d119f435c3be320aee159f5c7eb67797c1cb5 |
| native_slat_genrecon_no_vggt.py | 32e12cb3bb6709313095f2c5ab7a3c510ee091476d4ebab3b561005ae59205bb |
| native_slat_genrecon_v2.py | 88a480a751adea51a2f023ea6a0f3772577400d222ceea3f71fe858f70774648 |
| direct_slat_data.py | 794461a80f30e0ef581474a718d6ce828ee83860b5ca95725f27305e3cc445d9 |
| no_vggt_ss_evidence.py | d3e50138d70b1532756a4dc0748413bd74cbfb1c89decddeb0e65108758e0a98 |
| native_3d_condition.py | 7e45b05e64516c0dc918c5cdb81bc148ac3e014d000688d227202fdff38121d9 |
| proobjaverse_official_slat_training.py | a87c97561981e8b34735d668d1d24973975617cdbd67ce51921f1ee909e71978 |
| preflight_proobjaverse_official_slat_resume.py | 5fb26d123ae0e4bb259305637f70e2064792d24860332c4f448d0092faf2445b |
| ar_ss_flow/local_pose_lifting_flow.py | 4d25dc862f5a46430625eae52c9d9da11e76308c61f7a378d4b563d880c1aff3 |
| train_direct_flow.py | fe8fb1bd9e4f0ac420e43a3a0cf4aaa7259250a123e455d23cb224e7ef9f56d5 |
| train_pointpose_ss_lora.py | d10bc27beb35536784c098f7f8385c70d56459c596cfdd0752ff11a5dc87a4d7 |
| ReconViaGen/trellis/pipelines/base.py | fb5534e7c757440d3e065ecb7ae5ccfae149d6a2bb4abcb71fcaf3aa9b5e2a70 |
| ReconViaGen/trellis/pipelines/trellis_image_to_3d.py | 07f7d2b1374dd3531052b32309b3dbd0530bc7a890177d8c668f3e99071fef78 |
| ReconViaGen/trellis/models/__init__.py | 76a1002a7c8fd6e5d84019e54e4559d2cbf1aeb0cdcc37a3c4928af87085befa |
| conv_spconv.py | 69174887a0b378851ced8fd35fe52f5a37c4c7ef2067daac80ad2980bdddb1b1 |
| full_attn.py | 18f6fe3168efff666ad7ef5321ff493280eafc95d69bc24a47938b9febbb8551 |
| trellis_point_prior_mv/common.py | a22c5a55a2258375e45082d33f6c6134a286f5a270a2e919fbcf49f2a34aac23 |

### 1.4 真实 artifact SHA256

| Artifact | SHA256 |
|---|---|
| slat_manifest.json | ead078ec423475ddbd2e4272e990404f74cddcc39b2332d42b4e2da266aab737 |
| lifting_manifest.json | 4a275be6d4b013a378d4aab325e48617cf1b5b7b77aeee30964558260be1d20b |
| Native-SS report | d2106fb85aecbfe7368ece963dbe976ddea3a9ecd558bf64f374d34066a6ff31 |
| Native-SS checkpoint | 7c036321e8d89c131a65f236ebe0df713c5bf7b95b84cd838b661be1fb09cd09 |
| stock freeze 文件 | 131f9612c736a57cf41ed2161ee102ce8bcab3b9f39fd1a2473ce7919b290bc6 |
| stock freeze 内部 freeze_sha256 | aeef32f5c17c1d039f93f1237b7c012f51d9854ad21a027a8c6645d09ceed91f |
| official decoder audit | fc1ab7b211c005eddb46d546139fe669f646b3fb009edf478943b7ef48330329 |
| step_008000.pt | 49edb3bbdbd86b10c5eea14e9c80a9996076b6fd65a459db12b130b6560bda4d |
| step8000 report.json | 35260349ff653107cdae14edb842d2cc8c029e99fa4185bb5bb043ade45ecf7b |

stock freeze 的“文件 SHA”与 JSON 中冻结的“模型状态 SHA”是两个不同概念，不能互换。

## 2. 正式训练入口完整调用链

### 2.1 wrapper 链

~~~text
python -m pose_point_depth_mv.train_proobjaverse_official_slat_condition_lora
  └─ official.main
       ├─ base.validate_decoder_audit = validate_official_decoder_audit
       └─ no_vggt_wrapper.main
            ├─ argv 缺 architecture 时插入 --architecture v2
            ├─ 非 v2 直接拒绝
            ├─ PoseLiftingCacheDataset(...)
            ├─ validate_dino_only_lifting_contract(...)
            ├─ patch base v2 version/checkpoint validator/component builder
            ├─ patch Native-SS eval format、loader、upstream projection
            └─ base.main
                 ├─ DDP init / device / rank seed
                 ├─ NativeConditionSLatDataset
                 │    ├─ DirectSLatCacheDataset
                 │    └─ PoseLiftingCacheDataset
                 ├─ Native-SS full deployment A↔B
                 ├─ official decoder audit
                 ├─ current data_identity
                 ├─ DataLoader + ObjectBalancedDistributedSampler
                 ├─ no-VGGT v2 component builder
                 │    ├─ 临时 TrellisVGGTTo3DPipeline := TrellisImageTo3DPipeline
                 │    ├─ Stable-X pipeline.from_pretrained
                 │    ├─ DINOv2 startup load
                 │    ├─ stock SLat flow freeze/hash
                 │    ├─ LoRA 注入
                 │    └─ 新 condition adapter
                 ├─ DDP(model)
                 ├─ AdamW / AMP scaler / EMA
                 ├─ resume checkpoint validation + load
                 ├─ training loop
                 └─ checkpoint + stage/final report
~~~

### 2.2 每层替换的 symbol

official wrapper 只替换 base.validate_decoder_audit，使 decoder report 绑定官方 ProObjaverse protocol、pretrained 和 cache target_source。

no-VGGT wrapper：

- 强制 architecture=v2；
- 在进入 base.main 前独立验证 lifting manifest 符合 raw_dino_only 合同；
- 将 NATIVE_SLAT_GENRECON_V2_VERSION 替换为 no-VGGT checkpoint format；
- 将普通 v2 checkpoint validator 替换为 no-VGGT validator；
- 将普通 v2 component builder 替换为 no-VGGT builder；
- 将 Native-SS 证据 format、loader 和 upstream_binding 改为 no-VGGT 版本。

no-VGGT builder 暂时把 trellis.pipelines.TrellisVGGTTo3DPipeline 指向 TrellisImageTo3DPipeline，再复用 v2 builder，最后恢复原 symbol。因此：

- 不执行 VGGT 模型；
- lifting contract 要求 vggt_feature_dim=0、vggt_model_executed=false；
- 训练仍消费缓存的 per-view raw DINO token context；
- v2 trainable parameter names/shapes 保持原设计；
- 它不是普通 v2 checkpoint 的任意兼容模式，而是单独 format/contract。

## 3. 正式入口全部 CLI 参数

“写 ckpt”表示 checkpoint_args 是否保存。override 和 run_until_step 被明确剔除；v2 下两个 v3-only 参数也剔除。

| 参数 | 类型；默认值 | 写 ckpt | 影响与 resume 规则 |
|---|---|---:|---|
| architecture | choice v2/v3；v3，official 强制 v2 | 是 | 模型/format；不得改 |
| cache_manifest | str；required | 是 | dataset/data identity；仅同一物理文件 relocation |
| lifting_cache_manifest | str；required | 是 | lifting/data identity；仅同一物理文件 relocation |
| target_decoder_audit | str；required | 是 | official target；仅 audit.path 同物理文件 relocation |
| native_ss_report | str；required | 是 | Native-SS deployment/identity；report 与其 checkpoint 仅同物理文件 relocation |
| stock_slat_freeze | str；required | 是 | stock 模型身份；不在 relocation whitelist |
| output_dir | str；required | 是 | 输出位置；当前 strict validator 不比较，可换到新持久目录 |
| pretrained | str；Stable-X/trellis-vggt-v0-2 | 是 | 模型、freeze、decoder；不得改 |
| indices | str；all | 是 | 样本集合/顺序/identity；不得改 |
| resume | str；空 | 是 | resume 来源；每段可指向新的 checkpoint |
| init_checkpoint | str；空 | 是 | 新 identity 初始化；与 resume 互斥 |
| init_weights | raw/ema；ema | 是 | init_checkpoint 权重选择；resume 不使用 |
| max_steps | int；2000 | 是 | 总训练 horizon；只允许显式严格增大 |
| run_until_step | int；0 | 否 | 本次进程停止边界；0 等于 max_steps |
| save_every | int；200 | 是 | I/O 频率；当前 validator 允许改，属运行策略 |
| log_every | int；10 | 是 | 日志频率；当前 validator 允许改 |
| grad_accum | int；4 | 是 | effective batch/DDP sync；拓扑变化时必须保持 world_size×grad_accum |
| num_workers | int；0 | 是 | DataLoader 性能；当前 validator 允许改，需 smoke |
| seed | int；42 | 是 | sampler/noise/t/view/uncond；代码未显式 strict compare，但语义上不得改 |
| lora_rank | int；8 | 是 | 架构/optimizer shape；strict |
| lora_alpha | int；16 | 是 | 模型语义；strict |
| condition_channels | int；1024 | 是 | 架构；strict，且当前要求 1024 |
| view_fusion_hidden_dim | int；64 | v2 否 | v3-only；official v2 不适用 |
| geometry_logit_scale_init | float；1.0 | v2 否 | v3-only；official v2 不适用 |
| new_lr | float；1e-4 | 是 | condition 参数组 LR；strict |
| lora_lr | float；3e-5 | 是 | LoRA 参数组 LR；strict |
| new_weight_decay | float；0.01 | 是 | optimizer；strict |
| adam_beta1 | float；0.9 | 是 | optimizer；未显式 compare，但 optimizer state 会恢复，禁止随意改 |
| adam_beta2 | float；0.95 | 是 | optimizer；同上 |
| grad_clip | float；1.0 | 是 | 优化语义；未显式 compare，禁止改 |
| warmup_steps | int；-1 | 是 | LR schedule；strict |
| warmup_ratio | float；0.02 | 是 | LR schedule；strict |
| ema_decay | float；0.9995 | 是 | EMA；strict |
| amp_dtype | bf16/fp16/none；bf16 | 是 | 数值路径/scaler/性能；未显式 compare，canonical resume 保持 |
| amp_init_scale | float；8192 | 是 | fp16 scaler 初值；canonical resume 保持 |
| p_uncond | float；0.1 | 是 | 训练分布/随机性；strict |
| t_logit_mean | float；1.0 | 是 | t 分布；strict |
| t_logit_std | float；1.0 | 是 | t 分布；strict |
| t_schedule | logit_normal/uniform；默认 logit_normal，step8000=uniform | 是 | 训练分布；strict |
| min_condition_views | int；1 | 是 | 训练分布/计算；strict |
| max_condition_views | int；16，step8000=8 | 是 | 训练分布/计算；strict |
| stock_context_views | all/first；all | 是 | stock cross-attention 语义；strict |
| gradient_checkpointing | flag；false，step8000=true | 是 | 内存/执行次序；canonical resume 保持 |
| verify_cache_hashes | flag；false | 是 | 启动完整性开销；不改训练语义 |
| allow_resume_max_steps_extension | flag；false | 否 | 仅配合 resume；批准严格延长 horizon |
| allow_resume_topology_change | flag；false | 否 | 仅配合 resume；批准同 effective batch 的 topology 改变 |
| allow_resume_data_path_relocation | flag；false | 否 | 仅配合 resume；只批准白名单路径指向同一现存物理对象 |

step8000 的有效关键值是：v2、grad_accum=2、seed=42、LoRA 8/16、condition_channels=1024、new_lr=1e-4、lora_lr=3e-5、weight_decay=0.01、betas=(0.9,0.95)、clip=1、EMA=.9995、bf16、p_uncond=.1、t_schedule=uniform、views=1..8、stock_context_views=all、gradient_checkpointing=true。

### run_until_step 与 max_steps

- max_steps 是训练合同和 LR schedule horizon；checkpoint 保存它。
- run_until_step 只是本进程的 stage boundary；不写 checkpoint args。
- 0 被归一化为 max_steps。
- 8000→8002 smoke 应保持 max_steps=10000，只令 run_until_step=8002。
- 如果误把 max_steps=8002，会把“迁移 smoke”变成另一条训练 horizon 合同，warmup ratio 派生值也可能改变。

## 4. checkpoint schema 与 resume 顺序

### 4.1 真实 step8000

文件 581559030 bytes，format=pose_point_depth_mv.native_slat_genrecon_no_vggt.v1。

| 顶层 key | 类型 | 用途 |
|---|---|---|
| format | str | checkpoint 合同版本 |
| step | int | 已完成 optimizer step；8000 |
| micro_step | int | 已消费的 per-rank micro-step 计数；16000 |
| model_trainable_state | dict[str,Tensor] | 仅 trainable LoRA/new-condition 状态；298 tensors |
| ema_trainable_state | dict[str,Tensor] | 同一 trainable key 集的 EMA；298 tensors |
| ema | dict | target_decay、updates、last_decay |
| optimizer | dict | AdamW state/param_groups；298 states、2 groups |
| scaler | dict | AMP scaler；bf16 下为空 dict |
| args | dict | checkpoint_args 过滤后的训练参数 |
| model_summary | dict | 架构、trainable/frozen hashes、upstream binding、resume transitions |
| data_identity | dict | cache/lifting/Native-SS/freeze/decoder 的冻结身份 |
| history | list[dict] | 每个 optimizer step 的指标；8000 项 |
| rng | dict | python、numpy、torch CPU、当前 CUDA device RNG |

optimizer 两组：

- lora：240 参数，lr=3e-5，weight_decay=0；
- new_condition：58 参数，lr=1e-4，weight_decay=0.01。

模型摘要：24 blocks、120 LoRA modules、2,555,904 LoRA 参数、33,584,129 new-condition 参数，总 trainable 36,140,033。EMA updates=8000，target_decay=.9995，last_decay=.998876404494382。

### 4.2 构造和恢复的精确顺序

checkpoint 并不是程序启动时首先 load。当前顺序是：

1. 从环境读取 rank/world_size 并 init NCCL；
2. set CUDA device，设置 rank seed；
3. 构造 dataset，验证 lifting、A↔B deployment、decoder audit，构造 current data_identity；
4. 构造 sampler/DataLoader；
5. 加载 Stable-X/DINO，构造模型，freeze/hash，做 dataset[0] initial stock-equivalence audit；
6. DDP 包装；
7. 新建 optimizer、scaler、EMA；
8. torch.load resume checkpoint；
9. 验证 checkpoint format/model_summary/data_identity/training transition/strict args；
10. load trainable model state；
11. optimizer.load_state_dict；
12. scaler.load_state_dict；
13. load EMA state；
14. 恢复 step、micro_step、history；
15. topology 未变化时 restore_rng_state；变化时执行 rank-specific boundary reseed；
16. 进入 loop。

因此失败在 Stable-X、DINO、dataset 或 initial audit 时，甚至还没有读 checkpoint。initial audit 检查的是 freshly built zero-init adapter 对 stock 的等价性，不是在检查 resume 后的 step8000 模型。

save_checkpoint 仅 rank0 执行，先写临时文件再 os.replace；step_xxxxxx.pt 与 last.pt 使用相同 payload。

### 4.3 RNG 恢复的实现边界

4→8 时实现明确不做 exact RNG 恢复，而按 seed + rank×100003 + step×1000003 在迁移边界重新播种；因此 optimizer/EMA 可继承，但 continuation 不可能 bitwise identical。

还发现一个更细的审计边界：只有 rank0 保存一份 Python/NumPy/CPU-Torch/当前 CUDA RNG。相同 topology resume 时，每个 rank 都会恢复这同一份 rank0 bundle；同时 trainer 没有 checkpoint 当前 sampler epoch/epoch 内偏移，epoch 从 0 重新开始。所以 validator 在“topology 不变”时给出的 per_rank_rng_stream_exact=true 比当前实现能证明的更强。对本次 4→8 没有误导，因为输出本来就是 false；但不能把将来的 8→8 resume 宣称为全 rank、全 sample-stream bitwise exact。

### 4.4 step8002

当前源码会生成同一 13-key schema，step=8002；但源端没有真实文件可核验。A72 smoke 后必须确认：

- step_008002.pt 和 last.pt 存在且可 load；
- step=8002、format 正确；
- args.max_steps=10000、args.grad_accum=1；
- model_summary 的 saved/current topology transition 记录合理；
- stage_report_step_008002.json 中 stage_complete=true；
- 再以 step8002 作为 continuation 的 resume 输入。

## 5. 三层 identity / deployment 合同

定义：

- A = dataset.config["native_ss_deployment"]
- B = load_no_vggt_ss_evidence(native_ss_report) 返回的完整 binding
- C = no_vggt_upstream_binding(B)
- D = checkpoint["data_identity"]["native_ss"]

### 5.1 A/B：完整 deployment

A/B 完整字段：

- report、report_sha256
- checkpoint、checkpoint_sha256、checkpoint_step
- weights
- cfg_strength、steps、cfg_interval、guidance_rescale、rescale_t
- amp_dtype
- false_checks

validate_native_ss_deployment(frozen_deployment, runtime_deployment, *, allow_path_relocation) 的规则：

1. 原 dict 完全相等，直接通过；
2. 不相等且未允许 relocation，拒绝；
3. 唯一白名单是 report 和 checkpoint；
4. 每个路径必须 Path(value).expanduser().resolve(strict=True)，两边结果完全相等；
5. 只有验证为同一现存 filesystem object 后，copy 中才可换成共同 marker；
6. copy 的剩余 dict 必须深度完全相等。新增、删除、unknown field 都会拒绝。

因此 amp_dtype、false_checks、hash、step、weights、所有 CFG 字段永远不会因路径迁移被忽略。

full-deployment relocation 只在 args.resume 且 allow_resume_data_path_relocation 时启用；fresh training 仍要求 A==B 原始严格相等。

### 5.2 C/D：训练 identity 投影

C/D 是历史上明确投影后的字段：

- report、report_sha256
- checkpoint、checkpoint_sha256、checkpoint_step
- weights
- cfg_strength、steps、cfg_interval、guidance_rescale、rescale_t

amp_dtype 和 false_checks 不在 C/D，但仍在更早的 A/B 完整验证中严格检查；这不是放宽。

validate_resume_data_identity(saved_identity, current_identity, *, allow_path_relocation) 的唯一路径白名单：

- cache_manifest
- lifting_cache_manifest
- native_ss.report
- native_ss.checkpoint
- target_decoder_audit.path

也是 resolve(strict=True) 同对象后用 marker，再严格比较所有剩余字段。cache_manifest_sha256、lifting SHA、config_hash、sample/object count、object_uids/hash、Native-SS semantic fields、freeze SHA、decoder audit hash/summary/thresholds 均保持严格。

checkpoint.model_summary.upstream_native_ss 还要与 checkpoint.data_identity.native_ss 内部一致；不能只让 live identity 通过而忽略 checkpoint 自洽性。

### 5.3 training transition

validate_resume_training_contract(checkpoint, args, world_size=...)：

- max_steps 变化：必须是严格增大、checkpoint.step < current max_steps，并给 extension flag；
- topology 变化：必须给 topology flag，且 saved world_size×saved grad_accum 等于 current；
- 当前目标：4×2=8，8×1=8；
- optimizer_inherited=true、ema_inherited=true；
- topology_changed=true 时 per_rank_rng_stream_exact=false。

路径 relocation 不是训练语义迁移，不能被用来批准任何 hash、CFG、dataset 或模型变化。

## 6. 从 manifest 到一个 sample 的数据加载

### 6.1 数据集与 join

实际 dataset 是 NativeConditionSLatDataset。它构造：

- DirectSLatCacheDataset(slat_manifest)
- PoseLiftingCacheDataset(lifting_manifest)

并按 UID 做严格一对一 join；长度 2000、object_count 2000。ObjectBalancedDistributedSampler 每 epoch 每 object 取一条 row。

slat manifest row 关键字段：

- uid、object_uid、support_seed
- target_file / SHA
- support_file / SHA
- physical_file / SHA
- condition_file / SHA
- source_lh_slat / SHA
- source_glb / SHA
- ss_latent / SHA

_resolve 将相对路径按 manifest/cache 根解析；A72 的 /data/zjr 逻辑软链接可能最终 resolve 到 /dev/shm 或 /root/autodl-fs。

### 6.2 __getitem__ 的真实 I/O

DirectSLatCacheDataset.__getitem__：

1. np.load(target_file)；
2. torch.load(support_file, map_location=cpu)；
3. torch.load(physical_file, map_location=cpu)；
4. torch.load(condition_file, map_location=cpu)；
5. 对 shape/dtype/finite/UID/hash/metadata 做 CPU 验证；
6. 返回 target、support、physical、condition 与元数据。

PoseLiftingCacheDataset.__getitem__：

1. torch.load(lifting payload, map_location=cpu)；
2. 大张量转 float32 做 isfinite、shape、相机与预处理合同检查；
3. 验证 K_feature=A@K_source、geometry metadata/hash；
4. np.load placeholder SS latent；
5. 返回 lifting tensors 与 metadata。

NativeConditionSLatDataset.__getitem__ 同时调用二者，并再次检查 UID/object/condition 对齐；collate_native_one 要求 batch size 恰为 1，直接返回单样本。

### 6.3 第一条真实样本

| 文件 | 大小/格式 | 主要内容 |
|---|---|---|
| target .npz | 388694 B，ZIP DEFLATE | coords [12267,3] uint8→int32；feats [12267,8] fp32 |
| support .pt | 1328944 B，PyTorch ZIP stored | corrected_ss [1,8,16,16,16] fp32；occupancy [1,1,64,64,64] fp32；coords [12267,3] int32 |
| physical .pt | 1140 B，ZIP stored | native placeholder |
| condition .pt | 44864862 B，ZIP stored | cond 与 neg_cond 各 8×[1,1369,1024] fp16 |
| lifting .pt | 43704818 B，ZIP stored | visual [8,1369,1024] fp16；depth/conf/mask 各 [8,518,518] fp16；K/extrinsics；prior；stock condition |
| placeholder SS .npz | 779 B | z [8,16,16,16] fp32；target coords [1,3] |

这些 .pt 的大 tensor block 实际是 stored，不是压缩流；target NPZ 才是 DEFLATE。

主 SLat step 真正数值消费 target coords/features、cond/neg_cond、lifting visual features、K/extrinsics、depth shape 与几何元数据。support volume、physical placeholder、depth 数值/conf/mask、prior、stock_condition、SS placeholder 等仍被 load、验证并被 pin_memory 遍历，却不进入当前主 forward 的数值计算。这是性能事实，不授权删字段；删之前要改并重审数据合同。

### 6.4 DataLoader 当前构造

| 项 | 当前值 |
|---|---|
| batch_size | 1 / rank，硬编码 |
| sampler | ObjectBalancedDistributedSampler |
| shuffle | 未传；有 sampler，等价 false |
| num_workers | CLI，默认 0；step8000 也是 0 |
| pin_memory | true |
| drop_last | 未传，PyTorch 默认 false |
| persistent_workers | 未传，默认 false |
| prefetch_factor | 未传；仅 workers>0 时 PyTorch 默认生效 |
| worker_init_fn | 无 |
| generator | 无 |
| collate_fn | collate_native_one |

num_workers=0 意味着全部 np.load、torch.load、解压、Python loop、float/isfinite、相机/哈希检查都在每个 rank 的主训练进程里同步完成，GPU 在取下一 batch 时直接等待。

## 7. 一个 optimizer step 的 CPU/GPU 执行图

| 阶段 | 位置 | 每 rank / rank0 | 同步与阻塞 |
|---|---|---|---|
| sampler 选 row | CPU | 每 rank | epoch permutation 本地生成 |
| dataset __getitem__ | CPU/I/O | 每 rank | num_workers=0，阻塞 rank 主进程 |
| np/torch load + validation | CPU/I/O | 每 rank | 大量 memcpy、float、finite、allclose、Python loop |
| pin-memory 处理 | CPU | 每 rank | 全 sample，包含未使用大 tensor |
| host→device | DMA + GPU alloc | 每 rank | non_blocking，但随后使用会形成依赖 |
| condition/view 随机选择 | CPU+GPU | 每 rank | Python/torch RNG；部分 scalar 取值会同步 |
| DINO/condition | GPU | 每 rank | DINO 网络本身不跑；处理缓存 DINO tokens |
| sparse tensor/坐标 | CPU+GPU | 每 rank | 坐标转换、索引、spconv rulebook 可耗 CPU |
| model forward | GPU | 每 rank | v2 condition adapter + frozen stock flow/LoRA |
| flow/stock loss | GPU | 每 rank | autocast bf16 |
| backward | GPU | 每 rank | accumulation 非末 micro-step 用 DDP no_sync |
| DDP all-reduce | GPU/NCCL | 每 rank | optimizer boundary 同步 |
| finite/grad norms | GPU→CPU | 每 rank | 每 tensor .item，强同步热点 |
| clip + optimizer | GPU | 每 rank | AdamW |
| parameter/optimizer finite | GPU→CPU | 每 rank | 再次逐 tensor .item，强同步热点 |
| EMA | GPU/CPU book-keeping | 每 rank | 所有 trainable tensor update |
| metric all-reduce | GPU/NCCL | 每 rank | 每 optimizer step |
| JSON log | CPU stdout | rank0 | log_every |
| checkpoint/report | CPU+storage | rank0 | save boundary，可能长暂停 |

## 8. CPU thread / DataLoader 性能诊断

项目没有设置：

- torch.set_num_threads
- torch.set_num_interop_threads
- OMP_NUM_THREADS
- MKL_NUM_THREADS
- OPENBLAS_NUM_THREADS
- NUMEXPR_NUM_THREADS

源环境观察值是 torch intraop=32、interop=32、OpenMP/MKL max=32。8 ranks 若每 rank 可展开约 32 线程，仅 PyTorch/OpenMP 就可能形成约 256 个 runnable threads，与 A72 观察到的 runnable 250~500、user CPU≈95%、每 rank≈2000% CPU 高度一致。

### 8.1 高置信性能瓶颈

1. 每 rank/sample 约 44.86 MB condition + 43.70 MB lifting，再加 support/target；workers=0。
2. cond/neg_cond/visual/depth 等反复 float() 和 isfinite().all().item() 全量扫描。
3. 全部正负视图和 visual features 在选择最终 views 前即 H2D，约 67 MB/rank/sample 的主要 context 搬运。
4. pin_memory 递归处理完整 sample，包括当前 forward 未用的大字段。
5. 每 optimizer step 的 gradients_finite、gradient_norms、parameters_finite、optimizer_state_finite 对数百 tensor 分别 .item()。保守估计每 rank 每次更新超过 1000 个 GPU→CPU 同步点，能直接产生“GPU 突发 75~100%，随后 SM=0”的锯齿。
6. sparse coordinate/rulebook 构造、排序/unique/indexing 以及 spconv CPU preprocessing 也可能贡献 CPU 峰值。

可能并行的库：PyTorch CPU、OpenMP、Intel MKL、OpenBLAS/NumPy、Numba，以及 spconv 的 CPU preprocessing。源端没有发现 TBB Python distribution 或系统动态库记录。

### 8.2 只读审计后的 benchmark 矩阵

先不改算法代码，按单变量测试：

| 轴 | 候选 |
|---|---|
| OMP_NUM_THREADS / MKL_NUM_THREADS | 1、2、4、8 |
| OPENBLAS_NUM_THREADS / NUMEXPR_NUM_THREADS | 建议先固定 1，再单测 |
| num_workers | 0、1、2、4 |
| persistent_workers | false/true；仅 workers>0，当前需小代码改动 |
| prefetch_factor | 1、2、4；仅 workers>0，当前需小代码改动 |

每点至少跨过 startup 后测 5~20 optimizer steps，记录 step wall time、samples/s、CPU run queue、磁盘读取、pinned memory、H2D、GPU utilization/SM occupancy。先测试线程帽，再测试 workers，避免 8 ranks×workers×BLAS threads 二次过度订阅。

分类：

- OMP/MKL/OpenBLAS/NumExpr、CPU affinity：纯运行性能参数，不进入 checkpoint/data identity。
- num_workers：当前 dataset 的 row 内容是确定性的，采样顺序由 sampler 主进程决定，通常是性能项；但 worker 进程 RNG/调度和 CPU RNG checkpoint 行为会变化，必须 smoke。
- persistent_workers/prefetch_factor：性能/实现参数；当前无 CLI，若添加就是代码变更，但不应改变训练数学合同。仍需正负回归与 smoke。
- 删除数据字段、跳过验证、改变 view 选择/H2D 时机：可能改变数据/训练实现，不能当作纯环境调优直接上线。

## 9. 模型与网络依赖

### 9.1 Stable-X/trellis-vggt-v0-2

- loader：no-VGGT 临时类替换后的 TrellisImageTo3DPipeline.from_pretrained。
- HF snapshot revision：647659a5ad5fbf67e22793e7b5e2cee4b30c5d13。
- A72 cache：/root/autodl-fs/huggingface/hub/models--Stable-X--trellis-vggt-v0-2。
- pipeline base 会按 pipeline.json 加载全部配置模型，而非只加载本训练最终保留的模块。
- snapshot 的 pipeline.json 实际列出：sparse_structure_decoder=ckpts/ss_dec_conv3d_16l8_fp16、sparse_structure_flow_model=ckpts/ss_flow_img_dit_L_16l8_fp16、slat_decoder_gs=ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16、slat_decoder_mesh=ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16、slat_flow_model=ckpts/slat_flow_img_dit_L_64l8p2_fp16、sparse_structure_vggt_cond=ckpts/ss_vggt_cond、slat_vggt_cond=ckpts/slat_vggt_cond。
- 最终保留：slat_flow_model、slat_sampler、slat_normalization。
- slat_decoder_mesh 会被加载、做 state hash 验证，然后丢弃；不进入训练 GPU loop。
- 其他 pipeline-configured models 也可能在 startup 被加载后释放，因此模型缓存必须完整。

### 9.2 DINOv2

- 名称：dinov2_vitl14_reg。
- loader：先 torch.hub.load(local_repo, source="local", pretrained=True)。
- A72 repo：/root/.cache/torch/hub/facebookresearch_dinov2_main。
- A72 weight：/root/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth。
- 本训练 loop 不执行 DINO 编码；使用 cache 中的 DINO token。
- 但 startup 仍实例化/加载 DINO。
- local load 被 broad except 包围；任何本地异常都会触发 torch.hub 的 GitHub fallback。HF_HUB_OFFLINE 不会阻止这个显式 fallback，所以离线前必须单独验证本地 repo 和 weight。

### 9.3 Native-SS、stock SLat、decoder

- Native-SS report/checkpoint 用于证据和 identity；SLat trainer 不把 Native-SS checkpoint weights 加载进模型，但会 hash 整个 checkpoint 文件。
- stock SLat flow 来自 Stable-X，完全冻结，并以 stock freeze 记录/runtime state hash 验证。
- mesh decoder 用于 identity/decoder trust 验证，不在训练 loop 执行。

建议离线变量：

~~~bash
export HF_HOME=/root/autodl-fs/huggingface
export HF_HUB_CACHE=/root/autodl-fs/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/root/.cache/torch
~~~

## 10. native extension 与环境依赖

| 组件 | 源服务器 | A72/要求 |
|---|---|---|
| Python | 3.10.18 | A72 镜像实际版本需记录 |
| PyTorch | 2.4.0 + cu121 | 已知 2.10 + cu130 |
| CUDA toolkit | 12.1.66 | 13.0 |
| driver | 595.71.05 | A72 当前 driver 与 CUDA13 匹配 |
| GPU arch | 3090/Ampere | RTX PRO 6000 Blackwell sm_120 |
| torchvision | 0.19.0 | 必须与 torch ABI 匹配 |
| xformers | 0.0.27.post2 | 必须是 torch/CUDA/sm_120 兼容 build |
| flash-attn | 2.7.0.post2 | 2.8.3，sm_120 build |
| spconv | spconv-cu120 2.3.6 | CUDA13/sm_120 可运行 build |
| cumm | cumm-cu120 0.4.11 | 与 spconv 配套 |
| nvdiffrast | 0.3.3 | 见下方 A72 patch |
| NumPy | 2.0.1 | cache/object 反序列化需兼容 |
| Numba | 0.65.0 | cache dir 可写 |
| TBB | 未发现 | 不应假定存在 |
| huggingface_hub | 0.36.2 | 可离线命中 snapshot |
| transformers | 4.57.3 | 可离线 |
| peft | 0.18.1 | LoRA runtime |
| safetensors | 0.7.0 | Stable-X weights |
| BLAS | Intel oneAPI MKL 2023.1 | 控制线程防过订阅 |

已知 A72 兼容事项：

1. nvdiffrast 0.3.3：在 A72/PyTorch 2.10 上，cpp_extension.load 返回 module，应直接使用返回值，不依赖随后 importlib.import_module。属于环境/ABI 兼容，不是训练语义。
2. PyTorch ≥2.6：torch.load 默认 weights_only=True。历史可信 checkpoint/cache 含普通 Python dict/NumPy 状态；正式 trainer 多数 load site 未显式传 weights_only。A72 必须设置 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1，或另做经审计的显式 weights_only=False patch。不要对不可信文件这样做。
3. FlashAttention：A72 使用 2.8.3 sm_120 build；这会改变 backend/build 的数值实现细节，但不改变 checkpoint schema。必须先 smoke。
4. 环境模块导入会 setdefault：SPCONV_ALGO=native、MPLCONFIGDIR=/tmp/matplotlib、NUMBA_CACHE_DIR=/tmp/numba_cache、TORCH_EXTENSIONS_DIR=/tmp/torch_extensions。

## 11. runtime-v2 闭包中的 torch.load / torch.save

### 11.1 torch.load 静态清单

weights_only “未传”表示会受到 PyTorch 2.6 默认变化影响。

| 文件:行 | 用途 | weights_only | 正式训练主路径 |
|---|---|---|---|
| ar_ss_flow/correspondence_lifting.py:1199 | correspondence checkpoint | 未传 | 否 |
| ar_ss_flow/local_pose_lifting_flow.py:104 | lifting cache dict | 未传 | 是 |
| pose_point_depth_mv/direct_flow.py:219 | frozen correspondence head | 未传 | 否 |
| direct_slat_data.py:142 | support cache | 未传 | 是 |
| direct_slat_data.py:143 | physical cache | 未传 | 是 |
| direct_slat_data.py:144 | condition cache | 未传 | 是 |
| eval_direct_flow.py:665 | eval checkpoint | 未传 | 否 |
| evaluate_native_ss_genrecon.py:376 | Native-SS checkpoint | 未传 | 否 |
| evaluate_native_ss_stock_slat_mesh.py:599 | Native-SS checkpoint | 未传 | 否 |
| evaluate_native_ss_stock_slat_mesh.py:799 | condition cache | 未传 | 否 |
| evaluate_native_ss_stock_slat_mesh.py:943 | SLat payload | 未传 | 否 |
| export_direct_flow_mesh_pairs.py:244 | sparse payload | 未传 | 否 |
| export_direct_flow_mesh_pairs.py:305 | condition artifact | 未传 | 否 |
| export_direct_flow_mesh_pairs.py:1262 | checkpoint | 未传 | 否 |
| export_direct_flow_mesh_pairs.py:1518 | condition cache | 未传 | 否 |
| export_direct_flow_mesh_pairs.py:1715 | SLat payload | 未传 | 否 |
| export_direct_flow_mesh_pairs.py:1721 | repeat SLat payload | 未传 | 否 |
| preflight...py:60 | resume checkpoint | false,mmap=true | preflight |
| preflight...py:64 | old-Torch fallback | 未传 | preflight |
| prepare_direct_flow_mesh_protocol.py:118 | cache sample | 未传 | 否 |
| train_direct_flow.py:850 | Direct-Flow checkpoint | 未传 | 否 |
| train_direct_slat_flow.py:1332 | Direct-SLat checkpoint | 未传 | 否 |
| train_native_slat_genrecon.py:1086 | resume checkpoint | 未传 | 是 |
| train_native_slat_genrecon.py:1200 | init checkpoint | 未传 | 条件路径 |
| train_pointpose_ss_lora.py:786 | PointPose checkpoint | 未传 | 否 |

这些对象都是项目自产、经 SHA/合同约束的可信文件；TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 只应在这个信任边界内使用。

### 11.2 torch.save 静态清单

- ar_ss_flow/correspondence_lifting.py:1190
- evaluate_native_ss_stock_slat_mesh.py:109
- export_direct_flow_mesh_pairs.py:152
- omni_real_benchmark_common.py:52
- train_direct_flow.py:560
- train_direct_slat_flow.py:669
- train_native_slat_genrecon.py:797（正式路径，atomic）
- reconvggt_ar_adapter_a/inspect_and_sanity.py:268
- reconvggt_ar_adapter_a/train_pointpose_ss_lora.py:674

## 12. 输出、checkpoint 与 report

output_dir 下：

~~~text
checkpoints/step_00xxxx.pt
checkpoints/last.pt
stage_report_step_00xxxx.json   # 本次边界小于 max_steps
report.json                     # 达到 max_steps
~~~

save_every 控制定期 checkpoint；无论是否整除 save_every，达到 run_until_step 时也会保存边界 checkpoint。log_every 只控制 stdout JSON，不控制 history（history 每 optimizer step 追加）。

精确定义：

- completed：step == max_steps。
- stage_complete：到达 run_until_step，且所有 rank 的 trainable parameters、optimizer state、EMA 均 finite。
- passed：stage_complete and completed。

因此：

- max_steps=10000、run_until_step=8002：completed=false、stage_complete=true、passed=false，正常 exit code 0；不是失败。
- max_steps=10000、run_until_step=10000：三者均 true。

如果 8002 stage report 缺失，不应仅凭训练进程“结束”认定 smoke 成功；同时检查 checkpoint、last.pt、stage report 和 preflight。

## 13. 日志格式与健康判断

前缀是 [native_slat_genrecon]，后接一行 JSON。

| 字段 | 含义 | 健康观察 |
|---|---|---|
| step | optimizer step | 单调 +1 |
| micro_step | per-rank micro-step 累计 | 旧 4卡acc2 每 step +2；新 8卡acc1 每 step +1 |
| global_micro_samples | 累计全局样本 | 两种配置每 optimizer step 都 +8 |
| flow_loss | 当前 trainable flow objective | 必须 finite；看趋势/分布，不用单点判定 |
| stock_loss | frozen stock 对同目标的 loss | finite；作为同 batch 基线 |
| gain | stock_loss 相对 flow_loss 的差 | 长期正值通常表示优于 stock；看滑动均值 |
| t_mean | 本 step 采样时间均值 | uniform 下不应长期卡在端点 |
| unconditional_fraction | classifier-free dropout 比例 | 长期围绕 p_uncond=.1；小窗口会抖动 |
| view_count_mean | 条件视图数均值 | 应落在 1..8 |
| flow_delta_rms | trainable 输出相对 stock 的变化量 | 非零且稳定；突增需检查 |
| supported_fraction | target/support 有效比例 | 突然塌到 0 或分布改变应停 |
| active_point_count_mean | 稀疏 active points | 应与数据基线同量级；极端跳变查 cache |
| fusion_gate_mean | v3 fusion gate | official v2 固定日志 0，预期 |
| effective_view_weight_deviation | v3-only view fusion | v2 为 0，预期 |
| target_view_weight_entropy | v3-only | v2 为 0，预期 |
| gradient_norm_before_clip | clip 前总 norm | 必须 finite；持续异常尖峰/为0需排查 |
| gradient_norms | 分模块 norm | LoRA/new condition 应有合理非零；view_fusion 在 v2 为0 |
| learning_rate_scale | warmup/调度缩放 | resume 后应与 max_steps=10000 合同一致 |
| learning_rates | 两 optimizer group 实际 LR | 对照 lora/new_condition |
| ema_decay | 当前动态 EMA decay | finite，逐步趋近 target .9995 |

源码没有编码通用的“正常数值阈值”，因此不要发明绝对阈值；应与 step7990~8000 的同字段、8001/8002 smoke 和多 step 滑动统计比较。立即停止条件包括 non-finite、合同 validator 失败、loss/grad 连续爆炸、supported_fraction/active points 结构性塌陷、某应训练模块梯度持续精确为 0。

fusion_gate_mean=0、effective_view_weight_deviation=0、target_view_weight_entropy=0、gradient_norms.view_fusion=0 在 official v2 是 logger 对不存在 v3 view_fusion 模块的默认值，不是故障。

## 14. 随机性与 reproducibility

- 主 seed=42。
- 启动时每 rank 使用 rank 派生 seed。
- sampler 使用独立 torch.Generator(seed+epoch)，先对排序后的 object UID 做 permutation，再为每 object 选 support row，pad 后按 rank stride 切分。
- p_uncond、t sampling、condition view count/subset、flow noise 使用运行进程 RNG。
- DataLoader 未传 generator/worker_init_fn。
- checkpoint 保存 Python、NumPy、torch CPU、当前 CUDA RNG，但只由 rank0 写一份。
- 4→8 拓扑改变时显式 boundary reseed；sample order、每 rank stochastic stream、reduction order 都改变。
- effective batch 保持不代表 bitwise identical；optimizer/EMA state inheritance 才是被保证的部分。
- num_workers 从 0 改为 >0 理论上不改变当前确定性 __getitem__ 的样本内容或 sampler 主序列，但会改变 worker process/RNG/调度；必须用短 smoke 验证，不能承诺 bitwise equality。

## 15. batch 的真实语义

batch_size 在 DataLoader 硬编码为 1/rank，没有 CLI batch_size。

旧训练：

- 4 GPU；
- 每 GPU micro batch=1；
- global micro batch=4；
- grad_accum=2；
- 每 optimizer step=8 samples。

A72：

- 8 GPU；
- 每 GPU micro batch=1；
- global micro batch=8；
- grad_accum=1；
- 每 optimizer step=8 samples。

所以 world_size×batch_per_rank×grad_accum 都为 8。日志 global_micro_samples 每 optimizer step 增加 8。DDP reduction 拓扑和每 rank 随机流仍变了，所以只叫“合法的同 effective batch continuation”，不叫 bitwise-identical continuation。

## 16. DDP / NCCL

- LOCAL_RANK、RANK、WORLD_SIZE 从环境读取，缺省均为单进程值。
- world_size>1 时调用 torch.distributed.init_process_group(backend="nccl", timeout=12h)。
- 当前代码在 init_process_group 之后才 torch.cuda.set_device(local_rank)，且 init 没传 device_id。
- DDP 参数：device_ids=[local_rank]、output_device=local_rank、broadcast_buffers=false、find_unused_parameters=false。
- accumulation 的非边界 micro-step 使用 model.no_sync；最后一个 micro-step才同步 gradients。

警告：

~~~text
ProcessGroupNCCL: Guessing device ID based on global rank
~~~

根因是 process group/barrier 早于显式 device 绑定，init 也没给 device_id。单机同质 8 卡且 global rank==local rank 时通常只是 warning，不改变当前合同；但值得以后做 DDP runtime compatibility 修正并 smoke。若多机、非连续 CUDA_VISIBLE_DEVICES 或 rank 映射复杂，风险会提高。

## 17. A72 canonical 命令

以下参数由当前 parser、真实 step8000 args 和 validator 反推，不是抄历史文档。先确认 /data/zjr 逻辑路径软链接全部存在、OUT 实际落在持久盘。

### 17.1 公共变量与环境

~~~bash
cd /root/Tracker

export RUN=/data/zjr/proobjaverse_official_slat_train2000_20260813_v1
export CACHE=$RUN/cache_train2000_protocol2128_views8_v1
export SSROOT=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1
export FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json
export STEP8000=$RUN/B_condition_lora_train2000_step8000_seed42_4gpu_v1/checkpoints/step_008000.pt
export OUT=$RUN/B_condition_lora_train2000_step10000_seed42_8gpu_v1

export PYTHONPATH=/root/Tracker:/root/Tracker/ReconViaGen:/root/Tracker/ReconViaGen/wheels/vggt
export HF_HOME=/root/autodl-fs/huggingface
export HF_HUB_CACHE=/root/autodl-fs/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/root/.cache/torch
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export ATTN_BACKEND=flash_attn
export SPCONV_ALGO=native

# 保守起点，不是已证明最优值
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
~~~

### A. read-only preflight

~~~bash
python -m pose_point_depth_mv.preflight_proobjaverse_official_slat_resume \
  --cache_manifest $CACHE/slat_manifest.json \
  --lifting_cache_manifest $CACHE/lifting_manifest.json \
  --target_decoder_audit $RUN/decoder_audit32_protocol2128_v1/report.json \
  --native_ss_report $SSROOT/ss_eval_synthetic_dev32_fixedcfg3_count125_v3/report.json \
  --stock_slat_freeze $FREEZE \
  --resume $STEP8000 \
  --max_steps 10000 \
  --world_size 8 \
  --grad_accum 1 \
  --allow_resume_max_steps_extension \
  --allow_resume_topology_change \
  --allow_resume_data_path_relocation
~~~

三个 allow flag 分别批准 8000→10000 horizon、4×2→8×1 topology、逻辑路径→相同物理 artifact 的 relocation；都不是“忽略检查”。

### B. 8000→8002 migration smoke

~~~bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  -m pose_point_depth_mv.train_proobjaverse_official_slat_condition_lora \
  --architecture v2 \
  --cache_manifest $CACHE/slat_manifest.json \
  --lifting_cache_manifest $CACHE/lifting_manifest.json \
  --target_decoder_audit $RUN/decoder_audit32_protocol2128_v1/report.json \
  --native_ss_report $SSROOT/ss_eval_synthetic_dev32_fixedcfg3_count125_v3/report.json \
  --stock_slat_freeze $FREEZE \
  --output_dir $OUT \
  --pretrained Stable-X/trellis-vggt-v0-2 \
  --indices all \
  --resume $STEP8000 \
  --max_steps 10000 \
  --run_until_step 8002 \
  --save_every 1000 \
  --log_every 1 \
  --grad_accum 1 \
  --num_workers 0 \
  --seed 42 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --condition_channels 1024 \
  --new_lr 1e-4 \
  --lora_lr 3e-5 \
  --new_weight_decay 0.01 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --grad_clip 1.0 \
  --warmup_steps -1 \
  --warmup_ratio 0.02 \
  --ema_decay 0.9995 \
  --amp_dtype bf16 \
  --amp_init_scale 8192 \
  --p_uncond 0.1 \
  --t_logit_mean 1.0 \
  --t_logit_std 1.0 \
  --t_schedule uniform \
  --min_condition_views 1 \
  --max_condition_views 8 \
  --stock_context_views all \
  --gradient_checkpointing \
  --allow_resume_max_steps_extension \
  --allow_resume_topology_change \
  --allow_resume_data_path_relocation
~~~

1~20 step smoke 只需把 run_until_step 改为 8001..8020，max_steps 始终保持 10000。

### C. 8002→10000 continuation

先对真实 step8002 重跑 A 命令，把 --resume 改为该文件；通过后：

~~~bash
export STEP8002=$OUT/checkpoints/step_008002.pt

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  -m pose_point_depth_mv.train_proobjaverse_official_slat_condition_lora \
  --architecture v2 \
  --cache_manifest $CACHE/slat_manifest.json \
  --lifting_cache_manifest $CACHE/lifting_manifest.json \
  --target_decoder_audit $RUN/decoder_audit32_protocol2128_v1/report.json \
  --native_ss_report $SSROOT/ss_eval_synthetic_dev32_fixedcfg3_count125_v3/report.json \
  --stock_slat_freeze $FREEZE \
  --output_dir $OUT \
  --pretrained Stable-X/trellis-vggt-v0-2 \
  --indices all \
  --resume $STEP8002 \
  --max_steps 10000 \
  --run_until_step 10000 \
  --save_every 1000 \
  --log_every 10 \
  --grad_accum 1 \
  --num_workers 0 \
  --seed 42 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --condition_channels 1024 \
  --new_lr 1e-4 \
  --lora_lr 3e-5 \
  --new_weight_decay 0.01 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --grad_clip 1.0 \
  --warmup_steps -1 \
  --warmup_ratio 0.02 \
  --ema_decay 0.9995 \
  --amp_dtype bf16 \
  --amp_init_scale 8192 \
  --p_uncond 0.1 \
  --t_logit_mean 1.0 \
  --t_logit_std 1.0 \
  --t_schedule uniform \
  --min_condition_views 1 \
  --max_condition_views 8 \
  --stock_context_views all \
  --gradient_checkpointing \
  --allow_resume_max_steps_extension \
  --allow_resume_topology_change \
  --allow_resume_data_path_relocation
~~~

## 18. 启动与训练故障树

| 失败阶段 | 最可能类别 | 第一检查项 | 可否原 checkpoint 重试 |
|---|---|---|---|
| preflight import | runtime closure/PYTHONPATH/native extension | 43-file增量、ReconViaGen、trellis_point_prior_mv、nvdiffrast | 是，未写训练状态 |
| lifting contract | manifest 搬错/no-VGGT cache 不匹配 | lifting format/config hash/raw_dino_only 字段 | 是；绝不能改 manifest |
| Native-SS deployment | A/B 路径或 semantic/hash 不同 | report/checkpoint resolve 与 SHA、CFG、amp、false_checks | 是；绝不能 subset compare |
| official decoder | audit/pretrained/target_source 不符 | report format/passed/pretrained/protocol SHA | 是；绝不能改 report |
| data identity | cache/UID/hash/freeze/audit 不符 | validator relocation detail | 是；只修路径映射或搬正确 artifact |
| DINOv2 startup | 本地 repo/weight/ABI 异常导致远程 fallback | TORCH_HOME 两个固定路径；先本地 import/load | 是 |
| Stable-X startup | HF snapshot不全、symlink断、offline miss | refs/snapshots/blobs、revision、pipeline.json | 是 |
| flash_attn | sm_120/torch/CUDA ABI | import、backend、版本、简单 kernel smoke | 是 |
| nvdiffrast | cpp extension build/import 方式 | load 返回 module patch、TORCH_EXTENSIONS_DIR | 是 |
| checkpoint torch.load | PyTorch weights_only 默认、文件不完整 | SHA/size、TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 | 是；不要重存旧 checkpoint |
| resume contract | max/topology/strict arg 不合法 | saved/current transition JSON、三个 allow flag | 是；修命令，不改 checkpoint |
| DDP/NCCL | rank/device/NCCL/共享路径 | CUDA_VISIBLE_DEVICES、8卡可见、rank mapping、NCCL日志 | 是 |
| first forward | dtype/backend/shape/cache | 最后一个通过的 validator、首样本 shape、CUDA trace | 是 |
| backward | unused param、nonfinite、OOM、extension | grad字段、DDP报错、显存、bf16/flash/spconv | 是；若无保存写入 |
| checkpoint save | OUT 非持久/空间/权限/原子 rename | df、路径实际指向、临时文件、rank0日志 | 从最近已校验 checkpoint 重试 |

任何合同失败时都不要改 step_008000.pt、manifest、Native-SS report、stock freeze 或 decoder audit 来“让它过”。先分辨是路径 relocation、搬错 artifact、源码版本、环境 ABI 还是实际 semantic mismatch。

## 19. 哪些东西绝不能随便改

### 19.1 训练语义 / checkpoint contract

禁止随意改变：

- architecture、pretrained、LoRA rank/alpha、condition_channels；
- dataset indices、manifest 内容、UID 集、cache hashes；
- Native-SS report/checkpoint/hash/step/weights/CFG/amp/false_checks；
- stock freeze、decoder audit；
- LR、weight decay、betas、clip、warmup、EMA；
- p_uncond、t schedule/parameters、condition view 范围、stock context views；
- seed；
- gradient checkpointing、amp dtype 虽未全部进入 strict-field compare，也应保持 canonical 值；
- global effective batch，除非另立训练实验；
- optimizer/EMA 状态。

受控可变：

- max_steps 只可严格延长并给 allow_resume_max_steps_extension；
- world_size/grad_accum 只可在 global effective batch 相等时配套改变，并给 allow_resume_topology_change；
- 白名单路径只可 resolve(strict=True) 到同一现存对象，并给 allow_resume_data_path_relocation。

### 19.2 可 benchmark 的性能项

- OMP/MKL/OpenBLAS/NumExpr threads；
- CPU affinity / NUMA placement；
- num_workers（先 smoke）；
- log_every、save_every；
- persistent_workers、prefetch_factor（需要单独代码改动与测试）；
- storage/RAM cache 布局，只要逻辑路径和 identity validator 仍通过。

### 19.3 环境兼容项

- nvdiffrast loader 适配；
- PyTorch weights_only 显式兼容；
- flash-attn/spconv/cumm 的 CUDA13/sm_120 build；
- NCCL device binding warning 的修复；
- cache/offline 环境变量。

这些不是训练数据语义，但可能改变数值执行顺序或运行稳定性；必须以 1~20 step smoke 和日志/finite 检查验证。

## 20. 源服务器真实 preflight JSON

以下是 2026-08-15 用当前正式 helper、真实 step8000 与真实 artifact 运行所得，exit code=0：

~~~json
{
  "dataset_contract": {
    "config_hash": "b242881421ed122458403beaa97b7dcd2752c6d2a25cc7feb4f82e8d963ccfbb",
    "object_count": 2000,
    "passed": true,
    "sample_count": 2000
  },
  "lifting_contract": {
    "config_hash": "9c0e465e3597f7c3d5ac93a341f8d2b1f748ba51e1cb7a4661f916e0e2c8cec0",
    "context_source": "raw_dino_only",
    "dino_feature_dim": 1024,
    "passed": true,
    "patch_count": 1369,
    "patch_side": 37,
    "vggt_feature_dim": 0,
    "vggt_model_executed": false
  },
  "native_ss_deployment": {
    "all_non_path_fields_exact": true,
    "passed": true,
    "path_relocated": false,
    "relocations": {}
  },
  "official_decoder_audit": {
    "format": "pose_point_depth_mv.proobjaverse_official_slat_decoder_audit.v1",
    "passed": true,
    "protocol_sha256": "36c2147c9d3d37b5dc867ff3d277a4af8f7ad9f2f5099edcc032719a0fe5c241",
    "sha256": "fc1ab7b211c005eddb46d546139fe669f646b3fb009edf478943b7ef48330329"
  },
  "passed": true,
  "resume_checkpoint": {
    "passed": true,
    "sha256": "49edb3bbdbd86b10c5eea14e9c80a9996076b6fd65a459db12b130b6560bda4d",
    "step": 8000
  },
  "resume_data_identity": {
    "all_non_path_fields_exact": true,
    "passed": true,
    "path_relocated": false,
    "relocations": []
  },
  "resume_training_contract": {
    "checkpoint_step": 8000,
    "current_global_effective_batch": 8,
    "current_grad_accum": 1,
    "current_max_steps": 10000,
    "current_world_size": 8,
    "ema_inherited": true,
    "max_steps_extended": true,
    "optimizer_inherited": true,
    "per_rank_rng_stream_exact": false,
    "saved_global_effective_batch": 8,
    "saved_grad_accum": 2,
    "saved_max_steps": 8000,
    "saved_world_size": 4,
    "topology_changed": true
  },
  "stock_slat_freeze": {
    "file_sha256": "131f9612c736a57cf41ed2161ee102ce8bcab3b9f39fd1a2473ce7919b290bc6",
    "freeze_sha256": "aeef32f5c17c1d039f93f1237b7c012f51d9854ad21a027a8c6645d09ceed91f",
    "passed": true
  }
}
~~~

源端没有 relocation，所以两个 path_relocated 都是 false；A72 软链接布局下应为 true，但 resolved target 与所有非路径字段仍必须 exact。

## 21. 机器合同与源码摘录索引

A72_RUNTIME_CONTRACT.json 包含：

- 47 个 parser 参数的真实 default/effect/resume 分类；
- 43-file runtime increment v2；
- artifact 与源码 hash；
- resume/deployment/data identity whitelist；
- checkpoint/dataset/DataLoader schema；
- 25 个 torch.load 与 9 个 torch.save site；
- DDP、环境、模型依赖、输出与 canonical commands。

A72_RUNTIME_SOURCE_EXCERPTS.txt 包含带 nl -ba 行号的：

- official/no-VGGT wrappers；
- parser、validate_args、checkpoint_args；
- full deployment / resume identity / transition validators；
- Direct/Lifting/Join dataset 与 sampler；
- model builder、Stable-X/DINO loaders；
- DDP、optimizer、resume、RNG、checkpoint save；
- forward/loss/backward/finite/EMA/log/report；
- spconv 和 attention backend；
- 完整 pre-output git status 与 v2 whitelist。

## 22. 自检

- CONTRACT.json 已通过 python -m json.tool。
- 两条 canonical torchrun 命令已由当前 make_parser().parse_args + validate_args 实际解析通过：smoke=(max=10000, run_until=8002, grad_accum=1, v2)，continuation=(10000,10000,1,v2)。
- resume identity 与 full Native-SS deployment 两组测试共 18 项，Ran 18 tests，OK。
- 源服务器真实 preflight exit code=0、passed=true。
- 排除本次三个输出文件后的原工作树 status snapshot SHA256 在生成前后保持 553ec4d3b8446698aa2472791af825728f67363643fbed7a5e976ae5b240f3ca。
- 仅凭三份文件是否足以构造正确 A72 resume 命令？是。命令、环境、固定参数、三个 override 的条件和真实 preflight 证据均已给出。
- 是否足以判断参数能否修改？是。CLI 表和第 19 节区分 strict semantic、受控 transition、性能和环境兼容；还明确指出了 validator 未覆盖但语义上仍不能改的参数。
- 是否足以定位 DataLoader/CPU 瓶颈？是。已给真实每样本 I/O/shape、workers=0、线程未限制、CPU validation、H2D/pin overfetch 和逐 tensor .item 同步热点，并给 benchmark 矩阵。
- 是否还有关键运行代码未覆盖？对当前 official/no-VGGT SLat resume 的关键 Python 路径已覆盖。runtime increment v2 是增量而非裸机闭包；Stable-X snapshot 内模型配置/权重和全部第三方 CUDA kernel 源码没有复制进摘录，而是以 revision/cache/version 绑定。唯一关键 artifact 缺口是真实 A72 step_008002.pt 不在源服务器，必须在 A72 生成后再 preflight。

最终放行条件：A72 上先运行 preflight；再完成 8000→8002；确认 stage_complete=true、checkpoint 可 load、日志 finite；对真实 step8002 再跑 preflight；最后才启动 8002→10000。
