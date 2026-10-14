# 2026-10-09：扩大语言 LoRA 的匹配 SFT 实验

代码交付与验证时间：2026-10-09 09:14–10:30 UTC。
初始交付状态（2026-10-09）：实现、局部真实模型验证和已有资产核对完成，当时正式训练尚未启动。
最新状态（2026-10-10）：5,000 步训练和三个检查点的完整开发集评分已完成；报告模块校验误判已修复，开发集报告已生成。扩大 LoRA final 为 60.42%，旧 QKV final 为 60.46%；扩大 LoRA 的官方 500 题评测尚未完成。

## 这次只改变什么

按照用户最新决定，暂停自动搜证／裁剪输入路线，保留 MPP-Qwen-Next、原始五帧、
原 QA、完整 Q-Former 和 Projector。先扩大语言 LoRA 的覆盖范围，继续纯 CE SFT，
不同时引入新数据、区域观察、教师、OPD 或 RL。

| 部分 | 历史 CE 对照 | 本次扩展 |
|---|---|---|
| Qwen 每层语言 LoRA | `attn.c_attn`，融合 QKV | QKV ＋ `attn.c_proj` ＋ `mlp.w1/w2/c_proj` |
| LoRA rank / alpha / 实际 dropout | 16 / 32 / 0 | 不变 |
| 32 层语言 LoRA 模块数 | 32 | 160 |
| 语言 LoRA 可训练参数 | 8,388,608 | 按本地 Qwen7B 配置计算为 35,782,656；启动时再记录实际值 |
| Q-Former／query tokens、Projector | 完整训练 | 不变 |
| EVA、Qwen 原始参数、embedding、LM head | 冻结 | 不变 |

这是扩大低秩适配范围，不是解冻 Qwen 全量参数，也不是声称复现了 LingoQA 官方
完整训练配方。问题仍是“看驾驶片段并回答原问题”；loss 仍只监督 assistant 的
参考答案 token，不把提示或视觉占位符作为答案训练。

## 初始化与预算

为能与历史实验比较，两组从同一完整父快照开始：
`outputs/mpp_lingoqa_qformer_all_1epoch_v1` 的 `final_adapter`、
`final_projector.pt`、`final_qformer_adapter.pt`。它已经训练过全部 406,212 条 QA。
**本次不是从当前 55.4% 的 CE final 再追加 5,000 步**，否则覆盖范围和追加训练量
两个变量会混在一起。

历史对照为 `outputs/mpp_feedback_sft_control_40k_2epoch_v2`；复用其答案和评分，
不重训、不覆盖已有文件。当前保留的历史结果是 parent / CE：dev 60.00% / 60.46%，
反复使用的 official500 为 54.2% / 55.4%。本次是否更好尚未知。

- 数据：原有 `datasets/processed/lingoqa_feedback_opd_40k_v2/train.jsonl`。
  Action 25,800、Scenery 14,200，共 40,000 QA，覆盖 27,599 个训练 segment。
- train SHA256：`d885dc2f2c59e2a8ad4c01a8240e471ed3d04e4602e887dd7ef680febb770712`。
  不改问答、不重新抽样、不使用 dev 证据框；沿用已隔离的划分。
- 四卡、每卡 batch1、梯度累积4、2 epoch：有效 batch16，共 **5,000 optimizer steps**，
  80,000 QA 访问。这里是匹配覆盖范围实验，不是把全数据再训练一遍。
- LR：语言 LoRA `2.5e-6`、Projector `2.5e-6`、Q-Former `5e-6`；AdamW、
  weight decay0.01、warmup3%、cosine、seed44、原 checkpointing 和梯度同步保持一致。
- max text1536；生成 greedy96 token、batch1；任务提示仍 `legacy_v1`。
- 新输出：`outputs/mpp_language_lora_decoder_40k_2epoch_v1`；
  新评测：`outputs/language_lora_eval_v1`；不重新缓存原始五帧 EVA。

扩展时将旧 QKV 的 A/B 直接复制进一个新的扩大 adapter。先转 FP32 再复制，避免
经过 BF16 舍入；不将旧增量 merge 到冻结主干。新增 A 正常初始化，B 全零，初始
增量为零。真实小 Qwen 检查已验证扩展前后初始 logits 完全相等。

首次扩展只是权重初始化，重新开始与历史 CE 一致的 optimizer／scheduler。
断点续训则加载完整扩大 adapter、optimizer、scheduler、数据位置及各 rank RNG，
**不再次扩展或清零新增模块**。推理也按保存的完整 `target_modules` 加载。

## 代码变动

- `mpp_video_qa/lora_expansion.py`：明确五类 decoder 线性层的完整路径；继承旧
  FP32 QKV、验证新 B 为零，输出范围和参数计数。只覆盖 decoder，不匹配 LM head。
- `mpp_video_qa/model.py`：首次训练接入扩展；正常加载／推理／resume 仍读取保存配置。
  原 QKV 精度加载路径保留。
- `scripts/feedback_opd_common.py`：增加 `--language-lora-scope decoder_linear`；
  仅新范围加入身份字段，不改变旧 QKV 实验的身份和恢复契约。
- `scripts/train_feedback_opd.py`：复用已有 `sft_control` 训练循环；纯 CE 不要求
  教师 probe、不创建 Teacher 或 Judge；新增输出层／MLP 的更新跟踪和每 rank 初始化记录。
  新 CE 身份明确为 `assistant_only_ce`，旧 OPD 和旧 CE resume 元数据保持兼容。
  文件名虽含 OPD，这个入口执行的仍是纯 CE。
- `scripts/run_language_lora_sft.sh`：训练、恢复、轻量状态、dev／official 评测入口。
- `scripts/report_language_lora_sft.py`：核对同父权重、原 QA 和匹配预算；复用历史评分；
  按相同步数比较，并输出配对修复／退化、segment bootstrap 区间。
- `tests/test_lora_expansion.py`：真实本地 Qwen／PEFT 权重迁移、更新、保存恢复、
  CE 无教师依赖和 dev-only 选择的四项测试。

启动后 `lora_initialization_rank_*.json` 应记录实际 160 模块、旧 64 个张量继承、
新增 256 个张量、`inherited_weights_exact=true`、`new_B_zero=true`。
训练结束的 summary 含三组参数计数，以及旧 QKV、新输出／MLP、Projector、
Q-Former 的代表参数更新幅度。不能仅根据 loss 下降就认定回答能力提升。

## 命令

以下默认使用已存在的 `/tmp/zjr_mpp_fast` 模型及 EVA 缓存；若缺失，入口回退到持久路径。
状态入口使用绝对路径，避免从 Tracker 根目录运行时找不到相对脚本。
Shell 脚本必须交给 **bash，不要用 Python 执行**。

后台四卡启动（只启动一次）：

```bash
systemd-run --user --unit=mpp-language-lora-sft-v1 --collect \
  --property=WorkingDirectory=/home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa \
  --setenv=CUDA_VISIBLE_DEVICES=0,1,2,3 \
  --setenv=MASTER_PORT=29711 \
  /usr/bin/bash scripts/run_language_lora_sft.sh train
```

监视：

```bash
watch -n 10 'bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_language_lora_sft.sh status'
```

日志：

```bash
journalctl --user -fu mpp-language-lora-sft-v1.service -o cat
```

无需 systemd 时可用前台替代；**不要与后台命令同时启动**：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29711 \
  bash scripts/run_language_lora_sft.sh train
```

训练完成后：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
CUDA_VISIBLE_DEVICES=0 bash scripts/run_language_lora_sft.sh eval_dev
CUDA_VISIBLE_DEVICES=0 bash scripts/run_language_lora_sft.sh eval_selected
```

dev 对原 QKV 和扩大 LoRA 分别从 1000／2500／5000 步选择最佳，合并 Action3120 ＋
Scenery1200，共4320问，分数相同选较早步数。先看相同步数差异，再看各自最佳；
保持两组相同选择机会。父快照也保留为不继续训练的参考。
新报告固定选择后才评 official500；不能按官方结果换 checkpoint。该500问已经
多次用于项目迭代，**不是新的盲测集**。

如中断后恢复，可在确认旧任务已退出后使用下面的前台替代命令；checkpoint 路径
改成实际完整保存的步数，不要对仍在运行的任务同时启动恢复：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29711 \
LANGUAGE_LORA_RESUME_FROM=/home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/outputs/mpp_language_lora_decoder_40k_2epoch_v1/checkpoint-001000 \
  bash scripts/run_language_lora_sft.sh train
```

## 已完成的验证与实现检查问题

1. 本机真实父三路权重哈希、原40k数据哈希和已完成历史 CE 预算核对通过。
   `check` 输出40000条、5000步、同父快照和原QA；没有据此声称缓存重建或训练已完成。
2. 新增4项测试通过：使用本地旧 Qwen 的真实 modeling code 和 PEFT0.13.2，缩小为
   两层 CPU BF16 模型，验证 FP32 继承、初始 logits 相等、新模块 CE 更新、保存／加载
   和恢复 AdamW 后下一步与不中断训练一致。报告流程用固定小数据测试历史复用及
   official 不重选 checkpoint。
3. 回归：feedback OPD v2 的9项、执行优化8项、真实旧 Qwen core 的1项通过。
   共 **22项通过，无跳过**；执行优化包含两进程 Gloo 梯度同步检查。
4. Bash语法、Python3.8编译、CLI和轻量状态检查通过。正式 MPP 四卡完整训练／显存
   仍未运行；小模型和 Gloo 检查不等价于完整四卡 NCCL smoke，不承诺新的准确率或速度。

实现检查中遇到并处理的错误（不是用户正式训练报错）：

- 缩小词表的小 Qwen fixture 触发 PEFT 自动保存 embedding，源 adapter 张量数不符合
  真实 QKV-only 合约。测试保存时显式 `save_embedding_layers=False`，不放宽生产检查；
  本地真实父 adapter 已核对为64个 QKV 张量、8,388,608参数。
- 报告 fixture 缺少真实评分记录使用的 `references` 字段；补齐测试记录后通过。
- 两进程 Gloo 回归在沙箱内无法解析本机回环地址，报 `Cannot resolve 127.0.0.1 to a
  (local) address`。获准在沙箱外重跑后8项全部通过；没有为此更改训练通信逻辑。

本次未修改数据／参考答案／Judge协议，未重跑旧 benchmark，未停止其他任务，
未删除 `/tmp` 或持久缓存，未启动正式训练。

## 2026-10-10 09:55–10:00 UTC：正式结果校验误判修复与开发集报告

用户完成三检查点的开发集生成／评分后，报告在 `model_specs()` 报
`ValueError: saved adapter does not contain the intended language modules`。
此前一次排查只定位原因，没有修改脚本；用户再次执行仍报相同错误。本节记录实际修复。

根因不是训练缺失模块。本机 PEFT 0.13.2 在目标路径数量达到 20 时尝试压缩
`target_modules`。真实 32 层的 160 条完整路径保存为 `c_attn/c_proj/w1/w2`，
`c_proj` 同时覆盖 `attn.c_proj` 和 `mlp.c_proj`。报告旧实现只接受完整路径后缀，
因此把合法的短名称配置误判为模块缺失。原真实模型测试只用两层／10 个目标模块，
没有触发 PEFT 压缩；报告测试又 mock 了 `model_specs`，遗漏了这一交界条件。

实际权重检查：旧 QKV 的 1000／2500／final 各为 64 个张量；扩大 LoRA 对应三个
检查点各为 320 个张量，五类模块在 32 层均有 A/B，未发现额外 LM head 权重。
训练 summary 记录 5000 更新、80000 QA 访问，语言／Projector／Q-Former 以及四类
新增语言模块的代表参数均有非零变化。本轮故障仅阻断报告，不改变已经训练出的模型
或已完成的评分。

修复范围：

- `scripts/report_language_lora_sft.py` 新增 `validate_saved_language_adapter`：使用
  PEFT 的名称／后缀匹配语义，支持完整路径、模块类型路径及压缩后缀；随后读取
  safetensors 的真实键名和形状，验证 32 层逐模块 A/B 完整性、rank16／alpha32、
  rank 维度及无额外张量。没有通过取消检查来掩盖问题；缺层、缺矩阵或多余目标仍报错。
- 验证只读取 adapter 的键和形状，不加载 7B 模型，不使用 GPU；报告继续执行原有
  数据／模型哈希、问答一致性、生成／Judge 协议和开发集选模检查。覆盖信息写入
  `checkpoint_specs.*.adapter_validation`。
- `tests/test_lora_expansion.py` 的真实 Qwen fixture 改为四层／20 个目标，实际触发
  PEFT 压缩，并验证保存后的适配器仍能通过报告检查、正常恢复和继续更新。
- 新增 `tests/test_language_lora_report.py`，覆盖合法的三种配置形式、旧 QKV、伪装成
  decoder 的 QKV-only 权重、缺矩阵／缺层、多余张量／配置、错误秩形状和 alpha。

验证：新增 7 项检查和已有 4 项真实 Qwen／流程测试全部通过，无跳过。执行
`bash scripts/run_language_lora_sft.sh report` 成功，完整核验并复用已有预测及评分，
没有重新训练、重新生成或重新调用 Judge，也没有修改 adapter／原始评分文件。

开发集结果（Action3120＋Scenery1200）：

| 模型 | 1000 步 | 2500 步 | final／5000 步 |
|---|---:|---:|---:|
| 历史 QKV LoRA | 59.26% | 60.16% | 60.46% |
| 扩大 decoder LoRA | 59.38% | 60.14% | 60.42% |

父 SFT 为 60.00%。扩大 final 的 Action 为 58.40%，Scenery 为 65.67%，合并为
2610／4320；Scenery 单项 65.67% 不能当作整体或官方集分数。两分支均按开发集选择
final，跨分支保留 `qkv_final`。扩大 final 相对旧 QKV final 修复 67／退化 69，
净少 2 题（−0.0463 个百分点），segment bootstrap 95% CI 约 [−0.63,+0.56] 个百分点；
当前没有证据说明扩大语言 LoRA 已带来稳定增益。相对父模型修复 201／退化183，
净 +0.4167 个百分点，区间 [−0.55,+1.42] 也跨零。

报告：[Markdown](../outputs/language_lora_eval_v1/report.md)、
[JSON](../outputs/language_lora_eval_v1/report.json)。本轮没有启动新的官方评测，后续
仅需运行 `CUDA_VISIBLE_DEVICES=0 bash scripts/run_language_lora_sft.sh eval_selected`；
该入口沿用开发集选择、复用已有父模型／QKV 官方评分，为扩大 final 补齐官方评分。
官方500题已反复使用，仍不作为新的盲测集。
