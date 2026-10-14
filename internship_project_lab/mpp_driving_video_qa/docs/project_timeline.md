# MPP Driving Video QA：完整项目归档与时间线

> 最后更新：2026-10-14（补录扩大LoRA官方结果、权重3训练／评测，并整理源码上传与checkpoint索引，见第64–66节）
> 本文是项目事实的统一入口，按时间记录任务定义、数据、代码演进、训练、评测、
> 故障修复和当前状态。数值均来自已落盘产物；仍在运行的实验单独标为“进行中”。

## 1. 当前状态摘要

- 项目目标：基于官方 MPP-Qwen-Next checkpoint，对短时驾驶视频的 5 帧序列进行
  自由问答 SFT；不包含先前项目的时间定位器、候选池或 Verifier。
- 数据：首轮使用 Action 30,000 问；full-v2 已构建 Action + Scenery 406,212 问、
  27,599 个训练 segment，固定 dev 4,320 问/400 segment，official evaluation
  500 问/100 segment。三个 split 的 segment ID 与帧路径两两无交集。
- 视觉输入：固定完整5帧。历史projected缓存为 `[5,32,4096]`；历史OPD使用冻结
  Q-Former缓存 `[5,32,768]`，训练语言LoRA和投影层。视觉适配实验另有EVA缓存，
  三者不能混用。full-v2所需28,099段缓存已完成。
- 已完成：30k SFT、full-v2全量LoRA、LoRA+projector联合SFT、10k定向续训、20k
  Q-Former视觉续训；后两种续训未超过joint。现已完成100k及全量完整Q-Former训练，
  后续同预算CE控制是当前保留模型，见第49节。
- 同路径评测：Base/full-data LoRA/joint的官方500题Judge为43.0%/45.2%/46.4%；
  joint的Action3120题为51.47%，Scenery1200题为60.17%。早期projected路径数值
  保留在历史条目，不与这些同路径结果混比。官方500题已反复查看。
- 历史答案条件OPD：2000道训练候选，师生各8000次采样完成；学生47.325%、答案条件教师
  70.6375%，这是训练侧随机采样。筛选751条（Action614/Scenery137）。
- 2026-10-02：同751条的SFT控制组及答案条件OPD均已完成47次更新，完整开发评测
  已完成；本轮joint/SFT/OPD分别为54.05%/53.66%/53.22%，继续保留joint。
- 历史偏好实现：复用学生采样构造120对事实偏好＋40条回放，新增固定joint参考分数缓存、
  DPO＋匹配CE对照、四卡训练和评测报告。实际为160条小规模试验，不是1000对或全量训练。
  自动筛选没有冒充人工视觉审核。正式DPO/SFT及开发评测已完成，未超过joint，见第28节。
  断点恢复可运行但非逐位复现。详见第24–27节与
  [偏好训练实现及命令](preference_dpo_20261002.md)。历史条目的“未运行”指当时状态。
- 全量参考：完整Q-Former 406,212条×1epoch，历史dev60.05%、official500为54.2%；
  相对100k模型57.22%/51.0%继续提升。低学习率续训完成，开发选中24000为61.78%，
  官方52.2%，继续保留54.2%参考模型，见第35节。
- 外部4B教师600题完成：学生58.83%、教师42.33%，修复82/退化181，尚未进入蒸馏。
  完整说明五图／原生视频对照已完成，分别40.33%／38.83%，均未超越学生，见第37–39节。
- 最新反馈OPD v2：4万原QA、27599训练视频、两轮、四卡有效batch16，两分支各
  5000次更新。200条固定学生错误的教师条件对照已完成：reference_draft纠正通过
  103/200，reference_only为190/200，配对修复90／退化3；这是训练侧条件诊断，
  不是学生held-out准确率。代码、报错修复、82项回归与GPU检查见第40–45节。
- 最新反馈OPD v2于2026-10-06 22:41完成5000步，fast从1000恢复后的4000步耗时
  10h45m。完整dev及官方500评测于10-07完成：父模型／CE控制／OPD分别
  dev60.00%／60.46%／60.23%，官方54.2%／55.4%／54.8%；OPD相对同预算CE
  无独立收益，配对区间跨零。55.4%为本轮dev选定CE模型的观测分数，不是盲测
  或统计显著提升。第46–47节旧快照保留作为历史。
- 10-07已完成80共同失败＋30Judge正确控制的逐项复核、386份固定权重观察生成／
  评分；局部ROI相对同输入预算副本对照修复5、退化1，但还有语义细节损失。
  下一阶段候选为观察方式／问题目标绑定的SFT适配，不直接扩大旧OPD。没有
  启动新训练，没有拿dev框作训练数据。完整边界见第50节及专项文档。
- 10-08 07:02完成旧规则ROI的冻结dev4320三臂：full60.30%、duplicate56.88%、
  ROI53.52%；full→ROI修复238／退化531，净−293。旧自动候选不支持直接宣称
  观察收益，ROI SFT未启动，详见第52节。
- 10-08新增外部VLM完整问题驱动搜证：请求native crop、看返回图、选已观察证据，
  原五帧始终保留。只给冻结MPP图像，不传外部答案；默认两臂，不做外部C／D。
  代码及38项定向检查通过，见第53节；8B随后下载完成并实际完成四卡搜证。
- 10-08 18:42完成新搜证完整dev4320评测：full60.39%→evidence55.95%，修复291／
  退化483，净−192，按segment的95%区间[−6.02,−3.07]个百分点。selected3378问
  净退化193，执行failed434问均回退原五帧，不能把整体降分主要归因于执行失败。
  Action−5.32、Scenery−2.17个百分点。官方未运行、未冻结策略、无新训练；建议
  保留原五帧基线，先区分错误选区与附加视图适配问题，见第54–55节和
  [完整结果分析](evidence_search_results_20261009.md)。
- 10-09完成24个不同segment的失败审核和72份F/D/R生成评分。16道所选历史退化中
  12道在只追加全图副本时也退化；真实裁剪相对副本修复2／退化4。全dev795题的
  附加槽全是整图、净−38，实际局部裁剪2582题仍净−155。部分同义答案Judge翻转、
  个别回放不稳定，不能把全部483题当作人工确认退化。局部细节有受控个例收益，
  但冻结MPP无条件追加策略不成立；没有开始区域SFT或新OPD，见第56节和
  [案例审核](evidence_case_audit_20261009.md)。
- 10-09按用户决定暂停搜证分支，新增扩大语言LoRA的匹配纯CE实验：原QKV扩为
  注意力输出＋三类MLP，rank16不变，同父快照／40k原QA／两轮5000更新。权重继承、
  新模块更新、保存恢复、dev选模及历史结果复用已实现，22项测试与资产核对通过。
  尚未启动正式训练，无新准确率；命令和检查问题归档在第57节及
  [扩大语言LoRA实验](language_lora_sft_20261009.md)。

## 2. 2026-09-26：独立项目建立

### 2.1 任务定型

在 `internship_project_lab/mpp_driving_video_qa` 建立与视频定位项目完全隔离的工程。
稳定任务协议为：输入按时间排序的驾驶片段帧与自然语言问题，使用发布的
MPP-Qwen-Next SFT checkpoint 初始化，对 Qwen attention 做 LoRA SFT，输出简洁自由
文本答案，并在相同视觉输入、prompt 和 greedy 解码下比较 Base 与 SFT。

首批实现：

- `mpp_video_qa/core.py`：prompt、视觉占位协议、缓存 key 与 JSONL 工具；
- `mpp_video_qa/schema.py`：LingoQA 字段规范化；
- `mpp_video_qa/data.py`：缓存特征数据集和 collate；
- `mpp_video_qa/model.py`：官方权重加载、视觉 token 注入、LoRA 与生成；
- `scripts/prepare_lingoqa.py`：数据转换与按 segment 切分；
- `scripts/cache_visual_features.py`：MPP 视觉特征缓存；
- `scripts/train_sft.py`：DDP LoRA SFT；
- `scripts/generate_answers.py`、`evaluate_answers.py`、`compare_runs.py`：生成、评分、
  fixed/regressed 对比；
- `scripts/ask_video.py`：本地驾驶视频问答入口。

### 2.2 数据构建

读取 Action `train.parquet` 和 Evaluation `val.parquet`。最初 pandas 无 parquet
engine，安装 `pyarrow==14.0.2` 后解决。本轮实际没有把 scenery 加入训练。

| split | 问题数 | segment 数 | 用途 |
|---|---:|---:|---|
| train | 30,000 | 17,469 | LoRA SFT |
| dev | 3,120 | 300 | loss、checkpoint 与诊断 |
| eval | 500 | 100 | 最终 held-out 对比 |

`datasets/processed/lingoqa_v1/manifest.json` 记录 train/dev overlap 为 0；后续审计
确认 train/dev、train/eval、dev/eval 的 segment ID 两两无交集。所有有效样本均为
5 帧，train/dev/eval 问题共享同 segment 的缓存，但不会跨 split 共享 segment。

### 2.3 视觉缓存修复

最初缓存阶段不必要地构造完整 Qwen-7B 并读取约 15 GB 的 MPP language 权重，进程
长时间不产出缓存。代码改为视觉专用路径：只构造 EVA-ViT、Q-Former、`llm_proj`，
并只从 MPP shards 读取投影层所需张量。最终生成 17,869 份不同 segment 特征，形状
均为 `[5,32,4096]`。

## 3. 2026-09-26 至 09-27：训练启动故障与修复

### 3.1 机械盘 I/O 与临时 SSD 副本

`/home` 共享盘读取约 1.6--2 MB/s，单个 MPP shard 可停留十几分钟。为避免每次启动
等待，将可再生运行副本放到 `/tmp/zjr_mpp_fast`：

- Qwen7B-chat 约 15 GB；
- sharded MPP-Qwen-Next checkpoint 约 15 GB；
- LingoQA 视觉缓存约 22 GB。

这是性能副本，不是唯一数据源；删除前必须保证当前训练与评测已结束。

### 3.2 分布式加载超时

为避免四个 rank 同时争抢磁盘，模型按 rank 串行加载。机械盘读取超过默认通信超时，
其余 rank 在 barrier 报 `Socket Timeout`。`train_sft.py` 将 process-group timeout
提高到 12 小时；SSD 副本解决主要速度问题，长 timeout 只覆盖共享服务器波动。

### 3.3 `v5`：Qwen/PEFT core 解析错误

首次 forward 报 `RuntimeError: could not resolve Qwen core`。旧逻辑沿
`.base_model` 下钻到没有 `lm_head` 的 `QWenModel`。修复为优先使用 PEFT
`get_base_model()`，并在同时具有真实 `transformer` 与 `lm_head` 的 causal LM 停止。
新增 `--max-steps`、rank 加载日志、失败 JSON 以及 adapter 保存/重载回归检查。

真实单卡 smoke：8 micro steps / 2 optimizer steps，LoRA 最大参数变化
`4.0026887e-05`。

### 3.4 `v6`：首次 backward 显存失败

四卡完成模型加载后，rank 0 首次 backward 报
`CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)`；该次运行未产生有效
optimizer step。根因是每卡完整 7B、decoder 激活、DDP 临时张量及外部显存占用叠加，
不是数据或 loss 错误。

显存修复：

1. 启用 Qwen gradient checkpointing，训练时关闭 KV cache；
2. 拼接后的 `inputs_embeds.requires_grad_(True)`，避免冻结 embedding 时 checkpointing
   切断 LoRA 梯度；
3. DDP 使用 `gradient_as_bucket_view=True` 与 10 MB bucket；
4. 记录每 rank 上卡、DDP、首次 forward/backward/step 的显存；
5. 异常时写 `failure_rank_N.json`；
6. smoke 与正式训练统一由环境参数进入同一代码路径。

验收结果：

| 运行 | 卡数 | micro/optimizer | LoRA max delta | peak allocated |
|---|---:|---:|---:|---:|
| `gc_single_smoke_20260927` | 1 | 8 / 2 | `4.0026887e-05` | 14.82 GiB |
| `gc_4gpu_smoke_20260927` | 4 | 8 / 2 | `4.0026815e-05` | 16.04 GiB/rank0 |

四个 rank 均跨过首次 backward 和 optimizer step；这里只证明训练链路和 LoRA 更新
有效，不把近似相同的参数变化夸大为完整数值等价证明。

## 4. 2026-09-27：全量 LoRA SFT 完成

正式运行 `outputs/mpp_lingoqa_lora_gc_full_v1`：

| 配置 | 数值 |
|---|---|
| 数据 | Action train 30,000 问 |
| 模型 | MPP-Qwen-Next + Qwen-7B LoRA |
| LoRA | `c_attn`, r=16, alpha=32, dropout=0.05 |
| 冻结部分 | 视觉编码器、Q-Former、投影层、Qwen base |
| 卡数与 batch | 4 卡，batch 1/rank，grad accumulation 4 |
| epochs / optimizer steps | 3 / 5,625 |
| learning rate | `2e-5` |
| 训练耗时 | 5,958.85 秒（约 99.3 分钟） |
| LoRA max delta | `0.00805494` |
| rank0 peak allocated | 16.04 GiB |

| epoch | dev teacher-forced loss |
|---:|---:|
| 1 | 0.41455 |
| 2 | 0.37717 |
| 3 | 0.36229 |

loss 连续下降说明监督目标被拟合，但不等于自由生成正确率提升。

## 5. 2026-09-27：生成链路修复

首次 Base/SFT 生成在第一条前同时报 `bos_token_id has to be defined`。原因是只向
Hugging Face `generate()` 传 `inputs_embeds`。修复为同时传 prompt `input_ids` 与
替换后的 `inputs_embeds`，并只解码 prompt 后的新 token。

随后真实 smoke 输出恒为 `"0"`。定位到训练预处理会为答案追加 `<|im_end|>`，空答案
用于推理时等价于提前关闭 assistant turn。协议拆分为：训练输入包含标准答案和结束
标记；生成输入只保留开放的 assistant prefix。新增 `--max-records` 后，真实 Base
单条能生成完整自然语言答案。

## 6. 2026-09-27：official evaluation 与评分修正

Base 和 Epoch 3 SFT 在相同 500 条 evaluation、相同 5 帧、legacy prompt 与 greedy
解码下完成自由生成。

第一次本地 Judge 漏掉官方答案预处理，得分 Base 39.6%、SFT 34.6%。对照 LingoQA
官方 `judge.py` 后修正为：reference/prediction 执行 `lower().strip()`，question 保持
原文，多参考答案取最大 logit，`max_length=128`，`logit>0` 判正确。旧结果保留，修正
结果写入 `scored_official_v2`。

| 500 题 | Base | Epoch 3 SFT | 变化 |
|---|---:|---:|---:|
| Lingo-Judge | 42.2% | 37.4% | -4.8 pp |
| Token F1 | 0.21843 | 0.28543 | +0.06700 |
| Normalized exact | 0.0% | 0.4% | +0.4 pp |

Judge 配对：fixed 93、regressed 117，净少 24 题。Token F1 增加说明答案更贴近参考
文本风格，但不能抵消自动正确性下降。Lingo-Judge 是文本代理指标，已观察到与参考
答案明显冲突仍判正确的假阳性，不能等同于人工视觉真值。

错误集中于 “current action and why” 子集：

| 子集 | Base | SFT | fixed / regressed |
|---|---:|---:|---:|
| 当前动作及原因，100 题 | 55.0% | 22.0% | 5 / 38 |
| 其他 400 题 | 39.0% | 41.25% | 88 / 79 |

SFT 在 100 道动作题中 95 道以 `I am stopping` 开头；但训练集中对应 431 题仅
97 道（22.5%）以 stopping 开头，所以不能简单归因于训练标签的单一多数类。

## 7. 2026-09-27：固定开发集诊断完成

新增 `scripts/diagnose_sft.py`、`run_sft_diagnosis.sh`，固定 seed=42 的随机 dev 512
题，并补齐全部 48 道动作原问、48 道同义改写、48 道换视频和 48 道改写换视频；
每个模型生成 692 条。比较 Base、Epoch 1、Epoch 2、Epoch 3，不用 evaluation 选择
checkpoint。

### 7.1 随机 dev 主结果

| 模型 | Judge | Token F1 | 平均答案词数 | 对 Base fixed/regressed |
|---|---:|---:|---:|---:|
| Base | 36.13% | 0.24052 | 55.62 | - |
| Epoch 1 | 44.53% | 0.57940 | 18.93 | 119 / 76 |
| Epoch 2 | 44.53% | 0.60756 | 18.12 | 123 / 80 |
| Epoch 3 | **46.48%** | 0.60173 | 18.34 | 125 / 72 |

Epoch 3 是固定 dev 上最佳 checkpoint；这与 evaluation 退化同时成立，说明主要问题
不是“完全没训练动”，而是 dev 与 evaluation 的措辞/输出分布泛化不足。

### 7.2 动作题与视觉依赖

| 模型 | 原问法 Judge | 改写 Judge | 原问 stopping | 改写 stopping |
|---|---:|---:|---:|---:|
| Base | 0.00% | 29.17% | 0/48 | 0/48 |
| Epoch 1 | 14.58% | 10.42% | 16/48 | 6/48 |
| Epoch 2 | 16.67% | 20.83% | 6/48 | 9/48 |
| Epoch 3 | 18.75% | 25.00% | 19/48 | 27/48 |

Base 在 48 道原问法中全部产生 AI 自述式回答，说明 `you/your` 存在角色歧义。
Epoch 3 消除了 AI 自述，但又形成 stopping/decelerating 偏置：原问法 48 题中
stopping 19、decelerating 20。

Epoch 3 换视频后 43/48 答案文本变化、35/48 动作前缀变化；与原视频参考答案的一致率
从 18.75% 降至 6.25%。48 份缓存均不同、有限且形状正确。这证明输出受视觉输入影响，
但不能证明模型已经正确理解动作或时序。

数据模板审计发现：随机 dev 512 题中 509 题的精确问题措辞在 train 出现；official
evaluation 只有 11/500 出现。segment 没有泄漏，但 dev 对问题模板泛化的测量明显偏弱。

## 8. 2026-09-28：驾驶视角任务说明对照（已完成）

为隔离角色歧义，只在推理输入加入 `driving_perspective_v1`：明确画面来自自车前向
摄像头，问题中的 `you/your` 指自车或驾驶员。原始视频、问题、答案、权重和 greedy
解码均不变，`legacy_v1` 仍是默认协议。

比较 Base 与 Epoch 3，各 596 条：固定随机 dev 512，补齐动作题，以及 48 条动作
同义改写。旧 prompt 结果复用已完成诊断，新 prompt 单独生成并做逐题 paired 分析。
输出目录为 `outputs/task_instruction_ablation_v1`。

实验最终完成。新任务说明在随机 dev 上使 Base Judge 从 36.13% 变为 35.94%，
Epoch 3 从 46.48% 降到 41.80%；动作原问法中 Epoch 3 从 18.75% 降到 10.42%，
同义改写从 25.00% 降到 8.33%。因此拒绝该提示词版本，后续继续使用
`legacy_v1`。完整结果位于 `outputs/task_instruction_ablation_v1/report.json`。

## 9. 当前可复现产物索引

| 产物 | 路径 |
|---|---|
| 数据清单 | `datasets/processed/lingoqa_v1/manifest.json` |
| 全量 SFT | `outputs/mpp_lingoqa_lora_gc_full_v1/` |
| Base official eval | `outputs/eval_base_v1/scored_official_v2/` |
| SFT official eval | `outputs/eval_sft_v1/scored_official_v2/` |
| Base/SFT paired | `outputs/base_vs_sft_official_v2.json` |
| 开发集诊断 | `outputs/sft_diagnosis_v1/report.{md,json}` |
| 48 题对照 | `outputs/sft_diagnosis_v1/action_cases.jsonl` |
| 提示词对照 | `outputs/task_instruction_ablation_v1/`（已完成） |
| full-v2 数据清单 | `datasets/processed/lingoqa_full_v2/manifest.json` |

主要代码更新：

- `core.py`：legacy 与 driving-perspective 两个版本化任务说明；
- `model.py`：Qwen/PEFT core、训练/生成 prompt 分离、`input_ids+inputs_embeds`
  生成、版本化任务说明注入；
- `train_sft.py`：长 timeout、串行加载、gradient checkpointing、显存/失败记录；
- `evaluate_answers.py`：官方 Lingo-Judge 预处理协议；
- `diagnose_sft.py`：固定 dev、动作改写、换视频控制；
- `evaluate_task_instruction.py`：任务说明消融的准备、状态和 paired 报告。
- `prepare_lingoqa.py`：固定旧 dev、按来源留出、无限制训练导出、重叠与指纹审计；
- `cache_visual_features.py`：旧缓存内容校验、增量生成及独立视觉/MPP 权重根目录；
- `train_sft.py`：周期性可恢复 checkpoint、优化器/scheduler/位置与各 rank RNG；
- `generate_answers.py`：数据、adapter、prompt 和生成参数的运行指纹；
- `run_full_v2.sh`、`report_full_v2.py`：full-v2 编排和分项 paired 报告。

## 10. 2026-09-28 当时的下一阶段与结论边界（历史记录）

以下是 full-v2 训练前的计划与首轮结论，完成情况和最新结论见第 13–14 节。

1. 完成 full-v2 新增 10,230 个 segment 的视觉缓存。
2. 从官方 MPP 权重训练 full-v2 一个 epoch，以固定 Action 与 Scenery dev 自由生成选择
   checkpoint；不继续使用已退化的 driving-perspective prompt。
3. 只对选定候选执行 official evaluation，并保留 Base、旧 SFT、新 SFT paired 结果。
4. 对最终候选补充帧顺序打乱或单帧控制；在此之前不宣称获得“视频时序理解”。
5. `/tmp/zjr_mpp_fast` 是可再生运行副本，仅在全部相关服务结束后删除。

当前可严谨陈述：完成了独立数据—视觉缓存—四卡 LoRA SFT—官方评分—配对诊断的
完整工程闭环；SFT 提高了固定 dev 与文本贴合度，并消除了 Base 的 AI 自述问题，
但 official evaluation 总体 Judge 下降且动作泛化不足。不能把该版本包装为已获得稳定
驾驶理解提升。

## 11. 文档索引

- `implementation.md`：当前稳定实现与复现协议；
- `training_bringup_issues_20260927.md`：I/O、分布式和显存故障专项记录；
- `qwen_core_fix_20260927.md`：Qwen/PEFT core 修复；
- `sft_diagnosis_20260927.md`：首轮 SFT 评测和开发集诊断；
- `task_instruction_ablation_20260928.md`：当前提示词对照实验。
- `data_download_and_full_v2_bringup_20260928.md`：下载中断、分块恢复、full-v2 数据与
  首次缓存启动故障。
- `full_v2_action_diagnosis_20260929.md`：全量训练/评测补录，三个动作诊断的实现、
  数值、视觉复核、局限与下一轮决策。
- `../../mpp_llava_video_workspace/docs/project_master_timeline_20260924.md`：共享 MPP
  环境安装、权重下载和分片修复历史。

## 12. 2026-09-28：Action + Scenery 全量覆盖 v2

确认首轮所谓“全量”实际为数据构建器输出的 30,000 条 Action 子集：先固定随机留出
300 个视频，每视频最多 12 题，再从其余样本随机取 30,000 条。它只占排除 dev 后
Action 问答的约 11.45%，且 17,469 个入选视频中 14,483 个仅获得 1–2 道监督。

完成 Scenery 压缩包续传与 CRC 校验后，扩展数据构建协议：原样保留旧 Action dev
3,120 题并排除这 300 个视频的全部 3,318 条原始标注；Scenery 以 seed 42 固定留出
100 个视频，每视频取 12 题作为独立 dev；训练取消总条数和每视频题数上限。实际产物：

| split | 问答 | 视频 | 来源 |
|---|---:|---:|---|
| train | 406,212 | 27,599 | Action 262,005 + Scenery 144,207 |
| dev | 4,320 | 400 | Action 3,120 + Scenery 1,200 |
| official eval | 500 | 100 | Evaluation |

训练、dev、evaluation 的视频 ID 和帧路径交集均为 0，所有问答 ID 唯一，所有训练和
dev 样本均有 5 帧。产物及 SHA-256 记录在
`datasets/processed/lingoqa_full_v2/manifest.json`。

代码同步增加：已有视觉缓存的帧路径/协议/张量校验和增量生成；每 5,000 optimizer
step 保存 adapter、优化器、调度器、训练位置和各 rank RNG 的可恢复 checkpoint；
评测输出绑定数据、adapter、prompt 与生成参数指纹；Action/Scenery 分开评测并支持
Base、旧 SFT、新 SFT 的 paired 汇总。当前 `/tmp` 缓存已有 17,869 个视频，v2 共需
28,099 个，还需生成 10,230 个。训练尚未启动。

full-v2 首次缓存服务在任何新编码前失败：统一 checkpoint root 指向了只含 Qwen 与
MPP shards 的 `/tmp` 副本，缺少 BLIP2 视觉权重。现已拆分视觉权重根目录与快速 MPP
投影根目录，并用真实单段 Scenery 编码验证通过。完整故障链见
`docs/data_download_and_full_v2_bringup_20260928.md`。

## 13. 2026-09-28 至 09-29：full-v2 训练与评测完成

前节“训练尚未启动”为当时状态；本阶段已完成增量缓存及一个 epoch 的四卡训练。
`outputs/mpp_lingoqa_full_v2_lora_1epoch/summary.json` 记录 completed=true，
25,389 optimizer steps、101,553 micro steps、26,017.31 秒，rank 0 峰值 allocated
16.04 GiB。最终 adapter 保存于该目录的 `final_adapter`。

| 集合 | QA | Base Judge | 旧 SFT Judge | 新 SFT Judge |
|---|---:|---:|---:|---:|
| Action dev | 3,120 | 37.05% | 43.08% | 48.24% |
| Scenery dev | 1,200 | 24.92% | 25.33% | 59.33% |
| 合并 dev | 4,320 | 33.68% | 38.15% | 51.32% |
| 官方 evaluation | 500 | 42.20% | 37.40% | 44.60% |

新模型官方集 token F1=0.313187、normalized exact=0.004。相对 Base 修复124、
退化112，净+12题；相对旧 SFT 修复99、退化63，净+36题。使用修正后的统一官方
Judge 协议，不与早期错误评分协议混比。dev 来自训练来源的留出视频，不用于梯度训练。

产物：`outputs/full_v2_eval/report.json`（dev 分来源），
`outputs/full_v2_eval/new_eval/scored/`（官方 evaluation）。新结果支持数据覆盖改善，
但不支持把 Judge 上升直接等同于获得可靠的时序动作理解。

## 14. 2026-09-29：三个动作诊断完成及代码更新

固定沿用 Action dev 的48个同问句视频，比较原问 A、等义改写 B、B加格式要求 C。
Base、旧 SFT、新 SFT 各新增96条回答并完成评分，同时复用各48条原问回答。

| 模型 | A | B | C |
|---|---:|---:|---:|
| Base | 0/48 | 0/48 | 0/48 |
| 旧 SFT | 9/48 | 9/48 | 8/48 |
| 新 SFT | 12/48 | 13/48 | 11/48 |

新模型 A→B 修复2/退化1；B→C 修复2/退化4。格式要求会改变内容，但整体净变化小，
不能将全部动作失败归因于问句措辞。相同问题跨48视频的回答有变化，但变化不等于正确，
也不能证明时序理解。没有采用错误的视频置换总准确率作为视觉依赖指标。

Codex 逐一检查240帧及新模型144条回答，分开记录动作、原因、参考可观察性。A原问
动作与参考完全一致12、部分一致11、冲突15、未回答对应维度10；原因完全一致11、
部分一致9、不同/冲突28。参考视觉支持：明确8、部分33、不确定5、疑似问题2。
这些不是人工真值；没有把难以验证的参考标签自动当成脏数据，也没有删除或修改数据。

新增 `scripts/diagnose_full_sft.py` 与 `scripts/run_full_sft_diagnosis.sh`：包含固定
dev导出、数据及运行指纹核验、三条件生成评分汇总、跨视频统计、复核标签汇总、
联系表和内嵌图片的离线 HTML。通过编译/语法检查及实际GPU推理评分验证。

产物目录 `outputs/full_v2_action_diagnosis_v1/`，包括 report、case_predictions、
visual_audit、review.html。专项文档为 `docs/full_v2_action_diagnosis_20260929.md`。

下一轮优先在训练 split 做动作/原因对照与保真问句改写，兼顾速度、车道、方向等维度，
核验信号归属和后段状态证据。保留 full-v2 基线，不因48题立即推翻架构、不盲目增加
epoch、不直接上未验证教师的 OPD/RL。本轮仅诊断，未训练；dev复核不得转作训练标签。

## 15. 2026-09-30：实验与代码归档审计

对 full-v2 数据、缓存、训练配置、五个中间 checkpoint、开发/官方评测、48段诊断、
离线页面和相关代码进行只读核验。补齐了实际训练超参、epoch dev loss、checkpoint
内容与恢复验证边界、adapter identity、输入数据指纹及关键脚本 SHA-256；同步修正
`implementation.md` 中仍写作“缓存未完成/训练前”和只列首轮退化的过时状态。

核验确认诊断三模型均96/96生成并评分，页面有48段和240张内嵌帧，逐题产物共有
432条回答；official eval逐题重算仍为Base 211、旧SFT 187、新SFT 223。未启动新
训练，未改权重、数据或既有结果。完整清单见
`docs/experiment_code_result_archive_20260930.md`。

## 16. 2026-09-30：视觉投影层联合 SFT 路径实现

针对 full-v2 在官方 evaluation 上仅由 Base 42.2% 提升至 44.6%、且 Action 诊断仍有
大量方向、车道、信号归属和动作原因错误的问题，完成第一优先级改造：保持 EVA-ViT、
Q-Former、Qwen base、5 帧输入及 LingoQA 数据不变，将缓存边界从 MPP 投影层之后移到
Q-Former 输出，使官方初始化的 `llm_proj(768→4096)` 能与 Qwen `c_attn` LoRA 接受
同一个答案交叉熵监督并联合更新。

新增 `qformer` 缓存协议，张量形状为 `[frames,32,768]`，与原有 `projected`
`[frames,32,4096]` 缓存分目录并严格区分；旧缓存和旧训练入口仍保持兼容。训练入口新增
投影层独立学习率、梯度和参数变化记录；每个可恢复 checkpoint、epoch 产物与最终产物
都同时保存 PEFT adapter 和 `projector.pt`，恢复训练会一并恢复投影权重、优化器和调度器。
生成评测与真实视频入口也支持加载同一投影权重，避免训练与推理路径不一致。

新增统一入口 `scripts/run_projector_sft.sh`，覆盖缓存、缓存状态、四卡训练、训练状态、
Base/联合模型分 split 评测和评分。Python 编译、Shell 语法和全部 3 个本地测试通过；
新增回归测试确认答案损失对投影层产生非零梯度，且投影权重保存后可无损重载。本阶段
只完成代码与轻量验证，尚未生成全量 Q-Former 缓存，也未启动正式联合训练。

## 17. 2026-10-01 归档：同路径联合 SFT 重评与下一轮实现

Q-Former 同路径重评已经完成：Base / full-data LoRA / 联合模型的官方500题
Judge分别为43.0% /45.2% /46.4%；Action dev3120题为36.57% /48.27% /51.47%；
Scenery dev1200题为25.17% /59.75% /60.17%。这些是新同路径结果，不覆盖之前
projected 路径的历史记录。官方小集增益有限，不能把开发集收益推广为普遍读图提升。

本轮新增 train-only 10k 定向集构建器、已有 LoRA+projector 权重初始化入口、
固定48段同路径动作诊断、统一训练/评测 Shell 入口及回归测试。实际生成10,000条、
5,795段，所需6,295段缓存无缺失；100条视觉复核样本仅导出，尚未审核。
7个 unittest及编译/语法检查通过；当时的实现交付阶段未启动GPU诊断或训练，后续进展见第18节。
实现清单、限制和完整命令见 `docs/action_targeted_sft_20261001.md`。

## 18. 2026-10-01：10k定向续训完成、开发评测退化与48段诊断

03:22 UTC训练summary落盘：625次更新，训练耗时672.89秒，LoRA及投影参数均更新；
rank0峰值allocated16.05GiB。04:27/04:42分别完成Action3120题、Scenery1200题评分。
targeted相对joint的Judge正确率为51.47%→49.55%、60.17%→60.08%；
Action修复160/退化220，Scenery修复34/退化35，合计少答对61题。

05:01 UTC固定48段诊断报告完成：joint原问/改写/格式正确16/13/12，targeted为14/14/9；
两个模型全部三条件均判错29段。问句或格式变化没有稳定收益，仍需区分视觉错误、
动作推理、参考可观察性及Judge误判，尚未完成新回答的逐图复核。

采样复查：5000条重点原题中Action2316、Scenery2684；关键词分组没有按真实动作类别均衡，
保真改写没有新增视觉监督。此局限尚无独立消融，不能当作已证明的退化原因。
继续保留全量LoRA＋投影层联合模型。下一步计划为48段＋32条正确对照的强教师试验，
教师入口与蒸馏/OPD均未实现、未运行。本次只归档，未修改训练代码或实验产物。

详细时间、代码SHA-256、数据、配置、指标、失败案例、验证边界与产物路径见
[2026-10-01完整结果归档](action_targeted_results_20261001.md)。

## 19. 2026-10-01 06:22–10:06 UTC：视觉侧 SFT 与单教师 OPD 代码交付

根据下一阶段讨论，一起实现教师对照、Q-Former 视觉适配 SFT、单教师 OPD，
执行顺序仍是先验证 SFT 与教师，后决定是否运行 OPD。本节更新第18节“尚未实现”的历史状态。

06:22 UTC实际导出20,000条未改写的原始训练 QA、16,308段视频，Action/Scenery
为12,900/7,100；固定48条开发诊断加32条正确对照，另导出训练侧1000条教师候选。
train/dev/eval视频段隔离。未将开发诊断加入训练，也未继续使用退化的10k模型。

缓存新增 normalized EVA 阶段 [5,257,1408]，使 Q-Former query/value LoRA、query
tokens、projector 和 Qwen LoRA 可联合接受答案CE监督。保存、恢复、自由评测和视频问答
入口均接入视觉 adapter。教师对照支持同模型全图/全图加中央ROI，以及可选本地HF强VLM。
不同tokenizer的外部教师只用于对照，不直接进入当前token级OPD。

OPD采用学生原始token采样、固定教师同前缀分布、分块全词表JSD与独立CE回放；
共享冻结Qwen基座但保留独立teacher adapter/视觉分支。实现多卡梯度平均、完整checkpoint
和恢复、训练侧教师胜例/保留样本筛选及身份指纹。教师是否有效尚待实际对照。

通过15项unittest与4项原core函数检查、Python编译/Shell语法/入口参数检查。
真实MPP单卡短测输出 outputs/visual_opd_real_smoke_v1/summary.json：新视觉路径与旧
缓存最大误差0，四个可训练分支均有梯度并更新；采样19tokens含EOS，JSD=0.00239314，
教师权重不变，峰值allocated15.40GiB，耗时69.70秒。没有将短测当作准确率提升证据。
pytest缺失，未安装新依赖；改用现有unittest及原函数测试完成验证。

本轮未启动全量新缓存、正式四卡训练、教师质量对照或开发集自由评测；旧产物未删除。
完整架构、代码清单、数据指纹、验证边界、监视/训练/评测命令见
[视觉 SFT 与 OPD 实现记录](visual_sft_opd_20261001.md)。

### 19.1 补录：2026-10-01 12:28–14:58 UTC视觉续训与裁剪教师结果

时间依据为对应summary文件最后修改时间（落盘时间），不推测任务启动时间。
12:28:47，20k视觉续训完成1250次更新，耗时1582.34秒，峰值allocated16.25GiB；
LoRA、projector、query tokens均有更新。14:09:25/14:25:39，Action/Scenery评分
分别为50.2885%/60.0%，相对joint修复123/退化160、修复29/退化31。
14:55:56/14:58:01，80条诊断的完整帧/中央裁剪教师为60.0%/42.5%。
模型参数确实更新，但未获得开发集收益；裁剪观察未构成更强教师。
完整产物路径见 [10月2日补录归档](answer_opsd_events_20261002.md)。

## 20. 2026-10-01 23:20 UTC归档：完整五帧答案条件OPD与匹配SFT对照

更新第19节后续状态：读取已有visual开发对比，Action3120题修复123/退化160，
Scenery1200题修复29/退化31，相对joint合计少答对39题。80条开发教师诊断中，
joint完整帧48/80正确，full+中央裁剪34/80正确。不能据此把中央裁剪作为更强教师；
也不能称为OPD正式训练失败。保留joint LoRA＋projector初始化，旧产物未删除。

本轮实现新独立入口 `scripts/run_answer_opsd.sh`：原始训练候选四次随机采样、参考答案
仅进入固定教师、Judge多样本稳定性筛选、来源配额和回放、学生on-policy token级JSD+
CE、匹配同数据同更新次数的纯SFT控制组、无hint开发自由评测及paired报告。
冻结EVA/Q-Former，复用完整五帧Q-Former缓存，只训练语言LoRA与投影层。
旧crop/EVA入口保留。新增评分进度文件，不改变官方Judge规则。

23:01:34 UTC生成2000条训练候选、2000段，Action1290/Scenery710；train/dev/eval
段隔离，缓存文件无缺失。尚未运行正式16000条教师/学生采样筛选，最终训练条数未知。

真实单卡工程短测完成2次更新、每步累积2条，4条rollout均EOS结束；教师权重不变，
学生LoRA/projector均更新，峰值allocated15.02GiB，耗时43.53秒。
结果在 `outputs/answer_opsd_real_smoke_v1/summary.json`，未保存正式模型。
23项unittest、4项原core函数及编译/Shell语法检查通过。新模式四卡正式训练、匹配SFT
及开发质量评测未运行；不把hint教师高分或短测loss当作实际读图提升。

代码清单、数据哈希、运行/后台/监视命令、损失定义及限制见
[完整五帧答案条件OPD实现记录](answer_opsd_20261001.md)。

## 21. 2026-10-02 01:30–03:43 UTC：采样评分完成与751条筛选

学生8000条生成于01:30:46落盘，01:34:05评分完成；教师8000条生成于03:40:31落盘，
03:43:49评分完成；03:43:50筛选manifest落盘。评分时间来自进度字段，生成/筛选时间
来自文件mtime。两边每题4条完整，无重复、缺失或截断。

| 训练候选范围 | 学生Judge | 答案条件教师Judge | 差值 |
|---|---:|---:|---:|
| 全部2000题×4 | 47.3250% | 70.6375% | +23.3125 pp |
| Action1290题×4 | 45.7364% | 78.8760% | +33.1395 pp |
| Scenery710题×4 | 50.2113% | 55.6690% | +5.4577 pp |

教师4次均正确751题，学生4次均正确439题；按每题通过次数比较，教师更多1164题、
更少354题、相同482题。选出751条=543条教师稳定正确且学生不稳定+208条双方稳定
正确；其中学生4次全错108条，余435条偶有正确。原QA与帧路径不变。

Action614条占81.76%，Scenery137条占18.24%，未保持原64.5%/35.5%比例；原因是
Scenery稳定教师样本不足，当前代码在来源不足时允许其他来源补齐。应分来源评测。
部分教师出现答非所问（天气题回答交通状况），不能把hint成功率当作独立读图能力。

运行中出现 `resume_download` FutureWarning及`TypedStorage`弃用警告。随后评分和
教师生成均推进，未因此失败；无需据此改依赖或重跑。训练前候选、结果及原始身份见
`datasets/processed/lingoqa_answer_opsd_v1/train.manifest.json`。

## 22. 2026-10-02 06:35–06:43 UTC：SFT控制组和OPD训练完成

两者同751条、四卡、累积4、1epoch、LoRA/projector LR=2e-6，共47次更新，每rank
188个micro steps。DistributedSampler将751条补齐为752次全局样本处理，非新增标签。

| 训练 | summary落盘UTC | 训练秒数 | rank0峰值allocated | 全局rollouts | 截断 |
|---|---|---:|---:|---:|---:|
| SFT控制组 | 06:35:40 | 50.05 | 14.89GiB | 0 | 0 |
| 答案条件OPD | 06:43:00 | 335.88 | 15.05GiB | 752 | 0 |

两者final均有adapter、projector、training_state及metadata，未发现failure_rank文件。
训练日志各47条。OPD为JSD+0.2CE，控制组为权重1的CE；控制组summary沿用通用
`objective: jsd`及`teacher_context: answer_hint`字段，不能据此认为控制组用了蒸馏。
实际`training_mode=sft_control`、JSD=0、rollouts=0，与代码分支一致；这是记录语义局限，
本轮只归档，不修改运行代码。47步小于save_every=50，因此只保存final属预期。

已完成模型：`outputs/mpp_lingoqa_answer_opsd_sft_control_v1/final`及
`outputs/mpp_lingoqa_answer_opsd_v1/final`。损失、权重变化、配置及文件身份见专项归档。
训练完成只证明执行成功，最终效果待开发集评测。

## 23. 2026-10-02 07:32 UTC：归档核对与评测进行中

本次将未充分记录的视觉续训完成时间、OPD代码身份、警告、采样筛选、正式四卡训练
补入原时间线，更新README和OPD说明顶部，修正主文档仍停留在9月30日的当前状态摘要。
保留历史事件原始时态，通过后续条目更新状态。

核对快照中 `outputs/answer_opsd_eval_v1/joint_action/predictions.jsonl`已有2626条，
未评分；Scenery、SFT控制组和OPD评测尚无产物。最终report尚不存在，不能补写未得到
的准确率。按截至此快照的已知代码、对话故障和产物，本轮归档完成；进行中评测的后续
结果仍待下一事件记录。详细文件清单和SHA-256见
[2026-10-02完整事件补录](answer_opsd_events_20261002.md)。

## 24. 2026-10-02：答案提示OPD完整开发结果与下一步决策

`outputs/answer_opsd_eval_v1/report.md`已生成。Action3120题：joint51.5064%、
SFT50.9936%、OPD50.5128%；Scenery1200题：60.6667%、60.5833%、60.2500%。
合计joint2335/4320=54.0509%，SFT2318/4320=53.6574%，OPD2299/4320=53.2176%。
OPD相对joint修复170/退化206，净−36；相对SFT净−19。该轮没有超过joint，不扩大
当前答案提示OPD。该joint重评数值与旧快照有差异，比较采用本轮同路径结果。

对原2000训练题的学生四次采样重新梳理：491题全错，439题全对，1070题混合。
决定利用其中参考支持的明确事实差异进行离线偏好纠错，保留同数据/CE权重的SFT对照；
不将混合采样组数视为可靠偏好对数，也不将训练侧pass@4解释为测试准确率。

## 25. 2026-10-02 13:20–13:26 UTC：偏好流程实现及数据准备

新增`preference.py`、数据筛选/参考评分/训练/报告和Shell入口等7个实现文件，以及
`tests/test_preference.py`。DPO使用固定joint无hint参考、回答log-prob求和及EOS；
参考分数离线缓存，训练仅驻留学生。两组CE权重一致、复用原Q-Former缓存及四卡梯度
同步/累积。为断点恢复保留LoRA/projector原始fp32值，避免经bf16转换后舍入。

13:20:40 UTC首次构建120对＋40回放，整体来源134Action/26Scenery。发现回放没有
补偿偏好对来源偏斜，改为按整体来源缺口回放；首次构建保留在
`outputs/preference_prepare_initial_unbalanced_v1`。13:23:53 UTC正式构建仍为120对＋
40回放，整体108Action/52Scenery，偏好对本身108/12。默认数据路径
`datasets/processed/lingoqa_preference_v1`；没有重写原始full-v2问答。

严格文本规则只覆盖转向62、ego动作25、信号颜色11、车辆类型22。120对均未声称人工
视觉审核；内嵌五帧的`review_selected.html`可供检查，并支持复核JSONL排除候选。
10项新增测试及33项全项目unittest通过，Python/Shell语法检查通过。

首次GPU参考评分因沙箱CUDA不可用退出，错误为`No CUDA GPUs are available`，
主机GPU空闲；已在沙箱外申请并重试，日志分开保留。正式DPO/SFT训练尚未启动，
真实GPU验证完成后另补条目。数据哈希、默认30步预算、后台命令和可恢复断点见
[2026-10-02偏好训练记录](preference_dpo_20261002.md)。

## 26. 2026-10-02 13:29–13:46 UTC：偏好流程真实GPU验证完成

13:29:39参考概率缓存完成160/160，峰值14.52GiB；文件
`outputs/preference_reference_v1/reference.jsonl`的SHA-256为
`602e5360143ce5d1fc80c3288208fc0a67cc24bbceeba8e3b50367dc47e4a914`。
13:40:21 DPO、13:42:37 SFT对照分别完成四卡、累积4、两步验证。每rank8个micro
steps，两者LoRA及projector更新确认，四卡参数校验差均为0，rank0峰值分别14.92/14.79GiB。
短测保存在独立`preference_smoke_4gpu_v1`和`preference_control_smoke_4gpu_v1`目录。

13:46:29从第一步断点恢复的DPO完成第二步，保存最终bundle，四卡一致。参考初始
概率核对通过每token差不超过0.02的检查，但不是逐位一致。首次DPO损失约0.6939，
接近log(2)，不能据此判断任务正确率提升。正式30步DPO/SFT及held-out评测均未启动。

## 27. 2026-10-03 00:07 UTC：续查恢复差异及交付状态

读取前日CPU权重对比结果：连续与恢复运行的64个adapter张量最大差1.65333e-6，
projector最大差1.55112e-6。功能恢复成功，但未达到逐位复现；没有将其归因于已证实
的具体数值内核，也不宣称对准确率无影响。比较记录保存在
`outputs/preference_resume_smoke_4gpu_v1/resume_comparison.json`，正式默认30步试验
建议连续执行。全项目33项测试、Python/Shell语法通过；进度脚本在真实短测产物上通过。

本轮补齐数据构建、训练、固定参考、同预算控制、开发报告、监视及恢复命令，详见
[偏好流程说明](preference_dpo_20261002.md)。未改写原数据/历史实验结果，没有继续扩大
OPD或启动正式DPO。归档包括此前未收尾的GPU执行记录，不虚构尚未获得的任务收益。

## 28. 2026-10-03补录：正式偏好对比完成与训练规模复核

此前第27节描述的是当时交付状态。后续`outputs/preference_eval_v1/report.md`已经完成：
joint / SFT控制 / DPO在4320道开发题上分别54.19% / 53.73% / 53.61%。DPO修复83题、
退化108题、净−25题；视频bootstrap 95%区间为[−1.29,+0.12]个百分点，未证明收益。
不能把驾驶项目的30步离线DPO称为已经充分验证在线GRPO。OPD仅751条47次更新；全量
joint SFT为406,212条25,389次更新；局部Q-Former适配仅20k条1250次更新。

复核指出训练FP32投影与历史评测BF16投影加载不同，小更新可能被舍入；需要统一路径
重评，尚未证明其解释了性能差异。决定下一轮采用完整现有Q-Former、projector和语言
LoRA联合SFT，使用原始100k QA、2epoch，不从退化模型继续训练。

## 29. 2026-10-03：完整Q-Former实现与100k数据构建

新增full/frozen/lora视觉桥模式及完整权重格式，训练、断点、自由评测和视频问答入口
接通。新精度协议统一FP32视觉桥/projector/LoRA加载，冻结语言基座BF16，Q-Former AMP。
训练日志改为所有rank与梯度累积窗口的平均loss；新增各分支梯度和完整桥参数更新记录。
历史精度协议仍可用。EVA缓存summary此前将hidden_size误记为4096，本轮修正为1408，
不改变已有缓存张量。

09:27:35 UTC导出100k QA，Action64500、Scenery35500，覆盖全部27599训练片段。每段
四题无法满足Scenery来源配额，采用覆盖优先逐轮抽样，自适应上限11题。dev/eval保持
原视频隔离。EVA缓存初查缺11291段，尚未启动全量补缓存或正式训练。

CPU测试和四卡短测记录、完整实现清单及命令见
[完整Q-Former联合SFT记录](full_qformer_sft_20261003.md)。GPU短测独立于正式训练目录，
验证完成后补录真实结果；本节不声称已获得新的模型效果。

### 29.1 2026-10-03 10:47–10:56 UTC：完整桥真实四卡验证完成

37项unittest和编译/Shell语法通过。10:47:15四卡两步完成，完整桥1.052亿参数、语言LoRA
839万参数及投影315万参数参与训练，各分支均有非零梯度和权重更新，rank checksum差0。
rank0峰值allocated16.93GiB、reserved22.23GiB；其他rank的reserved峰值23.08GiB。
10:54:44重新加载完整权重后2条自由生成完成，峰值allocated15.23GiB。
10:56:42从第1步恢复至第2步成功，rank checksum差0。

GPU恢复与连续训练不逐位一致：桥/投影/语言LoRA最大参数差约1.65e-5/8.09e-6/8.44e-6。
来源未隔离，不把功能恢复等同于精确轨迹复现；完整CPU权重重载测试通过。
`outputs/full_qformer_resume_4gpu_v1/verification.json`保留比较证据。正式100k训练和补齐
11291段EVA缓存仍由用户命令启动，尚无新的开发或官方准确率。

## 30. 2026-10-03正式运行，2026-10-04补录：100k完整Q-Former结果

时间为产物mtime（UTC落盘时间，不是推定启动时间）：16:40:22训练summary；20:40:36开发报告；
23:40:56 epoch-02官方评分。训练2epoch/12500更新/50000每rank微步，训练阶段18559.48秒，
约5小时9分。rank0峰值allocated16.94GiB、reserved22.23GiB，参数checksum跨rank差0。
完整Q-Former、query tokens、projector、语言LoRA均有更新。12500步连续且loss/梯度有限；
两epoch平均训练loss0.46956/0.44522，64条teacher-forced dev loss0.50907/0.50751。

| 模型 | Action3120 | Scenery1200 | 开发4320 | 官方500 |
|---|---:|---:|---:|---:|
| 同精度joint | 51.31% | 60.58% | 53.89% | 47.2% |
| epoch-01 | 54.42% | 62.42% | 56.64% | 未测 |
| epoch-02 | 54.49% | 64.33% | 57.22% | 51.0% |

epoch-02相对joint开发修复387、退化243、净+144题/+3.33pp；400视频bootstrap95%区间
[+2.09,+4.67]pp。官方修复45、退化26、净+19题/+3.8pp；两者都正确210、都错误219。
100视频bootstrap10000次seed42区间[+0.4,+7.2]pp；27视频净改善、14退化、59持平。
官方Token F1为0.313405→0.323456；exact0.004→0.006。官方当前动作100题42%→48%；
按问句关键词统计交通灯38题44.74%→60.53%，这些不是新官方任务标签。

收益跨开发/官方方向一致；不等同于全部事实/因果正确。部分Judge接受遗漏变化过程或动作
不完全一致的回答，仍有灯色、障碍和通用加速模板退化。文本对照未冒充逐帧视觉审核。
官方500已反复查看；此轮与train/dev视频无重叠，但不是从未接触的最终测试。
统一精度后的joint47.2%不能与旧路径46.4%混用计算收益；本轮同时改变训练范围和预算，
不能把全部收益单独归因于Q-Former。

产物：`outputs/full_qformer_eval_v1/report.{json,md}`、`joint_eval/scored/summary.json`、
`epoch-02_eval/scored/summary.json`、`joint_vs_epoch-02_eval.json`。

## 31. 2026-10-04：全量原始QA规模实验实现与准备

回顾10k定向续训净−61题和20k局部视觉适配净−39题，决定扩大当前有效完整Q-Former配方，
不重新构造关键词纠错集。旧joint为相同起点；全量406,212问×1epoch、预期25389次更新，
对照当前100k×2epoch。新增视频数为0，增加同视频问答覆盖和总训练更新。

新增`run_full_qformer_scale.sh`、`report_full_qformer_scale.py`及对应测试；准备器增加
`--all-train`保留全部源行，trainer恢复元数据保留初始权重指纹。三模型报告直接校验和复用
旧joint/100k预测，主对照为100k_vs_full；按dev点估计选模、平局保留100k，另报区间。
仅选中full后生成新官方500，官方评分不参与重新选模。

00:14:39 UTC全量manifest落盘，406212问/27599训练段，Action262005、Scenery144207，
实际单视频最多65问。train SHA与full-v2完全相同，dev/eval与100k轮哈希相同。
缓存存在性检查28099/28099、missing0；参考模型权重、评测数据和生成/评分协议核验通过。
未启动正式训练，没有新增效果结论。目录、启动/监视/评测命令见
[全量Q-Former规模实验](full_qformer_scale_20261004.md)。

本次10项相关unittest全部通过，Python编译及Shell语法通过；验证全量保留、启动参数、
模型/数据/评分身份、三模型比较、dev选模隔离、恢复初始化溯源及原小型桥梯度/重载。
测试Sentry弃用/Matplotlib临时缓存提示无失败。正式规模训练未启动，status显示
initializing_or_not_started是当前正常状态。没有停止其他任务、重算全部缓存或覆盖100k权重。

## 32. 2026-10-04 00:31 UTC：归档完整性核对与全量训练启动

本节检查现有scripts、模型代码、tests、docs、outputs/logs中的结果与失败文件，以及全量服务
journal。第30–31节已包含100k效果、全量代码与验证，漏项主要是刚发生的启动状态及代码
文件时间/身份索引。第31节的“未启动”指实现交付时；此处更新其后发生的事件。

### 32.1 时间与代码更新索引

| UTC时间 | 事件 | 依据/说明 |
|---|---|---|
| 2026-10-03 16:40:22 | 100k训练完成 | summary落盘；12500更新，见第30节 |
| 2026-10-03 20:40:36 | 100k开发报告完成 | epoch-02为57.22%，见第30节 |
| 2026-10-03 23:40:56 | 官方500评分/成对报告完成 | 47.2%→51.0%；+45/−26，见第30节 |
| 2026-10-04 00:12:15 | 全量数据模式与独立Shell入口落盘 | `prepare_full_qformer_sft.py`、`run_full_qformer_scale.sh` |
| 2026-10-04 00:14:39 | 全量数据manifest落盘 | 406212问，train指纹与源相同 |
| 2026-10-04 00:17:08 | 三模型报告与恢复初始化溯源更新 | `report_full_qformer_scale.py`、`train_sft.py` |
| 2026-10-04 00:18:22 | 规模流程测试文件更新 | `test_full_qformer_scale.py`；随后10项相关测试通过 |
| 2026-10-04 00:28:29 | 全量训练服务启动 | systemd/journal实际启动时间 |
| 2026-10-04 00:28:34 | 启动前参考结果核验通过 | references_verified=true，406212条 |
| 2026-10-04 00:30:42 | rank2完成加载、rank3开始加载 | journal中的model_loaded/loading_model |
| 2026-10-04 00:31 | 本次归档快照 | 服务运行中，最后一张卡继续读取模型分片 |

文件mtime是当前文件最后落盘时刻，不等于首次实现时间。以上代码快照SHA256如下；本项目
在父仓库中尚未形成这些文件的独立Git提交，哈希用于指认当前版本，不虚构commit记录。

```text
01f82a570f640041571c34953afa869e6d4e83cf5355b92a1ae47cdd9f568267 scripts/prepare_full_qformer_sft.py
8ae72dd2336122b8da612c0823eb8d709cfe9e21bffc5bd780569958b9faa8cd scripts/run_full_qformer_scale.sh
6459b3e1905c8030a70dacbc7d494c1ab069fa596abaeb2a89f0f356812844a8 scripts/report_full_qformer_scale.py
34f45f1f1e51f96c38a26a4d4fd0b6e1c02dff3bfb0d3888b7fcc1ee557b15fb scripts/train_sft.py
763c4925f33e53f26abb1f93b5026ef0a0ad5d9758b24932f5293a5c7ac32d7e tests/test_full_qformer_scale.py
```

### 32.2 失败与负向结果索引

| 时间/阶段 | 问题 | 归档位置与状态 |
|---|---|---|
| 09-26～09-28数据准备 | 下载中断、gdown跳过残缺ZIP、相对路径错误、缺pyarrow | [下载与full-v2记录](data_download_and_full_v2_bringup_20260928.md)，已记录 |
| 09-26～09-27启动 | 慢盘I/O、串行加载导致NCCL等待超时 | [启动问题](training_bringup_issues_20260927.md)、第3节，已记录 |
| 09-27首轮反向 | CUBLAS_STATUS_ALLOC_FAILED | `outputs/mpp_lingoqa_lora_v2/failure_rank_0.json`与第3.4节；不是新全量实验故障 |
| 09-27生成评测 | inputs_embeds路径缺bos_token_id | 第5节附近的生成修复记录，已记录 |
| 09-28缓存 | 临时目录缺视觉BLIP2 checkpoint | 下载/full-v2记录，已记录 |
| 09-30操作命令 | 用Python执行run_projector_sft.sh，set -euo pipefail触发SyntaxError | 本次补明：Shell脚本应由bash执行；属调用错误，无训练参数更新 |
| 历史监视命令，精确日期未保留 | watch默认sh执行[[，报[[: not found | 本次补明：显式bash或POSIX [；不是模型训练失败 |
| 历史重评Shell，精确日期未保留 | data: unbound variable | 10k结果文档已有提及；独立Shell入口避免复杂内嵌引用，未推测缺失时间 |
| 10-01定向10k SFT | 开发净−61题 | 第18节、action_targeted_results_20261001.md |
| 10-01局部视觉20k/中央裁剪教师 | 开发净−39题；教师80题60%→42.5% | 第19.1节 |
| 10-02答案提示OPD | 开发净−36题 | 第24节；未证明提升 |
| 10-02偏好参考评分启动 | No CUDA GPUs are available | `logs/preference_reference_v1.log`、第25节附近；环境限制后重试，已记录 |
| 10-02～10-03 DPO与恢复 | DPO开发净−25；GPU恢复非逐位一致 | 第27–29节，不能称为已充分验证GRPO |
| 10-04验证 | Sentry弃用/Matplotlib临时目录提示 | 第31节，10项测试通过，未记作训练失败 |

本次只读服务查询首次受沙箱user bus限制，随后获准查询成功；这是归档工具访问限制，
不是训练服务故障。新全量目录快照无failure_rank文件、无summary；服务MainPID=3606564，
训练rank0 PID=3606701，状态为初始化加载中。service的Result=success不代表训练已完成。
没有为运行中的实验补写尚未产生的loss趋势、准确率或成功结论。

## 33. 2026-10-04 10:56–12:48 UTC：全量完整Q-Former训练与评测完成

完成时间按产物mtime：训练summary 10:56:55，开发report 12:40:43，官方report
12:48:53。406,212问、1epoch、25,389更新、101,553个每rank微步；训练阶段
37,490.86秒（约10小时25分），allocated峰值16.95GiB，reserved约22.23GiB。
语言LoRA、Projector、完整Q-Former/query tokens均更新，多卡参数校验差0。

| 模型 | Action3120 | Scenery1200 | 开发4320 | 官方500 |
|---|---:|---:|---:|---:|
| 同协议joint | 51.31% | 60.58% | 53.89% | 47.2% |
| 100k×2epoch | 54.49% | 64.33% | 57.22% | 51.0% |
| 全量×1epoch | 58.59% | 63.83% | 60.05% | 54.2% |

相对100k：Action修复267/退化139、净+128；Scenery修复53/退化59、净−6。
开发合计修复320/退化198、净+122/+2.82pp，视频bootstrap95%区间[+1.65,+4.07]pp。
官方修复36/退化20、净+16/+3.2pp，区间[+0.4,+6.0]pp；官方Token F1
0.323456→0.319993，不能称所有指标提高。Scenery−0.5pp区间跨零，尚非明确退化。
官方按问句包含current action分组100题：48%→62%，贡献14道净改善；这是文本关键词
分组和Judge判定，不是人工确认动作分类准确率。先前抽读发现动作矛盾仍被Judge接受，
不能将整体提升直接等同于完全可靠的视觉/时序理解。100k与全量仍有209道官方共同失败。

结论：扩大问答覆盖与训练更新的组合有效；仍为相同27599段训练视频，新增视频数0，
不能单独归因于QA数量或优化步数。当前保留全量模型，后续追加一轮低学习率SFT。
旧OPD仅751条47更新、旧DPO仅120对30更新，不能据此排除新模型上的OPD/RL；
再次蒸馏需要可靠教师，在线RL需要可靠视觉事实奖励，本阶段不加入这些损失。

证据：`outputs/mpp_lingoqa_qformer_all_1epoch_v1/{summary,run_config}.json`、
`outputs/full_qformer_scale_eval_v1/{report,official_report}.{json,md}`。

## 34. 2026-10-04 13:58 UTC起：全量最佳模型续训实现与核验

新增`run_full_qformer_continue.sh`与`report_full_qformer_continue.py`，复用原训练器、
生成器、官方Judge和视频级成对统计。独立加载完整父模型三组权重，重建优化器和调度；
数据仍406212问，1epoch，四卡batch1累积4，学习率Q-Former5e-6、投影/语言LoRA2.5e-6，seed43。
预计新增25389步，合计50778步。保存6000/12000/18000/24000/final五候选。
原启动脚本新增完整初始化、seed/warmup透传与checkpoint评测；训练器新增父阶段身份、
累计步数和续训恢复参数核验。原实验默认训练范围、损失和历史输出不变。

报告使用固定Action3120/Scenery1200自由评测选模，平分保留父模型/较早候选；另报
paired区间，区间跨零时主模型推荐仍为baseline。选定候选后才做官方500题；未选模型
官方状态标记not_requested。监视增加更新时间和估计剩余时间，不冒充进程存活查询。

首次真实基线核验报evaluation data or model identity mismatch。定位为torch.save使用
不同文件名封装使epoch-01/final的两份pt文件SHA不同；实际参数逐张量相同。
增加元数据、键、shape、dtype和张量逐项等价核验后成功复用旧基线，未改写原权重或评分。
真实check结果references_verified=true，4320题Judge0.60046296296。

15项相关CPU测试、Python编译与Shell语法检查通过；覆盖模型完整重载、初始化路径、
续训恢复来源、不同封装等价、开发选中间检查点、官方选模隔离。新一轮正式GPU训练
尚未启动，没有续训效果结论。配置、启动、恢复、监视和评测命令见
[续训说明](full_qformer_continue_20261004.md)。

## 35. 2026-10-05 00:44–07:30 UTC：全量完整Q-Former续训完成及负向官方结果

时间取产物mtime：训练summary 00:44:14，开发报告07:22:37，官方报告07:30:54。
追加25389更新、101553每rank微步；训练37546.87秒（约10小时26分），累计完整Q-Former
50778更新。继续使用406212问及相同结构。训练完成不等于官方效果改善。

| 模型 | Action3120 | Scenery1200 | 开发4320 | 官方500 |
|---|---:|---:|---:|---:|
| 父模型全量1epoch | 58.59% | 63.83% | 60.05% | 54.2% |
| 续训6000 | 57.37% | 64.00% | 59.21% | 未请求 |
| 续训12000 | 58.85% | 65.42% | 60.67% | 未请求 |
| 续训18000 | 60.32% | 65.50% | 61.76% | 未请求 |
| 续训24000，开发选中 | 60.13% | 66.08% | 61.78% | 52.2% |
| 续训final25389 | 60.10% | 65.33% | 61.55% | 未请求 |

开发选中24000：修复230/退化155，净+75/+1.74pp，视频bootstrap95%区间
[+0.62,+2.83]pp。它与18000只差1道开发题。官方修复15/退化25，净−10/−2.0pp，
区间[−4.2,+0.2]pp跨零；不能据此断言显著退化，也没有进一步改善证据。
官方Token F1由0.319993到0.317591。继续保留原54.2%模型作为后续实验参考。

这说明本轮相同数据追加训练的开发收益未转化为官方收益，不足以证明模型结构到顶，
也不能直接证明输入数据错误或所有SFT/OPD/RL路线无效。下一阶段验证外部视觉教师
的实际纠错优势，并通过同预算原答案SFT对照判断监督内容的作用。

证据：`outputs/mpp_lingoqa_qformer_continue_1epoch_v1/summary.json`、
`outputs/full_qformer_continue_eval_v1/{report,official_report}.{json,md}`。

## 36. 2026-10-05：外部视觉教师与离线答案蒸馏完整入口

新增`generate_vlm_teacher.py`、`teacher_distillation.py`、`run_teacher_distillation.sh`、
`mpp_video_qa/distillation.py`和10项测试。扩展`data.py`/`train_sft.py`目标来源，
保留原始GT，训练恢复时核验目标模式。旧SFT默认仍读取原始答案。

学生固定为原54.2%的完整三组权重。教师候选默认复用本地Qwen3-VL-4B-Instruct及
chartvlm环境，未修改其他项目。五帧原图与原问题独立生成，教师无参考答案提示。
开发600问/400视频，Action433/Scenery167；训练候选10000问/9858视频，
Action6450/Scenery3550。真实样本准备、划分隔离和旧学生结果复用通过。

教师判定使用视频级成对区间，未结束回答不计为可用正确答案。达标后在train生成
双方答案、选择教师正确学生错误的目标，并混入约30%原答案回放；原答案SFT对照
和蒸馏使用相同样本、三组初始化、3epoch与更新预算。完整开发4320问选择模型，
只对选中模型评官方500问。此次实现是离线答案蒸馏，不改写成OPD/RL实验。

核验发现并修复：当前AutoProcessor忽略顶层像素限制，可能保留16M像素/图默认值。
改为显式设置图像处理器实际size；1920×1080五帧测试每帧实际965632像素，符合1M上限。
首次GPU smoke受沙箱CUDA不可访问影响而回退CPU，已停止；新增CUDA检查并获准GPU
单条重跑通过。产物`outputs/teacher_distillation_backend_smoke_v1`为1/1、16token、
finished=true；这只验证接口，不证明教师答案正确或优于学生。

新增10项CPU测试通过，既有15项Q-Former/全量/续训回归测试通过；覆盖完整流程连接、
参考答案不进教师提示词、原始参考不被覆盖、截断过滤、同预算和官方选模隔离。
正式600条教师对照及后续10k候选推理、两组训练尚未启动，没有新增准确率结论。
完整启动、后台运行、监视与恢复命令见[教师蒸馏说明](teacher_distillation_20261005.md)。

## 37. 2026-10-05 10:33 UTC：600题外部教师对照完成

`outputs/teacher_distillation_v1/teacher_report.json`落盘。600问/400视频，Action433、
Scenery167。学生353/600=58.83%，Qwen3-VL-4B零样本教师254/600=42.33%；
修复82/退化181，净−99/−16.50pp，视频bootstrap95%区间[−22.02,−11.24]pp。
Action55.43%→38.57%，Scenery67.66%→52.10%；TokenF1为0.6416→0.1237。
教师生成约1243秒，599/600正常结束、无空答案。

检查代码和结果：题目ID/参考答案对齐，传入完整五帧；教师没有读取GT，没有加载Chart
训练adapter。旧模板实际换行/图像占位符正常。教师平均5.38词，374/600答案不超过3词，
同时存在遗漏解释及动作/对象与参考冲突，尚未逐图确认全部分歧。
原因关键词组202题，学生96对、教师56对。不能把低分全部解释成答案长度问题。

此轮是教师资格对照，未训练新的学生。没有证明蒸馏算法无效；现成通用4B教师缺少
驾驶领域适配，当前设置没有总体优势。旧总体区间门槛过于单一，后续报告分别描述总体
优势与潜在局部纠错；训练监督仍需验证质量。600道开发题不转作训练样本。

## 38. 2026-10-05：教师任务适配代码与验证

新增`driving_complete_v2`、`teacher_adaptation.py`、`run_teacher_adaptation.sh`与6项测试。
扩展`generate_vlm_teacher.py`支持multi_image/video、视频元数据和实际视觉token记录；
扩展`teacher_distillation.py`透传教师独立的输入/提示/时间设置并核验身份。

旧A和MPP学生结果只读复用，新B/C使用原600道题。B为完整驾驶说明＋五图，C为同说明＋
原生视频。相对时间默认按论文四秒五帧协议设1秒间隔；不冒充逐样本采集时间。
当前Processor对奇数帧的resize预算先取整；显式重复末帧补到6帧，并使用[0,1,2,3,4,4]
计算时间，可保证空间预算。视频不二次抽帧，时间块为0.5/2.5/4.0秒。

14:09 UTC前完成16项CPU测试。真实5帧Processor验证：B为5×[1,48,78]，C为[3,48,78]，
空间尺寸相同；视觉token4680/2808，视频具有时间压缩，不能称为等token比较。
报告包含A→B、B→C、student→B/C、分组、净修复区间、完成率、长度、成本与40条分歧HTML。
整体教师较弱仍能生成报告，潜在修复不自动当作已审核训练标签。

单条GPU检查首次在沙箱内因CUDA不可访问退出，未发生CPU权重加载；已申请GPU环境重试。
14:28 UTC B单条完成，14:35 UTC C单条完成，均EOS结束，23/47新token，约3.70/4.23秒。
产物`outputs/teacher_adaptation_smoke_v1/{multiimage_complete,video_complete}`；不据单条判断准确率。
实际600问/400视频的A/学生结果复用和身份核验也通过，验证配置保存在
`outputs/teacher_adaptation_validation_v1/run_config.json`。弃用/greedy采样参数警告不影响推理。
正式600题B/C对照尚未启动，无新增教师正确率或蒸馏训练结果。
命令、限制和完整输出说明见[教师任务适配记录](teacher_adaptation_20261005.md)。

## 39. 2026-10-05 15:41 UTC：教师任务适配 B/C 对照完成

时间取 `outputs/teacher_adaptation_v1/report.json` 的 mtime。
同一600道开发题／400视频：MPP学生58.83%、旧零样本教师42.33%、完整说明五图B
40.33%、完整说明原生视频C38.83%。B相对学生修复73／退化184、净−18.50pp；
C修复72／退化192、净−20.00pp。B/C相较旧教师的变化区间跨零，未得到教师优势。
更新任务说明改善TokenF1不等于事实回答能力提高。没有训练新学生，不把这600道开发
诊断题改为训练监督，不沿用B/C作为新的概率蒸馏教师。

## 40. 2026-10-05 17:27 UTC 起：反馈条件同模型 OPD 独立实现与数据准备

学生与冻结教师均从 `mpp_lingoqa_qformer_all_1epoch_v1` 初始化；学生该阶段使用
406212原始问答、25389更新，已有官方54.2%。教师权重不再额外训练，而是在训练侧
接收原参考答案和当前学生草稿。旧751条指旧学生训练数据，不指教师训练数据。
保留五帧完整画面，弃用中央裁剪和弱外部教师，学生推理不接收参考答案。

新增 `feedback_opd.py`、`feedback_opd_common.py`、prepare/probe/train/report/smoke
脚本及 `run_feedback_opd.sh`，对应测试 `tests/test_feedback_opd.py`。
扩展共享 `opd.py` 的 greedy教师生成、FP32教师LoRA精确重载和初始权重核验；
`data.py` 新增可选五帧原图缓存核验。旧默认行为不变，未修改其他项目或历史权重。

实际准备10000问／9858视频，Action6450、Scenery3550；其中2000问供每题两次采样。
原始问答未改写，train/dev/official视频、题目、图像路径交集为0。数据目录为
`datasets/processed/lingoqa_feedback_opd_v1`，train SHA256
`038eb2f003990bda26a1a12deff3c3299cc6b7ec871f6167f376ac37432c6d3f`；
开发3120+1200、官方500维持原划分。EVA缓存已存在，无额外数据下载。

训练完整Q-Former＋projector＋语言LoRA，共享冻结Qwen base、独立冻结教师视觉分支。
每次访问生成当前学生token；学生完成且Judge判错、教师纠正完成且Judge判对时，
额外计算同一学生token／相同生成前缀的全词表token-mean JSD。教师纠正文本不是
OPD目标序列。所有样本仍执行原答案CE；SFT对照CE权重／数据／初始化／LR／更新
预算相同。四卡batch1、累积4、1epoch默认625更新，不沿用旧47步或默认200步上限。
checkpoint保存三组权重、优化器、scheduler与各rank RNG；恢复时教师保持原始快照。

完成训练侧参考隔离、动态门控、teacher切换后恢复policy再反向、JSD／CE分别反向，
固定次序多卡梯度归约、实时PID／错误／进度／ETA、严格同协议开发选模和官方验证。
Judge仅是文本代理，不是逐图事实验证；诊断导出逐token分布及五帧内嵌HTML。
未把自动概率变化或参考提示下的高分当成视觉事实标签。

## 41. 2026-10-05：反馈 OPD GPU 检查中的数值误判及验证结果

新增9项CPU测试通过，全部unittest回归共73项通过；编译、Shell语法、真实数据准备
和任意工作目录下监视入口通过。完整GPU初试因沙箱CUDA不可访问立即退出，没有
CPU回退；申请真实GPU运行。

原先用无反馈师生logits最大差值≤0.05核验初始化，在完整BF16模型上误报：先后
观察到0.25／0.171875／0.15625，而同一学生重复计算也有0.125–0.15625差异。
逐项核验Q-Former、projector、LoRA权重、输入IDs、mask、视觉／提示嵌入完全一致。
关闭fused SDPA的对照没有消除现象，不宣称已确定底层kernel根因。改用全部对应权重
与输入的精确身份检查，同时记录无反馈JSD作为数值基线，不强行扩大raw-logit阈值。
正式模型attention设置未改动。

最终 `outputs/feedback_opd_real_smoke_v1/summary.json` 完成1次机械更新：
JSD0.083420、CE0.104004，语言／projector／Q-Former的JSD梯度均非零；
三组被跟踪学生参数更新约2.5e-6、三组被跟踪教师参数变化0。峰值allocated17.466GiB，
耗时36.67秒（含加载）。无反馈师生JSD1.194e-5、同学生重复计算1.094e-5。

该24token回答截断，smoke为检验反向链路强制一次JSD，无Judge门控且未保存训练
权重；不是训练质量或准确率实验。正式流程仍跳过截断回答的JSD、保留CE。
仅完成单GPU机械验证，不冒充四卡长训／实际4000次教师诊断已完成。
正式probe、两组625更新及新自由评测未启动，无新准确率结论。
完整命令、恢复、监视、限制见[反馈OPD记录](feedback_opd_20261005.md)。

## 42. 2026-10-06 01:18 UTC：训练侧反馈教师 probe 完成与监督方向诊断

完成时间取 `outputs/feedback_opd_probe_v1/probe_report.json` 的 mtime。
2,000 道训练题、每题2次，共4,000份学生采样；学生正确2,148份（53.70%），
错误1,852份，教师参考＋草稿纠正通过1,364份（错误中的73.65%，全采样34.10%）。
剩余488份教师仍错。Action错误1,212／纠正856（70.63%）；Scenery错误640／纠正
508（79.375%）。无截断／上下文超长；1,364份成功纠正覆盖989道题。

40份token诊断、27道题中36份门控通过；34份整段sampled-token平均logP提高。
不能把这34份自动判为坏监督，因为措辞／正确词／EOS也在内；但存在具体方向词
反例：qid `e06aae4dbd5a12c0f8f8c4c9d449a3e9`、draw0，参考right、学生left、
教师新回答正确，而沿原学生前缀left概率约0.290→0.985。
另有正确方向：qid `1b343138ea578edeb34e4ec32af51a18`、draw1，参考decelerate、
学生accelerate，错误词概率约0.350→0.042。教师独立生成与teacher-forcing信号
不能等同；不能凭少量反例宣称所有OPD不行，也不能凭73.65%证明蒸馏正确。

反馈JSD0.086645；无反馈师生JSD5.0104e-6，raw logits最大差0.1875仍在已发现
BF16数值噪声背景下解释。此为训练侧教师诊断，不是新学生held-out准确率。
旧10k正式反馈OPD／SFT尚未训练，决定另建v2预算，不把probe当作正式训练。

## 43. 2026-10-06 03:26 UTC 起：反馈 OPD v2 的代码、数据与对照预算实现

按用户要求实现4万原QA×2epoch，暂不扩大到10万。
新增两教师条件的固定学生token对照、独立v2训练wrapper、固定步数自由评测和
每分支dev选模报告；扩展训练原参考CE＋门控JSD、检查点、源统计、最近20步ETA。
教师和学生保留同一406212-QA父模型，五帧完整观察、完整Q-Former＋projector＋
语言LoRA；没有重用中央裁剪或改成弱外部教师。

`reference_draft`保留原行为，新增`reference_only`不额外给完整错误草稿。
两种条件的教师纠正生成和概率评分分别保持相同prompt，但都仍使用学生completion
前缀；新模式不承诺消除错误前缀延续。默认v2为reference_only候选，正式训练前
先看200条同token对照，不从自动概率升降推断事实标签，也不自动按该指标选教师。

新数据 `datasets/processed/lingoqa_feedback_opd_40k_v2` 已实际准备、重复准备成功：
40000题，Action25800／Scenery14200，覆盖全部27599视频（24191／3408）。
source比例＋视频轮转，实际每视频最多5题；原问答未改、原dev4320／官方500保持。
train SHA256 `d885dc2f2c59e2a8ad4c01a8240e471ed3d04e4602e887dd7ef680febb770712`。
三类train-heldout交集为0，选中EVA缓存文件均存在，不下载新数据。

200条诊断复用旧probe缓存，129Action／71Scenery、各题一份固定回答；旧eligible101／
旧教师未纠正99，混合用于对照而非无偏纠正率统计。已核验父权重、缓存、问答与
学生IDs身份，产物`outputs/feedback_teacher_compare_v2/selection_validation.json`。
仅完成选择与身份验证，尚未完成正式200条双条件推理。

两分支四卡batch1、累积4、2轮各5000更新；OPD预期80000份当前rollout。
固定dev评1000／2500／5000步；SFT和OPD分别以combined-dev Judge选最优、同分取
较早步，最后父模型＋两分支dev-best都跑官方500，不拿官方分数选checkpoint。
同更新预算不代表同GPU时间，保存完整恢复状态，教师恢复时仍冻结在原父模型。

测试发现adaptive新manifest直方图int键在JSON重读后变str导致重复准备误拒绝，已
稳定为str键，实际重复准备通过。新增8项测试，全回归81项通过，Python3.8编译与
Shell语法通过。两条真实GPU条件检查首次沙箱不可见CUDA，未加载权重，随后申请
真实GPU权限重试。没有自动启动正式长训，无v2学生准确率结果。
完整文件列表、启动／监视／恢复／评测见[反馈OPD v2记录](feedback_opd_v2_20261006.md)。

## 44. 2026-10-06 03:39 UTC：v2 两条真实条件推理完成及最终代码验证

`outputs/feedback_teacher_compare_smoke_v2/probe_report.json` 完成，取实际mtime。
在空闲GPU0、完整父模型、原五帧EVA缓存上，复用2份训练错误学生IDs，分别评两种
教师条件。两条件各2份纠正均EOS完成且Judge通过，原学生序列的两条件分布对齐
成功，导出两份内嵌原图review HTML；推理部分约8.09秒（不含权重／Judge加载）。
无optimizer更新，无新的学生准确率结果；不根据两条判定reference_only质量已胜出。
原沙箱CUDA不可访问失败记录已由重试完成进度更新，前面的环境失败仍在本文归档。

增加固定检查点评测报告集成测试：两组各自dev选2500步，故意设置官方偏好final，
确认官方不改选模；两分支dev-best都进入官方对比，并拒绝变更过的权重。
最终新增9项，全回归82项通过；Python3.8编译、Bash语法、真实40k重复准备、200条
诊断选择核验、任意cwd状态监视、训练CLI／wrapper参数转发通过。
正式两组各5000步、正式200条对照与其自由评测没有自动启动；交付启动命令给用户。

## 45. 2026-10-06 04:20 UTC：200条固定学生错误的双教师条件对照完成

完成时间取 `outputs/feedback_teacher_compare_v2/probe_report.json` 的mtime
（04:20:00 UTC）。核验报告 `completed=true`、200/200份对照，两个条件均200份
逐token诊断。复用相同原学生token，Action129／Scenery71、各题一份；其中旧
eligible101、旧教师未纠正99。输入、父权重与缓存身份沿用第43节核验，无权重更新。

| 指标 | reference_draft：参考＋完整草稿 | reference_only：仅参考 |
|---|---:|---:|
| 纠正通过门控 | 103/200（51.50%） | 190/200（95.00%） |
| 教师回答仍错、CE-only | 97 | 10 |
| 在通过门控的回答中，原学生整段平均logP提高 | 81/103 | 5/190 |
| 教师纠正文本原样重复错误草稿（lower/strip比较） | 58/200 | 0/200 |
| 所有诊断token的平均JSD | 0.083550 | 0.061774 |

reference_only相对reference_draft配对修复90、退化3，净+87份纠正／+43.50pp。
该200条按source分层并混合旧门控状态选取，不是所有训练错误的无偏随机抽样；
不能将这里51.50%／95.00%与先前全1852错误中的73.65%直接混比或描述为学生
准确率提升。教师看过训练参考答案，95%仅指该诊断集的条件纠正通过率。

这些信号支持移除额外完整草稿后继续验证reference_only候选：教师较少照抄错误
草稿，纠正门控覆盖提高。但整段sampled-token logP升降不是逐词事实错误标签，
JSD大小也不是质量排名；不能从5/190宣称其余185份都得到正确视觉监督。
报告仍保持 `visual_supervision_direction_verified=false`。最终收益须由无参考提示
的学生自由评测，以及同预算SFT控制组对比确定。

产物保存在 `outputs/feedback_teacher_compare_v2/`：

- `run_config.json`、`selection_validation.json`：复用与选择身份；
- `predictions.jsonl`：逐题两条件纠正、Judge及逐token分布；
- `probe_report.json`、`probe_report.md`：完整指标；
- `review_reference_draft.html`、`review_reference_only.html`：内嵌原五帧的可视复核页。

报告SHA256：`328bf4986c265a38f2f0e5a8ca1362e459369a5b4bcf566ea9a8cf4f12670794`。
正式训练的run_config记录了这一同样的probe报告hash。此阶段没有新增代码修改，
没有把诊断题改为额外人工事实标签，也没有新学生准确率结论。

## 46. 2026-10-06 04:37 UTC起：正式四卡SFT控制组启动、检查点与进行中快照

`outputs/mpp_feedback_sft_control_40k_2epoch_v2/run_config.json` 于04:37:46 UTC
落盘。记录模式 `sft_control`，teacher_mode配置为 `reference_only`，40000原QA、
world_size4、grad_accumulation4、epochs2、total_steps5000，完整Q-Former＋projector
＋语言LoRA，FP32 trainables／BF16 frozen base。初始LR语言／projector各2.5e-6，
Q-Former5e-6；原答案CE权重1、warmup_ratio0.03，使用已完成200条诊断报告身份。
该控制组实际只执行CE，配置中的OPD字段用于匹配实验；rollouts／eligible／JSD为0
是预期行为，不是OPD门控失效。它也不是正在训练参考条件教师。

早先读取的04:42:34 UTC快照为180/5000步。05:04:24 UTC保存完整
`checkpoint-001000`，其下存在adapter、projector.pt、qformer_adapter.pt、
training_state.pt与metadata.json；不是仅有目录名或未完成的.incomplete产物。
这是首个固定dev评测候选，但此时尚未运行该checkpoint自由评测。

本次归档采用 `progress.json` 的05:15:56 UTC快照（其后文件会继续更新）：

| 项 | 观察值 |
|---|---:|
| 更新步数 | 1436/5000（28.72%） |
| 全局累计样本访问 | 22976（Action14796／Scenery8180） |
| 当前累积窗口平均CE | 0.394226 |
| 当前梯度范数（裁剪前日志值） | 4.177663 |
| 最近20步平均耗时 | 1.579655秒/更新 |
| 当前进程训练耗时（不含先前模型加载） | 2290.87秒 |
| 当时估计本控制组剩余时间 | 1.5639小时；不含后续OPD或评测 |
| rank0峰值allocated显存 | 16.9468GiB |
| 当前reserved显存 | 18.4863GiB |

rank0 PID1541600在归档时以只读进程查询核验存活，命令行与该实验一致。检查时
控制组尚无summary.json、未发现failure_rank报错产物；OPD输出目录
`outputs/mpp_feedback_opd_40k_2epoch_v2` 尚不存在，因此未观察到OPD开始。
若按train_pair启动，控制组成功结束后才进入OPD；不能将上面的1.56小时当作整个
双分支实验剩余时间。未终止／重启／改动当前训练，未开展持续监视。

静态追溯hash（不对持续更新的progress文件宣称永久hash）：

- 控制组run_config：`4f177dda19007852f5f17737406b35940708ab9e1aac01a4be1c83ed7b1f946f`；
- checkpoint-001000 metadata：`d79c94a79386c48e3997a0b8df06109ad82fbdd71527bf61b87830fc57166632`。

更新首页摘要、README和[反馈OPD v2文档](feedback_opd_v2_20261006.md)，保留第43–44
节“尚未启动”作为当时实现交付边界，不倒改历史事件。本次仅补归档，未修改训练
代码或数据。下一阶段仍为完成两分支5000步，固定1000／2500／5000步dev自由评测，
各分支dev选模后进行父模型＋两组dev-best的官方500对比；目前没有v2效果结论。

## 47. 2026-10-06 09:48–10:38 UTC：保留四卡的反馈OPD执行加速实现与验证

用户要求优化OPD速度，并明确尽量保持四张训练卡。本次不改40k原QA、2epochs、
四卡×microbatch1×梯度累积4＝全局batch16、5000次更新的预算；不改原答案CE、
固定reference_only教师、学生在线采样和通过纠正门控后沿学生原token计算全词表JSD
的目标。没有停止、重启或热更新当前旧训练，也没有新增正式八卡实验。

新增执行优化：

- `model.py`的answer-only CE和`opd.py`的answer-only completion logits：保留完整
  上下文Transformer，仅对监督位置计算全词表LM head；原路径仍可用。
- 新增`opd_runtime.py`：按dtype/device固定梯度分桶，复用通信缓冲；采样窗口分项
  计时；固定教师门控答案缓存。每次学生回答和评分仍重新生成，教师蒸馏概率每次
  重新前向，缓存不保存学生输出或教师logits。
- `feedback_opd_common.py`：Judge可放在每个rank自己的GPU，保持FP32和官方评分
  协议；必须提供等价验证报告，靠近零阈值的结果回退CPU复算。缓存身份包括固定
  教师条件及视觉特征文件身份，避免跨教师／特征版本误复用。
- `train_feedback_opd.py`：接入分桶、缓存、GPU Judge、answer-only路径、分项计时
  和降低日志flush频率；新增runtime_config及runtime_history，与严格实验身份分开。
  旧checkpoint恢复仍核验训练身份，恢复优化器、调度器、样本游标和随机状态，并
  兼容旧17项累计计数迁移到新20项；不允许更改world_size／预算后伪装续训。
- 新增`run_feedback_opd_fast.sh`、`report_feedback_opd_speed.py`及真实模型／Judge／
  四rank通信验证脚本；原wrapper默认路径不变。fast入口默认四卡本卡GPU Judge，
  可显式保留CPU Judge。v2报告增加执行配置及历史hash便于追溯。

更新`README.md`和反馈OPD v2文档，新增
[四卡加速实现与切换命令](opd_speed_20261006.md)。新增`tests/test_opd_speed.py`覆盖
CE/JSD与梯度、缺失梯度分桶、固定教师缓存与学生动态门控、文件身份失效、CPU
边界回退、严格resume身份和旧计数迁移等行为。

### 实测结果（不是学生准确率结论）

产物：`outputs/feedback_opd_speed_validation_v1/`。短验证使用其他空闲卡，
没有给正式训练增加GPU，也没有保存测速模型作为正式checkpoint。

| 完成时间 UTC | 验证 | 结果 |
|---|---|---|
| 10:04:09 | 128条Judge CPU/GPU FP32比较 | 最大logit差1.1832e-5，原始决策翻转0；1条接近阈值，CPU回退后翻转0 |
| 10:04:45 | 真实Qwen两条样本旧／新CE与JSD | logits最大差0，CE/JSD一致；三路梯度最低余弦0.999851，相对L2差约1.06%–1.72% |
| 10:12:17 | 单卡实际OPD短测速，每版2次预热＋20次测量，累积4 | 更新中位10.7488→8.4463秒，约1.2726倍、耗时减少21.4% |
| 10:14:35 | 四rank NCCL，128组小张量旧／新归约 | 每rank全部通过，最大梯度差0，collectives 128→1 |

Judge单次评分中位663.90→25.28ms，是本次主要实测收益来源。短测速allocated
峰值17.3904→18.0798GiB，GPU Judge会额外占用显存。四rank小张量通信均值
约10.17–10.21→5.88–6.31ms，不能外推为完整训练加速倍数。

两版从相同权重、样本顺序和初始seed出发，但BF16运算形状变化造成梯度与采样
轨迹分叉：含预热的88次访问中legacy eligible39、fast46，不是固定逐题计算量的
严格测速。本批全是不同题，教师缓存命中0，第二epoch缓存收益尚未实测；单卡
1.27倍也不是正式四卡吞吐或剩余时间保证。更没有新dev／官方准确率提升结论。

最终90项unittest和4个test_core函数全部通过，Python3.8编译、训练CLI参数检查、
Bash语法及状态入口通过。验证时沙箱隐藏CUDA、限制Gloo本地网络，改为获准宿主
GPU／本地通信后通过；环境没有pytest，未安装依赖，改用已有unittest及直接调用
函数测试。BF16梯度非逐位一致与短测速门控数不同均如实记录，不声称精确复现。

### 正式实验进度与切换边界

补归档：SFT控制组已完成5000步／80000次样本访问，summary训练耗时7922.75秒
（约2小时12分钟）。10:14:52 UTC只读快照中OPD为794/5000，最近更新14.7102秒，
当时估计剩余17.19小时；这是旧执行路径快照，不是fast版本结果。该时刻OPD尚无
完整checkpoint metadata；不能把控制组checkpoint用于恢复OPD。

交付命令以OPD `checkpoint-001000`为例：先等待其完整落盘，再由用户结束旧训练，
用fast入口从该checkpoint续训。若已到更晚完整checkpoint则替换路径；不得同时
对同一输出目录运行两组写入进程。新代码不会进入已加载旧实现的Python进程。
后续仍保留1000／2500／5000步dev选模和选定模型官方500评测方案。

## 48. 2026-10-06：完整恢复、SIGHUP中断与四卡fast训练完成

补齐第47节之后的已完成事件。SFT控制summary于06:49:49 UTC完成5000步／80000
访问，训练7922.75秒。旧OPD到达完整checkpoint-001000后，按用户选择切换四卡
fast入口；仍是40k×2epoch、global batch16、相同三路初始化与reference_only教师。
恢复优化器、scheduler、样本游标、RNG与累计计数，不是从第1000步权重另开新训。

首次nohup fast续训日志`logs/feedback_opd_v2_fast_resume1000.log`中launcher PID1927816
收到signal1／SIGHUP并终止子rank，在约1079步中断；不是loss或蒸馏公式报错。
随后用systemd user service `mpp-feedback-opd-fast-resume-v1`从同一完整1000断点恢复，
未把中断尝试当作额外有效训练，也没有同时运行同输出目录的两组writer。
日志中旧PID和中断PID的重复step，在速度分析中按最终有效轨迹区分。

OPD final summary于**22:41:35 UTC**落盘，completed=true、5000/5000更新：

| 项目 | 已完成值 |
|---|---:|
| 实际样本／在线学生rollout | 80000／80000 |
| 教师条件纠正尝试／通过 | 36382／34898（95.92%） |
| 全访问中蒸馏eligible | 43.6225% |
| 蒸馏token | 737446 |
| 学生截断／上下文跳过 | 0／0 |
| 固定教师纠正缓存命中 | 7718 |
| fast续训4000步耗时 | 38705.081秒，约10h45m |
| rank0 peak allocated | 18.104GiB |

语言LoRA、projector与Q-Former被跟踪参数均更新，教师被跟踪参数变化均为0。
教师95.92%为有训练参考提示时的条件纠正率，不是学生准确率。原前1000有效步约
15076.965秒，fast后4000步平均9.676秒／更新；旧有效1000步均值15.077秒，实际
区间约1.56倍。两个区间样本与门控不同，不是严格同任务逐步测速。合计有效训练
约14h56m，不包括模型加载、中断浪费和评测。第47节单卡1.27倍结果保留。

本阶段未再改变训练目标、数据或预算，执行配置与历史保存在运行目录；没有把
加速当作效果提升。对应artifact为`outputs/mpp_feedback_opd_40k_2epoch_v2/summary.json`
及final三路权重。此时训练完成，held-out收益仍待下一节自由评测。

## 49. 2026-10-07 08:06／08:32 UTC：反馈OPD v2开发选模与官方评测完成

时间取实际报告mtime；dev报告08:06:21，官方报告08:32:20。原五帧、legacy_v1
说明、FP32 trainables／BF16 frozen base、greedy96、batch1、原官方Judge协议一致。
检查1000／2500／final三点，SFT与OPD按4320题合并dev均选择final，不按官方500
反选检查点。各目录数据／权重身份及question_id逐项核验，31740份生成和评分完成。

| 模型 | Action3120 | Scenery1200 | 合并dev4320 | 官方500 |
|---|---:|---:|---:|---:|
| 全量SFT父模型 | 58.37% | 64.25% | 60.00% | 54.2% |
| 同预算CE控制final | 58.75% | 64.92% | 60.46% | 55.4% |
| 反馈OPD final | 58.33% | 65.17% | 60.23% | 54.8% |

OPD相对CE：dev修复80／退化90、净−10（−0.23pp），segment-bootstrap95%区间
[−0.82,+0.38]pp；官方修复10／退化13、净−3（−0.60pp），区间[−2.4,+1.4]pp。
父模型→CE的dev／官方净增+20／+6题，父模型→OPD净增+10／+3题；相关区间都跨零。
两份报告均不支持OPD独立收益，不能用教师纠正率、参数更新或loss证明提升。

同一步OPD相对CE的dev净变化1000为+0.21pp、2500为−0.14pp、final为−0.23pp，
不存在随着更多OPD步数而稳定提高的证据。父模型不是raw MPP；当前协议下父dev
60.00%与旧报告60.05%的小差异分别保留，不覆盖历史评测。官方500已多次查看，
这里不是新盲测。本阶段没有新的模型代码修正或数据修改。

报告位于`outputs/feedback_opd_eval_v2/report.{json,md}`及`official_report.{json,md}`。
三模型共同失败1495题（Action1124／Scenery371），成为下一节开发诊断的候选池，
不是新训练集。当前停止扩大同一OPD，先确认失败来自视觉、关系还是评测口径。

## 50. 2026-10-07 09:38起：开发集共同失败复核与观察验证完成

按用户要求先完成开发集验证。新增`scripts/validate_dev_failures.py`提供固定选择、
图像预览、ROI预览、固定权重推理、相同官方评分的汇总和离线HTML；没有改训练
forward、蒸馏目标、模型权重或原参考。新增4项输入／配对测试和Python3.8编译通过。

09:38确定80共同失败＋30共同Judge正确控制，按source／问句分层，110不同segment，
和40k训练segment交集0，原五帧均存在。并非全部4320题重评或1495题完整人工复核，
也非无偏失败比例抽样；相邻segment可能共享原录制场景。看完22张原五帧概览、7张
ROI预览，并补查关键原图。人工标记110题，置信度／存疑均保留，未改变官方标签。

80失败暂定类型：空间／目标关系20、对象／细节12、场景原因15、回答具体性11、
时序／动作时刻6、计数3、参考／Judge争议13。参考完整支持38、部分29、无法确认6、
存疑7；这些是Codex审核意见而不是因果金标。Case5/16原图amber→red修正了概览
阶段的静态颜色判断；Case91发现Judge放过左右反转，30正确控制并非全部事实正确。

固定CE控制final三路权重；原五帧重放、末帧重复、倒序均110份。重放103/110逐字
一致，Case19同义措辞引起1题Judge翻转，所以后续以新重放为基线。末帧6修复／
7退化，倒序7修复／6退化，无稳定净收益，不能当作可部署修复或“不会用时间”的证明。

28题（20失败＋8控制）ROI与双全图对照于10:09生成、10:14评分完成。每时间full+
full或full+ROI、同paired说明、相同320视觉token，未传参考或目标文本。人工框在
推理前预览冻结，目标离开视野时回退全帧，没有根据输出改框。结果：

| 同28题 | 原五帧 | 全图副本 | 全图＋ROI |
|---|---:|---:|---:|
| Judge正确 | 8 | 10 | 14 |
| 20原失败中通过 | 0 | 3 | 8 |
| 8原正确控制中通过 | 8 | 7 | 6 |

ROI相对副本配对修复5／退化1：2/23/30/37/51经原图确认恢复正确摩托车、箱式货车
或前方汽车；88变generic object，损失car类别。97两组均评分失败，但ROI遗漏原有
信号原因，存在分数未体现的语义退化。路牌67、方向75、计数71仍失败。不能把8个
通过都归ROI：44/78副本已恢复、58原本低置信。人工框是观察诊断而不是自动定位结果。

386份生成和评分全部完成，59条逐条件语义复核完成，模型参数未更新。首次GPU
推理沙箱不可见CUDA，获准宿主GPU后成功；不是训练失败。修补HTML单份内嵌数据
避免重复base64、支持点击放大，离线无需服务器。产物与复现命令见
[开发集失败验证](dev_failure_validation_20261007.md)，输出`outputs/dev_failure_validation_v1/`。

结论：下一阶段候选为全局场景＋目标细节的视觉观察／关系SFT适配，先验证不依赖
人工dev框的机制与训练侧一致输入，不直接重复旧OPD或增加epoch。没有用开发题／框
作训练数据，也未启动后续SFT、OPD、RL。专项报告、首页与README同步更新；第48–49
节补齐了此前未归档的恢复中断、训练完成与最新负结果。

## 51. 2026-10-07 11:58起／14:47 UTC：自动问题相关局部观察代码交付

承接第50节的人工ROI诊断，按用户要求实现自动观察＋SFT适配。28题人工框的
5修复／1退化仍仅是诊断结果，不作为本节自动机制的效果。本次没有下载公开
检测器权重、运行正式候选检测／EVA缓存、启动SFT／OPD／RL或新增准确率结果。

### 实现路线与数据边界

原五帧→Grounding-DINO固定类别候选→仅按原问题选择候选→全景＋原像素裁剪→
原MPP-Qwen-Next回答→原参考答案assistant-only CE。检测器不接收参考答案；
问题选区用透明规则与弱方位／邻帧重叠偏好，不使用人工dev框、不采用固定中心
裁剪，也不把几何重叠声称为可靠跟踪或前车身份判断。每帧最多2区域，原5全景
始终保留、追加视图标为同一T0..T4；低置信／不适用问题回退全图。训练侧确定性
约20%原全图回放，不改原问题、参考、train/dev/eval segment划分。

复用固定40k训练QA、4320合并dev（Action3120／Scenery1200）和原官方500；
train/dev/dev_action/dev_scenery/eval原QA值保存在新目录，视图计划另存。这里不是
全量重新采样，也没有将110条人工诊断或其框用于监督。初始化为第49节最新CE
控制final的语言LoRA、projector、完整Q-Former三路权重，不是raw MPP或OPD final。

### 代码更新

| 新增／修改 | 实际职责 |
|---|---|
| `mpp_video_qa/region_views.py` | 固定候选协议、仅问题路由、原像素框、视图／时间顺序、计划SHA、去重与三条件提示 |
| `scripts/propose_regions.py`／`install_region_detector.sh` | 独立Python3.11环境的Grounding-DINO检测、torchrun四进程分片、图片SHA／检测模型身份、断点复用和进度 |
| `scripts/prepare_region_views.py` | 原QA保留、隔离检查、逐题计划、同图片SHA＋实际像素框去重、区域缓存体积估计 |
| `scripts/cache_region_features.py` | 仅加载冻结EVA，原RGB native crop→原224处理器→BF16[257,1408]，不重复缓存原五帧 |
| `mpp_video_qa/data.py` | 每题动态视图、原全图缓存身份、区域编码器／像素身份校验、提示与元数据传递 |
| `mpp_video_qa/model.py` | 可传每题提示；answer-only LM head计算通过DDP forward，保持同一因果CE与训练参数梯度 |
| `scripts/train_sft.py`／`generate_answers.py` | 正式region参数、变视图训练／生成、视图身份记录与恢复、开发提示、预测观察trace |
| `scripts/report_auto_regions.py` | 输入／缓存检查、分rank状态、同QA／权重／预算检查、dev选模、分来源／配对bootstrap和官方选定模型报告 |
| `scripts/preview_region_views.py` | 不需服务器的原图选框／裁剪HTML，内嵌图片去重、可放大查看 |
| `scripts/ask_region_video.py` | 自带短视频五帧取样，复用冻结检测版本／planner，分阶段释放模型，保存问答观察来源 |
| `scripts/run_auto_region_sft.sh` | 安装、检测、准备、预览、缓存、SFT、评测、恢复、状态与短视频问答统一入口 |

原全图EVA缓存保持`/tmp/zjr_mpp_fast/cache/lingoqa_5f_eva_v1/`，新选中区域缓存
默认持久化到项目`datasets/cache/lingoqa_selected_regions_eva_v1/`。按像素内容共享，
不为每题复制五帧。单区域裸特征约0.69MiB，总量随实际选区／去重而变，prepare
输出估计GiB。视觉权重使用持久目录，语言权重优先tmp已有快缓存，不重现缺失
tmp BLIP路径问题。检测环境不升级原MPP Python3.8环境；正式独立环境尚未安装。

### 训练与效果评测设计

默认每分支40k×2epoch、global batch16、5000更新，四卡batch1×accum4；冻结EVA
及语言base，更新完整Q-Former＋query tokens、projector、语言LoRA。FP32 trainables
／BF16 base、双侧梯度检查点，语言／projector lr2.5e-6、Q-Former lr5e-6、seed44，
每1000步完整断点；最多15视图／480视觉token，上下文1536。

三条件full／duplicate／roi。duplicate与roi在同一选区位置使用逐字相同提示、
框坐标、顺序、视觉token数，只将附加全图换为实际crop。冻结三输入先比较；
如运行匹配训练控制，duplicate和roi同初始三路权重、问答、预算与选模机会。
每分支两个epoch均做4320题自由评测，合并dev Judge选模、同分优先早epoch；
官方500只评所选模型，不参与选模。报告分Action／Scenery、配对修复／退化及
按segment bootstrap；原官方500已被多次查看，明确不是新盲测。

没有改变OPD目标／RL奖励。本节将可靠自动观察作为待验证候选，不保证自动
候选就是正确目标，不将人工框诊断外推为自动提升。正式四卡峰值显存、速度、
检测覆盖／错选率、SFT效果仍待运行，不声称多视图与原五帧同成本。

### 验证、错误与修复记录

最终完整unittest **105项：104通过、1跳过**；另4个原test_core函数显式通过。
跳过的是MPP旧Transformers不具备Grounding-DINO，该项已在现有较新环境用
本地随机小权重真实forward／postprocess／续跑单独通过，不是预训练模型质量
评估。新增10项region检查覆盖答案独立选区、原帧／顺序、去重、stale拒绝、
三条件、数据隔离、报告dev-only选模、实际Qwen／PEFT 15视图CE／梯度／生成，
以及双进程DDP／PEFT交替5／15视图、累积4次连续两步且参数校验和一致。

发现的代码错误：ego vehicle被外部车辆规则触发、signals被sign前缀触发，
已收紧原问题词匹配；不改参考标注。MPP环境无pytest，改用既有unittest，没有
安装训练环境新依赖。沙箱禁止Gloo本机回环导致已有多进程测试失败，获准宿主
运行后通过；新增GPT2 CPU测试静态Boolean mask不受旧Gloo广播支持，测试fixture
仅排除两个rank相同的静态mask，未修改正式Qwen／NCCL buffer配置。

Python3.8编译、Bash语法、训练／生成CLI及从Tracker根目录的状态入口通过；
接口验证的小模型不是完整MPP权重训练，CPU双进程也不是四卡显存／吞吐实测。
最终状态为候选未运行、缓存未运行、ROI训练not_started。没有停止任何用户作业、
覆写历史权重／原缓存、删除数据或启动长任务。

运行、后台启动、监视、完整恢复与三条件评测命令收录于
[自动局部观察实现](auto_region_views_20261007.md)，README同步新增入口。

## 52. 2026-10-07 16:24起／10-08 01:23–07:02 UTC：旧自动ROI预处理与冻结评测完成

本节补记第51节交付后实际产生的结果。时间以落盘文件mtime和历史错误记录为准，
不是猜测的训练开始时间。此次没有ROI模型训练结果。

### 故障与完成状态

- 10-07 16:24:54：`outputs/auto_region_proposals_v1/failure_rank_3.json`保存一次
  Grounding-DINO下载阶段HF Xet/CDN网络失败。这是历史错误文件，不能只因文件
  存在判定随后已经完成的候选／缓存仍失败。不在归档复制临时签名下载URL。
- 用户还贴过`cache_region_features.py`的四rank `ChildFailedError`摘要与
  `report_auto_regions.py`的“original QA preservation / split isolation not
  established”。前者缺底层rank Traceback，不能仅凭摘要将其归因于上面的下载
  失败；后者表示当时读取到的计划manifest未建立或身份条件不满足，不是读图
  准确率。现manifest／缓存完成情况如下，不把历史报错当新的训练失败。
- 10-08 01:23:59：`lingoqa_auto_regions_40k_v1/manifest.json`落盘；保留原QA和
  split、unique native regions144040。dev4320中2623题有ROI，追加24944个区域
  槽，mean views10.7741，旧规则最多15视图。
- 10-08 02:01:34：`lingoqa_selected_regions_eva_v1/cache_summary.json`完成，
  generated144040／stale0，耗时2240.95秒，BF16[257,1408]、原EVA／LN初始化
  身份。用户给出的preflight通过44820道train/dev/eval QA；max supervised1067、
  generation prompt+96为1107，低于1536；这证明输入／缓存契约，不证明效果。
- 10-08 04:16:26／05:38:48／07:02:45：full／duplicate／ROI各4320道dev生成及
  原官方Judge评分完成。07:02:53产生`outputs/auto_region_eval_v1/report.json`。

### 实验结果

三臂同CE控制final语言LoRA／projector／完整Q-Former，不是新训练或raw Base。
原问题和参考、贪心96token预算、同一合并dev一致：

| 输入 | Judge | Action3120 | Scenery1200 | Token F1 |
|---|---:|---:|---:|---:|
| frozen_full | 60.3009% | 58.4615% | 65.0833% | 0.652541 |
| frozen_duplicate | 56.8750% | 54.3910% | 63.3333% | 0.628884 |
| frozen_roi | 53.5185% | 51.2179% | 59.5000% | 0.616973 |

| 配对 | 修复 | 退化 | 净题数 | Judge变化pp | segment bootstrap95%区间pp |
|---|---:|---:|---:|---:|---|
| full→duplicate | 214 | 362 | −148 | −3.4259 | [−4.7553,−2.1062] |
| duplicate→ROI | 189 | 334 | −145 | −3.3565 | [−4.6204,−2.0597] |
| full→ROI | 238 | 531 | −293 | −6.7824 | [−8.4462,−5.1288] |

旧report的`development_complete=false`缺的是`roi_epoch-01/02`，冻结三臂实际
已完成；官方栏空，无本轮官方集或ROI SFT结果。不能把新回放full60.30%替换为
第49节历史同CE模型dev60.46%的原记录，两个生成回放数值保留各自来源。

结论：人工28题局部观察潜力不能外推给检测＋关键词规则选区。追加重复图也退化，
说明视图／提示变化本身有代价；相同提示和槽数ROI比duplicate更差，说明旧自动
候选内容也未建立收益。可能涉及错目标、无关细节和未适配多视图，当前指标不能
单独定量分解这些原因。暂不因人工诊断就直接启动自动ROI SFT。

原表格和配对数据来自[原报告](../outputs/auto_region_eval_v1/report.md)，旧资产保留。

## 53. 2026-10-08 08:00后：外部VLM问题相关证据搜索实现与验证

承接第52节负向结果，按用户“实现代码”交付新的离线自动搜证＋冻结MPP评测。
本节时间是实现／检查时段；不是8B批量搜索或训练完成时间。

### 已落实的流程

原完整问题＋原五帧 → 外部VLM的inspect JSON → 程序从原RGB取native crop并
返回真实图像 → VLM再观察／可丢弃或继续请求 → finish只选已返回region IDs →
选中图像＋原五帧给冻结MPP → 原问题自由回答 → 同官方Judge评分。

不使用固定中心框、不按车辆类别＋关键词冒充问题理解、不读取人工dev框／
失败清单／评分／教师答案，不把外部模型的purpose或答案传入MPP。宽关系上下文
框可用，不套旧面积65%拒绝。选择输入由独立allowlist文件导出，搜索进程只收到
问题和原帧；原QA放另一目录用于下游评分，没有改标注或划分。

搜索模型默认本地Qwen3-VL-8B-Instruct；本机已知的4B仅可显式用于独立接口检查，
不静默fallback。使用既有现代chartvlm，不升级MPP旧环境。每卡独立问题分片，
搜索不使用NCCL；后续仅选中crop的EVA编码复用既有四卡缓存，不缓存所有候选。
默认dev4320→显式freeze配置→同策略official500，官方500仍非盲测。
freeze同时固定三路MPP权重／精度身份，官方不能更换与dev不同的MPP快照。

预算2轮×最多5请求＋1次final-only；最终最多5crops、每帧最多2，最多10MPP视图。
搜索整个历史像素cap2007040、单图50176–200704、输入6144／输出512token；
不静默截断，实际grid／像素／token记录于trace。MPP仍96token、batch1、原五帧
原始时序，不声称选区已确认对象身份或推理成本不变。

### 新增／修改代码与产物

新增`evidence_search.py`、`evidence_hf.py`、`prepare_evidence_search.py`、
`search_video_evidence.py`、`evidence_workflow.py`和`run_evidence_search.sh`。
`region_views.py`加入`driving_vlm_evidence_views_v1`、新原像素契约与动态运行身份，
保持旧候选协议兼容；`data.py`附加证据状态／trace hash元数据，不修改旧loss、
SFT或OPD路径。新增两个测试文件；README、本文和专项命令文档同步更新。

新数据／裁剪缓存／输出分别用`lingoqa_vlm_evidence_v1`、
`lingoqa_vlm_evidence_eva_v1`、`vlm_evidence_search_v1`，不覆盖第51–52节资产。
固定MPP初始化为`mpp_feedback_sft_control_40k_2epoch_v2/final`三路权重。
默认只评MPP full和evidence，用户不需要的外部直接答题／搜索答题C、D未实现为
自动任务。duplicate可选但默认不跑。无SFT／OPD／RL训练入口。

逐题原子结果支持恢复；worker记录启动、加载、观察、final、失败及更新时间。
紧凑watch状态输出可从Tracker根目录调用，Bash入口不交给Python、不依赖`rg`。
有效full_only和执行failed区分，两者保留原五帧，失败不删除出评测分母。
报告校验同QA、同MPP权重、同视图和官方Judge协议，输出分来源、配对修复／退化、
按segment区间及搜索／EVA／MPP分阶段成本，不把旧MPP生成ms当端到端延迟。

### 检查中修正的问题、验证与边界

代码检查中修正独立子worker试图汇总尚未完成其它分片的问题，加入针对该行为
测试；完整生成恢复会直接复用答案，不重复加载MPP。调用失败／无EOS截断留下
明确失败trace；启动即写阶段进度。上述属于本次实现检查修正，不声称用户已在
正式搜索上遇到这些故障。
新增并行异常退出测试出现过launcher lock的ResourceWarning，随后改为with
上下文关闭，异常退出也释放锁；状态统计的预测文件读取也显式关闭。此问题不是
GPU运行报错或训练效果变化，修正后复测通过。

26项新流程测试＋8项原区域回归＋1项旧Qwen／PEFT真实CE／梯度／生成＋3项现代
Qwen3-VL接口，共38项通过，无跳过。另检查并修正父进程逐轮poll所有worker，
避免后置分片失败被首分片仍活跃的短路判断掩盖。现代接口检查使用本地真实processor和随机
小Qwen3-VL，验证5／10／15图的整个历史预算与真实generate，非8B质量评估。
Python3.8编译、Bash语法、CLI help和Tracker根目录紧凑status通过。旧checkpointing
deprecation提示仍是原环境告警，此次检查未报训练运行错误。

现代环境实际为Python3.11.16、torch2.11.0+cu128、transformers4.57.6；旧MPP环境
未变。默认8B权重未下载，未实测8B单卡显存／吞吐，未导出全量新搜索输入或启动
正式搜证／EVA缓存／MPP评测。因此只有代码交付，没有新准确率结果、自动框质量
保证或“已蒸馏”结论。没有停止用户任务、删除缓存、改源QA或重训旧模型。

模型下载、开发集全流程／后台启动、实时监视、HTML、freeze后官方500、分阶段和
恢复命令见[问题相关视觉证据搜索](evidence_search_20261008.md)。

## 54. 2026-10-08 09:43–11:35 UTC：新搜证启动顺序与8B权重下载问题

承接第53节代码交付，补记用户实际运行时的问题。该阶段没有模型训练。

- 09:43:19：无答案输入导出完成，见`stage_timings.jsonl`。尚无开发集搜索manifest
  时提前执行`freeze`，在`assemble_views`读取`search_dev/run_manifest.json`时报
  `FileNotFoundError`。缺的是开发集搜证结果，不是模型训练权重损坏。
- freeze与systemd启动是分开执行的命令，所以前者失败后仍显示官方服务
  `Running as unit`。官方入口随后报“run dev and explicitly freeze its policy
  before official evaluation”。服务创建成功不代表评测成功。后续给出的完整
  命令顺序改为下载→dev→查看报告→`freeze &&`后台official，避免冻结失败仍启动。
- 8B下载遇到`IncompleteRead`／`requests.exceptions.ChunkedEncodingError`：
  一次连接已读1310231987字节、仍需3605730509字节时中断。检查时第1、3、4
  分片完成，第2分片保留`.incomplete`。具体是代理还是远端断开，日志不足以确认。
- 续传命令沿用`/data/zjr/models/Qwen3-VL-8B-Instruct`，`max_workers=1`，下载
  timeout120、etag timeout60、禁用Xet和hf_transfer，未强制重下、未删除缓存。
  用户输出第2分片4.92GB及16/16文件完成。第2分片mtime为11:31:53，四分片齐全。
- 11:35:43：新`search_dev/run_manifest.json`落盘，外部8B正式搜证开始。

下载报错和首次freeze错误没有独立可靠的秒级时间，只记当日。此轮更正的是操作
顺序与下载重试命令，没有修改搜索算法、训练代码、已有数据或权重。

## 55. 2026-10-08 16:05–18:42／10-09归档：外部8B搜证完整开发集结果

16:05:18，4320/4320搜证完成；16:09:11，7755/7755去重crop缓存阶段完成；
17:22:32，MPP full生成＋评分完成；18:42:30，MPP evidence生成＋评分完成；
18:42:39，报告和workflow completed落盘。10-09监视的worker alive=False是
已完成退出；官方EVAL=not_started符合未启动下一split的状态，不是任务卡住。

两组均固定第49节CE控制final的语言LoRA、projector及完整Q-Former，原五帧保留，
只追加外部8B选中的图像及坐标／时间，不传外部答案或purpose。开发集400segment、
4320问；原问题／参考不变，greedy96token、batch1、原官方Judge协议一致。
归档时核对两组完整唯一qid覆盖、原QA、预测／评分对应关系及manifest权重身份。

| dev指标 | full | evidence | 差值 |
|---|---:|---:|---:|
| 合并Judge | 2609/4320，60.3935% | 2417/4320，55.9491% | −4.4444pp |
| Action3120 Judge | 58.6859% | 53.3654% | −5.3205pp |
| Scenery1200 Judge | 64.8333% | 62.6667% | −2.1667pp |
| Token F1 | 0.652813 | 0.626363 | −0.026450 |
| normalized exact | 10.7176% | 9.3056% | −1.4120pp |

配对修复291、退化483、净−192；两组均对2126、均错1420。按segment的差值95%
bootstrap区间为[−6.0205,−3.0658]pp，不跨零。Action修复218／退化384，Scenery
修复73／退化99。当前输入策略是开发集负向结果，不是新训练增益。

搜索状态：selected3378（78.19%）、full_only508（11.76%）、failed434（10.05%）。
selected只证明协议合法与图像被选择，不证明选对目标。selected组修复286／
退化479，净−193（−5.7134pp）；其余942问回退／不追加，净＋1。因而主要负向
变化出现在实际追加图像组，不能简单归因于搜证失败。942个控制中897个答案逐字
一致、45个有变化，9个Judge翻转（5修复／4退化）；提示／视图契约相同，具体回放
差异来源待查，不能算作视觉收益，也不足以解释selected组退化。

434次执行失败逐条统计：单轮请求超5为236、坐标协议错误73、单帧最终区域超2
为71、finish字段不合17、JSON解析13、总数／未观察ID合并校验11、未知action11、
输出未结束2。没有CUDA／网络／加载失败记录；最大实际输入3841<6144。223次
超量请求一次要6个区域，说明工具协议遵循还有改进空间，但仅修协议不等于修质量。

10163次外部调用、观察18652裁剪、选中9431槽、去重编码7755；平均全输入7.18视图。
四卡搜索wall4小时29分36秒，逐题均值14.62s；搜索到完整两臂评测约7小时07分。
MPP生成均值full965.68ms／evidence1031.18ms，不含搜证／缓存。外部8B实测峰值
allocated17.36GiB，MPP full15.25／evidence16.03GiB；不能称在线端到端延迟或训练显存。

结论及后续：保留原五帧CE基线，暂不推进当前策略official500或直接大规模SFT／
OPD／RL。先有限审核本轮修复／退化的真实裁剪是否对应问题目标、关系和时间；
复用同一视图计划做MPP duplicate对照，区分视图／提示代价与裁剪内容影响；并核验
无裁剪控制的回放一致性。若证据正确但MPP不会用，再在训练侧独立构造同输入做
受控SFT；若选区本身不可靠则先修搜证。没有证据支持当前证据教师的OPD优势。

旧ROI53.52%和本轮55.95%因选区、覆盖和输入预算不同，不能当成严格方法消融；
本轮full60.39%也不覆盖旧60.30%／60.46%的历史记录。人工28题的收益不能外推
自动完整dev。若有限诊断仍无可复现收益，应停止自动追加区域分支。

完整表格、失败类型、成本、事件和证据入口见
[本轮完整分析](evidence_search_results_20261009.md)。此次只更新文档和时间线，
没有改训练／推理代码、重新评分、修改旧输出或启动后续实验；官方结果仍为空。

## 56. 2026-10-09 00:19–00:24 UTC运行／同日分析：退化视觉审核与同预算副本对照

回应“理论上多看局部应该改善，为何退化”，实际固定抽取24题：历史退化16、修复6、
均对2，覆盖24个不同segment。按固定seed与qid哈希排序、在看图片前选定，仅限
selected样本。它是有意多选退化题的诊断，不是新总体benchmark或训练数据。

新增`build_evidence_case_audit.py`导出原五帧、实际裁剪、原QA／答案／搜索trace与
24张图册；`run_evidence_case_audit.sh`复用本轮计划、缓存、原CE三路权重和精度，
单GPU0顺序运行full／duplicate／roi。duplicate与roi使用完全相同提示、时间、
坐标、视图槽位，前者每个附加槽填同时间全图缓存，后者填真实crop。没有再次
调用外部8B，也没有训练或外部直接答题对照。

00:19:55固定选择落盘；00:21:28、00:22:50、00:24:02分别三路评分完成。诊断服务
`mpp-evidence-case-audit-v1.service`正常退出，共72份新答案／评分。F/D/R计数
18/9/7，不作总体准确率解释。历史16退化中D已退化12道；历史6修复中D已修复4道。
D→R修复2、退化4：银灰轿车和黑衣人物的具体属性修复有图像支持，其余部分修复
即使不放大也发生。F回放23/24文字相同、Judge24/24相同；R23/24相同，案例24
发生Judge翻转，具体回放差异来源仍未确认。

逐题看图发现：08把不同来车／公交说成同一前车轨迹；07没加强后期行人过街；12
从近车换成远货车；15框对前方出租车，D/R均答“身后”；16裁剪有左侧车辆但R答
没有。局部选错与下游附加视图使用问题并存，不能只归因搜索器。03早期amber而
末帧red，需区分原因时刻；10与14的F/R近义答案却Judge一正一负，13基线本已
把对向车辆答成本车道却判对，不能把评分翻转全部称为真实能力退化。

新增`summarize_evidence_case_audit.py`核验输入／权重身份并统计全dev计划：795题
只追加整图（448→410，修复70／退化108），2582题只有实际局部裁剪（1568→1413，
修复216／退化371），1题混合；1699个附加槽为整图。selected不代表实际放大，
没有新局部信息时目前仍切换提示并加图，是具体可改进的策略问题。但实际裁剪组
仍负向，单过滤整图不足以证明方向已经修好。

全dev事后按color/colour文字匹配129题，96→84（修复3／退化15）；其中Scenery81题
57→51。因此两例颜色修复不能推广成已验证的颜色题路由。原五帧始终保留，Q-Former
逐图压32token且不以问题条件化，新增多图／坐标／时间提示未在当前checkpoint
训练中适配；对照证明这种输入改变能引起错误，但没有分离提示、位置和token数。

建议保留五帧基线，停止无条件追加后直接扩量／蒸馏。如果继续，先消除无新增信息
的整图追加和明显目标／时间／关系错配，再在独立训练集构造一致接口、原QA及全图
回放做受控SFT，验证真实裁剪超过副本且完整dev净收益为正。不能承诺SFT必然修好，
不把本次dev框／成功标签加进训练，不据此宣称OPD教师更强。未执行上述新训练。

本轮只新增诊断辅助脚本、图册和报告，没有修改主模型／搜索／训练／Judge代码、
原4320输出或权重。Bash语法、Python3.8编译及统计运行通过，三臂真实GPU运行
全部完成。详细视觉依据、24题审核表和不确定性见
[退化案例分析](evidence_case_audit_20261009.md)，原始三路结果见
[诊断对照](../outputs/evidence_case_audit_20261009/comparison.md)。

## 57. 2026-10-09 09:14–10:30 UTC：暂停搜证分支，扩大语言 LoRA 的匹配纯 CE SFT 实现

承接第56节及后续讨论，用户决定停止当前自动证据／裁剪路线，先扩大语言 LoRA，
不再将外部VLM搜证补到训练侧。保留 MPP-Qwen-Next、五帧 EVA 输入、原问答、
已有效的完整 Q-Former＋Projector 训练成果。这是新的可训练范围实验，不是
继续裁剪 OPD，也没有实施语言模型全参数解冻。

为只比较语言覆盖范围，初始化取与第49节 CE 控制相同的
`mpp_lingoqa_qformer_all_1epoch_v1` 三路 final 权重，而不是从55.4%的控制 final
再追加5000步。复用未改的40k原QA池（Action25800／Scenery14200，27599segment），
train SHA为`d885dc2f2c59e2a8ad4c01a8240e471ed3d04e4602e887dd7ef680febb770712`。
四卡batch1／累积4、2epoch、5000更新／80000 QA访问、seed44、LR语言／投影2.5e-6、
Q-Former5e-6、AdamW／cosine／warmup3%均沿用历史CE，不重新采样或修改问答。

语言LoRA由每层融合QKV `attn.c_attn` 扩为QKV、注意力输出 `attn.c_proj` 及
MLP `w1/w2/c_proj`，rank16／alpha32／实际dropout0不变。32层对应32→160模块，
原真实adapter为64个QKV张量、8388608参数；新范围按本地配置应为35782656参数，
启动会记录实际数值。EVA、Qwen原始参数、embedding与LM head继续冻结。

新增`lora_expansion.py`，将旧A/B在FP32下精确继承，新B为零，不merge到BF16主干；
`model.py`只在首次扩展进入该分支，推理／断点恢复按已保存完整配置加载，不清零
新增模块。`feedback_opd_common.py`加入范围选项；`train_feedback_opd.py`复用纯CE
循环，跳过教师probe／Teacher／Judge初始化，新增各模块更新跟踪与初始化身份。
仅新实验显式记录`assistant_only_ce`；旧QKV／OPD身份及旧CE恢复元数据保持兼容。

新增`run_language_lora_sft.sh`、`report_language_lora_sft.py`和4项新测试。
输出与旧实验隔离在`mpp_language_lora_decoder_40k_2epoch_v1`／
`language_lora_eval_v1`。报告复用历史QKV答案，两分支同样从1000／2500／5000步
按dev4320选择各自最佳，平分取较早，输出同step及最佳配对修复／退化与segment
区间；official500必须沿用dev选择，且该集合仍非新盲测。

局部真实Qwen＋PEFT检查通过：旧FP32权重精确继承、初始logits完全相等、新模块
在CE下更新、完整保存／普通加载、恢复AdamW后下一步与不中断一致。新4项＋旧
feedback-v2 9项＋执行优化8项＋旧Qwen core 1项，共22项通过，无跳过。执行优化含
两进程Gloo同步；完整MPP四卡NCCL训练未启动，不将上述验证称为正式四卡smoke。
实际父权重／40k数据／历史预算核对、Bash语法、Python3.8编译、CLI、轻量status通过。

实现检查错误也归档：缩小词表的fixture导致PEFT自动附存embedding，测试显式不存
embedding后通过，未放宽生产QKV-only检查；报告fixture缺references，补齐后通过；
Gloo在沙箱内无法解析127.0.0.1，获准沙箱外重跑8项通过，未更改通信逻辑。这些
是本次验证问题，不是新的用户训练失败。

本次仅交付代码与文档，没有新训练准确率结果、教师增益或显存保证，没有修改旧
输出、原QA／Judge或删除tmp缓存。当前历史保留CE仍为dev60.46%／official55.4%。
完整实现说明、四卡后台训练／监视／评测／恢复命令及边界见
[扩大语言LoRA实验](language_lora_sft_20261009.md)。

## 58. 2026-10-10 09:55–10:00 UTC：扩大 LoRA 报告校验修复，开发集比较完成

用户完成扩大语言 LoRA 的 5000 步训练、1000／2500／final 三个检查点在
Action3120 和 Scenery1200 上的生成与评分后，报告仍报
`saved adapter does not contain the intended language modules`。此前一次排查仅解释
原因，代码尚未修改，所以再次运行重复报错；本节为实际修复与报告补齐记录。

根因是 PEFT 0.13.2 的目标路径压缩：160 条完整模块路径缩为
`c_attn/c_proj/w1/w2`，其中 `c_proj` 同时覆盖注意力输出与 MLP 输出。报告旧逻辑
按完整路径字符串判断，误报缺模块。三个真实扩大 adapter 实际都含 32 层×5 类×
A/B＝320 张量，旧 QKV 检查点各64张量；训练记录中新模块代表参数有非零更新，
不是模块没有训练，也不需要重训。原两层／10目标的小模型测试未触发20目标压缩
阈值，且报告 fixture mock 了 `model_specs`，因此漏测。

`report_language_lora_sft.py` 改为兼容完整路径和 PEFT 后缀，并读取 safetensors
真实键名／形状检查每层 A/B、rank16／alpha32和无额外模块，保留原数据／模型身份、
评分协议及dev选模约束。覆盖结果写入checkpoint specs，没有修改训练／推理代码、
adapter配置或权重，也未放宽为无条件通过。

新增7项报告验证测试；真实 Qwen 迁移测试扩为四层／20模块，覆盖压缩后保存、校验、
恢复和继续训练。合计11项通过，无跳过。运行 `run_language_lora_sft.sh report`
成功，复用全部原始预测／评分，生成`outputs/language_lora_eval_v1/report.json/.md`。

合并dev：旧QKV在1000／2500／5000步分别为59.26／60.16／60.46%；扩大decoder为
59.38／60.14／60.42%。两分支选中final，跨分支保留`qkv_final`。扩大final的
Scenery为65.67%，Action58.40%，合并2610／4320＝60.42%；不能把1200题的Scenery
分数当作整个dev或official分数。相对旧QKV修复67／退化69，净−2题、−0.0463pp，
segment bootstrap 95% CI [−0.63,+0.56]pp，未支持稳定增益。

官方500题的扩大final结果尚未生成；本轮没有启动GPU评测或停止其他任务。下一步
运行`eval_selected`即可复用历史父模型／QKV评分、补齐扩大final的官方结果，仍按
dev选模且官方集不是新的盲测。完整原因、改动与验证见
[扩大语言LoRA实验的2026-10-10修复记录](language_lora_sft_20261009.md)。

## 59. 2026-10-10 12:04–12:18 UTC：补齐 reference_only OPD 的固定轨迹诊断并启动

针对“教师带参考答案后纠正率较高，为何学生未超过同预算CE”，新增／接通
`mpp_video_qa/opd_diagnostics.py`、`scripts/diagnose_feedback_opd.py`、
`scripts/run_feedback_opd_diagnostics.sh`，与既有OPD-v2训练、数据和权重隔离。
原始固定教师仍为完整Q-Former SFT parent，学生分别为parent、CE final、OPD final。
不创建优化器；前向／反向用于诊断，运行前后核对模型分支参数哈希。

12:15真实数据准备成功。保留已完成条件比较的200道训练错误，并从同源probe选择
100道不同题的正确对照；总计Action194／Scenery106，原错误中190道通过
reference_only教师门控。保存原token IDs、原五帧、QA、原门控和权重／代码身份。
核对训练与dev／official在题目、视频、图像路径上隔离。自动样本选择不产生事实词标签。

回放原step 1／500／1000／2000／2501／3000／4000／5000的实际四rank×累积4
批次，共128份、55份eligible。按rank/micro_step保留最后的恢复日志；所选批次
无重复。全训练日志5163行按step去重为5000行，移除恢复重复163行。epoch1→2
CE均值0.432532→0.414547，eligible JSD均值0.064060→0.064265，门控比例
44.7125%→42.5325%，学生采样Judge正确率53.465%→55.580%。这些是变化轨迹的
历史统计，不能据此判断固定样本对齐是否改善。

三个学生均沿同一原学生前缀计算无条件parent与reference_only教师分布、逐token
logP／Top-3／JSD；额外记录参考CE和无答案提示的greedy自由生成，并由原官方协议
CPU Judge评分。梯度按有效batch=16累积CE与加权JSD后，按语言／Projector／
Q-Former分别统计范数比例、夹角、负夹角批次及全局裁剪系数。这里是在起始／
终点权重上回放历史批次，不是历史step即时梯度或AdamW更新。

同时验证并复用dev历史结果：OPD相对CE修复80／退化90，导出全部170例记录和
分层40例内嵌五帧HTML。训练侧另有40例逐词审核页与默认未审核的标注模板。
用户或后续审核明确标注事实词后，可在报告中统计错误／正确事实词的教师条件影响；
未经审核不能把词概率升高、Judge通过当作教师事实正确性证据。

6项针对性测试通过，覆盖有效批次梯度累积、冲突与缺失梯度、恢复去重与完整性、
正确对照抽样及QA保留、图像级隔离、审核位置重叠。Python3.8编译、Bash语法检查、
真实300条准备均通过。沙箱内NVIDIA驱动不可见，获准读取实际宿主GPU后确认
0–7全部空闲；没有更改驱动或其他作业。

12:16启动`mpp-feedback-opd-diagnostics-v1.service`，GPU0／1／2分别承载parent／
CE／OPD三个独立诊断进程，自动完成生成、梯度回放、评分和报告。启动时服务为
active/running，三个进程已加载到MPP第10/16权重分片。沙箱内status无法看到宿主
PID时可误报interrupted，本次用宿主systemctl和实际日志确认存活；用户宿主终端
直接运行监视命令即可。12:18三个分支均进入`fixed_trajectories`，已写出首批完整
记录：parent与CE各2条、OPD3条，完成真实分支加载、token对齐、分布计算和自由生成。

截至本条记录，完整诊断尚未结束，不能宣称教师信号错误、梯度冲突或蒸馏无效的
新归因已经证实。产物目录`outputs/feedback_opd_diagnostics_v1`，完成标志为
三模型各300条轨迹、8个批次、300条评分及`report.json.completed=true`。
运行／监视／重启及解释见[固定轨迹OPD诊断](feedback_opd_diagnostics_20261010.md)。

## 60. 2026-10-10 12:32 UTC完成／2026-10-11补归档：OPD固定轨迹与梯度诊断结果

承接第59节，`feedback_opd_diagnostics_v1/report.json`记录完成时间为
2026-10-10 12:32:27 UTC。Parent、CE final、OPD final各完成300条固定轨迹、八个
有效批次梯度回放、300条无参考提示自由回答评分；学生与冻结教师参数哈希未变。
原200训练错误中190条通过reference_only教师纠正门控。

原错误200题：Parent／CE／OPD的固定前缀JSD均值分别0.062809／0.064151／0.060926，
自由回答Judge分别77／83／89题正确。OPD相对CE有186/200题JSD下降，均值约降低
5.03%；原正确100题的自由回答分别92／89／90题正确。合计300题CE→OPD修复9、
退化2，净+7题。它们是按历史采样正误选择的训练样本，不能作为开发集或总体准确率。
原正确对照相对Parent仍有回归。

OPD终点八个有效批次的加权JSD／CE全参数梯度范数平均比0.101248，余弦均值约
0.19623，8/8批次为正；eligible微批次有3/55负余弦。各模型教师权重一致，CE与OPD
对固定前缀得到的条件教师sampled-logP／Top-3一致。不能把这些梯度几何量当作
历史step即时梯度或AdamW真实更新占比。

逐词与自由回答观察不总同向：一例行人在右侧题，OPD自由回答正确指出右侧，但在
原错误前缀上的left概率未下降；部分旧错误前缀下教师仍延续错误关系。事实词模板
仍未完整审核，不据此给出自动事实错误率或所有Judge修复均为真实修复的结论。

历史dev4320仍为60.00%／60.46%／60.23%，OPD对CE修复80、退化90，区间跨零；
复用official500为54.2%／55.4%／54.8%。本轮支持“已经学到部分教师分布，但尚无稳定
开发集优势”。据此提出一次同预算权重1→3对照，检验蒸馏信号偏弱的假设，尚不认为
它已被证实。原报告、案例与图册未改；补充解读见
[固定轨迹诊断文档结果节](feedback_opd_diagnostics_20261010.md)。

## 61. 2026-10-11 UTC：实现reference_only OPD权重3完整对照与统一报告

用户要求实现前述三组方案。新增`run_feedback_opd_weight.sh`、
`report_feedback_opd_weight.py`、`diagnose_feedback_opd_weight.py`及11项针对性测试。
原CE／OPD1模型、预测与诊断直接读取；新模型隔离在
`mpp_feedback_opd_weight3_40k_2epoch_v1`，新评测在`feedback_opd_weight_eval_v1`，
新诊断在`feedback_opd_weight_diagnostics_v1`。未修改旧训练核心、数据、Judge或模型结构。

新训练仍由`run_feedback_opd_fast.sh`进入原`train_feedback_opd.py`，固定原完整SFT
parent、40k原QA、四卡batch1／累积4、2epoch／5000更新、seed44与原三组LR。
只令CE权重1、门控JSD权重3；保留在线学生采样和冻结reference_only教师，不从旧
OPD／CE final继续添加预算。加速缓存仅复用教师独立纠正门控，不复用旧学生轨迹训练。

报告核对同源train SHA、原三路初始化、完整预算、checkpoint元数据与权重、生成
和Judge协议。CE／OPD1／OPD3在1000／2500／5000步分别配对，每组都按dev4320
独立选优、平分选较早step；官方500严格沿用dev选择。输出分项、修复／退化、原
parent正确／错误子集、segment bootstrap区间及raw／weighted loss口径。
旧OPD1第1000步才切换加速执行的差异显式记录；历史比较不能视为从step0完全相同
执行版本，如果收益很小须同版权重1复跑确认。

新增诊断复用第60节的同300条token前缀及同八个历史有效批次，仅计算OPD3 final。
旧CE／OPD1不重复加载或评分。核对实际teacher张量哈希、样本／前缀／模型身份、
无参数更新，报告区分加权与未加权梯度比例；输出比较JSON／MD及40题内嵌五帧图册。
固定历史前缀回放是训练侧诊断，不是新策略的在线训练或即时梯度，不产生自动事实标签。

验证：11项新测试、9项feedback-v2回归、6项既有诊断测试，共26项通过，无跳过。
新测试覆盖三组不同checkpoint选优、官方得分不回流选模、历史文件不重写、协议／
权重更改拒绝、相同前缀、诊断复用、加权梯度、CE-only分母，以及stub捕获的真实
Bash链路四卡／累积4／2epoch／权重3参数。后者未加载GPU模型，不称为四卡训练smoke。
Python3.8编译、Bash语法检查、真实40k／parent检查及两种status入口通过。
04:07 UTC的真实检查输出`new_training_started=false`。

实现过程中首次编译发现新报告Markdown列表漏闭合并修正；读取既有梯度接口后修正
新诊断的字段名及聚合函数引用。这些问题未进入训练，未影响历史实验结果。
一次文档补丁因定位上下文不匹配被拒绝，核对后重新应用；未覆盖已有记录。
检查输出用`check_passed=true`区分前置核对与训练完成。

交付时新训练为not_started，没有新准确率或显存结果，也没有启动GPU作业、停止其他
任务或删除tmp缓存。完整前台／后台、监视、恢复、评测与诊断命令见
[2026-10-11蒸馏权重对照](feedback_opd_weight_20261011.md)。

## 62. 2026-10-11 04:15 UTC失败／随后核查：权重3四卡加载显存冲突

用户启动`mpp-feedback-opd-weight3-v1.service`后出现torchrun ChildFailedError。
读取完整journal与`failure_rank_0.json`，定位到`build_student(...).to(device)`：
GPU0总23.56GiB，本进程已分配11.37GiB，只余71.69MiB，申请96MiB时OOM。
Qwen与MPP的全部分片均已读取，尚未创建训练更新；不是权重3标量损失引起的模型增大，
也不是缺少FlashAttention警告或末尾ChildFailedError本身导致。

NVIDIA与PID核对确认GPU0有另一个用户的训练进程占用约11.9GiB。随后检查时GPU0–4
均有其他用户作业，GPU5–7空闲；没有停止或修改他们的任务。新目录只存在failure／
progress，没有run_config、summary或checkpoint。原SFT、CE／OPD1模型与结果均未改。
沙箱内systemctl总线不可访问不等于服务故障，真正失败依据为journal中的GPU OOM。

## 63. 2026-10-11 UTC：用户要求两／三卡，保持真实batch16与5000步

新增`mpp_video_qa/global_batch.py`，按seed＋epoch与原DistributedSampler一致的全局
排列，在每次16题更新窗口中跨rank分发。两卡8＋8题；三卡六个同步槽但实际6＋5＋5题，
两个空槽不读取特征、不前向／反向、不采样、不计examples；所有rank仍在更新边界归约。
loss除以真实全局题数/world，随后原SUM/world reducer得到全部真实题上的均值。
尾批按真实count归一化，不用重复题填空。40k×2epoch保持80k访问／5000更新。

`train_feedback_opd.py`新增显式global batch参数，将原逐QA反向过程提为函数，CE与
eligible JSD均沿用同一分母。默认global参数0时保留旧采样器和配置，可恢复原四卡。
两／三卡checkpoint记录布局并要求同world恢复，不隐式改rank随机状态。启动重试把
旧rank failure改名留存，避免监视沿用旧失败；没有删除失败证据。

`run_feedback_opd_v2.sh`不再覆盖调用方进程／累积设置；权重入口按可见GPU数量自动
选择2/8、3/6或4/4，并向底层透传精确global参数。重复卡号和nproc不符在加载前拒绝。
训练结构、教师、Judge、数据、LR、损失权重和保存机会均未变；没有减少训练规模。

报告验证同源初始化／数据／真实batch16／完整预算，允许物理布局变化，并显示各组
world与累积、全局batch及随机轨迹差异。顺便修复有run_config时status复用`totals`
变量导致评测总数读取异常的问题。新诊断保留历史四卡八个有效batch的16题分母，
实际候选训练配置单独记录；不修改旧诊断worker，旧源码哈希与产物仍可核验。

验证共46项：新布局12＋权重对照11＋v2回归9＋诊断6＋加速8。实际CPU/Gloo两／
三进程六次更新包含尾批、无本地真实样本／缺失梯度，与真实全局batch16的参考梯度
和AdamW参数更新一致；启动完整Bash链路参数及40k无重复访问通过。
首次沙箱Gloo报“Cannot resolve 127.0.0.1 to a (local) address”，获准在宿主复跑后通过，
这是测试环境通信限制，不是训练代码失败。Python3.8编译与Bash语法检查通过；真实
两／三卡40k配置检查通过，总5000步，`new_training_started=false`。

本次未做完整MPP两／三卡GPU smoke、未启动正式训练，也没有新准确率或显存结果。
减少卡数不减少每卡模型占用；需要每张所选卡有足够空闲显存，耗时通常增加。
后台、监视、同world恢复及后续评测见
[蒸馏权重对照两／三卡节](feedback_opd_weight_20261011.md)。

## 64. 2026-10-10／10-11完成，2026-10-14补录：扩大LoRA官方结果与权重3三卡训练

扩大decoder LoRA正式评测已经结束，旧第58节“官方未生成”是当时状态。
`outputs/language_lora_eval_v1/official_report.md`于10-10落盘：parent54.2%、
QKV CE final55.4%、decoder final54.4%。decoder对QKV修复5、退化10，净−1.00pp，
按segment95%区间[−2.60,+0.40]pp；dev60.42%对60.46%，未支持稳定增益。
报告校验修复只处理PEFT模块后缀，没有修改训练参数或重新训练。

用户以`mpp-feedback-opd-weight3-3gpu-v2.service`运行原40k／2epoch权重3试验，
CUDA_VISIBLE_DEVICES=5,6,7、NPROC_PER_NODE=3。10-11约22:23 UTC保存final：
completed=true、5000更新、每rank30000同步micro槽、真实batch16、80000 QA访问／
在线rollout、35702 eligible、758515蒸馏token。教师37266次纠正尝试均完成，35702
通过，条件通过率95.803%；学生截断和context skips均0。该率不等于学生评测正确率。
训练56117.087秒约15h35m；rank0 peak allocated18.098GiB、reserved19.203GiB。
被跟踪语言／projector／Q-Former参数均更新，教师对应参数变化0。
真实三卡训练已完成，覆盖第63节交付时尚未运行的状态；没有混入第一次四卡OOM的预算。

## 65. 2026-10-12 06:12／06:33 UTC完成，2026-10-14补录：权重3完整dev／官方报告

新权重3三个checkpoint各完成Action3120及Scenery1200生成和评分；历史父模型、
CE、OPD1产物只读复用。dev报告06:12:26 UTC、official报告06:33:56 UTC落盘。
每组按同三点1000／2500／5000的combined-dev Judge选模：CE final、OPD1 final、
OPD3 checkpoint-002500。

| 模型 | dev4320 | official500 |
|---|---:|---:|
| 全量SFT parent | 60.00% | 54.20% |
| CE final | 60.46% | 55.40% |
| OPD1 final | 60.23% | 54.80% |
| OPD3 2500 | 60.39% | 54.20% |

OPD3 final dev仅60.00%，不按official反选它。开发CE→OPD3修复223／退化226，
净−3，95%区间[−1.17,+0.97]pp；官方修复17／退化23，净−6，区间[−3.80,+1.40]pp。
开发以parent正确性分层：原错1728题相对CE净+44、原对2592题净−47，纠错与退化
抵消。官方相应净0／−6；相对parent，CE修复23／退化17，OPD3修复23／退化23。
训练JSD执行和参数更新不代表泛化；这些结果未支持增大权重的稳定收益，也未证明
MPP能力上限或OPD普遍无效。三卡与历史四卡随机流／执行切换差异在报告中保留。
官方500已经多次使用，不是新盲测。权重3固定前缀诊断代码存在，但未发现完成报告。
用户随后决定停止继续调整这一分支；没有启动权重更大或更多epoch的试验。

## 66. 2026-10-14：按用户要求准备源码上传、SFT／简历成果索引与小型证据

新增`docs/resume_checkpoint_guide_20261014.md`，区分简历SFT父模型、当前保留CE、
简历OPD1和新权重3。按贡献列明数据／模型／训练／评测源码、生成权重的入口、三路
checkpoint作用、推理和恢复文件差异，记录实际本地位置；不上传大权重和数据。
新增`scripts/export_experiment_snapshot.py`，原样导出小型run_config、summary、adapter
配置与dev／official报告至`docs/artifacts/20261014`，记录源文件SHA256及tensor文件大小。
源码中未得到checkpoint的实现仍保留，并注明未训练或负结果，不编造成功产物。

复现实际upstream需要两个本地修改文件。归档固定MPP-LLaVA revision
`baf617b0f5217442ba20721be9aed710cdf999e2`的git diff至`patches/`，新增
`scripts/setup_upstream.sh`用于clone／版本核对／应用同补丁；已应用时只核对。
环境配置随项目保存。本节准备的是新问题感知Q-Former实现之前的源码快照；
新结构的训练／推理代码后续独立修改，不把历史分数标到尚未训练的新结构。
