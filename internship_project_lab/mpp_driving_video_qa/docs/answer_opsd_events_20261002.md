# 2026-10-02：视觉续训、答案条件OPD事件与结果补录

归档核对快照：2026-10-02 07:32 UTC。主入口为 [项目完整时间线](project_timeline.md)，本文件补充其第19.1、20–23节。项目根目录以下均指 `internship_project_lab/mpp_driving_video_qa`。

本轮只更新文档，不重启、停止训练或评测，不修改实验数据与模型。事件时间优先使用产物中的UTC时间字段；缺少字段时采用文件mtime并明确为落盘时间。mtime不能证明精确启动时间，本文没有回推未记录的启动时刻。

## E01 — 2026-10-01 12:28–14:58 UTC：视觉适配SFT及裁剪教师结果

| 事件 | 时间依据及UTC | 事实 |
|---|---|---|
| 20k视觉续训完成 | summary mtime 12:28:47 | 1250次更新，5000 micro steps，1582.34秒 |
| Action开发评分完成 | summary mtime 14:09:25 | 3120题，Judge50.2885%，F1 0.6406730 |
| Scenery开发评分完成 | summary mtime 14:25:39 | 1200题，Judge60.0%，F1 0.5807336 |
| 完整帧教师诊断评分 | summary mtime 14:55:56 | 80题，Judge48/80=60% |
| full+中央crop诊断评分 | summary mtime 14:58:01 | 80题，Judge34/80=42.5% |

视觉续训为EVA缓存输入，训练Q-Former、query tokens、projector和Qwen LoRA。rank0峰值allocated16.25197GiB；记录的LoRA、projector、query tokens最大参数变化分别为0.00037736、0.00044645、0.00085093。相对joint，Action修复123/退化160，Scenery修复29/退化31，合计少答对39题。参数发生更新不等于任务改善。

证据：

- `outputs/mpp_lingoqa_qformer_lora20k_v1/summary.json`
- `outputs/visual_adaptation_eval/visual_action/scored/summary.json`
- `outputs/visual_adaptation_eval/visual_scenery/scored/summary.json`
- `outputs/visual_adaptation_eval/joint_vs_visual_action.json`
- `outputs/visual_adaptation_eval/joint_vs_visual_scenery.json`
- `outputs/teacher_probe_v1/{joint,joint_crop}/scored/summary.json`

结论：继续保留joint LoRA+projector作为已验证基线；固定crop未产生更强教师。80题为选定开发诊断，不能当总体准确率，更不能将其描述为OPD正式训练的失败。

## E02 — 2026-10-01 22:52–23:20 UTC：新OPD实现与验证

本次代码文件mtime落在上述区间，具体设计和命令见 [实现说明](answer_opsd_20261001.md)。从joint初始化，冻结EVA/Q-Former和教师，完整五帧Q-Former缓存复用；只训练policy语言LoRA和projector。教师提示增加第一条训练参考答案，学生输入保持原问题和五帧。

新增/修改功能：多次随机采样且保留sample_id；训练数据与dev/eval视频隔离；正常EOS且教师4/4正确的筛选；按来源配额优先选差距样本并回放；原学生token逐位置JSD及0.2CE；匹配纯SFT控制组；独立命令、进度、paired评测；评分进度文件；恢复身份增加训练模式、提示协议、学习率、seed与选择manifest哈希。不同提示长度各自定位回答token，教师前向结束后恢复policy再反向。

23:01:34（manifest字段）生成2000道候选、2000段，Action1290、Scenery710；所需缓存无缺失。23:19:08（summary mtime）真实短测落盘：2次更新、每步累积2条、4次采样均结束，教师参数不变，LoRA/projector均更新，峰值15.02GiB，43.53秒。短测路径为 `outputs/answer_opsd_real_smoke_v1/summary.json`。

验证记录：23项unittest、4项core函数、编译/Shell语法及CLI检查通过。没有安装额外依赖。此次文档归档不重复执行GPU短测或训练。

实现过程有一次文档apply_patch因代码块的 `done` 行缺少补丁前缀而返回 `invalid hunk`；修正补丁格式后成功。此为文档编辑工具错误，未造成训练代码或实验产物损坏。

## E03 — 2026-10-02 01:30–03:43 UTC：师生采样、评分和运行警告

| 事件 | 时间UTC | 依据 |
|---|---|---|
| 学生8000条生成完成 | 01:30:46 | generation_summary mtime |
| 学生评分完成 | 01:34:05 | score_progress.updated_unix_seconds |
| 教师8000条生成完成 | 03:40:31 | generation_summary mtime |
| 教师评分完成 | 03:43:49 | score_progress.updated_unix_seconds |
| 751条训练选择落盘 | 03:43:50 | train.manifest mtime |

双方同2000道原始训练题，每题4次随机采样；temperature1、top_p1、top_k0、seed42、max_new_tokens96。adapter/projector身份相同，观察均为full，差别是teacher_context。全部16000条正常结束，无缺失、重复或截断。

| 范围 | 问题数 | 每模型回答数 | 学生Judge | 条件教师Judge | 差值pp |
|---|---:|---:|---:|---:|---:|
| 全部 | 2000 | 8000 | 47.3250% | 70.6375% | +23.3125 |
| Action | 1290 | 5160 | 45.7364% | 78.8760% | +33.1395 |
| Scenery | 710 | 2840 | 50.2113% | 55.6690% | +5.4577 |

整体normalized exact：学生0.07725、教师0.1805；token F1：0.5860669、0.6582050。评分协议为 `official_lower_strip_maxref_logit_gt0_maxlength128_v1`。

按每题四次通过数：

| 通过次数 | 学生题数 | 教师题数 |
|---:|---:|---:|
| 0/4 | 491 | 104 |
| 1/4 | 420 | 213 |
| 2/4 | 340 | 362 |
| 3/4 | 310 | 570 |
| 4/4 | 439 | 751 |

教师通过次数更多1164题、更少354题、相同482题。条件教师并非所有题都优于学生。观察到Scenery答非所问，例如题ID `2a021442070274160f62b9d9d0849c95` 问天气、参考为多云，teacher回答周围没有其他交通；学生4次通过、教师0次通过。该分析来自问题/参考/回答文本，未在此轮重新人工看图定真值。

用户报告两类警告：`resume_download` FutureWarning和`TypedStorage`弃用警告。核对时学生评分继续推进，随后教师开始生成；最终两组评分均完成。结论是本轮非阻断警告，无需因为它们重装依赖、重下模型或重跑。不能将warning误记为训练失败。

所有原始预测、分数、生成身份和进度位于 `outputs/answer_opsd_probe_v1/{student,teacher}/`。

## E04 — 2026-10-02 03:43 UTC：筛选结构与局限

| 分组 | Action | Scenery | 总数 |
|---|---:|---:|---:|
| 教师4/4，学生不足4/4 | 459 | 84 | 543 |
| 师生均4/4，回放 | 155 | 53 | 208 |
| 合计 | 614 | 137 | 751 |

543条纠错候选中，学生0/4、1/4、2/4、3/4分别为108、144、150、141条。不能称为“543条学生完全不会的题”。回放占27.70%，因为稳定正确回放只有208条，未硬凑30%。

Action从原候选64.5%上升到81.76%，Scenery降到18.24%；来源配额仅在合格样本足够时保持，当前代码用其他来源补齐不足配额。这一轮选中了全部751条稳定教师样本，未凑齐1000条上限，也未丢弃通过筛选的额外样本。原问题、参考、segment和frames逐项保持一致。

数据路径 `datasets/processed/lingoqa_answer_opsd_v1/train.jsonl`，身份如下：

| 对象 | SHA-256 |
|---|---|
| 2000条候选 | `91d68c9ef5d718fec6592a2ab5d29dd3a6acde9cf2a9f314872194687db245c5` |
| 751条训练集 | `783a717dfe3ab4b212798d3d8c7c1d772aa74f6bad51515f5f4cd001827739d6` |
| 筛选manifest | `abb90f568dfaabf3355aee1ed7d5105f86f4a131812067f2d094cd3e313a86a6` |
| 初始joint adapter身份 | `f753072958d16519dcbac2c9be61a1d40f1c5b0ed2234543c70363e82e098c01` |
| 初始joint projector | `28491589845697fdbb0fdd9289e7af8cc6ec052a607f7d87d8e08c15181d1bbc` |

adapter身份是项目directory_fingerprint算法，非单个权重文件哈希。前述身份来自已落盘manifest/run_config。

## E05 — 2026-10-02 06:34–06:43 UTC：四卡对照训练完成

共同设置：751条训练数据，world_size4，每卡batch1，grad_accumulation4，1epoch，LoRA/projector LR2e-6，bf16基座、梯度checkpoint、冻结Q-Former、max_new_tokens96、seed42。`max_steps200`是上限，实际47次更新。

| 项目 | SFT控制组 | 答案条件OPD |
|---|---:|---:|
| 第一个更新时间（日志UTC） | 06:34:52 | 06:37:32 |
| 最后更新时间（日志UTC） | 06:35:40 | 06:42:59 |
| summary落盘UTC | 06:35:40 | 06:43:00 |
| optimizer steps / 每rank micro steps | 47 / 188 | 47 / 188 |
| 训练耗时，不含模型初始化 | 50.05秒 | 335.88秒 |
| rank0峰值allocated | 14.8871GiB | 15.0493GiB |
| 全局样本处理数 | 752 | 752 |
| 学生rollouts / 截断数 | 0 / 0 | 752 / 0 |
| 首步CE / 末步CE | 0.249893 / 0.218109 | 0.249557 / 0.231750 |
| 首步JSD / 末步JSD | 0 / 0 | 0.042607 / 0.037905 |
| 被跟踪LoRA参数最大变化 | 6.15185e-5 | 6.66259e-5 |

各训练日志47条，无failure_rank文件。752是DistributedSampler为四卡补齐1次样本处理，不是751条之外新加了训练标签。首末loss来自不同mini-batch，不足以证明完整收敛或正确率提升。

控制组配置/summary保留了公共 `objective=jsd`、`teacher_context=answer_hint`、`sft_weight=0.2` 字段，实际行为应以 `training_mode=sft_control` 和代码分支为准：teacher=None，不生成rollout，不计算JSD，CE权重1。OPD才执行JSD+0.2CE。已用日志rollouts0/JSD0核对控制组；本轮记录这一元数据语义局限，没有改写历史summary。

最终产物分别位于：

- `outputs/mpp_lingoqa_answer_opsd_sft_control_v1/final/`
- `outputs/mpp_lingoqa_answer_opsd_v1/final/`

两目录均实际存在 `adapter/adapter_model.safetensors`（33,562,824字节）、`projector.pt`（12,600,431字节）、`training_state.pt`（92,406,357字节）、adapter配置与metadata。47步小于save_every50，未生成第50步中间checkpoint属正常。恢复需使用实际存在的final包；已经完成的同配置运行不能当成未完成任务直接续跑。

## E06 — 2026-10-02 07:32 UTC：自由评测进行中与归档完成

核对快照：`outputs/answer_opsd_eval_v1/joint_action/predictions.jsonl` 有2626/3120条，尚无评分summary。joint Scenery、SFT控制组、OPD的开发预测尚无产物；最终report未生成。后续任务可能继续推进，这个数量只代表核对时刻。

本次补齐主时间线的历史落盘事件与当前状态，更新README入口及OPD实现说明的最新状态。已有训练、采样、故障修复的早期记录继续保留在原docs。此轮没有最终学生正确率可归档；下一事件应补录无hint开发评测以及joint/SFT控制组/OPD的修复、退化和分来源差异。

## 当前相关代码身份（2026-10-02归档快照）

这是文件内容SHA-256快照，不是Git提交。对应功能在E02及实现说明中描述。项目目录在根仓库仍为未跟踪目录，本次未执行git add/commit。

| 文件 | SHA-256 |
|---|---|
| `mpp_video_qa/observations.py` | `bf817a9fb687024c675e6d33259749c046b9c05687e0771bfd6a6687ee57f202` |
| `mpp_video_qa/opd.py` | `cbc70db5dc4d101fb2b409a1827bf36be469e3f6d25bb4f9b75c78635da2b792` |
| `scripts/compare_teachers.py` | `cc980d568792f298b485abe2ba6ffccf1697094a53c7927c0a3d6874ee6902a0` |
| `scripts/prepare_answer_opsd.py` | `dd38c9c76473ece50b6ca0db1f57ff3cd5827da91c34d6fbbefa4dea2244e2ff` |
| `scripts/prepare_opd_data.py` | `195312726809e961ebe94466d8a067e2ca2c5bb9df591ab85c06a92188ffb2ce` |
| `scripts/train_opd.py` | `cb350a4f9043b79cab48818e1790bd3cdd219ec293239f9ad7342d6a089c2bc8` |
| `scripts/evaluate_answers.py` | `e351e7c7936a3e5f6da369d8b25ca448c33ec66526e5bfead3a8e7a53ffe5bdc` |
| `scripts/report_answer_opsd.py` | `70e67a6a2ffd3a4f6bbad4299017adc7babbb5b0f358b3b9961ccd06ac215a6b` |
| `scripts/run_answer_opsd.sh` | `3828eee08d14bd1d8b8dc549dd987e8b9c972180ce51ccf1d98c8caa7cd6c8f5` |
| `scripts/smoke_answer_opsd.py` | `414f9f934ca417467ab66653a9567370a31cbfd1b24929b8d084878795d5570a` |
| `tests/test_answer_opsd.py` | `30aa408f2992f7a9009a5a81d325947644c33d2705338e45ed9ddce3f0835418` |

决策维持：保留joint基线，等待同协议无hint开发评测。条件教师70.64%不等于学生70.64%，训练执行成功不等于OPD有效，Action占比增加后的Scenery退化需要单独观察。
