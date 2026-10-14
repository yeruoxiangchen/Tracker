# 实验、代码与结果增量归档（截至 2026-09-30 UTC）

本文补齐 2026-09-28 至 09-29 已运行但此前只在分散产物中记录的信息。它不新增实验，
只把数据指纹、实际训练配置、检查点、评测、诊断代码和验证状态统一落档。更早历史见
`project_timeline.md`。

## 1. 2026-09-28：full-v2 数据与视觉缓存最终状态

数据构建产物为 `datasets/processed/lingoqa_full_v2/manifest.json`：

| split | QA | 视频 | 构成 |
|---|---:|---:|---|
| train | 406,212 | 27,599 | Action 262,005 + Scenery 144,207 |
| dev | 4,320 | 400 | Action 3,120 + Scenery 1,200 |
| official eval | 500 | 100 | Evaluation |

训练、dev、eval 的视频 ID 和帧路径两两无交集；train/dev/eval JSONL SHA-256 分别为
`dfc14f...619e`、`838f90...6d6f`、`2f41b8...5aaf`。数据清单保存完整值和源 parquet
指纹。full-v2 需要的 28,099 个唯一训练/dev/eval 视频特征最终均已可用，否则后续
406,212 QA 训练和三 split 评测无法完成。运行时使用 `/tmp/zjr_mpp_fast` 可再生副本；
它不是持久数据源，不把其是否仍存在作为实验完成条件。

视觉缓存修复涉及：

- `mpp_video_qa/model.py`：允许 BLIP2 视觉权重与 MPP 投影 shards 使用不同根目录；
- `scripts/cache_visual_features.py`：只加载视觉路径和投影权重，校验已有缓存后增量编码；
- `scripts/run_full_v2.sh`：分别传递持久视觉权重根和快速 MPP checkpoint 根；
- `scripts/report_cache_progress.py`：报告已缓存/待缓存视频数。

首次 full-v2 缓存因 `/tmp` 根缺少 BLIP2 权重而在新编码前失败，随后已修复并用真实
Scenery 视频验证；失败产物没有混入训练缓存。详细故障链已在
`data_download_and_full_v2_bringup_20260928.md` 归档。

## 2. 2026-09-28：full-v2 一轮四卡 SFT

实际配置来自 `outputs/mpp_lingoqa_full_v2_lora_1epoch/run_config.json`，不是计划值：

| 项 | 实际值 |
|---|---|
| 初始化 | 官方 MPP-Qwen-Next SFT checkpoint |
| 数据 | 406,212 QA，SHA-256 `dfc14f...619e` |
| epoch / world size | 1 / 4 |
| 每卡 batch / 梯度累积 | 1 / 4 |
| optimizer steps / micro steps | 25,389 / 101,553 |
| LoRA | `c_attn`, r=16, alpha=32, dropout=0.05 |
| 优化器超参 | lr=2e-5，weight decay=0.01，warmup ratio=0.03 |
| 精度 / gradient checkpointing | bf16 / 开启 |
| 最大文本长度 | 1,024 |
| prompt | `legacy_v1` |
| seed | 42 |

训练从头完成，没有从 checkpoint 恢复（`resumed_global_step=0`）。总耗时
26,017.31 秒，约 7 小时 13 分钟；rank 0 峰值 allocated 16.04 GiB、峰值 reserved
21.28 GiB；LoRA 最大绝对参数变化 0.017250，证明确实发生参数更新。epoch-01 的
256 条 teacher-forced dev 抽样 loss 为 0.465559；它只用于运行监控，不能与自由生成
Judge 准确率等同。

保存了 5 个中间 checkpoint：step 5k、10k、15k、20k、25k。每个包含约 33.6 MB
adapter、约 67.2 MB `training_state.pt` 和记录 global/micro step、数据指纹、world
size 的 metadata。最终另存 `epoch-01/adapter` 和 `final_adapter`。

恢复代码已经写入并保存完整状态，但本次任务未发生真实中断恢复。因此可声称“具备
恢复产物”，不能声称“已完成中断恢复一致性验证”。

## 3. 2026-09-29：Base、旧 SFT、full-v2 SFT 统一评测

评测保持数据、5 帧缓存、`legacy_v1`、greedy decode、batch size 1、max new tokens
96 一致。新模型 adapter identity 为
`a442311f579dad3a32009e2a6011bae56864333b83eb6ff38a7367737dff0230`。

| 数据 | QA | Base Judge | 旧 SFT Judge | 新 SFT Judge |
|---|---:|---:|---:|---:|
| Action dev | 3,120 | 37.05% | 43.08% | 48.24% |
| Scenery dev | 1,200 | 24.92% | 25.33% | 59.33% |
| 合并 dev | 4,320 | 33.68% | 38.15% | 51.32% |
| official eval | 500 | 42.20% | 37.40% | 44.60% |

official eval 新模型 normalized exact=0.004、token F1=0.313187、Judge=0.446。
相对 Base 修复124/退化112；相对旧 SFT 修复99/退化63。开发集来源结果在
`outputs/full_v2_eval/report.json`，官方集新模型结果在
`outputs/full_v2_eval/new_eval/scored/`。

结果边界：新 SFT 明显优于旧 SFT并恢复官方集退化，但相对 Base 官方集只提高2.4 pp；
dev 和 official eval 均已用于迭代判断，不能继续当作完全未触碰的新测试集。

相关代码：

- `scripts/generate_answers.py`：保存数据、adapter、prompt、batch、解码配置指纹；
- `scripts/evaluate_answers.py`：官方 lower/strip、最大参考正类 logit、阈值 `>0`；
- `scripts/report_full_v2.py`：Action/Scenery 与 Base/旧/新模型 paired 汇总；
- `scripts/run_full_v2.sh`：缓存、训练、生成、评分和状态编排；
- `scripts/validate_full_v2.py`：split、路径、ID、来源及覆盖校验。

## 4. 2026-09-29：三个动作诊断

`scripts/diagnose_full_sft.py` 固定 Action dev 的48段视频，核验与旧诊断的题目、视频、
帧和参考答案完全相同。三种条件为原问 A、等义改写 B、B 加格式要求 C。每个模型
复用48条A，新增96条B/C；三模型共432条回答，其中新增288条且全部评分。

| 模型 | A | B | C | A→B 修复/退化 | B→C 修复/退化 |
|---|---:|---:|---:|---:|---:|
| Base | 0/48 | 0/48 | 0/48 | 0/0 | 0/0 |
| 旧 SFT | 9/48 | 9/48 | 8/48 | 5/5 | 2/3 |
| 新 SFT | 12/48 | 13/48 | 11/48 | 2/1 | 2/4 |

诊断输入 SHA-256 为 `2d4aa4...b2dc`。`scripts/run_full_sft_diagnosis.sh` 显式固定
batch size 1、max new tokens 96、legacy prompt，并按 Base/旧/new 指向正确 adapter。

同问不同视频统计使用全部1,128个无序视频对，只作为回答敏感性描述，不计算置换后的
伪准确率。新模型A/B/C中回答文本不同的配对分别为1,059/1,071/1,052；这说明输出
随视觉输入变化，但不能证明变化正确或获得时序理解。

48段的240帧已由Codex逐条复核，新模型A/B/C共144条回答按动作和原因分别审阅。
参考动作在五帧中明确支持8、部分支持33、不确定5、疑似标注问题2。原问A中动作
完全一致12、部分一致11、冲突15、未覆盖对应维度10；原因完全一致11、部分一致9、
不同/冲突28。该复核不是人工真值，也没有转成训练标签。

产物：

- `outputs/full_v2_action_diagnosis_v1/report.json`：机器可读完整统计；
- `case_predictions.jsonl`：48段关联三模型三问法全部432条回答；
- `visual_audit.json`：复核标签与说明；
- `review.html`：约80 MB、内嵌240帧的离线审阅页面；
- `review_assets/sheet_01.jpg` 至 `sheet_12.jpg`：12张五帧联系表。

完整解释和案例见 `full_v2_action_diagnosis_20260929.md`。

## 5. 2026-09-30：归档完整性核验

只读核验及语法检查结果：

- `diagnose_full_sft.py` 通过 Python 编译，`run_full_sft_diagnosis.sh` 通过 Bash 语法；
- Base/old/new 诊断目录均为96/96生成、96/96评分；
- 离线页面含48个案例、240张内嵌图片，无外部图片依赖；
- `case_predictions.jsonl` 含48段、432条模型回答；
- official eval 重新从逐题记录计数：Base 211/500、旧187/500、新223/500；
- paired 重新计算：新vs Base 修复124/退化112，新vs旧修复99/退化63；
- 没有重新训练、修改权重、删除数据或覆盖既有评测。

本次新增/最终相关代码 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `scripts/diagnose_full_sft.py` | `dccc1c74b25b8901e4811235aba8da5c7c38d00892b08ea628131f7e79900d22` |
| `scripts/run_full_sft_diagnosis.sh` | `93cb62e187f8dd7f11c4efd175d9e6979eabfa0c8e07ce315bece36c03ae6a57` |
| `scripts/run_full_v2.sh` | `0c724d969dac86ebdd61b0a589e6931f8e102fd172d0878efaff81e4c010a886` |
| `scripts/train_sft.py` | `981fe0a33dfb63525142c678cf3d3ee1ce31b6e72f51d21f2d6cacccda76a072` |
| `scripts/report_full_v2.py` | `37c000dcb4bf2f67980eae95b6e4b33e6ff3b38c989d704936c6b0e90d9f9aa9` |
| `mpp_video_qa/model.py` | `25c61b38401e7de9afa586f2dbcde6e40642ee629ef8d9c69a58da2f693cce8e` |

注意：整个项目目录当前在上层 Git 中仍显示为未跟踪目录，因此这些 SHA-256 是当前
文件快照的审计依据，不等同于 Git commit。后续若要形成可移交版本，应单独决定是否
纳入版本控制；本次没有擅自执行 `git add` 或提交。
