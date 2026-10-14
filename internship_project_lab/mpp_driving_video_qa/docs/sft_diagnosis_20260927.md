# 2026-09-27：首轮 SFT 退化诊断

## 目的与范围

当前任务仍为基于 MPP-LLaVA 的驾驶视频问答。输入为 LingoQA 提供的 5 帧及问题，
通过冻结的 MPP 视觉编码器、Q-Former、投影层得到 160 个视觉 token，再训练 Qwen7B
的 LoRA。3 万条 Action 训练问答、3 epochs 的首轮 SFT 已完成。本轮使用现有权重
诊断退化，不重新训练或修改训练数据。

## 官方评分口径修正：已完成

对照 https://github.com/wayveai/LingoQA/blob/main/benchmark/judge.py：

- 参考答案和预测答案均执行 `str(...).lower().strip()`；问题保持原样。
- 保留官方 `max_length=128` 截断、多参考答案最大 logit、`logit > 0` 判定。
- 修改 `scripts/evaluate_answers.py`，记录 `judge_protocol`。
- `tests/test_judge_protocol.py` 验证答案预处理、保留问题原文、多参考最大值和零阈值。
- 旧评分留在 `scored/`；新评分留在 `scored_official_v2/`，未重新生成测试答案。

| 500 题评测 | Base | SFT epoch 3 | 差值 |
|---|---:|---:|---:|
| 旧 Judge（缺少预处理） | 39.6% | 34.6% | -5.0 pp |
| 官方口径 Judge | 42.2% (211/500) | 37.4% (187/500) | -4.8 pp |
| Token F1 | 0.21843 | 0.28543 | +0.06700 |
| Normalized exact | 0% | 0.4% | +0.4 pp |

按官方口径，SFT 修复 93 题、退化 117 题，净少答对 24 题。
输出：`outputs/base_vs_sft_official_v2.json`。

| 官方口径子集 | Base | SFT | fixed / regressed |
|---|---:|---:|---:|
| 当前动作及原因，100 题 | 55% | 22% | 5 / 38 |
| 其他问题，400 题 | 39% | 41.25% | 88 / 79 |

退化主要集中于动作问题。SFT 在这 100 题中 95 题以 `I am stopping` 开头。
Lingo-Judge 是文本自动代理指标，不能把每个判定直接当作人工核实的视觉事实。
Token F1 上升反映词汇重合增加，不能单独证明理解能力提高。

按参考答案进行文本一致性抽查，发现 Judge 的假阳性。例如题
`09b2700134f738d7277082de652309f0` 的参考答案描述保持车速、绿灯且道路清晰；
Base 却描述停车等待行人，仍被 Judge 判为正确。这只能确认评分与参考答案存在
明显冲突，尚未进行该视频的人工视觉真值复核。因此“动作 55% → 22%”应严格表述
为自动 Judge 的变化，不能把所有计入 regressed 的题都当作真实能力退化。

## 训练数据动作分布审查：已完成

仅以问题包含 `current action` 划定本次重点子集；答案分类是前缀规则统计，不是
人工核验的动作真值，也不代表全部动作相关训练问题。

| 答案前缀类别 | train（431） | dev（48） |
|---|---:|---:|
| stopping | 97 | 8 |
| remaining stopped | 22 | 2 |
| accelerating | 69 | 6 |
| decelerating | 77 | 13 |
| starting | 33 | 0 |
| following | 36 | 7 |
| maintaining | 39 | 5 |
| turning | 15 | 2 |
| continuing | 19 | 1 |
| 其他措辞 | 24 | 4 |

train 中直接以 stopping 开头占 22.5%，不足以直接解释测试中 95% 的 stopping 输出。
训练中 426/431 条使用 `What is your current action and why?`；dev 48 条全部使用
此问法。测试的对应问题额外要求 `action, justification` 格式。因此加入开发集同义
改写对照，以区分问题措辞敏感与原问法下也存在的动作偏置。

## 固定开发集实验：已完成

输入：`outputs/sft_diagnosis_v1/diagnostic.jsonl`；清单：同目录 `manifest.json`。
不使用 eval 样本选择 checkpoint。

- 从全部 3,120 条 dev 以 seed=42 均匀抽取 512 条，作为主比较集。
- 补齐全部 48 道当前动作题；其中 12 题已在随机样本内，因此正确视频条件共 548 题。
- 48 道动作题再各构造：同义改写、替换视频、改写并替换视频，共 144 条对照。
- 每个模型共生成 692 条；比较 Base、epoch 1、epoch 2、epoch 3。
- 补充动作题不混入随机 512 题总分，避免人为改变主比较集分布。
- 视频替换为同一问题的另一条 dev 视频，采用 seed=43 循环错排，不跨到训练/测试集。
- 替换条件使用新视频自己的参考答案作为正确性监督，并单独计算与原视频参考答案
  的一致性。后者是视觉依赖诊断，不是新视频的正确率。
- 正确与替换视频条件是同问题样本的置换；确定性解码下，替换后的输出应与对应
  donor 正常输入一致。不能把替换后 donor 正确率不变误判为“不使用视觉”。

另行核对 train/dev、train/eval、dev/eval 的视频 ID 交集均为 0；全部替换样本的
帧路径与参考答案均匹配 donor，且 donor 与原视频不同。

生成采用原有 greedy 协议、batch size 1、max_new_tokens=96。仅新增诊断元数据传递，
没有改变模型、prompt 模板或原始预测文件。

读取全部 48 个动作视频的真实缓存检查：48 份不同的特征张量，形状均为
`[5,32,4096]`，无 NaN/Inf，float32 范数范围 3924.50–5003.74。未发现这些视频误用
同一视觉缓存；这项检查不等于验证了视觉特征的语义质量。

### 完整结果

四路均完成 692 条生成、官方口径评分与 paired 汇总。固定随机 512 题是主比较集：

| 模型 | Judge | Token F1 | 平均答案词数 | 相对 Base fixed / regressed |
|---|---:|---:|---:|---:|
| Base | 36.13% | 0.24052 | 55.62 | - |
| Epoch 1 | 44.53% | 0.57940 | 18.93 | 119 / 76 |
| Epoch 2 | 44.53% | 0.60756 | 18.12 | 123 / 80 |
| Epoch 3 | **46.48%** | 0.60173 | 18.34 | 125 / 72 |

开发集样本内 Epoch 3 最佳。动作控制结果：

| 模型 | 原问法 Judge | 改写 Judge | 原问 stopping | 改写 stopping |
|---|---:|---:|---:|---:|
| Base | 0.00% | 29.17% | 0/48 | 0/48 |
| Epoch 1 | 14.58% | 10.42% | 16/48 | 6/48 |
| Epoch 2 | 16.67% | 20.83% | 6/48 | 9/48 |
| Epoch 3 | 18.75% | 25.00% | 19/48 | 27/48 |

Base 对 48 道原问法全部产生 AI 自述式回答；SFT 修复了角色错误，但 Epoch 3 的原问
动作前缀集中为 stopping 19/48、decelerating 20/48。换视频后 Epoch 3 有 43/48 的
完整答案和 35/48 的动作前缀发生变化，与原视频参考答案的一致率从 18.75% 降至
6.25%。因此视觉输入确实影响生成，但模型仍没有稳定学会动作及原因判断。

问题模板审计进一步发现，随机 dev 512 题中 509 题的精确问题文本在 train 出现，
而 official evaluation 只有 11/500。这不是视频泄漏，却说明当前 dev 更接近已见
模板，能够解释“dev 提升、evaluation 退化”的一部分差异。

结果文件：`outputs/sft_diagnosis_v1/report.json`、`report.md` 和
`action_cases.jsonl`。

### 复现入口

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
python scripts/diagnose_sft.py prepare
# 分配各自空闲 GPU 后分别运行 base、epoch01、epoch02、epoch03
CUDA_VISIBLE_DEVICES=2 bash scripts/run_sft_diagnosis.sh base
# 四路均完成后汇总
python scripts/diagnose_sft.py report
```

历史实际服务为 `mpp-lingoqa-diag-{base,epoch01,epoch02,epoch03}-v1.service`，GPU
分别为 2、3、5、6。此处仅记录当时分配，不代表未来这些卡仍空闲。

另启动 `mpp-lingoqa-diag-report-v1.service`，等待四路评分结果后自动运行 report，
输出 `report.json`、`report.md` 和 48 题完整对照 `action_cases.jsonl`。最长等待 2 小时，
超时会报告失败，不伪造完整结果。监视命令：

```bash
watch -n 10 '/home/zjr/anaconda3/envs/mpp_llava/bin/python /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/diagnose_sft.py status'
```

状态中的条数表示已落盘输出，不保证进程存活；若复现时长时间不更新，应查看对应
服务日志。历史运行已全部完成并生成上述报告。

## 结论边界

已确认：SFT 并非完全无效——随机 dev Judge 提升 10.35 pp、回答更简洁且不再把
`you` 当作 AI；但 official evaluation Judge 下降 4.8 pp，动作内容出现明显偏置。
视觉替换会改变输出，只能证明模型使用了视觉信号，不能证明正确时序理解。Epoch 3
是 dev 内最佳 checkpoint，但不能据此宣称跨模板泛化成功。下一步先用版本化任务说明
隔离角色歧义，再决定是否调整训练数据或训练目标；不应直接用 evaluation 反复调参。
