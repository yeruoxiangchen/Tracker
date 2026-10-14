# 2026-09-28：驾驶视角任务说明对照

## 目的

首轮 SFT 在固定随机开发集上改善，但官方 evaluation 退化；Base 在开发集 48 道
`What is your current action and why?` 上全部将 `you` 理解为助手自身。为区分角色
歧义与驾驶动作识别错误，本实验只改变模型输入中的任务说明，不修改 LingoQA 的
视频、原始问题、参考答案、模型权重或 greedy 解码参数，也不重新训练。

旧说明 `legacy_v1`：

```text
The images are chronological frames from a driving video. Answer the question
using only visible evidence. Give a concise, factual answer.
```

候选说明 `driving_perspective_v1`：

```text
The images are chronological frames from a driving video captured by the ego
vehicle's forward-facing camera. In the question, 'you' and 'your' refer to the
ego vehicle or its driver, not the assistant. Answer the question using only
visible evidence. Give a concise, factual answer.
```

旧协议仍是代码默认值，确保既有训练和评测可复现。候选协议通过
`generate_answers.py --task-instruction-version driving_perspective_v1` 显式启用，且写入
`generation_summary.json`。

## 评测协议

- 模型：官方 MPP Base、首轮 SFT Epoch 3。
- 数据：此前固定的随机 512 道 dev，加上未落入随机样本的动作题，及全部 48 道
  动作题的改写版本；每个模型共生成 596 条。
- 配对：旧说明结果直接复用 `outputs/sft_diagnosis_v1`，新说明逐题使用相同视觉缓存、
  问题和参考答案。
- 指标：官方口径 Lingo-Judge、token F1、fixed/regressed、答案变化、AI 自述数量；
  动作原问法和动作改写分别报告。
- checkpoint 和提示选择只看 dev，不使用官方 evaluation 反复调参。若最终采用候选
  说明，再对官方 evaluation 做一次确认。
- Judge 是文本代理指标，动作题还需查看 `action_cases.jsonl` 的逐题内容。

实现文件：

- `mpp_video_qa/core.py`：两个稳定命名的任务说明。
- `mpp_video_qa/model.py`：将任务说明注入统一 prompt builder。
- `scripts/generate_answers.py`：显式协议参数及摘要记录。
- `scripts/evaluate_task_instruction.py`：准备、进度、自动汇总及配对分析。
- `scripts/run_task_instruction_ablation.sh`：Base/Epoch 3 生成与评分入口。

模板、语法、官方评分协议测试均通过；真实 Base 单条生成 smoke 完成，摘要确认使用
候选协议。完整评测已作为以下 systemd user services 启动：

- `mpp-lingoqa-taskprompt-base-v1.service`（GPU 0）
- `mpp-lingoqa-taskprompt-epoch03-v1.service`（GPU 1）
- `mpp-lingoqa-taskprompt-report-v1.service`（等待两路评分并自动汇总）

输出目录为 `outputs/task_instruction_ablation_v1`。最终报告为该目录下 `report.md`
和 `report.json`。

监视命令：

```bash
watch -n 10 '/home/zjr/anaconda3/envs/mpp_llava/bin/python /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/evaluate_task_instruction.py status'
```

## 最终结果与决定

两路各 596 条均完成生成、评分和 paired 汇总：

| 子集 | 模型 | legacy Judge | candidate Judge | fixed / regressed |
|---|---|---:|---:|---:|
| 随机 dev 512 | Base | 36.13% | 35.94% | 60 / 61 |
| 随机 dev 512 | Epoch 3 | 46.48% | 41.80% | 39 / 63 |
| 动作原问 48 | Epoch 3 | 18.75% | 10.42% | 3 / 7 |
| 动作改写 48 | Epoch 3 | 25.00% | 8.33% | 2 / 10 |

Epoch 3 随机集 token F1 从 0.60173 降到 0.57638。候选说明没有解决动作内容，反而使
整体和动作子集退化；Base 原动作题仍为 48/48 AI 自述。结论是拒绝
`driving_perspective_v1`，后续训练和评测继续使用 `legacy_v1`。这项失败消融保留，
用于说明角色提示不能替代视觉动作学习。
