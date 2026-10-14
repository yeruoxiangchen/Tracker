# Feedback OPD 固定轨迹诊断

实现日期：2026-10-10 UTC。完整事件记录见 [project_timeline.md](project_timeline.md)。

本轮回答：reference_only 教师的独立纠正率较高，为什么 OPD 仍未超越同预算 CE？
复用已完成训练的权重、原始学生 token 和门控结果，进行前向、自由生成和梯度回放。
不创建优化器、不执行参数更新，诊断前后核对学生可训练参数和教师分支的哈希。

## 1. 固定比较条件

| 分支 | 学生权重 | 教师权重 |
|---|---|---|
| parent | `mpp_lingoqa_qformer_all_1epoch_v1` 三路 final | 同一 parent |
| ce | `mpp_feedback_sft_control_40k_2epoch_v2/final` | 同一 parent |
| opd | `mpp_feedback_opd_40k_2epoch_v2/final` | 同一 parent |

三路指语言 QKV LoRA、Projector、完整 Q-Former。教师使用原先的 reference_only
条件；诊断各学生时不更换教师，也不把 CE 或 OPD final 当作新教师。
沿用 EVA 五帧缓存、原始问题／参考、legacy_v1 提示、FP32 可训练参数和 BF16 主干。

固定样本包括：

- 已完成教师条件比较中的 200 道训练错误，保留当时采样的精确 token IDs（包括 EOS）。
- 从同一源 probe 中按 Action／Scenery 比例选择 100 道不同题的原正确回答，检查回归。
- 从已完成 OPD 日志抽取 step 1、500、1000、2000、2501、3000、4000、5000
  的实际批次。每批次覆盖原四个 rank × 每 rank 四个微批次，共 128 份回答。
  同一 rank/micro_step 的断点恢复重复记录保留最后一份，缺少任何一份则报错。

这些是有意按原回答正误挑选的训练侧样本，不是新测试集或总体准确率估计。
开发集只复用历史 CE／OPD 修复、退化及对应的原五帧展示；不向开发集学生提示参考。

## 2. 诊断输出与解释

### 固定轨迹

每道题、每个模型都沿相同原学生 token 前缀计算：无参考的学生分布、无参考的 parent
分布、带参考的 parent 教师分布。保存 token logP、Top-3、教师条件前后 logP 差、
学生到固定教师的全词表 JSD；同时计算参考答案 CE，并使用原输入重新 greedy 回答。
新回答按项目既有官方 Lingo-Judge 协议在 CPU 上评分，分错误组／正确对照组汇总。

这可以检查 OPD 是否比 CE 更接近同一个教师，以及这种接近是否伴随同题自由回答改善。
训练日志中的 epoch JSD 单独列出：轨迹和门控随训练变化，不能拿它代替固定样本对齐。

### 梯度回放

对三个检查点分别计算 `CE` 和 `opd_weight × JSD` 梯度。先完成教师前向，再做学生
JSD 前向／反向，接着做参考 CE 前向／反向；避免在梯度检查点重算期间切换 adapter。
逐份梯度按原有效 batch=16 归一化，在 CPU 累积后再计算夹角和范数，不能平均逐题余弦
代替有效批次余弦。保留 CE-only 样本，JSD 不按 eligible 样本重新归一化。

分别报告语言／投影／Q-Former／全部参数的 CE 范数、加权 JSD 范数、两者比例、余弦、
负余弦批次数和全局裁剪系数。它们是在起始／终点权重上回放历史批次的梯度，
不是历史 step 当时的梯度，也不是考虑 AdamW 动量后的真实更新。8 个批次用于定位线索。

### 事实词审核

`review.html` 内嵌五帧、问题、参考、原错误草稿、原教师纠正、三个学生自由回答与逐词概率。
自动分层抽取 40 道供审核，`dev_review.html` 展示 40 道历史修复／退化。

`review_annotations.template.json` 默认全部 `reviewed=false`。有明确图像／参考依据时，
可在其副本中填写 `reviewer`、`reference_consistent` 和 token 位置及标签
`wrong_fact/correct_fact/wording`；`report --annotations PATH` 输出对应词的统计。
未审核时报告明确显示 `not_reviewed`。概率提高、Judge 通过、关键词匹配都不自动视为事实标签。

## 3. 运行与监视

单卡顺序运行三个模型，可从任意目录调用：

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_feedback_opd_diagnostics.sh run
```

三张空闲 GPU 各运行一个模型，后台运行完整诊断（与上面二选一）：

```bash
systemd-run --user --unit=mpp-feedback-opd-diagnostics-v1 --collect \
  --property=WorkingDirectory=/home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa \
  --setenv=CUDA_VISIBLE_DEVICES=0,1,2 \
  /usr/bin/bash scripts/run_feedback_opd_diagnostics.sh run_parallel
```

这不是分布式训练。三个进程分别复用一张 GPU，梯度只在各自进程内计算和汇总。
评分由各进程的 CPU 执行。已经完成的轨迹／批次会跳过，可用相同命令续跑。
同一输出目录不要并发重复启动任务；改变样本、代码、权重或协议需要新的
`FEEDBACK_DIAGNOSTIC_DIR`。

```bash
watch -n 15 'bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_feedback_opd_diagnostics.sh status'

journalctl --user -fu mpp-feedback-opd-diagnostics-v1.service -o cat

tail -F /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/outputs/feedback_opd_diagnostics_v1/opd/runtime.log
```

状态包括每模型固定轨迹 `300/300`、梯度批次 `8/8`、评分 `300/300`；全部完成后汇总
`report.json` 和 `report.md`。分别执行阶段：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
bash scripts/run_feedback_opd_diagnostics.sh prepare
CUDA_VISIBLE_DEVICES=0 bash scripts/run_feedback_opd_diagnostics.sh infer opd
bash scripts/run_feedback_opd_diagnostics.sh score --arm opd
bash scripts/run_feedback_opd_diagnostics.sh report
```

## 4. 代码与验证

- `scripts/diagnose_feedback_opd.py`：数据身份核对、固定推理、梯度回放、评分、HTML、报告和进度。
- `mpp_video_qa/opd_diagnostics.py`：固定抽样、split 隔离、恢复记录去重、有效批次梯度统计、审核校验。
- `scripts/run_feedback_opd_diagnostics.sh`：单卡／三卡后台入口和分阶段恢复。
- `tests/test_opd_diagnostics.py`：6 项测试覆盖冲突／缺失梯度、累积后再测角度、真实批次与恢复去重、
  对照抽样及 QA 保留、图像级 split 隔离、事实审核位置。

实现时上述 6 项测试与 Bash 语法检查通过。真实数据准备、GPU运行及结果状态另按时间线记录；
未获得运行结果之前，不用这份实现文档作“教师教错”“梯度冲突”或“学生没学会”的实验证据。

## 5. 2026-10-10 12:32 UTC完成结果（2026-10-11补归档）

三个分支均完成300条固定轨迹、八个梯度批次、300条自由回答评分，
`outputs/feedback_opd_diagnostics_v1/report.json`记录完成时间12:32:27 UTC。
学生和冻结教师未发生参数更新。原错误200题中190题通过教师纠正门控。

| 诊断组 | Parent | CE final | OPD final |
|---|---:|---:|---:|
| 原错误200题：固定前缀JSD | 0.062809 | 0.064151 | 0.060926 |
| 原错误200题：自由回答Judge | 77/200 | 83/200 | 89/200 |
| 原正确100题：自由回答Judge | 92/100 | 89/100 | 90/100 |

OPD相对CE在186/200原错误中的JSD更小；合计300题自由回答修复9、退化2。
原正确对照相比Parent仍有退化，不能将对CE的净正变化解读为没有回归。

OPD终点回放八批次的全参数加权JSD／CE梯度范数平均比0.101248，余弦均值0.19623，
8/8有效批次余弦为正；eligible单题有3/55负余弦。教师权重一致，CE和OPD对同前缀
得到的条件教师sampled-logP／Top-3一致。证据支持已学到部分教师分布，而不是完全
没有蒸馏信号；尚不证明梯度偏弱是开发集没有收益的因果解释。

固定旧前缀上的单词概率与自由回答不总同向。一例“行人在右侧”题中，OPD自由回答
由CE的左侧改为右侧，但旧错误前缀上的“left”概率仍较高。部分其他错误前缀下，
教师仍高概率延续错误关系。事实词审核模板尚未完整标注，不能据此给出教师事实错误率。

历史dev4320仍为Parent／CE／OPD＝60.00%／60.46%／60.23%，OPD相对CE修复80、
退化90，区间跨零。训练侧的小幅纠错不是新的开发集优势。下一轮仅比较蒸馏权重1和3，
保持原教师、输入、QA、初始化和5000步预算，详见
[2026-10-11权重对照](feedback_opd_weight_20261011.md)。
