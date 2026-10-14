# Feedback OPD v2：4 万题、两轮、5,000 次更新

日期：2026-10-06 UTC。对应完整时间线：[project_timeline.md](project_timeline.md)。
本次实现不增加到 10 万题，不下载新数据。实现交付时未自动启动正式长训练；
此后200条教师条件对照已完成、SFT控制组已经启动。最新归档快照见第7节，
历史验证边界见第6节，实时进度用status读取。

## 1. 现有证据与本轮要回答的问题

`outputs/feedback_opd_probe_v1/probe_report.json` 已完成：2,000 道训练题，
每题两次采样，共 4,000 份回答。学生采样正确率 53.70%；1,852 份学生错误中，
教师参考答案＋错误草稿条件下纠正通过 Judge 的为 1,364 份，即 73.65%。
这使 34.10% 的全部采样可进入原门控。无截断、无上下文超长。

但教师独立重新回答正确，不代表沿学生错误前缀计算的分布能纠正学生。
40 份逐 token 诊断中 36 份门控通过，其中 34 份整段 sampled-token 平均 logP
提高；这不能直接当作坏监督的占比，因为正确词、措辞和 EOS 也在其中。
有具体方向词反例：学生错误地回答 left、参考为 right，教师新回答正确，
但教师沿原学生前缀给 left 的概率由约 0.290 提高到 0.985。
另有速度动作正例：错误 accelerate 概率由约 0.350 降到 0.042。
因此需要同一批学生 token 的条件对照，而不是从纠正率直接宣称蒸馏有效或无效。

教师不另训，仍为完整 Q-Former SFT 快照；学生同样从该快照初始化。
这不是弱外部教师，也不是中央裁剪教师。`reference_only` 仍属于
参考答案条件的同模型蒸馏，不宣称新算法或必然解决错误前缀问题。

本轮问题：在足够覆盖、明确预算下，额外的门控 on-policy 分布损失是否比
同数据、同初始化、同学习率、同 CE、同更新次数的纯 SFT 控制组有效？

## 2. 数据和初始化

- 父模型：`outputs/mpp_lingoqa_qformer_all_1epoch_v1`，三组权重是
  `final_adapter`、`final_projector.pt`、`final_qformer_adapter.pt`。
  父阶段 406,212 道原 QA、25,389 次更新，历史 dev 60.05%、官方 54.2%。
- 新池：`datasets/processed/lingoqa_feedback_opd_40k_v2/train.jsonl`。
  Action 25,800 / Scenery 14,200，全部 27,599 个训练视频覆盖，其中
  Action 24,191 / Scenery 3,408；题目、答案、完整五帧路径不改写。
- 抽样：保持原 source 比例，每个 source 内按视频轮转取题。先覆盖视频，
  再取下一轮问答；实际最多每视频 5 道。Scenery 配额无法用旧每视频 2 道
  上限满足，故 `--max-qa-per-segment 0` 自动允许所需轮次。
- 40k QA × 2 epochs；每次访问重新生成当前学生回答，预期 80,000 次 rollout，
  不是固定缓存回答训练两遍。门控通过次数随当前策略变化，不预设为固定样本数。
- 保留原 dev Action 3,120 + Scenery 1,200，以及 official 500；视频、题目、
  图像路径 train/held-out 交集为 0。教师只在训练侧接触参考答案。
- 复用 `/tmp/zjr_mpp_fast/cache/lingoqa_5f_eva_v1`，不重新裁图、缓存或下载。
- 40k train SHA256：`d885dc2f2c59e2a8ad4c01a8240e471ed3d04e4602e887dd7ef680febb770712`。

## 3. 教师条件对照（不更新权重）

`compare_feedback_teachers.py` 复用已完成旧 probe 的回答和原 token IDs，
从属于本轮训练池的学生错误中确定性选 200 道不同题：Action 129 / Scenery 71。
混合旧门控通过与教师未纠正成功的样本；因此它是诊断集，不是无偏错误率估计。
无需重新生成 4,000 份学生回答，也无需重新构造问答或人工标错误 token。

每份固定学生 token 分别做：

1. `reference_draft`：参考答案＋完整错误草稿，保持旧教师条件。
2. `reference_only`：参考答案，不额外给完整错误草稿。

每种条件下，教师独立纠正生成与沿原学生 token 的概率计算都使用同一个 prompt。
两种条件均仍沿学生 completion prefix 计算，不能理解成去掉错误前缀。
这里比较教师条件，不把旧实现称为生成／打分提示不一致的 bug。

输出 `outputs/feedback_teacher_compare_v2/{probe_report.json,probe_report.md,predictions.jsonl}`，
并生成两个五帧内嵌的 `review_reference_*.html`。记录纠正门控、重复错误草稿、
原 token 概率、候选分布、JSD；不根据概率升降生成事实标签或自动选最优教师。
默认新训练条件为 `reference_only`；正式200条结果已完成，见第7节。它是通过
条件诊断后继续验证的候选，不能把去掉草稿当作已验证的学生收益。若它也持续
强化明确错误，先讨论监督而非盲训。
训练入口要求所选条件的对照已完整结束、有可用门控和 token 诊断、身份与训练一致，
这只是运行前提，不是正确性认证。`reference_draft` 可通过环境变量选择。

## 4. 正式训练

| 项 | OPD 与 SFT 控制组 |
|---|---|
| 训练模块 | 完整 Q-Former＋projector＋语言 attention LoRA |
| 冻结模块 | EVA、Qwen 基座；OPD 另有固定教师视觉桥和 LoRA |
| 显卡 / micro batch / 累积 | 四卡 / 每卡 1 / 4 |
| 有效 batch / epochs / updates | 16 / 2 / 5,000 |
| LR | 语言 2.5e-6，projector 2.5e-6，Q-Former 5e-6 |
| CE | 所有访问均使用原始答案，权重 1 |
| OPD | 仅 OPD 分支：门控通过后 token-mean 全词表 JSD，权重 1 |
| 学生采样 | temperature=1，top_p=1，top_k=0，dropout=0 |
| 教师纠正 | greedy；只有 Judge 通过且 EOS 完成才蒸馏 |
| 长度 | max_new_tokens=96，max_txt_len=1536 |
| 调度 | 3% warmup＋cosine；三组 FP32 trainables、BF16 frozen base |

OPD 的训练序列是当前学生 IDs，而非教师纠正答案；教师纠正用于动态门控。
教师条件提示下的全词表分布是 target。正确回答、截断、教师错误等保留 CE、
跳过 JSD；不通过猜测出的错误词 mask 改写标签。教师计算全部结束、恢复学生
adapter 后再进行学生反向；教师不更新，多卡固定次序同步梯度。

保存默认每 1,000 步及显式 2,500 步；5,000 步用 `final` 完整 bundle 保存，
不额外重复保存 checkpoint-005000。包含三组权重、优化器、scheduler、各 rank RNG。
恢复时学生从 checkpoint 继续，教师仍保持最初父模型，累计统计继续记录。

同预算是更新次数相同，不是时间相同。OPD 含学生生成、教师生成、Judge、双侧
概率计算，实际耗时明显高于纯 SFT；以运行中的最近 20 步 ETA 为准，不承诺几小时。

## 5. 启动、监视、恢复与评测

全部在项目目录执行。prepare 已完成，可重复执行核验：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
bash scripts/run_feedback_opd_v2.sh prepare

CUDA_VISIBLE_DEVICES=0 bash scripts/run_feedback_opd_v2.sh compare_teacher
```

阅读两种教师条件报告后，确定条件再训练。默认 `reference_only`。
两组使用同一值；变更条件必须改用新的输出目录，不覆盖既有 run。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29697 \
  bash scripts/run_feedback_opd_v2.sh train_pair
```

`train_pair` 依次运行控制组、OPD，不让同四卡两组同时竞争显存。也可单独使用
`train_control` / `train_opd`。任一分支失败时不会继续下一组，按对应 checkpoint 恢复。
后台运行（只需选一种启动方式，不与上面的前台命令重复执行）：

```bash
systemd-run --user --unit=mpp-feedback-opd-v2-pair --collect \
  --property=WorkingDirectory=/home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa \
  --setenv=CUDA_VISIBLE_DEVICES=0,1,2,3 --setenv=MASTER_PORT=29697 \
  bash scripts/run_feedback_opd_v2.sh train_pair
```

监视（Bash 运行 Shell 脚本，不用 Python；不依赖 `rg`）：

```bash
watch -n 10 'bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_feedback_opd_v2.sh status'
journalctl --user -fu mpp-feedback-opd-v2-pair.service -o cat
```

训练完之后，同样五帧、无参考答案提示，逐组完成自由生成和官方 Judge：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_feedback_opd_v2.sh eval_dev
CUDA_VISIBLE_DEVICES=0 bash scripts/run_feedback_opd_v2.sh eval_selected
```

dev 比较父模型及两组的 1,000、2,500、5,000 步，共 7 个模型、每模型 4,320 题。
输出同一步 OPD vs SFT、每个候选 vs 父模型、各分支最优 vs 父模型，以及最优
OPD vs 最优 SFT。以 combined dev Judge 为选模指标，同分取较早步；两分支均
获得相同选模次数。然后 official 分别评父模型、SFT dev-best、OPD dev-best，
共 3×500 题，不用 official 分数选 checkpoint。

报告：`outputs/feedback_opd_eval_v2/{report.json,report.md,official_report.json,official_report.md}`。
按视频做 paired bootstrap，记录 fixed/regressed/net。开发选模后的 CI 是探索性统计；
反复看过的 official 500 不是新盲测。若 OPD 未超越匹配 SFT，不能把 CE 续训收益
归因于 OPD；父模型仍可保留为部署基线。

2026-10-06四卡执行加速、真实测量和兼容恢复见
[四卡OPD加速记录](opd_speed_20261006.md)。旧v2入口默认计算路径保留；切换加速版
使用`run_feedback_opd_fast.sh train_opd`，须先结束旧任务并指定完整checkpoint。

恢复示例（把路径改成已存在、完整保存的实际 checkpoint；只恢复对应一组）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29697 \
FEEDBACK_RESUME_FROM=outputs/mpp_feedback_opd_40k_2epoch_v2/checkpoint-002500 \
  bash scripts/run_feedback_opd_v2.sh train_opd
```

`status` 显示教师比较进度、PID 存活、两组训练步数／门控／源分布／蒸馏 token 数／
recent ETA／显存，以及所有固定检查点自由生成和评分进度。最终生成数量不等于
评分已结束，报告须验证生成／评分文件、协议和模型身份。

## 6. 实现交付时的验证边界（后续运行见第7节）

新增：`compare_feedback_teachers.py`、`report_feedback_opd_v2.py`、
`run_feedback_opd_v2.sh`、`tests/test_feedback_opd_v2.py`。
更新：`feedback_opd.py`、`feedback_opd_common.py`、prepare/probe/train 与 v1 wrapper。
v1 教师默认条件、旧10k数据／已完成probe产物／旧权重保留。

实际 40k 准备、重复准备、缓存文件存在性和 200 条对照选择／初始化身份核验通过；
200 条包含旧 eligible 101 条、旧未纠正 99 条。全回归 82 项通过（新增 9 项），
Python 3.8 编译、Shell 语法通过。新增 manifest 直方图的整数键序列化造成重复
准备不一致，已改为稳定字符串键并回归验证。

真实两条教师条件检查首次因沙箱 CUDA 不可访问退出，尚未加载模型；随后申请
GPU 权限完成，产物 `outputs/feedback_teacher_compare_smoke_v2`。两种条件的两条
纠正均完成并通过 Judge、两条原学生序列概率对齐成功，生成两个内嵌五帧 HTML。
此处没有 optimizer 更新，不把 2/2 当作教师质量或学生准确率结论，也不宣称本轮
四卡正式训练已经验证。完整 200 条条件对照、两组 5,000 步训练和自由评测未启动。
任意 cwd 的监视入口、CLI 参数转发（四卡／两轮／无 max-step 上限／1000和2500保存）
均已核验；旧 73 项测试仍通过。

## 7. 2026-10-06运行补档：200条对照完成、SFT控制组进行中

按产物时间更新，详细时间线见主档第45–46节。

### 04:20 UTC：正式200条教师条件对照完成

报告 `outputs/feedback_teacher_compare_v2/probe_report.json`：200/200完成，两个条件
均记录200份token诊断。reference_draft纠正通过103/200（51.50%），reference_only
通过190/200（95.00%）；配对修复90、退化3。教师原样重复错误草稿从58份变为0份。
通过门控的回答中，整段原学生sampled-token平均logP提高的份数分别为81/103与5/190；
全部诊断token平均JSD分别0.083550／0.061774。

这支持继续验证reference_only，不能当作学生准确率提升、自动事实token标注或
已证实的视觉监督正确性。该集混合旧门控状态，不是无偏错误率样本；不可把其
51.50%／95%与先前全量错误条件中的73.65%直接混比。两份五帧HTML与逐题诊断
已经生成，报告保留 `visual_supervision_direction_verified=false`。本阶段无模型更新。

报告SHA256为`328bf4986c265a38f2f0e5a8ca1362e459369a5b4bcf566ea9a8cf4f12670794`，
控制组run_config所绑定的probe hash与它一致。

### 04:37 UTC起：正式SFT控制组运行

`outputs/mpp_feedback_sft_control_40k_2epoch_v2` 的run_config于04:37:46 UTC落盘，
确认四卡／40000QA／两轮／5000步，CE-only、完整Q-Former＋projector＋语言LoRA。
05:04:24 UTC保存完整checkpoint-001000，包含三组权重和优化器／scheduler／RNG
恢复状态。05:15:56 UTC进度快照为1436/5000步（28.72%），当前CE0.394226，
最近20步1.579655秒/更新，rank0峰值allocated16.9468GiB；当时控制组自身ETA为
1.5639小时，不包括OPD训练／评测。归档时核验rank0 PID1541600存活。

SFT控制组rollouts、eligible、JSD为0是预期行为。检查时尚无控制组summary／failure
产物，OPD运行目录尚不存在，未观察到OPD或本轮held-out自由评测启动。
这些是时间快照，不是文档自动持续监视；后续以status和实际summary为准。
本次只更新归档与入口摘要，没有修改正在运行的代码、数据或进程，也没有学生
效果结论。完成两组后仍按第5节执行dev选模与官方对比，不拿教师纠正率替代学生评测。

## 8. 2026-10-07：正式结果完成与后续验证

两组均5000步完成；OPD从1000的fast恢复、中途nohup SIGHUP、systemd恢复及最终
22:41完成情况，补入主时间线第48节。两分支合并dev均选择final：

| 模型 | dev4320 | 官方500 |
|---|---:|---:|
| 完整Q-Former全量SFT父模型 | 60.00% | 54.2% |
| 同预算CE控制final | 60.46% | 55.4% |
| 反馈OPD final | 60.23% | 54.8% |

OPD相对CE在dev修复80／退化90，官方修复10／退化13；配对区间都跨零，没有
独立收益证据。官方数据已反复使用，不是新盲测。正式训练80000在线rollout、
34898eligible、737446蒸馏token，教师条件纠正95.92%也不能代替学生评测。
不直接扩大当前OPD。报告：`outputs/feedback_opd_eval_v2/report.md`、`official_report.md`。

已完成[开发集共同失败复核与观察验证](dev_failure_validation_20261007.md)：80失败＋
30控制、386份观察生成／评分。28题匹配局部ROI对照修复5／退化1，但ROI人工选取，
不是全dev或自动机制收益。下一候选为观察方式和问题目标绑定的SFT适配，未启动。
这些dev题／框不进入训练，旧训练／评测产物和未运行入口继续保留。
