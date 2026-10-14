# Reference-only OPD 蒸馏权重对照

实现与验证日期：2026-10-11 UTC。完整归档见 [项目时间线](project_timeline.md)。

## 实验问题与依据

此前同预算训练的 dev4320 得分：起始 SFT 60.00%、继续 CE 60.46%、OPD 60.23%；
复用 official500 为54.2%、55.4%、54.8%。OPD 尚没有稳定超过同预算 CE。

2026-10-10 完成的固定轨迹诊断表明，OPD final 相对 CE final 在200道历史训练错误中，
186道的固定前缀 JSD 降低，平均0.064151→0.060926，约降低5.03%。无参考提示的
自由回答，CE→OPD 为83/200→89/200；100道原正确对照为89/100→90/100。
合计300道修复9／退化2。它们是按历史答对／答错选择的训练诊断，不是开发集收益。

八个有效批次在终点权重上的回放显示，OPD 梯度范数约为CE的0.101倍，全参数批次
余弦均值约0.196，8/8批次为正；个别 eligible 微批次仍有冲突。因此本轮测试
“蒸馏项是否偏弱”，而不是把普遍梯度冲突或教师没有被学生学习当作已证实原因。
梯度范数不是AdamW更新份额；提高权重3倍也不保证下游效果或实际更新占比3倍。

## 固定条件

| 组别 | 目标 | 默认处理 |
|---|---|---|
| CE | 原参考答案CE | 读取历史完整训练、三个checkpoint的评分 |
| OPD1 | CE + 1×门控JSD | 读取历史完整训练、三个checkpoint的评分 |
| OPD3 | CE + 3×门控JSD | 新建目录，从相同起点完整训练 |

三组均以 `outputs/mpp_lingoqa_qformer_all_1epoch_v1` 的语言QKV LoRA、Projector和完整
Q-Former final 初始化。OPD3 使用新优化器及5000步完整调度；不从OPD1／CE final追加
更新，也不使用后来扩大decoder LoRA的模型。

复用 `datasets/processed/lingoqa_feedback_opd_40k_v2` 原QA池：40,000条，Action25,800、
Scenery14,200，覆盖27,599个训练segment。Train SHA-256：
`d885dc2f2c59e2a8ad4c01a8240e471ed3d04e4602e887dd7ef680febb770712`。
历史四卡、每卡batch1、累积4、有效batch16、2epoch、80,000次QA访问、5000次更新；
10月11日新增两卡／三卡执行支持，仍保持每次更新16条真实QA和相同完整预算，详见后文。
seed44，语言／投影LR2.5e-6，Q-Former LR5e-6，AdamW、warmup3%、cosine、
全局梯度裁剪1.0、FP32可训练权重和BF16主干均沿用历史配置。

教师仍为冻结的起始MPP，条件为 `reference_only`。学生在线采样；教师在当前学生
前缀上提供分布。每道题均接受CE，学生答错且生成完整、教师纠正通过且完整时加入
JSD。门控不会按eligible数重新归一化。加速入口沿用仅答案位置logits、梯度bucket、
GPU Judge及其CPU边界复核、冻结教师纠正结果缓存。缓存的是与学生草稿无关的
教师门控结果；在线学生回答和当前前缀的教师分布仍每次重新计算。

历史OPD1前1000步使用原执行实现，随后切换加速版。报告保存运行历史，并明确
是否记录了从第0步开始一致的执行设置。本轮是匹配目标、数据和更新预算的历史比较；
如果只有小幅优势，需要同加速版本复跑权重1确认，不能声称完全相同执行或随机轨迹。

## 代码与输出

- `scripts/run_feedback_opd_weight.sh`：独立两／三／四卡训练、恢复、评测、报告、诊断和监视入口。
- `scripts/report_feedback_opd_weight.py`：核对三组配置／起点／训练池／完整预算、checkpoint
  元数据与权重，读取历史评分；每个固定step进行三组配对比较，每组独立按dev选最优。
- `scripts/diagnose_feedback_opd_weight.py`：复用原200错误＋100正确对照、原八个有效批次，
  只计算新学生的固定前缀／自由回答／梯度，读取历史CE和OPD1诊断，生成比较与离线图册。
- `tests/test_feedback_opd_weight.py`：覆盖单变量配置、隔离输出、三组比较、dev选模、
  评分协议、运行历史、同前缀诊断、梯度权重、有效batch分母及四卡入口参数。

首次交付未修改训练核心；之后为两／三卡支持加入精确全局批次采样和归一化。
QA、模型结构、Judge、旧权重和旧预测仍不修改。新训练目录为
`outputs/mpp_feedback_opd_weight3_40k_2epoch_v1`，新评测／报告目录为
`outputs/feedback_opd_weight_eval_v1`，新诊断为
`outputs/feedback_opd_weight_diagnostics_v1`。可以用 `OPD_WEIGHT_TRAIN_DIR`、
`OPD_WEIGHT_EVAL_DIR`、`OPD_WEIGHT_DIAGNOSTIC_DIR`覆盖新目录。
历史引用可用 `OPD_WEIGHT_REFERENCE_DIR`／`OPD_WEIGHT_REFERENCE_EVAL_DIR`指定；
`OPD_WEIGHT_SOURCE_DIAGNOSTIC_DIR`必须与所引用的CE和OPD1权重对应。

复用评分前后不重写历史目录。历史评分缺失时，在新评测目录中生成对应模型结果。
最终报告重新核对原问答、完整预测、评分摘要、权重身份和生成协议。权重3首次启动
在加载时显存不足，尚无训练更新或新准确率结果。`check_passed=true`只表示配置／原数据检查通过。

验证完成：11项新测试、9项feedback-v2回归、6项既有诊断测试，共26项通过，无跳过。
Python3.8编译、Bash语法、真实40k／parent检查、训练／诊断status均通过。
四卡入口由stub捕获完整命令链核对参数；本轮没有实际启动四卡GPU训练。

## 训练、监视与恢复

### 2026-10-11：改为两卡／三卡

04:15首个四卡服务在rank0把学生模型移到GPU0时OOM：自身已分配11.37GiB，
剩余71.69MiB，申请96MiB失败。核对GPU0有另一个用户的训练占用约11.9GiB。
这是显存资源冲突，不是把JSD权重改3导致的模型增大；还未执行优化器更新。
原目录只有失败／加载进度文件，没有run_config或checkpoint，允许从原SFT重新启动。

| GPU数 | 每卡同步槽／更新 | 每卡真实QA／完整更新 | 真实全局batch | 总更新 |
|---|---:|---|---:|---:|
| 2 | 8 | 8＋8 | 16 | 5000 |
| 3 | 6 | 6＋5＋5 | 16 | 5000 |
| 4 | 4 | 4＋4＋4＋4 | 16 | 5000 |

两／三卡使用`ExactGlobalBatchSampler`，沿用seed＋epoch全局乱序，每个16题窗口
分发到各rank。三卡的两个空槽不做模型前向／反向、不采样、不计examples，仍在共同
更新边界参与collective。loss先除以“真实全局题数／world”，再SUM/world归约，得到
全部真实题上的均值，CE-only题仍在分母。尾批按实际题数归一化，不重复QA来填空。
两epoch严格访问80,000条QA、5000次更新，1000／2500／5000步选模机会与历史相同。

报告接受合法少卡配置，但显示实际world、累积槽和有效batch；GPU数变化会改变
rank随机流、在线学生采样和门控，不能说运行／随机轨迹与历史四卡逐位一致。
复用诊断仍回放历史四卡batch16，不把三卡六槽错算为分母18；实际训练配置另存。

启动器自动按`CUDA_VISIBLE_DEVICES`数量设置进程和累积；拒绝重复卡号与进程数不符。
四卡默认和原恢复身份保留。已开始训练的checkpoint不能跨world恢复，须使用同卡数；
本次0更新失败不是恢复，无需`FEEDBACK_RESUME_FROM`。新启动自动将旧failure文件改名
留存，避免状态页持续显示旧失败；不删除其他作业、数据或tmp缓存。

三卡后台命令（先确认5／6／7没有其他任务；与两卡或四卡命令二选一）：

```bash
systemd-run --user --unit=mpp-feedback-opd-weight3-3gpu-v2 --collect \
  --property=WorkingDirectory=/home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa \
  --setenv=CUDA_VISIBLE_DEVICES=5,6,7 \
  --setenv=NPROC_PER_NODE=3 \
  --setenv=MASTER_PORT=29721 \
  /usr/bin/bash scripts/run_feedback_opd_weight.sh train
```

两卡后台命令（仅GPU5／6，自动累积8）：

```bash
systemd-run --user --unit=mpp-feedback-opd-weight3-2gpu-v2 --collect \
  --property=WorkingDirectory=/home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa \
  --setenv=CUDA_VISIBLE_DEVICES=5,6 \
  --setenv=NPROC_PER_NODE=2 \
  --setenv=MASTER_PORT=29721 \
  /usr/bin/bash scripts/run_feedback_opd_weight.sh train
```

状态和后续评测入口／目录不变。三卡日志：

```bash
watch -n 15 'bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_feedback_opd_weight.sh status'
journalctl --user -fu mpp-feedback-opd-weight3-3gpu-v2.service -o cat
```

验证：12项新布局测试＋11项权重对照＋9项v2＋6项诊断＋8项加速回归，共46项。
包含真实CPU/Gloo两／三进程、多次梯度归约／裁剪／AdamW更新、整批／尾批及空rank；
与单进程真实batch16参考计算一致。真实40k两／三卡配置检查通过。未加载完整MPP
做两／三卡GPU smoke，也未替用户启动正式训练，不能据此报告新显存或精度。

### 原四卡命令与恢复

四卡前台命令：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29721 \
  bash scripts/run_feedback_opd_weight.sh train
```

后台运行与上面的前台命令二选一：

```bash
systemd-run --user --unit=mpp-feedback-opd-weight3-v1 --collect \
  --property=WorkingDirectory=/home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa \
  --setenv=CUDA_VISIBLE_DEVICES=0,1,2,3 \
  --setenv=MASTER_PORT=29721 \
  /usr/bin/bash scripts/run_feedback_opd_weight.sh train
```

可从任意目录监视。状态显示加载rank、optimizer_step、总5000步、实时ETA、门控率、
显存及评测生成／评分进度；没有进度文件时为not_started，不根据配置文件宣称正在训练。

```bash
watch -n 15 'bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_feedback_opd_weight.sh status'
journalctl --user -fu mpp-feedback-opd-weight3-v1.service -o cat
```

中断后可恢复本组实际存在的checkpoint，例如第1000步；恢复优化器、调度器、随机状态
与已完成的预算，不把剩余更新当作新5000步。不能把OPD1的checkpoint用于这个命令。

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29721 \
FEEDBACK_RESUME_FROM=/home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/outputs/mpp_feedback_opd_weight3_40k_2epoch_v1/checkpoint-001000 \
  bash scripts/run_feedback_opd_weight.sh train
```

## 训练结束后的评测与诊断

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
CUDA_VISIBLE_DEVICES=0 bash scripts/run_feedback_opd_weight.sh eval_dev
CUDA_VISIBLE_DEVICES=0 bash scripts/run_feedback_opd_weight.sh eval_selected
CUDA_VISIBLE_DEVICES=0 bash scripts/run_feedback_opd_weight.sh diagnose
```

`eval_dev`评测1000／2500／5000步，每组各有同样的三个选优机会，dev为Action3120＋
Scenery1200。历史CE／OPD1与parent已有完整评分直接读取；新增OPD3的6路dev评分。
`report.json/.md`给出同step、各自dev最优、Action／Scenery、相对parent原正确／原错误
子集的修复／退化和按segment bootstrap区间。相同得分选较早step。

`eval_selected`只评测各组dev所选的checkpoint。官方500不能重新选模型；它是复用集，
不是新盲测。结果在`official_report.json/.md`。两份报告同时记录历史执行差异。

`diagnose`默认比较三组5000步终点，新增学生沿原300条历史token前缀及八个历史批次
回放；这些不是新策略训练轨迹，也不会用于更新模型。teacher和student前后参数哈希
不变，复用旧诊断，报告区分加权和未加权JSD／CE梯度比例。输出`comparison.json/.md`
和40例内嵌五帧`review.html`，可核对实际对象、动作、方向变化；Judge通过不自动代表事实修复。

```bash
watch -n 15 'bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_feedback_opd_weight.sh diagnostic_status'
```

诊断不是dev指标。只有固定JSD进一步下降，而dev没有超过同预算CE，不能认定这次OPD
有效；也不据此无限增加权重。经过dev选择的置信区间属于探索性分析。
