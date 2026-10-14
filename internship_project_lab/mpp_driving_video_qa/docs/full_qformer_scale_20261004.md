# 2026-10-04：完整 Q-Former 全量问答规模实验

完成补录：10:56 UTC训练完成，12:40开发报告完成，12:48官方报告完成（产物mtime）。
全量模型开发60.05%、官方54.2%，成为当前最佳；相对100k轮分别+2.82/+3.20个百分点。
详细成对结果见主时间线第33节。后续低学习率续训见
[续训实现](full_qformer_continue_20261004.md)。下文准备阶段状态为历史快照。

## 目的与当前状态

当前最佳模型是100k、2epoch的epoch-02：开发集57.22%，官方500题51.0%；同协议joint为
53.89%/47.2%。下一轮用406,212条原始训练问答、1epoch，与它直接比较。两轮从同一个
旧joint初始化；开放完整Q-Former、query tokens、projector和语言LoRA，EVA/Qwen基座冻结。
这同时改变QA覆盖与总更新量，不能单独归因于数据量或训练步数。

代码与全量数据已交付，参考结果核验通过，EVA缓存28,099/28,099，无缺失。
实现交付时未启动；2026-10-04 00:28:29 UTC用户已启动正式训练服务，00:31归档核对时
正在加载第4个rank模型。尚无全量模型的效果结论。CPU测试与验证见文末。

## 固定实验配置

| 项目 | 已完成100k轮 | 本轮全量 |
|---|---:|---:|
| 原始QA | 100,000 | 406,212 |
| Action / Scenery | 64,500 / 35,500 | 262,005 / 144,207 |
| 训练视频 | 27,599 | 27,599 |
| epoch | 2 | 1 |
| 优化器更新 | 12,500 | 25,389 |
| 输入 | 原始完整五帧 | 原始完整五帧 |

全量JSONL保持源行顺序与全部问题、答案、帧。DistributedSampler仍在每个epoch随机打乱。
不是扩大新视频数量，也不执行动作关键词筛选、问句改写或教师生成标签。
四卡batch1、累积4；Q-Former LR 1e-5，projector和语言LoRA LR 5e-6；原始回答CE及EOS；
FP32可训练权重、BF16计算路径和legacy_v1提示沿用100k轮。两侧梯度检查点开启。
从旧joint的语言LoRA/projector和原始视觉桥开始，不从100k epoch-02继续训练。
按上一轮实测速度估算训练阶段约10–11小时，加载/自由评测另计。

## 代码、目录与身份

- `scripts/prepare_full_qformer_sft.py`：新增互斥的`--all-train`，绕过子集抽样和每视频上限。
- `scripts/run_full_qformer_scale.sh`：独立编排，固定full模式/1epoch，复用现有训练器。
- `scripts/report_full_qformer_scale.py`：三模型比较、参考预测复用核验、视频bootstrap、dev选模、官方报告。
- `scripts/train_sft.py`：恢复时保留原始初始化哈希；仅修正溯源元数据，不改变训练更新算法。
- `tests/test_full_qformer_scale.py`：全量保留、启动参数、参考身份/旧评分、三模型比较、选模隔离及恢复溯源测试。

数据：`datasets/processed/lingoqa_qformer_all_v1`；训练：`outputs/mpp_lingoqa_qformer_all_1epoch_v1`；
评测：`outputs/full_qformer_scale_eval_v1`。原100k产物独立保留。
全量train SHA256：`dfc14f194da236ed9e35484a446144fe29425d7f3f71c7fab44c3ff3aa0c619e`，
与原full-v2训练文件相同。dev/eval哈希与100k轮完全相同；新数据manifest记录其余身份。

## 启动与监视

本次已经完成prepare、check_references和缓存文件存在性检查，可直接启动训练。卡号按空闲情况设置。

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa

systemd-run --user --unit=mpp-lingoqa-qformer-all-v1 --collect \
  --property=WorkingDirectory=/home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa \
  --setenv=CUDA_VISIBLE_DEVICES=0,1,2,3 \
  --setenv=MASTER_PORT=29686 \
  /usr/bin/bash scripts/run_full_qformer_scale.sh train
```

```bash
watch -n 15 'bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_full_qformer_scale.sh status'
```

加载日志：

```bash
journalctl --user -fu mpp-lingoqa-qformer-all-v1.service -o cat
```

前台训练等价命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29686 bash scripts/run_full_qformer_scale.sh train
```

复查或重新准备：

```bash
bash scripts/run_full_qformer_scale.sh prepare
bash scripts/run_full_qformer_scale.sh check_references
bash scripts/run_full_qformer_scale.sh cache_status
```

如果临时EVA缓存被清理，可用`CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_full_qformer_scale.sh cache`补齐。
`SCALE_DATA_DIR`、`SCALE_TRAIN_DIR`、`SCALE_EVAL_DIR`可指定新版本目录；改变目录后所有阶段使用同一组变量。

中断恢复使用实际存在的`checkpoint-NNNNNN`与`SFT_RESUME_FROM`，仅恢复同一全量实验。
不能用100k断点恢复到本轮数据/步数配置。GPU恢复沿用已有实现，功能可用但未证明逐位复现。

## 评测与决策

训练后：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
CUDA_VISIBLE_DEVICES=0 bash scripts/run_full_qformer_scale.sh eval_dev
CUDA_VISIBLE_DEVICES=0 bash scripts/run_full_qformer_scale.sh eval_selected
```

```bash
watch -n 15 'bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_full_qformer_scale.sh eval_status'
```

`eval_dev`只为新模型生成Action3120与Scenery1200条；旧joint和100k epoch-02直接读取
`outputs/full_qformer_eval_v1`已有预测。复用前检查数据、模型哈希、提示、精度、缓存路径、
生成配置、Judge协议、问题/参考一致性，以及评分对应的预测文本。

`report.json`/`report.md`报告三个模型、三组成对比较，主比较为`qformer100k_vs_full`。
按dev合计Judge点估计选模；平局保留100k模型。单独输出
`full_gain_supported_by_video_bootstrap`标识95%区间下界是否大于0，点估计领先不等于稳健提升。
仅64条teacher-forced loss不能用于替代此自由评测。

`eval_selected`重新核验dev选模：选中full才生成新的官方500条；否则复用已有官方结果，
报告`full_evaluated=false`，此时full/eval显示pending是预期行为。官方报告输出到
`official_report.json`/`official_report.md`，不反馈改变dev选择。
可单独用`report`重建开发报告，不重新生成回答。

## 验证与边界

2026-10-04 00:14 UTC全量数据落盘，406,212条/27,599视频，实际单视频最多65题，
没有沿用100k子集的11题上限。训练与固定开发/官方视频分离；数据指纹与源文件一致。
缓存状态completed、missing=0；是现有缓存文件存在性核对，没有重新读取全部张量。
参考预测和完整权重身份核验成功：joint dev53.8889%，100k dev57.2222%。

未启动GPU正式训练；本轮主要修改编排、数据和报告，沿用此前已完成真实四卡训练的核心算法。
新规模实验尚未产生新checkpoint，所有51.0%等效果数值均属于已完成100k轮。

本次10项相关unittest通过（原完整Q-Former4项＋新增规模流程6项），包括真实小型Q-Former
梯度/重载、全量行与holdout保留、Shell展开后的初始化/目录/epoch参数、旧评分错配拒绝、
三模型成对比较、官方结果不参与dev选模，以及恢复原始初始化指纹。Python编译及Shell语法
检查通过；没有启动完整7B模型的新GPU训练。测试中的Sentry弃用与Matplotlib临时缓存提示
没有导致失败。训练入口当前显示initializing_or_not_started，符合尚未由用户启动的状态。

以上验证状态对应00:21前交付时点。00:28:29用户启动`mpp-lingoqa-qformer-all-v1.service`，
00:28:34参考核验通过，00:30:42 rank2完成加载并开始加载rank3；00:31服务active/running。
后续训练结果需依据新summary和评测产物更新，当前不能把服务运行状态当作训练完成。
