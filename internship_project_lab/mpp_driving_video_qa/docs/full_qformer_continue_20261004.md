# 2026-10-04：完整 Q-Former 全量模型低学习率续训

## 已验证的起点

父实验 `outputs/mpp_lingoqa_qformer_all_1epoch_v1` 已完成406,212条原始QA、1epoch、
25,389次更新。开发4320题60.0463%，官方500题54.2%。本轮完整加载父实验的
`final_adapter/`、`final_projector.pt`、`final_qformer_adapter.pt`，新建训练阶段。
截至代码交付，仅完成CPU测试、真实权重等价核验和基线评测身份核验，未启动本轮正式GPU训练。

## 配置与实现

- 数据：复用 `datasets/processed/lingoqa_qformer_all_v1`，不重抽样、不改答案。
- 输入：完整五帧，复用EVA缓存；提示 `legacy_v1`，精度 `fp32_trainables_v1`。
- 更新范围：完整Q-Former及query tokens、Projector、语言LoRA；EVA/Qwen基座冻结。
- 四卡，每卡batch1，累积4；新增1epoch/25,389步；与父阶段合计50,778步。
- 学习率：Q-Former 5e-6、Projector/语言LoRA 2.5e-6；warmup 3%，线性衰减。
- seed43用于新阶段打乱；父阶段为42。新建AdamW和调度，不恢复父阶段优化器。
- 保存6000/12000/18000/24000步与最终完整模型；中断可恢复本阶段checkpoint。
- 输出：`outputs/mpp_lingoqa_qformer_continue_1epoch_v1`。
- 评测：`outputs/full_qformer_continue_eval_v1`。

`run_full_qformer_continue.sh`提供独立入口；复用`train_sft.py`的训练损失、DDP、
梯度累积及检查点。通用启动脚本新增完整初始化路径、seed/warmup透传和checkpoint评测。
训练配置/进度/总结记录`parent_run`、`parent_optimizer_steps`与累计步数；续训恢复
拒绝更换父模型、seed或学习率/调度参数。父实验及其他实验的默认入口保持原配置。

## 真实权重核验中发现的问题

首次`check`拒绝复用父模型的epoch-01评分：final与epoch-01的Projector/Q-Former
文件SHA不同。原因是torch.save的归档文件名不同；随后逐项核对了元数据、张量键、
形状、dtype与全部张量值，两份权重完全一致，语言LoRA目录指纹也一致。
新增等价核验后才允许复用原预测，仍按原epoch-01文件身份验证旧评分，未绕过身份检查。
该问题不代表父模型权重损坏，也不改变已经得到的60.05%/54.2%。

## 启动和监视

在项目目录执行，GPU编号按可用卡设置：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
bash scripts/run_full_qformer_continue.sh check
mkdir -p logs
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash scripts/run_full_qformer_continue.sh train \
  > logs/full_qformer_continue_v1.log 2>&1 &
```

```bash
watch -n 15 'bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_full_qformer_continue.sh status'
```

`status`显示加载阶段、新阶段/累计步数、loss、分组学习率、更新时间与按平均速度估算的
剩余时间。它是文件进度而非进程存活证明；超600秒未更新的running记录标记为stale。
也可`tail -F logs/full_qformer_continue_v1.log`查看原始日志。

恢复示例（仅在对应checkpoint已存在且训练已停止时使用）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
SFT_RESUME_FROM=outputs/mpp_lingoqa_qformer_continue_1epoch_v1/checkpoint-006000 \
bash scripts/run_full_qformer_continue.sh train
```

## 评测与选模

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_full_qformer_continue.sh eval_dev
CUDA_VISIBLE_DEVICES=0 bash scripts/run_full_qformer_continue.sh eval_selected
```

`eval_dev`依次为五个候选生成Action3120/Scenery1200答案，采用相同96-token上限和
官方Judge协议；当前全量基线复用既有结果。五套完整自由评测会额外占用推理时间。
输出`report.json`和`report.md`，包括各来源和合计准确率、F1、修复/退化、视频级bootstrap。
开发集Judge点估计选择`selected_model`，平分保留基线/较早检查点；区间跨零时
`recommended_model`保留baseline。多检查点选择后的bootstrap属于探索性证据。

`eval_selected`先核验完整开发结果，仅生成被选候选的官方500题答案，并与父模型官方
结果比较，输出`official_report.json/.md`。如果开发集选中baseline，不另做官方推理。
官方集不参与选模，且已被本项目多轮查看，不能视作全新盲测。

```bash
watch -n 15 'bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_full_qformer_continue.sh eval_status'
```

未选检查点的官方评测显示`not_requested`，而非待执行。训练过程中单独检查某个已经
完整保存的候选可执行（另选空闲GPU）：

```bash
CUDA_VISIBLE_DEVICES=4 bash scripts/run_full_qformer_continue.sh eval checkpoint-006000 action
CUDA_VISIBLE_DEVICES=4 bash scripts/run_full_qformer_continue.sh eval checkpoint-006000 scenery
```

最终报告要求五个候选的开发评分完整。若续训无稳定净收益，保留父模型，不自动增加轮数。

## 验证范围

15项相关CPU测试通过，涵盖完整Q-Former梯度与重载、精度一致性、父模型完整初始化、
阶段恢复来源、Shell参数展开、不同封装的张量等价、开发选中间检查点、官方集选模隔离
以及旧预测/评分身份。Python编译和两个Shell入口语法检查通过。
真实`check`成功复用4320题，得到60.0462963%；未启动新的正式训练或GPU长任务。
