# 2026-10-01：动作定向续训与同路径诊断

最新状态：四卡续训、Action/Scenery自由评测及joint/targeted的48段诊断均已完成。
Action正确率51.47%→49.55%，Scenery60.17%→60.08%；继续采用上一轮joint。
完整代码快照、时间、结果和下一阶段计划见[实验结果归档](action_targeted_results_20261001.md)。

## 目的与边界

保留 MPP-LLaVA、Qwen、五帧输入和原始答案监督，先做低学习率的定向联合 SFT，
暂不引入 OPD/RL。不是重新全量训练，也不把已分析的开发/评测失败题转成训练题。
动作、原因、速度、位置等标签仅用于采样，不是新增真值；不能据此声称视觉证据已人工审核。

## 已实现与实际数据

- `scripts/prepare_action_targeted_sft.py`：只从 full-v2 train 选取 5,000 条重点原问答、
  2,000 条保守问句改写及 3,000 条通用回放。答案、帧路径保持原样，记录原题 ID。
- 已生成 `datasets/processed/lingoqa_action_targeted_v1`：10,000 条、5,795 段视频；
  Action 来源 5,774 条、Scenery 来源 4,226 条。任务类型与来源 split 不是一回事。
- 每段最多 4 条原始题，含改写最多 8 条；单一原问句最多 300 条。随机种子 42。
- train/dev/eval 视频与帧路径无泄漏；沿用原 dev 4,320 条、eval 500 条。
- 6,295 段所需 Q-Former 缓存全部存在，确定性抽检 8 个张量协议正确；不是逐文件全量内容校验。
- `review_sample.jsonl` 导出 100 条重点原题，尚未完成图像级人工复核。
- `train_sft.py --init-adapter` 加载已有可训练 LoRA，配合 `--projector` 加载联合投影层，
  新建优化器/调度器；`--resume-from` 才是同一次实验的断点恢复，两者互斥。
- 默认初始化 `outputs/mpp_lingoqa_projector_lora_1epoch_v1` 的两个 final 权重；
  1 epoch、四卡、每卡 batch 1、累积 4、LoRA/投影层学习率均 5e-6，预计 625 次更新。
- 诊断沿用历史 48 个开发视频，A 原问复用同路径完整 Action 评测；B 改写、C 加格式，
  每模型新生成 96 条。比较权重身份、缓存、数据指纹、生成参数和评分协议。
- 新增诊断 JSON/Markdown、内嵌五帧 HTML，以及固定模型的逐题修复/退化报告。

## 启动命令

数据已生成，不必再次 prepare（构建器拒绝覆盖已有版本）。先跑同路径诊断：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
CUDA_VISIBLE_DEVICES=0 bash scripts/run_action_targeted_sft.sh diagnose base
CUDA_VISIBLE_DEVICES=0 bash scripts/run_action_targeted_sft.sh diagnose old_lora
CUDA_VISIBLE_DEVICES=0 bash scripts/run_action_targeted_sft.sh diagnose joint
bash scripts/run_action_targeted_sft.sh diagnostic_report
```

四卡须先确认空闲；以下 GPU 编号是示例。后台续训：

```bash
systemd-run --user --unit=mpp-lingoqa-action-targeted-v1 --collect \
  --property=WorkingDirectory=/home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa \
  --setenv=CUDA_VISIBLE_DEVICES=0,5,6,7 --setenv=MASTER_PORT=29661 \
  /usr/bin/bash scripts/run_action_targeted_sft.sh train
```

监视：

```bash
journalctl --user -fu mpp-lingoqa-action-targeted-v1.service -o cat
watch -n 10 'bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_action_targeted_sft.sh status'
```

训练后先检查两个完整开发子集，防止只改善动作措辞而损害场景回答：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_action_targeted_sft.sh eval action
CUDA_VISIBLE_DEVICES=0 bash scripts/run_action_targeted_sft.sh eval scenery
CUDA_VISIBLE_DEVICES=0 bash scripts/run_action_targeted_sft.sh diagnose targeted
bash scripts/run_action_targeted_sft.sh diagnostic_report base old_lora joint targeted
```

选定方案后再运行 `CUDA_VISIBLE_DEVICES=0 bash scripts/run_action_targeted_sft.sh eval eval`。
500 条官方评测已多次被分析，不应再称完全未见的最终测试；不据其单题结果构造训练标签。
成功标准看 Action 开发集净修复、Scenery 回退、48段视觉核验共同判断，不能只看 loss/F1。

## 验证与未完成项

Python 编译、Shell 语法检查通过；7 个 unittest 通过，覆盖采样约束、确定性、
配对计数、评分协议以及小型真实 Qwen/PEFT 前后向和投影保存重载。
环境未安装 pytest，未声称运行 pytest 全套。数据构建、缓存存在性校验及48段诊断准备已执行。
实现交付时尚未启动GPU诊断或续训；后续已完成625步四卡训练及评测，详见上方结果归档。
100条训练复核样本与本轮诊断回答的逐图人工裁定仍未完成。
统一 Shell 入口启用离线模型读取，避免评分阶段 Hugging Face 在线证书失败；
systemd 直接执行文件，避免内嵌长 Bash 的变量展开问题。
