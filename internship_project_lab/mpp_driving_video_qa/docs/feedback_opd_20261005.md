# 2026-10-05：失败反馈条件教师 OPD

## 本轮决策

学生从 `outputs/mpp_lingoqa_qformer_all_1epoch_v1` 的完整三组权重初始化：
`final_adapter`、`final_projector.pt`、`final_qformer_adapter.pt`。这是使用
406,212 道原始训练题、完成 25,389 更新的模型；已有开发准确率 60.05%、官方 54.2%。

教师是同一模型快照的冻结副本，使用相同 tokenizer 和同样的五帧原图；区别是教师
额外收到**训练参考答案、当前学生回答和纠错任务说明**。不沿用旧 751 条训练后的
学生，不使用此前未通过对照的零样本 Qwen3-VL-4B，不采用固定中央裁剪。
“条件教师”指监督条件增强，不意味着教师权重已单独训练得更强。

旧实验的 751 条是筛选出的学生训练数据，并非教师训练规模。本轮要单独验证
参考反馈能否提供有效纠错方向，以及这种监督能否优于继续原答案 SFT。

## 数据与训练流程

1. 从现有训练划分分层抽取 10,000 问，Action 6,450、Scenery 3,550，覆盖
   9,858 个视频，每视频最多两题；原问题、五帧路径、原答案保持不变。
2. 从这批训练题抽取 2,000 问，每题两次学生采样，共 4,000 次诊断。
   开发／官方题不参与教师提示、筛选或梯度更新。
3. 学生只看五帧和原问题，temperature=1、top_p=1、top_k=0，直接保留生成 token ID。
4. 学生 EOS 完成且 Lingo-Judge 判错时，冻结教师在参考答案和学生草稿条件下生成纠正。
   教师纠正 EOS 完成且 Judge 判对，才允许该次回答进入 OPD。
5. 对**同一条学生 token 序列、相同已生成前缀**分别计算学生和反馈教师分布。
   两者提示长度独立计算。教师新生成的纠正文本只用于资格筛选，不作为 OPD token 目标。
6. OPD 组每条样本都执行原答案 CE，合格回答额外执行全词表 token-mean JSD；
   SFT 对照每条样本执行相同权重原答案 CE。JSD 与 CE 分开反向，最后一起更新。
7. 完整开发 Action 3,120＋Scenery 1,200 问比较 baseline／SFT control／OPD，
   开发选模后仅评 baseline 和选中模型的官方 500 问。

门控失败、学生已答对或截断时，不执行 JSD，**仍训练 CE**。每次访问重新生成当前学生
回答和判断门控，不把初始筛选结果永久当作当前策略的行为。

Judge 是文本一致性代理，不读取图像。教师看到参考答案之后答对，并不能证明它真正
纠正了视觉事实。诊断导出逐 token 概率及五帧 HTML，供检查错误动作／方向／对象的
概率是否下降；程序不凭关键词自动构造“事实 token”监督。

## 配置与资源

| 项目 | 设置 |
|---|---|
| 学生可训练部分 | 完整 Q-Former＋projector＋Qwen attention LoRA |
| 视觉输入 | 已有 EVA 缓存，五帧完整画面；不用旧 Q-Former 输出缓存 |
| 参数／计算 | 三组可训练参数 FP32，冻结 Qwen BF16，Q-Former BF16 autocast |
| 教师 | 独立冻结 Q-Former、projector、LoRA；共享冻结 Qwen base |
| 每 rank batch／累积 | 1／4；四卡有效 batch 16 |
| 默认预算 | 10,000 问×1 epoch＝625 optimizer updates；max_steps=0 |
| 学习率 | language／projector 2.5e-6；Q-Former 5e-6 |
| 损失 | CE 权重 1；OPD 额外 JSD 权重 1 |
| 其他 | dropout=0；梯度检查点；CPU Judge；cosine schedule，warmup 3% |
| 保存 | 每 100 更新可恢复 checkpoint；final 三组权重与优化器状态 |

对照匹配数据、初始化、CE 权重、学习率和更新预算，并非相同墙钟时间。
CPU Judge 和双次生成使 OPD 显著慢于 SFT，不能借旧 47 步实验推算本轮耗时。
默认先做 10k 规模，而不是立即对 406k 全量进行多轮生成。

## 代码入口

- `mpp_video_qa/feedback_opd.py`：训练参考提示、门控、原始 token 概率诊断、匹配配置校验。
- `scripts/feedback_opd_common.py`：完整三组初始化与哈希、动态 rollout、教师纠正、Judge。
- `scripts/prepare_feedback_opd.py`：独立训练／诊断数据；视频、题目和图像路径隔离。
- `scripts/probe_feedback_opd.py`：可续跑的训练侧诊断、概率报告、内嵌五帧 `review.html`。
- `scripts/train_feedback_opd.py`：四卡 OPD／原答案 SFT 对照、固定通信顺序、恢复训练。
- `scripts/report_feedback_opd.py`：严格同协议对比、视频级 bootstrap、开发选模、实时状态。
- `scripts/smoke_feedback_opd.py`：单条完整模型机械更新，不保存训练权重，不测准确率。
- `scripts/run_feedback_opd.sh`：统一命令入口，从任意工作目录执行均可。

共享 `opd.py` 扩展了确定性教师生成与 FP32 教师 adapter 加载；保留旧默认接口。
`data.py` 新增可选的五帧原图缓存校验，旧数据流程默认不变。

恢复必须加载**学生**三组 checkpoint、优化器、scheduler 和各 rank RNG；
教师仍固定为最初的全量 SFT 快照，不随学生 checkpoint 改变。
基础 checkpoint 身份当前记录 index 文件哈希和路径，而非重新逐字节校验所有大分片。

## 启动与监视

数据已于 2026-10-05 17:27 UTC 准备完成，无需额外下载；可重复执行准备命令核验：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
bash scripts/run_feedback_opd.sh prepare

# 先验证反馈能否形成合理监督，不启动正式训练
CUDA_VISIBLE_DEVICES=0 bash scripts/run_feedback_opd.sh probe
```

完成后查看 `outputs/feedback_opd_probe_v1/probe_report.md` 和 `review.html`。
代码要求至少存在合格回答和 token 诊断记录才允许正式训练；这只是机械入口要求，
不是可靠视觉纠错的保证，也没有把任意小的正信号宣称为教师通过。

确认诊断后，依次运行匹配对照和 OPD：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29696 \
  bash scripts/run_feedback_opd.sh train_control

CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29696 \
  bash scripts/run_feedback_opd.sh train_opd

CUDA_VISIBLE_DEVICES=0 bash scripts/run_feedback_opd.sh eval_dev
CUDA_VISIBLE_DEVICES=0 bash scripts/run_feedback_opd.sh eval_selected
```

另一个终端监视（不依赖当前目录，也不使用 shell 不兼容的 `[[`）：

```bash
watch -n 10 'bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_feedback_opd.sh status'
```

OPD 中断后示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29696 \
FEEDBACK_RESUME_FROM=outputs/mpp_feedback_opd_10k_v1/checkpoint-000100 \
  bash scripts/run_feedback_opd.sh train_opd
```

使用新输出目录或更改预算时，SFT 和 OPD 的相应 `FEEDBACK_*` 设置必须一致。
完整配置见启动脚本。不要同时在同一四卡组运行两组训练。

## 验证状态和边界

新增 9 项反馈 OPD 测试通过，全部 unittest 回归共 73 项通过；含小型真实 Bert
Q-Former、LoRA 与 teacher 切换／冻结、JSD 梯度、不同提示长度 token 对齐、数据隔离、
匹配 CE 预算及官方选模隔离。Python 编译、Shell 语法和实际数据准备已检查。

完整 MPP 模型单 GPU 单步检查通过，产物为
`outputs/feedback_opd_real_smoke_v1/summary.json`：

- JSD 0.083420、原答案 CE 0.104004；三组都有非零 JSD 梯度。
- 三组被跟踪学生参数均更新约 2.5e-6，教师被跟踪三组参数变化均为 0。
- 峰值 allocated 17.466 GiB，耗时约 36.67 秒（含加载），没有保存训练权重。
- 24-token 学生回答截断；此检查强制执行一次 JSD，**不是通过正式门控的样本**。
  正式训练仍跳过截断回答的 JSD、保留 CE。

检查过程中发现同一学生对相同输入重复计算也存在 BF16 logits 最大差异约 0.125–0.156；
原先用最大差值 0.05 判定教师加载失败会误报。核验全部 Q-Former、projector、LoRA
权重及输入 token、attention mask、视觉／提示嵌入完全一致。无反馈师生 token-mean
JSD 为 1.194e-5，同学生重复计算为 1.094e-5，远小于此条反馈条件 JSD 0.0834。
关闭融合 SDPA 的试验没有消除差异，因此不声称已锁定某个 kernel 根因，也不修改
正式训练／评测的 attention 后端。改为精确检查加载权重与输入、记录无反馈分布噪声，
不以 raw-logit 最大值冒充权重身份检查。

首次受限环境 CUDA 不可访问时立即退出，没有回退 CPU 加载模型；真实 GPU 重试完成。
只完成单卡机械更新，未声称四卡长训练和真实教师门控诊断已验证。
正式 4,000 次诊断、两组 625 步训练及新准确率评测尚未启动。

只有不带参考反馈的学生自由评测超过匹配 SFT 对照，才支持 OPD 的独立收益。
官方 500 问在历史多轮实验中已使用过，不再称为完全盲测；开发、官方结果均需如实记录。
