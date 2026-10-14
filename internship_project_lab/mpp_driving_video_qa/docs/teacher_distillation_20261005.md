# 2026-10-05：外部视觉教师对照与离线答案蒸馏

## 当前状态

代码已实现；600条教师对照于2026-10-05 10:33 UTC完成，学生58.83%、教师42.33%，
修复82/退化181，未通过原总体优势门槛。10k训练候选推理和两组正式训练尚未运行。
后续使用[任务说明与原生视频对照](teacher_adaptation_20261005.md)检查教师任务适配。
本文末尾单条验证记录描述代码首次交付时状态，正式教师结果见时间线第37节。

学生初始化为`outputs/mpp_lingoqa_qformer_all_1epoch_v1`，同时读取语言LoRA、Projector、
完整Q-Former。该模型开发4320题60.05%、官方500题54.2%。续训检查点24000的对应结果
为61.78%/52.2%；本轮使用54.2%模型作为既定参考。官方集合已在历史实验中多次观察，
不将它描述为从未使用的盲测集。

默认候选教师是本地`chart_verification_vlm/checkpoints/Qwen3-VL-4B-Instruct`，使用现有
`chartvlm` Python3.11/Transformers4.57.6环境。只读复用权重与环境，不修改该项目。
4B是参数规模，不构成教师强于MPP的证据；必须通过开发对照。可用`TEACHER_MODEL`
和`TEACHER_PYTHON`切换其他已下载、兼容的HF教师，切换时使用新的`--output-dir`。

## 方法与数据

外部教师读取同一视频五帧完整图像和原问题，按时间顺序组成多图输入；没有参考答案提示、
中央裁剪或人工选择区域。每帧教师处理预算为50176至1003520像素；学生保留已有224输入
的EVA缓存。教师和学生的视觉处理分辨率不同，报告中有记录；教师能看到的细节不一定都能
通过蒸馏传给学生，需要用训练对照检验。

来源为`datasets/processed/lingoqa_qformer_all_v1`。新样本在
`outputs/teacher_distillation_v1/data`，不覆盖原始数据：

| 用途 | 问答 | 视频 | Action | Scenery |
|---|---:|---:|---:|---:|
| 教师开发对照 | 600 | 400 | 433 | 167 |
| 训练候选 | 10000 | 9858 | 6450 | 3550 |

seed44；按来源比例、不同视频轮转抽样，每视频最多2问，不根据已有正确/错误选开发题。
检查train/dev/eval的question_id、segment_id、frame路径隔离。完整开发集仍为4320题，
官方仍为500题。学生的600条开发预测从已核验的父模型结果中复用；epoch-01/final的
pt封装SHA不同但实际张量相同，沿用先前逐张量核验。

教师对照报告给出Judge、Token F1、fixed/regressed、来源分组和视频级bootstrap。
教师可用答案相对学生的95%区间下界大于0才允许进入训练准备，未正常结束的教师答案
在这个判定中按错误处理。原始Judge结果也单独保留。该门槛是实验规则，不能代替
视觉审查；`teacher_review/review.html`包含五帧、参考答案和双方回答，可抽查分歧。

训练候选在教师、学生都无参考提示的条件下生成，然后由相同官方Judge评分。
筛选教师正确且完整、学生错误的答案作为纠错目标；从学生原本正确的题中抽取约30%
原始答案回放，包括教师答错的题。最终有效样本数由实测决定，不宣称10k条全部进入训练。
`answers`始终保留原始GT；`teacher_answer`、`target_source`单独保存训练目标。

蒸馏组使用这个固定混合数据集，每个样本只做一次标准答案token交叉熵；约70%教师目标、
30%原始答案回放，相当于按实际比例混合两个CE期望。对照组使用完全相同的样本ID、顺序、
训练步数与初始化，全部读取原始`answers[0]`。训练目标来源是两组的区别。

默认3epoch、4卡、每卡batch1、累积4；语言LoRA/Projector LR2.5e-6，完整Q-Former
LR5e-6，FP32可训练参数、梯度检查点、seed44。每200步保存可恢复检查点。若最终N条，
默认更新数为`3 * ceil(ceil(N/4)/4)`。本轮预先固定比较两组final，开发选择baseline、
control或distill，平分保留较早者，然后仅评选中模型的官方集合。

这是离线答案蒸馏。教师字符串由学生tokenizer重新编码，无需词表对齐；旧`train_opd.py`
仍服务于同tokenizer的在线token分布蒸馏，没有把该流程称为OPD或RL。

## 完整命令

先在一张空闲GPU上运行教师对照。模型均已在本地，无需下载新数据或权重：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
CUDA_VISIBLE_DEVICES=0 bash scripts/run_teacher_distillation.sh teacher_compare
```

需要后台运行时，可使用：

```bash
systemd-run --user --unit=mpp-teacher-compare-v1 --collect \
  --property=WorkingDirectory=/home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa \
  --setenv=CUDA_VISIBLE_DEVICES=0 \
  /usr/bin/bash scripts/run_teacher_distillation.sh teacher_compare

journalctl --user -fu mpp-teacher-compare-v1.service -o cat
```

任意目录监视；状态以文件为依据，不冒充进程存活验证：

```bash
watch -n 15 'bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_teacher_distillation.sh status'
```

对照完成查看`outputs/teacher_distillation_v1/teacher_report.json`中的
`teacher_qualified`与`usable_teacher_paired`。如果为false，训练准备会解释原因并退出，
此时应阅读纠错/退化案例或换教师，不会自动启动两组训练。

教师达标后的训练候选推理和筛选：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
CUDA_VISIBLE_DEVICES=0 bash scripts/run_teacher_distillation.sh prepare
```

该步骤要生成学生和教师各10000个回答并评分，比开发教师对照耗时；支持生成断点续跑。
教师、学生、Judge顺序运行，进程退出释放显存。观察最终`data/train_manifest.json`中
`teacher_targets`、`reference_replay`和实际来源分布。

两组训练顺序执行，选取四张空闲卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_teacher_distillation.sh train_control
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_teacher_distillation.sh train_distill
```

两组完成后：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_teacher_distillation.sh eval_dev
CUDA_VISIBLE_DEVICES=0 bash scripts/run_teacher_distillation.sh eval_selected
```

完整开发比较在`report.{json,md}`，官方比较在`official_report.{json,md}`。比较
`baseline_vs_distill`以及`control_vs_distill`，同时看净修复和区间；只有前者提高不能
单独证明教师监督优于同预算SFT。

断点恢复示例（换成实际存在的检查点）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_teacher_distillation.sh train_distill \
  --resume-from outputs/teacher_distillation_v1/train_distill/checkpoint-000200
```

`--output-dir`、`--epochs`等非默认参数在各相关阶段保持一致。新teacher模型与新像素预算
使用新实验目录。`--dry-run`仅打印GPU命令，数据准备/身份校验仍执行，状态显示
`dry_run_completed`，不能当作正式评测完成。

## 实现、核验与本次发现

- `scripts/generate_vlm_teacher.py`：独立HF教师推理、实际像素限制、EOS完成标记、逐条续跑。
- `mpp_video_qa/distillation.py`：原始划分隔离、预测对齐、教师纠错与原答案回放选择。
- `scripts/teacher_distillation.py`与Shell入口：阶段衔接、教师判定、同预算训练、开发选模、报告与监视。
- `mpp_video_qa/data.py`：默认原答案行为保持一致，增加显式distillation目标模式。
- `scripts/train_sft.py`：目标模式透传、保存与恢复检查；开发loss始终使用原始参考答案。
- `tests/test_teacher_distillation.py`：数据/答案隔离、截断过滤、真实缓存读取、流水线衔接、
  两组命令匹配、预算不同拒绝比较、官方结果不反向选模等测试。

本机Processor测试发现Transformers4.57.6的AutoProcessor未把顶层min_pixels/max_pixels
转给Qwen3图像处理器，仍保留16777216像素默认上限。已显式设置实际image_processor的size；
1920×1080五帧测试得到每帧46×82个16像素patch，即965632像素，低于1003520。

首次单条生成测试在沙箱内不可访问CUDA、自动回退CPU。已停止该验证进程，增加CUDA可用性
检查，在获准GPU环境重跑通过；不是正式训练失败。GPU验证产物：
`outputs/teacher_distillation_backend_smoke_v1`，1/1，16个completion tokens，finished=true。
回答是否正确没有在该单条检查中得出结论。

单条GPU验证、真实数据准备、父模型开发答案身份核验通过。新增10项CPU测试通过；
此前已运行的15项完整Q-Former/全量/续训回归测试通过。没有运行两组正式GPU训练，
没有新增教师对照准确率或蒸馏收益。
