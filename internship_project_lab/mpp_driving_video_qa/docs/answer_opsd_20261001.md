# 2026-10-01：完整五帧、答案条件教师的OPD

原始记录时间：2026-10-01 23:20 UTC；以下实现与短测条目保留当时状态。

2026-10-02 07:32 UTC更新：师生各8000条采样评分与751条筛选已完成；SFT控制组与OPD四卡训练均完成47次更新，开发自由评测进行中。完整事件、警告和结果见 [10月2日归档](answer_opsd_events_20261002.md)，统一入口为 [主时间线](project_timeline.md)。

## 动机与边界

已有 `teacher_probe_v1` 的80条开发诊断中，joint完整帧教师Judge正确48条（60%），full+中央裁剪34条（42.5%）。不能支持把固定中央裁剪当作更强教师。该80条是选择过的诊断集，不是总体准确率基准，也不进入本轮训练。

视觉侧20k续训相对joint在Action开发集修复123/退化160，在Scenery修复29/退化31，未替换joint基线。新入口从 `outputs/mpp_lingoqa_projector_lora_1epoch_v1/final_adapter` 和 `final_projector.pt` 初始化，不从退化的Q-Former续训权重开始。

本轮不是“更高清教师”的OPD，也不是Agent：固定同一MPP教师以训练参考答案作为额外上下文，指导学生自己采样的回答前缀。条件教师容易答对是预期现象，不能证明视觉能力变强。价值必须由不带答案提示的学生开发评测证明，并与同数据继续SFT比较。

## 模型与数据

学生：原始五帧Q-Former特征 → 可训练projector → Qwen LoRA → 自由回答。

教师：同一五帧特征 → 独立冻结projector → 冻结teacher LoRA → 问题加第一条训练参考答案的提示。

- EVA和Q-Former固定，复用 `/tmp/zjr_mpp_fast/cache/lingoqa_5f_qformer_v1`；没有裁剪、伪造帧或修改原始QA。
- 共享冻结Qwen基座，teacher/policy LoRA独立；teacher projector独立。只训练语言LoRA与projector。
- 学生任务说明 `legacy_v1` 不变。hint协议：`full_frames_first_reference_teacher_only_v1`。
- 教师hint仅用于训练候选生成与训练教师打分。学生评测入口不接收hint。

23:01:34 UTC已导出 `datasets/processed/lingoqa_answer_opsd_v1/probe.jsonl`：2000条QA、2000段，Action1290/Scenery710，与dev/eval视频段交叉0，缺失Q-Former缓存文件0。缓存存在性全量检查，内容在加载时按已有协议检查；没有重新校验全部特征值。

源全量训练SHA256：`dfc14f194da236ed9e35484a446144fe29425d7f3f71c7fab44c3ff3aa0c619e`。
候选SHA256：`91d68c9ef5d718fec6592a2ab5d29dd3a6acde9cf2a9f314872194687db245c5`。
其余split哈希见数据目录 `manifest.json`。

## 筛选、损失与对照

1. 学生和条件教师每题各随机采样4次：temperature=1、top_p=1、top_k=0；不是重复greedy。`question_id + sample_id` 唯一，逐draw固定seed，可继续未完成的生成。
2. 用官方LingoJudge协议评分。要求4次教师回答均EOS结束且Judge判对；缺失或重复采样拒绝筛选。
3. 在Action/Scenery配额内优先选师生成功率差较大的题；约30%配入二者都稳定答对的回放题。最多1000条，实际可能不足。
4. 训练时重新让当前学生采样。教师与学生对同一学生token序列打分，分别按自己的提示长度对齐；没有decode/re-encode，也不拿教师生成答案冒充学生rollout。
5. 默认回答token全词表分块JSD + 0.2×原始答案CE回放。EOS参与，提示不参与；截断rollout跳过蒸馏但保留CE，不补造EOS。
6. 教师no_grad forward完成并恢复policy后才执行学生forward/backward，避免共享PEFT与梯度checkpoint重算冲突。

`sft_control` 使用同一选中数据、初始化、可训练参数、学习率、样本顺序和更新次数，只做原始答案CE（权重1），不加载教师、不采样。匹配的是更新次数，不是耗时。

默认4卡、每卡1条、梯度累积4、1epoch、LR=2e-6、max_steps=200（上限）。若选中1000条，每rank250条，实际 `ceil(250/4)=63` 次更新，不会为凑200步自动重复训练。DistributedSampler必要时补齐尾部样本。保存adapter/projector、优化器、各rank RNG及数据/教师身份，可从checkpoint恢复。

## 代码清单

| 文件 | 改动 |
|---|---|
| `mpp_video_qa/observations.py` | teacher-only答案上下文及版本，保留旧crop |
| `mpp_video_qa/opd.py` | 采样支持独立提示，保留原token和EOS |
| `scripts/compare_teachers.py` | 多次随机采样、训练hint、逐draw恢复与身份 |
| `scripts/prepare_answer_opsd.py` | 训练候选抽样与视频段隔离 |
| `scripts/prepare_opd_data.py` | 多采样筛选、来源配额、差距优先及回放 |
| `scripts/train_opd.py` | hint教师、同缓存复用、匹配SFT控制组、进度与身份 |
| `scripts/evaluate_answers.py` | 评分进度文件；评分规则不变 |
| `scripts/report_answer_opsd.py` | 采样/评分/训练状态，完整同题paired报告 |
| `scripts/run_answer_opsd.sh` | 独立统一入口 |
| `scripts/smoke_answer_opsd.py` | 真实权重2步、累积2、教师冻结验证 |
| `tests/test_answer_opsd.py` | 提示/数据隔离、采样、EOS、筛选CLI和报告测试 |

## 命令

无需下载或重新缓存。候选prepare已经完成，重复运行会检查内容一致性。选择空闲GPU，不要同时前台和后台运行同一个probe。

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
bash scripts/run_answer_opsd.sh prepare
CUDA_VISIBLE_DEVICES=0 bash scripts/run_answer_opsd.sh probe
```

`probe`依次生成学生8000条和教师8000条回答、评分、筛选；这是训练数据准备，不是训练。后台版本：

```bash
systemd-run --user --unit=mpp-answer-opsd-probe-v1 --collect \
  --property=WorkingDirectory=/home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa \
  --setenv=CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_answer_opsd.sh probe
```

监视（用bash执行sh）：

```bash
watch -n 10 'bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_answer_opsd.sh status'
journalctl --user -fu mpp-answer-opsd-probe-v1.service -o cat
```

状态来自文件，不验证进程存活。选择报告在 `datasets/processed/lingoqa_answer_opsd_v1/train.manifest.json`；无稳定教师胜例时明确失败，不绕过筛选自动训练。

筛选完成后顺序训练，避免同时抢占同四卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29675 \
  bash scripts/run_answer_opsd.sh train_control
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29675 \
  bash scripts/run_answer_opsd.sh train
```

输出分别在 `outputs/mpp_lingoqa_answer_opsd_sft_control_v1`、`outputs/mpp_lingoqa_answer_opsd_v1`；最终模型在 `final/adapter`、`final/projector.pt`。后台照probe使用systemd-run，换unit名、四卡环境及末尾train/train_control即可。

恢复仅用于未完成运行，保持原始数据/配置，将 `OPSD_RESUME_FROM` 设为实际存在的checkpoint目录。10月2日这轮只运行47步，小于save_every50，只有final包，不存在 `checkpoint-000050`，也无需恢复已完成任务。调整实验参数或数据时必须更换输出目录，不能混写。

开发集自由评测（joint、SFT控制组、OPSD，各Action3120/Scenery1200题）：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_answer_opsd.sh eval_dev
bash scripts/run_answer_opsd.sh report
```

支持单独 `eval opsd action` 等入口。报告在 `outputs/answer_opsd_eval_v1/report.json`、`report.md`，含joint对两种续训以及SFT对OPSD的fixed/regressed。默认不反复用官方500题调参；开发方案固定后可运行：

```bash
for model in joint sft_control opsd; do
  CUDA_VISIBLE_DEVICES=0 bash scripts/run_answer_opsd.sh eval "$model" eval
done
bash scripts/run_answer_opsd.sh report
```

旧500题已多次查看，不能重新称为完全未见的独立测试集。可调环境变量：`OPSD_PROBE_SIZE`、`OPSD_NUM_SAMPLES`、`OPSD_TRAIN_SIZE`、`OPSD_EPOCHS`、`OPSD_MAX_STEPS`、`OPSD_LR`、`OPSD_PROJECTOR_LR`、`NPROC_PER_NODE`、`GRAD_ACCUMULATION`、`SEED`；自定义路径见Shell开头。

## 验证结果与限制

真实短测：`outputs/answer_opsd_real_smoke_v1/summary.json`。

- 2次更新，每次累积2条；4条rollout均EOS结束，长度13/26/12/24。
- JSD=0.0566950/0.0473870/0.0149880/0.0275705；不同样本，不解读为收敛趋势。
- 教师LoRA/projector无梯度且权重不变；学生两分支最大变化约4.00e-6。
- 峰值allocated15.02GiB，43.53秒；不代表所有长回答的显存上限。
- 短测直接取训练候选，不绕过正式筛选供质量训练；未保存或覆盖正式模型。

10月1日交付时23项unittest、4项原core函数、Python编译和Shell语法检查通过，包含真实小型Qwen/Q-Former回归、提示长度对齐及教师冻结。当时四卡正式训练未运行；10月2日已完成，见文首后续归档。工程短测不是准确率证据。

条件教师在错误学生前缀上未必能纠正；文本Judge可能误判动作和方向；筛选可能偏向易题；反复用开发集也有方案选择偏差。最终看无hint学生LingoJudge、各来源fixed/regressed、相对SFT净增益，并抽查视觉事实。若只胜过joint而不胜过SFT，不宣称OPD有独立收益。本版不包含动态ROI、外部大教师、视觉对比蒸馏、RL或Agent。
