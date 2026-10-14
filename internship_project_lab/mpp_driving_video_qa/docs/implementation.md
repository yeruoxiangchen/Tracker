# 当前稳定实现与复现协议

> 更新：2026-09-28 UTC。完整历史、故障与实验结果见
> [project_timeline.md](project_timeline.md)。本文描述首轮已完成结果和当前 full-v2
> 有效入口，不把旧失败命令当作推荐入口。

## 任务范围

模型接收短驾驶片段的 5 张按时间排序图像和一个自然语言问题，生成简洁自由文本
答案。基线为官方 MPP-Qwen-Next SFT checkpoint，领域训练只对 Qwen-7B attention
`c_attn` 注入 LoRA。视觉编码器、Q-Former、视觉投影和 Qwen base 冻结。

本项目不预测时间边界、不排序视频片段、不使用车辆隐式 telemetry，也不加载兄弟
项目的 Grounder 或 Verifier。

## 数据协议

首轮已完成数据来源：

- train/dev：`/data/zjr/datasets/lingoqa/action/train.parquet`；
- held-out eval：`/data/zjr/datasets/lingoqa/evaluation/val.parquet`；
- 图像：对应 `action/extracted` 与 `evaluation/extracted`；
- 本轮未将 scenery 加入训练。

已生成：train 30,000 问 / 17,469 segments，dev 3,120 问 / 300 segments，eval
500 问 / 100 segments。train/dev/eval 的 segment ID 两两不重合，每条记录严格包含
5 帧。

复现转换：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa

PYTHONPATH=. /home/zjr/anaconda3/envs/mpp_llava/bin/python \
  scripts/prepare_lingoqa.py \
  --train-input action=/data/zjr/datasets/lingoqa/action/train.parquet \
  --eval-input /data/zjr/datasets/lingoqa/evaluation/val.parquet \
  --frames-root-map action=/data/zjr/datasets/lingoqa/action/extracted \
  --frames-root-map evaluation=/data/zjr/datasets/lingoqa/evaluation/extracted \
  --output-dir datasets/processed/lingoqa_v1 \
  --train-size 30000 \
  --dev-segments 300 \
  --max-qa-per-segment 12 \
  --check-files
```

转换后以 `datasets/processed/lingoqa_v1/manifest.json` 为准，不应根据命令文字猜测
实际数据组成。

当前 full-v2 已加入完整可用 Action 与 Scenery 监督，并固定保留旧 Action dev：

| split | 问答 | segment |
|---|---:|---:|
| train | 406,212 | 27,599 |
| dev | 4,320 | 400 |
| eval | 500 | 100 |

训练为 Action 262,005 + Scenery 144,207；dev 为旧 Action 3,120 + 固定 Scenery
1,200。清单和指纹位于 `datasets/processed/lingoqa_full_v2/manifest.json`。入口为：

```bash
bash scripts/run_full_v2.sh prepare
bash scripts/run_full_v2.sh cache
bash scripts/run_full_v2.sh cache_status
bash scripts/run_full_v2.sh train
bash scripts/run_full_v2.sh status
```

## 模型与视觉缓存

每个 segment 的 5 帧只编码一次。冻结 MPP 视觉路径输出 `[5,32,4096]`，同一
segment 的不同问题共享缓存。缓存实现不会构造完整 Qwen，也只从 MPP shards 读取
视觉投影层所需权重。首轮缓存包含 17,869 个唯一 segment，约 22 GB；full-v2 所需
28,099 个训练/dev/eval 唯一 segment 已完成增量缓存，并已用于全量训练与评测。

full-v2 允许 BLIP2 视觉权重和 MPP 投影 shards 使用不同根目录。默认从持久项目目录
读取 BLIP2、从 `/tmp/zjr_mpp_fast` 读取 MPP shards；这是首次缓存启动失败后的修复。

训练 prompt 中，system、问题、视觉占位 token 和 padding 均被 mask，只监督参考
答案与 assistant 结束标记。生成 prompt 必须保持 assistant turn 开放，不能复用会
自动追加 `<|im_end|>` 的训练预处理。

## SFT 配置与已完成结果

### 首轮 Action 30k

```text
LoRA target: c_attn
r / alpha / dropout: 16 / 32 / 0.05
learning rate: 2e-5
epochs: 3
world size: 4
batch size: 1 per rank
gradient accumulation: 4
optimizer steps: 5625
gradient checkpointing: enabled
```

完成产物：`outputs/mpp_lingoqa_lora_gc_full_v1`。训练耗时约 99.3 分钟，LoRA 最大
参数变化 `0.00805494`；dev teacher-forced loss 为 0.41455 → 0.37717 → 0.36229。

训练入口仍由：

```bash
bash scripts/run_pipeline.sh cache
bash scripts/run_pipeline.sh train
```

控制项通过 `SFT_MAX_STEPS`、`SFT_MAX_DEV_RECORDS`、`SFT_EPOCHS`、
`SFT_GRAD_ACCUMULATION` 和 `SFT_GRADIENT_CHECKPOINTING` 传入。共享盘很慢时可使用
`/tmp/zjr_mpp_fast` 的可再生权重与缓存副本；当前副本约 52 GB。

### full-v2 Action + Scenery

full-v2 已完成一轮四卡训练：406,212 QA、25,389 optimizer steps、101,553 micro
steps，耗时约7小时13分钟。LoRA参数与首轮相同，采用bf16、gradient checkpointing、
每卡batch 1、累积4、lr 2e-5、legacy_v1 prompt。产物为
`outputs/mpp_lingoqa_full_v2_lora_1epoch`，包含step 5k至25k每5k一次的恢复状态、
epoch adapter与final adapter。恢复状态已保存但未做真实中断恢复一致性实验。

## 生成协议

生成同时传入 prompt `input_ids` 和已替换视觉位置的 `inputs_embeds`，只解码 prompt
后的 completion。当前采用 greedy decode；Base/SFT 对比必须保持帧、prompt、
`max_new_tokens` 与解码参数一致。

任务说明有两个版本：

- `legacy_v1`：历史默认协议；
- `driving_perspective_v1`：明确图像来自 ego vehicle 前向摄像头，`you/your` 指自车
  或驾驶员。

后者只用于 2026-09-28 的 dev 消融。它使 Epoch 3 随机 dev Judge 从 46.48% 降到
41.80%，已经拒绝；`legacy_v1` 继续作为 full-v2 训练和评测协议。

标准入口：

```bash
bash scripts/run_pipeline.sh eval_base
bash scripts/run_pipeline.sh eval_sft
bash scripts/run_pipeline.sh score_base
bash scripts/run_pipeline.sh score_sft
```

## 评分协议

`scripts/evaluate_answers.py` 实现三类指标：normalized exact、token F1、可选
Lingo-Judge。Judge 与官方协议对齐：

- reference 和 prediction：`str(...).lower().strip()`；
- question 保留原文；
- tokenizer `max_length=128`；
- 多参考答案取最大 positive-class logit；
- `logit > 0` 判为正确。

Judge 是文本代理指标而非人工视觉真值。完整 official evaluation 的已确认结果是：
Base 42.2%，首轮Epoch 3 SFT 37.4%，full-v2 SFT 44.6%；token F1分别为
0.21843、0.28543、0.31319。新模型相对Base修复124/退化112，相对旧SFT修复99/
退化63。full-v2 结果位于 `outputs/full_v2_eval/`。

不要再使用旧 `scored/` 中缺少官方预处理的 39.6%/34.6% 作为最终结果；正确目录为
`scored_official_v2/`。

## 诊断入口

固定开发集诊断：

```bash
PYTHONPATH=. /home/zjr/anaconda3/envs/mpp_llava/bin/python scripts/diagnose_sft.py status
```

完整结果已生成于 `outputs/sft_diagnosis_v1/report.{json,md}`。随机 dev 512 题中，
Base/Epoch 1/Epoch 2/Epoch 3 Judge 分别为 36.13%/44.53%/44.53%/46.48%。

任务说明对照已完成，状态入口仍可用于查询归档结果：

```bash
watch -n 10 '/home/zjr/anaconda3/envs/mpp_llava/bin/python \
  /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/evaluate_task_instruction.py status'
```

报告位于 `outputs/task_instruction_ablation_v1/report.{json,md}`。

full-v2 的48段动作诊断入口：

```bash
/home/zjr/anaconda3/envs/mpp_llava/bin/python scripts/diagnose_full_sft.py status
```

结果位于 `outputs/full_v2_action_diagnosis_v1/`。它比较三模型的原问、等义改写和格式
要求，并提供240帧离线视觉复核页面；这些开发集诊断不得转为训练标签或独立测试成绩。

## 本地应用入口

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=.:/home/zjr/Tracker/internship_project_lab/mpp_llava_video_workspace/third_party/MPP-LLaVA \
/home/zjr/anaconda3/envs/mpp_llava/bin/python scripts/ask_video.py \
  --video /path/to/dashcam.mp4 \
  --question "Why is the vehicle slowing down?" \
  --adapter outputs/mpp_lingoqa_lora_gc_full_v1/final_adapter \
  --upstream-root /home/zjr/Tracker/internship_project_lab/mpp_llava_video_workspace/third_party/MPP-LLaVA \
  --checkpoint-root /home/zjr/Tracker/internship_project_lab/mpp_llava_video_workspace/checkpoints
```

## 能力声明边界

- 可以声明：数据转换、MPP 视觉缓存、四卡 LoRA SFT、官方评分与 paired 诊断已形成
  完整工程闭环。
- 可以声明：full-v2 SFT 在Action/Scenery dev上均优于Base与首轮SFT，在official
  evaluation上恢复首轮退化并较Base提高2.4 pp；同时修复Base的部分角色理解错误。
- 不能声明：2.4 pp 已证明稳定、大幅提高跨分布驾驶问答；官方集和dev均已用于实验
  决策，且48题复核仍显示动作、原因、方向与信号状态错误。
- 不能声明：已获得时序理解；还需要单帧或打乱帧顺序的控制实验。
