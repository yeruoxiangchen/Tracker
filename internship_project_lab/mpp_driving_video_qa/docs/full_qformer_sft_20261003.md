# 2026-10-03：完整 Q-Former 联合 SFT

## 当前状态与目的

在当前最佳 joint LoRA＋projector 上，开放 MPP 现有 Q-Former 全部参数和 query tokens，
用更多原始驾驶 QA 进行视觉适配。EVA 和视觉归一化层仍冻结，Qwen 基座仍冻结。
输入维持原始完整五帧和 `legacy_v1`，监督维持回答及 EOS 的交叉熵。
本轮不使用答案提示、教师生成标签或偏好损失。

2026-10-04补录：正式100k训练、开发和官方500题评测均已完成，epoch-02成为当前最佳。
开发集joint/epoch-01/epoch-02为53.89%/56.64%/57.22%；官方joint/epoch-02为47.2%/51.0%。
具体时间和成对统计见主时间线第30节。后续[全量规模实验](full_qformer_scale_20261004.md)已实现但未训练。
下文验证章节中的“未启动”保留为当时交付时状态，不代表当前状态。

## 数据与配置

来源：`datasets/processed/lingoqa_full_v2/train.jsonl`；输出：
`datasets/processed/lingoqa_qformer_full100k_v1`。

- 100,000 条原始 QA，Action 64,500、Scenery 35,500。
- 覆盖全部27,599个训练片段：Action 24,191、Scenery 3,408。
- 来源比例分层，每来源先覆盖不同片段，再逐轮补充同片段问题；自适应上限为每段11题。
- 保留原问题、原答案、原五帧；dev/eval视频隔离及原拆分不变。
- seed=42，train SHA-256：`a43bc099f51c456a917d7a30266530f008d991772e66ce66650b59b828deff72`。
- 四卡、每卡batch1、累积4、2epoch，共12,500次更新。
- Q-Former LR=1e-5；projector及语言LoRA LR=5e-6；沿用现有warmup/线性衰减。
- Qwen与Q-Former启用梯度检查点。默认每2000步保存可恢复checkpoint，每epoch保存评测权重。

初次核对EVA缓存：需要28,099段（含开发/官方评测），已有16,808段，缺11,291段。
现有缓存约57GiB，补齐额外约38GiB；使用 `/tmp/zjr_mpp_fast/cache/lingoqa_5f_eva_v1`。
必须先完成 `cache`，不能把旧的Q-Former输出缓存当作EVA输入。缓存状态中的旧
rank_progress/final_summary可能来自历史小任务，应以当前`missing`及本次cache日志为准。

## 模型、精度和权重

`VisualBridge`增加`frozen/lora/full`模式，默认`auto`维持历史适配。
`full`不注入视觉LoRA，训练现有self/cross-attention、query专用FFN、LayerNorm及query tokens。
不恢复MPP已移除的文本embedding、文本FFN和分类头。

新增`--precision-protocol fp32_trainables_v1`：Qwen冻结基座BF16；视觉桥和projector保存
FP32参数，Q-Former采用BF16 autocast计算，projector FP32计算后转为语言激活类型；语言
LoRA升至FP32后重新载入原始checkpoint，避免PEFT初始化时舍入。训练和自由评测采用同一
路径。历史入口默认`legacy`，不会把新预测追加到旧精度的输出目录。

完整Q-Former保存格式为`mpp_visual_bridge_full_v1`，包含全部Q-Former及query参数。
为复用已有bundle命名，文件仍叫`qformer_adapter.pt`或`final_qformer_adapter.pt`，**内容是
完整权重**。断点还含语言adapter、projector、优化器、scheduler及各rank RNG状态。
恢复校验模式、精度、数据身份和总步数。各epoch目录用于评测；恢复训练使用
`checkpoint-NNNNNN`，不使用只含评测权重的epoch目录。

`ask_video.py`在新精度协议下在线计算EVA特征，并复用`CachedVideoQA`的视觉替换、
prompt和生成路径；EVA输出按缓存协议转BF16后再进入可训练桥。

## 实现文件

| 文件 | 改动 |
|---|---|
| `mpp_video_qa/visual_bridge.py` | 完整训练、AMP、梯度检查点、完整权重读写及旧adapter兼容 |
| `mpp_video_qa/model.py` | 各分支精度、FP32加载、完整桥接入 |
| `scripts/train_sft.py` | full模式、恢复身份、全rank累积平均loss、分支梯度与权重变化、rank参数校验 |
| `scripts/generate_answers.py` | 精度身份、完整权重评测、峰值显存 |
| `scripts/ask_video.py` | 原视频与缓存评测统一路径 |
| `scripts/prepare_full_qformer_sft.py` | 独立100k原题抽样，无旧教师诊断依赖 |
| `scripts/run_full_qformer_sft.sh` | 数据、缓存、训练、评测、状态及视频问答入口 |
| `scripts/report_full_qformer.py` | 同精度/数据/权重核对，分来源、成对变化、视频bootstrap及dev选模 |
| `scripts/cache_visual_features.py` | 修正EVA summary的hidden_size为1408（之前误写4096；缓存张量本身未改） |
| `tests/test_full_qformer.py` | 完整梯度、checkpoint/reload、精度、覆盖抽样和报告验证 |
| `scripts/verify_full_qformer_smoke.py` | 汇总真实短测、恢复及重新加载验证，并实测权重差异 |

## 正式执行

以下卡号运行前应确认空闲。数据已构造；`prepare`可重复核对。

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
bash scripts/run_full_qformer_sft.sh prepare
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29684 bash scripts/run_full_qformer_sft.sh cache
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29685 bash scripts/run_full_qformer_sft.sh train
```

后台依次补缓存、训练：

```bash
systemd-run --user --unit=mpp-lingoqa-full-qformer-v1 --collect \
  --property=WorkingDirectory=/home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa \
  --setenv=CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /usr/bin/bash -c 'bash scripts/run_full_qformer_sft.sh cache && bash scripts/run_full_qformer_sft.sh train'
```

监视（可在任意目录执行；`.sh`由bash运行）：

```bash
journalctl --user -fu mpp-lingoqa-full-qformer-v1.service -o cat
watch -n 10 bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_full_qformer_sft.sh cache_status
watch -n 10 bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_full_qformer_sft.sh status
```

训练完成后，单卡依次评测joint及两个epoch，自动生成报告：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_full_qformer_sft.sh eval_dev
bash scripts/run_full_qformer_sft.sh report
CUDA_VISIBLE_DEVICES=0 bash scripts/run_full_qformer_sft.sh eval_selected
```

报告为`outputs/full_qformer_eval_v1/report.{json,md}`。开发集4320题按Action/Scenery拆分，
报告fixed/regressed及按视频bootstrap区间。按dev点估计选模，平局保留较早模型；统计区间
跨零时不宣称稳定提升。`eval_selected`只评估joint与开发集选中模型的官方500题；官方结果
落各自`*_eval/scored/summary.json`，不用于重新选模型。

原视频问答示例（替换视频路径和问题；模型可用joint、epoch-01、epoch-02、final）：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_full_qformer_sft.sh ask epoch-01 /path/to/clip.mp4 'Why is the ego vehicle slowing down?'
```

恢复示例（只在该checkpoint真实存在且训练未完成时）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
SFT_RESUME_FROM=outputs/mpp_lingoqa_qformer_full100k_2epoch_v1/checkpoint-002000 \
  bash scripts/run_full_qformer_sft.sh train
```

可选同预算冻结Q-Former对照：`QFORMER_MODE=frozen`并指定独立`FULL_QFORMER_TRAIN_DIR`和
`FULL_QFORMER_EVAL_DIR`。主实验相对旧joint同时改变训练范围与训练预算，若归因于完整
Q-Former本身，需要这个同数据同预算对照。当前没有启动该对照。

## 验证记录

新增真实小型Q-Former测试确认cross-attention key和query-FFN都有非零梯度，全部参数均
有梯度；完整保存再加载后的CPU输出逐位一致。FP32投影在训练/评测加载后逐位保留。
GPU短测使用训练集已有缓存的长文本64条，四卡、累积4、两次更新，独立输出目录
`outputs/full_qformer_smoke_4gpu_v1`，日志`logs/full_qformer_smoke_4gpu_v1.log`。
37项unittest、Python编译和Shell语法检查通过。不能把短测当作准确率提升。

### 2026-10-03 GPU验证（UTC按summary落盘时间）

| 时间 | 验证 | 结果 |
|---|---|---|
| 10:47:15 | 四卡、累积4、连续两步 | 成功；每rank 8 micro steps；rank间参数checksum差0 |
| 10:54:44 | 重新加载完整视觉桥、projector和LoRA，生成2条回答 | 2/2成功，推理峰值allocated15.23GiB |
| 10:56:42 | 从checkpoint-000001恢复到第2步 | 成功；rank间checksum差0 |

训练参数：语言LoRA 8,388,608、projector 3,149,824、Q-Former及queries 105,162,240。
所有rank的64个LoRA参数张量、2个projector张量、255个Q-Former/query张量都有非零梯度。
连续短测的语言LoRA/projector/桥内query投影最大更新分别约1.00e-5、1.00e-5、1.99e-5。
rank0峰值allocated16.93GiB、reserved22.23GiB；其他rank日志中的reserved峰值达23.08GiB。
四卡3090短测通过，但不由两步结果保证任意长度/外部GPU占用下均无OOM。

恢复与连续运行最终权重最大差：Q-Former1.64995e-5、projector8.09133e-6、语言LoRA
8.43656e-6；第二步loss分别1.08521与1.08588。数据指纹、训练参数、步数和rank同步一致，
但**恢复并非逐位复现**，差异来源未单独隔离，不宣称对准确率无影响。默认正式命令连续
训练；恢复功能可用，若实际恢复应保留这一事件记录。CPU完整权重保存/加载测试逐位一致，
不等同于GPU续训轨迹逐位一致。结果保存在
`outputs/full_qformer_resume_4gpu_v1/verification.json`。

短测保留约3.5GiB、恢复验证约2.2GiB产物，均在项目outputs下，不占用新的/tmp模型副本。
没有删除用户已有缓存或模型，没有启动正式训练。
