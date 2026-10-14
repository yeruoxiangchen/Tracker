# 2026-10-02：驾驶问答事实偏好训练

本轮从已验证的 joint LoRA＋projector 出发，实现离线 DPO＋匹配 CE 对照。输入为原始完整五帧，任务提示固定 `legacy_v1`，不向学生或参考模型提供参考答案提示。模型仍为官方 MPP-Qwen-Next / Qwen7B。

## 1. 为什么进行这轮实验

上一轮完整开发评测已完成（`outputs/answer_opsd_eval_v1/report.md`）：

| 模型 | Action 3120题 | Scenery 1200题 | 合计4320题 |
|---|---:|---:|---:|
| joint 重评 | 51.5064% | 60.6667% | 54.0509% |
| SFT控制 | 50.9936% | 60.5833% | 53.6574% |
| 答案提示OPD | 50.5128% | 60.2500% | 53.2176% |

OPD相对joint修复170题、退化206题，净少36题；相对控制少19题。保留joint，不扩大该OPD配方。这些是开发结果，不能替代官方500题结果；本轮没有补跑官方评测。

学生2000题×4次采样中，0/4、1/4、2/4、3/4、4/4通过的题数分别为491、420、340、310、439。1070题同时存在Judge通过与未通过的回答，但不代表1070对都是可靠视觉事实负例。

## 2. 数据构建及实际规模

输入：

- `datasets/processed/lingoqa_answer_opsd_v1/probe.jsonl`：2000道原始训练题；
- `outputs/answer_opsd_probe_v1/student/scored/scored_predictions.jsonl`：8000条学生回答；
- `datasets/processed/lingoqa_full_v2/{train,dev,eval}.jsonl`：原始身份和视频隔离核对。

默认筛选要求：完整结束、正确回答Judge logit ≥1、错误回答 ≤−1、文本不重复、长度比不超过2.5；参考与chosen必须具有相同且单一的可抽取事实，rejected为相反事实。支持转向、明确ego主语的动作、信号颜色和车辆类型；含否定/假设、参考冲突或仅措辞差异的自动候选排除。每题最多1对。

自动规则验证的是**文本与参考的一致性，不是人工视觉真值**。行中记录 `visual_review_status=not_reviewed`，没有伪造人工审核。`review_selected.html`内嵌完整五帧的缩略图，可直接用浏览器打开，无需服务器。`review_candidates.jsonl`保留自动未采纳的候选。

2026-10-02 13:23:53 UTC生成：

| 数据 | 数量 |
|---|---:|
| 事实偏好对 | 120 |
| 稳定正确题的原始参考答案回放 | 40 |
| 合计 | 160 |
| 偏好对Action / Scenery | 108 / 12 |
| 总体Action / Scenery | 108 / 52 |
| 转向 / ego动作 / 信号颜色 / 车辆类型 | 62 / 25 / 11 / 22 |
| 已人工逐图验证 | 0 |

偏好对偏向Action，回放优先补充来源缺口，最终总体为67.5% Action、32.5% Scenery。DPO项本身仍偏向Action，不能只展示整体比例掩盖这一点。

数据目录为 `datasets/processed/lingoqa_preference_v1`；`train.jsonl` SHA-256：
`b78acdb1c901113fb049f2a3d94e4fa0cb8bd87e96d146f8ecc2cc70bb9f0d5a`。
初次构建时回放独立按来源抽样，使整体偏为134/26；修正为按总体缺口回放。初次产物保留在 `outputs/preference_prepare_initial_unbalanced_v1`，不用于训练。

可以提供JSONL复核决定，每行示例：

```json
{"example_id":"候选文件中的完整ID","approved":false,"visual_verified":false,"reason":"视觉依据不足"}
```

通过 `PREFERENCE_REVIEW_JSONL=/absolute/review.jsonl PREFERENCE_DATA_DIR=/absolute/new_data_dir bash scripts/run_preference.sh prepare` 生成新版本。未声明 `visual_verified` 时不会记为视觉审核通过。来源数据及筛选配置相同的 `prepare` 可重复执行，验证现有数据后返回；配置改变须使用新目录。

## 3. 模型、概率与损失

固定参考为 `outputs/mpp_lingoqa_projector_lora_1epoch_v1/{final_adapter,final_projector.pt}`，同时作为学生初始化。复用 `[5,32,768]` Q-Former缓存，训练Qwen `c_attn` LoRA和projector，冻结EVA、Q-Former及语言基座。

回答文本统一编码，追加恰好一个`im_end`。参考与学生评分完全相同的token序列；只累计回答和EOS，不累计prompt/视觉位置/padding。采用回答token log-prob求和，fp32交叉熵、按8 token分块，保留投影层和语言LoRA梯度。

```
margin = beta * [(logπ(chosen)-logπ(rejected)) - (logref(chosen)-logref(rejected))]
L_dpo = -logsigmoid(margin)

偏好样本：DPO组 = L_dpo + 0.2 * mean_CE(chosen)
          对照组 =         0.2 * mean_CE(chosen)
回放样本：两组均 =         0.2 * mean_CE(original_reference)
```

两组使用相同数据顺序、初始化、CE权重、学习率、梯度累积及更新预算。固定参考先单独计算log-prob缓存，训练不常驻第二份7B模型，也不在线采样。概率缓存绑定数据、回答token、tokenizer、原joint权重及选中视觉缓存的哈希。

新增路径中，参考评分与学生均使用bf16冻结基座＋fp32 LoRA/projector。复用加载器后重新加载小型adapter/projector的原始fp32值，避免经过bf16转换后再升回fp32导致断点更新舍入；未修改历史训练或评测入口。

默认4卡、每卡1条记录、梯度累积4、3epoch、LoRA/projector LR均1e-6、beta0.1、dropout0、梯度裁剪1.0。当前160条因此每epoch10步、共30步；`max_steps=0`表示按epoch完成。`max_pairs=1000`是上限，不代表实际有1000对。每epoch及最终均保存可恢复bundle，另支持按步保存。

## 4. 代码与命令

| 文件 | 功能 |
|---|---|
| `mpp_video_qa/preference.py` | 数据契约、回答概率、DPO损失、参考缓存身份 |
| `scripts/prepare_preference_data.py` | 偏好筛选、来源回放、检查页面 |
| `scripts/preference_common.py` | 模型构建、fp32加载、joint身份 |
| `scripts/cache_preference_reference.py` | 固定参考离线评分，可续算 |
| `scripts/train_preference.py` | 四卡DPO/匹配SFT、梯度累积、断点恢复 |
| `scripts/run_preference.sh` | 全流程入口 |
| `scripts/report_preference.py` | 进度、成对对比、按视频bootstrap区间 |
| `tests/test_preference.py` | 概率位置、梯度、泄漏、身份与筛选检查 |

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa

bash scripts/run_preference.sh prepare
CUDA_VISIBLE_DEVICES=0 bash scripts/run_preference.sh reference

CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29681 \
  bash scripts/run_preference.sh train_control
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29681 \
  bash scripts/run_preference.sh train_dpo

CUDA_VISIBLE_DEVICES=0 bash scripts/run_preference.sh eval_dev
bash scripts/run_preference.sh report
```

以上卡号应选空闲GPU。`reference`完成后重跑只校验并返回。每个模型训练输出已有完成结果时不覆盖，需修改`PREFERENCE_TRAIN_DIR`或`PREFERENCE_CONTROL_DIR`进行新实验。

后台依次计算参考及训练两组：

```bash
systemd-run --user --unit=mpp-lingoqa-preference-v1 --collect \
  --property=WorkingDirectory=/home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa \
  --setenv=CUDA_VISIBLE_DEVICES=0,1,2,3 --setenv=MASTER_PORT=29681 \
  /usr/bin/bash -c 'bash scripts/run_preference.sh reference && bash scripts/run_preference.sh train_control && bash scripts/run_preference.sh train_dpo'
```

监视（在任意目录）：

```bash
watch -n 10 bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_preference.sh status
journalctl --user -fu mpp-lingoqa-preference-v1.service -o cat
```

`status`是文件进度，明确标记未验证进程存活，并显示最近更新时间。恢复未完成的DPO示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PREFERENCE_RESUME_FROM=/absolute/path/checkpoint-000050 \
  bash scripts/run_preference.sh train_dpo
```

当前30步正式配方不会产生step50；实际可用断点是`epoch-01`、`epoch-02`等，应填写真实存在且预算未完成的路径。恢复必须保持模型、数据、参考分数、world size和优化参数一致。

开发评测可选epoch，但应另设评测输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 PREFERENCE_CHECKPOINT=epoch-01 \
PREFERENCE_EVAL_DIR=outputs/preference_eval_epoch01_v1 \
  bash scripts/run_preference.sh eval_dev
```

主报告包含完整4320题加权汇总、Action/Scenery、fixed/regressed及按segment重采样的置信区间。评测使用原问题和五帧，不带GT提示。训练偏好胜率不是任务正确率。官方500题可在开发决策后用 `eval MODEL eval` 运行；不自动反复测试。

## 5. 验证与运行记录

截至2026-10-02 13:26 UTC：10项新增偏好测试通过；全项目unittest discover共33项通过；Python编译和Shell语法通过。核心覆盖回答/EOS位置、分块log-prob梯度、DPO梯度正负方向、视觉梯度、训练/开发视频与帧隔离、参考缓存篡改、复核排除、视频级bootstrap。

真实参考评分第一次在沙箱内加载权重后遇到 `RuntimeError: No CUDA GPUs are available`，是该执行环境无法初始化CUDA；当时主机`nvidia-smi`显示8张卡空闲。随后获准在沙箱外GPU执行，原日志保留 `logs/preference_reference_v1.log`，成功重试日志为 `logs/preference_reference_v1_gpu.log`。

### 2026-10-02 GPU验证（2026-10-03 00:07 UTC续查归档）

| UTC完成时间 | 验证 | 结果 | rank0峰值allocated |
|---|---|---|---:|
| 13:29:39 | 固定参考评分 | 160/160，回答含EOS，无hint | 14.52 GiB |
| 13:40:21 | 四卡DPO，累积4，两步 | 每rank 8个micro steps，参数更新、四卡校验一致 | 14.92 GiB |
| 13:42:37 | 四卡匹配SFT，同预算 | 参数更新、四卡校验一致 | 14.79 GiB |
| 13:46:29 | DPO从第一步恢复至第二步 | 步数与优化器恢复，可继续更新 | 14.93 GiB |

短测输出分别为 `outputs/preference_smoke_4gpu_v1`、`outputs/preference_control_smoke_4gpu_v1`、`outputs/preference_resume_smoke_4gpu_v1`。两步DPO的LoRA层0及projector最大更新约2.00e-6，所有rank的可训练参数总和差为0。参考概率初始核对通过每token差异不超过0.02的检查，但不是逐位相等；第一步平均DPO损失0.693898，接近相同policy/reference时的log(2)。这只验证数值链路，不代表任务收益。

断点恢复的最终64个adapter张量相对连续运行最大差1.65333e-6，projector最大差1.55112e-6。因此仅声称**可恢复运行，不声称逐位可复现**；差异原因尚未单独隔离，不从该短测推断是否影响最终正确率。对比落盘为 `outputs/preference_resume_smoke_4gpu_v1/resume_comparison.json`。本轮默认30步正式试验可连续运行。

短测没有写入正式DPO/SFT目录，也没有启动完整开发集或官方500题评测。正式模型效果仍未知。完成后再次运行全项目33项unittest及Python/Shell语法检查，均通过；进度命令已用真实短测产物核对。

方法参考：[DPO原论文](https://arxiv.org/abs/2305.18290)。本实现是DPO＋CE约束，不是GRPO或在线OPD。
