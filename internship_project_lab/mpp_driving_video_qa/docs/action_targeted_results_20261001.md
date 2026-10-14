# 2026-10-01：定向 SFT 实验、代码与动作诊断归档

本记录承接 [实现方案](action_targeted_sft_20261001.md) 和
[项目时间线](project_timeline.md)。项目目标为基于 MPP-LLaVA 的驾驶视频场景理解与问答。
时间采用 UTC；下列时刻来自产物文件修改时间，代表产物落盘时刻，不等同于任务启动时刻。

## 1. 当前结论

全量数据的 LoRA＋投影层联合 SFT 仍是当前开发集表现最好的方案。
10k 定向续训使 Action 净少答对60题，Scenery净少答对1题，不作为新的主模型。
48段诊断中，同义改写、增加输出格式均没有稳定收益。
这否定了本轮定向续训方案的有效性，不能推导为所有 SFT 或 MPP 架构均无效。

保留主模型：

- `outputs/mpp_lingoqa_projector_lora_1epoch_v1/final_adapter`
- `outputs/mpp_lingoqa_projector_lora_1epoch_v1/final_projector.pt`

本轮实验模型与原始结果保留用于分析，没有删除或覆盖已有模型。

## 2. 实验过程与时间

| 日期/时间（UTC） | 阶段 | 已完成内容 |
|---|---|---|
| 2026-10-01，训练前 | 实现与数据准备 | 新增10k构建器、权重初始化入口、同路径诊断、统一Shell入口；数据及缓存检查通过 |
| 2026-10-01 02:57 | 诊断数据准备 | 固定48段开发视频，构建每模型96条B/C问答 |
| 2026-10-01 03:22 | 定向训练完成 | 625次优化器更新，final adapter与projector及summary已保存 |
| 2026-10-01 04:27 | Action评测完成 | 3120条自由生成、Lingo-Judge评分与联合模型逐题比较 |
| 2026-10-01 04:42 | Scenery评测完成 | 1200条自由生成、评分与逐题比较 |
| 2026-10-01 05:01 | 48段诊断完成 | joint、targeted各96条新增回答；汇总JSON/MD和内嵌帧HTML已生成 |
| 2026-10-01，本次归档 | 结果复核 | 补齐采样构成、负向实验、持续失败及后续教师对照计划 |

## 3. 代码改动

| 文件 | 作用与实现边界 |
|---|---|
| `scripts/prepare_action_targeted_sft.py` | 只在训练split抽样；关键词分组、问句均衡、每视频上限；保守改写；保留原答案和帧；记录provenance及数据指纹 |
| `scripts/train_sft.py` | 新增`--init-adapter`；加载已有可训练LoRA，配合`--projector`初始化投影层；新建优化器与调度器；与恢复同一次实验的`--resume-from`互斥 |
| `scripts/diagnose_action_targeted.py` | 固定48段；复用A原问，生成B改写/C格式；检查权重身份、缓存、数据与生成协议；输出配对统计和HTML |
| `scripts/run_action_targeted_sft.sh` | 串接prepare、diagnose、diagnostic_report、train、eval、status；离线评分；可由systemd直接执行 |
| `tests/test_action_targeted.py` | 覆盖采样上限、确定性、样本不足报错、关键词/模板及修复退化计数 |

实现阶段执行了Python编译、Shell语法、7个unittest以及4个无参数基础测试函数，均通过。
环境没有pytest，未运行pytest全套。本次归档未改训练实现，也未重跑训练。
本轮后续四卡训练已实际完成；参数变化能证明执行了更新，不能单独证明学习效果。

实现和运行过程中沿用的注意事项：

- systemd执行独立`.sh`，避免之前内嵌长Shell的`data: unbound variable`问题。
- 入口设置`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，使用已缓存评分模型。
- 本轮评分日志中的`resume_download`及`TypedStorage`弃用提示没有中断评分。
- 本轮采用Q-Former输出缓存和gradient checkpointing；历史加载慢、显存故障见主时间线，不能归为本轮新故障。
- 原有阶段记录中的“尚未启动GPU训练/诊断”是实现交付时的状态；当前已由本记录更新。

代码SHA-256为本次归档时的文件快照，不代替不可变Git提交：

```text
101d03896d3b890bcce441da429325f76a6b47131b6d5a8d6e84630992640bbe  scripts/train_sft.py
892882d0a3dc36816b10d18edc39d9da5b5366e634f88bf2aa4bdfc9c007272b  scripts/prepare_action_targeted_sft.py
e2d142f7d75af0b5afefaaad6be946071fe07159b2f9d343c01c52b7bdd39752  scripts/diagnose_action_targeted.py
bc0f3cb2365ee4bd493ac9d6c34157cfada410dbb8cfce6f54c181c26d20678f  scripts/run_action_targeted_sft.sh
3315b647dad1567205b6de126ef367bad08242faa5345f8f9fa83083ac184219  tests/test_action_targeted.py
```

## 4. 数据构建及采样局限

目录：`datasets/processed/lingoqa_action_targeted_v1`。

| 类型 | Action来源 | Scenery来源 | 合计 |
|---|---:|---:|---:|
| 重点原始问答 | 2316 | 2684 | 5000 |
| 问句改写 | 1958 | 42 | 2000 |
| 通用回放 | 1500 | 1500 | 3000 |
| 合计 | 5774 | 4226 | 10000 |

覆盖5795段训练视频；种子42；每视频最多4条原始题、含改写最多8条；单一原问句最多300条。
改写题与原题共享答案和五帧，记录原题ID，不是2000条新视觉观察。
train与dev/eval视频和帧路径无重叠，开发/官方评测沿用原split。
所需6295段缓存均存在；抽检8个张量协议，未逐文件全量验证内容。

训练集SHA-256：`ac809e9a83224de115d9ac54081bca8a07b5ff746214881b9f2c0070c1a5d14d`。
`review_sample.jsonl`中100条样本仅导出，尚未完成图像级复核。

关键词采样有实际局限：它按问题中的speed、direction、position、action、why等分组，
没有按真实答案中的加速/减速/停止/转向类别均衡，也没有确保每条都补充关键视觉区别。
重点原题有53.68%来自Scenery；来源不等于任务标签，不能据此认定这些样本无效，
但不能再把该集合描述为经过精确标注、平衡的动作纠错数据集。
这是一项可能影响效果的设计局限，当前没有消融证明它单独导致退化。

## 5. 实际训练配置与完成情况

产物目录：`outputs/mpp_lingoqa_action_targeted_v1`。

| 项目 | 实际值 |
|---|---|
| 初始化 | 上一轮joint final_adapter＋final_projector.pt |
| 可训练部分 | Qwen c_attn LoRA＋768→4096投影层 |
| 缓存 | `/tmp/zjr_mpp_fast/cache/lingoqa_5f_qformer_v1`，5帧Q-Former特征 |
| LoRA | rank16、alpha32、dropout0.05 |
| 训练规模 | 10000题，1 epoch，4卡，每卡batch1、累积4 |
| 学习率 | LoRA及投影层均5e-6，warmup0.03，weight decay0.01 |
| 精度/显存策略 | bf16，gradient checkpointing |
| 实际更新 | 625 optimizer steps，2500 micro steps（每rank） |
| 训练耗时 | 672.89秒，约11.21分钟；不是整个服务含加载的墙钟耗时 |
| 跟踪LoRA参数最大变化 | 0.0004297777 |
| 跟踪投影参数最大变化 | 0.0003862381 |
| rank0峰值allocated/reserved | 16.05 /22.16 GiB |

保存了checkpoint-000200/000400/000600、epoch-01、final_adapter、final_projector.pt。
summary未提供完整loss轨迹，不能用该摘要声称loss收敛或无过拟合。

## 6. 开发集自由生成评测

协议：`official_lower_strip_maxref_logit_gt0_maxlength128_v1`。
四模型在同一split使用相同题目、参考答案、Q-Former缓存、legacy_v1任务说明、batch1及
max_new_tokens96；本轮生成条数与评分条数完整。表中正确率是文本Lingo-Judge判定，
不是人工视觉动作识别准确率。

| 模型 | Action3120 | Scenery1200 | 合计4320 |
|---|---:|---:|---:|
| 官方MPP Base | 36.57%（1141） | 25.17%（302） | 33.40%（1443） |
| 全量LoRA | 48.27%（1506） | 59.75%（717） | 51.46%（2223） |
| 联合SFT joint | 51.47%（1606） | 60.17%（722） | 53.89%（2328） |
| 定向续训 targeted | 49.55%（1546） | 60.08%（721） | 52.48%（2267） |

| 相对joint | Action | Scenery |
|---|---:|---:|
| Judge修复/退化/净变化 | 160 /220 /-60 | 34 /35 /-1 |
| Judge变化（百分点） | -1.9231 | -0.0833 |
| Token F1，joint→targeted | 0.64447→0.63566 | 0.58249→0.57533 |
| Normalized exact，joint→targeted | 8.75%→7.92% | 11.42%→10.83% |

分析时对视频成组重采样，numpy default_rng(42)、10000次bootstrap，按抽中视频包含的
问题数计算加权准确率差：Action差值95%区间[-3.19,-0.64]个百分点；Scenery[-1.25,1.17]。
这是当前开发集与本轮模型的抽样不确定性，不是多随机种子训练方差。

此前同路径官方500题：Base43.0%、全量LoRA45.2%、joint46.4%。targeted尚无官方500题结果。
官方集已反复分析，不描述为完全未见的最终测试集。

### 6.1 退化题型与输出倾向

| 原问句/类型 | 题数 | joint正确 | targeted正确 |
|---|---:|---:|---:|
| Should you stop or brake in this situation? | 124 | 52 | 43 |
| What are you paying attention to and why? | 229 | 87 | 80 |
| Why is it unsafe for you to accelerate in this situation? | 99 | 43 | 38 |
| What is the reason for decelerating? | 58 | 26 | 21 |
| What are you currently doing and why? | 117 | 35 | 33 |

最后一组117题中，回答以“I am decelerating”开头的数量由73增加至90。
该现象说明输出更偏向“减速”，不是已经证明所有这些回答都错误。
对应198条重点原始训练题中，68条答案提及decelerating/slowing/slow down（关键词统计，
不是人工动作分类）；不能简单断言是训练答案全部为减速导致。

逐题文本示例：参考“绿灯且交通有空隙，所以加速”，joint回答加速，targeted回答
“直行红灯，所以减速”。另有左转变成直行/右转、停止避让变成继续行驶等语义变化。
这些提示关注视觉、动作和原因判断，但需要看图复核，不能把Judge判定直接作为视觉真值。

## 7. 固定48段动作诊断

题目来自Action开发集，A为原问“What is your current action and why?”，
B为“What are you doing at this moment, and what is the reason?”，
C在B后加`Answer in the form "action, justification".`。
每模型A复用完整开发评测；新增B/C共96条。当前报告只有joint和targeted，
没有声称本轮重新生成了base/old_lora的B/C回答。

| 条件 | joint正确率 | targeted正确率 | targeted相对joint修复/退化 |
|---|---:|---:|---:|
| A原问 | 16/48，33.33% | 14/48，29.17% | 1 /3 |
| B改写 | 13/48，27.08% | 14/48，29.17% | 2 /1 |
| C格式 | 12/48，25.00% | 9/48，18.75% | 0 /3 |

同一模型改变提示：joint A→B修复1/退化4，B→C修复2/退化3；
targeted A→B修复1/退化1，B→C修复0/退化5。
joint三条件均错30段，targeted均错33段；两个模型全部六次均判错29段：
`2,3,4,5,6,7,11,14,15,18,19,21,22,23,24,25,26,27,28,29,30,31,33,35,36,38,39,40,41`。

三条件共享48段视频，不是144个独立视频；重复改写没有提供稳定收益。
当前不能把29段全部定为视觉失败：自动评分误差、参考可观察性、时序输入、动作推理均需区分。
Case43中joint被Judge判正确，但参考是保持左侧位置，回答却增加了为左转定位的理由，
也说明Judge通过不等于每个语义细节均已验证。

报告和内嵌五帧HTML已生成，HTML约153MB；新一轮逐图人工裁定尚未完成。

## 8. 当前决策与下一阶段（尚未实施）

1. 主模型继续使用joint；保存本轮负向实验，停止沿这套10k配方继续增加epoch。
2. 固定48段诊断视频，另加32条joint原本答对的开发样本作保留能力对照。
3. 接入候选强教师，使用相同五帧及原问题独立回答，不把参考答案放入教师输入。
4. 统一评分并看图核验修复/退化，判断教师是否存在真实的净纠错收益。
5. 教师有效后，才在训练split构造纠错监督并选择离线蒸馏/OPD方案；开发样本不变为训练标签。
6. 若教师同样无法根据五帧判断，先处理可观察性、时序输入或标注问题。暂无证据支持直接RL。

教师型号、环境、推理入口、蒸馏代码均未落实；未启动教师推理或新训练。
该计划不能当作已经完成的项目成果，也没有承诺OPD必然提高正确率。

## 9. 产物索引

所有路径相对项目根目录：

- 数据及采样信息：`datasets/processed/lingoqa_action_targeted_v1/manifest.json`
- 训练配置/结果：`outputs/mpp_lingoqa_action_targeted_v1/{run_config,summary}.json`
- Action逐题评分：`outputs/projector_sft_eval/targeted_action/scored/scored_predictions.jsonl`
- Scenery逐题评分：`outputs/projector_sft_eval/targeted_scenery/scored/scored_predictions.jsonl`
- 修复/退化：`outputs/projector_sft_eval/joint_vs_targeted_{action,scenery}.json`
- 诊断：`outputs/action_targeted_diagnosis_v1/{report.json,report.md,case_predictions.jsonl,review.html}`
- 对照模型：`outputs/projector_sft_eval/{base,old_lora,joint}_{action,scenery}/`

主要执行入口：`bash scripts/run_action_targeted_sft.sh train`、`eval action`、`eval scenery`、
`diagnose joint`、`diagnose targeted`、`diagnostic_report joint targeted`。
完整环境与启动命令见实现方案；本次归档仅更新Markdown，未变更上述原始实验产物。
