# 2026-10-01：视觉侧 SFT、教师对照与单教师 OPD 实现

归档核验时间：2026-10-01 10:06 UTC。这里记录代码交付及短测，不是新一轮准确率结果。
长时间缓存、正式训练、教师对照和开发集自由评测均未在本轮启动。

## 1. 延续的结论与本轮目的

目前继续选用全量数据训练的 Qwen LoRA + projector 联合模型：
outputs/mpp_lingoqa_projector_lora_1epoch_v1/。
其同路径开发集 Judge：Action 51.47%、Scenery 60.17%、合计 53.89%。
10k 定向续训退化，不能以“继续 SFT 必然更好”或“OPD 必然有效”为前提。

本轮一起实现三个入口，但执行顺序不是同时训练：

1. 视觉侧 SFT：补上原来完全冻结的 Q-Former 的可训练适配。
2. 教师对照：查明额外视觉观察或更强模型是否能纠错，以及是否破坏原本正确答案。
3. 单教师 OPD：只有教师存在可靠纠错信号时再启动，保留监督回放，控制退化。

任务始终是原始驾驶片段的五帧 + 原问题 → 自由生成驾驶问答答案。
没有改成定位、候选排序、Evidence JSON 或 Agent；没有改写原数据集答案。

## 2. 数据与隔离

已实际生成 datasets/processed/lingoqa_visual20k_v1/：

| 项目 | 数量 / 说明 |
|---|---|
| 训练 QA | 20,000 条原始问答 |
| 训练视频段 | 16,308 |
| Action / Scenery | 12,900 / 7,100 |
| 同段 QA 上限 | 4 |
| 教师诊断 | 固定 48 条诊断 + 32 条已答对、不同视频的对照 |
| 教师训练候选 | 训练子集中的 1,000 条，用于后续生成/筛选 |
| dev | 保留 Action 3,120 + Scenery 1,200 |
| official eval | 保留 500，不进入训练 |

使用原始训练数据的来源占比配额，按视频轮转抽样，避免少量视频占据大部分训练问答。
它是计算成本受控的续训子集，不是重新从头训练，也不是全量重训。
20k 与旧 10k 采样不同，单靠这一轮对比不能把收益全部归于 Q-Former；
如需归因，用 TRAIN_QFORMER=0 在相同 20k 数据上补一个冻结视觉分支对照。

数据构建时间：2026-10-01T06:22:25.171591+00:00。
原训练 JSONL SHA-256：
dfc14f194da236ed9e35484a446144fe29425d7f3f71c7fab44c3ff3aa0c619e。
本轮训练 JSONL SHA-256：
21f9e8a1a379b4a3f387364b4af7b8b703d3871e4da842553bb7c2fc27556a5c。

构建器检查 train/dev/eval 视频段隔离。教师 80 条开发诊断不能变成训练标签。
OPD 构建器只接收来自训练 split 的逐题结果，并核验问题、参考答案、视频段和生成权重身份。

## 3. 模型与 SFT 实现

旧路径：冻结 EVA → 冻结 Q-Former → 缓存 [5,32,768] → 可训练 projector → Qwen LoRA。

新路径：冻结 EVA + ln_vision → 缓存 [5,257,1408] →
在线 Q-Former（query/value LoRA + 32 个 query tokens）→ projector → Qwen LoRA。

- 保留 MPP 的 EVA、Q-Former、Qwen7B 和每帧 32 个视觉 query。
- 新缓存阶段名为 eva，缓存的是经过 ln_vision 的 EVA 特征。
- Q-Former 从原 BLIP 初始化；新增 rank=8、alpha=16 的注意力 query/value LoRA，
  B 初始为零，因此不训练时应复现原 Q-Former。
- Qwen adapter 和 projector 从当前最佳 joint 加载，不使用退化的 targeted 权重。
- 不加载 EVA 到 SFT 显存；Q-Former 在线计算以保留视觉分支梯度。
- 三组学习率：Qwen LoRA 5e-6、projector 5e-6、Q-Former 1e-5。
- 默认 4 卡，每卡 batch=1、累积 4、1 epoch，共 1,250 次优化器更新。
- 使用梯度检查点，答案及 EOS 接受 CE，问题/视觉占位/填充不接受答案监督。
- 可训练参数 FP32，冻结模型和激活主要 BF16。
- checkpoint/epoch/final 同时保存语言 adapter、projector、Q-Former adapter/query。
  恢复时一并加载，不能只拿语言 LoRA 当成完整新模型。

完整模型三件套：
final_adapter/、final_projector.pt、final_qformer_adapter.pt。
同时保留 checkpoints/visual_bridge_init.pt，作为固定视觉初始化资产。

缓存预计：train+dev+eval 共约 16,808 段，BF16 特征净体积约 56.6 GiB，
另有文件元数据开销；不是原来的小型 Q-Former 缓存。默认放 /tmp，不覆盖旧缓存。
可以用 MPP_EVA_CACHE_DIR 指定其他位置；后续命令必须使用相同目录。
训练/评测结束前不要删缓存。临时权重目录仍不自动删除。

## 4. 教师对照

默认实现两个同模型条件：

- joint：原五帧。
- joint_crop：同五帧全图 + 每帧中央道路区域裁剪，总计十张观察图。
  ROI 为归一化 (0.2, 0.1, 0.8, 0.8)，按时间逐对输入。
  提示明确裁剪与全图来自同一时刻，不伪装成十个时间点。

固定裁剪不保证覆盖交通灯、侧向车辆或目标；它不是框标注、GT 工具或已证实有效的教师。
保留全图提供上下文，比较修复/退化之后才能决定是否采用。
它也不是原图细节的凭空恢复：只是让原图区域占据更多输入像素。

另有 hf_vlm 后端用于本地强 VLM 在同五帧上回答。
该后端需要独立的较新 Transformers 环境和用户提供的模型路径，本轮未下载或实测强教师。
不同 tokenizer 的外部教师目前仅用于答案级比较，不能直接喂给这里的 token-level OPD。

评分沿用修正的官方 LingoJudge 协议。输出 report.json、report.md 和内嵌五帧的 review.html。
80 条为偏向失败的诊断样本，不能报告为整个开发集准确率。
Judge 是文本判别，不是视觉真值；检查纠错是否确实受视频支持。

## 5. OPD 实现与边界

这里是 student on-policy 采样的分布蒸馏，不是教师离线写答案后的 SFT：

1. 学生在原五帧上采样实际 token IDs。
2. 固定教师看全图+裁剪，在相同 sampled prefix 上给出 token 分布。
3. 学生在自己的前缀上计算分布，分别按各自 prompt 长度对齐答案位置。
4. 全词表 JSD（默认）或 forward KL；按 token 分块 FP32 计算。
5. 同题原参考答案另作一次 CE forward/backward 回放，默认权重 0.2。

关键实现：

- 采样 temperature=1、top_p=1、top_k=0、repetition_penalty=1，训练时 dropout=0。
- 不先 decode 再 tokenize；答案 EOS 纳入，prompt/padding 不纳入蒸馏。
- 未完成的截断 rollout 不蒸馏，但仍做监督回放。
- 一份冻结 Qwen base + 独立 policy/teacher PEFT adapter，教师投影/视觉分支单独冻结。
- 教师 forward 完成后切回 policy，再执行学生 forward/backward；
  不在学生梯度检查点重计算期间切换教师 adapter。
- 多卡对固定参数序列平均梯度；某卡无梯度时用零参与，避免跳过 collective。
- 保存优化器、RNG、数据/教师/学生初始化指纹和进度，支持按 checkpoint 恢复。
- teacher 与筛选时的权重、视图必须一致；student 初始化同样受指纹约束。
- 当前 token OPD 只接受相同 MPP base/tokenizer；不支持任意外部强教师换头即蒸馏。

训练样本从 teacher_train_probe 生成两个条件的回答并评分：
优先 teacher-correct/student-wrong，约 30% both-correct 回放。
不改变原 QA，保留筛选来源及 paired fixed/regressed/net。
这些是 Judge 筛选标签，仍需抽查，不是保证教师全部正确。
若无 teacher-only correct 样本，构建器停止；只有少数胜例但整体大量退化时，也不建议开训。

默认 OPD pilot 上限 200 步、1 epoch。
实际步数还受筛选后样本数量限制：四卡累积4时，一轮 N 条只有 ceil(ceil(N/4)/4) 步；
例如 1,000 条约 63 步，不会自动重复到200步。不要把上限误读为固定训练步数。

## 6. 本轮验证

已完成：

- 15 项 unittest（其中新增 8 项），涵盖原路径、视觉梯度、初始化等价、
  Q-Former adapter 保存恢复、JSD 数值/梯度、不同前缀答案对齐、教师冻结、
  train-only 筛选与采样、裁剪时间提示。
- 原 4 个函数式 core 测试单独执行。
- Python 编译、Shell 语法、训练/评测命令展开检查。
- 真实 MPP 单卡 smoke：outputs/visual_opd_real_smoke_v1/summary.json。

真实 smoke 结果：

| 检查 | 结果 |
|---|---|
| EVA 输入形状 | [1,5,257,1408] |
| 在线 Q-Former 与旧缓存最大误差 | 0.0 |
| 两次累积 CE loss | 0.26171875 / 0.26171875 |
| Qwen / projector / query / Q-Former 梯度及更新 | 四者均非零 |
| 学生采样 | 19 tokens，包含 EOS |
| JSD | 0.00239314 |
| 教师参数 | 不变 |
| 峰值 allocated | 15.40 GiB |
| 耗时 | 69.70 秒 |

短测只证明新增路径能执行和更新，不证明泛化提升。
未做四卡完整新路径训练、恢复后的长跑、全开发集评测；
单样本峰值显存不是所有长度的峰值保证。
真实视频 ask_video.py 已接入 --qformer-adapter 与 --bridge-init；
该原始视频入口本轮只做代码/参数检查，没有单独完成视频质量评测。
本环境未安装 pytest，因此未通过安装依赖来运行 pytest；
使用 unittest 与原无参 core 函数直接执行，不将缺失 pytest 算作测试通过。

## 7. 执行命令

以下脚本都用 bash，不要用 Python 执行 .sh。
prepare 已执行，可重复检查相同配置。正式缓存与训练尚未启动。

### 先缓存，再做 SFT

~~~bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa

CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_visual_adaptation.sh cache

CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_visual_adaptation.sh train
~~~

另一个终端监视，绝对路径避免 cwd 错误：

~~~bash
watch -n 10 'bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_visual_adaptation.sh cache_status'
watch -n 10 'bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_visual_adaptation.sh status'
~~~

需要后台训练时（先完成缓存，不要与前台训练重复启动）：

~~~bash
systemd-run --user --unit=mpp-lingoqa-visual-sft-v1 --collect \
  --property=WorkingDirectory=/home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa \
  --setenv=CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash scripts/run_visual_adaptation.sh train

journalctl --user -fu mpp-lingoqa-visual-sft-v1.service -o cat
~~~

训练后先评开发集，别只比较 teacher-forced loss：

~~~bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_visual_adaptation.sh eval visual action
CUDA_VISIBLE_DEVICES=0 bash scripts/run_visual_adaptation.sh eval visual scenery
~~~

脚本自动评分并与已有同问题 joint 结果比较，输出到 outputs/visual_adaptation_eval/。
可额外运行 eval joint action/scenery，在新的 EVA 在线路径做完整同路径基线。
官方集可用 eval visual eval，但不要用其反复挑选配置。

### 教师小对照

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_visual_adaptation.sh teacher_cache
CUDA_VISIBLE_DEVICES=0 bash scripts/run_visual_adaptation.sh teacher_compare
bash scripts/run_visual_adaptation.sh teacher_report
~~~

强 VLM 为可选：设置 STRONG_TEACHER_MODEL=本地权重路径 和 STRONG_TEACHER_PYTHON=对应Python，
运行 teacher_strong，再重新 teacher_report。不要随意替换 MPP 的旧环境依赖。

### SFT 与教师对照后才考虑 OPD

默认从本轮 visual SFT 起步；若其退化，统一设置 OPD_STUDENT_KIND=joint，
不要在数据筛选和训练之间更换模型身份。以下不是本轮要求马上运行的命令：

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_visual_adaptation.sh opd_cache
CUDA_VISIBLE_DEVICES=0 bash scripts/run_visual_adaptation.sh opd_select

# 先阅读筛选 manifest 的修复/退化，抽查教师纠错。
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_visual_adaptation.sh opd
bash scripts/run_visual_adaptation.sh opd_status

CUDA_VISIBLE_DEVICES=0 bash scripts/run_visual_adaptation.sh eval opd action
CUDA_VISIBLE_DEVICES=0 bash scripts/run_visual_adaptation.sh eval opd scenery
~~~

SFT 恢复：SFT_RESUME_FROM=完整checkpoint目录 后运行 train。
OPD 恢复：OPD_RESUME_FROM=完整checkpoint目录 后运行 opd；保留原始学生/教师选择配置。
若改变训练集、权重或视图，另选数据/输出目录，不覆盖历史运行。

## 8. 代码清单

新增：
- mpp_video_qa/visual_bridge.py：视觉 LoRA、初始化提取、视觉权重保存恢复。
- mpp_video_qa/observations.py：同时间全图/ROI 观察与提示。
- mpp_video_qa/opd.py：on-policy 采样、对齐、散度、共享冻结教师、梯度同步。
- scripts/prepare_visual_adaptation.py：20k 原 QA 与教师诊断/训练候选集。
- scripts/compare_teachers.py：两类教师后端及对照报告。
- scripts/prepare_opd_data.py：训练侧教师胜例+保留样本筛选。
- scripts/train_opd.py：OPD 训练与恢复。
- scripts/run_visual_adaptation.sh：统一入口。
- scripts/smoke_visual_opd.py：真实权重短测。
- tests/test_visual_opd.py：新增回归测试。

修改：
- model.py / data.py：EVA 缓存到在线 Q-Former 路径。
- cache_visual_features.py：EVA 阶段、全图/裁剪视图。
- train_sft.py：视觉适配、三组优化器、初始化/恢复/权重记录。
- generate_answers.py：完整视觉权重加载及评测身份。
- ask_video.py：视频问答可加载新 Q-Former adapter。
- README.md / project_timeline.md：最新入口与阶段记录。

本轮没有删除旧数据、旧缓存或旧 adapter，没有改动兄弟项目训练代码。
