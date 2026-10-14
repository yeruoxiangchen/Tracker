# 2026-10-07：自动问题相关局部观察＋MPP SFT

## 状态与目的

代码已接通候选检测、问题视图规划、选中裁剪的 EVA 缓存、四卡 SFT、三种输入评测、
开发集选模、官方评测和自带短视频问答。**本次没有启动正式检测、缓存生成或训练，
也没有新增准确率结果。**检测器的公开权重及独立环境仍需按下方命令安装／下载。

前一轮 28 题人工 ROI 诊断相对重复全图修复 5 题、退化 1 题，说明有值得验证的
观察路径；人工选框收益不能直接变成自动系统收益。这里把观察机制部署到原训练
问题上，测试自动候选是否有用，再用 SFT 适配输入。不是继续固定中心裁剪 OPD，
没有把 110 条开发诊断题或人工框加入训练，也未改变原始问题和参考答案。

## 输入与职责

原始五帧 → 图像候选检测 → 仅按问题选少量候选 → 全景＋原分辨率局部图 → 原 MPP → 原答案 CE。

检测采用 `IDEA-Research/grounding-dino-tiny`。检测器只接收原图和固定类别词汇，
不接收参考答案，也不判断“答案是否正确”。车辆、两轮车、行人、交通灯、路牌、
路障等候选输出包含类别、置信度、原图尺寸、图片 SHA、归一化框与模型身份。
参考：[官方代码](https://github.com/IDEA-Research/GroundingDINO)、
[Transformers 接口](https://huggingface.co/docs/transformers/model_doc/grounding-dino)。

问题选区是透明的第一版路由规则：显式对象词决定候选组；明确左／右时提供弱
方位偏好；相邻原帧同类框重叠提供弱连续性加分。最多每帧 2 个候选，不用“画面
中央就是前车”的规则，不把检测分数当作目标身份置信度，不声称完成了目标跟踪。
问“ego vehicle 是否加速”不会仅因 vehicle 一词触发旁车裁剪。泛化原因题可以保留
多个类别候选；无可信候选时保留原五帧。

该规则**不能保证识别出语义上的正确目标**，例如跟随车／停放车、多车之间的
关系、标志与道路关系仍可能混淆。它提供少量可追溯观察，不是人工 ROI 的等价替代。
因此后续必须看自动框预览及同权重／同预算对比，不能以“接入检测器”宣布问题解决。
检测类别／颜色文本不作为给 MPP 的答案提示，原问题保留原文。

局部框按边长扩展 20% 上下文，从原始 RGB 像素裁剪，再使用原
`BlipImageEvalProcessor(224)` 和冻结 EVA。不是先缩成 224 再裁剪／放大。
每个时间点按 full、region1、region2 排列；提示标为同一 `T0..T4`，说明附加视图
不是新的时间点。五张原全景始终保留，局部只补充对象细节，MPP 仍要自己理解
对象、关系、动作和原因。训练侧确定性地保留约 20% full-only QA 回放；未命中
候选的问题也是 full-only。开发与官方输入不做随机回放。

## 文件与缓存

| 实现 | 职责 |
|---|---|
| `mpp_video_qa/region_views.py` | 问题路由、像素框、视图顺序、提示、计划身份与特征组装 |
| `scripts/propose_regions.py` | 独立环境的图像候选检测、分片、进度、断点复用 |
| `scripts/prepare_region_views.py` | 原 QA 不变，生成每题计划、去重裁剪目录和 manifest |
| `scripts/cache_region_features.py` | 只加载 EVA，缓存实际选中的 native crops |
| `mpp_video_qa/data.py` | 共享五帧＋每题局部计划，按条件组装动态视图 |
| `scripts/train_sft.py`／`generate_answers.py` | 正式接收每题提示和视图，保留三路权重与身份 |
| `scripts/report_auto_regions.py` | 输入 token 检查、阶段进度、配对统计与 dev 选模 |
| `scripts/preview_region_views.py` | 无服务器的原图框／裁剪 HTML 检查 |
| `scripts/ask_region_video.py` | 自带短视频使用同一检测器版本和选区设置问答 |
| `scripts/run_auto_region_sft.sh` | 各阶段、恢复与比较的统一 Bash 入口 |

默认源数据是 `datasets/processed/lingoqa_feedback_opd_40k_v2/`，复用原 40,000
训练 QA 和 4,320 dev（Action 3,120／Scenery 1,200）、官方 500。仍覆盖原 27,599
训练片段，没有重新挑开发失败题来训练。新的数据目录为
`datasets/processed/lingoqa_auto_regions_40k_v1/`。

`train/dev/dev_action/dev_scenery/eval.jsonl` 保留原 QA 值；各自的 `*.views.jsonl`
单独保存问题 SHA、帧顺序、候选来源、裁剪像素与计划 SHA。`regions.jsonl`只保存
被选中裁剪，并按**图片内容 SHA＋实际像素框**去重，不按 question_id 重复编码。

原五帧继续使用 `/tmp/zjr_mpp_fast/cache/lingoqa_5f_eva_v1/`，不重写，不给每题
复制一份。新局部缓存默认放在项目的
`datasets/cache/lingoqa_selected_regions_eva_v1/`，不默认占用 `/tmp`。单个区域的
BF16 `[257,1408]` 裸特征约 0.69 MiB；实际总量取决于候选覆盖与去重。`prepare`
输出 estimated_region_feature_gib，生成缓存前据此检查磁盘。检测器与 EVA 分阶段
运行，不同时占用显存；检测器不加载 Qwen。

语言权重仍优先使用 `/tmp/zjr_mpp_fast/checkpoints/`，视觉权重使用原持久
`mpp_llava_video_workspace/checkpoints/`，避免再次引用不完整的 tmp BLIP 路径。
新检测器在独立 Python3.11 环境 `mpp_regions`，不升级原 Python3.8 MPP 环境。
安装脚本固定 Torch2.5.1/cu118、Transformers4.57.6，模型下载放在项目
`checkpoints/detector_cache/`。接口检查使用了现有较新环境，不把那个环境变成正式依赖。

## 训练与对照

三种条件共用原问答、原帧和冻结视图清单：

| 条件 | 输入 | 用途 |
|---|---|---|
| full | 原五帧 | 观察／继续训练基线 |
| duplicate | 在每个已选区域位置重复该时间点的完整图 | 控制增加图像／token 和空间提示的效应 |
| roi | 同一位置用真实 native crop | 检查局部像素细节的额外价值 |

duplicate 与 roi 使用**逐字相同的提示、框坐标、视图顺序和视觉 token 数**。提示
如实说明附加视图可能是全图或裁剪，没有把副本谎称为裁剪。full 保持旧五帧提示。
无候选或 full 回放的 QA 在三条件中退化为相同原输入。

三个冻结输入的起点都是
`outputs/mpp_feedback_sft_control_40k_2epoch_v2/final/`，即最新 CE 对照模型，不是
raw MPP，也不是旧 OPD 模型。训练同时初始化其 `adapter/`、`projector.pt` 和
`qformer_adapter.pt`；新优化器／新 schedule。默认更新完整 Q-Former＋query tokens、
projector、语言 LoRA，冻结 EVA 和语言 base；保持 FP32 trainables／BF16 base、
双侧梯度检查点、四卡 batch1×accum4、原参考答案 assistant-only CE。

默认每分支 40k×2 epoch＝80k QA 访问、global batch16、5,000 次更新。
LoRA／projector lr=2.5e-6，Q-Former lr=5e-6，seed44，warmup3%，每 1,000 步存
可恢复状态。上下文上限 1,536、单题最多 15 视图／480 视觉 token。LM head 只投影
被监督答案位置，仍是同一因果 CE；真实 Qwen 小模型验证了 loss 和训练参数梯度
与原完整 logits 路径一致。**本次未实测正式四卡峰值显存／吞吐，不宣称新输入
与原五帧速度、显存相同。**

开发评测为相同 legacy_v1、greedy96、官方 Lingo-Judge 协议，报告完整 dev 及
Action／Scenery、修复／退化、按 segment bootstrap。每分支均评两个 epoch，
按合并 dev Judge 选模、同分优先早 epoch；官方 500 不参与选模，且已多次查看，
不是新的盲测。配对 SFT 分支的预算／初始权重不一致时报告拒绝混为同预算结果。

## 执行命令

首次准备（长阶段前台运行，或由你自行放后台；这里没有自动启动）：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa

bash scripts/run_auto_region_sft.sh install_detector
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_auto_region_sft.sh propose
bash scripts/run_auto_region_sft.sh prepare
bash scripts/run_auto_region_sft.sh preview dev
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_auto_region_sft.sh cache
bash scripts/run_auto_region_sft.sh preflight
```

先获取同权重的输入对照，再训练 ROI 分支：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_auto_region_sft.sh eval_frozen
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29703 \
  bash scripts/run_auto_region_sft.sh train roi
CUDA_VISIBLE_DEVICES=0 bash scripts/run_auto_region_sft.sh eval_dev roi
CUDA_VISIBLE_DEVICES=0 bash scripts/run_auto_region_sft.sh eval_selected roi
```

需要完整同预算 SFT 对照时，两组依次各使用四卡，不要求八卡同时运行：

```bash
export REGION_BRANCHES='duplicate roi'
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29703 \
  bash scripts/run_auto_region_sft.sh train_pair
CUDA_VISIBLE_DEVICES=0 bash scripts/run_auto_region_sft.sh eval_dev duplicate
CUDA_VISIBLE_DEVICES=0 bash scripts/run_auto_region_sft.sh eval_dev roi
bash scripts/run_auto_region_sft.sh report
```

`train full` 是可选的原五帧同预算继续 SFT；若运行，也把 full 加入 REGION_BRANCHES。
指定独立新 TRAIN_ROOT 才能改变训练规模，避免覆盖老配置；默认不增加数据量。

后台 ROI 正式训练与监视：

```bash
systemd-run --user --unit=mpp-auto-region-sft-v1 --collect \
  --property=WorkingDirectory=/home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa \
  --setenv=CUDA_VISIBLE_DEVICES=0,1,2,3 --setenv=MASTER_PORT=29703 \
  /usr/bin/bash scripts/run_auto_region_sft.sh train roi

journalctl --user -fu mpp-auto-region-sft-v1.service -o cat

watch -n 10 'cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa && bash scripts/run_auto_region_sft.sh status'
```

JSON 进度是文件状态，未验证进程存活；结合 service 状态看是否健康运行。检测与
EVA 阶段在加载前写进度，分别列出四个 rank。训练沿用原 per-rank 失败与显存记录。
**入口是 `.sh`：用 bash，不要用 Python 运行它。**绝对 cd 可避免再次从 Tracker
根目录调用不存在的 `scripts/report_progress.py`。

恢复完整 checkpoint（不要同时运行两组 writer）：

```bash
REGION_RESUME_FROM="$PWD/outputs/mpp_auto_region_sft_40k_2epoch_v1/roi/checkpoint-001000" \
  CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29703 \
  bash scripts/run_auto_region_sft.sh train roi
```

恢复检查原问答 SHA、视图计划 SHA、condition、裁剪编码器、优化器／schedule／游标
和 RNG，不是加载 checkpoint 重新开始计步。新初始权重路径无需复用旧训练游标。

自带**短驾驶片段**使用同一训练 planner 和固定检测模型 revision：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_auto_region_sft.sh ask \
  roi epoch-02 /absolute/path/short_driving_clip.mp4 \
  'What vehicle are we following and what is it doing?'
```

应用入口从 clip 均匀取五帧，原 RGB 无损导出 PNG；候选检测之后释放检测器，EVA
编码之后释放 EVA，再加载 MPP。frame/plan/answer 写在
`outputs/auto_region_eval_v1/user_clips/` 便于追溯。长录像应先提供所问的短片段；
这里没有新增长视频定位／选片能力。

## 验证、遇到的问题与边界

新增 10 项 region 回归包含：答案改变不影响选框、不同问题不同计划、原五帧与
时间顺序不变、同图片同实际框去重、stale region 拒绝、roi/duplicate 提示与 token
预算一致、开发完整性与 dev-only 选模、实际 Qwen/PEFT tokenizer＋15 视图的
CE 梯度／生成，以及双进程变视图＋四次梯度累积的连续两步更新。

CPU 多进程检查验证 DDP/PEFT 执行路径，不等于正式四卡 NCCL 的显存／速度测试。
现代环境另有一项本地随机小 Grounding-DINO 权重的真实 forward、后处理、
per-frame 元数据和跳过已完成检测验证；**它不是公开预训练检测器质量评估**。

实现期间发现并修正：ego vehicle 被外部车辆规则匹配、signals 被 sign 前缀匹配。
MPP 环境没有 pytest，改用已有 unittest，另显式运行 4 个旧函数测试。沙箱禁止
本机 Gloo 回环导致旧双进程测试失败，获准宿主运行后通过；新 CPU GPT2 测试的
静态 Boolean mask 也不受旧 Gloo 广播支持，测试中排除两 rank 相同的静态 mask。
没有因此改变正式 Qwen/NCCL 的 buffer 配置。

最后一次完整回归为 **105 项：104 通过、1 跳过**；跳过的是旧 MPP 环境不支持的
Grounding-DINO 接口测试，已在现有 Python3.11／Transformers4.57.6 环境单独通过。
另显式执行 4 个旧 `test_core.py` 函数，全部通过。Python3.8 编译、Bash 语法、
训练／生成 CLI 和从 Tracker 根目录调用状态入口通过；状态为 `not_started`。
这些验证不使用公开预训练检测器或完整 MPP 训练权重，不是完整训练效果验证。

复现回归检查（双进程测试需要允许本机 Gloo 回环通信）：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
/home/zjr/anaconda3/envs/mpp_llava/bin/python -m unittest discover -s tests -v
bash -n scripts/run_auto_region_sft.sh scripts/install_region_detector.sh
```

独立检测环境安装完成后，可在那里运行实际检测架构的随机小模型接口检查：

```bash
/home/zjr/anaconda3/envs/mpp_regions/bin/python -m unittest discover \
  -s tests -p test_region_detector.py -v
```

预训练 Grounding-DINO 下载、真实数据自动框覆盖／错误率、正式四卡显存、SFT
准确率都还未测。本轮保持旧 SFT／OPD 实验和所有原始缓存不变。只有自动观察在
完整 dev 上有实际收益、纠错伴随的退化可控，才讨论把其作为更强教师／特权视图
继续 OPD；当前没有新的蒸馏目标、RL 奖励或 Agent 策略。
