# 2026-10-08：问题相关视觉证据搜索 → 冻结 MPP 视频问答

最新状态（2026-10-09归档）：8B下载及完整dev4320流程已完成，报告于10-08
18:42:39 UTC落盘。full Judge60.39%→evidence55.95%，修复291／退化483，
净−192；434次搜证执行失败回退原五帧，主要退化发生在成功追加图像的3378题。
官方500尚未运行，当前建议先诊断再决定是否继续，不直接执行freeze／official。
本轮没有训练MPP。详见[结果、失败类型和下一步分析](evidence_search_results_20261009.md)。
以下保留实现与复现命令；文中接口测试通过不代表准确率提升。

## 目标和边界

原问题＋五帧 → 外部 VLM 提出裁剪请求 → 返回原像素裁剪图 → VLM 再观察、
丢弃无关区域或继续搜索 → 冻结最终图像证据 → 原五帧＋选中裁剪 → 原 MPP 回答。

不是“检测车辆后挑高置信框”，也不把问题改成关键词。搜索模型接收完整问题，
需区分问题目标、相对 ego 的关系、动作／场景变化，并允许关系上下文框。
只有实际返回的裁剪 ID 才能被选择。没有可靠局部证据时可以不追加图像。
没有使用旧人工框，也不向搜索模型提供答案、评分、失败清单或教师反馈。

程序仅将选中的**图像**、原帧索引和框坐标交给 MPP，不传外部模型的答案、
解释或 `purpose`。默认只比较 MPP 原五帧和 MPP 证据输入；不运行外部模型
直接答题、外部模型搜索后答题的 C／D 对比。可选 duplicate 是 MPP 输入控制，
默认关闭。

这是自动外部 AI 预处理，不是人工标注；同时不是“MPP 已学会自主搜证”。
原五帧仍保留，问答、参考及 split 不变。dev 可用于决策搜索配置，随后显式
冻结配置再运行官方集。原官方 500 已多次使用，不称为新的盲测。

采用搜索／观察／停止的工具循环思想，但没有接入 ZoomEye 原仓库、权重或
原算法实现，因此名称是 `external VLM-driven budgeted evidence search`，
不能称为 ZoomEye 复现，也不是 Visual OPD。

## 模型、预算与输入

正式默认外部模型：本地 `Qwen3-VL-8B-Instruct`，路径
`/data/zjr/models/Qwen3-VL-8B-Instruct`，可用 `EVIDENCE_MODEL` 显式覆盖。
既有 `chartvlm` 现代环境用于它的推理，不更新 MPP Python3.8 环境。
本次核实 chartvlm 为 Python3.11.16／Torch2.11.0+cu128／Transformers4.57.6；
实现交付时默认8B尚未下载；10-08续传完成四个分片，正式开发集使用该8B。

| 设置 | 默认值 |
|---|---:|
| 原始时间帧 | 5，T0–T4，不额外取帧、不伪造新时间点 |
| 裁剪请求轮次 | 最多 2 |
| 每轮请求数量 | 最多 5 |
| 最后一次调用 | 只能从已观察 ID 中选择，不能再裁剪 |
| 最终附加裁剪 | 最多 5，总计最多 10 个 MPP 视图 |
| 单时间点最终裁剪 | 最多 2 |
| 搜索图像最小／最大像素 | 50,176／200,704 |
| 整个历史最大空间像素 | 2,007,040，随历史图片数动态分摊 |
| 搜索输入／新生成 token | 6,144／512，不静默截断 |
| MPP 生成预算 | 原 96 token、batch1、上下文1,536 |

裁剪框是原图 0–1000 坐标，按原分辨率向外取整到实际整数像素；不使用固定中心
裁剪、不套旧候选面积 65% 阈值。外部处理器对原图和裁剪图按固定总预算缩放，
MPP 使用这些 native crops 经原 224 处理器得到的 EVA 特征。

搜索为每卡一个模型副本、独立问题分片，不初始化 DDP／NCCL；裁剪 EVA 编码
复用既有 torchrun 四卡入口。搜索、EVA 和 MPP 生成依次执行并释放前一阶段
模型；不会同时把搜索模型和 MPP 放到同一张卡。本轮8B BF16搜索实测峰值
PyTorch allocated为17.36GiB（不是整卡占用或训练显存）。

## 代码与产物

| 文件 | 职责 |
|---|---|
| `mpp_video_qa/evidence_search.py` | 问题／帧 allowlist、JSON 动作、原像素 crop、实际图像回传、已观察 ID 选择、trace／失败回退 |
| `mpp_video_qa/evidence_hf.py` | 现代 Qwen3-VL 多轮多图输入、整个历史像素和 token 预算、贪心生成 |
| `scripts/prepare_evidence_search.py` | 原 QA 独立保留、无答案搜索输入、train/dev/eval segment 隔离、最终视图及去重裁剪目录 |
| `scripts/search_video_evidence.py` | 独立多卡推理、逐题原子结果、恢复／失败重试、模型与代码身份、实时阶段进度 |
| `scripts/evidence_workflow.py` | 分阶段和完整流水线、同权重 MPP 两臂评测、官方 Judge、分来源和配对报告、dev 配置冻结、状态 |
| `scripts/run_evidence_search.sh` | 从任意工作目录调用的 Bash 统一入口 |
| `region_views.py`／`data.py` | 新证据协议兼容原 native-crop 缓存和多视图输入；原候选协议继续兼容 |
| `tests/test_evidence_search.py`／`test_evidence_hf.py` | 语义工具循环、QA 隔离、恢复／接入／报告和真实 HF 接口测试 |

默认新目录：

```text
datasets/processed/lingoqa_vlm_evidence_v1/
  inputs/dev.jsonl, eval.jsonl       # 搜索进程只接收这些无答案文件
  original/dev.jsonl, eval.jsonl     # 原 QA，仅供 MPP 回答后评分
  dev/data.jsonl, views.jsonl, regions.jsonl, manifest.json
  eval/data.jsonl, views.jsonl, regions.jsonl, manifest.json
datasets/cache/lingoqa_vlm_evidence_eva_v1/dev/, eval/
outputs/vlm_evidence_search_v1/
  search_dev/, search_eval/          # manifest、每 worker 状态、逐题 trace／结果
  mpp_full_dev/, mpp_evidence_dev/   # 预测＋原官方协议评分
  mpp_full_eval/, mpp_evidence_eval/
  report_dev.json/.md, report_eval.json/.md
  frozen_search_config.json
  workflow_progress.json, stage_timings.jsonl
```

使用原 `lingoqa_feedback_opd_40k_v2` 的 dev4320（Action3120／Scenery1200）和
official500，不筛选失败题，不扩大训练数据。train 只用于 exporter 的隔离检查，
不进入新搜索流程。MPP 固定为 CE 控制组 final 的语言 LoRA＋projector＋完整
Q-Former：`outputs/mpp_feedback_sft_control_40k_2epoch_v2/final`，不是 raw Base
或 OPD final。原全图 EVA 缓存 `/tmp/zjr_mpp_fast/cache/lingoqa_5f_eva_v1` 继续复用。

`selected`、有效的 `full_only`、执行 `failed` 分别计数。后两者使用原五帧，
失败题不从指标分母删除。报告含分 Action／Scenery、配对修复／退化、按 segment
bootstrap、选区／失败状态分组；是否提升以完整 dev 的 MPP Judge 为准。

搜索模型身份记录 JSON metadata 内容 SHA＋权重文件大小／mtime stat identity，
不称为整个大权重文件的内容 SHA。配置、提示、代码、输入身份变更必须新开目录。
逐题结果同时保存真实输入 hash、原图 hash、实际像素框、工具调用和已返回图像 ID。

成本记录包含逐题搜证耗时、调用次数、裁剪 EVA 批处理耗时、MPP 自身生成耗时及
阶段 wall time。MPP 原生成延迟不包括搜证和缓存，不能将它当新流程端到端延迟；
当前不是在线部署延迟测量。

## 运行命令

### 1. 下载外部 8B 模型

代码没有自动下载。使用独立现代环境；不需要给 MPP 安装新 Transformers：

```bash
mkdir -p /data/zjr/models
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HUB_DISABLE_XET=1 \
  /home/zjr/anaconda3/envs/chartvlm/bin/python -c \
  "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-8B-Instruct', local_dir='/data/zjr/models/Qwen3-VL-8B-Instruct')"
```

下载完整并结束后再启动。中断可重跑同一命令；不要在模型仍下载时开始正式搜证。
如果权重已有在其它目录，只设置 `EVIDENCE_MODEL=/绝对/模型目录`，无需重复下载。

### 2. 开发集完整流水线

选择空闲四卡；下面的 0,1,2,3 是调用示例，不是本次确认的 GPU 空闲状态：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa

CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29712 \
  bash scripts/run_evidence_search.sh dev
```

自动顺序：导出无答案输入 → 4320题搜证 → 选中区域计划 → EVA 缓存 → 同一个
MPP 分别 full／evidence 生成和 Judge 评分 → `report_dev.md/json`。
不会自动训练，也不会自动开启 C／D 或 duplicate。

后台运行（避免交互终端退出造成 SIGHUP）：

```bash
systemd-run --user \
  --unit=mpp-evidence-dev-v1 --collect \
  --property=WorkingDirectory=/home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa \
  --setenv=CUDA_VISIBLE_DEVICES=0,1,2,3 \
  --setenv=MASTER_PORT=29712 \
  /usr/bin/bash scripts/run_evidence_search.sh dev
```

模型放在其它目录时增加 `--setenv=EVIDENCE_MODEL=/绝对/模型目录`。

### 3. 监视与预览

以下可直接从 `~/Tracker` 运行，不依赖相对路径、不使用 `[[`，`.sh` 必须交给 bash：

```bash
watch -n 10 'bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_evidence_search.sh status'

journalctl --user -fu mpp-evidence-dev-v1.service -o cat
```

每 worker 显示加载／观察／最终选取阶段、当前题、已完成数、失败数、最近耗时、
ETA、PID 是否存活及最后更新时间。后续显示 crop cache 以及两臂生成／评分进度。
仅 `process_alive=true` 不能证明正在有效推进，要结合阶段和 completed／更新时间。

搜索／assemble 后可生成内嵌图片的 HTML；这只是查看，不要求人工确认才能评测：

```bash
bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_evidence_search.sh preview dev
```

输出 `outputs/vlm_evidence_search_v1/evidence_preview_dev.html`，无需 localhost 服务。
对应逐题调用／原因在 `search_dev/results/<hash前两位>/<qid哈希>.json`。
页面参考答案仅用于事后查看，不参与搜索模型或 MPP 输入。

### 4. 开发集结果支持继续时，冻结并运行官方 500

2026-10-09的本轮dev为负向结果，目前建议暂不执行这一步。以下保留后续复现入口。

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
bash scripts/run_evidence_search.sh freeze && \
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29713 \
  bash scripts/run_evidence_search.sh official
```

也可用 systemd，将 unit 改为 `mpp-evidence-official-v1`、端口29713、动作为
`official`。官方阶段要求外部模型、提示、代码和预算严格与 dev 冻结配置一致，
不能根据官方错误调配置再称为同一次测试。
freeze也固定三路MPP权重与精度身份，官方两臂不能另换MPP checkpoint。

### 分阶段运行和恢复

```bash
bash scripts/run_evidence_search.sh prepare
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_evidence_search.sh search_dev
bash scripts/run_evidence_search.sh assemble_dev
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29712 bash scripts/run_evidence_search.sh cache_dev
CUDA_VISIBLE_DEVICES=0 bash scripts/run_evidence_search.sh eval_dev
bash scripts/run_evidence_search.sh report dev
```

同配置直接重跑 `dev` 会复用完整题结果／选中缓存／完整 MPP 生成／已有有效评分。
中断没有提交结果的题会重新处理；生成过完整答案的条件不再无谓重新加载 MPP。
可改 worker 数恢复，但要同时设置可见 GPU 数，例如 `EVIDENCE_WORKERS=1` 配合
`CUDA_VISIBLE_DEVICES=0`，完整流水线还需 `NPROC_PER_NODE=1`。

执行失败结果默认保留、计入完整评测；只有**尚未 assemble 的时候**才建议显式
`EVIDENCE_RETRY_FAILED=1 ... search_dev` 重试。同一确定性策略的无效动作可能仍会失败，
不能保证重试会修正。assemble 后的 view／catalog 不修改，若确需改结果或策略，
用 `EVIDENCE_RUN_DIR`、`EVIDENCE_DATA_DIR`、`EVIDENCE_CACHE_ROOT` 三个新版本目录。
GPU OOM 会保存该题失败记录并停止自己的搜索 workers；不要将进程消失误认成功。

四卡是独立推理，不是把一份 8B 跨四卡切分。若单卡显存不足，先查看其它占用，
或显式调整总视觉预算且使用新输出目录；本次没有实测 8B 峰值或自动改成 4B。

4B 仅需检查接口时可手动调用 `scripts/search_video_evidence.py --max-records 1`
并指定单独 output-dir 和 `--split dev`，不可把这个结果 assemble 为完整评测。
正式搜索可以显式选择 4B，但必须从 dev 到官方冻结同一模型，不能混为 8B 结果。

## 已完成的验证和未完成的验证

2026-10-08 本次：

- 新搜证26项单元／集成测试通过：包括实际原像素裁剪回传、两轮＋final、未观察
  ID拒绝、上下文大框、预算、失败／no-crop区分、去重、QA／split隔离、恢复、
  原缓存dataset接入、配对报告、dev冻结；使用合成图和固定动作，不是质量评测。
- 原 RegionViewTests 8项和真实旧 Qwen／PEFT tokenizer→CE／梯度／生成1项通过，
  没有重跑已有多进程DDP测试或正式四卡训练。
- chartvlm真实 Qwen3-VL处理器＋随机小权重模型3项通过：5／10／15图片历史预算、
  不静默截断、真实多模态 `generate`。只验证接口，不验证预训练模型理解能力。
- 合计38项通过，无跳过；Python3.8编译、Bash语法、CLI help和Tracker根目录status通过。
- 实现检查中已修正：独立worker不提前汇总其它未完成分片；记录失败调用尝试／
  截断原因；完整生成恢复不再次加载 MPP；加载阶段进度从启动就可见；父进程
  每轮检查所有worker，后置分片报错不会被仍活跃的首分片掩盖。
- 新并行退出测试曾提示 launcher lock 的 ResourceWarning，已改为上下文管理，
  异常退出也显式关闭；状态读预测文件同样显式关闭。不涉及模型或目标变化。

未进行8B下载、真实8B搜证、全量输入导出、正式native-crop EVA缓存或MPP评测，
未启动／停止用户作业，没有修改旧模型、源QA、旧缓存／实验。没有准确率提升承诺。

## 上一轮实验为何不继续直接训练

2026-10-08 07:02 生成的
`outputs/auto_region_eval_v1/report.json` 已完成冻结 full／duplicate／规则ROI三臂，
虽然 `development_complete=false`，缺失的是尚未启动的 ROI SFT epoch1／2，
不是这三臂未完成。

| 输入，dev4320 | Judge | Action | Scenery | Token F1 |
|---|---:|---:|---:|---:|
| 原五帧 | 60.30% | 58.46% | 65.08% | 0.6525 |
| 追加全图 duplicate | 56.88% | 54.39% | 63.33% | 0.6289 |
| 检测＋规则ROI | 53.52% | 51.22% | 59.50% | 0.6170 |

full→ROI：238修复、531退化，净−293题／−6.7824pp；按segment95%区间
[-8.4462,-5.1288]pp。duplicate→ROI也是−145题／−3.3565pp。
所以结果不支持把“自动加图”视为已验证收益，也不能把28题人工ROI结果外推。
新阶段是验证真正问题相关的证据选择是否改变这一结论，不是默认再训一轮ROI。
详细事件和缓存完成记录放在 `project_timeline.md` 第52–53节。
