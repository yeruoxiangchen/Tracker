# 2026-10-14：SFT 保留版本、简历实验与 checkpoint 索引

这是本项目代码上传前的固定说明。范围为 MPP-LLaVA 驾驶视频问答；个人简历 Word、
数据、图像、模型权重和临时缓存留在本机。仓库公开源码、运行方法、实验历史和小型
配置／结果快照。所有分数为同协议 Lingo-Judge 判定，官方 500 题已多次使用，不是新盲测。

## 三个需要区分的模型

| 模型用途 | 本地输出目录（相对项目根） | 开发 4320 题 | 官方 500 题 |
|---|---|---:|---:|
| 简历领域 SFT 起点／OPD 实际父模型 | `outputs/mpp_lingoqa_qformer_all_1epoch_v1` | 60.00% | 54.20% |
| 当前保留的 SFT 续训模型 | `outputs/mpp_feedback_sft_control_40k_2epoch_v2/final` | 60.46% | 55.40% |
| 简历反馈 OPD 模型 | `outputs/mpp_feedback_opd_40k_2epoch_v2/final` | 60.23% | 54.80% |

“最佳 SFT”在这里指项目保留的 CE 续训模型，不宣称它是所有历史开发实验的最高分或
新盲测选出的最优模型。全量续训曾取得 dev61.78%，但其 dev 选定检查点官方仅52.2%；
扩大语言 LoRA 官方54.4%，均有原始报告保留。不同历史回放中父dev60.05%与60.00%
分别记录，本表采用同一反馈 OPD 对照协议，不能拼接不同轮次的分数。

简历展示真实 `SFT parent → OPD weight1` 链路：OPD 比起点观测 +0.23／+0.60 个百分点，
却低于同预算继续 SFT。配对置信区间跨零，未证明 OPD 独立增益。最新权重3在dev选中
2500步（60.39%），官方54.20%；它与简历的权重1模型不同，不替换简历指标。

## 数据与模型理解对应哪些代码

| 简历贡献 | 代码入口 | 实际行为 |
|---|---|---|
| 五帧视频与多模态架构适配 | `mpp_video_qa/model.py`、`visual_bridge.py` | EVA→32 query／帧的 Q-Former→768到4096 Projector→Qwen7B ChatML；EVA／Qwen基座冻结 |
| 原始问答构建、片段隔离 | `scripts/prepare_lingoqa.py`、`prepare_full_qformer_sft.py`、`prepare_feedback_opd.py` | 406212原QA、27599训练片段；dev4320、official500；五帧、问题、答案不重写 |
| 领域 SFT | `scripts/train_sft.py`、`run_full_qformer_scale.sh` | 全量QA×1epoch、25389更新；完整Q-Former＋Projector＋语言QKV LoRA |
| 保留模型的 CE 续训 | `scripts/train_feedback_opd.py` 的 `sft_control`、`run_feedback_opd_v2.sh train_control` | 40k×2epoch、5000更新；只有原答案CE，没有教师／rollout／JSD |
| 反馈条件 OPD | `mpp_video_qa/feedback_opd.py`、`opd.py`、`scripts/train_feedback_opd.py` | 冻结同模型教师获得训练参考答案；学生在线生成；Judge门控后在学生token前缀上计算全词表JSD；所有样本保留CE |
| 四卡执行、恢复及加速 | `opd_runtime.py`、`run_feedback_opd_fast.sh`、`global_batch.py` | bf16冻结基座／fp32训练参数，累积、梯度检查点、共享基座、答案位置logits、梯度bucket与RNG恢复 |
| 自由生成与配对评测 | `generate_answers.py`、`evaluate_answers.py`、`report_feedback_opd_v2.py` | 原五帧／问题、greedy96；官方多参考Judge；dev选1000／2500／5000检查点；统计修复／退化与视频bootstrap |
| 本地视频问答入口 | `scripts/ask_video.py` | 短视频均匀五帧采样并生成回答；未声称流式部署或自动驾驶控制 |

历史 OPD／DPO／区域／外部教师代码都随本次源码提交保留；没有成功 checkpoint 的分支
也保留实现和当时的未运行／负结果记录。自动区域 SFT、外部教师答案蒸馏未因代码存在
就被记作已训练成功。后续问题感知 Q-Former 在这个代码快照之后独立实现。

## Checkpoint 的组成和位置

项目绝对根：`/home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa`。

父模型目录使用 `final_adapter/`、`final_projector.pt`、`final_qformer_adapter.pt`；
CE和OPD bundle使用 `adapter/`、`projector.pt`、`qformer_adapter.pt`：

- 语言 adapter约32MiB：保存 LoRA A/B，不是完整7B权重；需要官方MPP语言基座。
- Projector约12MiB：把Q-Former的768维输出映射到Qwen4096维；不能任意与别轮混用。
- Q-Former约401MiB：文件名虽含adapter，实际为完整Q-Former和32个query tokens。
- `training_state.pt`约891MiB：优化器、scheduler、各rank随机状态和样本位置，只在恢复
  对应训练时需要，推理不加载。权重评测快照不一定含此文件，不能当作可恢复断点。
- `metadata.json`、`run_config.json`、`summary.json`：记录模型／数据身份、预算、状态。

实测每份推理三路权重合计约446MiB；带训练状态的bundle约1.3GiB。它们被Git忽略，
不上传二进制；逐文件真实大小与存在性见
[checkpoint清单](artifacts/20261014/manifest.json)。训练状态仅允许同布局／协议恢复。

外部基座位于兄弟目录 `../mpp_llava_video_workspace/checkpoints/`，快读副本位于
`/tmp/zjr_mpp_fast/checkpoints/`。Qwen7B、MPP已发布SFT权重、EVA、BLIP2、BERT均需保留；
官方MPP checkpoint覆盖已发布语言参数后，项目三路权重再覆盖领域适配参数。
原图位于 `/data/zjr/datasets/lingoqa/`；EVA cache优先
`/tmp/zjr_mpp_fast/cache/lingoqa_5f_eva_v1`，临时目录清理后应从持久资产重建。

## 仓库里的小型证据

`docs/artifacts/20261014/`原样复制训练配置／summary／adapter配置及五组dev、official
报告，manifest记录源路径、字节数和SHA256。没有复制逐题预测、图像或tensor。
这是截至2026-10-14的结果快照，不会因以后更新模型而自动替换。

重新导出（读取本机原始产物）：

```bash
python3 scripts/export_experiment_snapshot.py
```

## 复现环境与上游补丁

上游固定在 `Coobiw/MPP-LLaVA@baf617b0f5217442ba20721be9aed710cdf999e2`。
该checkout实际修改了BaseModel分片加载和Qwen生成／设备兼容；归档在
`patches/mpp_llava_baf617b_local.patch`。在新机器运行
`bash scripts/setup_upstream.sh`即可获得相同版本并应用补丁；已应用时只检查。
它同时连接upstream的cache／ckpt与持久资产目录，使原图视频推理可找到EVA权重。
配置样例见`configs/environment.yml`和`configs/requirements-runtime.txt`。
torch2.0.1+cu118、transformers4.39.2、Python3.8为原训练环境；PEFT实际使用0.13.2。

新机器安装（环境与权重分开准备）：

```bash
conda env create -f configs/environment.yml
conda activate mpp_llava
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r configs/requirements-runtime.txt
bash scripts/setup_upstream.sh
python -m pip install --no-deps -e ../mpp_llava_video_workspace/third_party/MPP-LLaVA
```

复现简历实验与保留SFT的主要命令（需要先恢复对应数据和大模型资产）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_full_qformer_scale.sh train
CUDA_VISIBLE_DEVICES=0 bash scripts/run_full_qformer_scale.sh eval_dev
CUDA_VISIBLE_DEVICES=0 bash scripts/run_full_qformer_scale.sh eval_selected

CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_feedback_opd_v2.sh train_control
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_feedback_opd_fast.sh train_opd
CUDA_VISIBLE_DEVICES=0 bash scripts/run_feedback_opd_v2.sh eval_dev
CUDA_VISIBLE_DEVICES=0 bash scripts/run_feedback_opd_v2.sh eval_selected
```

这些命令用于复现，不能在已有完整输出目录中覆盖旧实验。已完成运行只需读取归档
报告；新机器缺权重时运行产生checkpoint的源码，而不是伪造已有checkpoint的存在。
