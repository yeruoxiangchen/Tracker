# 2026-09-27：首个 forward 失败修复

> 后续四卡 `v6` 已通过模型加载和首个 forward，但首个 backward 遇到
> `CUBLAS_STATUS_ALLOC_FAILED`。当前状态及训练前问题时间线见
> `docs/training_bringup_issues_20260927.md`；下方 v6 命令仅作历史记录，不应原样重启。

## 失败原因

`mpp-lingoqa-sft-v5` 已完成四卡 MPP 权重加载，但首个 forward 报
`RuntimeError: could not resolve Qwen core`，没有完成 optimizer step。
原代码沿 `.base_model` 不断解包；Qwen 的 `base_model_prefix="transformer"`，
因此在到达 `QWenLMHeadModel` 后又进入不含 `lm_head` 的 `QWenModel`。
这是模型访问代码错误，不是缓存、训练数据或 SSD 故障。

## 修改

- `qwen_core`：PEFT 使用 `get_base_model()`，在拥有实际 `transformer` 和
  `lm_head` 子模块的完整 causal LM 处停止；保留模型内部注入的 LoRA 层。
- `train_sft.py`：新增 `--max-steps`，可用真实训练入口验证有限步数；默认 0
  仍执行全部 epoch。每个 rank 打印加载开始/完成；Python 异常写入
  `failure_rank_N.json`，监视脚本可显示失败原因。
- 保留此前 12 小时 process-group timeout；此次未更改训练数据、loss 或 LoRA 超参数。
- 回归测试使用本地 Qwen modeling 代码构造小模型和真实 PEFT，检查基座及
  LoRA 包装的解包、视觉 embedding forward、反向梯度、参数更新、adapter 保存重载。

## 验证结果

- 原有 core 检查：3 项通过；修改文件通过 Python 语法检查。
- 真实 Qwen modeling 小模型 + PEFT 回归：通过，包括梯度、更新和保存重载。
- GPU 5 单卡真实 MPP 7B + 原始 LingoQA 缓存：通过，使用正式
  `train_sft.py`、`grad_accumulation=4`，仅用 `--max-steps 2` 限制长度。
  共 8 个 micro-batch、2 个 optimizer steps；记录 loss 为 1.171875、2.140625；
  `tracked_lora_max_abs_delta=4.002688729087822e-05`，非零更新已确认。
  训练和保存阶段耗时 56.24 秒（不含环境导入及权重加载）。
  结果位于 `outputs/qwen_core_fix_smoke_20260927/summary.json`。
- 本次没有重跑完整四卡 smoke；单卡验证证明此次模型解析、反向和更新错误已解决，
  不代表全量训练完成或评测能力提升。未改变 DDP 训练逻辑。

## 重启全量训练

使用独立输出目录，避免和 v5 的失败记录混淆。暂存权重和缓存继续从 SSD 读取。

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
systemd-run --user --unit=mpp-lingoqa-sft-v6 --collect \
  --property=WorkingDirectory=/home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa \
  --setenv=CUDA_VISIBLE_DEVICES=0,5,6,7 \
  --setenv=OMP_NUM_THREADS=1 \
  --setenv=MASTER_PORT=29637 \
  --setenv=MPP_CHECKPOINT_ROOT=/tmp/zjr_mpp_fast/checkpoints \
  --setenv=LINGO_CACHE_DIR=/tmp/zjr_mpp_fast/cache/lingoqa_5f_mpp \
  --setenv=SFT_OUTPUT_DIR=/home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/outputs/mpp_lingoqa_lora_v2 \
  bash scripts/run_pipeline.sh train

journalctl --user -fu mpp-lingoqa-sft-v6.service
```

监视：

```bash
watch -n 10 '/home/zjr/anaconda3/envs/mpp_llava/bin/python /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/report_progress.py --output-dir /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/outputs/mpp_lingoqa_lora_v2'
```

之后 `eval_sft` 也需设置 `SFT_OUTPUT_DIR` 到 `outputs/mpp_lingoqa_lora_v2`。
验证/训练期间保留 `/tmp/zjr_mpp_fast`。已有 v5 输出仅含配置，不可作为续训 checkpoint。
