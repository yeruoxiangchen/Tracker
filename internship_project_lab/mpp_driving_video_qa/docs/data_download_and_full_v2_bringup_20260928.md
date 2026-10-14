# LingoQA 下载、解压与 full-v2 启动问题记录

> 日期：2026-09-26 至 2026-09-28 UTC。本文记录原始数据落盘、失败下载、恢复方式、
> 数据扩量代码和 full-v2 首次缓存启动故障。模型训练效果不在本文判断。

## 1. 原始数据与最终落盘状态

原始数据位于 `/data/zjr/datasets/lingoqa`，没有放入项目仓库：

| 子集 | ZIP 字节数 | 解压 JPG | 标注 |
|---|---:|---:|---:|
| Action | 57,072,793,178 | 122,480 | 265,323 行 |
| Scenery | 8,273,918,069 | 17,540 | 148,506 行 |
| Evaluation | 232,101,697 | 500 | 1,000 行、合并后 500 问 |

三个 ZIP 最终均可读取；Scenery 在 2026-09-28 由分块脚本完成后执行全量 CRC，日志以
`ZIP VERIFIED` 结束。Scenery 解压体积约 7.8 GiB。

## 2. Google Drive 大文件下载失败链

最初使用 `gdown` 同时下载 Action、Scenery 与 Evaluation。小文件和 Evaluation
完成，但 Action、Scenery 的 HTTPS 连接多次在约 5 分钟处结束，只留下前缀文件。
典型残片包括：Action 632,258,536 字节；Scenery 首轮约 565 MB、重试后
811,909,035 字节。进度条停在某个百分比不代表任务仍在运行，应同时检查 PID。

随后使用 `gdown --continue -O 已有文件`，`gdown` 输出
`Skipping already downloaded file`，文件大小没有增加。外层 shell 虽称“resume”，
实际无法推进并以 `download made no progress` 退出。ZIP 因缺少尾部 central directory
而无法被 `unzip -t` 打开，不代表已有字节不能作为远端文件前缀继续下载。

最终采用短 HTTP Range 请求：每块下载后核对 `206`、`Content-Range` 和精确字节数，
完整块才追加到目标文件；连接断开只重试当前块。Action 使用当时位于数据目录的
`resume_drive.py` 完成；Scenery 的可复现实现已归档为
`scripts/resume_scenery_download.py`，并增加已有文件远端前缀校验、文件锁和最终 ZIP
CRC 检查。

一次 Scenery 重下命令中的 `mv scenery/images.zip ...` 在项目目录执行，因相对路径
错误报 `cannot stat`；后续 `gdown -O /data/.../images.zip` 使用绝对路径，仍成功启动，
但再次在约 812 MB 退出。该 `mv` 报错不是下载退出的原因。

## 3. parquet 读取依赖

首次运行 `prepare_lingoqa.py` 时，pandas 报没有 `pyarrow` 或 `fastparquet`。安装
`pyarrow==14.0.2` 的第一次命令在 wheel 下载完成后被手动 `Ctrl-C` 中断，随后重新
执行安装并完成。该错误只影响读取 parquet，不涉及图片、模型或标签内容。

## 4. 首轮 30k Action 抽样的真实含义

旧构建流程先按 segment 固定随机留出 300 个 Action dev 视频，每个视频最多保留
12 道问答，再将其余问答打乱取前 30,000 条。最终覆盖 17,469 个训练视频，其中
8,857 个只有 1 道监督、5,626 个只有 2 道监督。该运行是“构建后 30k 全量训练”，
不是原始 Action 或 LingoQA 全量训练。

## 5. Action + Scenery full-v2 数据修复

`prepare_lingoqa.py` 现支持：

- `--train-size 0`：取消总问题数上限；
- `--max-qa-per-segment 0`：取消每视频问答上限，修复旧代码 `[:0]` 会清空数据的问题；
- `--fixed-dev-data`：原样保留旧 dev，并排除对应视频的全部原始问答；
- `--dev-segments-by-source`：按来源固定新增 dev；
- 问题 ID、跨来源 segment ID、split 视频及帧路径重叠检查；
- 输入、输出 SHA-256 和分来源统计。

实际生成 `datasets/processed/lingoqa_full_v2`：训练 406,212 问/27,599 视频；dev
4,320 问/400 视频；official evaluation 500 问/100 视频。旧 Action dev 3,120 条完全
保留；100 个 Scenery dev 视频的全部原始问答均从训练排除，其中固定 1,200 条用于
生成评测。所有重叠检查为 0。

## 6. full-v2 首次缓存启动失败与修复

2026-09-28 09:59 UTC，`mpp-lingoqa-full-v2-cache-v1` 启动。10:01 UTC，rank 0 在
加载 BLIP2 时失败：

```text
RuntimeError: checkpoint url or path is invalid
```

原因是 `run_full_v2.sh` 把统一 `checkpoint_root` 指向 `/tmp/zjr_mpp_fast/checkpoints`。
该 SSD 副本只包含 Qwen7B-chat 和 MPP sharded checkpoint，没有持久目录中的
`ckpt/blip2/blip2_pretrained_flant5xxl.pth`。这是资产路由错误，发生在任何新视频编码
之前，不影响旧缓存，也没有产生训练 checkpoint。

修复后两个来源分开：

- BLIP2 视觉权重：项目持久 checkpoint 目录；
- MPP 投影 shards：`/tmp/zjr_mpp_fast/checkpoints` 快速副本。

`run_full_v2.sh cache` 在 torchrun 前检查两项资产；`mpp_config()` 新增可选
`visual_checkpoint_root`，不改变训练时 Qwen/MPP 的快速路径。真实单卡 smoke 已完成
一段 Scenery 视频的 5 帧编码：BLIP2 成功加载、MPP 投影加载 2 个张量并跳过 259 个
无关 Qwen 张量，最终生成合法缓存。

## 7. full-v2 工程能力与验证边界

当前还增加了：旧缓存协议/segment/帧路径/形状/有限值校验，缺失缓存增量生成；训练
每 5,000 optimizer step 保存 adapter、优化器、scheduler、数据位置和各 rank RNG；
生成输出绑定数据、adapter、prompt 与参数指纹；Action/Scenery 分项 paired 报告。

数据不变量、Python 编译、shell 语法、旧 dev 完整保留、缓存匹配辅助函数和单条真实
视觉缓存均已验证。项目环境没有安装 `pytest`，所以本轮没有在该环境运行原 pytest
套件。可恢复 checkpoint 代码尚未经过 full-v2 四卡中断恢复实跑，应在首个 5,000-step
checkpoint 后再做事实确认，不能提前写成“恢复能力已实测通过”。

## 8. 曾出现但不属于模型故障的运行信息

- `nohup: ignoring input`：表示后台进程不读取终端 stdin，是正常信息。
- `flash_attn ... fail`：旧版 Qwen 找不到可选 FlashAttention 扩展，会走普通
  attention；影响速度和显存效率，不会改变标签或直接说明答案错误。
- `do_sample=False` 同时出现 `top_p/top_k` warning：greedy 生成会忽略采样参数，
  当前 Base/SFT 对比仍是确定性解码。
- `report_progress.py: No such file or directory`：一次监视命令在 `~/Tracker` 执行却
  使用项目相对路径；改为绝对路径后解决，不是训练退出。
- `initializing_or_not_started`：只表示输出目录还没有运行状态文件，不能据此判定
  systemd 服务死亡；模型加载期间应同时检查 `systemctl` 和 journal。

## 9. 上游 MPP 工作区问题的归档边界

更早的共享 MPP-LLaVA 环境问题包括 PyTorch wheel 哈希/不完整下载、ModelScope
新版本在 Python 3.8 使用 `list[int]` 导致异常、官方权重下载、MPP checkpoint 分片
转换与逐 tensor 核验。这些已归档在兄弟目录
`mpp_llava_video_workspace/docs/project_master_timeline_20260924.md` 和
`mpp_checkpoint_sharded_loading_fix_20260924.md`。驾驶问答项目引用其最终环境和权重，
不把这些历史重复记成自己的模型实验。
