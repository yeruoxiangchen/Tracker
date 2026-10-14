# 2026-10-05：教师任务说明与原生视频输入对照

## 已完成的旧教师结果

来源：`outputs/teacher_distillation_v1/teacher_report.json`，2026-10-05 10:33 UTC落盘。
600道开发题、400个视频，Action433题、Scenery167题。教师为原始Qwen3-VL-4B-Instruct，
只读复用Chart项目目录下的基础权重，没有加载Chart训练adapter，也没有进行LingoQA适配。

| 分组 | 现有MPP学生 | 旧教师 | 变化 |
|---|---:|---:|---:|
| 全体600题 | 58.83% | 42.33% | −16.50pp |
| Action433题 | 55.43% | 38.57% | −16.86pp |
| Scenery167题 | 67.66% | 52.10% | −15.57pp |

Judge修复82、退化181、净−99；按视频bootstrap95%区间[−22.02,−11.24]pp。
教师599/600正常结束，没有空回答。平均5.38个英文空格分词，374/600个答案≤3词；
学生平均16.83词。原因关键词题202道，学生96/202、教师56/202，教师其中64条≤3词。
短回答不能独立证明视觉能力不足；逐题可见遗漏原因，也有动作和对象与参考冲突。
这些是参考答案/Judge层面的观察，并非本轮已经逐图人工确认。

当前结果是教师推理对照，不是新增蒸馏训练结果。600道开发题不转作训练数据。

## 本轮对照

| 名称 | 输入 | 任务说明 | 执行 |
|---|---|---|---|
| student | 原MPP五帧EVA路径 | legacy_v1 | 复用已完成预测 |
| legacy_teacher / A | 五张独立图片 | legacy_v1 | 复用已完成预测 |
| multiimage_complete / B | 相同五张图片 | driving_complete_v2 | 新增600题生成/评分 |
| video_complete / C | 相同五帧的视频张量 | driving_complete_v2 | 新增600题生成/评分 |

B明确自车视角、完整回答问题请求、当前动作与下一步建议的区别，依据帧间变化回答运动问题。
提示中不带参考答案，不强制所有题输出原因。保持原模型、greedy、96最大新token和像素预算。

C通过`videos=`、`video_metadata`进入安装环境Transformers4.57.6的原生视频接口。
`do_sample_frames=False`保留所有原帧。数据JSONL没有逐帧采集时间；默认1秒间隔依据
[LingoQA论文的四秒五帧协议](https://arxiv.org/html/2312.14115v3)作均匀相对时间约定，
不是读取到的每个片段绝对时间。可用`--frame-interval-seconds`显式覆盖，新值需新输出目录。

实现时发现当前video smart_resize会先对奇数T取整再检查预算。为保证空间预算一致，
先把五帧重复最后一帧补成六帧，传入帧索引[0,1,2,3,4,4]和fps=1；重复帧时间保持4秒。
视频总像素上下限为补齐后帧数乘以原每图上下限。不会补出新的画面信息。
Qwen3两个帧一组，最终生成0.5、2.5、4.0秒的时间块标记。

真实五帧处理器检查：B每帧grid[1,48,78]，C视频grid[3,48,78]，空间分辨率一致，
即768×1248；B视觉token4680、C2808。原生视频包含时间压缩，因此B/C不是等token预算实验，
报告记录实际grid、token和延迟，不把差异单独归因于时间戳。

## 代码

- `mpp_video_qa/core.py`：新增`driving_complete_v2`。
- `scripts/generate_vlm_teacher.py`：增加输入模式、相对帧间隔、视频像素预算与时间元数据；
  每题记录实际观察参数和耗时。默认旧多图/legacy的manifest保持兼容，可继续旧任务。
- `scripts/teacher_adaptation.py`与`run_teacher_adaptation.sh`：复用原600题及旧结果，
  B/C生成、评分、报告、分歧HTML和进度。每组使用独立目录，可续跑。
- `scripts/teacher_distillation.py`：教师独立的任务说明/输入模式/帧间隔参数贯穿生成、
  报告和身份检查；学生仍读取其训练时任务说明。
- `tests/test_teacher_adaptation.py`：补齐/时间信息/参考隔离、配置对齐、未结束答案、
  弱教师互补性报告及监视入口测试。

报告比较A→B、B→C、student→B、student→C；包含总体、Action、Scenery和原因关键词组，
以及fixed/regressed和视频级bootstrap。回答长度仅是诊断指标。

适配报告不会因教师总体较弱而拒绝生成：同时呈现总体优势与潜在修复数。
“潜在修复”仍是Judge标签，分歧HTML默认导出40条，不声称已经完成视觉复核。
该入口不启动监督生成或训练。旧`teacher_distillation.sh prepare`仍使用原总体优势门槛；
局部教师如何选择训练监督应在看到本轮报告和案例后确定，不能直接取消质量要求开始训练。

## 运行与监视

前台一次运行B、C及评分汇总（同一GPU顺序执行）：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
CUDA_VISIBLE_DEVICES=0 bash scripts/run_teacher_adaptation.sh compare
```

后台运行，可与上面的前台方式二选一：

```bash
systemd-run --user --unit=mpp-teacher-adaptation-v1 --collect \
  --property=WorkingDirectory=/home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa \
  --setenv=CUDA_VISIBLE_DEVICES=0 \
  /usr/bin/bash scripts/run_teacher_adaptation.sh compare
```

任意目录监视：

```bash
watch -n 15 'bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_teacher_adaptation.sh status'
journalctl --user -fu mpp-teacher-adaptation-v1.service -o cat
```

发生中断后重跑同一个`compare`命令，生成按题跳过已有结果，评分按预测文件指纹复用。
`multiimage`或`video`可单独执行一组；两组完成后运行`report`汇总。建议用顺序`compare`。

输出：

```text
outputs/teacher_adaptation_v1/
  run_config.json
  workflow_progress.json
  multiimage_complete/predictions.jsonl
  multiimage_complete/scored/summary.json
  video_complete/predictions.jsonl
  video_complete/scored/summary.json
  report.json
  report.md
  review/review.html
```

## 验证状态

2026-10-05：新增6项及原教师蒸馏10项测试全部通过。真实五帧Processor检查通过，
确认两类视觉张量键、同空间网格、时间块、补齐方式与视觉token数量。
真实旧结果复用核验通过：600问/400视频，师生各600条、数据指纹一致。

14:28 UTC多图GPU单条、14:35 UTC原生视频GPU单条完成，均正常EOS；分别23/47个新token，
单题处理及生成约3.70/4.23秒。产物在`outputs/teacher_adaptation_smoke_v1`。
这验证输入与真实模型接口，不作为教师正确率改善证据。首次沙箱内不可访问CUDA，
在获准的GPU环境重试通过。`torch_dtype`弃用及greedy忽略采样参数警告未影响完成。
正式B/C各600题的结果需运行上述命令获取。
