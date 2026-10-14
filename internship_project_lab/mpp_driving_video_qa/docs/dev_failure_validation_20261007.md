# 开发集共同失败复核与固定权重观察验证

日期：2026-10-07 UTC。数据选择09:38；局部生成10:09完成，评分10:14完成。
本次仅诊断、推理、评分和归档：没有训练或改变模型权重，没有修改原问答／参考
或官方Judge分数。人工判断为Codex暂定视觉／文本复核，不是独立人工金标。

## 1. 结果与下一步结论

不支持直接扩大同一种OPD，也不能把全部失败归为训练量不够。最具体的新证据：
保留全图并加入问题相关局部观察，能恢复部分对象类别和跟随目标；但不能普遍
解决左右关系、信号时刻、道路规则和计数。

28题匹配对照中，ROI相对全图副本Judge修复5、退化1，正确数10→14。5题经原图
核对确实恢复摩托车、同向箱式货车或前方汽车。ROI为人工选取，不是自动定位。
这14.29个百分点是定向28题诊断差值，不是全开发集提升，不能宣称统计显著。

下一阶段优先考虑**改观察方式和问题目标绑定的SFT**，而不是重复旧SFT／OPD：

1. 保留原五帧全图，验证不依赖参考答案或人工dev框的细节观察机制，如通用多尺度
   视图或问题相关目标选择。不能退回固定中央裁剪，也不能假设自动机制已有收益。
2. 训练侧使用相同视图机制、原QA，适配完整Q-Former＋projector＋语言LoRA；
   保留全图样本，检验正确样本和因果细节保持。同数量／同预算对照区分真正
   细节收益、增加输入和单纯继续CE训练。不是把这80道dev失败重新拿去训练。
3. 区分问题对应时刻、左右与对象关系，有限核查训练侧同类标签。本次dev题与框
   不进入训练。若反复依据dev设计方法，需要另留新评估数据；官方500已反复
   查看，也不再视作盲测。本次没有启动新的SFT、OPD或RL。
4. 先证实观察适配改善自由回答，再考虑蒸馏真正有视觉优势的教师。参考条件下
   教师答对，不等于其沿学生错误前缀的概率分布能教授正确视觉事实。

## 2. 固定模型、样本与身份

既有完整开发集：Action3120＋Scenery1200＝4320题。最新同协议评测：

| 模型 | Action | Scenery | 合并dev |
|---|---:|---:|---:|
| 完整Q-Former全量SFT父模型 | 58.37% | 64.25% | 60.00% |
| 同预算CE控制组final | 58.75% | 64.92% | 60.46% |
| 反馈OPD v2 final | 58.33% | 65.17% | 60.23% |

OPD相对控制修复80、退化90、净−10题；按segment bootstrap的95%区间
[−0.82,+0.38]pp，无独立优势。两分支dev均选5000步。baseline是已全量SFT的
父模型，不是未适配的官方MPP。报告见`outputs/feedback_opd_eval_v2/report.md`。

从三模型共同Judge失败的1495题（Action1124、Scenery371）中，按问题类型／source
分层选80题：60Action＋20Scenery。另选三模型均Judge正确30题：22Action＋8Scenery。
共110个不同segment，和40k训练池segment交集0，全部原五帧存在。原图路径含train
不代表进入训练，以固定segment划分为准。

这是问句分层诊断，不是失败分布无偏估计；不同segment可能来自同一路段相邻
时刻（例如62/75/78），无原录制ID，不能声称110个统计独立视频。110题总体分数
不等于4320题总体分数。固定seed20261007，cases SHA256为
`de504d797145cb2a6e7a9334918f1c6dd52be82675d4e0c02b39580ee5861687`。

观察模型固定为`outputs/mpp_feedback_sft_control_40k_2epoch_v2/final/`的adapter、
projector.pt和qformer_adapter.pt；原五帧EVA缓存、legacy_v1任务说明、BF16冻结
base／FP32可训练权重、batch1、greedy、max_new_tokens96、max_txt_len1024。
三路权重与数据hash见[manifest](../outputs/dev_failure_validation_v1/manifest.json)
和各组run_manifest.json。原问题、参考答案及权重在所有条件下不变。

## 3. 110题逐项图像与文本复核

已看完22张五题×五帧概览、7张ROI预览，并补看关键原图。80失败的暂定主问题
类型如下，每题只归一类，不解释为已证明的唯一因果：

| 类型 | 数量 | 现象 |
|---|---:|---|
| 对象／左右／车道关系 | 20 | 看到对象却选错跟随目标、方向、避让侧 |
| 对象类别／细节 | 12 | 摩托车认成汽车、前方车辆遗漏、车色错配 |
| 动作与场景原因对应 | 15 | 套用enough space等模板，忽略信号／危险对象 |
| 回答具体性／额外断言 | 11 | 缺少限定或加了没有依据的细节 |
| 时序／动作粒度／时刻对齐 | 6 | amber→red，stopping/decelerating差异 |
| 计数 | 3 | 数量错误或跨帧去重口径不清 |
| 参考或Judge争议 | 13 | 参考方向存疑、宽泛参考遗漏合理答案 |

参考完整支持38、部分支持29、无法确认6、存疑7；复核置信度高41／中34／低5。
部分支持不等于整题标错，存疑不等于已正式修正。存疑题14/28/33/35/43/52/64
仍保留原参考与原分数。重要例子：

- Case14连续帧左侧红色厢式车，模型left而参考right；Case10原图确有白三角
  减速带，模型给出此原因，但参考强调traffic。
- Case5/16首帧amber后转red。Case16原始0.jpg亮中间灯，2.jpg亮顶部灯，补查
  后修正了初步静态颜色判断；不能把全部red回答归为认不出amber。
- Case91原为Judge正确控制：左侧道路施工障碍，参考right避left，模型left避right
  却被接受。Case108控制附加near speed limit，五帧没有实际速度证据。

30个Judge正确控制不等于30个人工金标正确。没有伪造“修正后的真实准确率”。
逐项证据见[review_audit.json](../outputs/dev_failure_validation_v1/review_audit.json)。

## 4. 固定权重时序对照

full_replay为原五帧；last_repeat将末帧重复五次；reverse倒序同五帧。均160视觉
token、原任务说明。原重放103/110逐字一致，7题有措辞变化，Case19从ensure safe
变give way后Judge通过，没有获得新视觉证据。具体数值不稳定来源尚未证明；
身份一致不代表逐位复现，因此下表以新full_replay为基线，不将该1题计作改进。

| 条件 | N | Judge正确 | 原80失败中通过 | 原30控制失败 | 相对新full修复／退化 |
|---|---:|---:|---:|---:|---:|
| 历史原预测 | 110 | 30 | 0 | 0 | — |
| 原五帧重放 | 110 | 31 | 1 | 0 | — |
| 末帧重复 | 110 | 30 | 6 | 6 | 6／7 |
| 倒序五帧 | 110 | 32 | 7 | 5 | 7／6 |

末帧／倒序分别改变51／34题回答，无稳健净优势。部分目标恢复、部分行人／停车
原因丢失，19/20/77等分数变化主要是措辞，91的退化其实是不再放过原方向反转。
保留时序全帧，不把末帧替换或倒序用于部署。本对照既不能证明完全没学会时间，
也不能证明加时间模块就一定有效。

## 5. 全图＋局部对照

选20失败＋8控制。ROI逐帧人工圈问题相关对象，预览后、推理前冻结；5/101/106
部分帧目标离开视野则回退全图，没有根据生成结果调框。Case23原概览低置信，
原始2.jpg补查确认同向白色箱式货车。每时刻两view、同一paired提示、均320视觉token：
paired_duplicate为full+full，paired_roi为full+ROI。裁剪按相同224 EVA预处理。
未给模型参考、目标名称／颜色等隐藏答案。该双视图输入尚未专门SFT，有分布变化。

| 同28题条件 | Judge正确 | 20失败中通过 | 8控制中通过 |
|---|---:|---:|---:|
| 原五帧重放 | 8 | 0 | 8 |
| 全图＋全图副本 | 10 | 3 | 7 |
| 全图＋目标ROI | 14 | 8 | 6 |

ROI相对副本修复5、退化1、净+4；17/28回答变化。不能把原五帧→ROI的8个通过
都算ROI收益：44/78副本已经恢复，58跟随关系低置信且副本已经通过。

| ROI特有修复Case | 副本回答 | ROI回答 | 原图证据 |
|---|---|---|---|
| 2 | car | motorcycle | 正前方踏板摩托车 |
| 23 | car | truck | 同向白色箱式货车，不是右侧对向van |
| 30 | car | motorcycle | 低照度前方摩托车 |
| 37 | no vehicle | car | 正前方白色汽车 |
| 51 | car | truck | 正前方同向箱式货车 |

Case88新增Judge退化：stationary car→stationary object，含义没有反转但丢失类别。
Case97两组都Judge失败，ROI却从red停车转为行人距离，遗漏原有信号约束，说明
分数不翻转仍可能有语义退化。Case67路牌、75方向、71计数仍未解决。
这支持局部细节路线，不代表它是完整场景理解方案、自动路由收益或已完成训练。

59条逐条件语义复核覆盖时序所有Judge翻转、重放翻转、双全图翻转及全部28ROI题；
不是59个独立视频，见[intervention_review.json](../outputs/dev_failure_validation_v1/intervention_review.json)。

## 6. 实现、产物与复现

新增`scripts/validate_dev_failures.py`：prepare/assets/roi_assets/infer/report。
不改训练、模型forward或原评分实现。新增4项测试（时间输入变换、ROI／副本
交错与预算、重放基线、配对子集）及Python3.8编译通过。
首次推理因沙箱GPU不可见报No CUDA GPUs are available，获准宿主GPU后成功，
不是模型崩溃。最终386份生成和评分全部完成。

- [JSON总报告](../outputs/dev_failure_validation_v1/report.json)、
  [指标MD](../outputs/dev_failure_validation_v1/report.md)、case_results.jsonl。
- [80失败离线HTML](../outputs/dev_failure_validation_v1/review_failure.html)、
  [30控制离线HTML](../outputs/dev_failure_validation_v1/review_control.html)：原图、
  冻结ROI、参考、预测、Judge、审核全部内嵌，无需localhost。
- cases.jsonl／baseline_predictions.jsonl固定选择，roi_review_assets保存局部预览。
  各条件run_manifest、predictions、scored保存完整身份／生成／评分，roi_cache是
  本次局部EVA缓存。没有改变旧实验或启动新训练。

已执行，通常无需重跑；完整流程命令：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
  /home/zjr/anaconda3/envs/mpp_llava/bin/python -u scripts/validate_dev_failures.py infer \
  --conditions full_replay last_repeat reverse paired_duplicate paired_roi

for condition in full_replay last_repeat reverse paired_duplicate paired_roi; do
  CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
    /home/zjr/anaconda3/envs/mpp_llava/bin/python -u scripts/evaluate_answers.py \
    --predictions "outputs/dev_failure_validation_v1/$condition/predictions.jsonl" \
    --output-dir "outputs/dev_failure_validation_v1/$condition/scored" --lingo-judge
done

/home/zjr/anaconda3/envs/mpp_llava/bin/python scripts/validate_dev_failures.py report
/home/zjr/anaconda3/envs/mpp_llava/bin/python -m unittest discover \
  -s tests -p 'test_dev_failure_validation.py' -v
```

更改数据／权重／框时用新输出目录，不混入本报告。后续训练尚未执行。
