# 2026-10-14：问题感知 Q-Former 原始问答 SFT

用户先要求上传历史代码，再实现新结构。历史SFT／OPD及checkpoint索引已提交并推送
`origin/main@6b5e7d0b`。该提交可用于读取旧模型实现；本轮结构变化单独记录。

## 机制与现有权重

迁移InstructBLIP的问题条件视觉特征提取：原问题经BERT tokenizer，和每帧32个query
一起经过Q-Former双向自注意力；query通过交叉注意力读取EVA patch。保留query输出
`[B,5,32,768]`，经过原Projector到Qwen，Qwen继续读取原问题。输入没有参考答案提示。
这是内部软特征选择，未新增框／裁剪／外部教师或时序模块，不保证定位或准确率提升。

现有桥包含256个state键，无词嵌入／位置嵌入／文本FFN。本地BERT-base提供74个文本
参数键；safetensors中的LayerNorm gamma/beta映射为weight/bias。旧视觉注意力、交叉
注意力、query FFN和query tokens逐键继承FP32训练值，初始化验证256键完全一致。
Projector和语言QKV LoRA继承同一父模型；新增分支从BERT初始化，未下载整个InstructBLIP。
新增文本权重会增加可训练参数和计算；不能将结构对照的增益全部归因于问题语义。

源参考：[InstructBLIP](https://arxiv.org/abs/2305.06500)、
[LAVIS官方实现](https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_vicuna_instruct.py)。
该官方路径已有逐帧视频推理；本轮把问题输入与已有五帧SFT链路统一。

## 代码与持久化

- `mpp_video_qa/question_qformer.py`：BERT键映射、仅加载文本键、tokenizer／初始化身份。
- `visual_bridge.py`：`auto/none/question`模式，原权重升级、原问题tokenize、按batch×frame
  重复、padding mask、双向注意力、完整权重保存和自动重载。
- `model.py`：CE、答案位置CE、自由生成共享`_embed`，将原`questions`传给视觉桥。
  问题分支要求EVA patch缓存；旧视觉／投影缓存路径维持原行为。
- `feedback_opd_common.py`、`train_feedback_opd.py`：只启用`sft_control`，三组原LR和
  同步／恢复逻辑；记录问题协议、原视觉继承、额外文本参数更新。Teacher和Judge不初始化。
- `generate_answers.py`、`ask_video.py`：保存自动识别的问题分支身份，视频入口走FP32桥
  精度；加载后核对tensor中的身份与生成manifest。
- `report_question_qformer_sft.py`、`run_question_qformer_sft.sh`：完整训练、进度、历史
  对照只读复用、同step比较、dev选模、official锁定选模与原正确／错误分层。

新bridge格式`mpp_visual_bridge_question_full_v1`，保存330个完整state键、初始化／
tokenizer身份。每份`qformer_adapter.pt`旁附`qformer_adapter.question.json`，随bundle
一起复制。保存的模型推理／恢复只需BERT config和vocab，不重复读取初始BERT大权重。
第一次升级需要BERT model.safetensors。auto模式读保存格式，不能把新checkpoint降级
为visual-only；原格式的checkpoint仍走原实现。

EVA特征与问题无关，可复用现有28099段缓存；问题条件Q-Former输出不能按segment
单独缓存。原问题、答案、帧和split不改写，不用开发／官方参考训练。

## 实验规模

父目录`outputs/mpp_lingoqa_qformer_all_1epoch_v1`，新目录
`outputs/mpp_question_qformer_40k_2epoch_v1`，评测
`outputs/question_qformer_eval_v1`。原40k QA（Action25800／Scenery14200）、两轮、
四卡每卡1题累积4、batch16、5000更新；seed44，语言／Projector2.5e-6、Q-Former5e-6，
3%warmup＋cosine、权重衰减.01。保存1000／2500／final；仅原答案与EOS的CE。
bf16冻结语言基座、fp32训练参数；梯度检查点开启。

历史对照为`mpp_feedback_sft_control_40k_2epoch_v2`同三点，从同父模型、原QA池和
预算训练。新report验证这些相同条件及无语言范围扩展；生成保持五帧、160视觉token、
greedy96、batch1。问题最大128个BERT token，无静默截断；Qwen文本最大1536。
两分支各按dev4320选最佳，平分取较早，再评复用官方500；官方不是新的盲测。

## 2026-10-14 已执行验证

44820条train／dev／official输入合同检查完成；28099段EVA缓存齐全，问题最长29个
BERT token，低于128，不需要改数据。`preflight.json`已生成，未启动正式5000步训练。

CPU新增12项：问题影响视觉输出、文本梯度、batch/frame顺序与padding mask、旧权重
精确继承、gamma/beta、完整保存自动重载、恢复AdamW后下一步一致、无静默截断、旧桥
兼容、Bash四卡完整预算、合成完整dev／official报告及错配拒绝。旧完整Q-Former4项、
反馈OPD-v2 9项和实际小Qwen／PEFT 1项回归通过，共26项。合成报告不作为真实准确率。

GPU4真实MPP验证：两次optimizer更新／累积4，共8次原训练QA访问；语言／Projector／
视觉交叉注意力／新增文本FFN代表参数均改变约1e-5；词嵌入梯度非零。同一视频换问题
视觉token最大差2.1803，完整bridge保存重载逐键完全一致，自由生成成功。peak allocated
18.839GiB、reserved19.121GiB；是单卡模型验证，不是四卡正式训练或新benchmark结果。
验证仅保留小型summary，大临时checkpoint自动清理，不留下额外tensor副本。

测试中曾误把末层文本FFN的梯度一定断言为None；实际拼接／checkpoint产生全零tensor，
已改成接受None或全零。该末层文本输出不直接进入Qwen；前部文本FFN／embedding有
非零梯度。GPUsummary的mem_get_info因无索引cuda设备字串产生仪表字段错误，allocated／
reserved／peak仍有效，脚本现改为明确cuda:0；它不影响模型计算或已测峰值。
原验证快照见`docs/artifacts/question_qformer_20261014/`。

## 启动与监视

正式训练尚未启动。数据／缓存已核对，直接启动（运行时选空闲的四张卡）：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
CUDA_VISIBLE_DEVICES=4,5,6,7 MASTER_PORT=29731 \
  bash scripts/run_question_qformer_sft.sh train
```

后台长期运行：

```bash
systemd-run --user --unit=mpp-question-qformer-sft-v1 --collect \
  --property=WorkingDirectory=/home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa \
  --setenv=CUDA_VISIBLE_DEVICES=4,5,6,7 --setenv=MASTER_PORT=29731 \
  /usr/bin/bash scripts/run_question_qformer_sft.sh train
```

监视（任意当前目录）：

```bash
watch -n 10 'bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_question_qformer_sft.sh status'
journalctl --user -fu mpp-question-qformer-sft-v1.service -o cat
```

完成后：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
CUDA_VISIBLE_DEVICES=4 bash scripts/run_question_qformer_sft.sh eval_dev
CUDA_VISIBLE_DEVICES=4 bash scripts/run_question_qformer_sft.sh eval_selected
```

同布局断点恢复：`QUESTION_QFORMER_RESUME_FROM=outputs/mpp_question_qformer_40k_2epoch_v1/checkpoint-002500`
再执行train。保持原新目录和预算，不用只含评测权重的epoch目录恢复，也不把旧模型
checkpoint作为新分支优化器恢复状态。旧历史输出只读复用；新报告和权重独立保存。
