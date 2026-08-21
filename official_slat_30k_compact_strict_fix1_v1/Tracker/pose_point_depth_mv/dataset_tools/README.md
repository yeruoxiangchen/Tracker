# 数据构建工具

这里是 `pose_point_depth_mv` 当前数据流水线的统一入口。真实扫描几何与
Objaverse 合成几何的 mixed-10k 构建脚本不再分散在
`ar_pose_trellis/dataset_tools` 和 `pixal3d_multiview/dataset_tools`。

## Mixed-10k 流水线

按执行顺序：

1. `download_objaverse.py`：下载冻结的 Objaverse UID 集。
2. `download_omniobject3d.py`：查询或下载 OmniObject3D。
3. `download_omniobject3d_filewise.py`：按冻结文件清单断点续传 OmniObject3D。
4. `freeze_mixed10k_source_inputs.py`：按 pilot 冻结 Objaverse 候选预算，并生成带
   显式排除审计的 Omni archive 清单。
5. `prepare_mixed_mesh10k_sources.py`：审计、解包并冻结两个来源的 Mesh 清单。
6. `run_mixed_multiview_render_worker.py`：把冻结分片分配给 GPU worker。
7. `build_objaverse_multiview_sparse_data.py`：生成 AR 风格轨迹、从 24 个候选中
   选取 8 个视角，并构建图像、mask、相机和 SS latent。
8. `blender_pbr_render_multiview.py`：由上一步调用的 Blender PBR 渲染入口。
9. `audit_mixed_mesh_blender_frames.py`：在不渲染图像的情况下，对比 trimesh/Pixal3D
   与 Blender 导入后的规范化 AABB，先阻断 OBJ/GLB 坐标轴或中心变换错误。
10. `analyze_mixed_render_pilot.py`：汇总正式渲染前的多卡 pilot，核对分片 marker、
   基础设施失败、AABB 误差和低纹理配额，并估算达到目标数据量所需的候选对象数。
11. `render_failure_taxonomy.py`：在不改写原始 manifest 的前提下，统一 worker
   admission 和 pilot aggregation 使用的失败分类。
12. `finalize_mixed_multiview_10k.py`：冻结 object-disjoint 的 10k
   train/val/test 数据。
13. `validate_multiview_sparse_data.py`：检查最终 manifest、图像、相机和 latent。
14. `check_multiview_sparse_data_quality.py`：重算投影、mask 与 visual-hull 质量指标。

Objaverse 5K 渲染进行期间可并行运行的增量缓存链：

- `process_objaverse5k_cache_shards.py`：只消费完成 marker，逐 shard 编排
  A/B/C/R、point-prior、PointPose 和 direct-DINO 阶段；
- `build_dino_only_lifting_cache_direct.py`：直接加载冻结 DINO，完全不导入、加载或
  执行 VGGT，输出现有 SS/SLat 训练代码可读的 lifting cache；
- `merge_dino_only_lifting_shards.py`：以绝对路径引用方式冻结多个完成 cache shard，
  不复制大 tensor 文件。

完整命令见：

`pose_point_depth_mv/Objaverse5K完成Shard并行后处理与DirectDINOOnlyCache命令_20260811.txt`

约 1k 单主体初始微调池另使用：

- `prepare_objaverse_semantic_review.py`：生成逐对象 contact sheet，并冻结显式人工语义决定；
- `freeze_reviewed_mixed1k_dataset.py`：只合并人工确认的 Objaverse 与完成 marker 的
  Omni，校验 source Mesh 去重、旧 val/holdout 排除和资源完整性，再冻结 object/source
  mesh disjoint 的 train/val/test。

Objaverse 缺口渲染完成后的逐段命令见：

`pose_point_depth_mv/Objaverse补充576完成后D8汇总语义审查与Mixed1k冻结命令_20260730.txt`

完整、可复制的运行命令见：

`pose_point_depth_mv/Mixed10k_Omni_OBJ坐标修复后重启命令_20260729.txt`

2026-07-28 的 v3 命令与输出保留为 OBJ 二次旋转失败现场，不可原地续跑。当前正式输入
显式排除无法作为扫描类别使用的 `package.tar.gz`，因此预期为 215 个 archive。

`prepare_mixed_mesh10k_sources.py` 对 Omni 扫描采用两级失败策略：

- 单个对象缺少/空 `Scan.obj`、`Scan.mtl` 或纹理时，写入拒绝审计并跳过该对象；
- 类别没有任何有效对象、tar 成员不安全、archive 缺失、坏对象超过绝对/比例预算，
  或最终有效对象不足时，仍然硬失败。

这避免一个坏对象丢掉整个类别，同时不允许静默吞掉大范围源数据损坏。构建状态可用
只读工具查看：

```bash
/home/zjr/anaconda3/envs/reconviagen/bin/python -u \
  pose_point_depth_mv/dataset_tools/inspect_mixed10k_build.py \
    --source_inputs /data/zjr/dataset10k_20260726/source_inputs_obj11546_omni215_v1 \
    --omni_extract_root /data/zjr/OmniObject3D/raw_scans_extracted_omni215_v1 \
    --source_freeze /data/zjr/dataset10k_20260726/source_freeze_obj11546_omni215_v2 \
    --render_root /data/zjr/dataset10k_20260726/strict_render_mixed10k_cyclescuda_v3 \
    --final_root /data/zjr/pixal3d_multiview/mixed_objaverse6000_omni4000_10k_cyclescuda_v3
```

旧文档保留作历史记录，不应再从其中启动新的 source freeze 或正式渲染。

2026-07-27 的 Objaverse 256-object pilot 若处于 13/16 markers 的已审计状态，恢复和
汇总使用：

`pose_point_depth_mv/Objaverse256_pilot恢复与汇总命令_20260727.txt`

Objaverse pilot 通过后，冻结 11546 个 Objaverse 候选并显式排除 Omni
`package.tar.gz` 的下一步命令见：

`pose_point_depth_mv/混合10k_Objaverse11546_Omni215来源冻结命令_20260727.txt`

## 兼容性

旧路径保留轻量转发脚本，因此已经启动的服务、旧命令和历史记录仍可继续使用。
新命令和后续修改应只引用本目录。
