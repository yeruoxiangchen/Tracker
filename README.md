# Tracker

Tracker 是一个以**相机位姿约束下的三维物体重建与生成**为核心的研究代码仓库，包含模型训练、推理、评估、真实数据处理和 AR 展示等相关代码。

仓库的重点是 **[`pose_aligned_reconstruction`](pose_aligned_reconstruction/)**：将已知相机几何与预训练三维生成先验结合，从稀疏多视图 RGB 观测中重建完整的物体 Mesh。建议从这个目录开始阅读。

## 核心项目：Pose-Aligned Generative Object Reconstruction

### 解决什么问题

稀疏图像只能提供物体的部分观测，生成模型可以补全未观测区域，但重建结果还需要与实际输入的视角和几何关系保持一致。本项目利用同步的相机内参与位姿，为生成过程提供显式的多视图几何约束。

- **输入**：同一物体的稀疏 RGB 视图，以及对应的相机内参和位姿；目标物体的裁剪、坐标规范化和特征缓存由前处理完成。
- **输出**：规范化物体坐标系下的完整三维 Mesh，可结合坐标变换用于 AR 中的物体建模、放置和几何遮挡等应用。
- **模型重点**：利用带位姿的 RGB 观测重建物体几何，论文版本不依赖 VGGT 特征或输入深度观测。

### 方法概览

在 TRELLIS / ReconViaGen 的预训练生成骨干上，将三维查询位置投影到各输入视图，采样并聚合对应位置的图像特征，构建与空间位置相关的多视图条件。该条件参与稀疏结构与结构化 latent 的生成，使模型能够同时利用图像证据、相机几何和学习到的形状先验。

```text
多视图 RGB + 相机内参 / 位姿
             ↓
图像特征提取与基于投影的多视图聚合
             ↓
稀疏结构生成（Native-SS）
             ↓
结构化潜变量生成（Native-SLat）
             ↓
冻结的 Mesh 解码器 → 物体 Mesh
```

当前整理的论文模型使用 **Native-SS 30,000 步 EMA + Native-SLat v2 30,000 步 EMA**。这里的步数对应固定的 checkpoint；模型绑定、数据版本与验证记录均保留在项目文档中。

`pose_aligned_reconstruction` 包含数据缓存制备、训练、推理、评估和实验审计代码，已与早期 `pose_point_depth_mv` 源码解耦。具体入口和依赖见：

- [项目说明与运行入口](pose_aligned_reconstruction/README.md)
- [整理后的验证记录](pose_aligned_reconstruction/VALIDATION.md)
- [模型与实验审计](pose_aligned_reconstruction/审计.md)
- [历史实验与失败分析索引](pose_aligned_reconstruction/audits/README.md)

## 仓库导航

| 目录 | 用途 |
| --- | --- |
| [`pose_aligned_reconstruction/`](pose_aligned_reconstruction/) | 核心模型：位姿约束的生成式物体重建，以及论文版本的训练、推理与评估 |
| [`manual_mesh_reconstruction/`](manual_mesh_reconstruction/) | Mesh 与真实场景的对齐、轮廓优化及 AR 展示工具 |
| [`ar_ss_flow/`](ar_ss_flow/)、[`reconvggt_ar_adapter_a/`](reconvggt_ar_adapter_a/)、[`trellis_point_prior_mv/`](trellis_point_prior_mv/) | 重建流程使用的几何条件、适配器与共享组件 |
| `ReconViaGen/` | 预训练生成骨干及相关依赖 |
| [`internship_project_lab/`](internship_project_lab/) | 其他独立实验项目 |

训练数据、模型权重、缓存和生成结果需要单独准备；相关路径与环境要求见各项目说明。第三方库的使用遵循其各自的许可证。
