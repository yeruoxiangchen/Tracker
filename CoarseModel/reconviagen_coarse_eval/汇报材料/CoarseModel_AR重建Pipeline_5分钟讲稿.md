# AR 系统下的物体级三维重建 Pipeline：5 分钟讲稿

## 1. 标题页
这次汇报的主线是：AR 系统已经能稳定提供相机位姿和空间感知，但物体级应用最终需要一个真正可用的 Mesh。我们的目标是把手机采集到的 RGB、mask 和 AR pose，变成一个能放回世界坐标、能被后续优化使用的物体 Mesh。

## 2. AR 背景：为什么还缺 Mesh
早期 AR 更偏姿态感知，比如 VIO 或水平仪式的方向估计，知道手机怎么动，但没有物体模型。后来 SLAM 和深度融合能得到轨迹、平面、稀疏点云或局部深度，但这些通常不是干净、封闭、可复用的物体 Mesh。真实任务需要的是可以碰撞、渲染、编辑和对齐到世界坐标的物体资产。

## 3. 我们实现的 AR 重建 Pipeline
当前 pipeline 是：Unity 前端采集 RGB 和 AR pose；server.py 保存 session；用户用 SAM2 做监督分割；服务端做输入 QC；ReconViaGen/TRELLIS 生成粗 Mesh；CoarseModel 把 Mesh 对齐回 AR 世界。新增的 pose/SLAM prior 实验放在独立目录里，不直接污染主流程。

## 4. CoarseModel 思路
CoarseModel 不是生成器，它假设已经有一个粗 Mesh。输入包括粗 Mesh、多视图 RGB/mask、AR 或 COLMAP 相机位姿和内参。输出是 Mesh 到世界坐标的 T_M2W、final_scale、投影可视化，以及可选的 deformation/refined_model.obj。这里放了 GOOD_MESH_TEST 的例子：左边是输入图像，右边是 CoarseModel 对齐后的投影轮廓。即使粗 Mesh 不是完美的，只要整体形状可用，它仍然可以基本估计出世界坐标对齐。

## 5. 为什么要接 Mesh 生成模型
CoarseModel 需要一个粗 Mesh，但 AR/SLAM 本身通常只给稀疏点、轨迹和局部几何。传统深度融合也很难直接得到干净的物体 Mesh。所以当前方案是：ReconViaGen/TRELLIS 负责补全物体整体形状，给 CoarseModel 一个可用初值；CoarseModel 再把这个 Mesh 放回真实世界坐标。

## 6. 当前进展
真实实验里，ReconViaGen 对输入图像覆盖度要求较高。GOOD_MESH_TEST 这种覆盖好的序列，生成的粗 Mesh 已经能进入 CoarseModel 使用；但一些 reconviagen_2026 序列看起来相似，Mesh 质量差异很大。这说明覆盖、主体尺度、mask 边界、模糊和重复帧都会影响生成模型。我们的思路是先做输入筛选，如果仍不稳定，再利用 AR 系统传回的 pose 或 pose+点云增强生成过程。

## 7. Points-to-3D 和我们的修改
参考论文是 Points-to-3D: Structure-Aware 3D Generation with Point Cloud Priors，arXiv:2603.18782。论文基于 TRELLIS，用 point cloud prior 约束 sparse structure，再由模型补全不可见区域。我们的修改是把点云来源换成 AR/SLAM-like prior：由相机 pose 和图像匹配三角化，或者后续由手机 AR 系统直接上传点云，再用 mask 和 pose 筛出物体点。宏观目标是让生成模型不仅看图像，也看到真实可观测几何。

## 8. 下一步
下一步先用真实 SLAM-like prior 做端到端 Mesh smoke，判断 pose/点云是否真的提升粗 Mesh。然后补一组更贴近手机系统输入的 ARPointCloud 采集，让前端直接上传 SLAM 点云。只有当这条路线在 Mesh 层面稳定有收益时，再继续投入更完整的模型训练。
