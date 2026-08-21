# 人工 Mesh 重建工具（独立目录）

该目录是人工真实数据测试的统一入口，不再要求从
`pose_point_depth_mv/background_jobs` 中拼接多段命令。

手机界面由根目录的 `ARposeTTracker.cs` 在运行时统一整理：录制/重新录制
固定为安全区最底部的全宽主按钮；Mesh 显示、轮廓方式、快速校准和位姿诊断四项固定在
主按钮上方，不再遮挡它。旧 Unity Scene 中已绑定的按钮会被自动重排并统一
颜色、字号和最小触控高度，不需要重新拖拽 Inspector 引用。

统一入口及其各阶段的可执行实现都位于本目录。模型网络定义、TRELLIS
sampler/decoder 和已审计的 checkpoint loader 仍以 `pose_point_depth_mv`、
`ReconViaGen` 为只读库依赖；没有复制训练器或另造一份网络结构，避免人工测试版
与正式训练版发生参数名、采样或 decoder 语义漂移。具体依赖文件及 SHA256 记录在
[`PACKAGE_MANIFEST.json`](PACKAGE_MANIFEST.json)。

当前固定模型身份：

- 当前模型：official no-VGGT Native-SS step 30000 + no-VGGT Native-SLat step 30000；
- 对照：strict original ReconViaGen（VGGT → Stock SS → Stock SLat）；
- 当前模型的 Mesh 使用 TRELLIS decoder 原生 sparse-grid/runtime-O 坐标，禁止再施加历史错误的 `(x,y,z)->(x,z,-y)` 旋转；
- 原相机轮廓使用 `Mesh_O → T_O2W → Mesh_W → T_W2C → K_raw + distortion`；
- 所有结果只用于人工定性检查，不自动升级成 held-out 科学结论。

## 手机第一阶段服务：全候选先固定 O，再做训练一致球面 FPS8

新服务位于 [`server.py`](server.py)，没有修改
`pose_point_depth_mv/server.py`。旧服务只作为已经可用的手机上传、交互式 SAM2
分割和会话管理层；从 raw-cache、runtime-O、DINO-only、SS30K+SLat30K、Mesh
导出到轮廓回投均由本目录代码执行。

```bash
cd /home/zjr/Tracker
conda activate reconviagen
GPU=4 python manual_mesh_reconstruction/server.py
```

它保持 `/input_qc`、`/generate` 和返回的 `mobile_ar.mesh_url` HTTP 合同，并在两者之间
增加 `/prepare_runtime_o`。为修复
Android 上 Anchor 长期处于 Limited 时 Mesh 永久隐藏的问题，当前根目录
`ARposeTTracker.cs` 已同步更新；手机端需要用该脚本重新编译安装。每一轮采集严格执行：

1. 保留所有已上传且具有 RGB、最终 Mask、K 和 AR Pose 的候选帧；不接受客户端
   选帧，也不按质量、roll、轨迹跳变、mask-ray 或点云删帧。
2. 原生 `XRCpuImage.Transformation.None` RGB/Mask 做物理 x/y 转置，K 同步交换，
   `T_W2C` 只做 Unity→CV 转换，不把 `displayMatrix` 当外参。
3. 用完整候选域构建 official-compatible Z-up model-O；冻结 O 后只执行一次官方
   训练同款的单起点 3D 球面最远点采样。Mask 仅用于前景有效性，不枚举候选起点，
   不按质量门排序，也不再运行时间均匀、轨迹均匀或随机分支。
4. 服务端先停在冻结 Runtime-O 的边界，随后运行 no-VGGT SS30K+SLat30K，得到
   原生 Mesh-O。Mesh 几何从此冻结；服务端从原始完整候选域中以球面 FPS 选最多
   24 帧，其中交替分成训练/留出视角，只优化一个有界的 O2W 旋转、平移和尺度。
   3 万面代理只用于可微轮廓梯度，原始完整 Mesh 用于全部输入帧的最终分辨率验收。
   若训练视角、留出视角或全输入验收任一退化，保留原始 O2W。
5. 将通过安全门的 `T_O2A0`（或被保留的原始值）直接烘入 A0-local Unity
   `.armesh`；同时保留
   runtime-O OBJ、internal-world OBJ/GLB、Mesh
   预览、完整输入域优化前后轮廓、标定后模型输入 8 帧轮廓，以及转回原始手机
   图像朝向的轮廓总览。

### Mesh 后完整输入图像 O2W 优化与旧结果复用

该阶段已经是 `server.py` 正常 `/generate` 流程的一部分，不需要额外按按钮或再次
运行模型。默认使用最多 24 个球面分散视角、40 次迭代、长边 160 的可微轮廓，
并把 decoder 完整 Mesh 投到全部有效输入帧进行最终验收。原始 runtime cache、
Mesh-O、DINO/SS/SLat 结果均保持只读。

每次新重建的报告与可视化位于：

```text
reconstructions/<session>/branches/01_training_spherical_farthest8/
  04b_input_o2w_refinement/
    report.json
    selected_T_O2W.json
    selected_T_O2W.npz
    optimization_proxy_mesh_o.obj
    01_初始O2W输入轮廓/
    02_优化后O2W输入轮廓/
    04_初始O2W总览/
    05_优化后O2W总览/
    07_优化后世界Mesh/
```

对已经完成、已有 Mesh 的旧手机重建，可直接运行下列命令。入口会自动解析该会话
的 runtime manifest、seed-42 Mesh-O 与 Mesh frame report，不重新执行 DINO、SS、
SLat 或 decoder：

```bash
cd /home/zjr/Tracker
conda activate reconviagen

RECON='/home/zjr/Tracker/pose_point_depth_mv/outputs2/手机AR第一阶段_全视图统一O_训练一致球面最远8帧/reconstructions/你的session'
GPU=4 python -u -m manual_mesh_reconstruction.optimize_o2w \
  --reconstruction_dir "${RECON}" \
  --gpu 4 \
  --resume
```

也可传入 branch 目录或 `branch_report.json`；多分支目录用 `--branch_name` 明确选择。
默认输出到该 branch 的 `04b_input_o2w_refinement/`。若要保留多个参数版本，使用
`--output_dir /path/to/new_output`。`report.json` 中 `accepted=false` 不是运行失败，
而是候选未通过留出/全帧安全门，`selected_T_O2W.npz` 会明确保存原始 O2W。

手机端会同时核对 `.armesh` 的字节数、SHA-256、二进制头、顶点/索引上限和索引
范围。当前 Anchor 合同只有采集/显示共用的 A0。每次新录制先递增本地 capture generation，停止
复用上一轮应用状态，按 Mesh→A0 销毁，并至少跨过一个 `WaitForEndOfFrame`。普通重录明确
不调用 `ARSession.Reset()`：各轮共享同一个持续增强的 ARCore 环境地图，而 Mesh、A0、
网络协程等应用状态仍按 generation 隔离。新 A0 只有在旧 A0 的 `TrackableId` 已从
`ARAnchorManager` 消失、清理后的新相机帧已经到达、并且连续稳定 Tracking 后才能建立；
任一条件超时都会停止本轮，而不是带着旧 Trackable 继续录制。
再次按“录制”走上述软隔离，不 Reset Session；只有人工审核界面的“退出/重置”按钮会
销毁本轮 Mesh/A0 后调用 `ARSession.Reset()`，并等待新的相机帧与稳定 Tracking 再解锁录制。

录制开始前在首张相机帧位置建立真实 A0 `ARAnchor`，旋转保持 Unity 重力/世界轴；所有上传
相机位姿都以 A0 为父坐标。Mask 确认后，服务端用全视图 pose+mask 求得并冻结
`T_O2A0`，按 `Mesh_A0=T_O2A0@Mesh_O` 导出 `unity_capture_anchor_a0`。手机只在 A0
连续稳定 Tracking 时，以 `localPosition=0`、`localRotation=identity`、
`localScale=1` 把 Mesh 挂到 A0；禁止在 Unity 再次应用 `T_O2A0`。

代码强制 `Screen.sleepTimeout=NeverSleep`。若仍切到后台，则保留同一 Session 内的 A0
但立即隐藏 Mesh；回到应用后，只有 A0
连续处于 `Tracking` 0.75 秒才重新显示。`Limited`/`None` 绝不会被冻结 Unity 世界位姿
伪装成稳定。这一合同支持同一应用进程、未 Reset 的 AR Session 恢复；如果 Android 杀掉
进程或 Session 被重建，仍需 Cloud Anchor/视觉 Marker 才能跨 Session 恢复。

由于手机位姿协议已更换为 `camera_frame_received_anchor_a0_relative_v1`，必须将根目录
[`ARposeTTracker.cs`](../ARposeTTracker.cs) 复制到 Unity 项目后重新编译并安装 APK。旧 APK
会被服务端明确拒绝，不会被当成 A0-relative 位姿继续重建。Inspector 不需新增
Anchor 组件；保持现有 `AR Anchor Manager` 已绑定即可，A0 由该 Manager 动态创建。
当前第一阶段明确假设用户在 Mask/Runtime-O/模型等待期间留在原场景，避免突然大位移、遮挡
镜头或系统重建 AR Session；跨 Session 持久恢复仍不在该最小方案范围内。

### Unity 双轮廓显示与切换按钮

根目录 [`ARposeTTracker.cs`](../ARposeTTracker.cs) 现在保留两种互斥显示路径：

- `实时3D剪影边`：每 0.1 秒在 CPU 上判断三角面正反向，只用
  `MeshTopology.Lines` 画剪影拓扑边；
- `服务器式屏幕轮廓`：用第二台低分辨率 Camera 将完整 Mesh 栅格化为二值
  silhouette，再在屏幕空间显示 `dilate(mask) - mask` 的青色外边，与服务端轮廓语义一致。

把 [`unity/ARMeshSilhouetteMask.shader`](unity/ARMeshSilhouetteMask.shader) 和
[`unity/ARMeshScreenSpaceOutline.shader`](unity/ARMeshScreenSpaceOutline.shader) 复制到 Unity
`Assets/Shaders/`，分别创建 Material，并绑定到 `Server Style Mask Material Template` 和
`Server Style Outline Material Template`。保留原 `Mesh Display Button` 用于显示/隐藏。
`Mesh Outline Method Button` 留空时，代码会在运行时复制原显示按钮并自动排在其后；
也可在场景中新建 UI Button 并显式绑定，以完全控制布局。`Mesh Outline Method Text`
是可选状态文字。默认仍启动3D边；两种路径不会同时运行，服务器式的副相机、
RenderTexture 和填充 Mesh 也只在第一次切换到该方式时才延迟创建。`Server Style Outline Layer`
默认为 31，该 Layer 不应再放其他场景对象。Android 构建必须通过 Inspector 绑定 Material，
避免运行时 `Shader.Find` 所需 Shader 被剥离。服务器式 mask 默认使用 0.5 倍长宽（约 1/4
像素）且限制在 30fps，可分别用 `Server Style Render Scale` 与
`Server Style Max Frames Per Second` 调整。
成功下载后 logcat 会包含类似
`[ARMesh] received_bytes=... vertices=... triangles=... reference=...` 的记录。

### 返回现场后的 SAM2 Tiny 快速轮廓校准

Mesh 首次显示后，手机界面会出现“快速校准轮廓”按钮。该功能只修正已经存在的
A0-local Mesh 的显示位姿，不重新运行 SS/SLat、不改 Mesh 顶点，也不移动或重建 A0：

1. 第一次按按钮后，手机以 A0 为坐标系，每 0.25 秒上传一张同步 RGB、K 和相机
   位姿；候选帧没有硬上限。用户围绕物体中心缓慢移动，至少采集 16 帧，建议
   32 帧或更多。
2. 再按一次按钮后，服务端以“当前已显示 Mesh 经现有 Transform 变换后的 AABB
   中心”为物体中心，从全部候选相机方向执行球面最远点采样，固定选出 16 帧。
   选中集合按原始时间顺序交给常驻的 SAM2.1 Hiera Tiny video predictor，避免
   打乱视频传播；球面 FPS 顺序交替分成 8 个训练视角和 8 个留出视角，使两组
   都保持视角分散。当前 Mesh 投影只用于在少数训练帧生成宽松 box、少量正点和
   背景负点；最终 Mask 来自 RGB 视频传播，不直接把 Mesh 投影当作观测真值。
3. 服务端固定 Mesh 几何，只优化一个严格有界的相似变换：旋转不超过 12 度、
   平移不超过 0.12 米、尺度限制在 0.88 到 1.12。只有整体 IoU、训练视角、留出
   视角和逐视角胜率同时过门才把新变换返回手机；任一异常、分割失败或门未过，
   手机继续使用优化前的精确位姿。

快速校准 v4 与主重建严格共用同一相机合同：手机上传
`XRCpuImage.Transformation.None`，服务端只做一次图像 x/y 转置与 K 交换，Unity
Pose 转 CV 外参时额外光轴旋转固定为 0 度。旧 v2 曾在这里重复加入 90 度光轴
旋转；v3 使用时间均匀 8 帧，不能保证围绕物体的角度覆盖。旧 `attempt_NNN` 会
原样保留作历史审计，但不能作为 v4 校准证据，也不会被新版结果复用。

SAM2 位于 `any6d_sam3d` 环境，主服务仍在 `reconviagen` 环境。主服务会在正常 Mesh
重建完成后自动启动并预热一个 localhost-only 常驻进程，因此启动命令不变：

```bash
cd /home/zjr/Tracker
conda activate reconviagen
GPU=4 python manual_mesh_reconstruction/server.py
```

Hydra 只负责 SAM2 Tiny 的模型配置与构建；真正避免每次重复加载权重的是这个常驻
worker。默认解释器为
`/home/zjr/anaconda3/envs/any6d_sam3d/bin/python`，端口为 `127.0.0.1:5091`，可分别
通过 `--sam2_python` 和 `--sam2_worker_port` 覆盖。每次校准的原始帧、Mesh prompt、
SAM2 Mask 和接受/拒绝报告均保存在：

```text
reconstructions/<session>/08_fast_a0_silhouette_refinement/attempt_NNN/
  00_phone_rgb_a0/
  04_alignment_optimization/
    01_corrected_rgb/
    02_mesh_prompt_projection/
    03_sam2_tiny_observed_masks/
    report.json
```

`Alignment Refine Button` 在 Inspector 中是可选绑定；留空时脚本会复制现有 Mesh
显示按钮并自动创建，不需要给场景新增组件。但由于根目录 `ARposeTTracker.cs` 已
更新，仍须替换 Unity 项目中的脚本并重新编译安装 APK。

### 校准前/后显式位姿诊断录制

Mesh 显示后，底部操作区会出现“录制位姿诊断”按钮。默认不再自动回传；
用户可以在快速校准前按一次，校准后再按一次，得到两个独立、明确标记为
`pre_fast_alignment` 和 `post_fast_alignment_*` 的 attempt。服务器会返回本次 Mesh
重建实际消费的 8 个原始 `poses.txt` 相机位姿；手机逐一寻找尚未采集的最近目标，
同时显示厘米/角度残差，并且只有原始相机帧和最终屏幕截图都满足默认 2.5 cm、3°
门限时才接受。每个重建输入位姿恰好保存一次，不能用任意新视角替代。

手机实显截图使用原生屏幕分辨率、无损 PNG，不再缩到长边 1080，也不再做 JPEG
压缩；原始相机证据仍以 CPU image 原生尺寸、高质量 JPEG 单独保存。后者的分辨率
由 ARCore CPU image 决定，不能拿它代表手机屏幕截图清晰度。

手机每帧同时上传两张真实证据：

- `ScreenCapture.CaptureScreenshotAsTexture` 在 `WaitForEndOfFrame` 后截取的实际手机画面，
  包含 AR 相机、当前 Mesh 轮廓和 UI；
- 同期原始 `XRCpuImage.Transformation.None` JPEG，以及对应的 K、A0-relative
  相机位姿和时间戳。

同步保存 A0 的 `TrackableId`/跟踪状态/世界位姿，相机世界位姿，Mesh 的
A0-local 与世界 Transform，display/projection matrix，AR Session/应用前后台/设备/电池
状态，以及最近一次快速校准的接受状态和报告路径。服务器不信任手机截图中的
轮廓，而是读取已返回的不可变 `.armesh`，使用该帧上传的当前 Mesh Transform、
A0-relative 相机位姿和 K 在原始传感器域重新栅格化，再严格使用 ARFoundation
`displayMatrix` 变换到手机显示域。像素旋转/裁剪/镜像只在这个显示变换里完成，
不会通过额外旋转相机外参来“修图”。此外还保存现场帧与其严格匹配的原始重建
输入帧对照。

结果位于：

```text
reconstructions/<session>/09_mobile_render_overlay_audit/attempt_NNN/
  00_mobile_screen_composite/
  01_mobile_outline_texture/
  02_frame_metadata/
  03_raw_camera_rgb/
  04_server_raw_sensor_reprojection/
  05_server_display_aligned_reprojection/
  06_original_reconstruction_input/
  07_live_vs_reconstruction_input/
  08_phone_vs_server_display_comparison/
  手机实际最终渲染总览.jpg
  服务器显示方向同位姿Mesh复投影总览.jpg
  手机实际与服务器复算逐帧对照总览.jpg
  重建原始输入与现场严格同位姿图像对照总览.jpg
  report.json
```

该目录具有明确的 `diagnostic-only` scope guard，服务器不会把其中任何图像或位姿送入
选帧、SAM2、SS/SLat、快速校准或指标计算。对照图的快速判读是：

- 服务器同位姿复算对齐，但手机实显不对：优先查 Unity Mesh/相机/屏幕渲染链；
- 两者以相同方式偏移或旋转：优先查当前 O2W/Mesh Transform 或 A0 相机位姿；
- 校准前对、校准后两者均变差：快速校准返回的相似变换有问题；
- 原重建的 `07_original_phone_input_contours` 对，而新诊断的服务器复算不对：
  优先查现场 A0 重定位、当前相机位姿或 RGB-pose 时间同步。

`Pose Diagnostic Record Button` 在 Inspector 中是可选绑定；留空时会复制现有 Mesh
按钮并自动放入底部操作区，不需要改 Scene 组件。但 C# 逻辑已改，仍需替换脚本后
重新编译并安装 APK。高级用户可开启 `Auto Start Mobile Overlay Audit` 恢复自动开始。

默认输出目录为：

```text
pose_point_depth_mv/outputs2/手机AR第一阶段_全视图统一O_训练一致球面最远8帧/
  runtime/                              # 手机上传与 SAM2 会话
  reconstructions/<session>/
    00_all_view_capture_report.json
    01_all_view_raw_cache/
    selection_plan.json
    branches/01_training_spherical_farthest8/
    phase1_session_report.json
```

仅检查上传、像素轴、全视图 O 和严格球面 FPS8 而不占用模型推理时，可加
`--capture_only`。服务端不再暴露 `--selection_mode`，避免误启动旧选帧分支。

## 最短命令：直接输入数据集

对于已经有 COLMAP 的 CoarseModel/Omni 数据集，最推荐显式指定
`--colmap-mode reuse`。下面一条命令会沿用既有相机内外参，先用全部“已注册且
有 mask”的输入帧冻结 official-compatible model-O，再按时间轴均匀选择 8 帧作为
模型输入，随后完成两套重建、Mesh 预览和当前模型回投原图的轮廓：

```bash
cd /home/zjr/Tracker

DATA=/home/zjr/Tracker/CoarseModel/datasets/heimei
OUT=/home/zjr/Tracker/manual_mesh_reconstruction/output/heimei_colmap_reuse_v1

bash manual_mesh_reconstruction/run_reconstruction.sh \
  --dataset-path "${DATA}" \
  --dataset-type colmap \
  --colmap-mode reuse \
  --frame-selection time_uniform \
  --selected-view-count 8 \
  --output-dir "${OUT}" \
  --gpu 0
```

## 既有 COLMAP 与重新构建 COLMAP

`--colmap-mode` 有三个互斥值：

- `reuse`：严格要求数据集内已经存在完整的 COLMAP model，或通过
  `--colmap-sparse /path/to/sparse/0` 明确给出；若不存在就直接失败。不会重新
  feature extraction、matching 或 mapping。若源模型只有 `.bin`，只会调用
  `model_converter` 转成只读文本副本，不重新求相机位姿。
- `rebuild`：忽略数据集中的既有 model，在本次输出的
  `00_dataset_adapter/colmap_workspace/` 内对完整 RGB 序列重新运行 COLMAP；不
  覆盖源数据集。COLMAP 完成后保留全部注册成功且有 mask 的帧；先构建 O，再从
  同一完整帧域里选择 8 帧。
- `auto`（默认）：发现完整既有 model 时采用 `reuse`；否则采用 `rebuild`。
  若要做严格可解释对照，建议不要依赖自动判断，而是显式写 `reuse` 或
  `rebuild`。

强制沿用自定义既有 model：

```bash
bash manual_mesh_reconstruction/run_reconstruction.sh \
  --dataset-path /path/to/你的数据集 \
  --dataset-type colmap \
  --colmap-mode reuse \
  --colmap-sparse /path/to/你的数据集/sparse/0 \
  --frame-selection time_uniform \
  --selected-view-count 8 \
  --output-dir /home/zjr/Tracker/manual_mesh_reconstruction/output/实验名 \
  --gpu 0
```

在独立目录重新构建 COLMAP：

```bash
bash manual_mesh_reconstruction/run_reconstruction.sh \
  --dataset-path /path/to/你的数据集 \
  --dataset-type colmap \
  --colmap-mode rebuild \
  --colmap-matcher sequential \
  --frame-selection time_uniform \
  --selected-view-count 8 \
  --output-dir /home/zjr/Tracker/manual_mesh_reconstruction/output/实验名 \
  --gpu 0
```

视频/有序采集默认使用 `sequential` matcher；确实无时间邻接关系时可改为
`--colmap-matcher exhaustive`。如果 COLMAP 的 CUDA/SIFT 不可用，可加
`--colmap-cpu`。`--colmap-use-foreground-masks` 只控制重建模式下的特征提取；
复用模式不会重新提特征，因此该参数在 `reuse` 中不起作用。

报告会明确记录：请求模式、实际模式、选中的原始 model、原始 model 文件
SHA256、是否重新求解 COLMAP、是否仅执行了 model converter、参与 O 构建的
全部输入帧，以及最终采用的 8 个模型输入帧。源 `points3D` 只用于确认 COLMAP
model 完整，不作为 runtime-O 物体点云；runtime-O 仍严格由 pose+mask 得到。

## Omni 数据集

已经整理成 `images/masks/sparse/0` 的 Omni 回放可直接传数据集根目录：

```bash
bash manual_mesh_reconstruction/run_reconstruction.sh \
  --dataset-path \
  'pose_point_depth_mv/outputs/可视AR/OmniHoldout64复杂样本真实采集流程回放_20260811_v1/datasets/omni_plant_012_replay64_v2' \
  --dataset-type colmap \
  --colmap-mode reuse \
  --frame-selection time_uniform \
  --selected-view-count 8 \
  --output-dir manual_mesh_reconstruction/output/omni_plant_012_reuse_v1 \
  --gpu 0
```

官方 Omni 解包对象若目录结构为 `standard/images`、`standard/matting`、
`standard/sparse/0`，应把 `standard` 目录作为 `--dataset-path`。

## 手机采集数据

可以直接输入已完成重建目录；适配器会优先绑定其中已冻结的 raw-cache：

```bash
bash manual_mesh_reconstruction/run_reconstruction.sh \
  --dataset-path \
  'pose_point_depth_mv/outputs/可视AR/reconstructions/real_official_slat_step25000_retest_20260812_171117_303_seed42_spherical_v1' \
  --dataset-type phone \
  --frame-selection time_uniform \
  --selected-view-count 8 \
  --output-dir manual_mesh_reconstruction/output/phone_retest_v1 \
  --gpu 0
```

也可以直接输入采集 runtime 根目录并给 session：

```bash
bash manual_mesh_reconstruction/run_reconstruction.sh \
  --dataset-path 'pose_point_depth_mv/outputs/可视AR/runtime' \
  --dataset-type phone \
  --session-id 20260816_040547_970 \
  --frame-selection time_uniform \
  --selected-view-count 8 \
  --output-dir manual_mesh_reconstruction/output/phone_runtime_v1 \
  --gpu 0
```

也接受 `runtime/data/<session>`、`runtime/masks/<session>` 或包含
`images|rgb + masks + poses.txt` 的固化数据集目录。手机分支使用冻结的
ARFoundation 相机位姿和 CPU 图像轴合同，不运行 COLMAP。

## Objectron：pose+mask O 或真值物体位姿 O

pose+mask（与普通真实采集相同的 O 定义）：

```bash
bash manual_mesh_reconstruction/run_reconstruction.sh \
  --dataset-path yxc/datasets/Objectron_real_pose_2clips_20260819_v1 \
  --dataset-type objectron \
  --objectron-clip camera/batch-7/24 \
  --objectron-object-id 0 \
  --objectron-o pose_mask \
  --frame-selection time_uniform \
  --selected-view-count 8 \
  --output-dir manual_mesh_reconstruction/output/objectron_camera_pose_mask_v1 \
  --gpu 0
```

使用 Objectron 官方物体旋转、平移和尺度定义 O（oracle diagnostic）：

```bash
bash manual_mesh_reconstruction/run_reconstruction.sh \
  --dataset-path yxc/datasets/Objectron_real_pose_2clips_20260819_v1 \
  --dataset-type objectron \
  --objectron-clip camera/batch-7/24 \
  --objectron-object-id 0 \
  --objectron-o true_object_pose \
  --frame-selection time_uniform \
  --selected-view-count 8 \
  --output-dir manual_mesh_reconstruction/output/objectron_camera_true_o_v1 \
  --gpu 0
```

两种 Objectron 分支使用完全相同的官方相机位姿、图像、mask 和选中帧；
pose+mask 分支先用全部输入视图构建 O，true-object-pose 分支则在选帧前已经由
官方物体标注冻结 O。后者只替换 O 的定义，因此可用于隔离 pose+mask 估计误差。报告会标记
`oracle_object_pose_consumed=true`，不能把它混写成一般真实部署结果。

## 选帧与恢复

- 数据集适配器不再提前删减到 8 帧，而是无损保留全部有效 RGB、mask、K 和
  `T_W2C`。runtime 阶段先对全部 mask 执行与模型前端一致的去畸变，再用全部
  输入视图估计中心、尺度和轴，冻结唯一 `T_O2W/T_W2O`；完成后才执行选帧。
- 默认 model-O 轴策略是 `official_compatible_z_up_v1`：model +X 等于旧
  runtime-O +X，model +Y 等于旧 runtime-O -Z，model +Z 等于旧 runtime-O +Y。
  该映射是行列式为 +1 的纯旋转，不改变中心和尺度。仅做旧结果配对回放时可显式
  加 `--pose-mask-object-frame-policy legacy_y_up_v1`。
- `--frame-selection time_uniform`：冻结 O 后按原始时间顺序均匀取帧，包含首尾；
  默认 8 帧。
- `--frame-selection random --selection-seed 20260819`：冻结同一个 O 后无放回
  随机抽帧，之后按时间顺序执行，且固定 seed 可复现。
- 高级 raw-cache 入口可用
  `--view-selection-policy fixed_frame_names_valid_mask` 并重复传入
  `--fixed-frame-name 00001.jpg`，在完整输入序列冻结 O 后严格重放指定帧；名称缺失、
  重复、数量不等于 `--selected-view-count` 或去畸变后 Mask 为空都会直接失败。
- 高级 raw-cache 入口的球面分散策略同样先冻结全视图 O；候选打分只读取这个 O，
  不允许为候选子集重新估计临时 O。
- 完整阶段结束后可在相同命令上加 `--resume`。已有 COLMAP 的完整阶段和已完成
  推理会按报告及 SHA256 复用；旧的“先选帧再构建 O”适配器产物会 fail-closed，
  必须换新输出目录，避免静默混用旧坐标合同。

## 高级入口：已有 raw-cache report

如果已经有标准 raw-cache report（包含原图、mask、相机内参和 `T_W2C`），
可跳过数据集适配阶段：

```bash
cd /home/zjr/Tracker

bash manual_mesh_reconstruction/run_reconstruction.sh \
  --raw-cache-report /path/to/raw_cache_report.json \
  --object category:object_id \
  --output-dir /home/zjr/Tracker/manual_mesh_reconstruction/output/你的实验名 \
  --gpu 0
```

这个高级入口默认在 raw report 的全部输入视图上先冻结 O，再使用 8 帧球面最远
点策略。若 raw report 只有一个物体，可省略 `--object`。直接使用
`--dataset-path` 时则按上面的 `--frame-selection` 在 O 冻结之后选帧。

## 只查看计划，不占 GPU

```bash
bash manual_mesh_reconstruction/run_reconstruction.sh \
  --raw-cache-report /path/to/raw_cache_report.json \
  --object category:object_id \
  --output-dir /tmp/manual_reconstruction_plan \
  --gpu 0 \
  --dry-run
```

## 复用已有 runtime-O

支持当前 v3 和冻结的 v2 runtime-O manifest；更早、缺少
`T_O2C_lifting` 的 v1 会 fail-closed。

```bash
bash manual_mesh_reconstruction/run_reconstruction.sh \
  --runtime-input-manifest /path/to/runtime_input_manifest.json \
  --object category:object_id \
  --output-dir /home/zjr/Tracker/manual_mesh_reconstruction/output/你的实验名 \
  --gpu 0
```

中断后给同一命令增加 `--resume`。每个模型在独立 Python 子进程中执行，
所以 current 与 ReconViaGen 之间不会保留上一模型的 GPU 状态。

## 单次输出结构

```text
00_dataset_adapter/                     # 使用 --dataset-path 时存在
  raw_cache/
  colmap_workspace/                     # rebuild 或二进制转文本时存在
01_runtime_o_pose_mask/
02_dino_only_model_input/
03_current_no_vggt_ss30k_slat30k/
04_strict_reconviagen/
05_current_original_camera_contours/
06_mesh_previews/current/
06_mesh_previews/reconviagen/
pipeline.log
run_manifest.json
```

`output/已有SS30K_SLat30K结果/` 保存从旧 `outputs2` 迁入的、最终 PASS 且未被
后续坐标修正版取代的人工测试结果；`MIGRATION_MANIFEST.json` 记录来源和哈希。
这些历史 JSON 不做内容改写，因此其中的旧绝对路径仍指向原 `outputs2`；迁移清单
提供新位置与逐目录 tree SHA256，实际图像和 Mesh 已完整复制到新目录。

## 代码分工

- `pipeline.py`：唯一端到端调度入口；
- `data_adapters/cli.py`：手机、COLMAP/Omni/CoarseModel、Objectron 的统一数据集入口；
- `data_adapters/phone.py`：冻结 ARFoundation pose 与保存图像轴合同；
- `data_adapters/colmap.py`：既有 COLMAP 复用或隔离式重建，并冻结注册帧身份；
- `data_adapters/objectron.py`：官方相机合同及 pose+mask/真值物体位姿两种 O；
- `runtime_o.py`、`pose_mask.py`、`canonicalization.py`：pose+mask 选帧及 runtime-O；
- `model_inputs.py`、`dino_condition.py`：no-VGGT DINO-only 条件；
- `current_model.py`：official SS30K + SLat30K 推理；
- `reconviagen.py`：strict original ReconViaGen 对照；
- `mesh_coordinates.py`：decoder 原生 sparse-grid/runtime-O 坐标合同；
- `projection.py`、`contours.py`：物理相机链投影和轮廓；
- `preview_backend.py`、`render_mesh.py`：两路 Mesh 的 display-only normal 预览；
- `defaults.py`：冻结模型/报告/bridge 的路径、SHA256 和绑定检查。
