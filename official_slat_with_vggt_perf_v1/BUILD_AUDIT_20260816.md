# Build audit — official with‑VGGT strict perf v1

## 状态

`READY FOR SOURCE 3090 CUDA RETEST`

不能写 CUDA PASS：本次 Codex 运行环境不可见 NVIDIA driver。下一门是8-object
sidecar、1GPU step0/2-update、2GPU DDP/resume、8GPU 100-step benchmark。

## 隔离边界

- 新文件只位于 `/home/zjr/Tracker/official_slat_with_vggt_perf_v1/`。
- 未修改 `pose_point_depth_mv/` 中任何既有文件。
- 未写入或覆盖任何 cache、checkpoint、report、manifest、decoder audit 或 freeze。
- 真实 Train2000 preflight 是只读模式，且确认指定 `/tmp/...preflight...` 输出目录
  没有被创建。

## 复用闭包

- strict trainer SHA256：
  `45978dd8a809e12245bdd9d831345f9cb7e5ea4526893e31f81cfe32c20664d5`
- strict projection SHA256：
  `349a4df4402997d0f5858ea967faec0e4b18a60a4f4f9889c27edbb899633987`
- with‑VGGT cache/model/science-wrapper SHA256：
  - `c9335be738bcfbc7e0876d0c34c898bde829dd74d085e395e6224b5e7e94826a`
  - `fea22e3f4f9ef35ab3254acc8619e70e5f19312f7e854fe68c3bfbb0e2bafb26`
  - `db9927351c90b81b2c90fbf275d974a5f1e2c2f8a83537599482a552a4749b55`

动态 strict 文件使用绝对 import 的关键 helper 也已逐文件锁 SHA；完整列表由：

```bash
python -m official_slat_with_vggt_perf_v1.preflight_runtime
```

结构化打印。

## 新代码 SHA256

```text
b97b7f95c6ca001d6c689af5577aba51e50c73e245b8ef8c93159ae638179ada  __init__.py
489a3ce07f47d18f2b838bfceb83ad41d15c2a32b0c4cb1ef54b35e2ea7e0bc1  dataset.py
d9f221f1f90cc1fea3e367020392afa2b86f7da020bd4e91c5e5f923d3d9c4cb  preflight_runtime.py
4bb8441a1d3c6ac77c0841b7218c782ea9a95e1032c3fb18a6900531dff340e2  runtime.py
f1211b83f0f6fc50459b7f94a4dbd5b28aec5ba70acf77275a9b7b19953e9a87  test_runtime.py
12c67018b70f3009c1ba3b0548b2a133f4b2e40d5cca8c860dce8c2c6f3a33cf  train.py
dcfda951fe277eebcde1669c856b9e1af292ebcf9e9e9ef9c54e7e95d2460c6c  train_proobjaverse_official.py
```

## 回归结果

1. 新旁路 + with‑VGGT + resume/deployment：38 tests，全部 PASS。
2. 原 strict-fix1：36 tests，35 PASS，1项真实CUDA对照因本环境无CUDA而SKIP；
   CPU DDP gate、2/4/8-view selection、RNG、finite、EMA、resume均PASS。
3. `py_compile`：PASS。
4. 正式入口 `--help` import：PASS。
5. trailing-whitespace/tokenize：PASS。

## 真实 Train2000 只读 preflight

- object_count：8
- protocol SHA：
  `36c2147c9d3d37b5dc867ff3d277a4af8f7ad9f2f5099edcc032719a0fe5c241`
- base SLat manifest SHA：
  `ead078ec423475ddbd2e4272e990404f74cddcc39b2332d42b4e2da266aab737`
- base lifting manifest SHA：
  `4a275be6d4b013a378d4aab325e48617cf1b5b7b77aeee30964558260be1d20b`
- exact_base_view_ids_only：true
- vggt_camera_consumed：false
- builder SHA：
  `bfda60739794995ce0777a7bd2eb1ceba176ae3c0e4049b98aaf76671d709991`

## 训练数学影响

新增代码只改变 host I/O/H2D/DDP 输入搬运策略；with‑VGGT 本身的唯一科学变量仍是
V0 native `slat_vggt_cond` 替换 N0 DINO-only Stock context。checkpoint/report 的
model summary 会记录 `strict_with_vggt_runtime`，但不新造第二套科学 identity。
