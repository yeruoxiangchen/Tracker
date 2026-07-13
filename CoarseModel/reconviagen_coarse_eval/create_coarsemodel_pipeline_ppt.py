#!/usr/bin/env python3
from __future__ import annotations

import html
import zipfile
from datetime import datetime, timezone
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent / "汇报材料"
PPTX_PATH = OUT_DIR / "CoarseModel_AR重建Pipeline_5分钟汇报.pptx"
NOTES_PATH = OUT_DIR / "CoarseModel_AR重建Pipeline_5分钟讲稿.md"
GOODMESH_PREVIEW = Path("/home/zjr/Tracker/CoarseModel/datasets/GOOD_MESH_TEST/images/frame_0013.jpg")
GOODMESH_ALIGNED = Path("/home/zjr/Tracker/CoarseModel/results/refine_model/GOOD_MESH_TEST_原始/rigid_after/projected_frame_0013.jpg")
PPT_MEDIA = {
    4: [
        {"rid": "rId2", "target": "../media/goodmesh_preview.jpg", "name": "goodmesh_preview.jpg", "path": GOODMESH_PREVIEW},
        {"rid": "rId3", "target": "../media/goodmesh_aligned.jpg", "name": "goodmesh_aligned.jpg", "path": GOODMESH_ALIGNED},
    ]
}

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}

SLIDE_W = 12192000
SLIDE_H = 6858000


def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def emu(inch: float) -> int:
    return int(inch * 914400)


def text_shape(shape_id: int, x: float, y: float, w: float, h: float, paragraphs: list[dict]) -> str:
    p_xml = []
    for para in paragraphs:
        text = para.get("text", "")
        size = int(para.get("size", 24) * 100)
        color = para.get("color", "1F2937")
        bold = ' b="1"' if para.get("bold") else ""
        align = para.get("align", "l")
        p_xml.append(
            f"""
            <a:p>
              <a:pPr algn="{align}"/>
              <a:r>
                <a:rPr lang="zh-CN" sz="{size}"{bold}>
                  <a:solidFill><a:srgbClr val="{color}"/></a:solidFill>
                  <a:latin typeface="Microsoft YaHei"/>
                  <a:ea typeface="Microsoft YaHei"/>
                </a:rPr>
                <a:t>{esc(text)}</a:t>
              </a:r>
              <a:endParaRPr lang="zh-CN" sz="{size}">
                <a:solidFill><a:srgbClr val="{color}"/></a:solidFill>
                <a:latin typeface="Microsoft YaHei"/>
                <a:ea typeface="Microsoft YaHei"/>
              </a:endParaRPr>
            </a:p>
            """
        )
    return f"""
    <p:sp>
      <p:nvSpPr>
        <p:cNvPr id="{shape_id}" name="TextBox {shape_id}"/>
        <p:cNvSpPr txBox="1"/>
        <p:nvPr/>
      </p:nvSpPr>
      <p:spPr>
        <a:xfrm>
          <a:off x="{emu(x)}" y="{emu(y)}"/>
          <a:ext cx="{emu(w)}" cy="{emu(h)}"/>
        </a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        <a:noFill/>
        <a:ln><a:noFill/></a:ln>
      </p:spPr>
      <p:txBody>
        <a:bodyPr wrap="square" anchor="t"/>
        <a:lstStyle/>
        {''.join(p_xml)}
      </p:txBody>
    </p:sp>
    """


def box_shape(shape_id: int, x: float, y: float, w: float, h: float, text: str, fill: str, line: str = "D1D5DB", font: int = 17) -> str:
    return f"""
    <p:sp>
      <p:nvSpPr>
        <p:cNvPr id="{shape_id}" name="Box {shape_id}"/>
        <p:cNvSpPr/>
        <p:nvPr/>
      </p:nvSpPr>
      <p:spPr>
        <a:xfrm>
          <a:off x="{emu(x)}" y="{emu(y)}"/>
          <a:ext cx="{emu(w)}" cy="{emu(h)}"/>
        </a:xfrm>
        <a:prstGeom prst="roundRect"><a:avLst/></a:prstGeom>
        <a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>
        <a:ln w="12700"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>
      </p:spPr>
      <p:txBody>
        <a:bodyPr wrap="square" anchor="ctr"/>
        <a:lstStyle/>
        <a:p>
          <a:pPr algn="ctr"/>
          <a:r>
            <a:rPr lang="zh-CN" sz="{font * 100}" b="1">
              <a:solidFill><a:srgbClr val="111827"/></a:solidFill>
              <a:latin typeface="Microsoft YaHei"/>
              <a:ea typeface="Microsoft YaHei"/>
            </a:rPr>
            <a:t>{esc(text)}</a:t>
          </a:r>
        </a:p>
      </p:txBody>
    </p:sp>
    """


def image_shape(shape_id: int, rel_id: str, x: float, y: float, w: float, h: float, name: str) -> str:
    return f"""
    <p:pic>
      <p:nvPicPr>
        <p:cNvPr id="{shape_id}" name="{esc(name)}"/>
        <p:cNvPicPr/>
        <p:nvPr/>
      </p:nvPicPr>
      <p:blipFill>
        <a:blip r:embed="{rel_id}"/>
        <a:stretch><a:fillRect/></a:stretch>
      </p:blipFill>
      <p:spPr>
        <a:xfrm>
          <a:off x="{emu(x)}" y="{emu(y)}"/>
          <a:ext cx="{emu(w)}" cy="{emu(h)}"/>
        </a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        <a:ln w="12700"><a:solidFill><a:srgbClr val="CBD5E1"/></a:solidFill></a:ln>
      </p:spPr>
    </p:pic>
    """


def footer(slide_no: int) -> str:
    return text_shape(
        900 + slide_no,
        0.55,
        7.05,
        12.1,
        0.22,
        [{"text": f"CoarseModel / ReconViaGen / TRELLIS point-prior pipeline    {slide_no}/8", "size": 8, "color": "6B7280"}],
    )


def slide_xml(slide_no: int, title: str, subtitle: str | None, body_shapes: list[str]) -> str:
    title_shapes = [
        text_shape(10, 0.55, 0.28, 12.2, 0.55, [{"text": title, "size": 25, "bold": True, "color": "111827"}])
    ]
    if subtitle:
        title_shapes.append(text_shape(11, 0.58, 0.82, 12.0, 0.32, [{"text": subtitle, "size": 10, "color": "6B7280"}]))
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="{NS['a']}" xmlns:r="{NS['r']}" xmlns:p="{NS['p']}">
  <p:cSld>
    <p:bg><p:bgPr><a:solidFill><a:srgbClr val="F8FAFC"/></a:solidFill></p:bgPr></p:bg>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm><a:off x="0" y="0"/><a:ext cx="{SLIDE_W}" cy="{SLIDE_H}"/><a:chOff x="0" y="0"/><a:chExt cx="{SLIDE_W}" cy="{SLIDE_H}"/></a:xfrm>
      </p:grpSpPr>
      {''.join(title_shapes)}
      {''.join(body_shapes)}
      {footer(slide_no)}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>
"""


def bullets(items: list[str], x: float, y: float, w: float, h: float, size: int = 18, color: str = "1F2937") -> str:
    return text_shape(100 + int(y * 100), x, y, w, h, [{"text": f"• {item}", "size": size, "color": color} for item in items])


def make_slides() -> list[str]:
    slides: list[str] = []
    slides.append(
        slide_xml(
            1,
            "基于 AR 采集的 CoarseModel 三维重建 Pipeline",
            "5 分钟汇报：背景、当前方案、正在验证的 pose / SLAM point-prior 方向",
            [
                text_shape(20, 0.9, 1.65, 11.5, 0.7, [{"text": "目标：从手机多视图 RGB + mask + AR pose 得到可用于世界坐标对齐和后续优化的 Mesh", "size": 20, "bold": True, "color": "0F172A"}]),
                box_shape(21, 1.0, 3.0, 2.45, 0.8, "手机 AR 采集", "DBEAFE", "60A5FA"),
                box_shape(22, 3.75, 3.0, 2.45, 0.8, "ReconViaGen 生成 Mesh", "DCFCE7", "4ADE80"),
                box_shape(23, 6.5, 3.0, 2.45, 0.8, "CoarseModel 对齐 / 位姿", "FEF3C7", "FBBF24"),
                box_shape(24, 9.25, 3.0, 2.45, 0.8, "Pose / SLAM prior 增强", "EDE9FE", "A78BFA"),
                text_shape(25, 1.08, 4.0, 10.8, 0.45, [{"text": "讲述重点：当前不是替换 CoarseModel，而是把输入质量、Mesh 初值和 pose/point prior 串成可验证流程。", "size": 14, "color": "374151"}]),
            ],
        )
    )
    slides.append(
        slide_xml(
            2,
            "背景：为什么只靠图像生成 Mesh 不稳定",
            "真实 AR 采集和合成训练数据之间存在明显分布差异",
            [
                bullets(
                    [
                        "ReconViaGen/TRELLIS 对输入帧覆盖、遮挡、模糊、mask 边界和主体尺度非常敏感。",
                        "同一物体、不同采集序列可能得到差异很大的 Mesh；GOOD_MESH_TEST 与 reconviagen_2026 批次已经表现出这种差异。",
                        "手机 AR pose 是稳定的几何信息，但原版生成模型并不会直接利用它约束 Mesh。",
                        "CoarseModel 后续能做 TM2W / pose-scale 优化，但前提是输入 Mesh 不能太差。",
                    ],
                    0.85,
                    1.55,
                    11.6,
                    3.4,
                    18,
                ),
                box_shape(40, 1.1, 5.35, 3.1, 0.72, "问题 1：输入帧质量", "FEE2E2", "FCA5A5"),
                box_shape(41, 5.05, 5.35, 3.1, 0.72, "问题 2：生成 Mesh 初值", "FFEDD5", "FDBA74"),
                box_shape(42, 9.0, 5.35, 3.1, 0.72, "问题 3：pose 没进入生成", "E0E7FF", "818CF8"),
            ],
        )
    )
    slides.append(
        slide_xml(
            3,
            "CoarseModel 的角色：不是从零生成，而是把粗 Mesh 放进世界坐标",
            "当前重点是验证“可用粗 Mesh + AR pose”能否支撑后续位姿/尺度估计",
            [
                bullets(
                    [
                        "输入：分割后的多视图图像、相机位姿、ReconViaGen 生成的粗 Mesh。",
                        "输出：Mesh 到世界坐标的变换 TM2W，以及后续几何/边缘优化结果。",
                        "现实观察：如果 ReconViaGen Mesh 已经大体正确，CoarseModel 位姿部分可以得到可用结果。",
                        "当前不优先把 CoarseModel 几何优化做复杂；先保证 Mesh 初值和位姿对齐链路稳定。",
                    ],
                    0.85,
                    1.45,
                    11.6,
                    3.3,
                    18,
                ),
                box_shape(60, 1.0, 5.15, 2.5, 0.72, "粗 Mesh", "DCFCE7", "4ADE80"),
                text_shape(61, 3.65, 5.27, 0.5, 0.3, [{"text": "→", "size": 22, "bold": True}]),
                box_shape(62, 4.3, 5.15, 2.5, 0.72, "TM2W 对齐", "FEF3C7", "F59E0B"),
                text_shape(63, 6.95, 5.27, 0.5, 0.3, [{"text": "→", "size": 22, "bold": True}]),
                box_shape(64, 7.6, 5.15, 2.5, 0.72, "位姿 / 尺度优化", "DBEAFE", "60A5FA"),
                text_shape(65, 10.25, 5.27, 0.5, 0.3, [{"text": "→", "size": 22, "bold": True}]),
                box_shape(66, 10.9, 5.15, 1.5, 0.72, "可用结果", "EDE9FE", "A78BFA", 15),
            ],
        )
    )
    slides.append(
        slide_xml(
            4,
            "当前在线 Pipeline",
            "Unity 前端采集，server.py 完成分割、输入筛选、重建和 CoarseModel 调用",
            [
                box_shape(80, 0.75, 1.65, 1.75, 0.74, "Unity ARPoseTracker\nRGB + pose", "DBEAFE", "60A5FA", 14),
                box_shape(81, 2.85, 1.65, 1.75, 0.74, "server.py\n保存 session", "E0F2FE", "38BDF8", 14),
                box_shape(82, 4.95, 1.65, 1.75, 0.74, "SAM2\n用户监督 mask", "DCFCE7", "4ADE80", 14),
                box_shape(83, 7.05, 1.65, 1.75, 0.74, "输入 QC\n覆盖度/帧筛选", "FEF3C7", "F59E0B", 14),
                box_shape(84, 9.15, 1.65, 1.75, 0.74, "ReconViaGen\n生成粗 Mesh", "FCE7F3", "F472B6", 14),
                box_shape(85, 11.25, 1.65, 1.2, 0.74, "CoarseModel", "EDE9FE", "A78BFA", 13),
                text_shape(90, 0.95, 3.05, 11.2, 2.6, [{"text": "当前已经加入的工程侧重点：", "size": 18, "bold": True}] + [{"text": f"• {t}", "size": 16} for t in [
                    "前端采集多帧 RGB 与 AR pose，服务端写成 CoarseModel 数据集格式。",
                    "分割后先做输入覆盖/视角检查，避免明显不足的序列直接进入生成。",
                    "ReconViaGen 结果先单独评估，再进入 CoarseModel 位姿对齐。",
                    "新增真实/SLAM-like point prior 实验目录，不直接污染现有 CoarseModel 主代码。",
                ]]),
            ],
        )
    )
    slides.append(
        slide_xml(
            5,
            "当前发现：输入筛选比盲目改模型更先影响结果",
            "真实采集中，帧覆盖和 mask 质量直接决定 ReconViaGen Mesh 是否可用",
            [
                bullets(
                    [
                        "GOOD_MESH_TEST 这类环绕覆盖较好的序列，ReconViaGen 生成的粗 Mesh 已经能进入 CoarseModel 使用。",
                        "一些 reconviagen_2026 序列外观看起来相似，但 Mesh 差异很大，说明模型存在隐藏输入要求。",
                        "有效处理方向：覆盖均匀、主体尺度稳定、mask 干净、减少重复/模糊/极端视角。",
                        "当前 pipeline 更像“先让输入接近模型喜欢的分布，再利用 pose/point prior 修正”。",
                    ],
                    0.85,
                    1.55,
                    11.5,
                    3.25,
                    18,
                ),
                box_shape(100, 1.0, 5.25, 2.6, 0.75, "帧覆盖", "DBEAFE", "60A5FA"),
                box_shape(101, 3.9, 5.25, 2.6, 0.75, "mask 边界", "DCFCE7", "4ADE80"),
                box_shape(102, 6.8, 5.25, 2.6, 0.75, "主体尺度", "FEF3C7", "F59E0B"),
                box_shape(103, 9.7, 5.25, 2.6, 0.75, "视角均匀性", "EDE9FE", "A78BFA"),
            ],
        )
    )
    slides.append(
        slide_xml(
            6,
            "Pose / SLAM prior：不再简单把 pose 塞进模型，而是生成可观测几何先验",
            "当前更合理的方向：AR/SLAM sparse points + mask 投影过滤 -> TRELLIS sparse inpainting",
            [
                box_shape(120, 0.95, 1.75, 2.05, 0.78, "AR / COLMAP pose", "DBEAFE", "60A5FA", 14),
                box_shape(121, 3.25, 1.75, 2.05, 0.78, "SIFT / SLAM-like\n三角化点云", "E0F2FE", "38BDF8", 14),
                box_shape(122, 5.55, 1.75, 2.05, 0.78, "mask + pose\n筛物体点", "DCFCE7", "4ADE80", 14),
                box_shape(123, 7.85, 1.75, 2.05, 0.78, "point prior\n64³ sparse coords", "FEF3C7", "F59E0B", 14),
                box_shape(124, 10.15, 1.75, 2.05, 0.78, "Stage2 sparse\ninpainting", "EDE9FE", "A78BFA", 14),
                text_shape(130, 0.9, 3.15, 11.7, 2.55, [{"text": "关键转变：", "size": 18, "bold": True}] + [{"text": f"• {t}", "size": 16} for t in [
                    "pose head / reranker 对当前 CoarseModel 主流程帮助有限，因为输入 pose 是指定的。",
                    "更有价值的是利用 pose 产生可观测点云先验，让 sparse flow 在已观测区域更稳。",
                    "离线 SLAM-like 点云已经能在 GOOD_MESH_TEST 得到较多点；弱纹理序列点数明显下降。",
                ]]),
            ],
        )
    )
    slides.append(
        slide_xml(
            7,
            "正在训练 / 验证的模型方向",
            "训练部分只保留结论：point-prior sparse flow 有局部收益，但还需要端到端 Mesh 评估",
            [
                bullets(
                    [
                        "当前训练重点：TRELLIS sparse flow 的 point-prior / masked inpainting Stage2。",
                        "已验证：point-prior 能提升 observed/known 区域的 sparse ranking，但会带来过填充、尺度和 downstream slat 不匹配问题。",
                        "当前较稳配置：anti-overfill + weak hard-negative ranking + late clamp + relative top-k mesh eval。",
                        "下一步不是立即大规模训练 slat flow，而是先用真实 SLAM-like prior 做 end-to-end mesh smoke 与 baseline 对比。",
                    ],
                    0.85,
                    1.45,
                    11.5,
                    3.9,
                    18,
                ),
                box_shape(140, 1.3, 5.55, 2.4, 0.62, "stock_sparse", "F3F4F6", "9CA3AF", 14),
                box_shape(141, 4.25, 5.55, 2.4, 0.62, "prior_sparse", "FEF3C7", "F59E0B", 14),
                box_shape(142, 7.2, 5.55, 2.4, 0.62, "stage2_correct", "EDE9FE", "A78BFA", 14),
                text_shape(143, 10.0, 5.62, 2.1, 0.45, [{"text": "三者必须一起评估", "size": 14, "bold": True, "color": "374151"}]),
            ],
        )
    )
    slides.append(
        slide_xml(
            8,
            "下一步计划",
            "目标是把“真实输入可用性”和“模型改进收益”分开验证",
            [
                bullets(
                    [
                        "1. 跑真实 SLAM-like prior mesh smoke：stock_sparse / prior_sparse / stage2_correct 同场比较。",
                        "2. 做 strict-mask vs relaxed SLAM 点云 ablation，判断背景点是否污染 prior。",
                        "3. 把 AR 前端扩展到上传原生 ARPointCloud，替代离线 SIFT 三角化。",
                        "4. 如果 sparse->mesh 端到端确实有收益，再考虑训练/适配 slat flow；否则优先完善输入筛选和候选 rerank。",
                    ],
                    0.85,
                    1.4,
                    11.6,
                    4.1,
                    18,
                ),
                text_shape(160, 0.9, 5.85, 11.6, 0.45, [{"text": "一句话总结：当前 CoarseModel 主线先保证“好输入 + 可用粗 Mesh + 可对齐 pose”，point-prior 是利用 pose 提升 Mesh 初值的增量路线。", "size": 16, "bold": True, "color": "0F172A"}]),
            ],
        )
    )
    return slides


def make_updated_slides() -> list[str]:
    slides: list[str] = []
    slides.append(
        slide_xml(
            1,
            "AR 系统下的物体级三维重建 Pipeline",
            "5 分钟汇报：为什么需要 Mesh、CoarseModel 怎么接入、以及当前 Mesh 生成模型改进方向",
            [
                text_shape(20, 0.85, 1.45, 11.7, 0.78, [{"text": "核心问题：AR 系统能稳定追踪相机和空间，但物体级任务最终需要一个可对齐、可渲染、可优化的 Mesh。", "size": 20, "bold": True, "color": "0F172A"}]),
                box_shape(21, 0.95, 2.9, 2.25, 0.78, "AR 采集\nRGB / mask / pose", "DBEAFE", "60A5FA", 14),
                box_shape(22, 3.55, 2.9, 2.25, 0.78, "Mesh 初值\n生成模型", "DCFCE7", "4ADE80", 14),
                box_shape(23, 6.15, 2.9, 2.25, 0.78, "CoarseModel\n世界坐标对齐", "FEF3C7", "F59E0B", 14),
                box_shape(24, 8.75, 2.9, 2.25, 0.78, "pose / SLAM prior\n提升 Mesh 精度", "EDE9FE", "A78BFA", 14),
                text_shape(25, 1.0, 4.25, 11.1, 0.7, [{"text": "叙事顺序：AR 三维重建背景 → 我们的 AR pipeline → CoarseModel → 接 Mesh 生成模型 → Points-to-3D 启发下的修改。", "size": 15, "color": "374151"}]),
            ],
        )
    )
    slides.append(
        slide_xml(
            2,
            "背景：AR 系统为什么还缺真正的 Mesh",
            "从姿态感知到 SLAM/深度融合，AR 已经能理解空间，但物体级几何仍然不足",
            [
                bullets(
                    [
                        "早期 AR 更偏 VIO / 水平仪式姿态估计：能知道手机怎么动，但没有稳定物体模型。",
                        "后续 SLAM 和深度融合可以得到相机轨迹、平面、稀疏点云或局部深度，但通常不是干净的物体 Mesh。",
                        "很多真实任务需要可碰撞、可渲染、可编辑、可复用的物体级 Mesh，而不只是相机轨迹或点云。",
                        "因此我们要把 AR 采集到的 RGB、mask、pose 转成一个能放回世界坐标的 Mesh。",
                    ],
                    0.85,
                    1.55,
                    11.6,
                    3.4,
                    18,
                ),
                box_shape(40, 0.9, 5.25, 2.15, 0.72, "VIO / 姿态", "DBEAFE", "60A5FA"),
                box_shape(41, 3.35, 5.25, 2.15, 0.72, "SLAM 轨迹", "E0F2FE", "38BDF8"),
                box_shape(42, 5.8, 5.25, 2.15, 0.72, "深度 / 点云", "FEF3C7", "F59E0B"),
                box_shape(43, 8.25, 5.25, 2.15, 0.72, "物体 Mesh", "DCFCE7", "4ADE80"),
                box_shape(44, 10.7, 5.25, 1.65, 0.72, "可用资产", "EDE9FE", "A78BFA", 14),
            ],
        )
    )
    slides.append(
        slide_xml(
            3,
            "我们实现的 AR 物体重建 Pipeline",
            "先把 AR 系统数据整理成可计算输入，再由 Mesh 生成模型和 CoarseModel 分工处理",
            [
                box_shape(60, 0.75, 1.75, 1.65, 0.78, "手机 AR\nRGB + pose", "DBEAFE", "60A5FA", 14),
                box_shape(61, 2.75, 1.75, 1.65, 0.78, "SAM2\nmask", "DCFCE7", "4ADE80", 14),
                box_shape(62, 4.75, 1.75, 1.65, 0.78, "输入 QC\n覆盖筛选", "FEF3C7", "F59E0B", 14),
                box_shape(63, 6.75, 1.75, 1.65, 0.78, "Mesh 生成\nReconViaGen", "FCE7F3", "F472B6", 14),
                box_shape(64, 8.75, 1.75, 1.65, 0.78, "CoarseModel\nTM2W", "EDE9FE", "A78BFA", 14),
                box_shape(65, 10.75, 1.75, 1.65, 0.78, "世界坐标\n可用 Mesh", "E0F2FE", "38BDF8", 14),
                text_shape(70, 0.85, 3.15, 11.7, 2.55, [{"text": "当前工程主线：", "size": 18, "bold": True}] + [{"text": f"• {t}", "size": 16} for t in [
                    "server.py 把前端采集的原图、位姿和分割结果整理成 CoarseModel 数据集。",
                    "ReconViaGen/TRELLIS 负责从多视图图像生成物体粗 Mesh。",
                    "CoarseModel 负责把粗 Mesh 对齐到 AR 世界坐标，并输出 TM2W / scale / refine 结果。",
                    "新增实验用于验证 pose/SLAM point prior 是否能改善 Mesh 初值，不直接污染主流程。",
                ]]),
            ],
        )
    )
    slides.append(
        slide_xml(
            4,
            "CoarseModel 思路：给定粗 Mesh，估计 Mesh 到世界坐标的变换",
            "CoarseModel 不是生成器，而是利用多视图、mask 和相机位姿做对齐与优化",
            [
                text_shape(80, 0.75, 1.28, 4.6, 1.15, [{"text": "输入", "size": 18, "bold": True}] + [{"text": f"• {t}", "size": 14} for t in [
                    "粗 Mesh / model_norm.obj",
                    "多视图 RGB 与 mask",
                    "COLMAP / AR camera pose",
                    "相机内参 K",
                ]]),
                text_shape(81, 0.75, 3.15, 4.6, 1.0, [{"text": "输出", "size": 18, "bold": True}] + [{"text": f"• {t}", "size": 14} for t in [
                    "T_M2W：Mesh 到世界坐标",
                    "final_scale：尺度估计",
                    "投影可视化和诊断图",
                    "可选 deformation/refined_model.obj",
                ]]),
                box_shape(82, 0.85, 5.0, 1.95, 0.58, "Stage 1\n每帧数据", "DBEAFE", "60A5FA", 11),
                box_shape(83, 3.05, 5.0, 1.95, 0.58, "Stage 2\npose + scale", "FEF3C7", "F59E0B", 11),
                box_shape(84, 5.25, 5.0, 1.95, 0.58, "Stage 3\n变形优化", "DCFCE7", "4ADE80", 11),
                box_shape(85, 7.45, 5.0, 1.95, 0.58, "Stage 4\n导出可视化", "EDE9FE", "A78BFA", 11),
                image_shape(86, "rId2", 5.65, 1.33, 3.05, 2.28, "GOOD_MESH_TEST 原始输入"),
                image_shape(87, "rId3", 9.25, 1.33, 3.05, 2.28, "GOOD_MESH_TEST CoarseModel 对齐"),
                text_shape(88, 5.65, 3.72, 3.05, 0.28, [{"text": "输入 preview", "size": 11, "bold": True, "align": "ctr", "color": "334155"}]),
                text_shape(89, 9.25, 3.72, 3.05, 0.28, [{"text": "CoarseModel 投影对齐", "size": 11, "bold": True, "align": "ctr", "color": "334155"}]),
                text_shape(90, 0.85, 5.85, 11.4, 0.5, [{"text": "例子：即使 ReconViaGen 给的是粗 Mesh，只要整体形状可用，CoarseModel 仍能估计出基本正确的世界坐标对齐。", "size": 14, "bold": True, "color": "374151"}]),
            ],
        )
    )
    slides.append(
        slide_xml(
            5,
            "为什么要接 Mesh 生成模型",
            "CoarseModel 需要一个粗 Mesh；AR/SLAM 自身通常只提供点云、深度或局部几何",
            [
                bullets(
                    [
                        "AR 系统能提供相机位姿和空间点，但这些点稀疏、噪声大，且通常混有背景。",
                        "传统 SLAM / 深度融合可以重建部分表面，但难以直接得到干净、封闭、可用的物体 Mesh。",
                        "ReconViaGen/TRELLIS 可以从多视图图像生成粗 Mesh，补足不可见区域和整体形状。",
                        "因此当前方案是：生成模型给 Mesh 初值，CoarseModel 用 AR pose 把它放回真实世界。",
                    ],
                    0.85,
                    1.45,
                    11.5,
                    3.4,
                    18,
                ),
                box_shape(100, 0.9, 5.15, 2.5, 0.75, "SLAM 点云\n真实但稀疏", "E0F2FE", "38BDF8", 13),
                box_shape(101, 3.65, 5.15, 2.5, 0.75, "生成 Mesh\n完整但可能偏", "DCFCE7", "4ADE80", 13),
                box_shape(102, 6.4, 5.15, 2.5, 0.75, "CoarseModel\n对齐真实世界", "FEF3C7", "F59E0B", 13),
                box_shape(103, 9.15, 5.15, 2.5, 0.75, "pose/point prior\n提升初值", "EDE9FE", "A78BFA", 13),
            ],
        )
    )
    slides.append(
        slide_xml(
            6,
            "当前进展：ReconViaGen 对输入图像覆盖度要求较高",
            "覆盖好时 Mesh 可以进入 CoarseModel；覆盖不足或 mask/尺度不稳时，结果差异明显",
            [
                bullets(
                    [
                        "GOOD_MESH_TEST 这种视角覆盖较好的序列，生成 Mesh 虽然不是完美，但已经能支撑 CoarseModel 对齐。",
                        "一些 reconviagen_2026 序列看起来类似，但 Mesh 质量差异很大，说明模型对覆盖、尺度、mask 和模糊较敏感。",
                        "因此我们先做输入帧筛选和覆盖度检查，避免明显不合格序列直接进入生成。",
                        "进一步考虑利用 AR 系统已有 pose，或 pose + SLAM 点云，让生成模型显式看到真实可观测几何。",
                    ],
                    0.85,
                    1.45,
                    11.5,
                    3.55,
                    18,
                ),
                box_shape(120, 1.0, 5.35, 2.6, 0.72, "覆盖均匀", "DBEAFE", "60A5FA"),
                box_shape(121, 3.9, 5.35, 2.6, 0.72, "主体尺度稳定", "DCFCE7", "4ADE80"),
                box_shape(122, 6.8, 5.35, 2.6, 0.72, "mask 干净", "FEF3C7", "F59E0B"),
                box_shape(123, 9.7, 5.35, 2.6, 0.72, "pose / 点云增强", "EDE9FE", "A78BFA"),
            ],
        )
    )
    slides.append(
        slide_xml(
            7,
            "模型改进方向：借鉴 Points-to-3D，但输入换成 AR/SLAM prior",
            "论文：Points-to-3D: Structure-Aware 3D Generation with Point Cloud Priors, arXiv:2603.18782",
            [
                box_shape(140, 0.75, 1.55, 2.1, 0.75, "Point cloud prior", "DBEAFE", "60A5FA", 13),
                box_shape(141, 3.15, 1.55, 2.1, 0.75, "TRELLIS sparse\nstructure", "E0F2FE", "38BDF8", 13),
                box_shape(142, 5.55, 1.55, 2.1, 0.75, "Structure\ninpainting", "FEF3C7", "F59E0B", 13),
                box_shape(143, 7.95, 1.55, 2.1, 0.75, "SLAT / Mesh\ndecoding", "DCFCE7", "4ADE80", 13),
                box_shape(144, 10.35, 1.55, 2.1, 0.75, "Complete 3D\nasset", "EDE9FE", "A78BFA", 13),
                text_shape(145, 2.88, 1.75, 0.22, 0.22, [{"text": "→", "size": 18, "bold": True, "align": "ctr"}]),
                text_shape(146, 5.28, 1.75, 0.22, 0.22, [{"text": "→", "size": 18, "bold": True, "align": "ctr"}]),
                text_shape(147, 7.68, 1.75, 0.22, 0.22, [{"text": "→", "size": 18, "bold": True, "align": "ctr"}]),
                text_shape(148, 10.08, 1.75, 0.22, 0.22, [{"text": "→", "size": 18, "bold": True, "align": "ctr"}]),
                text_shape(149, 0.9, 3.05, 5.65, 2.35, [{"text": "论文给我们的启发", "size": 18, "bold": True}] + [{"text": f"• {t}", "size": 15} for t in [
                    "点云先验可以约束 TRELLIS sparse structure。",
                    "保留已观测区域，再由模型补全不可见结构。",
                    "比单纯把 pose 编成特征更直接地影响几何。",
                ]]),
                text_shape(150, 6.85, 3.05, 5.65, 2.35, [{"text": "我们的宏观修改", "size": 18, "bold": True}] + [{"text": f"• {t}", "size": 15} for t in [
                    "点云来源换成 AR/SLAM-like prior。",
                    "先验证 sparse flow 是否能利用真实观测几何。",
                    "再看端到端 Mesh 是否优于原始生成结果。",
                ]]),
            ],
        )
    )
    slides.append(
        slide_xml(
            8,
            "下一步计划",
            "目标：让 AR 系统得到真正可用、可对齐、可优化的物体 Mesh",
            [
                bullets(
                    [
                        "1. 先跑真实 SLAM-like prior mesh smoke：stock_sparse / prior_sparse / stage2_correct 同场比较。",
                        "2. 补 strict-mask vs relaxed SLAM 点云 ablation，判断背景点是否污染 prior。",
                        "3. 让 Unity 前端上传原生 ARPointCloud，替代离线 SIFT 三角化，更贴近真实系统输入。",
                        "4. 如果 sparse->mesh 端到端收益稳定，再考虑 slat flow 适配；否则优先完善输入筛选和候选 rerank。",
                    ],
                    0.85,
                    1.4,
                    11.6,
                    4.1,
                    18,
                ),
                text_shape(160, 0.9, 5.85, 11.6, 0.45, [{"text": "一句话总结：CoarseModel 负责把 Mesh 放进 AR 世界；ReconViaGen/TRELLIS 负责给 Mesh 初值；pose/SLAM prior 的价值是让这个初值更贴近真实观测。", "size": 15, "bold": True, "color": "0F172A"}]),
            ],
        )
    )
    return slides


def content_types(num_slides: int) -> str:
    slide_overrides = "\n".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, num_slides + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="jpg" ContentType="image/jpeg"/>
  <Default Extension="jpeg" ContentType="image/jpeg"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  {slide_overrides}
</Types>
"""


def presentation_xml(num_slides: int) -> str:
    ids = "\n".join(f'<p:sldId id="{255 + i}" r:id="rId{i + 1}"/>' for i in range(1, num_slides + 1))
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="{NS['a']}" xmlns:r="{NS['r']}" xmlns:p="{NS['p']}">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst>{ids}</p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle/>
</p:presentation>
"""


def presentation_rels(num_slides: int) -> str:
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
    ]
    for i in range(1, num_slides + 1):
        rels.append(
            f'<Relationship Id="rId{i + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {''.join(rels)}
</Relationships>
"""


def rels_root() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""


def slide_master() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="{NS['a']}" xmlns:r="{NS['r']}" xmlns:p="{NS['p']}">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{SLIDE_W}" cy="{SLIDE_H}"/><a:chOff x="0" y="0"/><a:chExt cx="{SLIDE_W}" cy="{SLIDE_H}"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="1" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>
"""


def slide_layout() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="{NS['a']}" xmlns:r="{NS['r']}" xmlns:p="{NS['p']}" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{SLIDE_W}" cy="{SLIDE_H}"/><a:chOff x="0" y="0"/><a:chExt cx="{SLIDE_W}" cy="{SLIDE_H}"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>
"""


def simple_theme() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="CoarseModel Theme">
  <a:themeElements>
    <a:clrScheme name="Office">
      <a:dk1><a:srgbClr val="111827"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="1F2937"/></a:dk2><a:lt2><a:srgbClr val="F8FAFC"/></a:lt2>
      <a:accent1><a:srgbClr val="2563EB"/></a:accent1><a:accent2><a:srgbClr val="16A34A"/></a:accent2>
      <a:accent3><a:srgbClr val="F59E0B"/></a:accent3><a:accent4><a:srgbClr val="7C3AED"/></a:accent4>
      <a:accent5><a:srgbClr val="DB2777"/></a:accent5><a:accent6><a:srgbClr val="0891B2"/></a:accent6>
      <a:hlink><a:srgbClr val="2563EB"/></a:hlink><a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="Microsoft YaHei"><a:majorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/></a:majorFont><a:minorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="Default"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
</a:theme>
"""


def core_props() -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>CoarseModel AR Reconstruction Pipeline</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
"""


def app_props(num_slides: int) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex PPTX Generator</Application>
  <PresentationFormat>On-screen Show (16:9)</PresentationFormat>
  <Slides>{num_slides}</Slides>
  <Company></Company>
</Properties>
"""


def write_pptx(slides: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PPTX_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types(len(slides)))
        z.writestr("_rels/.rels", rels_root())
        z.writestr("docProps/core.xml", core_props())
        z.writestr("docProps/app.xml", app_props(len(slides)))
        z.writestr("ppt/presentation.xml", presentation_xml(len(slides)))
        z.writestr("ppt/_rels/presentation.xml.rels", presentation_rels(len(slides)))
        z.writestr("ppt/slideMasters/slideMaster1.xml", slide_master())
        z.writestr(
            "ppt/slideMasters/_rels/slideMaster1.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/></Relationships>""",
        )
        z.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout())
        z.writestr(
            "ppt/slideLayouts/_rels/slideLayout1.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/></Relationships>""",
        )
        z.writestr("ppt/theme/theme1.xml", simple_theme())
        written_media = set()
        for media_items in PPT_MEDIA.values():
            for media in media_items:
                source = Path(media["path"])
                if not source.exists():
                    raise FileNotFoundError(source)
                archive_name = f"ppt/media/{media['name']}"
                if archive_name not in written_media:
                    z.write(source, archive_name)
                    written_media.add(archive_name)
        for idx, slide in enumerate(slides, start=1):
            z.writestr(f"ppt/slides/slide{idx}.xml", slide)
            media_rels = ""
            for media in PPT_MEDIA.get(idx, []):
                media_rels += (
                    f'<Relationship Id="{media["rid"]}" '
                    f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                    f'Target="{media["target"]}"/>'
                )
            z.writestr(
                f"ppt/slides/_rels/slide{idx}.xml.rels",
                f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>{media_rels}</Relationships>""",
            )


def write_notes() -> None:
    NOTES_PATH.write_text(
        """# CoarseModel AR 重建 Pipeline：5 分钟讲稿

## 1. 标题页
今天讲的是当前 AR 采集到三维重建的主线。目标不是单纯展示模型训练，而是说明为什么现在 pipeline 重点放在输入质量、ReconViaGen 粗 Mesh、CoarseModel 位姿对齐，以及 pose/SLAM point-prior 的增量利用。

## 2. 背景问题
真实 AR 采集和合成训练数据分布差异很大。ReconViaGen/TRELLIS 对输入视角覆盖、mask 边界、主体尺度和重复帧都比较敏感。同一个物体，在不同采集序列上 Mesh 质量差异明显。AR pose 是可靠几何信息，但原版生成模型没有直接利用它。

## 3. CoarseModel 的角色
CoarseModel 当前更像后端几何对齐模块：给它一个粗 Mesh，再结合多视图和相机，估计 Mesh 到世界坐标的 TM2W，并做简单优化。前提是粗 Mesh 不能太差。所以现阶段先保证 Mesh 初值可用，再谈几何优化。

## 4. 当前在线 pipeline
Unity 前端采集 RGB 和 AR pose，server.py 保存 session；用户通过 SAM2 监督分割；服务端做输入 QC；ReconViaGen 生成粗 Mesh；最后进入 CoarseModel。我们新增的实验都放在独立目录，不直接污染主流程。

## 5. 当前发现
输入筛选比盲目改模型更直接影响结果。GOOD_MESH_TEST 这类覆盖好的序列已经能得到可用 Mesh；部分 reconviagen_2026 序列看起来相似但结果差很多，说明模型对输入有隐藏要求。现在需要让输入更接近模型喜欢的分布。

## 6. Pose / SLAM prior
直接用 pose head 对当前主流程帮助有限，因为输入 pose 已经指定。更合理的是用 pose 生成几何先验：AR/COLMAP pose 加图像匹配或 SLAM 点云，经 mask 投影筛出物体点，再进入 TRELLIS sparse inpainting。

## 7. 当前训练方向
正在验证的是 TRELLIS sparse flow 的 point-prior / masked inpainting Stage2。结果显示已观测区域有提升，但还存在过填充、尺度和 downstream slat 不匹配问题。所以必须用 stock_sparse、prior_sparse、stage2_correct 三者一起做 end-to-end mesh eval。

## 8. 下一步
下一步先跑真实 SLAM-like prior mesh smoke，同时比较 baseline；补 strict-mask 和 relaxed SLAM 点云 ablation；之后再考虑让 Unity 前端上传原生 ARPointCloud。如果 sparse 到 mesh 端到端收益稳定，再投入 slat flow 训练。
""",
        encoding="utf-8",
    )


def write_updated_notes() -> None:
    NOTES_PATH.write_text(
        """# AR 系统下的物体级三维重建 Pipeline：5 分钟讲稿

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
""",
        encoding="utf-8",
    )


def main() -> None:
    slides = make_updated_slides()
    write_pptx(slides)
    write_updated_notes()
    print(PPTX_PATH)
    print(NOTES_PATH)


if __name__ == "__main__":
    main()
