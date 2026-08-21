import os
os.environ['SPCONV_ALGO'] = 'native'
import torch
import numpy as np
import imageio
from PIL import Image
import argparse

# 导入 ReconViaGen (TRELLIS) 核心依赖
from trellis.pipelines import TrellisVGGTTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
from peft import LoraConfig, get_peft_model

# 1. 目录配置
DATA_DIR = "./data"
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. 生成超参数设定
PARAMS = {
    "seed": 0,
    "ss_sampling_steps": 30,
    "ss_guidance_strength": 7.5,
    "ss_guidance_rescale": 0.7,
    "ss_rescale_t": 5.0,
    "slat_sampling_steps": 12,
    "slat_guidance_strength": 7.5,
    "slat_guidance_rescale": 0.5,
    "slat_rescale_t": 3.0,
    "multiimage_algo": "multidiffusion",
    "mesh_simplify": 0.95,
    "texture_size": 1024,
    "low_vram": True
}

def load_local_images(pipeline, data_dir):
    """从本地读取图像并使用 pipeline 去除背景，返回 (图像, 文件名) 列表"""
    print(f"Reading images from {data_dir}...")
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    image_files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(valid_exts)])
    
    if not image_files:
        raise ValueError(f"No valid images found in {data_dir}. Please put some images there.")

    processed_images_with_names = []
    for img_name in image_files:
        img_path = os.path.join(data_dir, img_name)
        raw_img = Image.open(img_path).convert("RGBA")
        processed_img = pipeline.preprocess_image(raw_img)
        processed_images_with_names.append((processed_img, img_name))
        print(f"  Processed: {img_name}")
        
    return processed_images_with_names

def select_images_interactive(images_with_names):
    """弹出窗口可视化处理后的图像，并让用户在终端选择要保留的图像"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping visualization and keeping all images.")
        return [img for img, name in images_with_names]

    num_images = len(images_with_names)
    cols = min(4, num_images)
    rows = (num_images + cols - 1) // cols

    try:
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if num_images == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, ((img, name), ax) in enumerate(zip(images_with_names, axes)):
            # PIL RGBA 图像可以直接用 imshow 渲染透明度
            ax.imshow(img)
            ax.set_title(f"[{idx}] {name}")
            ax.axis('off')

        # 隐藏多余的子图空白区域
        for i in range(num_images, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show(block=False)  # 非阻塞显示，允许终端继续接收输入
        plt.pause(0.1)

        print("\n" + "="*50)
        print("🔍 请在弹出的窗口中预览图像 (如无弹窗请检查本地环境)")
        print("="*50)
        user_input = input("请输入你想【保留】的图像编号，用英文逗号分隔 (例如: 0,1,3)。\n直接按回车键 (Enter) 保留所有图像: ").strip()

        plt.close(fig) # 获取输入后关闭窗口

        if not user_input:
            print(">> 默认保留所有图像。")
            return [img for img, name in images_with_names]

        # 解析用户输入的编号
        selected_indices = [int(x.strip()) for x in user_input.split(',')]
        selected_images = [images_with_names[i][0] for i in selected_indices if 0 <= i < num_images]
        
        if not selected_images:
            print(">> 未选中任何有效图像，自动退回保留所有图像。")
            return [img for img, name in images_with_names]

        print(f">> 已成功选择 {len(selected_images)} 张图像进行后续推理。")
        return selected_images

    except Exception as e:
        print(f"Visualization failed (Headless server without X11?): {e}")
        print(">> Auto-selecting all images.")
        return [img for img, name in images_with_names]

def main(args):
    # ==========================
    # 步骤 A: 加载模型与 LoRA
    # ==========================
    print("Initializing Pipeline...")
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(args.pretrained)

    # 注入 SS LoRA
    if args.ss_ckpt is not None:
        print(f"Loading SS checkpoint from {args.ss_ckpt}")
        ss_lora_cfg = LoraConfig(
            r=64, lora_alpha=128, lora_dropout=0.0,
            target_modules=["to_q", "to_kv", "to_out", "to_qkv"]
        )
        ss_states = torch.load(args.ss_ckpt, map_location="cpu")["state_dict"]
        peft_ss = get_peft_model(pipeline.models['sparse_structure_flow_model'], ss_lora_cfg)
        peft_ss.load_state_dict({k.replace("ss_flow_model.", ""): v for k, v in ss_states.items()}, strict=False)
        pipeline.models['sparse_structure_flow_model'] = peft_ss.merge_and_unload()
        pipeline.sparse_structure_flow_model = pipeline.models['sparse_structure_flow_model']
        pipeline.models['sparse_structure_vggt_cond'].load_state_dict(
            {k.replace("ss_cond.", ""): v for k, v in ss_states.items()}, strict=False)

    # 注入 SLAT LoRA
    if args.slat_ckpt is not None:
        print(f"Loading SLAT checkpoint from {args.slat_ckpt}")
        slat_lora_cfg = LoraConfig(
            r=128, lora_alpha=256, lora_dropout=0.0,
            target_modules=["to_q", "to_kv", "to_out", "to_qkv"]
        )
        slat_states = torch.load(args.slat_ckpt, map_location="cpu")["state_dict"]
        peft_slat = get_peft_model(pipeline.models['slat_flow_model'], slat_lora_cfg)
        peft_slat.load_state_dict({k.replace("slat_flow_model.", ""): v for k, v in slat_states.items()}, strict=False)
        pipeline.models['slat_flow_model'] = peft_slat.merge_and_unload()
        pipeline.slat_flow_model = pipeline.models['slat_flow_model']
        pipeline.models['slat_vggt_cond'].load_state_dict(
            {k.replace("slat_cond.", ""): v for k, v in slat_states.items()}, strict=False)

    # VRAM 与设备配置
    pipeline._device = torch.device('cuda')
    pipeline.low_vram = PARAMS["low_vram"]
    pipeline.birefnet_model.cuda()
    
    if not pipeline.low_vram:
        for model in pipeline.models.values():
            model.to(pipeline._device)
        pipeline.VGGT_model.to(pipeline._device)

    # ==========================
    # 步骤 B: 图像输入、可视化筛选与推理
    # ==========================
    images_with_names = load_local_images(pipeline, DATA_DIR)
    
    # === 新增：交互式筛选步骤 ===
    images = select_images_interactive(images_with_names)
    
    print("\nStarting 3D Generation...")
    outputs, _, _ = pipeline.run(
        image=images,
        seed=PARAMS["seed"],
        formats=["gaussian", "mesh"],
        preprocess_image=False, 
        sparse_structure_sampler_params={
            "steps": PARAMS["ss_sampling_steps"],
            "cfg_strength": PARAMS["ss_guidance_strength"],
            "cfg_interval": [0.6, 1.0],
            "guidance_rescale": PARAMS["ss_guidance_rescale"],
            "rescale_t": PARAMS["ss_rescale_t"],
        },
        slat_sampler_params={
            "steps": PARAMS["slat_sampling_steps"],
            "cfg_strength": PARAMS["slat_guidance_strength"],
            "cfg_interval": [0.6, 1.0],
            "guidance_rescale": PARAMS["slat_guidance_rescale"],
            "rescale_t": PARAMS["slat_rescale_t"],
        },
        mode=PARAMS["multiimage_algo"],
    )

    gs = outputs['gaussian'][0]
    mesh = outputs['mesh'][0]
    torch.cuda.empty_cache()

    # ==========================
    # 步骤 C: 提取和导出文件
    # ==========================
    base_filename = os.path.join(OUTPUT_DIR, "reconstructed_object")

    print("Extracting and saving Gaussian Splatting (.ply)...")
    gs.save_ply(f"{base_filename}.ply")

    print("Extracting and saving Mesh (.glb)...")
    glb = postprocessing_utils.to_glb(gs, mesh, simplify=PARAMS["mesh_simplify"], texture_size=PARAMS["texture_size"], verbose=False)
    glb.export(f"{base_filename}.glb")
    del glb

    print("Rendering preview video (.mp4)...")
    video_color = render_utils.render_video(gs, num_frames=120)['color']
    video_geo = render_utils.render_video(mesh, num_frames=120)['normal']
    video = [np.concatenate([video_color[i], video_geo[i]], axis=1) for i in range(len(video_color))]
    imageio.mimsave(f"{base_filename}.mp4", video, fps=15)
    
    print(f"\n✅ All tasks finished! Results are saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--ss_ckpt", default=None, help="Path to SS checkpoint (.ckpt)")
    parser.add_argument("--slat_ckpt", default=None, help="Path to SLAT checkpoint (.ckpt)")
    args = parser.parse_args()
    
    main(args)