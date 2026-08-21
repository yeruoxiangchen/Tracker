import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

def load_birefnet(device="cuda"):
    """
    加载 BiRefNet 模型
    """
    print("Loading BiRefNet model...")
    model = AutoModelForImageSegmentation.from_pretrained(
        'ZhengPeng7/BiRefNet',
        trust_remote_code=True
    ).to(device)
    model.eval()
    return model

def get_birefnet_mask(image: Image.Image, model, device="cuda") -> np.ndarray:
    """
    输入 PIL 图像，返回二值化的 Mask (Numpy 数组格式, 0 或 1)
    """
    image = image.convert('RGB')
    
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_images = transform_image(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask_pil = pred_pil.resize(image.size)
    mask_np = np.array(mask_pil)

    # 返回二值化 mask，大于 128 的像素为前景(1)，否则为背景(0)
    return (mask_np > 128).astype(np.uint8)

def extract_foreground(image_path: str, output_mask_path: str, output_rgba_path: str, model, device="cuda"):
    """
    读取单张图像，提取 Mask 并保存
    """
    original_img = Image.open(image_path)
    
    # 获取 0 和 1 组成的 mask
    mask_array = get_birefnet_mask(original_img, model, device)
    
    # ---------------- 1. 保存黑白 Mask ----------------
    mask_image = Image.fromarray(mask_array * 255, mode='L')
    mask_image.save(output_mask_path)
    
    # ---------------- 2. 保存去背景的 RGBA 图像 ----------------
    input_rgba = original_img.convert('RGBA')
    input_array = np.array(input_rgba)
    input_array[:, :, 3] = mask_array * 255
    foreground_img = Image.fromarray(input_array)
    foreground_img.save(output_rgba_path)

def process_directory(input_dir: str, output_dir: str, model, device="cuda"):
    """
    遍历目录，批量处理图像
    """
    # 如果输出目录不存在，则创建它
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义支持的图像格式
    valid_exts = ('.jpg', '.jpeg', '.png')
    
    # 过滤出符合格式的文件
    try:
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
    except FileNotFoundError:
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    if not files:
        print(f"No valid images found in '{input_dir}'.")
        return
        
    print(f"Found {len(files)} images in '{input_dir}'. Starting batch processing...")

    for filename in files:
        input_path = os.path.join(input_dir, filename)
        
        # 提取不带后缀的文件名，用于拼接输出文件名
        base_name = os.path.splitext(filename)[0]
        
        # 统一输出为 .png 格式（因为 .jpg 不支持 RGBA 透明通道）
        out_mask = os.path.join(output_dir, f"{base_name}_mask.png")
        out_rgba = os.path.join(output_dir, f"{base_name}_rgba.png")
        
        print(f"Processing: {filename} ... ", end="", flush=True)
        try:
            extract_foreground(input_path, out_mask, out_rgba, model, device)
            print("Done")
        except Exception as e:
            print(f"Failed! Error: {e}")

    print(f"\nBatch processing complete! All outputs saved to '{output_dir}'.")

if __name__ == "__main__":
    # 配置设备
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 指定输入和输出目录的相对或绝对路径
    INPUT_DIR = "data/images"   # 请替换为你的输入文件夹路径
    OUTPUT_DIR = "data/masks" # 请替换为你的输出文件夹路径
    
    # 2. 初始化模型 (只需要加载一次，避免显存溢出或重复加载耗时)
    birefnet = load_birefnet(device=DEVICE)
    
    # 3. 执行批量处理
    process_directory(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        model=birefnet,
        device=DEVICE
    )