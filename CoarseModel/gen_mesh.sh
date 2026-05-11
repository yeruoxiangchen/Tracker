#!/bin/bash

# --- 配置区域 ---
# 输入输出路径
RGB_PATH="/home/zjr/Tracker/CoarseModel/datasets/conch/images/00000.jpg"
MASK_PATH="/home/zjr/Tracker/CoarseModel/datasets/conch/masks/00000.png"
TEMP_IMAGE="/home/zjr/Tracker/CoarseModel/datasets/conch/obj.png"
OUTPUT_DIR="/home/zjr/Tracker/CoarseModel/datasets/conch/models"

# 环境与目录配置
CONDA_PATH="/home/zjr/anaconda3/etc/profile.d/conda.sh" # 你的 conda 路径
ENV_NAME="instantmesh"
TRIPOSR_DIR="/home/zjr/Tracker/TripoSR"
GPU_ID=2

# --- 执行开始 ---

# 1. 图像预处理 (调用上面的 Python 脚本)
echo "--- Step 1: Extracting and Centering Object ---"
python /home/zjr/Tracker/CoarseModel/datasets/mask_obj.py \
    --rgb "$RGB_PATH" \
    --mask "$MASK_PATH" \
    --output "$TEMP_IMAGE"

# # 2. 激活环境并运行模型生成
# echo "--- Step 2: Running TripoSR ---"
# # 在脚本中激活 conda 需要 source conda.sh
# source "$CONDA_PATH"
# conda activate "$ENV_NAME"

# 切换到工作目录并执行
CUDA_VISIBLE_DEVICES=$GPU_ID python /home/zjr/Tracker/TripoSR/run.py "$TEMP_IMAGE" --output-dir "$OUTPUT_DIR"

echo "--- All Done! ---"