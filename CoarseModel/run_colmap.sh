#!/bin/bash

# ==========================================
# 1. 设置基础路径（在此处修改路径）
# ==========================================
BASE_PATH="/home/zjr/Tracker/CoarseModel/datasets/conch"

# 定义子目录变量
DATABASE_PATH="$BASE_PATH/database.db"
IMAGE_PATH="$BASE_PATH/images"
SPARSE_PATH="$BASE_PATH/sparse"
DENSE_PATH="$BASE_PATH/dense"

# 如果你 undistorter 使用的是不同的文件夹（如脚本中写的 rgb）
RGB_PATH="$BASE_PATH/rgb"

# ==========================================
# 2. 执行流程
# ==========================================

echo ">>> [1/6] 特征提取 (Feature Extraction)..."
colmap feature_extractor \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGE_PATH" \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 0 \

echo ">>> [2/6] 特征匹配 (Exhaustive Matching)..."
colmap exhaustive_matcher \
    --database_path "$DATABASE_PATH" \
    --SiftMatching.use_gpu 0

echo ">>> [3/6] 稀疏重建 (Sparse Mapping)..."
mkdir -p "$SPARSE_PATH"
colmap mapper \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGE_PATH" \
    --output_path "$SPARSE_PATH"

echo ">>> [4/6] 转换模型为文本格式 (Model Converter TXT)..."
# 注意：mapper 默认输出在 $SPARSE_PATH/0
colmap model_converter \
    --input_path "$SPARSE_PATH/0" \
    --output_path "$SPARSE_PATH/0" \
    --output_type TXT

echo ">>> [5/6] 导出 PLY 点云 (Model Converter PLY)..."
colmap model_converter \
    --input_path "$SPARSE_PATH/0" \
    --output_path "$SPARSE_PATH/0/model.ply" \
    --output_type PLY

# echo ">>> [6/6] 图像去畸变 (Image Undistorter)..."
# mkdir -p "$DENSE_PATH"
# colmap image_undistorter \
#     --image_path "$RGB_PATH" \
#     --input_path "$SPARSE_PATH/0" \
#     --output_path "$DENSE_PATH" \
#     --output_type COLMAP

# echo ">>> 全部任务完成！"