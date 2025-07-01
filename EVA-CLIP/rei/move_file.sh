#!/bin/bash

# 设置源目录和目标目录
SRC_DIR="/llm_reco_ssd/yangzhuoran/dataset/laion-conceptual-captions-12m-webdataset/embedding/image_embedding"
DST_DIR="${SRC_DIR}/00000"

# 如果目标目录不存在则创建
mkdir -p "$DST_DIR"

# 移动所有 .npy 文件（仅限当前目录，不递归子目录）
for file in "$SRC_DIR"/*.npy; do
    if [ -f "$file" ]; then
        mv "$file" "$DST_DIR/"
        echo "Moved $(basename "$file") → 00000/"
    fi
done

echo "All .npy files moved to 00000."
