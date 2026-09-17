#!/bin/bash
# 初始化脚本：创建必要目录、安装依赖等

echo "=== 初始化环境 ==="

# 创建目录
mkdir -p /app/output
mkdir -p /app/temp
mkdir -p /app/model

pip install -r /app/code/requirements.txt

echo "=== 初始化完成 ==="