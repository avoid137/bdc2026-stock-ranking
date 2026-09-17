#!/bin/bash
# 训练脚本：从训练数据开始训练模型

echo "=== 开始训练 ==="

cd /app/code/src

# 设置随机种子（保证可复现）
export PYTHONHASHSEED=42

# 运行训练
python train.py

# 检查模型是否生成
if [ -f /app/model/best_model.pth ]; then
    echo "训练完成，模型已保存"
else
    echo "训练失败，未生成模型"
    exit 1
fi

echo "=== 训练完成 ==="