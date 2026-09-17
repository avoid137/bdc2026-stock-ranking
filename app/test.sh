#!/bin/bash
# 推理脚本：加载模型，生成预测结果

echo "=== 开始预测 ==="

cd /app/code/src

# 运行预测
python test.py

# 检查结果是否生成
if [ -f /app/output/result.csv ]; then
    echo "预测完成，结果已保存"
else
    echo "预测失败，未生成结果"
    exit 1
fi

echo "=== 预测完成 ==="