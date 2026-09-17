"""
推理脚本 - 用某个滚动窗口的最佳权重生成 Top-5 选股结果

窗口序号通过环境变量指定（编号与 train.py 存盘的 splitN_best.pth 一致）：
    Windows:  set BEST_SPLIT=7
    Linux:    export BEST_SPLIT=7
未指定时优先读取 train.py 落盘的 best_split_idx.pkl，读不到再回退默认值 8
（本机训练日志中最佳窗口为 Split 8，最佳 FS = 0.1307）。
模型与标准化器始终取同一个 splitN，避免两者不同源。
"""
import os
import sys
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import joblib
from tqdm import tqdm

from config import config
from model import GNNStockRankingModel, NEG_INF
from utils import engineer_features_39, engineer_features_158plus39
from config import feature_columns_map

# ============================================================
# 路径配置
# ============================================================
# 数据文件
data_file = os.path.join(config['data_path'], config['data_file'])

# 选择滚动窗口：环境变量 > best_split_idx.pkl > 默认 8（本机日志中的最佳窗口）
_ind_pkl = os.path.join(config['output_dir'], 'best_split_idx.pkl')
if os.environ.get('BEST_SPLIT'):
    split_idx = int(os.environ['BEST_SPLIT'])
elif os.path.exists(_ind_pkl) and os.path.getsize(_ind_pkl) > 0:
    split_idx = int(joblib.load(_ind_pkl))
    print(f"从 best_split_idx.pkl 读取最佳窗口: split{split_idx}")
else:
    split_idx = 8
    print(f"未找到 best_split_idx.pkl，使用默认窗口: split{split_idx}")

# 模型与其配套的 Scaler —— 必须来自同一个 split
model_path = os.path.join(config['output_dir'], f'split{split_idx}_best.pth')
scaler_path = os.path.join(config['output_dir'], f'split{split_idx}_scaler.pkl')

# 映射文件
stockid2idx_path = os.path.join(config['output_dir'], 'stockid2idx.pkl')
ind_map_path = os.path.join(config['output_dir'], 'industry_map.pkl')
num_ind_path = os.path.join(config['output_dir'], 'num_industries.pkl')

# 输出路径
output_dir = os.path.join(os.path.dirname(os.path.dirname(config['output_dir'])), "output")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f'result_split{split_idx}.csv')

print("=" * 60)
print(f"🔍 加载 Split {split_idx} 最佳模型")
print("=" * 60)

# ============================================================
# 加载映射
# ============================================================
stockid2idx = joblib.load(stockid2idx_path)
industry_map = joblib.load(ind_map_path)
num_industries = joblib.load(num_ind_path)
stock_ids = list(stockid2idx.keys())

print(f"✅ 股票总数: {len(stock_ids)}")
print(f"✅ 行业总数: {num_industries}")
print(f"✅ 模型: split{split_idx}_best.pth")
print(f"✅ Scaler: split{split_idx}_scaler.pkl")

# ============================================================
# 特征工程函数映射
# ============================================================
feature_engineer_func_map = {
    '39': engineer_features_39,
    '158+39': engineer_features_158plus39,
}


def preprocess_predict_data(df, stockid2idx):
    """预测数据预处理"""
    assert config['feature_num'] in feature_engineer_func_map
    fe = feature_engineer_func_map[config['feature_num']]
    cols = [c for c in feature_columns_map[config['feature_num']] if c != "instrument"]

    df = df.copy().sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [g for _, g in df.groupby('股票代码', sort=False)]

    num_proc = min(10, mp.cpu_count())
    with mp.Pool(num_proc) as pool:
        processed_list = list(tqdm(pool.imap(fe, groups), total=len(groups), desc="特征工程"))

    processed = pd.concat(processed_list).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed['industry_id'] = processed['股票代码'].map(industry_map).fillna(0).astype(np.int64)
    processed = processed.dropna(subset=['instrument'])
    processed['instrument'] = processed['instrument'].astype(np.int64)
    processed['日期'] = pd.to_datetime(processed['日期'])

    processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    processed.ffill(inplace=True)

    trade_cols = ["成交量", "成交额", "涨跌额", "换手率", "volume_change"]
    exist = [c for c in trade_cols if c in processed.columns]
    processed[exist] = processed[exist].fillna(0)
    processed.fillna(0, inplace=True)

    return processed, cols


def build_inference_sequences(data, feats, seq_len, stock_ids, latest_date, sid2idx):
    """构建推理序列"""
    seqs, sid_list, ind_list = [], [], []

    for sid in tqdm(stock_ids, desc="构建序列"):
        hist = data[(data['股票代码'] == sid) & (data['日期'] <= latest_date)].sort_values('日期').tail(seq_len)
        if len(hist) == seq_len:
            seqs.append(hist[feats].values.astype(np.float32))
            sid_list.append(sid2idx[sid])
            ind_list.append(int(hist['industry_id'].iloc[-1]))

    if len(seqs) == 0:
        raise ValueError("无有效股票序列")

    return np.array(seqs), np.array(sid_list), np.array(ind_list)


def main():
    print("\n" + "=" * 60)
    print(f"📊 使用 Split {split_idx} 模型进行预测")
    print("=" * 60)

    # ============================================================
    # 1. 检查文件
    # ============================================================
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 模型文件不存在: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"❌ Scaler文件不存在: {scaler_path}")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"❌ 数据文件不存在: {data_file}")

    print(f"✅ 数据文件: {data_file}")
    print(f"✅ 模型文件: {model_path}")
    print(f"✅ Scaler文件: {scaler_path}")

    # ============================================================
    # 2. 加载数据
    # ============================================================
    raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
    raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])

    print(f"原始数据行数: {len(raw_df)}")
    print(f"数据日期范围: {raw_df['日期'].min()} ~ {raw_df['日期'].max()}")

    # 使用 2023 年之后的数据（与训练一致）
    raw_df = raw_df[raw_df['日期'] >= '2023-01-01'].copy()
    print(f"过滤后数据行数: {len(raw_df)}")

    if len(raw_df) == 0:
        raise ValueError("数据为空")

    latest_date = raw_df['日期'].max()
    print(f"预测基准日: {latest_date.date()}")

    # ============================================================
    # 3. 特征工程
    # ============================================================
    processed, feats = preprocess_predict_data(raw_df, stockid2idx)
    print(f"特征数量: {len(feats)}")
    print(f"特征工程后数据行数: {len(processed)}")

    if len(processed) == 0:
        raise ValueError("特征工程后数据为空")

    # ============================================================
    # 4. 标准化
    # ============================================================
    processed[feats] = processed[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = joblib.load(scaler_path)
    processed[feats] = scaler.transform(processed[feats])

    # ============================================================
    # 5. 构建推理序列
    # ============================================================
    seq_len = config['sequence_length']
    seq_np, sid_np, ind_np = build_inference_sequences(
        processed, feats, seq_len, stock_ids, latest_date, stockid2idx
    )

    print(f"有效股票数: {len(seq_np)}")

    if len(seq_np) < 5:
        raise ValueError(f"有效股票不足5只，当前仅有 {len(seq_np)} 只")

    # ============================================================
    # 6. 加载模型
    # ============================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = GNNStockRankingModel(
        input_dim=len(feats),
        config=config,
        num_stocks=len(stock_ids),
        num_industries=num_industries
    )

    # 加载模型权重
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("✅ 模型加载成功")

    # ============================================================
    # 7. 推理
    # ============================================================
    print("\n正在预测...")
    with torch.no_grad():
        x = torch.from_numpy(seq_np).unsqueeze(0).to(device)
        sid_tensor = torch.from_numpy(sid_np).unsqueeze(0).to(device)
        ind_tensor = torch.from_numpy(ind_np).unsqueeze(0).to(device)
        masks = torch.ones(1, len(seq_np), device=device)

        scores, _, _, _, _ = model(x, masks, sid_tensor, ind_tensor)
        scores = scores.squeeze(0).detach().cpu().numpy()

    # ============================================================
    # 8. 排序并输出 Top5
    # ============================================================
    order = np.argsort(scores)[::-1]
    top5 = [stock_ids[i] for i in order[:5]]
    top5_score = scores[order[:5]]

    # 权重生成
    soft_w = F.softmax(torch.tensor(top5_score) / 0.3, dim=0).numpy()
    soft_w = np.clip(soft_w, 0, 0.3)
    soft_w = soft_w / soft_w.sum()

    print("\n" + "=" * 60)
    print(f"📈 Split {split_idx} 模型 Top5 选股结果")
    print("=" * 60)
    for i, (s, sc, w) in enumerate(zip(top5, top5_score, soft_w)):
        print(f"  {i+1}. {s}  分数: {sc:.4f}  权重: {w:.3f}")
    print("=" * 60)

    # ============================================================
    # 9. 保存结果
    # ============================================================
    out_df = pd.DataFrame({
        "stock_id": top5,
        "weight": soft_w
    })
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ 结果保存至: {output_path}")
    print(f"📅 预测日期: {latest_date.date()}")
    print(f"📊 参与排序股票数: {len(order)}")
    print(f"📌 使用模型: Split {split_idx}（权重 {model_path}，Scaler {scaler_path}）")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()