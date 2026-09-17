import os
from pathlib import Path

# 配置参数
sequence_length = 60
feature_num = '158+39'

# 项目根目录（即 app/ 目录）。默认取本文件上两级目录，无需改代码即可迁移：
#   Windows:  set APP_ROOT=D:\path\to\app   （PowerShell: $env:APP_ROOT="D:\path\to\app"）
#   Linux:    export APP_ROOT=/path/to/app
APP_ROOT = Path(os.environ.get('APP_ROOT', Path(__file__).resolve().parents[2])).resolve()
LOCAL_ROOT = str(APP_ROOT)  # 兼容旧变量名

config = {
    # ===== 核心模型 =====
    'sequence_length': 60,
    'd_model': 128,              
    'nhead': 8,                  
    'num_layers': 3,             
    'dim_feedforward': 256,      

    # ===== 训练 =====
    'batch_size': 8,
    'num_epochs': 50,
    'learning_rate': 5e-5,
    'dropout': 0.45,             # 0.4 → 0.45（轻微增加防过拟合）
    'weight_decay': 1e-4,        # 新增（L2正则化）

    # ===== GAT =====
    'gat_heads': 2,
    'k_neighbors': 10,
    'industry_dim': 16,

    # ===== 损失 =====
    'loss_temperature': 0.7,     # 1.0 → 0.7（更聚焦头部）
    'pairwise_weight': 0.3,
    'topk_metric': 5,            # 10 → 5
    'top5_weight': 2.0,
    'base_weight': 1.0,

    # ===== 其他 =====
    'feature_num': '158+39',
    'max_grad_norm': 5.0,
    'fp16_support': False,
    'seed': 42,

    # ===== 路径 =====
    'output_dir': str(APP_ROOT / "model"),
    'data_path': str(APP_ROOT / "model" / "data"),
    'temp_dir': str(APP_ROOT / "temp" / "train_batches"),
    'data_file': 'stock_data.csv',
    'industry_map_csv': str(APP_ROOT / "model" / "data" / "industry_map.csv"),
}

feature_columns_map = {
    '39': ['instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
           'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv', 'volume_ma_5',
           'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60',
           'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10', 'high_low_spread',
           'open_close_spread', 'high_close_spread', 'low_close_spread'],
    '158+39': ['instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
               'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0',
               'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5',
               'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10',
               'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20',
               'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30',
               'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30',
               'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60',
               'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60',
               'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
               'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60',
               'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60',
               'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60', 'VMA5',
               'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5', 'WVMA10',
               'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5', 'VSUMN10',
               'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60', 'sma_5', 'sma_20',
               'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20',
               'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 'volatility_10',
               'volatility_20', 'return_1', 'return_5', 'return_10', 'high_low_spread', 'open_close_spread',
               'high_close_spread', 'low_close_spread']  # 已删除 market_cap, log_mcap
}