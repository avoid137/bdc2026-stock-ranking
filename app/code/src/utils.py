import pandas as pd
import numpy as np
import joblib
import os

from tqdm import tqdm
import gc


def _rolling_linear_regression(x, y):
    x = np.vstack([np.ones(len(x)), x]).T
    beta, res, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return beta[1], res[0] if len(res) > 0 else 0.0


def engineer_features_158plus39(df):
    """
    差异化填充策略：
    - 时序指标：ffill 前向填充
    - 交易类指标：停牌时天然为0，单独填0
    - 剩余NaN：fillna(0) 兜底
    """
    df_copy = df.copy()
    df_158 = engineer_features(df_copy)
    df_39 = engineer_features_39(df_copy)
    feature_cols_39 = [
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal',
        'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio',
        'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60',
        'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
    ]
    df_39_exist = df_39[feature_cols_39]
    df_final = pd.concat([df_158, df_39_exist], axis=1)
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]

    # 统一处理 inf 和 NaN
    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    # 1. 时序数据：前向填充全表
    df_final.ffill(inplace=True)
    # 2. 交易类指标：停牌时天然为0，单独覆盖对应列
    trade_cols = ["成交量", "成交额", "涨跌额", "换手率", "volume_change"]
    existing_trade_cols = [col for col in trade_cols if col in df_final.columns]
    if existing_trade_cols:
        df_final[existing_trade_cols] = df_final[existing_trade_cols].fillna(0)
    # 3. 剩余 NaN（极少）全表兜底填 0
    df_final.fillna(0, inplace=True)
    return df_final


def engineer_features_39(df):
    """
    计算39个技术指标特征。
    差异化填充策略：
    - 时序指标：ffill 前向填充
    - 交易类指标：停牌时天然为0，单独填0
    - 剩余NaN：fillna(0) 兜底
    """
    try:
        import talib
    except ImportError:
        print("请安装TA-Lib: pip install TA-Lib")
        raise

    df = df.copy()
    open_ = df['开盘'].astype(float)
    high = df['最高'].astype(float)
    low = df['最低'].astype(float)
    close = df['收盘'].astype(float)
    volume = df['成交量'].astype(float)

    # 移动平均线
    df['sma_5'] = talib.SMA(close, 5)
    df['sma_20'] = talib.SMA(close, 20)
    df['ema_12'] = talib.EMA(close, 12)
    df['ema_26'] = talib.EMA(close, 26)
    df['ema_60'] = talib.EMA(close, 60)

    # MACD
    macd_line, macd_signal_line, _ = talib.MACD(close, 12, 26, 9)
    df['macd'] = macd_line
    df['macd_signal'] = macd_signal_line

    # RSI
    df['rsi'] = talib.RSI(close, 14)

    # KDJ
    df['kdj_k'], df['kdj_d'] = talib.STOCH(high, low, close, 9, 3, 3)
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']

    # Bollinger Bands
    df['boll_mid'], df['boll_upper'], df['boll_lower'] = talib.BBANDS(close, 20, 2, 2)
    df['boll_std'] = (df['boll_upper'] - df['boll_mid']) / 2
    df.drop(['boll_upper', 'boll_lower'], axis=1, inplace=True)

    # ATR
    df['atr_14'] = talib.ATR(high, low, close, 14)

    # OBV
    df['obv'] = talib.OBV(close, volume)

    # Volume-related features
    df['volume_change'] = volume.pct_change(fill_method=None)
    df['volume_ma_5'] = talib.SMA(volume, 5)
    df['volume_ma_20'] = talib.SMA(volume, 20)
    df['volume_ratio'] = df['volume_ma_5'] / (df['volume_ma_20'] + 1e-12)

    # Returns and Volatility
    df['return_1'] = close.pct_change(1)
    df['return_5'] = close.pct_change(5)
    df['return_10'] = close.pct_change(10)
    df['volatility_10'] = df['return_1'].rolling(10).std()
    df['volatility_20'] = df['return_1'].rolling(20).std()

    # Spreads
    df['high_low_spread'] = high - low
    df['open_close_spread'] = open_ - close
    df['high_close_spread'] = high - close
    df['low_close_spread'] = low - close

    # 处理 inf 和 NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 差异化填充策略
    # 1. 时序数据：前向填充
    df.ffill(inplace=True)

    # 2. 交易类指标：停牌时天然为0，单独覆盖
    trade_cols = ["成交量", "成交额", "涨跌额", "换手率", "volume_change"]
    existing_trade_cols = [col for col in trade_cols if col in df.columns]
    df[existing_trade_cols] = df[existing_trade_cols].fillna(0)

    # 3. 剩余 NaN（极少）填 0 兜底
    df.fillna(0, inplace=True)

    return df


def engineer_features(df):
    """
    使用talib加速特征计算（158个Alpha特征）。
    差异化填充策略：
    - 时序指标：ffill 前向填充
    - 交易类指标：停牌时天然为0，单独填0
    - 剩余NaN：fillna(0) 兜底
    """
    try:
        import talib
    except ImportError:
        print("请安装TA-Lib: pip install TA-Lib")
        raise

    df = df.copy()
    open_ = df['开盘'].astype(float)
    high = df['最高'].astype(float)
    low = df['最低'].astype(float)
    close = df['收盘'].astype(float)
    volume = df['成交量'].astype(float)
    vwap = df['成交额'] / (volume + 1e-12)

    features = []
    feature_names = []

    # 1. K-line features (9 features)
    features.extend([
        (close - open_) / (open_ + 1e-12),
        (high - low) / (open_ + 1e-12),
        (close - open_) / (high - low + 1e-12),
        (high - pd.concat([open_, close], axis=1).max(axis=1)) / (open_ + 1e-12),
        (high - pd.concat([open_, close], axis=1).max(axis=1)) / (high - low + 1e-12),
        (pd.concat([open_, close], axis=1).min(axis=1) - low) / (open_ + 1e-12),
        (pd.concat([open_, close], axis=1).min(axis=1) - low) / (high - low + 1e-12),
        (2 * close - high - low) / (open_ + 1e-12),
        (2 * close - high - low) / (high - low + 1e-12)
    ])
    feature_names.extend(['KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2'])

    # 2. Price-related features (4 features)
    features.extend([
        open_ / (close + 1e-12),
        high / (close + 1e-12),
        low / (close + 1e-12),
        vwap / (close + 1e-12)
    ])
    feature_names.extend(['OPEN0', 'HIGH0', 'LOW0', 'VWAP0'])

    windows = [5, 10, 20, 30, 60]

    # 3. Price change features
    for w in windows:
        features.append(close.shift(w) / (close + 1e-12))
        feature_names.append(f'ROC{w}')

    # 4. Moving average features
    for w in windows:
        features.append(talib.SMA(close, w) / (close + 1e-12))
        feature_names.append(f'MA{w}')

    # 5. Standard deviation features
    for w in windows:
        features.append(talib.STDDEV(close, w) / (close + 1e-12))
        feature_names.append(f'STD{w}')

    # 6. Regression-based features
    for w in windows:
        slope = talib.LINEARREG_SLOPE(close, w)
        features.append(slope / (close + 1e-12))
        feature_names.append(f'BETA{w}')

        corr = close.rolling(w).corr(pd.Series(range(w), index=close.index[:w]))
        features.append(corr ** 2)
        feature_names.append(f'RSQR{w}')

        intercept = talib.LINEARREG_INTERCEPT(close, w)
        resi = close - (slope * (w - 1) + intercept)
        features.append(resi / (close + 1e-12))
        feature_names.append(f'RESI{w}')

    # 7. Max/Min features
    for w in windows:
        features.append(talib.MAX(high, w) / (close + 1e-12))
        feature_names.append(f'MAX{w}')
    for w in windows:
        features.append(talib.MIN(low, w) / (close + 1e-12))
        feature_names.append(f'MIN{w}')

    # 8. Quantile features
    for w in windows:
        features.append(close.rolling(w).quantile(0.8) / (close + 1e-12))
        feature_names.append(f'QTLU{w}')
    for w in windows:
        features.append(close.rolling(w).quantile(0.2) / (close + 1e-12))
        feature_names.append(f'QTLD{w}')

    # 9. Rank features
    for w in windows:
        features.append(close.rolling(w).rank(pct=True))
        feature_names.append(f'RANK{w}')

    # 10. Stochastic oscillator features
    for w in windows:
        minlow = low.rolling(w).min()
        maxhigh = high.rolling(w).max()
        features.append((close - minlow) / (maxhigh - minlow + 1e-12))
        feature_names.append(f'RSV{w}')

    # 11. Index of Max/Min features
    for w in windows:
        features.append(high.rolling(w).apply(np.argmax, raw=True) / w)
        feature_names.append(f'IMAX{w}')
    for w in windows:
        features.append(low.rolling(w).apply(np.argmin, raw=True) / w)
        feature_names.append(f'IMIN{w}')
    for w in windows:
        imax = high.rolling(w).apply(np.argmax, raw=True)
        imin = low.rolling(w).apply(np.argmin, raw=True)
        features.append((imax - imin) / w)
        feature_names.append(f'IMXD{w}')

    # 12. Correlation features
    logvol = np.log(volume + 1)
    for w in windows:
        features.append(talib.CORREL(close, logvol, w))
        feature_names.append(f'CORR{w}')

    cr = close / close.shift(1)
    vr = volume / (volume.shift(1) + 1e-12)
    logvr = np.log(vr + 1)
    for w in windows:
        df_corr = pd.concat([cr, logvr], axis=1).fillna(0)
        features.append(talib.CORREL(df_corr.iloc[:, 0], df_corr.iloc[:, 1], w))
        feature_names.append(f'CORD{w}')

    # 13. Count features
    pos = close > close.shift(1)
    neg = close < close.shift(1)
    for w in windows:
        features.append(pos.rolling(w).mean())
        feature_names.append(f'CNTP{w}')
    for w in windows:
        features.append(neg.rolling(w).mean())
        feature_names.append(f'CNTN{w}')
    for w in windows:
        features.append(pos.rolling(w).mean() - neg.rolling(w).mean())
        feature_names.append(f'CNTD{w}')

    # 14. Sum of price change features
    absret = (close - close.shift(1)).abs()
    upret = (close - close.shift(1)).clip(lower=0)
    downret = -(close - close.shift(1)).clip(upper=0)
    for w in windows:
        sumabs = absret.rolling(w).sum()
        sumup = upret.rolling(w).sum()
        features.append(sumup / (sumabs + 1e-12))
        feature_names.append(f'SUMP{w}')
    for w in windows:
        sumabs = absret.rolling(w).sum()
        sumdown = downret.rolling(w).sum()
        features.append(sumdown / (sumabs + 1e-12))
        feature_names.append(f'SUMN{w}')
    for w in windows:
        sumabs = absret.rolling(w).sum()
        sumup = upret.rolling(w).sum()
        sumdown = downret.rolling(w).sum()
        features.append((sumup - sumdown) / (sumabs + 1e-12))
        feature_names.append(f'SUMD{w}')

    # 15. Volume-related features
    for w in windows:
        features.append(talib.SMA(volume, w) / (volume + 1e-12))
        feature_names.append(f'VMA{w}')
    for w in windows:
        features.append(talib.STDDEV(volume, w) / (volume + 1e-12))
        feature_names.append(f'VSTD{w}')

    # 16. Weighted volume features
    volret_abs = ((close / close.shift(1) - 1).abs()) * volume
    for w in windows:
        meanvr = volret_abs.rolling(w).mean()
        stdvr = volret_abs.rolling(w).std()
        features.append(stdvr / (meanvr + 1e-12))
        feature_names.append(f'WVMA{w}')

    # 17. Volume change sum features
    absvol = (volume - volume.shift(1)).abs()
    upvol = (volume - volume.shift(1)).clip(lower=0)
    downvol = -(volume - volume.shift(1)).clip(upper=0)
    for w in windows:
        sumabs = absvol.rolling(w).sum()
        sumup = upvol.rolling(w).sum()
        features.append(sumup / (sumabs + 1e-12))
        feature_names.append(f'VSUMP{w}')
    for w in windows:
        sumabs = absvol.rolling(w).sum()
        sumdown = downvol.rolling(w).sum()
        features.append(sumdown / (sumabs + 1e-12))
        feature_names.append(f'VSUMN{w}')
    for w in windows:
        sumabs = absvol.rolling(w).sum()
        sumup = upvol.rolling(w).sum()
        sumdown = downvol.rolling(w).sum()
        features.append((sumup - sumdown) / (sumabs + 1e-12))
        feature_names.append(f'VSUMD{w}')

    # Combine all features
    feat_df = pd.concat(features, axis=1)
    feat_df.columns = feature_names
    df = pd.concat([df, feat_df], axis=1)

    # 处理 inf 和 NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 差异化填充策略
    # 1. 时序数据：前向填充
    df.ffill(inplace=True)

    # 2. 交易类指标：停牌时天然为0，单独覆盖
    trade_cols = ["成交量", "成交额", "涨跌额", "换手率", "volume_change"]
    existing_trade_cols = [col for col in trade_cols if col in df.columns]
    df[existing_trade_cols] = df[existing_trade_cols].fillna(0)

    # 3. 剩余 NaN（极少）填 0 兜底
    df.fillna(0, inplace=True)

    return df


def process_single_date(date, features, sequence_length, stock_groups, day_labels):
    """
    修复版本：使用 np.searchsorted，统一使用字符串格式比较
    """
    import numpy as np

    date_str = date.strftime('%Y-%m-%d')

    day_seqs = []
    day_tgts = []
    day_stks = []
    day_inds = []

    for sid, cache in stock_groups.items():
        sid_int = int(sid) if not isinstance(sid, int) else sid
        if sid_int not in day_labels:
            continue

        dt_arr = cache['dt']
        pos = np.searchsorted(dt_arr, date_str, side='right')

        if pos < sequence_length:
            continue

        seq = cache['feat'][pos - sequence_length: pos]
        target = day_labels[sid_int]
        ind = cache['ind'][pos - 1] if cache['ind'] is not None else 0

        day_seqs.append(seq)
        day_tgts.append(target)
        day_stks.append(sid_int)
        day_inds.append(int(ind))

    if len(day_seqs) < 3:
        return None

    day_tgts = np.array(day_tgts)
    sort_idx = np.argsort(day_tgts)[::-1]
    rel = np.zeros_like(day_tgts, dtype=np.float32)
    for rk, i in enumerate(sort_idx):
        rel[i] = np.exp(-rk / 5.0)

    return {
        'sequences': day_seqs,
        'targets': day_tgts,
        'relevance': rel,
        'stock_indices': day_stks,
        'industry_ids': day_inds,
        'date': date
    }


def create_ranking_dataset_multiprocess(data, features, sequence_length, ranking_data_path=None, max_workers=None,
                                        return_dates=False):
    """
    输入：股票历史数据 DataFrame，特征列名列表，序列长度
    return_dates: True 同步返回每条样本对应的交易日列表
    """
    if ranking_data_path and os.path.exists(ranking_data_path):
        cached = joblib.load(ranking_data_path)
        if return_dates and len(cached) == 6:
            return cached
        elif not return_dates and len(cached) == 5:
            return cached
        print("缓存格式不匹配，重新生成数据集")

    print("构建排序数据集")
    data = data.copy()

    # 日期列处理
    if "datetime" in data.columns:
        if not pd.api.types.is_datetime64_any_dtype(data["datetime"]):
            data["datetime"] = pd.to_datetime(data["datetime"])
    elif "日期" in data.columns:
        data["datetime"] = pd.to_datetime(data["日期"])
    else:
        raise ValueError("数据中缺少日期列 ('datetime' 或 '日期')")

    # ✅ 关键修复：在构建 stock_groups 之前过滤 label 为 NaN 的行
    data = data.dropna(subset=['label'])
    print(f"  过滤 label NaN 后: {len(data)} 行")

    # ✅ 过滤无效的 instrument
    data = data[data['instrument'] >= 0]
    print(f"  过滤无效 instrument 后: {len(data)} 行")

    data = data.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    all_dates = sorted(data['datetime'].unique())

    if len(all_dates) < sequence_length:
        print("交易日不足窗口长度，返回空")
        if return_dates:
            return [], [], [], [], [], []
        return [], [], [], [], []

    min_dt = all_dates[sequence_length - 1]
    valid_dates = [d for d in all_dates if d >= min_dt]
    if not valid_dates:
        if return_dates:
            return [], [], [], [], [], []
        return [], [], [], [], []

    print(f"  有效交易日: {len(valid_dates)}")
    print(f"  有效股票: {data['instrument'].nunique()}")

    # 预构建 numpy 数组字典
    print("预处理股票数据为 numpy 数组...")
    stock_groups = {}
    has_industry = 'industry_id' in data.columns
    for sid, g in tqdm(data.groupby('instrument'), desc="构建numpy缓存"):
        sid_int = int(sid)
        stock_groups[sid_int] = {
            'feat': g[features].values.astype(np.float32),
            'dt': g['datetime'].dt.strftime('%Y-%m-%d').values,
            'ind': g['industry_id'].values if has_industry else None
        }

    print(f"  stock_groups 大小: {len(stock_groups)}")

    seqs, tgts, rels, stkidx, indlist, date_list = [], [], [], [], [], []

    for d in tqdm(valid_dates, desc="处理日期"):
        day_data = data[data['datetime'] == d]

        if len(day_data) < 3:
            continue

        day_labels = {}
        for _, row in day_data.iterrows():
            sid = int(row['instrument'])
            day_labels[sid] = row['label']

        res = process_single_date(
            date=d,
            features=features,
            sequence_length=sequence_length,
            stock_groups=stock_groups,
            day_labels=day_labels
        )

        if res is not None:
            seqs.append(res['sequences'])
            tgts.append(res['targets'])
            rels.append(res['relevance'])
            stkidx.append(res['stock_indices'])
            indlist.append(res['industry_ids'])
            if return_dates:
                date_list.append(res['date'])

        del day_data, day_labels
        gc.collect()

    print(f"生成有效样本天数: {len(seqs)}")

    if len(seqs) > 0:
        daily_counts = [len(s) for s in seqs]
        print(f"每日股票数量: 最少={min(daily_counts)}, 最多={max(daily_counts)}, 平均={np.mean(daily_counts):.1f}")

    if ranking_data_path:
        save_tuple = (seqs, tgts, rels, stkidx, indlist, date_list) if return_dates else (seqs, tgts, rels, stkidx,
                                                                                          indlist)
        joblib.dump(save_tuple, ranking_data_path)

    if return_dates:
        return seqs, tgts, rels, stkidx, indlist, date_list
    else:
        return seqs, tgts, rels, stkidx, indlist


def create_dataset(data, features, sequence_length, ranking_data_path=None):
    """保持原有接口"""
    return create_ranking_dataset_multiprocess(data, features, sequence_length, ranking_data_path)


def create_ranking_dataset_vectorized(data, features, sequence_length, ranking_data_path=None,
                                      min_window_end_date=None):
    print("正在创建排序数据集（向量化加速版本）...")
    data = data.copy()
    if '日期' in data.columns and 'datetime' not in data.columns:
        data['datetime'] = pd.to_datetime(data['日期'])
    data = data.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    data = data.dropna(subset=['label'])

    has_industry = 'industry_id' in data.columns

    all_windows = []
    print("Step 1: 为每只股票生成滑动窗口...")
    grouped = data.groupby('instrument')
    for stock_code, group in tqdm(grouped, desc="Processing stocks"):
        if len(group) < sequence_length:
            continue
        feature_values = group[features].values.astype(np.float32)
        labels = group['label'].values.astype(np.float32)
        dates = group['datetime'].values
        industry_val = group['industry_id'].values if has_industry else np.zeros(len(group), dtype=np.int64)
        n = len(group)
        num_windows = n - sequence_length + 1
        for i in range(num_windows):
            end_idx = i + sequence_length - 1
            seq = feature_values[i: i + sequence_length]
            target = labels[end_idx]
            end_date = dates[end_idx]
            ind_id = industry_val[end_idx]
            all_windows.append((end_date, stock_code, seq, target, ind_id))

    print("Step 2: 按日期聚合窗口...")
    window_df = pd.DataFrame(all_windows, columns=['date', 'stock_code', 'seq', 'target', 'industry_id'])
    del all_windows
    gc.collect()

    sequences = []
    targets = []
    relevance_scores = []
    stock_indices = []
    industry_ids_list = []

    print("Step 3: 构建每日样本并计算 relevance...")
    grouped_by_date = window_df.groupby('date')
    if min_window_end_date is not None:
        min_window_end_date = pd.to_datetime(min_window_end_date)
    for date, group in tqdm(grouped_by_date, desc="Aggregating by date"):
        if min_window_end_date is not None and pd.Timestamp(date) < min_window_end_date:
            continue
        if len(group) < 3:
            continue
        day_seqs = np.stack(group['seq'].values)
        day_targets = group['target'].values
        day_stocks = group['stock_code'].tolist()
        day_inds = group['industry_id'].values.astype(np.int64)

        sorted_indices = np.argsort(day_targets)[::-1]
        relevance = np.zeros_like(day_targets, dtype=np.float32)
        for rank, idx in enumerate(sorted_indices):
            relevance[idx] = len(day_targets) - rank

        sequences.append(day_seqs)
        targets.append(day_targets)
        relevance_scores.append(relevance)
        stock_indices.append(day_stocks)
        industry_ids_list.append(day_inds)

    print(f"成功创建 {len(sequences)} 个训练样本")
    if len(sequences) > 0:
        avg_stocks = np.mean([len(seq) for seq in sequences])
        print(f"每个样本平均包含 {avg_stocks:.1f} 只股票")
    return sequences, targets, relevance_scores, stock_indices, industry_ids_list


def create_ranking_dataset_streaming(data, features, sequence_length, temp_dir,
                                     batch_size=50, max_workers=1, return_dates=False, force_refresh=False):
    """
    流式生成训练批次，支持缓存复用，不重复计算
    """
    import gc
    from pathlib import Path

    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    date_min = data["datetime"].min()
    date_max = data["datetime"].max()
    tag_str = f"{date_min}_{date_max}_{sequence_length}_{batch_size}"
    tag_file = temp_dir / "data_tag.txt"

    old_tag = ""
    if tag_file.exists():
        with open(tag_file, "r", encoding="utf-8") as f:
            old_tag = f.read().strip()

    exist_batch_files = sorted(list(temp_dir.glob("batch_*.pkl")))

    if old_tag == tag_str and len(exist_batch_files) > 0 and not force_refresh:
        print(f"✅ 检测到匹配缓存，复用已有批次，跳过数据集构建")
        return exist_batch_files

    print(f"⚠️ 数据参数变更或强制刷新，清理旧缓存重新生成")
    for f in temp_dir.glob("*"):
        try:
            f.unlink()
        except:
            pass
    with open(tag_file, "w", encoding="utf-8") as f:
        f.write(tag_str)

    data = data.copy()
    if "datetime" in data.columns:
        if not pd.api.types.is_datetime64_any_dtype(data["datetime"]):
            data["datetime"] = pd.to_datetime(data["datetime"])
    elif "日期" in data.columns:
        data["datetime"] = pd.to_datetime(data["日期"])
    else:
        raise ValueError("数据中缺少日期列 ('datetime' 或 '日期')")

    data = data.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    all_dates = sorted(data['datetime'].unique())

    if len(all_dates) < sequence_length:
        return []

    min_dt = all_dates[sequence_length - 1]
    valid_dates = [d for d in all_dates if d >= min_dt]
    if not valid_dates:
        return []

    # 构建 numpy 数组缓存，dt 使用字符串格式
    print("预处理股票数据为 numpy 数组缓存...")
    has_industry = 'industry_id' in data.columns
    stock_groups = {}
    for sid, g in data.groupby('instrument'):
        sid_int = int(sid)
        stock_groups[sid_int] = {
            'feat': g[features].values.astype(np.float32),
            'dt': g['datetime'].dt.strftime('%Y-%m-%d').values,  # 字符串格式
            'ind': g['industry_id'].values if has_industry else None
        }

    seqs, tgts, rels, stkidx, indlist = [], [], [], [], []
    date_list = [] if return_dates else None
    batch_count = 0
    batch_files = []

    for d in tqdm(valid_dates, desc="流式处理日期"):
        day_df = data[data['datetime'] == d]

        day_labels = {}
        for _, row in day_df.iterrows():
            sid = int(row['instrument'])
            day_labels[sid] = row['label']

        res = process_single_date(
            date=d,
            features=features,
            sequence_length=sequence_length,
            stock_groups=stock_groups,
            day_labels=day_labels
        )

        if res is not None:
            seqs.append(res['sequences'])
            tgts.append(res['targets'])
            rels.append(res['relevance'])
            stkidx.append(res['stock_indices'])
            indlist.append(res['industry_ids'])
            if return_dates:
                date_list.append(res['date'])

            if len(seqs) >= batch_size:
                if return_dates:
                    save_data = (seqs, tgts, rels, stkidx, indlist, date_list)
                else:
                    save_data = (seqs, tgts, rels, stkidx, indlist)
                batch_file = temp_dir / f"batch_{batch_count:06d}.pkl"
                joblib.dump(save_data, batch_file, compress=0)
                batch_files.append(batch_file)
                batch_count += 1
                seqs, tgts, rels, stkidx, indlist = [], [], [], [], []
                if return_dates:
                    date_list = []
                gc.collect()

        del day_df, day_labels
        gc.collect()

    if len(seqs) > 0:
        if return_dates:
            save_data = (seqs, tgts, rels, stkidx, indlist, date_list)
        else:
            save_data = (seqs, tgts, rels, stkidx, indlist)
        batch_file = temp_dir / f"batch_{batch_count:06d}.pkl"
        joblib.dump(save_data, batch_file, compress=0)
        batch_files.append(batch_file)
        batch_count += 1
        seqs, tgts, rels, stkidx, indlist = [], [], [], [], []
        if return_dates:
            date_list = []
        gc.collect()

    print(f"生成 {batch_count} 个批次文件")
    return batch_files