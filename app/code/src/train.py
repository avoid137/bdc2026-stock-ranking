import bisect

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config import config
from utils import engineer_features_39, engineer_features_158plus39
from utils import create_ranking_dataset_multiprocess
from model import GNNStockRankingModel, NEG_INF, EMA
from config import feature_columns_map
import joblib
import os
import json
import multiprocessing as mp
import random
import gc
import tempfile
from pathlib import Path
from utils import create_ranking_dataset_streaming
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from torch import amp

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


feature_engineer_func_map = {
    '39': engineer_features_39,
    '158+39': engineer_features_158plus39
}


# ==================== NDCG Loss ====================
def ndcg_loss(y_pred, y_true, temperature=1.0, k=5):
    """计算 NDCG Loss（返回 0~1 之间的正值）"""
    batch_size, n = y_pred.shape

    actual_k = min(k, n)
    if actual_k == 0:
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

    pred_score = y_pred / temperature

    true_sort = torch.sort(y_true, dim=-1, descending=True)[0]
    true_sort = true_sort[:, :actual_k]

    pos = torch.arange(1, actual_k + 1, device=y_pred.device).float()
    discount = torch.log2(pos + 1)

    dcg_true = torch.sum(true_sort / discount, dim=-1)
    dcg_true = torch.clamp(dcg_true, min=1e-8)

    pred_idx = torch.sort(pred_score, descending=True)[1]
    pred_rank = torch.gather(y_true, -1, pred_idx)[:, :actual_k]
    dcg_pred = torch.sum(pred_rank / discount, dim=-1)

    ndcg = dcg_pred / dcg_true
    ndcg = torch.clamp(ndcg, 0.0, 1.0)

    return (1.0 - ndcg).mean()


# ==================== 🔥 Margin Ranking Loss ====================
def margin_ranking_loss(y_pred, y_true, margin=0.1, k=5):
    """Top5 和 Bottom5 之间的 margin 约束"""
    batch_size, n = y_pred.shape
    actual_k = min(k, n)

    if actual_k == 0:
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

    # Top5 索引
    top_idx = torch.topk(y_true, actual_k, dim=1)[1]
    # Bottom5 索引
    bottom_idx = torch.topk(y_true, actual_k, dim=1, largest=False)[1]

    top_scores = torch.gather(y_pred, 1, top_idx)
    bottom_scores = torch.gather(y_pred, 1, bottom_idx)

    # Top5 平均分应该比 Bottom5 平均分高 margin
    diff = top_scores.mean(dim=1) - bottom_scores.mean(dim=1)
    loss = F.relu(margin - diff)

    return loss.mean()


# ==================== 加权排序损失 ====================
class WeightedRankingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = config['loss_temperature']
        self.k = config['topk_metric']
        self.weight_factor = config['top5_weight']
        self.pairwise_weight = config['pairwise_weight']
        self.base_weight = config['base_weight']

    def listwise_loss(self, y_pred, y_true, weights):
        pred_probs = F.softmax(y_pred / self.temperature, dim=1)
        target_probs = F.softmax(y_true / self.temperature, dim=1)
        weighted_ce = -(target_probs * torch.log(pred_probs + 1e-12) * weights)
        ce_loss = (weighted_ce.sum(dim=1) / (weights.sum(dim=1) + 1e-12)).mean()
        return ce_loss

    def pairwise_loss(self, y_pred, y_true, weights):
        batch_size, num_items = y_pred.size()
        pred_diff = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)
        true_diff = y_true.unsqueeze(2) - y_true.unsqueeze(1)
        mask = (true_diff != 0).float()
        weight_matrix = weights.unsqueeze(2) + weights.unsqueeze(1)
        pairwise_loss = torch.sigmoid(-pred_diff * torch.sign(true_diff))
        weighted_loss = pairwise_loss * mask * weight_matrix
        num_pairs = mask.sum(dim=[1, 2]).clamp(min=1)
        loss = (weighted_loss.sum(dim=[1, 2]) / num_pairs).mean()
        return loss

    def forward(self, y_pred, y_true):
        batch_size, num_items = y_true.size()
        k = min(self.k, num_items)
        _, top_indices = torch.topk(y_true, k, dim=1)
        weights = torch.full_like(y_true, fill_value=self.base_weight)
        for i in range(batch_size):
            weights[i, top_indices[i]] = self.weight_factor
        ndcg = ndcg_loss(y_pred, y_true, temperature=self.temperature, k=self.k)
        pairwise = self.pairwise_loss(y_pred, y_true, weights)
        # 🔥 加入 Margin Loss
        margin = margin_ranking_loss(y_pred, y_true, margin=0.1, k=5)
        total_loss = ndcg + self.pairwise_weight * pairwise + 0.1 * margin
        return total_loss


# ==================== 涨跌停过滤标签构建 ====================
def _build_label_and_clean(processed, drop_small_open=True, verbose=True):
    """构建标签并清理数据"""
    if verbose:
        print(f"\n  [构建标签] 输入: {len(processed)} 行")

    processed = processed[processed['instrument'] >= 0]
    if verbose:
        print(f"  过滤无效 instrument 后: {len(processed)} 行")

    if 'datetime' not in processed.columns:
        processed['datetime'] = pd.to_datetime(processed['日期'])

    processed['open_t1'] = processed.groupby('instrument')['开盘'].shift(-1)
    processed['open_t5'] = processed.groupby('instrument')['开盘'].shift(-5)

    if verbose:
        print(f"  计算开盘价后: open_t1有效={processed['open_t1'].notna().sum()}")

    processed['ret_t1'] = (processed['open_t1'] - processed['收盘']) / (processed['收盘'] + 1e-12)
    processed['ret_t5'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)

    if verbose:
        print(f"  计算收益率后: ret_t1有效={processed['ret_t1'].notna().sum()}")

    processed = processed.dropna(subset=['ret_t1', 'ret_t5', 'open_t1'])

    if verbose:
        print(f"  去除NaN后: {len(processed)} 行")

    limit_thresh = 0.095
    mask_limit = (processed['ret_t1'].abs() < limit_thresh) & (processed['ret_t5'].abs() < limit_thresh)
    processed = processed[mask_limit]

    if verbose:
        print(f"  涨跌停过滤后: {len(processed)} 行")

    if drop_small_open:
        processed = processed[processed['open_t1'] > 1e-4]
        if verbose:
            print(f"  开盘价过滤后: {len(processed)} 行")

    processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed = processed[processed['label'].abs() < 0.5]

    if verbose:
        print(f"  标签过滤后: {len(processed)} 行")
        print(f"  标签范围: [{processed['label'].min():.4f}, {processed['label'].max():.4f}]")

    if 'industry_id' in processed.columns:
        day_ind_mean = processed.groupby(["datetime", "industry_id"])["label"].transform("mean")
        processed["label"] = processed["label"] - day_ind_mean
        if verbose:
            print(f"  行业中性化完成")

    processed = processed.dropna(subset=['label'])
    processed.drop(columns=['open_t1', 'open_t5', 'ret_t1', 'ret_t5'], inplace=True)

    if verbose:
        print(f"  最终输出: {len(processed)} 行")
        daily_counts = processed.groupby('datetime').size()
        print(f"  每日平均股票数: {daily_counts.mean():.1f}")

    return processed


def _preprocess_common(df, stockid2idx, industry_map, desc, drop_small_open=True):
    assert config['feature_num'] in feature_engineer_func_map
    assert stockid2idx is not None, "stockid2idx 不能为空"
    feature_engineer = feature_engineer_func_map[config['feature_num']]
    feature_columns = feature_columns_map[config['feature_num']]

    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    print(f"正在使用多进程进行{desc}...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    min_length = config['sequence_length']
    original_count = len(groups)

    print(f"  原始股票数量: {original_count}")

    groups = [g for g in groups if len(g) >= min_length + 10]
    if len(groups) < original_count:
        print(f"  过滤掉 {original_count - len(groups)} 只数据不足{min_length + 10}天的股票")

    if len(groups) == 0:
        raise ValueError(f"{desc}输入为空，所有股票数据均不足{min_length}天")

    num_processes = min(4, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc=desc))

    processed = pd.concat(processed_list).reset_index(drop=True)
    print(f"  特征工程后: {len(processed)} 行")

    processed['股票代码'] = processed['股票代码'].astype(str).str.zfill(6)

    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    processed = processed[processed['instrument'] >= 0]
    print(f"  过滤无效 instrument 后: {len(processed)} 行")
    print(f"  instrument 唯一值数量: {processed['instrument'].nunique()}")
    print(f"  instrument 范围: {processed['instrument'].min()} - {processed['instrument'].max()}")

    processed['datetime'] = pd.to_datetime(processed['日期'])

    if industry_map is not None:
        processed['industry_id'] = processed['股票代码'].map(industry_map).fillna(0).astype(np.int64)
    else:
        processed['industry_id'] = 0

    max_date_per_stock = processed.groupby('instrument')['datetime'].transform('max')
    processed = processed[processed['datetime'] < max_date_per_stock - pd.Timedelta(days=5)]
    print(f"  过滤未来数据后: {len(processed)} 行")

    print(f"  开始构建标签...")
    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)
    print(f"  标签构建后: {len(processed)} 行")

    return processed, feature_columns


def preprocess_data(df, is_train=True, stockid2idx=None, industry_map=None):
    if not is_train:
        return _preprocess_common(df, stockid2idx, industry_map, desc="特征工程", drop_small_open=False)
    return _preprocess_common(df, stockid2idx, industry_map, desc="特征工程", drop_small_open=True)


def preprocess_val_data(df, stockid2idx=None, industry_map=None):
    return _preprocess_common(df, stockid2idx, industry_map, desc="验证集特征工程", drop_small_open=True)


def calculate_ranking_metrics(y_pred, y_true, masks, k=5):
    metrics = {
        'pred_return_sum': 0.0,
        'max_return_sum': 0.0,
        'random_return_sum': 0.0,
        'ratio_pred': 0.0,
        'ratio_random': 0.0,
        'final_score': 0.0
    }

    pred_return_sum_list = []
    max_return_sum_list = []
    random_return_sum_list = []
    ratio_pred_list = []
    ratio_random_list = []
    final_score_list = []

    batch_size = y_pred.size(0)

    for i in range(batch_size):
        mask = masks[i]
        valid_indices = mask.nonzero().squeeze()

        if valid_indices.numel() < k:
            continue

        valid_pred = y_pred[i][valid_indices]
        valid_true = y_true[i][valid_indices]

        actual_k = min(k, valid_pred.numel())

        _, pred_indices = torch.topk(valid_pred, actual_k)
        pred_top_returns = valid_true[pred_indices]
        pred_return_sum = pred_top_returns.sum().item()

        _, true_indices = torch.topk(valid_true, actual_k)
        true_top_returns = valid_true[true_indices]
        max_return_sum = true_top_returns.sum().item()

        random_return_sum = actual_k * valid_true.mean().item()

        ratio_pred = pred_return_sum / (max_return_sum + 1e-9) if abs(max_return_sum) > 1e-9 else 0.0
        ratio_random = random_return_sum / (max_return_sum + 1e-9) if abs(max_return_sum) > 1e-9 else 0.0

        denominator = max_return_sum - random_return_sum
        final_score = (pred_return_sum - random_return_sum) / (denominator + 1e-6) if abs(denominator) > 1e-6 else 0.0

        pred_return_sum_list.append(pred_return_sum)
        max_return_sum_list.append(max_return_sum)
        random_return_sum_list.append(random_return_sum)
        ratio_pred_list.append(ratio_pred)
        ratio_random_list.append(ratio_random)
        final_score_list.append(final_score)

    if len(pred_return_sum_list) > 0:
        metrics['pred_return_sum'] = np.mean(pred_return_sum_list)
        metrics['max_return_sum'] = np.mean(max_return_sum_list)
        metrics['random_return_sum'] = np.mean(random_return_sum_list)
        metrics['ratio_pred'] = np.mean(ratio_pred_list)
        metrics['ratio_random'] = np.mean(ratio_random_list)
        metrics['final_score'] = np.mean(final_score_list)

    return metrics


class RankingDataset(Dataset):
    def __init__(self, sequences, targets, relevance_scores, stock_indices, industry_ids):
        self.sequences = sequences
        self.targets = targets
        self.relevance_scores = relevance_scores
        self.stock_indices = stock_indices
        self.industry_ids = industry_ids

    def __getitem__(self, idx):
        arr_seq = np.array(self.sequences[idx], dtype=np.float32)
        arr_tgt = np.array(self.targets[idx], dtype=np.float32)
        arr_rel = np.array(self.relevance_scores[idx], dtype=np.int64)
        arr_stk = np.array(self.stock_indices[idx], dtype=np.int64)
        arr_ind = np.array(self.industry_ids[idx], dtype=np.int64)
        return {
            'sequences': torch.from_numpy(arr_seq),
            'targets': torch.from_numpy(arr_tgt),
            'relevance': torch.from_numpy(arr_rel),
            'stock_indices': torch.from_numpy(arr_stk),
            'industry_ids': torch.from_numpy(arr_ind)
        }

    def __len__(self):
        return len(self.sequences)


def collate_fn(batch):
    sequences = [item['sequences'] for item in batch]
    targets = [item['targets'] for item in batch]
    relevance = [item['relevance'] for item in batch]
    stock_indices = [item['stock_indices'] for item in batch]
    industry_ids = [item['industry_ids'] for item in batch]
    max_stocks = max(s.size(0) for s in sequences)
    seq_len = sequences[0].size(1)
    feat_dim = sequences[0].size(2)
    pad_seq, pad_tgt, pad_rel, pad_stk, pad_ind, masks = [], [], [], [], [], []
    for seq, tgt, rel, stk, ind in zip(sequences, targets, relevance, stock_indices, industry_ids):
        n = seq.size(0)
        pad_s = torch.zeros(max_stocks - n, seq_len, feat_dim)
        pad_t = torch.zeros(max_stocks - n)
        pad_r = torch.zeros(max_stocks - n, dtype=torch.long)
        pad_st = torch.zeros(max_stocks - n, dtype=torch.long)
        pad_in = torch.zeros(max_stocks - n, dtype=torch.long)
        seq = torch.cat([seq, pad_s], dim=0)
        tgt = torch.cat([tgt, pad_t], dim=0)
        rel = torch.cat([rel, pad_r], dim=0)
        stk = torch.cat([stk, pad_st], dim=0)
        ind = torch.cat([ind, pad_in], dim=0)
        mask = torch.ones(max_stocks)
        mask[n:] = 0
        pad_seq.append(seq)
        pad_tgt.append(tgt)
        pad_rel.append(rel)
        pad_stk.append(stk)
        pad_ind.append(ind)
        masks.append(mask)
    return {
        'sequences': torch.stack(pad_seq),
        'targets': torch.stack(pad_tgt),
        'relevance': torch.stack(pad_rel),
        'stock_indices': torch.stack(pad_stk),
        'industry_ids': torch.stack(pad_ind),
        'masks': torch.stack(masks)
    }


class LazyRankingDataset(Dataset):
    def __init__(self, batch_files, features_dim, seq_len, cache_size, temp_dir):
        self.batch_files = batch_files
        self.features_dim = features_dim
        self.seq_len = seq_len
        self.cache_size = cache_size
        self.temp_dir = temp_dir
        self.cache = {}
        self.cache_order = []
        self.batch_lengths = []
        self.cumsum = [0]
        print("正在扫描批次文件（读取元数据）...")
        for f in tqdm(batch_files, desc="扫描批次"):
            meta_file = self.temp_dir / f"{f.stem}_meta.pkl"
            if meta_file.exists():
                try:
                    meta = joblib.load(meta_file)
                    num_samples = meta.get("num_samples", 0)
                    if num_samples <= 0:
                        print(f"跳过无效批次: {f.name} (样本数为0)")
                        continue
                except Exception as e:
                    print(f"跳过损坏meta: {f.name} ({e})")
                    continue
            else:
                try:
                    data = joblib.load(f)
                    num_samples = len(data[0])
                    del data
                    gc.collect()
                except Exception as e:
                    print(f"跳过损坏批次 {f.name}: {e}")
                    continue
            self.batch_lengths.append(num_samples)
            self.cumsum.append(self.cumsum[-1] + num_samples)
        self.total = self.cumsum[-1]
        print(f"总样本数（天数）: {self.total}")
        print(f"有效批次文件数: {len(self.batch_lengths)} / {len(batch_files)}")

    def _load_batch(self, batch_idx):
        if batch_idx in self.cache:
            return self.cache[batch_idx]
        batch_file = self.batch_files[batch_idx]
        try:
            data = joblib.load(batch_file)
            if len(data) == 6:
                seq_list, tgt_list, rel_list, idx_list, ind_list, _ = data
            elif len(data) == 5:
                seq_list, tgt_list, rel_list, idx_list, ind_list = data
            else:
                return None
            cached_data = (seq_list, tgt_list, rel_list, idx_list, ind_list)
            if len(self.cache) >= self.cache_size:
                old_key = self.cache_order.pop(0)
                old_data = self.cache.pop(old_key)
                del old_data
            self.cache[batch_idx] = cached_data
            self.cache_order.append(batch_idx)
            return cached_data
        except Exception as e:
            print(f"加载批次 {batch_idx} 失败: {e}")
            return None

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        batch_idx = bisect.bisect_right(self.cumsum, idx) - 1
        if batch_idx < 0:
            batch_idx = 0
        sample_idx = idx - self.cumsum[batch_idx]
        data = self._load_batch(batch_idx)
        if data is None:
            return self._empty_sample()
        seq_list, tgt_list, rel_list, idx_list, ind_list = data
        single_seq = np.array(seq_list[sample_idx], dtype=np.float32)
        single_tgt = np.array(tgt_list[sample_idx], dtype=np.float32)
        single_rel = np.array(rel_list[sample_idx], dtype=np.int64)
        single_idx = np.array(idx_list[sample_idx], dtype=np.int64)
        single_ind = np.array(ind_list[sample_idx], dtype=np.int64)
        return {
            'sequences': torch.from_numpy(single_seq),
            'targets': torch.from_numpy(single_tgt),
            'relevance': torch.from_numpy(single_rel),
            'stock_indices': torch.from_numpy(single_idx),
            'industry_ids': torch.from_numpy(single_ind)
        }

    def _empty_sample(self):
        return {
            'sequences': torch.zeros(1, self.seq_len, self.features_dim),
            'targets': torch.zeros(1),
            'relevance': torch.zeros(1, dtype=torch.long),
            'stock_indices': torch.zeros(1, dtype=torch.long),
            'industry_ids': torch.zeros(1, dtype=torch.long)
        }


def lazy_collate_fn(batch):
    if len(batch) == 0:
        return {
            'sequences': torch.zeros(1, 1, 1),
            'targets': torch.zeros(1, 1),
            'relevance': torch.zeros(1, 1, dtype=torch.long),
            'stock_indices': torch.zeros(1, 1, dtype=torch.long),
            'industry_ids': torch.zeros(1, 1, dtype=torch.long),
            'masks': torch.zeros(1, 1)
        }
    B = len(batch)
    max_stocks = max(item['sequences'].size(0) for item in batch)
    seq_len = batch[0]['sequences'].size(1)
    feature_dim = batch[0]['sequences'].size(2)

    sequences = torch.zeros(B, max_stocks, seq_len, feature_dim)
    targets = torch.zeros(B, max_stocks)
    relevance = torch.zeros(B, max_stocks, dtype=torch.long)
    stock_indices = torch.zeros(B, max_stocks, dtype=torch.long)
    industry_ids = torch.zeros(B, max_stocks, dtype=torch.long)
    masks = torch.zeros(B, max_stocks)

    for i, item in enumerate(batch):
        n = item['sequences'].size(0)
        sequences[i, :n] = item['sequences']
        targets[i, :n] = item['targets']
        relevance[i, :n] = item['relevance']
        stock_indices[i, :n] = item['stock_indices']
        industry_ids[i, :n] = item['industry_ids']
        masks[i, :n] = 1.0

    return {
        'sequences': sequences,
        'targets': targets,
        'relevance': relevance,
        'stock_indices': stock_indices,
        'industry_ids': industry_ids,
        'masks': masks
    }


# ============================================================
# 🔥 训练循环（含 EMA 支持）
# ============================================================
def train_ranking_model(model, dataloader, criterion, optimizer, scheduler, device, epoch, writer, ema=None):
    model.train()
    scaler = GradScaler(enabled=config['fp16_support'])

    total_loss = 0.0
    total_metrics = {
        'pred_return_sum': 0.0,
        'max_return_sum': 0.0,
        'random_return_sum': 0.0,
        'ratio_pred': 0.0,
        'ratio_random': 0.0,
        'final_score': 0.0
    }
    topk = config['topk_metric']
    local_step = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch + 1}")):
        industry_ids = batch['industry_ids'].to(device, non_blocking=True)
        sequences = batch['sequences'].to(device, non_blocking=True)
        targets = batch['targets'].to(device, non_blocking=True)
        masks = batch['masks'].to(device, non_blocking=True)
        stock_indices = batch['stock_indices'].to(device, non_blocking=True)

        valid_count = masks.sum(dim=1)
        if valid_count.sum() == 0:
            continue

        valid_mask = valid_count >= 2
        if valid_mask.sum() == 0:
            continue

        optimizer.zero_grad()

        with autocast(enabled=config['fp16_support']):
            scores, pred_ret, pred_cls, pred_vol, ind_emb = model(sequences, masks, stock_indices, industry_ids)

            batch_size = sequences.shape[0]
            loss = 0.0
            for b in range(batch_size):
                mask_b = masks[b] == 1
                if mask_b.sum() < 2:
                    continue
                s_b = scores[b][mask_b].unsqueeze(0)
                t_b = targets[b][mask_b].unsqueeze(0)
                loss += criterion(s_b, t_b)

            if loss > 0:
                loss = loss / batch_size
                batch_total = loss
            else:
                continue

        if batch_total.item() > 1e-8:
            scaler.scale(batch_total).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # 🔥 更新 EMA
            if ema is not None:
                ema.update()

            total_loss += batch_total.item()

            masked_scores = scores * masks + (1 - masks) * (-1e9)
            with torch.no_grad():
                metrics = calculate_ranking_metrics(masked_scores, targets, masks, k=topk)
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0.0) + v
            local_step += 1

            if writer is not None and local_step % 50 == 0:
                gs = epoch * len(dataloader) + local_step
                writer.add_scalar('train/loss', batch_total.item(), gs)
                for k, v in metrics.items():
                    writer.add_scalar(f'train/{k}', v, gs)

        del scores, pred_ret, pred_cls, pred_vol, masked_scores, batch_total

    if local_step > 0:
        for k in total_metrics:
            total_metrics[k] /= local_step
    avg_loss = total_loss / local_step if local_step > 0 else 0.0

    print(f"\n【Train Epoch {epoch + 1}】Avg Loss: {avg_loss:.4f} | FinalScore: {total_metrics['final_score']:.4f}")
    return avg_loss, total_metrics


def evaluate_ranking_model(model, val_loader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0.0
    total_metrics = {
        'pred_return_sum': 0.0,
        'max_return_sum': 0.0,
        'random_return_sum': 0.0,
        'ratio_pred': 0.0,
        'ratio_random': 0.0,
        'final_score': 0.0
    }
    num_batches = 0
    topk = config['topk_metric']

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Evaluating Epoch {epoch + 1}", disable=True):
            industry_ids = batch['industry_ids'].to(device, non_blocking=True)
            sequences = batch['sequences'].to(device, non_blocking=True)
            targets = batch['targets'].to(device, non_blocking=True)
            masks = batch['masks'].to(device, non_blocking=True)
            stock_indices = batch['stock_indices'].to(device, non_blocking=True)

            scores, _, _, _, _ = model(sequences, masks, stock_indices, industry_ids)
            masked_outputs = scores * masks + (1 - masks) * (-1e9)

            batch_loss = 0.0
            valid_sample_count = 0

            for b in range(masks.size(0)):
                valid_indices = (masks[b] == 1).nonzero().squeeze()
                if valid_indices.numel() < 2:
                    continue

                v_scores = masked_outputs[b][valid_indices].unsqueeze(0)
                v_targets = targets[b][valid_indices].unsqueeze(0)

                if torch.isnan(v_scores).any() or torch.isnan(v_targets).any():
                    continue

                loss = criterion(v_scores, v_targets)
                batch_loss += loss.item()
                valid_sample_count += 1

            if valid_sample_count > 0:
                total_loss += batch_loss / valid_sample_count

            metrics = calculate_ranking_metrics(masked_outputs, targets, masks, k=topk)
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            num_batches += 1

            del scores, masked_outputs
            torch.cuda.empty_cache()
            gc.collect()

    if num_batches == 0:
        return 0.0, total_metrics

    avg_loss = total_loss / num_batches if num_batches else 0.0
    for k in total_metrics:
        total_metrics[k] /= num_batches

    if writer:
        writer.add_scalar('eval/loss', avg_loss, global_step=epoch)
        for k, v in total_metrics.items():
            writer.add_scalar(f'eval/{k}', v, global_step=epoch)

    print(f"【Val Epoch {epoch + 1}】Avg Loss: {avg_loss:.4f} | FinalScore: {total_metrics['final_score']:.4f}\n")
    return avg_loss, total_metrics


# ============================================================
# 🔥 改进的滚动窗口划分（验证集增大）
# ============================================================
def walk_forward_split(df, seq_len, train_window=450, val_window=90, roll_step=20):
    df['日期'] = pd.to_datetime(df['日期'])
    all_dates = sorted(df['日期'].unique())
    total_days = len(all_dates)
    splits = []
    start = 0
    while start + train_window + val_window <= total_days:
        train_end = start + train_window
        val_end = train_end + val_window
        val_context_start = train_end
        train_dates = all_dates[start:train_end]
        val_context_dates = all_dates[val_context_start:val_end]
        train_df = df[df['日期'].isin(train_dates)]
        val_df = df[df['日期'].isin(val_context_dates)]
        val_start_date = all_dates[train_end]

        print(f"\n  split {len(splits)}: 训练集日期 {train_dates[0]} ~ {train_dates[-1]}")
        print(f"    训练集股票数: {train_df['股票代码'].nunique()}")
        print(f"    训练集行数: {len(train_df)}")
        if len(val_df) < 30:
            start += roll_step
            continue
        splits.append((train_df, val_df, val_start_date))
        start += roll_step
    return splits


# ============================================================
# 主程序
# ============================================================
def main():
    mp.set_start_method('spawn', force=True)
    set_seed(config['seed'])
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log'))

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("使用CPU")

    data_file = os.path.join(config['data_path'], config['data_file'])
    full_df = pd.read_csv(data_file, dtype={'股票代码': str})
    full_df['日期'] = pd.to_datetime(full_df['日期'])
    full_df = full_df[full_df['日期'] >= '2023-01-01'].copy()
    print(f"使用数据范围: {full_df['日期'].min()} ~ {full_df['日期'].max()}")
    all_stock_ids = sorted(full_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(all_stock_ids)}
    num_stocks = len(stockid2idx)
    joblib.dump(stockid2idx, os.path.join(output_dir, 'stockid2idx.pkl'))
    print(f"全局股票总数 {num_stocks}，映射已保存")

    # 行业映射加载
    import csv
    industry_csv_path = os.path.join(config['data_path'], "industry_map.csv")
    industry_map = {}

    with open(industry_csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = str(row["stock_code"]).zfill(6)
            ind_id = int(row["industry_id"])
            industry_map[code] = ind_id

    if len(industry_map) == 0:
        raise ValueError("行业映射加载失败，请检查 industry_map.csv 文件格式")

    all_ind_ids = list(industry_map.values())
    num_industries = max(all_ind_ids) + 1
    joblib.dump(industry_map, os.path.join(output_dir, "industry_map.pkl"))
    joblib.dump(num_industries, os.path.join(output_dir, "num_industries.pkl"))
    print(f"行业映射加载完成，共 {len(industry_map)} 只股票，总行业数：{num_industries}")

    temp_dir = Path(config['temp_dir'])
    temp_dir.mkdir(parents=True, exist_ok=True)
    splits = walk_forward_split(full_df, config['sequence_length'], train_window=450, val_window=90)
    print(f"生成滚动窗口数量: {len(splits)}")

    best_avg_score = -float('inf')
    total_epochs = config['num_epochs']

    for split_idx, (train_df, val_df, val_start) in enumerate(splits):
        print(f"\n===== 滚动窗口 {split_idx + 1}/{len(splits)} =====")

        # 预处理原始数据
        train_data, features = preprocess_data(train_df, is_train=True, stockid2idx=stockid2idx,
                                               industry_map=industry_map)
        val_data_full, _ = preprocess_val_data(val_df, stockid2idx=stockid2idx, industry_map=industry_map)

        # 标准化
        all_features = features
        feature_cols_only = [c for c in all_features if c != 'instrument']

        scaler = StandardScaler()
        train_feat = train_data[feature_cols_only].copy()
        train_feat = train_feat.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
        train_data[feature_cols_only] = scaler.fit_transform(train_feat)
        joblib.dump(scaler, os.path.join(output_dir, f'split{split_idx}_scaler.pkl'))

        val_feat = val_data_full[feature_cols_only].copy()
        val_feat = val_feat.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
        val_data_full[feature_cols_only] = scaler.transform(val_feat)

        # 构建训练集
        train_sequences, train_targets, train_relevance, train_stock_idx, train_ind_list = create_ranking_dataset_multiprocess(
            train_data,
            feature_cols_only,
            config['sequence_length'],
            max_workers=1,
            return_dates=False
        )

        print(f"  训练集样本数: {len(train_sequences)}")
        if len(train_sequences) > 0:
            print(f"  第一个样本股票数量: {len(train_sequences[0])}")

        train_dataset = RankingDataset(
            train_sequences,
            train_targets,
            train_relevance,
            train_stock_idx,
            train_ind_list
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True
        )

        # 构建验证集
        val_sequences, val_targets, val_relevance, val_stock_idx, val_ind_list, val_dates = create_ranking_dataset_multiprocess(
            val_data_full, feature_cols_only, config['sequence_length'], max_workers=1, return_dates=True
        )

        filtered_idx = [i for i, dt in enumerate(val_dates) if dt >= val_start]
        if len(filtered_idx) > 0:
            val_sequences = [val_sequences[i] for i in filtered_idx]
            val_targets = [val_targets[i] for i in filtered_idx]
            val_relevance = [val_relevance[i] for i in filtered_idx]
            val_stock_idx = [val_stock_idx[i] for i in filtered_idx]
            val_ind_list = [val_ind_list[i] for i in filtered_idx]

        val_dataset = RankingDataset(val_sequences, val_targets, val_relevance, val_stock_idx, val_ind_list)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )

        # 模型初始化
        model = GNNStockRankingModel(
            input_dim=len(feature_cols_only),
            config=config,
            num_stocks=num_stocks,
            num_industries=num_industries
        )
        model.to(device)
        criterion = WeightedRankingLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                      weight_decay=config.get('weight_decay', 1e-5))

        # 🔥 使用 CosineAnnealingWarmRestarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6,
        )

        # 🔥 初始化 EMA
        ema = EMA(model, decay=0.999)

        split_best_score = -float('inf')

        # ============================================================
        # 🔥 Early Stopping
        # ============================================================
        best_score = -float('inf')
        best_epoch = 0
        patience = 10
        patience_counter = 0
        early_stop_triggered = False

        for epoch in range(total_epochs):
            # 训练
            train_loss, train_metrics = train_ranking_model(
                model, train_loader, criterion, optimizer, scheduler, device, epoch, writer, ema
            )

            # 验证（使用 EMA 模型）
            ema.apply_shadow()
            eval_loss, eval_metrics = evaluate_ranking_model(
                model, val_loader, criterion, device, writer, epoch
            )
            ema.restore()

            cur_score = eval_metrics.get('final_score', 0)

            # Early Stopping 判断
            if cur_score > best_score:
                best_score = cur_score
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(output_dir, f"split{split_idx}_best.pth"))
                print(f"  ✅ 新最佳 | Epoch {best_epoch} | FS: {best_score:.4f}")
            else:
                patience_counter += 1
                print(
                    f"  ⏳ Patience: {patience_counter}/{patience} | 当前 FS: {cur_score:.4f} | 最佳 FS: {best_score:.4f}")

            # 早停
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping at epoch {epoch + 1}")
                print(f"📌 最佳 Epoch: {best_epoch} | 最佳 FS: {best_score:.4f}")
                early_stop_triggered = True
                break

        if not early_stop_triggered:
            print(f"\n✅ Split {split_idx} 完成 | 最佳 Epoch: {best_epoch} | 最佳 FS: {best_score:.4f}")
        else:
            print(
                f"\n✅ Split {split_idx} 早停于 Epoch {epoch + 1} | 最佳 Epoch: {best_epoch} | 最佳 FS: {best_score:.4f}")

        del train_dataset, train_sequences, train_targets, train_relevance, train_stock_idx, train_ind_list
        gc.collect()
        torch.cuda.empty_cache()

        # 本窗口的最好成绩（修正：原先漏了这一步赋值，导致 split_best_score 恒为 -inf）
        split_best_score = best_score

        if split_best_score > best_avg_score:
            best_avg_score = split_best_score
            best_ckpt_path = os.path.join(output_dir, f"split{split_idx}_best.pth")
            best_state = torch.load(best_ckpt_path, map_location='cpu')
            torch.save(best_state, os.path.join(output_dir, "best_model.pth"))
            joblib.dump(split_idx, os.path.join(output_dir, "best_split_idx.pkl"))
            with open(os.path.join(output_dir, "best_score.txt"), "w", encoding="utf-8") as f:
                f.write(f"{split_best_score:.6f}")
            print(f"🏆 全局最佳窗口更新: split{split_idx} | FS: {split_best_score:.4f}")

        gc.collect()

    writer.close()
    return best_avg_score


if __name__ == "__main__":
    best_score = main()
    print(f"########## 训练结束，最优评分: {best_score:.4f} ##########")