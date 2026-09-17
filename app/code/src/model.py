import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch_geometric.nn import GATConv, HeteroConv
from config import config

NEG_INF = -1e2
from config import feature_columns_map


# ============================================================
# 位置编码模块
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ============================================================
# VCTimeAttn（时序注意力）
# ============================================================
class VCTimeAttn(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.1):
        super().__init__()
        self.time_score_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
        self.softmax = nn.Softmax(dim=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        bsn, T, C = x.shape
        time_weight = self.time_score_net(x)
        time_weight = self.softmax(time_weight)
        weighted_x = x * time_weight
        out = torch.sum(weighted_x, dim=1)
        return self.drop(out)


# ============================================================
# 改进的 ScoreHead（含残差连接）
# ============================================================
class ScoreHead(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model // 2, d_model // 4)
        self.fc2 = nn.Linear(d_model // 4, 1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout * 0.5)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # 残差连接：使用全局平均池化作为残差
        residual = x.mean(dim=-1, keepdim=True) * 0.1
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x + residual


# ============================================================
# EMA 辅助类（用于训练）
# ============================================================
class EMA:
    """指数移动平均模型"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# ============================================================
# 主模型
# ============================================================
class GNNStockRankingModel(nn.Module):
    def __init__(self, input_dim, config, num_stocks, num_industries):
        super(GNNStockRankingModel, self).__init__()
        self.config = config
        self.num_stocks = num_stocks
        self.num_industries = num_industries
        self.k_neighbors = config.get('k_neighbors', 10)
        self.gat_heads = config.get('gat_heads', 2)
        self.industry_dim = config['industry_dim']
        self.d_model = config['d_model']
        self.combined_dim = self.d_model + self.industry_dim

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, self.d_model * 2),
            nn.LayerNorm(self.d_model * 2),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model)
        )

        self.pos_encoder = PositionalEncoding(self.d_model, config['dropout'])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])
        self.vc_att = VCTimeAttn(config['sequence_length'], self.d_model, config['dropout'])

        # 行业 Embedding
        self.industry_emb = nn.Embedding(num_industries, self.industry_dim)

        # HeteroGAT
        edge_types = [
            ("stock", "price_sim", "stock"),
            ("stock", "same_industry", "stock"),
        ]
        conv_dict = {}
        for et in edge_types:
            head_dim = self.d_model // self.gat_heads
            conv_dict[et] = GATConv(
                in_channels=self.combined_dim,
                out_channels=head_dim,
                heads=self.gat_heads,
                dropout=config['dropout'],
                concat=True
            )
        self.hetero_gat = HeteroConv(conv_dict, aggr='sum')
        self.gat_norm = nn.LayerNorm(self.combined_dim)
        self.raw_norm = nn.LayerNorm(self.d_model)

        # GAT门控融合
        self.gate_fuse = nn.Linear(self.d_model, self.d_model)

        # Ranking Layers
        self.ranking_layers = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.LayerNorm(self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(self.d_model // 2, self.d_model // 2),
            nn.LayerNorm(self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout'])
        )

        # 使用改进的 ScoreHead
        self.score_head = ScoreHead(self.d_model, config['dropout'])

        # 辅助头
        self.ret_head = nn.Linear(self.d_model, 1)
        self.cls_head = nn.Linear(self.d_model, 2)
        self.vol_head = nn.Linear(self.d_model, 1)
        self.head_fuse = nn.Linear(self.d_model, self.d_model)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def _build_knn_edge_index(self, x, k):
        N = x.size(0)
        if N <= 1:
            return self._build_full_edge_index(N, x.device)
        if N <= k:
            return self._build_full_edge_index(N, x.device)

        x_norm = F.normalize(x, p=2, dim=1)
        sim_matrix = torch.mm(x_norm, x_norm.t())
        sim_matrix = sim_matrix - torch.eye(N, device=x.device) * 1e9
        actual_k = min(k, N - 1)
        topk_sim, topk_indices = torch.topk(sim_matrix, actual_k, dim=1)

        row = torch.arange(N, device=x.device).repeat_interleave(actual_k)
        col = topk_indices.flatten()
        valid_mask = topk_sim.flatten() > -1e8
        row = row[valid_mask]
        col = col[valid_mask]
        return torch.stack([row, col], dim=0)

    def _build_full_edge_index(self, n, device):
        idx = torch.arange(n, device=device)
        row = idx.repeat(n)
        col = idx.repeat_interleave(n)
        mask = row != col
        return torch.stack([row[mask], col[mask]], dim=0)

    def forward(self, src, masks=None, stock_indices=None, industry_ids=None):
        batch_size, num_stocks, seq_len, feature_dim = src.size()
        if masks is None:
            masks = torch.ones(batch_size, num_stocks, device=src.device)
        if stock_indices is None:
            stock_indices = torch.zeros(batch_size, num_stocks, dtype=torch.long, device=src.device)

        # ========== 1. 时序编码 ==========
        src_reshaped = src.view(batch_size * num_stocks, seq_len, feature_dim)
        src_proj = self.input_proj(src_reshaped)
        src_proj = self.pos_encoder(src_proj)

        stock_valid_mask = masks.view(-1).bool()
        pad_mask = ~stock_valid_mask.unsqueeze(1).repeat(1, seq_len)
        temporal_out = self.temporal_encoder(src_proj, src_key_padding_mask=pad_mask)
        vc_feat = self.vc_att(temporal_out)
        vc_feat = vc_feat.view(batch_size, num_stocks, -1)

        # ========== 2. 行业嵌入 ==========
        max_ind_idx = self.num_industries - 1
        ind_clamped = torch.clamp(industry_ids, 0, max_ind_idx)
        industry_emb_all = self.industry_emb(ind_clamped)
        stock_features = torch.cat([vc_feat, industry_emb_all], dim=-1)

        # ========== 3. 批量 GAT ==========
        valid_counts = masks.sum(dim=1).long()
        total_nodes = valid_counts.sum().item()

        if total_nodes == 0:
            scores = torch.full((batch_size, num_stocks), NEG_INF, device=src.device)
            pred_ret = torch.zeros(batch_size, num_stocks, device=src.device)
            pred_cls = torch.zeros(batch_size, num_stocks, 2, device=src.device)
            pred_vol = torch.zeros(batch_size, num_stocks, device=src.device)
            return scores, pred_ret, pred_cls, pred_vol, industry_emb_all

        node_offsets = torch.zeros(batch_size, dtype=torch.long, device=src.device)
        node_offsets[1:] = torch.cumsum(valid_counts[:-1], dim=0)

        all_x_list = []
        all_ind_list = []
        for b in range(batch_size):
            valid_mask = masks[b] == 1
            all_x_list.append(stock_features[b][valid_mask])
            all_ind_list.append(industry_ids[b][valid_mask])

        all_x = torch.cat(all_x_list, dim=0)
        all_ind = torch.cat(all_ind_list, dim=0)

        # 构建边
        price_edges_list = []
        industry_edges_list = []

        for b in range(batch_size):
            n_valid = valid_counts[b].item()
            if n_valid <= 1:
                continue

            offset = node_offsets[b].item()
            start = offset
            end = offset + n_valid

            x_b = all_x[start:end, :self.d_model]
            ind_b = all_ind[start:end]

            # 1. price KNN边
            if n_valid <= self.k_neighbors:
                idx = torch.arange(n_valid, device=x_b.device)
                row = idx.repeat(n_valid)
                col = idx.repeat_interleave(n_valid)
                mask = row != col
                edge_price = torch.stack([row[mask], col[mask]], dim=0)
            else:
                x_norm = F.normalize(x_b, p=2, dim=1)
                sim_matrix = torch.mm(x_norm, x_norm.t())
                sim_matrix = sim_matrix - torch.eye(n_valid, device=x_b.device) * 1e9
                actual_k = min(self.k_neighbors, n_valid - 1)
                topk_sim, topk_indices = torch.topk(sim_matrix, actual_k, dim=1)

                row = torch.arange(n_valid, device=x_b.device).repeat_interleave(actual_k)
                col = topk_indices.flatten()
                valid_mask_edge = topk_sim.flatten() > -1e8
                row = row[valid_mask_edge]
                col = col[valid_mask_edge]
                edge_price = torch.stack([row, col], dim=0)

            price_edges_list.append(edge_price + offset)

            # 2. 行业边
            unique_ind, inv = torch.unique(ind_b, return_inverse=True)
            row = inv.unsqueeze(1).repeat(1, len(inv)).flatten()
            col = inv.unsqueeze(0).repeat(len(inv), 1).flatten()
            mask = row != col
            edge_industry = torch.stack([row[mask], col[mask]], dim=0)
            if edge_industry.shape[1] > 0:
                industry_edges_list.append(edge_industry + offset)

        all_price_edges = torch.cat(price_edges_list, dim=1) if len(price_edges_list) > 0 else torch.zeros(2, 0, dtype=torch.long, device=src.device)
        all_industry_edges = torch.cat(industry_edges_list, dim=1) if len(industry_edges_list) > 0 else torch.zeros(2, 0, dtype=torch.long, device=src.device)

        graph_dict = {
            ("stock", "price_sim", "stock"): all_price_edges,
            ("stock", "same_industry", "stock"): all_industry_edges,
        }

        out_dict = self.hetero_gat(x_dict={"stock": all_x}, edge_index_dict=graph_dict)
        hetero_out = out_dict["stock"]

        gate_weight = torch.sigmoid(self.gate_fuse(hetero_out))
        fused_temporal = gate_weight * hetero_out[:, :self.d_model] + (1 - gate_weight) * all_x[:, :self.d_model]
        combined = torch.cat([fused_temporal, all_x[:, self.d_model:]], dim=-1)
        x_gat = self.gat_norm(combined)

        # 回填特征
        gat_outputs = []
        for b in range(batch_size):
            n_valid = valid_counts[b].item()
            offset = node_offsets[b].item()
            full_out = stock_features[b].clone()
            if n_valid > 0:
                valid_mask = masks[b] == 1
                full_out[valid_mask, :self.d_model] = x_gat[offset:offset + n_valid, :self.d_model]
            gat_outputs.append(full_out)

        interactive = torch.stack(gat_outputs, dim=0)

        # ========== 4. 排序打分 ==========
        inter_feat = interactive[..., :self.d_model]
        inter_flat = inter_feat.view(batch_size * num_stocks, self.d_model)
        rank_feat = self.ranking_layers(inter_flat)
        scores = self.score_head(rank_feat).view(batch_size, num_stocks)

        pred_ret = self.ret_head(inter_flat).view(batch_size, num_stocks)
        pred_cls = self.cls_head(inter_flat).view(batch_size, num_stocks, 2)
        pred_vol = self.vol_head(inter_flat).view(batch_size, num_stocks)

        # 直接返回 scores（不填充 NEG_INF）
        del all_x, all_price_edges, all_industry_edges, graph_dict

        return scores, pred_ret, pred_cls, pred_vol, industry_emb_all