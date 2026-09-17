# 归属声明 / Attribution Notice

## 1. 本仓库的性质

本仓库是 **2026 中国高校计算机大赛——大数据挑战赛**（主办：清华大学、大数据系统软件国家工程研究中心）
**赛事官方 baseline 的改造版**。

- 官方 baseline：`https://github.com/Sherlock1956/THU-BDC2026`
- 官方 baseline **未附带任何 LICENSE 文件**（默认「保留所有权利」），其著作权归赛事主办方所有
- 本仓库**只包含我改写过的文件与自写文件**；官方 baseline 中我未改动的文件（`get_stock_data.py`、
  `predict.py`、`pyproject.toml`、`uv.lock`、`README.md`、`GUIDE.md`、`asset/` 等）**没有复制进来**，
  需要时请从官方仓库获取

如果你只应引用一个版本的代码，请引用官方 baseline；本仓库的意义是「在它之上改了什么」。

## 2. 逐文件归属

| 文件 | 归属 | 说明 |
|---|---|---|
| `app/code/src/model.py` | **本人改写** | 官方 `StockTransformer`（`CrossStockAttention` 稠密自注意力）被整体替换为 `GNNStockRankingModel`：余弦 Top-10 KNN 稀疏图 + 同行业图 → `HeteroConv`/`GATConv`、行业嵌入、门控融合、含残差的 `ScoreHead`、`EMA` 类。`PositionalEncoding` 沿用官方实现 |
| `app/code/src/train.py` | **本人改写** | 保留官方的 `set_seed`、`preprocess_data`/`preprocess_val_data` 骨架、`WeightedRankingLoss` 类名、`RankingDataset`、`collate_fn`、`train_ranking_model`、`evaluate_ranking_model`、`calculate_ranking_metrics`；本人改动见下方第 3 节 |
| `app/code/src/utils.py` | **本人改写** | 官方特征工程（`engineer_features_158plus39` / `engineer_features_39`）与 `create_ranking_dataset_*` 保留；本人扩展：样本额外携带 `industry_ids`、`return_dates` 开关（返回每个样本对应的交易日）、新增 `create_ranking_dataset_streaming` |
| `app/code/src/config.py` | **本人改写** | 官方 `config.py` 基础上扩参（GAT / 行业 / 损失温度 / weight_decay / 路径），并新增 `feature_columns_map` |
| `app/code/src/test.py` | **本人自写** | 官方推理入口是 `predict.py` 及 `train.py` 内的 `predict_top_stocks`，本文件独立实现「取最新交易日 → 构建推理序列 → Top-5 → softmax 权重」 |
| `app/code/src/industry_map.py` | **本人自写** | 调 baostock 拉全市场行业分类并映射为整数 ID，产出 `industry_map.csv` |
| `Dockerfile` | **本人自写** | 官方 Dockerfile 是 `python:3.12-slim` + `uv sync`；本文件是 `python:3.10-slim` + 源码编译 TA-Lib + 本地 whl + 清华源分段安装 |
| `app/init.sh` `app/train.sh` `app/test.sh` | **本人自写** | 官方 `init.sh` 为空、`train.sh` 只有一行 `python code/src/train.py`、`test.sh` 调 `predict.py` |
| `app/requirements.txt` `app/readme.md` | **本人自写** | 官方依赖声明在 `pyproject.toml` / `uv.lock` |
| 赛题数据（`train.csv` / `test.csv` / 权重 / 提交结果） | **未包含** | 竞赛规程要求赛后销毁已下载数据，本仓库不含任何数据文件与模型权重 |

## 3. 相对官方 baseline 的改动清单（`code/src` 内）

### 3.1 模型（`model.py`，整体重写）

| 项目 | 官方 baseline | 本方案 |
|---|---|---|
| 股票间交互 | `CrossStockAttention`：`nn.MultiheadAttention(d_model, nhead)`，Q=K=V 同一张量 → 稠密自注意力，全对全建边 | `HeteroConv` 异构图：余弦 Top-10 KNN 稀疏图 + 同行业图，各接 `GATConv(heads=2, concat=True)` |
| 建边复杂度 | `O(N²)`（N = 当日股票数） | `O(N·k)`，`k=10` |
| 行业信息 | 无 | `nn.Embedding(num_industries, 16)`，拼到节点特征上 |
| 融合方式 | 残差 + LayerNorm | `sigmoid` 门控线性插值（GAT 输出 ↔ 原始时序特征） |
| 时序聚合 | `FeatureAttention`（`Linear→Tanh→Linear→Softmax`，作为 `Sequential`） | `VCTimeAttn`（同结构但独立模块、含 dropout 输出） |
| 输入投影 | 单层 `Linear` | 两段 `Linear + LayerNorm` 瓶颈结构 |
| 输出头 | 单一分数头 | `ScoreHead`（含均值残差）+ `ret_head`/`cls_head`/`vol_head` 三个辅助头（**未接入损失**） |
| 训练辅助 | 无 | `EMA` 类（decay 0.999） |

### 3.2 训练（`train.py`）

- **验证协议**：官方 `split_train_val_by_last_month`（按最后一个月切一刀）→ 本人 `walk_forward_split`
  （`train_window=450` / `val_window=90` / `roll_step=20`，滚动前推），每个窗口各存一份 `splitN_best.pth`
  （本机目录中为 `split0`–`split16` 共 17 份；控制台日志记录到 Split 15，其中 Split 8 最佳，FS = 0.1307）
- **损失主项**：官方 listwise 交叉熵 → 本人 `ndcg_loss`（`temperature=0.7`, `k=5`），
  组合为 `NDCG + 0.3×Pairwise + 0.1×Margin`
- **标签**：新增**行业中性化**（`label ← label − 同日同行业均值`）
- **标准化**：由全局一个 scaler → **每个滚动窗口单独 `fit`**，各自落盘 `splitN_scaler.pkl`
- **优化器/调度**：官方 `Adam` → 本人 `AdamW`（`weight_decay=1e-4`）+ `CosineAnnealingWarmRestarts`
- **训练技巧**：新增 EMA、早停（`patience=10`）、逐窗口最优窗口记录
- **数据范围**：显式过滤到 2023-01-01 之后

### 3.3 超参（`config.py`）

| 参数 | 官方 baseline | 本方案 |
|---|---|---|
| `d_model` | 256 | 128 |
| `nhead` | 4 | 8 |
| `num_layers` | 3 | **3（未改）** |
| `dim_feedforward` | 512 | 256 |
| `batch_size` | 4 | 8 |
| `learning_rate` | 1e-5 | 5e-5 |
| `dropout` | 0.1 | 0.45 |
| `weight_decay` | 无 | 1e-4 |
| `pairwise_weight` | 1 | 0.3 |
| `topk_metric` / `loss_temperature` | 无 / 无 | 5 / 0.7 |
| `top5_weight` / `base_weight` | 2.0 / 1.0 | 2.0 / 1.0（未改） |
| `sequence_length` / `feature_num` | 60 / `158+39` | 60 / `158+39`（未改） |

## 4. 为本仓库做的整理（与「当时训练跑的那份」的差异）

为了让代码能在别人的机器上跑起来，上传时做了以下**三处**修改，均已在代码注释中标明：

1. **路径去硬编码**：原 `config.py` / `test.py` / `industry_map.py` 里写死了
   `C:\Users\Administrator\Desktop\project-windows`，`train.py` 里写死了 `D:\project_temp\train_batches`。
   现改为以 `config.py` 上两级目录为项目根，并支持环境变量 `APP_ROOT` 覆盖
2. **`test.py` 模型与标准化器改为同源**：原代码加载 `split7_best.pth`，却用 `split8_scaler.pkl` 做标准化
   （窗口编号 0-based / 1-based 混用导致的 off-by-one）。现统一由 `BEST_SPLIT` / `best_split_idx.pkl` 决定，
   默认回退到 Split 8——即训练日志里 FS 最高的那个窗口
3. **`train.py` 补上最优窗口记录**：原代码 `split_best_score` 初始化后再未赋值，恒为 `-inf`，
   导致「全局最优窗口」永远不会随训练更新（`best_model.pth` / `best_split_idx.pkl` 停留在初始状态）。
   现改为每个窗口结束后更新 `split_best_score`，并把成绩写入 `best_score.txt`

因此本仓库代码**不等于**逐字节等于当时训练时运行的那份，也**没有重跑验证**上述修改；
如需引用任何成绩数字，请以自己重跑得到的训练日志为准。

## 5. 合规

- 竞赛规程要求「大赛结束之后，参赛者在拥有模型和代码的知识产权的情况下可自行选择公开分享」，
  且「完成比赛使用后应及时销毁已下载数据」——本仓库**不含数据、不含权重、不含提交结果**
- 本仓库**不主张**官方 baseline 部分的著作权；如主办方对本仓库的存在有异议，将立即下架
- 联系：`avoid137`（GitHub）
