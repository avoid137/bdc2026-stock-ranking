# 大数据挑战赛 - 股票排序预测方案

> 本方案基于赛事官方 baseline（THU-BDC2026）改造，逐文件归属见仓库根目录 `NOTICE.md`。

## 环境配置

- Python: 3.10
- PyTorch: 2.7.1+cu126 / CUDA 12.6（开发环境实测）
- 依赖库: numpy, pandas, scikit-learn, tqdm, joblib, tensorboardX, torch-geometric, TA-Lib, baostock
- 目录不固定：`config.py` 以「本文件上两级目录」为项目根（即 `app/`），也可用环境变量 `APP_ROOT` 覆盖

## 数据

- 沪深300成分股日线，来源 baostock（免费公开接口）
- 原始数据范围: 2020-01-01 至 2026-07-01
- 实际训练只用 **2023-01-01 之后** 的数据（`train.py` / `test.py` 内均有该过滤）
- 数据文件：`app/model/data/stock_data.csv`（字段：股票代码, 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌额, 换手率, 涨跌幅）
- 行业映射：`app/model/data/industry_map.csv`，由 `industry_map.py` 调 baostock `query_stock_industry()` 生成
- 本仓库不含任何数据文件

## 预训练模型

- 无外部预训练模型，所有模型参数随机初始化

## 算法

### 整体思路

排序学习（Learning to Rank）框架：以「过去 60 个交易日的量价特征」为输入，输出每只股票的排序分数，
目标是选出未来 5 个交易日开盘价收益率最高的 5 只股票。

### 网络结构（`model.py` 的 `GNNStockRankingModel`）

1. **输入投影**：`Linear → LayerNorm → ReLU → Dropout → Linear → LayerNorm`，把 197 维特征投到 `d_model=128`
2. **时序编码**：位置编码 + `TransformerEncoder`（`num_layers=3`, `nhead=8`, `dim_feedforward=256`, `norm_first=True`）
3. **时序聚合**：`VCTimeAttn`——对时间步做加性打分后 softmax 加权求和，压成每只股票一个向量
4. **行业嵌入**：`nn.Embedding(num_industries, 16)`，与 128 维时序特征拼接 → 144 维节点特征
5. **股票间交互**：`HeteroConv` 异构图 + 两种边
   - `(stock, price_sim, stock)`：节点特征做 L2 归一化后算余弦相似度，取 **Top-10 KNN 稀疏图**（`O(N·k)`）
   - `(stock, same_industry, stock)`：同一行业内部全连接
   - 两个边类型各接一个 `GATConv(heads=2, concat=True)`
6. **门控融合**：`sigmoid(W·h_gat)` 作门，线性插值融合 GAT 输出与原始时序特征
7. **排序头**：两层 `Linear+LayerNorm+ReLU+Dropout` → `ScoreHead`（带均值残差支路）→ 标量分数
8. **辅助头**：`ret_head` / `cls_head` / `vol_head` 已定义并在 `forward` 中返回，但**当前训练损失未接入**（见「已知局限」）

相对官方 baseline 的关键点：baseline 用 `CrossStockAttention`（`nn.MultiheadAttention`，Q=K=V 同一张量，
即稠密自注意力，股票两两全部建边，`O(N²)`）；本方案换成「余弦 Top-10 KNN 稀疏图 + 同行业图」的异构图 GAT，
把上下文建边的复杂度降到 `O(N·k)`，并把行业先验显式注入。

### 损失函数

- `NDCG loss`（`temperature=0.7`, `k=5`）为主项
- `+ 0.3 × Pairwise`（sigmoid 形式的成对排序损失，Top-5 样本权重 2.0）
- `+ 0.1 × Margin`（Top-5 均值分数 − 末 5 名均值分数，margin=0.1）

> baseline 的 `WeightedRankingLoss` 是「listwise 交叉熵 + pairwise」，本方案把主项换成 NDCG，
> 对「Top-5 排序位置」更直接。

### 标签与防泄露

- 标签：`label = (open_t5 − open_t1) / open_t1`，即 T+1 开盘买入、T+5 开盘卖出的收益率
- 过滤：`open_t1 ≈ 0`（停牌/一字板）与 `|label| > 0.5`（异常值）的样本剔除
- **行业中性化**：`label ← label − 同日同行业均值`，去掉行业整体涨跌带来的共同项
- 标准化：`StandardScaler` 只在**当前训练窗口**上 `fit`，验证/推理窗口只 `transform`；
  每个滚动窗口单独保存 `splitN_scaler.pkl`，不会跨期借用统计量
- 特征工程（158 Alpha158 因子 + 39 技术指标）全部是**逐股票按时间顺序**滚动计算的，不使用未来数据

### 数据扩增 / 模型集成

- 无人工扩增，使用原始时序数据
- 单模型，无集成

### 算法的其他细节

- 特征工程: 158 个 Alpha158 同名算子 + 39 个技术指标（TA-Lib/自算），共 197 维
- 序列长度: 60 个交易日
- KNN 邻居数: 10；行业嵌入维度: 16

## 训练流程

1. 读取 `stock_data.csv`，过滤到 2023-01-01 之后
2. 加载 `industry_map.csv` 得到行业 ID
3. **滚动窗口切分**（`walk_forward_split`，`train_window=450` / `val_window=90` / `roll_step=20` 个交易日）
   —— 每个窗口「用前 450 天训练、紧接的 90 天验证」，窗口间不重叠未来信息
4. 特征工程 (197 维) → 按窗口 `fit` 标准化器 → 构建 `[样本数, 股票数, 60, 197]` 张量
5. 逐窗口训练 `GNNStockRankingModel`：AdamW（`lr=5e-5`, `weight_decay=1e-4`）+
   `CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6)` + EMA（decay 0.999）+
   梯度裁剪（`max_grad_norm=5.0`）+ 早停（`patience=10`，监控验证集 `final_score`）
6. 每个窗口保存自己的最优权重 `model/splitN_best.pth`；全局最优窗口另存为
   `model/best_model.pth` / `best_split_idx.pkl` / `best_score.txt`

## 推理流程

1. 读取 `stock_data.csv`，取最新交易日为预测基准日
2. 特征工程 → 用**与所选窗口配套的** `splitN_scaler.pkl` 标准化
3. 加载 `splitN_best.pth`，对全部股票打分
4. 分数降序取 Top-5
5. 权重生成：`softmax(score / 0.3)` → 截断到 0.3 → 归一化
6. 输出 `output/result_splitN.csv`（`stock_id, weight`）

窗口序号可用环境变量 `BEST_SPLIT` 指定；未指定时优先读 `best_split_idx.pkl`，读不到才回退默认值 7。

## 实测训练记录（自测口径，非官方榜分）

指标 `final_score = (pred_return_sum − random_return_sum) / (denominator)`，由 `train.py::calculate_ranking_metrics` 计算。

逐窗口早停结果（控制台日志实录，按 FS 降序）：

| Split | 最佳 Epoch | 最佳 FS | | Split | 最佳 Epoch | 最佳 FS |
|---|---|---|---|---|---|---|
| **8** | 8 | **0.1307** | | 12 | 12 | 0.0338 |
| 4 | 19 | 0.1291 | | 5 | 4 | −0.0029 |
| 0 | 17 | 0.1184 | | 11 | 5 | −0.0055 |
| 7 | 22 | 0.1108 | | 13 | 14 | −0.0168 |
| 9 | 25 | 0.1004 | | | | |
| 3 | 31 | 0.0997 | | | | |
| 10 | 13 | 0.0807 | | | | |
| 6 | 4 | 0.0709 | | | | |
| 2 | 2 | 0.0641 | | | | |
| 15 | 3 | 0.0517 | | | | |
| 14 | 7 | 0.0434 | | | | |
| 1 | 10 | 0.0381 | | | | |

**怎么读这张表**：所有窗口都触发了早停（patience=10），最优 epoch 普遍落在 2–31 之间；
但窗口之间 FS 从 **−0.0168 到 0.1307** 的跨度说明——**同一套模型在不同时间切分上表现极不稳定**，
单看某个窗口的数字没有意义，只能作为「流程跑通 + 相对趋势」的证据。这也是本方案不敢报稳定收益的原因。

**诚实声明**：`app/model/best_score.txt` 里的 `0.074503` 与上表任何一行都不吻合，
推测是另一次运行的残留（可能是不同超参/数据切片），**不作为成绩引用**；
tensorboard 事件文件在本机副本中为 0 字节，训练曲线已无法重建。

## 已知局限（不做超出代码事实的宣称）

- **辅助头未启用**：`ret_head` / `cls_head` / `vol_head` 有前向计算，但没有对应的损失项参与反传，实际不起作用
- **懒加载路径未接入主流程**：`LazyRankingDataset`、`create_ranking_dataset_streaming` 已实现但 `main()` 走的是
  `create_ranking_dataset_multiprocess` + `RankingDataset`；`config['temp_dir']` 目前只被创建、未被写入
- **AMP 未启用**：`fp16_support=False`，`autocast` / `GradScaler` 处于关闭状态
- 训练数据依赖 baostock 的可得性，接口调整会影响复现
- **滚动窗口划分有一处口径要留意**：`walk_forward_split` 里训练窗口与验证窗口的日期是**相邻不重叠**的，
  但验证样本的 60 日序列会向前借用训练窗口末尾的历史（这是必要的上下文，不是泄露）；
  真正要防的泄露是「标准化统计量跨期借用」，已用「每窗口单独 `fit` scaler」堵住

## 其他注意事项

- 显存/内存：单窗口训练集一次性构建后常驻内存，`batch_size=8`；如需更大数据量请改用 streaming 路径
- 可复现性：`set_seed(42)`，并固定 `PYTHONHASHSEED`
