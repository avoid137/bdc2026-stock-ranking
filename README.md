# 沪深300 排序选股 · 在赛事官方 baseline 上做图结构改造

> ⚠️ **归属声明**：本项目是 **2026 中国高校计算机大赛——大数据挑战赛** 赛事官方 baseline
> （[Sherlock1956/THU-BDC2026](https://github.com/Sherlock1956/THU-BDC2026)，**未附带 LICENSE**）的改造版。
> 官方 baseline 的著作权归赛事主办方所有。**逐文件归属、改动清单、以及「为本仓库做的整理」见 [`NOTICE.md`](NOTICE.md)。**
> 本仓库只有我改写和自写的文件，不含赛题数据、不含模型权重、不含提交结果。

## 这个项目在做什么

给定沪深300成分股过去 60 个交易日的量价数据，给每只股票打一个排序分，
选出未来 5 个交易日（T+1 开盘买入 → T+5 开盘卖出）收益率最高的 5 只股票，并给出权重。

技术路线是 Learning to Rank：**时序编码（Transformer）→ 股票间建立关系图（异构图 GAT）→ 排序打分**。

## 相对官方 baseline 改了什么

一句话概括：**把「股票两两全连接的稠密自注意力」换成「余弦 Top-10 KNN 稀疏图 + 同行业图」的异构图 GAT。**

| 维度 | 官方 baseline | 本方案 |
|---|---|---|
| 股票间交互 | `nn.MultiheadAttention`，Q=K=V 同一张量（稠密自注意力，全对全） | `HeteroConv` + `GATConv(heads=2)`，两种边：余弦 Top-10 KNN 稀疏图、同行业全连接 |
| 建边复杂度 | `O(N²)` | `O(N·k)`（k=10） |
| 行业先验 | 无 | 行业 `Embedding(16)` 拼进节点特征 |
| 融合方式 | 残差 + LayerNorm | sigmoid 门控插值 |
| 验证协议 | 按最后一个月切一刀 | **滚动窗口前推**（训练 450 天 / 验证 90 天 / 步长 20 天），本机跑了 17 个窗口 |
| 损失主项 | listwise 交叉熵 | **NDCG**（+ 0.3×Pairwise + 0.1×Margin） |
| 标签处理 | 收益率 + 异常值过滤 | 追加**行业中性化**（减去同日同行业均值） |
| 标准化 | 全局一个 scaler | **每个滚动窗口单独 fit**，不跨期借用统计量 |
| 优化器 | Adam | AdamW + `CosineAnnealingWarmRestarts` + EMA + 早停 |

超参对照表（`d_model` / `nhead` / `learning_rate` / `dropout` 等）见 `NOTICE.md` 第 3.3 节；
注意 `num_layers` 与 `sequence_length`、`feature_num` **未改动**。

## 代码结构

```
bdc2026-stock-ranking/
├── Dockerfile                  # 自写：python3.10 + 源码编译 TA-Lib + 本地 whl + 清华源分段安装
├── README.md
├── NOTICE.md                   # 归属声明 / 逐文件来源 / 改动清单
└── app/
    ├── init.sh  train.sh  test.sh
    ├── requirements.txt
    ├── readme.md               # 方案说明（网络结构、训练流程、推理流程、已知局限）
    └── code/src/
        ├── config.py           # 全部超参与路径（路径支持 APP_ROOT 环境变量覆盖）
        ├── model.py            # GNNStockRankingModel：时序编码 + 异构图 GAT + 排序头
        ├── train.py            # 滚动窗口切分、NDCG 损失、EMA、早停、逐窗口训练
        ├── utils.py            # 197 维特征工程（158 Alpha158 算子 + 39 技术指标）、样本构建
        ├── test.py             # 推理：取最新交易日 → Top-5 → softmax 权重
        └── industry_map.py     # 调 baostock 生成行业映射
```

## 怎么跑

**前置**：本仓库不含数据。特征工程与取数脚本中，`get_stock_data.py` 属于官方 baseline 未改动文件，
没有复制进来，请从[官方仓库](https://github.com/Sherlock1956/THU-BDC2026)获取。

```bash
# 0) 环境
pip install -r app/requirements.txt
# 目录不固定：config.py 默认取自身所在位置的上两级（即 app/）为项目根
# 如需指定： export APP_ROOT=/abs/path/to/app   （Windows: set APP_ROOT=D:\path\to\app）

# 1) 取数据 → app/model/data/stock_data.csv（用官方 get_stock_data.py，需联网）
# 2) 生成行业映射 → app/model/data/industry_map.csv（需联网）
cd app/code/src && python industry_map.py

# 3) 训练（滚动窗口逐个训练，权重存 app/model/splitN_best.pth）
cd app/code/src && python train.py

# 4) 推理（窗口序号可用 BEST_SPLIT 指定，默认读 best_split_idx.pkl）
cd app/code/src && python test.py
```

也可用 Docker：`docker build -t bdc2026 .`，镜像内 `app/` 挂到 `/app`，`train.sh` / `test.sh` 已就位。

## 已知局限（不夸大）

- **成绩**：逐滚动窗口早停，最佳窗口为 **Split 8，最佳 FS = 0.1307**（口径 `final_score = (pred − random) / (max − random)`，
  自测口径、非官方榜分）。17 个窗口间波动很大，从 −0.0168 到 0.1307 都有，**只能当趋势看，不能当稳定收益**。
  完整逐窗口记录见 `app/readme.md`。tensorboard 事件文件在本机副本中为空（0 字节），曲线已无法重建。
- **辅助头未启用**：`ret_head` / `cls_head` / `vol_head` 有前向计算但没有损失项参与反传。
- **懒加载路径未接入主流程**：`LazyRankingDataset` / `create_ranking_dataset_streaming` 已实现但未被 `main()` 使用。
- **AMP 未启用**：`fp16_support=False`。
- **上传版做过整理**：路径去硬编码、`test.py` 模型与 scaler 改为同源（修 off-by-one）、
  `train.py` 补上最优窗口记录（原 `split_best_score` 恒为 `-inf`）。三处均已标注，**未重跑验证**。
  详见 `NOTICE.md` 第 4 节。

## 相关仓库

- [supermarket](https://github.com/avoid137/supermarket) —— 无人零售导购与自适应视觉结账系统（FastAPI + Vue3 + 多模态大模型）
- [tanker-port-pose-estimation](https://github.com/avoid137/tanker-port-pose-estimation) —— 罐车加注口位姿估计（点云，无训练模板几何先验）
- [kaggle-s6e9-postmortem](https://github.com/avoid137/kaggle-s6e9-postmortem) —— Kaggle 电动汽车购买预测复盘
