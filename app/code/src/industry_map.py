import baostock as bs
import pandas as pd
import joblib
import os
from config import config

data_file = os.path.join(config['data_path'], config['data_file'])
out_dir = config['output_dir']
os.makedirs(out_dir, exist_ok=True)
df = pd.read_csv(data_file, dtype={"股票代码":str})
all_stock_ids = sorted(df["股票代码"].unique())
stockid2idx = {s:i for i,s in enumerate(all_stock_ids)}
joblib.dump(stockid2idx, os.path.join(out_dir, "stockid2idx.pkl"))
print("自动生成stockid2idx完成")

# ===================== 路径配置 =====================
output_csv = config['industry_map_csv']
stockid_pkl_path = os.path.join(out_dir, "stockid2idx.pkl")
# ==========================================================================
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# 1. 加载本地股票列表
print("加载本地股票映射 stockid2idx.pkl ...")
stockid2idx = joblib.load(stockid_pkl_path)
target_stock_set = set(stockid2idx.keys())
print(f"项目内股票总数：{len(target_stock_set)}")

# 2. 登录baostock
lg = bs.login()
if lg.error_code != "0":
    raise Exception(f"Baostock登录失败：{lg.error_msg}")
print("Baostock登录成功，拉取全市场股票行业...")

# 3. 修复：不用get_data()，逐行读取数据规避append报错
rs = bs.query_stock_industry()
data_rows = []
while (rs.error_code == '0') & rs.next():
    data_rows.append(rs.get_row_data())
# 手动构造DataFrame
columns = rs.fields
all_ind_data = pd.DataFrame(data_rows, columns=columns)
bs.logout()

# 4. 处理股票代码 sz.000001 -> 000001
all_ind_data["stock_code"] = all_ind_data["code"].str[3:].str.zfill(6)

# 5. 行业转数字ID
ind_name_list = sorted(all_ind_data["industry"].unique())
name2id = {name: idx + 1 for idx, name in enumerate(ind_name_list)}
all_ind_data["industry_id"] = all_ind_data["industry"].map(name2id)

# 6. 只保留你项目里的股票
df_filter = all_ind_data[all_ind_data["stock_code"].isin(target_stock_set)].copy()
stock_ind_dict = dict(zip(df_filter["stock_code"], df_filter["industry_id"]))

# 7. 补齐所有股票，无行业填0
final_rows = []
for code in target_stock_set:
    iid = stock_ind_dict.get(code, 0)
    final_rows.append([code, iid])

# 8. 输出csv
df_out = pd.DataFrame(final_rows, columns=["stock_code", "industry_id"])
df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")

# 统计打印
fail_count = sum(1 for _,iid in final_rows if iid == 0)
print("="*60)
print(f"文件生成完成：{output_csv}")
print(f"识别一级行业数量：{len(name2id)}")
print(f"未匹配行业股票：{fail_count}")
print("\n行业-ID对照表：")
for name, idx in sorted(name2id.items(), key=lambda x:x[1]):
    print(f"{idx:2d} | {name}")
print("="*60)