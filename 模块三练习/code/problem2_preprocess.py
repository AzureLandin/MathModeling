# -*- coding: utf-8 -*-
"""
问题二 - 302家企业特征提取 + PCA投影 + 标准化
关键: 使用问题一的标准化参数和PCA载荷矩阵进行投影, 保证因子空间一致
输出: results/features_302_raw.csv (投影后未聚类标准化的8特征)
      results/features_123_raw.csv (123家同口径特征, 用于事后验证)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
import warnings
warnings.filterwarnings('ignore')

# ============ 路径配置 ============
FILE1 = r'D:\建模\MathModeling\模块三练习\2026C附件\附件1：123家有信贷记录企业的相关数据.xlsx'
FILE2 = r'D:\建模\MathModeling\模块三练习\2026C附件\附件2：302家无信贷记录企业的相关数据.xlsx'
DIR_RESULTS = r'D:\建模\MathModeling\模块三练习\results'

cols_19 = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8',
           'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16',
           'x17', 'x18', 'x19']
EPSILON = 1.0


def extract_invoice_features(df_in, df_out, enterprise_ids):
    """从进项/销项发票提取19个汇总变量 + 稳定度 + 增长率"""
    # 进项侧
    dfi = df_in.copy()
    dfi['is_valid'] = (dfi['发票状态'] == '有效发票').astype(int)
    dfi['is_void'] = (dfi['发票状态'] == '作废发票').astype(int)
    dfi['is_negative'] = ((dfi['发票状态'] == '有效发票') & (dfi['价税合计'] < 0)).astype(int)
    dfi['valid_amount'] = dfi['金额'] * dfi['is_valid']
    dfi['valid_tax'] = dfi['税额'] * dfi['is_valid']
    dfi['void_total'] = dfi['价税合计'] * dfi['is_void']
    dfi['valid_total'] = dfi['价税合计'] * dfi['is_valid']
    g_in = dfi.groupby('企业代号').agg(
        x1=('发票号码', 'count'), x2=('is_valid', 'sum'), x3=('is_void', 'sum'),
        x4=('is_negative', 'sum'), x5=('valid_amount', 'sum'), x6=('valid_tax', 'sum'),
        x7=('void_total', 'sum'), x8=('valid_total', 'sum'),
    ).reset_index()
    g_in['x7'] = g_in['x7'].abs()

    # 销项侧
    dfo = df_out.copy()
    dfo['is_valid'] = (dfo['发票状态'] == '有效发票').astype(int)
    dfo['is_void'] = (dfo['发票状态'] == '作废发票').astype(int)
    dfo['is_negative'] = ((dfo['发票状态'] == '有效发票') & (dfo['价税合计'] < 0)).astype(int)
    dfo['valid_amount'] = dfo['金额'] * dfo['is_valid']
    dfo['valid_tax'] = dfo['税额'] * dfo['is_valid']
    dfo['void_total'] = dfo['价税合计'] * dfo['is_void']
    dfo['valid_total'] = dfo['价税合计'] * dfo['is_valid']
    g_out = dfo.groupby('企业代号').agg(
        x9=('发票号码', 'count'), x10=('is_valid', 'sum'), x11=('is_void', 'sum'),
        x12=('is_negative', 'sum'), x13=('valid_amount', 'sum'), x14=('valid_tax', 'sum'),
        x15=('void_total', 'sum'), x16=('valid_total', 'sum'),
    ).reset_index()
    g_out['x15'] = g_out['x15'].abs()

    df_feat = pd.DataFrame({'企业代号': enterprise_ids})
    df_feat = df_feat.merge(g_in, on='企业代号', how='left')
    df_feat = df_feat.merge(g_out, on='企业代号', how='left')
    df_feat = df_feat.fillna(0)

    # 衍生变量
    df_feat['x17'] = df_feat['x16'] - df_feat['x8']
    total_inv = df_feat['x1'] + df_feat['x9']
    df_feat['x18'] = np.where(total_inv > 0, (df_feat['x3'] + df_feat['x11']) / total_inv, 0)
    df_feat['x19'] = np.where(total_inv > 0, (df_feat['x4'] + df_feat['x12']) / total_inv, 0)

    # 稳定度
    F7, F8, F9 = [], [], []
    for eid in enterprise_ids:
        inp = df_in[df_in['企业代号'] == eid]
        outp = df_out[df_out['企业代号'] == eid]
        f7 = inp['销方单位代号'].value_counts().head(3).sum() / len(inp) if len(inp) > 0 else 0
        f8 = outp['购方单位代号'].value_counts().head(3).sum() / len(outp) if len(outp) > 0 else 0
        f9 = ((f7**2 + f8**2) / 2) ** 0.5
        F7.append(f7); F8.append(f8); F9.append(f9)
    df_feat['F7'] = F7
    df_feat['F8'] = F8
    df_feat['F9'] = F9

    # 平均营收增长率 (2017-2019)
    dfo_valid = df_out[df_out['发票状态'] == '有效发票'].copy()
    dfo_valid['开票日期'] = pd.to_datetime(dfo_valid['开票日期'])
    dfo_valid['年份'] = dfo_valid['开票日期'].dt.year
    dfo_valid = dfo_valid[dfo_valid['年份'].isin([2017, 2018, 2019])]
    yearly = dfo_valid.groupby(['企业代号', '年份'])['价税合计'].sum().reset_index()
    yearly.columns = ['企业代号', '年份', 'R']
    F10 = []
    for eid in enterprise_ids:
        sub = yearly[yearly['企业代号'] == eid].set_index('年份')['R']
        r17, r18, r19 = sub.get(2017, 0), sub.get(2018, 0), sub.get(2019, 0)
        g1 = 0 if (r17 == 0 and r18 == 0) else (r18 - r17) / (abs(r17) + EPSILON)
        g2 = 0 if (r18 == 0 and r19 == 0) else (r19 - r18) / (abs(r18) + EPSILON)
        F10.append((g1 + g2) / 2)
    df_feat['F10'] = np.clip(F10, -1.0, 5.0)
    return df_feat


# ============ 1. 读取问题一参数 ============
print("=" * 60)
print("1. 读取问题一的标准化参数和PCA载荷矩阵...")
df_scaler = pd.read_csv(f'{DIR_RESULTS}\\pca_scaler_params.csv')
scaler_dict = dict(zip(df_scaler['变量'], zip(df_scaler['均值'], df_scaler['标准差'])))
df_loadings = pd.read_csv(f'{DIR_RESULTS}\\pca_loadings.csv', index_col=0)
loadings_matrix = df_loadings.values
n_factors = loadings_matrix.shape[1]
print(f"   PCA因子数: {n_factors}, 载荷矩阵形状: {loadings_matrix.shape}")

# ============ 2. 处理302家企业 ============
print("\n2. 读取附件2 (302家企业)...")
df_info2 = pd.read_excel(FILE2, sheet_name='企业信息')
df_in2 = pd.read_excel(FILE2, sheet_name='进项发票信息')
df_out2 = pd.read_excel(FILE2, sheet_name='销项发票信息')
ids_302 = df_info2['企业代号'].tolist()
print(f"   企业: {len(ids_302)} 家, 进项: {len(df_in2)} 条, 销项: {len(df_out2)} 条")

print("\n3. 提取302家特征...")
df_302 = extract_invoice_features(df_in2, df_out2, ids_302)
print(f"   x4非零企业: {(df_302['x4']>0).sum()}, x12非零企业: {(df_302['x12']>0).sum()}")

# ============ 4. Winsorize + 用问题一参数Z-score + PCA投影 ============
print("\n4. Winsorize + 问题一参数标准化 + PCA投影...")
for col in cols_19 + ['F10']:
    arr = winsorize(df_302[col].values, limits=[0.01, 0.01])
    df_302[col] = np.array(arr)
df_302['F10'] = np.clip(df_302['F10'].values, -1.0, 5.0)

# 用问题一的均值/标准差标准化19变量
for col in cols_19:
    mu, sigma = scaler_dict[col]
    df_302[col] = (df_302[col] - mu) / sigma if sigma > 0 else 0

# PCA投影: 因子得分 = X_std @ L
X_302 = df_302[cols_19].values
scores_302 = X_302 @ loadings_matrix
for i in range(n_factors):
    df_302[f'F{i+1}'] = scores_302[:, i]

pca_cols = [f'F{i+1}' for i in range(n_factors)]
feature_cols = pca_cols + ['F7', 'F8', 'F9', 'F10']
print(f"   8个特征: {feature_cols}")

# 保存投影后的原始特征(未做聚类标准化)
df_302_out = df_302[['企业代号'] + feature_cols].copy()
df_302_out.to_csv(f'{DIR_RESULTS}\\features_302_raw.csv', index=False, encoding='utf-8-sig')
print("   已保存 features_302_raw.csv")

# ============ 5. 同样处理123家(用于事后验证映射) ============
print("\n5. 提取123家同口径特征(用于验证映射)...")
df_info1 = pd.read_excel(FILE1, sheet_name='企业信息')
df_in1 = pd.read_excel(FILE1, sheet_name='进项发票信息')
df_out1 = pd.read_excel(FILE1, sheet_name='销项发票信息')
ids_123 = df_info1['企业代号'].tolist()

df_123 = extract_invoice_features(df_in1, df_out1, ids_123)
df_123['信誉评级'] = df_info1.set_index('企业代号').loc[ids_123, '信誉评级'].values

for col in cols_19 + ['F10']:
    arr = winsorize(df_123[col].values, limits=[0.01, 0.01])
    df_123[col] = np.array(arr)
df_123['F10'] = np.clip(df_123['F10'].values, -1.0, 5.0)
for col in cols_19:
    mu, sigma = scaler_dict[col]
    df_123[col] = (df_123[col] - mu) / sigma if sigma > 0 else 0
X_123 = df_123[cols_19].values
scores_123 = X_123 @ loadings_matrix
for i in range(n_factors):
    df_123[f'F{i+1}'] = scores_123[:, i]

df_123_out = df_123[['企业代号', '信誉评级'] + feature_cols].copy()
df_123_out.to_csv(f'{DIR_RESULTS}\\features_123_raw.csv', index=False, encoding='utf-8-sig')
print("   已保存 features_123_raw.csv")

print("\n" + "=" * 60)
print("问题二预处理完成！")
print(f"302家特征均值概览:")
print(df_302_out[feature_cols].mean().round(3).to_string())
