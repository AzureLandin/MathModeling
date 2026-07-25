# -*- coding: utf-8 -*-
"""
问题一 - 数据预处理 + 特征工程 + PCA降维 (修订版)
变更: x4/x12 定义为"有效发票中价税合计<0的条数", 19个变量全部进PCA
输出: results/features_123.csv, results/pca_loadings.csv, results/pca_scaler_params.csv
      figures/fig1_pca_scree.png, figures/fig2_loading_heatmap.png
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from factor_analyzer.factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============ 路径配置 ============
FILE1 = r'E:\MathModeling\模块三练习\2026C附件\附件1：123家有信贷记录企业的相关数据.xlsx'
DIR_RESULTS = r'E:\MathModeling\模块三练习\results'
DIR_FIGURES = r'E:\MathModeling\模块三练习\figures'

# ============ 1. 数据读取 ============
print("=" * 60)
print("1. 读取数据...")
df_info = pd.read_excel(FILE1, sheet_name='企业信息')
df_in = pd.read_excel(FILE1, sheet_name='进项发票信息')
df_out = pd.read_excel(FILE1, sheet_name='销项发票信息')
print(f"   企业信息: {len(df_info)} 行")
print(f"   进项发票: {len(df_in)} 条")
print(f"   销项发票: {len(df_out)} 条")

enterprise_ids = df_info['企业代号'].tolist()

# ============ 2. 特征工程: 19个汇总变量 ============
print("\n2. 提取19个汇总变量...")

# --- 进项侧 ---
dfi = df_in.copy()
dfi['is_valid'] = (dfi['发票状态'] == '有效发票').astype(int)
dfi['is_void'] = (dfi['发票状态'] == '作废发票').astype(int)
# x4: 有效发票中价税合计<0的条数(负数发票/红冲发票)
dfi['is_negative'] = ((dfi['发票状态'] == '有效发票') & (dfi['价税合计'] < 0)).astype(int)
dfi['valid_amount'] = dfi['金额'] * dfi['is_valid']
dfi['valid_tax'] = dfi['税额'] * dfi['is_valid']
dfi['void_total'] = dfi['价税合计'] * dfi['is_void']
dfi['valid_total'] = dfi['价税合计'] * dfi['is_valid']

g_in = dfi.groupby('企业代号').agg(
    x1=('发票号码', 'count'),
    x2=('is_valid', 'sum'),
    x3=('is_void', 'sum'),
    x4=('is_negative', 'sum'),
    x5=('valid_amount', 'sum'),
    x6=('valid_tax', 'sum'),
    x7=('void_total', 'sum'),
    x8=('valid_total', 'sum'),
).reset_index()
g_in['x7'] = g_in['x7'].abs()

# --- 销项侧 ---
dfo = df_out.copy()
dfo['is_valid'] = (dfo['发票状态'] == '有效发票').astype(int)
dfo['is_void'] = (dfo['发票状态'] == '作废发票').astype(int)
dfo['is_negative'] = ((dfo['发票状态'] == '有效发票') & (dfo['价税合计'] < 0)).astype(int)
dfo['valid_amount'] = dfo['金额'] * dfo['is_valid']
dfo['valid_tax'] = dfo['税额'] * dfo['is_valid']
dfo['void_total'] = dfo['价税合计'] * dfo['is_void']
dfo['valid_total'] = dfo['价税合计'] * dfo['is_valid']

g_out = dfo.groupby('企业代号').agg(
    x9=('发票号码', 'count'),
    x10=('is_valid', 'sum'),
    x11=('is_void', 'sum'),
    x12=('is_negative', 'sum'),
    x13=('valid_amount', 'sum'),
    x14=('valid_tax', 'sum'),
    x15=('void_total', 'sum'),
    x16=('valid_total', 'sum'),
).reset_index()
g_out['x15'] = g_out['x15'].abs()

# --- 合并 ---
df_feat = df_info[['企业代号', '信誉评级']].copy()
df_feat = df_feat.merge(g_in, on='企业代号', how='left')
df_feat = df_feat.merge(g_out, on='企业代号', how='left')
df_feat = df_feat.fillna(0)

# --- 衍生变量 ---
df_feat['x17'] = df_feat['x16'] - df_feat['x8']  # 有效收入
total_inv = df_feat['x1'] + df_feat['x9']
df_feat['x18'] = np.where(total_inv > 0, (df_feat['x3'] + df_feat['x11']) / total_inv, 0)  # 作废发票率
df_feat['x19'] = np.where(total_inv > 0, (df_feat['x4'] + df_feat['x12']) / total_inv, 0)  # 负数发票率

cols_19 = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8',
            'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16',
            'x17', 'x18', 'x19']

print(f"   19个汇总变量: {cols_19}")
print(f"   x4(进项负数发票数) 非零企业: {(df_feat['x4'] > 0).sum()} 家, 均值: {df_feat['x4'].mean():.2f}")
print(f"   x12(销项负数发票数) 非零企业: {(df_feat['x12'] > 0).sum()} 家, 均值: {df_feat['x12'].mean():.2f}")
print(f"   x19(负数发票率) 均值: {df_feat['x19'].mean():.4f}, 范围: [{df_feat['x19'].min():.4f}, {df_feat['x19'].max():.4f}]")

# ============ 3. 供应链稳定度 ============
print("\n3. 计算供应链稳定度...")
F7_list, F8_list, F9_list = [], [], []
for eid in enterprise_ids:
    inp = df_in[df_in['企业代号'] == eid]
    outp = df_out[df_out['企业代号'] == eid]
    f7 = inp['销方单位代号'].value_counts().head(3).sum() / len(inp) if len(inp) > 0 else 0
    f8 = outp['购方单位代号'].value_counts().head(3).sum() / len(outp) if len(outp) > 0 else 0
    f9 = ((f7**2 + f8**2) / 2) ** 0.5
    F7_list.append(f7)
    F8_list.append(f8)
    F9_list.append(f9)

df_feat['F7'] = F7_list
df_feat['F8'] = F8_list
df_feat['F9'] = F9_list
print(f"   F7(进稳定度) 均值: {df_feat['F7'].mean():.4f}")
print(f"   F8(销稳定度) 均值: {df_feat['F8'].mean():.4f}")
print(f"   F9(进销稳定度) 均值: {df_feat['F9'].mean():.4f}")

# ============ 4. 平均营收增长率 ============
print("\n4. 计算平均营收增长率...")
EPSILON = 1.0

dfo_valid = df_out[df_out['发票状态'] == '有效发票'].copy()
dfo_valid['开票日期'] = pd.to_datetime(dfo_valid['开票日期'])
dfo_valid['年份'] = dfo_valid['开票日期'].dt.year

# 仅用2017/2018/2019完整年份
dfo_valid = dfo_valid[dfo_valid['年份'].isin([2017, 2018, 2019])]
yearly_rev = dfo_valid.groupby(['企业代号', '年份'])['价税合计'].sum().reset_index()
yearly_rev.columns = ['企业代号', '年份', 'R']

F10_list = []
for eid in enterprise_ids:
    sub = yearly_rev[yearly_rev['企业代号'] == eid].set_index('年份')['R']
    r17 = sub.get(2017, 0)
    r18 = sub.get(2018, 0)
    r19 = sub.get(2019, 0)
    if r17 == 0 and r18 == 0:
        g1 = 0
    else:
        g1 = (r18 - r17) / (abs(r17) + EPSILON)
    if r18 == 0 and r19 == 0:
        g2 = 0
    else:
        g2 = (r19 - r18) / (abs(r18) + EPSILON)
    f10 = (g1 + g2) / 2
    F10_list.append(f10)

df_feat['F10_raw'] = F10_list
# 根据数据分布确定截断阈值
q01 = np.percentile(F10_list, 1)
q99 = np.percentile(F10_list, 99)
print(f"   F10原始分布: 1%分位={q01:.4f}, 99%分位={q99:.4f}")
print(f"   F10原始范围: [{min(F10_list):.4f}, {max(F10_list):.4f}]")
# 截断至[-1, 5]（即-100%~500%）
df_feat['F10'] = np.clip(F10_list, -1.0, 5.0)
print(f"   F10截断后均值: {df_feat['F10'].mean():.4f}, 范围: [{df_feat['F10'].min():.4f}, {df_feat['F10'].max():.4f}]")

# ============ 5. Winsorize缩尾 ============
print("\n5. Winsorize缩尾 (1%/99%)...")
winsor_cols = cols_19 + ['F10']
for col in winsor_cols:
    arr = winsorize(df_feat[col].values, limits=[0.01, 0.01])
    df_feat[col] = np.array(arr)
df_feat['F10'] = np.clip(df_feat['F10'].values, -1.0, 5.0)

# ============ 6. Z-score标准化 (19个汇总变量) ============
print("\n6. Z-score标准化...")
scaler_params = []
for col in cols_19:
    mu = df_feat[col].mean()
    sigma = df_feat[col].std()
    if sigma > 0:
        df_feat[col] = (df_feat[col] - mu) / sigma
    else:
        df_feat[col] = 0
    scaler_params.append({'变量': col, '均值': mu, '标准差': sigma})

df_scaler = pd.DataFrame(scaler_params)
df_scaler.to_csv(f'{DIR_RESULTS}\\pca_scaler_params.csv', index=False, encoding='utf-8-sig')
print("   已保存 pca_scaler_params.csv")
# 检查是否有零方差列
zero_var = [col for col in cols_19 if df_feat[col].std() == 0]
if zero_var:
    print(f"   [警告] 零方差列: {zero_var}")

# ============ 7. KMO & Bartlett检验 ============
print("\n7. KMO & Bartlett检验...")
X = df_feat[cols_19].values
chi2, p_val = calculate_bartlett_sphericity(X)
kmo_all, kmo_overall = calculate_kmo(X)
print(f"   Bartlett球形检验: chi2={chi2:.1f}, p={p_val:.2e}")
print(f"   KMO = {kmo_overall:.3f}")

# ============ 8. PCA (Kaiser准则 + Varimax旋转) ============
print("\n8. PCA主成分分析...")
fa_init = FactorAnalyzer(rotation=None, method='principal')
fa_init.fit(X)
eigenvalues_raw, _ = fa_init.get_eigenvalues()
n_factors = int(np.sum(eigenvalues_raw > 1))
print(f"   特征根>1的因子数: {n_factors}")
print(f"   特征根: {eigenvalues_raw[:8].round(3)}")

# Varimax旋转
fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax', method='principal')
fa.fit(X)
loadings = fa.loadings_
scores = fa.transform(X)

# 方差解释
variances = fa.get_factor_variance()
var_explained = variances[1]
cum_var = np.cumsum(var_explained)
print(f"   各因子方差贡献: {var_explained.round(4)}")
print(f"   累计方差贡献: {cum_var[-1]:.4f}")

# 保存载荷矩阵
cols_factor = [f'F{i+1}' for i in range(n_factors)]
df_loadings = pd.DataFrame(loadings, index=cols_19, columns=cols_factor)
df_loadings.to_csv(f'{DIR_RESULTS}\\pca_loadings.csv', encoding='utf-8-sig')
print("   已保存 pca_loadings.csv")

# 打印载荷矩阵摘要(每个因子的高载荷变量)
print("\n   载荷矩阵摘要 (|loading| > 0.6):")
for i, fname in enumerate(cols_factor):
    high_vars = [(cols_19[j], loadings[j, i]) for j in range(len(cols_19)) if abs(loadings[j, i]) > 0.6]
    high_vars.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"   {fname}: {[(v, round(l, 3)) for v, l in high_vars]}")

# 保存因子得分
for i in range(n_factors):
    df_feat[f'F{i+1}'] = scores[:, i]

# ============ 9. 保存最终特征表 ============
pca_factor_cols = [f'F{i+1}' for i in range(n_factors)]
feature_cols = pca_factor_cols + ['F7', 'F8', 'F9', 'F10']
df_out_final = df_feat[['企业代号', '信誉评级'] + feature_cols].copy()
df_out_final.to_csv(f'{DIR_RESULTS}\\features_123.csv', index=False, encoding='utf-8-sig')
print(f"\n9. 已保存 features_123.csv ({len(df_out_final)} 行)")
print(f"   入模特征({len(feature_cols)}个): {feature_cols}")

# ============ 10. 可视化 ============
print("\n10. 生成可视化...")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 图1: PCA碎石图
fig, ax1 = plt.subplots(figsize=(8, 5))
x_idx = range(1, len(eigenvalues_raw) + 1)
ax1.bar(x_idx, eigenvalues_raw, alpha=0.6, color='steelblue', label='特征根')
ax1.axhline(y=1, color='red', linestyle='--', label='特征根=1')
ax1.set_xlabel('主成分编号')
ax1.set_ylabel('特征根', color='steelblue')
ax1.set_xticks(list(x_idx))

ax2 = ax1.twinx()
cum_pct = np.cumsum(eigenvalues_raw) / np.sum(eigenvalues_raw) * 100
ax2.plot(x_idx, cum_pct, 'ro-', markersize=5, label='累计方差贡献率(%)')
ax2.set_ylabel('累计方差贡献率 (%)', color='red')
ax2.set_ylim(0, 105)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
plt.title('PCA碎石图')
plt.tight_layout()
plt.savefig(f'{DIR_FIGURES}\\fig1_pca_scree.png', dpi=150, bbox_inches='tight')
plt.close()
print("   已保存 fig1_pca_scree.png")

# 图2: 载荷热力图
fig, ax = plt.subplots(figsize=(8, 9))
sns.heatmap(df_loadings, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, ax=ax, linewidths=0.5)
ax.set_title('Varimax旋转后因子载荷矩阵')
plt.tight_layout()
plt.savefig(f'{DIR_FIGURES}\\fig2_loading_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("   已保存 fig2_loading_heatmap.png")

# 图3: 各评级等级特征箱线图
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
plot_cols = feature_cols[:min(7, len(feature_cols))]
for idx, col in enumerate(plot_cols):
    ax = axes[idx]
    data_plot = df_feat[['信誉评级', col]].copy()
    sns.boxplot(x='信誉评级', y=col, data=data_plot, ax=ax,
                order=['A', 'B', 'C', 'D'], palette='Set2')
    ax.set_title(col)
    ax.set_xlabel('')
# 隐藏多余子图
for idx in range(len(plot_cols), len(axes)):
    axes[idx].set_visible(False)
plt.suptitle('各评级等级特征分布箱线图', fontsize=14)
plt.tight_layout()
plt.savefig(f'{DIR_FIGURES}\\fig3_feature_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("   已保存 fig3_feature_boxplot.png")

print("\n" + "=" * 60)
print("预处理完成！")
