# -*- coding: utf-8 -*-
"""
问题一 - 多元Logit模型训练与评估 + 时效性稳健性检验 (修订版)
输入: results/features_123.csv
输出: results/logit_summary.txt, results/logit_confusion_matrix.csv
      figures/fig4_logit_confusion.png, figures/fig7_robustness.png
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============ 路径配置 ============
DIR_RESULTS = r'D:\建模\MathModeling\模块三练习\results'
DIR_FIGURES = r'D:\建模\MathModeling\模块三练习\figures'
FILE1 = r'D:\建模\MathModeling\模块三练习\2026C附件\附件1：123家有信贷记录企业的相关数据.xlsx'

# ============ 1. 读取特征 ============
print("=" * 60)
print("1. 读取特征数据...")
df = pd.read_csv(f'{DIR_RESULTS}\\features_123.csv')
print(f"   样本数: {len(df)}, 列: {list(df.columns)}")

# 动态识别特征列(除企业代号和信誉评级外的所有列)
feature_cols = [c for c in df.columns if c not in ['企业代号', '信誉评级']]
print(f"   入模特征({len(feature_cols)}个): {feature_cols}")

LABEL_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
LABEL_NAMES = ['A', 'B', 'C', 'D']

X = df[feature_cols].values
y = df['信誉评级'].map(LABEL_MAP).values
print(f"   评级分布: {dict(zip(*np.unique(y, return_counts=True)))}")

# ============ 2. 多元Logit建模 ============
print("\n2. 拟合多元Logit模型 (基准组=A)...")
X_const = sm.add_constant(X)
model = sm.MNLogit(y, X_const)
result = model.fit(method='lbfgs', maxiter=1000, disp=0)

# 保存summary
with open(f'{DIR_RESULTS}\\logit_summary.txt', 'w', encoding='utf-8') as f:
    f.write("多元Logit回归结果 (基准组=A)\n")
    f.write(f"入模特征: {feature_cols}\n")
    f.write("=" * 60 + "\n\n")
    f.write(str(result.summary()) + "\n\n")
    f.write("=" * 60 + "\n")
    f.write("系数表 (含P值)\n")
    f.write("=" * 60 + "\n")
    f.write(result.summary2().tables[1].to_string() + "\n\n")
    f.write("=" * 60 + "\n")
    f.write(f"伪R² = {result.prsquared:.4f}\n")
    f.write(f"对数似然 = {result.llf:.4f}\n")
    f.write(f"AIC = {result.aic:.4f}\n")
    f.write(f"BIC = {result.bic:.4f}\n")
print("   已保存 logit_summary.txt")
print(f"   伪R² = {result.prsquared:.4f}")
print(f"   LLR p-value = {result.llr_pvalue:.2e}")

# ============ 3. 预测与评估 ============
print("\n3. 模型预测与评估...")
probs = result.predict(X_const)
y_pred = probs.argmax(axis=1)

cm = confusion_matrix(y, y_pred, labels=[0, 1, 2, 3])
acc = accuracy_score(y, y_pred)
kappa = cohen_kappa_score(y, y_pred)

print(f"   总体准确率: {acc:.4f} ({int(acc*123)}/123)")
print(f"   Cohen's Kappa: {kappa:.4f}")
print(f"\n   混淆矩阵 (行=实际, 列=预测):")
cm_df = pd.DataFrame(cm, index=[f'实际_{n}' for n in LABEL_NAMES],
                     columns=[f'预测_{n}' for n in LABEL_NAMES])
print(cm_df.to_string())

print("\n   分级别准确率:")
for i, name in enumerate(LABEL_NAMES):
    total_i = cm[i, :].sum()
    acc_i = cm[i, i] / total_i if total_i > 0 else 0
    print(f"     {name}级: {acc_i:.4f} ({cm[i,i]}/{total_i})")

cm_df.to_csv(f'{DIR_RESULTS}\\logit_confusion_matrix.csv', encoding='utf-8-sig')
print("\n   已保存 logit_confusion_matrix.csv")

# ============ 4. 时效性稳健性检验 ============
print("\n4. 时效性稳健性检验 (仅用2019-01~2020-02数据)...")

from scipy.stats.mstats import winsorize as winsorize_fn
from factor_analyzer.factor_analyzer import FactorAnalyzer

# 读取原始数据并筛选近期
df_info = pd.read_excel(FILE1, sheet_name='企业信息')
df_in = pd.read_excel(FILE1, sheet_name='进项发票信息')
df_out_raw = pd.read_excel(FILE1, sheet_name='销项发票信息')

df_in['开票日期'] = pd.to_datetime(df_in['开票日期'])
df_out_raw['开票日期'] = pd.to_datetime(df_out_raw['开票日期'])

cutoff = pd.Timestamp('2019-01-01')
df_in_recent = df_in[df_in['开票日期'] >= cutoff].copy()
df_out_recent = df_out_raw[df_out_raw['开票日期'] >= cutoff].copy()
print(f"   近期进项发票: {len(df_in_recent)} 条, 近期销项发票: {len(df_out_recent)} 条")

enterprise_ids = df_info['企业代号'].tolist()

# 重新提取19个特征
dfi = df_in_recent.copy()
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

dfo = df_out_recent.copy()
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

df_recent = df_info[['企业代号', '信誉评级']].copy()
df_recent = df_recent.merge(g_in, on='企业代号', how='left')
df_recent = df_recent.merge(g_out, on='企业代号', how='left')
df_recent = df_recent.fillna(0)

cols_19 = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8',
            'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16',
            'x17', 'x18', 'x19']
df_recent['x17'] = df_recent['x16'] - df_recent['x8']
total_inv = df_recent['x1'] + df_recent['x9']
df_recent['x18'] = np.where(total_inv > 0, (df_recent['x3'] + df_recent['x11']) / total_inv, 0)
df_recent['x19'] = np.where(total_inv > 0, (df_recent['x4'] + df_recent['x12']) / total_inv, 0)

# 稳定度
F7_r, F8_r, F9_r = [], [], []
for eid in enterprise_ids:
    inp = df_in_recent[df_in_recent['企业代号'] == eid]
    outp = df_out_recent[df_out_recent['企业代号'] == eid]
    f7 = inp['销方单位代号'].value_counts().head(3).sum() / len(inp) if len(inp) > 0 else 0
    f8 = outp['购方单位代号'].value_counts().head(3).sum() / len(outp) if len(outp) > 0 else 0
    f9 = ((f7**2 + f8**2) / 2) ** 0.5
    F7_r.append(f7)
    F8_r.append(f8)
    F9_r.append(f9)
df_recent['F7'] = F7_r
df_recent['F8'] = F8_r
df_recent['F9'] = F9_r

# 增长率(近期数据只有2019，无法算同比，设F10=0)
df_recent['F10'] = 0.0

# Winsorize
for col in cols_19 + ['F10']:
    arr = winsorize_fn(df_recent[col].values, limits=[0.01, 0.01])
    df_recent[col] = np.array(arr)

# Z-score: 使用全数据的均值和标准差
df_scaler = pd.read_csv(f'{DIR_RESULTS}\\pca_scaler_params.csv')
scaler_dict = dict(zip(df_scaler['变量'], zip(df_scaler['均值'], df_scaler['标准差'])))
for col in cols_19:
    mu, sigma = scaler_dict[col]
    if sigma > 0:
        df_recent[col] = (df_recent[col] - mu) / sigma
    else:
        df_recent[col] = 0

# PCA投影: 使用全数据的载荷矩阵
df_loadings = pd.read_csv(f'{DIR_RESULTS}\\pca_loadings.csv', index_col=0)
loadings_matrix = df_loadings.values
n_factors = loadings_matrix.shape[1]

X_recent = df_recent[cols_19].values
scores_recent = X_recent @ loadings_matrix
for i in range(n_factors):
    df_recent[f'F{i+1}'] = scores_recent[:, i]

# 用全数据模型预测近期数据
X_recent_model = sm.add_constant(df_recent[feature_cols].values)
probs_recent = result.predict(X_recent_model)
y_pred_recent = probs_recent.argmax(axis=1)
y_recent = df_recent['信誉评级'].map(LABEL_MAP).values

cm_recent = confusion_matrix(y_recent, y_pred_recent, labels=[0, 1, 2, 3])
acc_recent = accuracy_score(y_recent, y_pred_recent)
kappa_recent = cohen_kappa_score(y_recent, y_pred_recent)

print(f"   近期数据准确率: {acc_recent:.4f} (全数据: {acc:.4f})")
print(f"   近期数据Kappa: {kappa_recent:.4f} (全数据: {kappa:.4f})")

# ============ 5. 可视化 ============
print("\n5. 生成可视化...")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 图4: Logit混淆矩阵
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_df.astype(int), annot=True, fmt='d', cmap='Blues', ax=ax,
            linewidths=0.5, cbar_kws={'label': '样本数'})
ax.set_title(f'多元Logit混淆矩阵 (Acc={acc:.2%}, Kappa={kappa:.3f})')
ax.set_xlabel('预测评级')
ax.set_ylabel('实际评级')
plt.tight_layout()
plt.savefig(f'{DIR_FIGURES}\\fig4_logit_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("   已保存 fig4_logit_confusion.png")

# 图7: 全数据 vs 近期数据对比
fig, ax = plt.subplots(figsize=(7, 5))
metrics = ['总体准确率', "Kappa系数"]
full_vals = [acc, kappa]
recent_vals = [acc_recent, kappa_recent]
x = np.arange(len(metrics))
w = 0.3
bars1 = ax.bar(x - w/2, full_vals, w, label='全数据 (2017-2020)', color='steelblue')
bars2 = ax.bar(x + w/2, recent_vals, w, label='近期数据 (2019-2020)', color='coral')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.0)
ax.legend()
ax.set_title('时效性稳健性检验: 全数据 vs 近期数据')
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{bar.get_height():.3f}', ha='center', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{bar.get_height():.3f}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(f'{DIR_FIGURES}\\fig7_robustness.png', dpi=150, bbox_inches='tight')
plt.close()
print("   已保存 fig7_robustness.png")

print("\n" + "=" * 60)
print("多元Logit建模完成！")
