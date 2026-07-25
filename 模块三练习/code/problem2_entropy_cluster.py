# -*- coding: utf-8 -*-
"""
问题二 - 步骤一+二: 熵权法客观赋权 + 8维加权K-Means重构标签
输入: results/features_123_raw.csv (8特征)
输出: results/entropy_weights.csv, results/new_labels_123.csv
      results/cluster_centers_p2.csv, results/silhouette_by_k_p2.csv
      results/topsis_label_comparison.csv (新标签vs银行: ARI/NMI)
      results/p2_std_params.csv (123家min/max标准化参数, 供302复用)
      figures/fig8_cluster_profile.png
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                             normalized_mutual_info_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

DIR_RESULTS = r'E:\MathModeling\模块三练习\results'
DIR_FIGURES = r'E:\MathModeling\模块三练习\figures'

feature_cols = ['F1', 'F2', 'F3', 'F4', 'F7', 'F8', 'F9', 'F10']
directions = {'F1': 1, 'F2': 1, 'F3': -1, 'F4': -1,
              'F7': -1, 'F8': -1, 'F9': -1, 'F10': 1}

# ============ 1. 读取123家特征 ============
print("=" * 60)
print("1. 读取123家企业8维特征...")
df = pd.read_csv(f'{DIR_RESULTS}\\features_123_raw.csv')
X_raw = df[feature_cols].values
n, m = X_raw.shape
print(f"   样本: {n}, 特征: {feature_cols}")

# ============ 2. 按方向min-max标准化 ============
print("\n2. 按方向min-max标准化(统一为正向)...")
fmin = X_raw.min(axis=0)
fmax = X_raw.max(axis=0)
Z = np.zeros_like(X_raw)
for j, col in enumerate(feature_cols):
    rng = fmax[j] - fmin[j]
    if rng > 0:
        if directions[col] == 1:
            Z[:, j] = (X_raw[:, j] - fmin[j]) / rng
        else:
            Z[:, j] = (fmax[j] - X_raw[:, j]) / rng
    else:
        Z[:, j] = 0.5

# 保存标准化参数(供302复用)
df_std = pd.DataFrame({'特征': feature_cols, '方向': [directions[c] for c in feature_cols],
                       'min': fmin, 'max': fmax})
df_std.to_csv(f'{DIR_RESULTS}\\p2_std_params.csv', index=False, encoding='utf-8-sig')
print("   已保存 p2_std_params.csv")

# ============ 3. 熵权法 ============
print("\n3. 熵权法确定权重...")
Z_safe = Z + 1e-10
P = Z_safe / Z_safe.sum(axis=0, keepdims=True)
e = -1 / np.log(n) * np.sum(P * np.log(P), axis=0)
d = 1 - e
w = d / d.sum()
print("   熵权法权重:")
for j, col in enumerate(feature_cols):
    print(f"     {col}: e={e[j]:.4f}, d={d[j]:.4f}, w={w[j]:.4f}")

pd.DataFrame({'特征': feature_cols, '方向': [directions[c] for c in feature_cols],
              '信息熵e': e, '差异系数d': d, '权重w': w}).to_csv(
    f'{DIR_RESULTS}\\entropy_weights.csv', index=False, encoding='utf-8-sig')
print("   已保存 entropy_weights.csv")

# ============ 4. 加权K-Means (等价变换: Z' = Z * sqrt(w)) ============
print("\n4. 加权K-Means聚类...")
sqrt_w = np.sqrt(w)
Z_weighted = Z * sqrt_w

# 轮廓系数分析 k=2..6
print("   不同k的轮廓系数:")
sil_list = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, init='k-means++', n_init=50, random_state=42)
    lab = km.fit_predict(Z_weighted)
    sil = silhouette_score(Z_weighted, lab)
    sil_list.append({'k': k, '轮廓系数': sil})
    print(f"     k={k}: {sil:.4f}")
pd.DataFrame(sil_list).to_csv(f'{DIR_RESULTS}\\silhouette_by_k_p2.csv', index=False, encoding='utf-8-sig')

# k=4
km4 = KMeans(n_clusters=4, init='k-means++', n_init=50, random_state=42)
labels = km4.fit_predict(Z_weighted)
centers_weighted = km4.cluster_centers_
centers_std = centers_weighted / sqrt_w  # 还原到标准化空间
print(f"\n   k=4 各簇规模: {dict(zip(*np.unique(labels, return_counts=True)))}")

# ============ 5. 聚类→评级映射 ============
print("\n5. 聚类中心→评级映射...")
# Score_l = Σ w_j * c_lj (标准化空间中心, 已统一正向)
scores_cluster = (centers_std * w).sum(axis=1)
print(f"   各簇综合得分: {scores_cluster.round(4)}")
rank_order = np.argsort(-scores_cluster)
cluster_to_label = {}
new_ratings = ['A*', 'B*', 'C*', 'D*']
for rank, c in enumerate(rank_order):
    cluster_to_label[c] = new_ratings[rank]
print(f"   映射: {cluster_to_label}")

df_centers = pd.DataFrame(centers_std, columns=feature_cols)
df_centers['cluster'] = range(4)
df_centers['映射评级'] = [cluster_to_label[i] for i in range(4)]
df_centers['综合得分'] = scores_cluster
df_centers['样本数'] = [int((labels == i).sum()) for i in range(4)]
df_centers.to_csv(f'{DIR_RESULTS}\\cluster_centers_p2.csv', index=False, encoding='utf-8-sig')
print("   已保存 cluster_centers_p2.csv")

# 新标签
df['新标签'] = [cluster_to_label[l] for l in labels]
print(f"\n   新标签分布:")
print(df['新标签'].value_counts().sort_index().to_string())

# 各等级特征画像(原始8特征均值)
print("\n   各等级原始特征画像(均值):")
profile = df.groupby('新标签')[feature_cols].mean().round(3)
print(profile.loc[['A*', 'B*', 'C*', 'D*']].to_string())

# ============ 6. 新标签 vs 银行标签 ============
print("\n6. 新标签 vs 银行标签...")
y_bank = df['信誉评级'].values
y_new = df['新标签'].str.replace('*', '', regex=False).values
ari = adjusted_rand_score(y_bank, y_new)
nmi = normalized_mutual_info_score(y_bank, y_new)
print(f"   ARI = {ari:.4f}, NMI = {nmi:.4f}")
cm = confusion_matrix(y_bank, y_new, labels=['A', 'B', 'C', 'D'])
cm_df = pd.DataFrame(cm, index=[f'银行_{r}' for r in ['A','B','C','D']],
                     columns=[f'新_{r}' for r in ['A','B','C','D']])
print(cm_df.to_string())

df[['企业代号', '信誉评级'] + feature_cols + ['新标签']].to_csv(
    f'{DIR_RESULTS}\\new_labels_123.csv', index=False, encoding='utf-8-sig')
pd.DataFrame({'指标': ['ARI', 'NMI'], '值': [ari, nmi]}).to_csv(
    f'{DIR_RESULTS}\\topsis_label_comparison.csv', index=False, encoding='utf-8-sig')
print("\n   已保存 new_labels_123.csv, topsis_label_comparison.csv")

# ============ 7. 可视化: 聚类中心画像 ============
print("\n7. 生成可视化...")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 5))
center_plot = df_centers.set_index('映射评级')[feature_cols].loc[['A*', 'B*', 'C*', 'D*']]
sns.heatmap(center_plot, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax,
            linewidths=0.5, center=0.5)
ax.set_title('加权K-Means各评级聚类中心特征值 (方向统一后标准化空间)')
ax.set_ylabel('信用评级')
plt.tight_layout()
plt.savefig(f'{DIR_FIGURES}\\fig8_cluster_profile.png', dpi=150, bbox_inches='tight')
plt.close()
print("   已保存 fig8_cluster_profile.png")

print("\n" + "=" * 60)
print("熵权 + 加权K-Means重构标签完成！")
