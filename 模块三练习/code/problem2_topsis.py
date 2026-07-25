# -*- coding: utf-8 -*-
"""
问题二 - 熵权TOPSIS综合信用评分 (步骤一: 无监督重构标签的前半段)
输入: results/features_123_raw.csv (123家8特征), results/features_302_raw.csv (302家8特征)
输出: results/topsis_scores.csv (425家TOPSIS得分)
      results/entropy_weights.csv (熵权法权重)

修订: Z-score标准化全部8个特征后再做熵权TOPSIS,避免PCA因子(F1/F2 std~7)
      压倒稳定度/增长率(F7~F10 std~0.2)的问题。
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE = r'D:\建模\MathModeling\模块三练习'
DIR_RESULTS = f'{BASE}\\results'
DIR_FIGURES = f'{BASE}\\figures'

FEATURE_COLS = ['F1', 'F2', 'F3', 'F4', 'F7', 'F8', 'F9', 'F10']
POSITIVE_FEATURES = ['F1', 'F2', 'F10']
NEGATIVE_FEATURES = ['F3', 'F4', 'F7', 'F8', 'F9']

EPSILON = 1e-8


def zscore_col(X_123, X_302):
    """用123家的均值/std对全部数据做Z-score"""
    mu = X_123.mean(axis=0)
    sigma = X_123.std(axis=0)
    sigma[sigma < EPSILON] = 1.0
    return (X_123 - mu) / sigma, (X_302 - mu) / sigma, mu, sigma


def range_normalize(X_123, X_302):
    """基于123家的min/max对全部数据做 [epsilon, 1-epsilon] 归一化"""
    xmin = X_123.min(axis=0)
    xmax = X_123.max(axis=0)
    X_123_norm = np.zeros_like(X_123)
    X_302_norm = np.zeros_like(X_302)
    for j in range(X_123.shape[1]):
        if xmax[j] - xmin[j] < EPSILON:
            X_123_norm[:, j] = 0.5
            X_302_norm[:, j] = 0.5
        else:
            X_123_norm[:, j] = (X_123[:, j] - xmin[j]) / (xmax[j] - xmin[j])
            X_123_norm[:, j] = np.clip(X_123_norm[:, j], EPSILON, 1 - EPSILON)
            X_302_norm[:, j] = (X_302[:, j] - xmin[j]) / (xmax[j] - xmin[j])
            X_302_norm[:, j] = np.clip(X_302_norm[:, j], EPSILON, 1 - EPSILON)
    return X_123_norm, X_302_norm


def entropy_weights(X_norm):
    """熵权法: X_norm ∈ [epsilon, 1-epsilon] → 权重 w_j"""
    n = X_norm.shape[0]
    P = X_norm / X_norm.sum(axis=0, keepdims=True)
    e = -np.sum(P * np.log(P), axis=0) / np.log(n)
    d = 1 - e
    w = d / d.sum()
    return w, e, d


def topsis(X_norm, w, positive_mask):
    """TOPSIS 计算相对贴近度"""
    v = X_norm * w
    vpos = np.where(positive_mask, v.max(axis=0), v.min(axis=0))
    vneg = np.where(positive_mask, v.min(axis=0), v.max(axis=0))
    d_pos = np.sqrt(((v - vpos) ** 2).sum(axis=1))
    d_neg = np.sqrt(((v - vneg) ** 2).sum(axis=1))
    C = d_neg / (d_pos + d_neg + EPSILON)
    return C, d_pos, d_neg, vpos, vneg


# ============ 1. 读取特征 ============
print("=" * 60)
print("1. 读取特征数据...")
df_123 = pd.read_csv(f'{DIR_RESULTS}\\features_123_raw.csv')
df_302 = pd.read_csv(f'{DIR_RESULTS}\\features_302_raw.csv')

X_123 = df_123[FEATURE_COLS].values
X_302 = df_302[FEATURE_COLS].values

print(f"   123家特征形状: {X_123.shape}")
print(f"   302家特征形状: {X_302.shape}")

# ============ 2. Z-score标准化全部8个特征 ============
print("\n2. Z-score标准化全部8个特征 (基于123家统计量)...")
X_123_z, X_302_z, mu, sigma = zscore_col(X_123, X_302)

# 截断极端值到 ±3σ (防止个别超大企业主导TOPSIS得分)
CUTOFF = 3.0
n_capped_123 = (np.abs(X_123_z) > CUTOFF).sum()
n_capped_302 = (np.abs(X_302_z) > CUTOFF).sum()
X_123_z = np.clip(X_123_z, -CUTOFF, CUTOFF)
X_302_z = np.clip(X_302_z, -CUTOFF, CUTOFF)
print(f"   截断至±{CUTOFF}: 123家修改{n_capped_123}个值, 302家修改{n_capped_302}个值")

print("   标准化前(123家) 各特征std:")
for i, f in enumerate(FEATURE_COLS):
    print(f"     {f}: {X_123[:, i].std():.3f}")
print("   截断后(123家) 各特征std:")
for i, f in enumerate(FEATURE_COLS):
    print(f"     {f}: {X_123_z[:, i].std():.3f}")

df_scaler = pd.DataFrame({'特征': FEATURE_COLS, '均值': mu, '标准差': np.where(sigma < EPSILON, 1.0, sigma)})
df_scaler.to_csv(f'{DIR_RESULTS}\\topsis_scaler_params.csv', index=False, encoding='utf-8-sig')
print("   已保存 topsis_scaler_params.csv")

# ============ 3. Min-Max归一化 (用于熵权计算) ============
print("\n3. Min-Max归一化 [epsilon, 1-epsilon]...")
X_123_nm, X_302_nm = range_normalize(X_123_z, X_302_z)

positive_mask = np.array([f in POSITIVE_FEATURES for f in FEATURE_COLS], dtype=bool)

# ============ 4. 基于123家计算熵权 ============
print("\n4. 基于123家企业计算熵权...")
w, e, d = entropy_weights(X_123_nm)

df_entropy = pd.DataFrame({
    '特征': FEATURE_COLS,
    '信息熵e': e,
    '差异系数d': d,
    '权重w': w
})
df_entropy.to_csv(f'{DIR_RESULTS}\\entropy_weights.csv', index=False, encoding='utf-8-sig')
print("   已保存 entropy_weights.csv")
print("\n   熵权法权重:")
for _, row in df_entropy.iterrows():
    direction = '正向' if row['特征'] in POSITIVE_FEATURES else '负向'
    print(f"     {row['特征']}({direction}): e={row['信息熵e']:.4f}, d={row['差异系数d']:.4f}, w={row['权重w']:.4f}")

# ============ 5. TOPSIS评分 ============
print("\n5. TOPSIS综合评分...")
C_123, dpos_123, dneg_123, vpos, vneg = topsis(X_123_nm, w, positive_mask)
C_302, dpos_302, dneg_302, _, _ = topsis(X_302_nm, w, positive_mask)

df_123_out = df_123[['企业代号', '信誉评级']].copy()
df_123_out['TOPSIS得分'] = C_123
df_123_out['D+'] = dpos_123
df_123_out['D-'] = dneg_123

df_302_out = df_302[['企业代号']].copy()
df_302_out['TOPSIS得分'] = C_302
df_302_out['D+'] = dpos_302
df_302_out['D-'] = dneg_302

df_scores = pd.concat([df_123_out, df_302_out], ignore_index=True)
df_scores.to_csv(f'{DIR_RESULTS}\\topsis_scores.csv', index=False, encoding='utf-8-sig')
print("   已保存 topsis_scores.csv")

print(f"\n   123家 TOPSIS得分: mean={C_123.mean():.4f}, std={C_123.std():.4f}, "
      f"min={C_123.min():.4f}, max={C_123.max():.4f}")
print(f"   302家 TOPSIS得分: mean={C_302.mean():.4f}, std={C_302.std():.4f}, "
      f"min={C_302.min():.4f}, max={C_302.max():.4f}")

print("\n   123家各评级TOPSIS得分:")
for grade in ['A', 'B', 'C', 'D']:
    sub = df_123_out[df_123_out['信誉评级'] == grade]
    print(f"     {grade}级: n={len(sub)}, mean={sub['TOPSIS得分'].mean():.4f}, "
          f"std={sub['TOPSIS得分'].std():.4f}")
    if len(sub) <= 5:
        print(f"      得分明细: {sub['TOPSIS得分'].values.round(4)}")

# ============ 6. 可视化 ============
print("\n6. 生成可视化...")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 图8: TOPSIS得分分布
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

color_map = {'A': '#2ecc71', 'B': '#3498db', 'C': '#f39c12', 'D': '#e74c3c'}
for grade in ['A', 'B', 'C', 'D']:
    sub = df_123_out[df_123_out['信誉评级'] == grade]
    axes[0].hist(sub['TOPSIS得分'], bins=12, alpha=0.5, label=f'{grade}级',
                 color=color_map[grade], edgecolor='black', linewidth=0.5)
axes[0].set_xlabel('TOPSIS综合得分')
axes[0].set_ylabel('企业数')
axes[0].set_title('123家企业TOPSIS得分分布(按银行评级)')
axes[0].legend()

axes[1].hist(df_302_out['TOPSIS得分'], bins=25, color='#9b59b6', alpha=0.7,
             edgecolor='black', linewidth=0.5)
axes[1].set_xlabel('TOPSIS综合得分')
axes[1].set_ylabel('企业数')
axes[1].set_title('302家企业TOPSIS得分分布')
plt.tight_layout()
plt.savefig(f'{DIR_FIGURES}\\fig8_topsis_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   已保存 fig8_topsis_distribution.png")

# 图9: TOPSIS得分核密度图
fig, ax = plt.subplots(figsize=(10, 5))
for grade in ['A', 'B', 'C', 'D']:
    sub = df_123_out[df_123_out['信誉评级'] == grade]
    if len(sub) > 2:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(sub['TOPSIS得分'])
        x_grid = np.linspace(0, 1, 200)
        ax.plot(x_grid, kde(x_grid), color=color_map[grade], linewidth=2, label=f'{grade}级')

kde_all = gaussian_kde(C_123)
ax.plot(x_grid, kde_all(x_grid), 'k--', linewidth=1.5, alpha=0.6, label='全部123家')
ax.set_xlabel('TOPSIS综合得分')
ax.set_ylabel('核密度')
ax.set_title('123家企业TOPSIS得分核密度图')
ax.legend()
plt.tight_layout()
plt.savefig(f'{DIR_FIGURES}\\fig9_topsis_kde.png', dpi=150, bbox_inches='tight')
plt.close()
print("   已保存 fig9_topsis_kde.png")

# 图10: 熵权权重条形图
fig, ax = plt.subplots(figsize=(7, 5))
colors_w = ['#2ecc71' if f in POSITIVE_FEATURES else '#e74c3c' for f in FEATURE_COLS]
bars = ax.barh(FEATURE_COLS, w, color=colors_w, edgecolor='black', linewidth=0.5)
ax.set_xlabel('权重')
ax.set_title('熵权法特征权重 (绿色=正向, 红色=负向)')
ax.invert_yaxis()
for bar, val in zip(bars, w):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(f'{DIR_FIGURES}\\fig10_entropy_weights.png', dpi=150, bbox_inches='tight')
plt.close()
print("   已保存 fig10_entropy_weights.png")

print("\n" + "=" * 60)
print("熵权TOPSIS综合评分完成！")
print(f"输出文件: {DIR_RESULTS}\\topsis_scores.csv, {DIR_RESULTS}\\entropy_weights.csv")
