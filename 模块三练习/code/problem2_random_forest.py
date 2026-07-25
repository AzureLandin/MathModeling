# -*- coding: utf-8 -*-
"""
问题二 - TOPSIS分位数标签划分 + 随机森林分类预测
输入: results/topsis_scores.csv, results/features_123_raw.csv, results/features_302_raw.csv
输出: results/rf_pred_302.csv, results/rf_feature_importance.csv
      figures/fig11-16

标签划分: 按123家TOPSIS得分的25%/50%/75%分位数生成A/B/C/D四档。
该方案避免一维K-Means受长尾得分影响产生极不均衡的小簇。
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (confusion_matrix, accuracy_score, cohen_kappa_score,
                               adjusted_rand_score, normalized_mutual_info_score)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

BASE = r'D:\建模\MathModeling\模块三练习'
DIR_RESULTS = f'{BASE}\\results'
DIR_FIGURES = f'{BASE}\\figures'

FEATURE_COLS = ['F1', 'F2', 'F3', 'F4', 'F7', 'F8', 'F9', 'F10']
LABEL_MAP = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
LABEL_NAMES = ['A', 'B', 'C', 'D']
BANK_LABEL_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
GRADE_COLORS = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']


def quantile_labels(scores_123, scores_302):
    """基于123家TOPSIS得分的25%/50%/75%分位数作为断点。"""
    q25, q50, q75 = np.percentile(scores_123, [25, 50, 75])

    def assign(s):
        if s >= q75:
            return 0  # A
        elif s >= q50:
            return 1  # B
        elif s >= q25:
            return 2  # C
        else:
            return 3  # D

    labels_123 = np.array([assign(s) for s in scores_123])
    labels_302 = np.array([assign(s) for s in scores_302])

    bps = [q75, q50, q25]
    return labels_123, labels_302, bps


def evaluate_labels(labels_123, bank_labels, method_name):
    """评估新标签 vs 银行标签的一致性"""
    ari = adjusted_rand_score(bank_labels, labels_123)
    nmi = normalized_mutual_info_score(bank_labels, labels_123)
    cm = confusion_matrix(bank_labels, labels_123, labels=[0, 1, 2, 3])
    acc = accuracy_score(bank_labels, labels_123)
    kappa = cohen_kappa_score(bank_labels, labels_123)

    print(f"\n   [{method_name}] vs 银行标签:")
    print(f"   ARI={ari:.4f}, NMI={nmi:.4f}, 吻合率={acc:.4f}, Kappa={kappa:.4f}")
    print(f"   分布: {dict(zip(LABEL_NAMES, [(labels_123 == i).sum() for i in range(4)]))}")

    cm_df = pd.DataFrame(cm, index=[f'银行{n}' for n in LABEL_NAMES],
                         columns=[f'新{n}' for n in LABEL_NAMES])
    print(cm_df.to_string())

    for i, name in enumerate(LABEL_NAMES):
        total_i = cm[i, :].sum()
        acc_i = cm[i, i] / total_i if total_i > 0 else 0
        print(f"   银行{name}→新{name}: {acc_i:.4f} ({cm[i,i]}/{total_i})")

    return ari, nmi, acc, kappa


def train_and_evaluate_rf(X_123, y_123, X_302, method_name):
    """训练RF、CV评估、预测302"""
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=5,
        random_state=42, n_jobs=-1
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(rf, X_123, y_123, cv=skf)

    cm = confusion_matrix(y_123, y_pred_cv, labels=[0, 1, 2, 3])
    acc = accuracy_score(y_123, y_pred_cv)
    kappa = cohen_kappa_score(y_123, y_pred_cv)

    print(f"\n   [{method_name}] RF 5折CV:")
    print(f"   准确率: {acc:.4f} ({int(acc*123)}/123), Kappa: {kappa:.4f}")
    for i, name in enumerate(LABEL_NAMES):
        total_i = cm[i, :].sum()
        acc_i = cm[i, i] / total_i if total_i > 0 else 0
        print(f"   {name}级: {acc_i:.4f} ({cm[i,i]}/{total_i})")

    # 全量训练
    rf.fit(X_123, y_123)
    proba_302 = rf.predict_proba(X_302)
    pred_302 = rf.predict(X_302)
    imp = rf.feature_importances_

    return acc, kappa, pred_302, proba_302, imp


# ============ 1. 读取数据 ============
print("=" * 60)
print("1. 读取数据...")
df_123 = pd.read_csv(f'{DIR_RESULTS}\\features_123_raw.csv')
df_302 = pd.read_csv(f'{DIR_RESULTS}\\features_302_raw.csv')
df_scores = pd.read_csv(f'{DIR_RESULTS}\\topsis_scores.csv')

X_123 = df_123[FEATURE_COLS].values
X_302 = df_302[FEATURE_COLS].values
C_123 = df_123[['企业代号']].merge(
    df_scores[['企业代号', 'TOPSIS得分']], on='企业代号', how='left', validate='one_to_one'
)['TOPSIS得分'].values
C_302 = df_302[['企业代号']].merge(
    df_scores[['企业代号', 'TOPSIS得分']], on='企业代号', how='left', validate='one_to_one'
)['TOPSIS得分'].values
if np.isnan(C_123).any() or np.isnan(C_302).any():
    raise ValueError('TOPSIS得分与企业代号未完全匹配，请先运行 problem2_topsis.py。')
bank_labels = df_123['信誉评级'].map(BANK_LABEL_MAP).values

print(f"   123家TOPSIS得分: [{C_123.min():.3f}, {C_123.max():.3f}]")
print(f"   302家TOPSIS得分: [{C_302.min():.3f}, {C_302.max():.3f}]")

# ============ 2. 分位数标签 ============
print("\n" + "=" * 60)
print("2. 分位数断点 (25%/50%/75%)")
print("=" * 60)
final_labels_123, final_labels_302, final_bps = quantile_labels(C_123, C_302)
ari, nmi, label_acc, label_kappa = evaluate_labels(
    final_labels_123, bank_labels, "分位数")
final_acc, final_kappa, final_pred_302, final_proba_302, final_imp = train_and_evaluate_rf(
    X_123, final_labels_123, X_302, "分位数")
use_method = '分位数'
print("\n   采用分位数断点：避免长尾TOPSIS得分导致K-Means产生极小等级簇。")

# ============ 3. 最终方案详细输出 ============
print("\n" + "=" * 60)
print(f"3. 最终方案: {use_method} — 详细结果")
print("=" * 60)

print(f"\n   断点: A≥{final_bps[0]:.4f} > B≥{final_bps[1]:.4f} > C≥{final_bps[2]:.4f} > D")

cnt_302 = [sum(final_pred_302 == i) for i in range(4)]
print(f"\n   302家评级分布:")
for i, name in enumerate(LABEL_NAMES):
    print(f"     {name}*级: {cnt_302[i]} 家 ({cnt_302[i]/302*100:.1f}%)")

# 预测302
df_pred = df_302[['企业代号']].copy()
df_pred['预测评级'] = [LABEL_NAMES[p] for p in final_pred_302]
for i, name in enumerate(LABEL_NAMES):
    df_pred[f'P({name})'] = final_proba_302[:, i]
df_pred['最大概率'] = final_proba_302.max(axis=1)
df_pred.to_csv(f'{DIR_RESULTS}\\rf_pred_302.csv', index=False, encoding='utf-8-sig')
print("   已保存 rf_pred_302.csv")

# 特征重要性
df_imp = pd.DataFrame({'特征': FEATURE_COLS, '重要性': final_imp}
                      ).sort_values('重要性', ascending=False).reset_index(drop=True)
df_imp['排名'] = df_imp.index + 1
df_imp.to_csv(f'{DIR_RESULTS}\\rf_feature_importance.csv', index=False, encoding='utf-8-sig')
print("   已保存 rf_feature_importance.csv")
print("\n   特征重要性:")
for _, row in df_imp.iterrows():
    print(f"     #{int(row['排名'])} {row['特征']}: {row['重要性']:.4f}")

# 302家特征画像
print("\n   302家各等级特征画像:")
profile_lines = []
for i, name in enumerate(LABEL_NAMES):
    mask = final_pred_302 == i
    n = mask.sum()
    if n > 0:
        means = X_302[mask].mean(axis=0)
        profile_lines.append(f"     {name}*: n={n} | " +
                             " | ".join(f"{f}={means[j]:.3f}" for j, f in enumerate(FEATURE_COLS)))
    else:
        profile_lines.append(f"     {name}*: n=0")
for line in profile_lines:
    print(line)

# 保存123家验证
df_val = pd.DataFrame({
    '企业代号': df_123['企业代号'],
    '银行评级': df_123['信誉评级'],
    '新评级': [LABEL_NAMES[l] for l in final_labels_123],
    'TOPSIS得分': C_123
})
df_val.to_csv(f'{DIR_RESULTS}\\topsis_label_comparison.csv', index=False, encoding='utf-8-sig')

# ============ 4. 可视化 ============
print("\n4. 生成可视化...")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 用分位数方案的标签
y_vis = final_labels_123
bp_vis = final_bps

# 图11: TOPSIS得分 → 4档
fig, ax = plt.subplots(figsize=(10, 5))
sorted_idx = np.argsort(C_123)
C_sorted = C_123[sorted_idx]
y_sorted = y_vis[sorted_idx]
for gid, gname in enumerate(LABEL_NAMES):
    mask = y_sorted == gid
    ax.scatter(np.arange(len(C_123))[mask], C_sorted[mask],
               c=GRADE_COLORS[gid], label=f'{gname}*', s=30, alpha=0.8)
for label, bp, color in zip(['Q75', 'Q50', 'Q25'], bp_vis, GRADE_COLORS[:3]):
    ax.axhline(y=bp, color=color, linestyle='--', linewidth=1.5, alpha=0.9)
    ax.text(len(C_123) - 1, bp + 0.006, f'{label}={bp:.4f}',
            color=color, ha='right', va='bottom', fontsize=9,
            bbox={'facecolor': 'white', 'edgecolor': color, 'alpha': 0.85, 'pad': 2})
ax.set_xlabel('企业排名(按TOPSIS得分升序)')
ax.set_ylabel('TOPSIS综合得分')
ax.legend()
plt.tight_layout()
plt.savefig(f'{DIR_FIGURES}\\fig11_topsis_clusters.png', dpi=150, bbox_inches='tight')
plt.close()
print("   已保存 fig11_topsis_clusters.png")

# 图12: 新标签 vs 银行标签
cm_final = confusion_matrix(bank_labels, y_vis, labels=[0, 1, 2, 3])
cm_final_df = pd.DataFrame(cm_final,
    index=[f'银行{n}' for n in LABEL_NAMES],
    columns=[f'新{n}' for n in LABEL_NAMES])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_final_df, annot=True, fmt='d', cmap='Purples', ax=ax,
            linewidths=0.5, cbar_kws={'label': '样本数'})
ax.set_xlabel('TOPSIS新标签')
ax.set_ylabel('银行原标签')
plt.tight_layout()
plt.savefig(f'{DIR_FIGURES}\\fig12_label_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   已保存 fig12_label_comparison.png")

# 图13: RF CV混淆矩阵
rf_final = RandomForestClassifier(
    n_estimators=200, max_depth=None, min_samples_split=5,
    random_state=42, n_jobs=-1
)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv_final = cross_val_predict(rf_final, X_123, y_vis, cv=skf)
cm_cv = confusion_matrix(y_vis, y_pred_cv_final, labels=[0, 1, 2, 3])
cm_cv_df = pd.DataFrame(cm_cv,
    index=[f'实际{n}' for n in LABEL_NAMES],
    columns=[f'预测{n}' for n in LABEL_NAMES])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_cv_df, annot=True, fmt='d', cmap='Blues', ax=ax,
            linewidths=0.5, cbar_kws={'label': '样本数'})
ax.set_xlabel('预测评级')
ax.set_ylabel('实际评级')
plt.tight_layout()
plt.savefig(f'{DIR_FIGURES}\\fig13_rf_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("   已保存 fig13_rf_confusion.png")

# 图14: 302家评级分布
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(LABEL_NAMES, cnt_302, color=GRADE_COLORS, edgecolor='black', linewidth=0.5)
ax.set_xlabel('预测信用评级')
ax.set_ylabel('企业数量')
for bar, val in zip(bars, cnt_302):
    pct = val/302*100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{val}\n({pct:.1f}%)', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(f'{DIR_FIGURES}\\fig14_302_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   已保存 fig14_302_distribution.png")

# 图15: 特征画像
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
x_idx = np.arange(len(FEATURE_COLS))
w_bar = 0.18

for ax_i, (Xdata, labels) in enumerate([
    (X_302, final_pred_302),
    (X_123, y_vis)
]):
    ax = axes[ax_i]
    for idx, (name, color) in enumerate(zip(LABEL_NAMES, GRADE_COLORS)):
        mask = labels == idx
        if mask.sum() > 0:
            means = Xdata[mask].mean(axis=0)
            means_std = (means - Xdata.mean(axis=0)) / (Xdata.std(axis=0) + 1e-8)
            ax.bar(x_idx + idx * w_bar, means_std, w_bar,
                   label=name, color=color, alpha=0.8)
    ax.set_xticks(x_idx + 1.5 * w_bar)
    ax.set_xticklabels(FEATURE_COLS, fontsize=8)
    ax.set_ylabel('标准化特征均值')
    ax.legend()
    ax.axhline(y=0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig(f'{DIR_FIGURES}\\fig15_feature_profile.png', dpi=150, bbox_inches='tight')
plt.close()
print("   已保存 fig15_feature_profile.png")

# 图16: 特征重要性
fig, ax = plt.subplots(figsize=(7, 5))
colors_imp = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(df_imp)))
bars = ax.barh(df_imp['特征'], df_imp['重要性'], color=colors_imp, edgecolor='black', linewidth=0.5)
ax.set_xlabel('特征重要性')
ax.invert_yaxis()
for bar, val in zip(bars, df_imp['重要性']):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(f'{DIR_FIGURES}\\fig16_rf_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("   已保存 fig16_rf_importance.png")

# ============ 5. 摘要 ============
print("\n" + "=" * 60)
print("问题二 建模完成！")
print(f"\n--- 标签划分方案: {use_method} ---")
print(f"  断点: {', '.join(f'{bp:.3f}' for bp in final_bps)}")
print(f"  123家分布: A={(y_vis==0).sum()}, B={(y_vis==1).sum()}, C={(y_vis==2).sum()}, D={(y_vis==3).sum()}")
print(f"  新标签 vs 银行: ARI={ari:.3f}, NMI={nmi:.3f}")

print(f"\n--- 随机森林 ---")
print(f"  5折CV: Acc={final_acc:.4f}, Kappa={final_kappa:.4f}")
print(f"  302家分布: A={cnt_302[0]}({cnt_302[0]/302*100:.1f}%), B={cnt_302[1]}({cnt_302[1]/302*100:.1f}%), "
      f"C={cnt_302[2]}({cnt_302[2]/302*100:.1f}%), D={cnt_302[3]}({cnt_302[3]/302*100:.1f}%)")
print(f"\n输出文件: {DIR_RESULTS}\\ 和 {DIR_FIGURES}\\")
