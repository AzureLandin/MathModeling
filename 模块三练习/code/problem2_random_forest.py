# -*- coding: utf-8 -*-
"""
问题二 - K-Means标签划分 + 随机森林分类预测 (步骤一后半段 + 步骤二)
输入: results/topsis_scores.csv, results/features_123_raw.csv, results/features_302_raw.csv
输出: results/rf_pred_302.csv, results/rf_feature_importance.csv
      figures/fig11-16

标签划分: 两种方案对比
  方案1(K-Means): 1维TOPSIS得分上K-Means(k=4) — 数据驱动断点
  方案2(分位数): 按TOPSIS得分25/50/75分位数断点 — 均衡分组
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
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


def kmeans_labels(scores_123, scores_302):
    """方案1: K-Means (k=4) 在1维得分上"""
    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    c123 = kmeans.fit_predict(scores_123.reshape(-1, 1))
    c302 = kmeans.predict(scores_302.reshape(-1, 1))

    centers = kmeans.cluster_centers_.ravel()
    rank = np.argsort(np.argsort(-centers))
    grade_map = {i: LABEL_MAP[r] for i, r in enumerate(rank)}

    labels_123 = np.array([BANK_LABEL_MAP[grade_map[c]] for c in c123])
    labels_302 = np.array([BANK_LABEL_MAP[grade_map[c]] for c in c302])

    # 断点
    bps = []
    for g in ['A', 'B', 'C']:
        gid = BANK_LABEL_MAP[g]
        nid = gid + 1
        mx = scores_123[labels_123 == gid].max() if (labels_123 == gid).any() else 0
        mn = scores_123[labels_123 == nid].min() if (labels_123 == nid).any() else 1
        bps.append((mx + mn) / 2)

    return labels_123, labels_302, bps, centers, rank


def quantile_labels(scores_123, scores_302):
    """方案2: 基于123家TOPSIS得分的25%/50%/75%分位数作为断点"""
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
C_123 = df_scores[df_scores['企业代号'].isin(df_123['企业代号'])]['TOPSIS得分'].values
C_302 = df_scores[df_scores['企业代号'].isin(df_302['企业代号'])]['TOPSIS得分'].values
bank_labels = df_123['信誉评级'].map(BANK_LABEL_MAP).values

print(f"   123家TOPSIS得分: [{C_123.min():.3f}, {C_123.max():.3f}]")
print(f"   302家TOPSIS得分: [{C_302.min():.3f}, {C_302.max():.3f}]")

# ============ 2. 方案1: K-Means ============
print("\n" + "=" * 60)
print("2. 方案1: K-Means (k=4) 断点划分")
print("=" * 60)
l1_123, l1_302, bp1, centers, rank = kmeans_labels(C_123, C_302)
ari1, nmi1, acc1, kap1 = evaluate_labels(l1_123, bank_labels, "K-Means")
acc1_rf, kap1_rf, pred1_302, proba1_302, imp1 = train_and_evaluate_rf(
    X_123, l1_123, X_302, "K-Means")

# ============ 3. 方案2: 分位数 ============
print("\n" + "=" * 60)
print("3. 方案2: 分位数断点 (25%/50%/75%)")
print("=" * 60)
l2_123, l2_302, bp2 = quantile_labels(C_123, C_302)
ari2, nmi2, acc2, kap2 = evaluate_labels(l2_123, bank_labels, "分位数")
acc2_rf, kap2_rf, pred2_302, proba2_302, imp2 = train_and_evaluate_rf(
    X_123, l2_123, X_302, "分位数")

# ============ 4. 选择最优方案 ============
print("\n" + "=" * 60)
print("4. 方案比较与选择")
print("=" * 60)

print(f"\n   {'指标':<20} {'K-Means':>12} {'分位数':>12}")
print(f"   {'-'*44}")
print(f"   {'ARI(新标签vs银行)':<20} {ari1:>12.4f} {ari2:>12.4f}")
print(f"   {'NMI(新标签vs银行)':<20} {nmi1:>12.4f} {nmi2:>12.4f}")
print(f"   {'RF CV准确率':<20} {acc1_rf:>12.4f} {acc2_rf:>12.4f}")
print(f"   {'RF CV Kappa':<20} {kap1_rf:>12.4f} {kap2_rf:>12.4f}")

# 选择RF CV准确率更高的方案
if acc2_rf >= acc1_rf:
    use_method = '分位数'
    final_labels_123 = l2_123
    final_labels_302 = l2_302
    final_bps = bp2
    final_imp = imp2
    final_acc = acc2_rf
    final_kappa = kap2_rf
    final_pred_302 = pred2_302
    final_proba_302 = proba2_302
    print(f"\n   → 选择方案2(分位数), RF CV准确率={final_acc:.4f}")
else:
    use_method = 'K-Means'
    final_labels_123 = l1_123
    final_labels_302 = l1_302
    final_bps = bp1
    final_imp = imp1
    final_acc = acc1_rf
    final_kappa = kap1_rf
    final_pred_302 = pred2_302  # 仍然用分位数预测302
    final_proba_302 = proba2_302
    print(f"\n   → 选择方案1(K-Means), RF CV准确率={final_acc:.4f}")

# 强制使用分位数方案(因为分布更合理)
use_method = '分位数'
final_labels_123 = l2_123
final_labels_302 = l2_302
final_bps = bp2
final_imp = imp2
final_acc = acc2_rf
final_kappa = kap2_rf
final_pred_302 = pred2_302
final_proba_302 = proba2_302
print(f"\n   → 最终选用: {use_method} (分布均衡, 符合信用评级惯例)")

# ============ 5. 最终方案详细输出 ============
print("\n" + "=" * 60)
print(f"5. 最终方案: {use_method} — 详细结果")
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

# ============ 6. 可视化 ============
print("\n6. 生成可视化...")
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
for bp in bp_vis:
    ax.axhline(y=bp, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax.set_xlabel('企业排名(按TOPSIS得分升序)')
ax.set_ylabel('TOPSIS综合得分')
ax.set_title(f'TOPSIS得分 {use_method}划分结果')
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
ax.set_title(f'新标签 vs 银行标签 (ARI={ari2:.3f}, 吻合率={acc2:.2%})')
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
ax.set_title(f'RF 5折CV混淆矩阵 ({use_method})\nAcc={final_acc:.2%}, Kappa={final_kappa:.3f}')
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
ax.set_title(f'302家无信贷记录企业预测评级分布 ({use_method})')
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

for ax_i, (Xdata, labels, title) in enumerate([
    (X_302, final_pred_302, '302家企业各等级特征画像'),
    (X_123, y_vis, '123家企业新标签特征画像')
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
    ax.set_title(title)
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
ax.set_title(f'随机森林特征重要性排序 ({use_method})')
ax.invert_yaxis()
for bar, val in zip(bars, df_imp['重要性']):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(f'{DIR_FIGURES}\\fig16_rf_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("   已保存 fig16_rf_importance.png")

# ============ 7. 摘要 ============
print("\n" + "=" * 60)
print("问题二 建模完成！")
print(f"\n--- 标签划分方案: {use_method} ---")
print(f"  断点: {', '.join(f'{bp:.3f}' for bp in final_bps)}")
print(f"  123家分布: A={(y_vis==0).sum()}, B={(y_vis==1).sum()}, C={(y_vis==2).sum()}, D={(y_vis==3).sum()}")
print(f"  新标签 vs 银行: ARI={ari2:.3f}, NMI={nmi2:.3f}")

print(f"\n--- 随机森林 ---")
print(f"  5折CV: Acc={final_acc:.4f}, Kappa={final_kappa:.4f}")
print(f"  302家分布: A={cnt_302[0]}({cnt_302[0]/302*100:.1f}%), B={cnt_302[1]}({cnt_302[1]/302*100:.1f}%), "
      f"C={cnt_302[2]}({cnt_302[2]/302*100:.1f}%), D={cnt_302[3]}({cnt_302[3]/302*100:.1f}%)")
print(f"\n输出文件: {DIR_RESULTS}\\ 和 {DIR_FIGURES}\\")
