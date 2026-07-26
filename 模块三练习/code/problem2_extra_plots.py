# -*- coding: utf-8 -*-
"""
问题二补充绘图: fig13轮廓系数 / fig14标签对比混淆矩阵 / fig15置信度分布
并修复编号冲突: fig10_entropy_weights.png -> fig8b_entropy_weights.png
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

DIR_RESULTS = r'E:\MathModeling\模块三练习\results'
DIR_FIGURES = r'E:\MathModeling\模块三练习\figures'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============ 0. 修复编号冲突 ============
old = f'{DIR_FIGURES}\\fig10_entropy_weights.png'
new = f'{DIR_FIGURES}\\fig8b_entropy_weights.png'
if os.path.exists(old):
    os.rename(old, new)
    print("0. 已重命名 fig10_entropy_weights.png -> fig8b_entropy_weights.png")
else:
    print("0. fig10_entropy_weights.png 不存在, 跳过")

# ============ 图1: 轮廓系数折线图 ============
print("\n1. 绘制 fig13_silhouette_by_k.png ...")
df_sil = pd.read_csv(f'{DIR_RESULTS}\\silhouette_by_k_p2.csv')
ks = df_sil['k'].values
sils = df_sil['轮廓系数'].values

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ks, sils, 'b-o', markersize=8, linewidth=2, label='轮廓系数')
# k=4 红色虚线
ax.axvline(x=4, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
ax.text(4.05, ax.get_ylim()[0] + 0.01, 'k=4（业务约束）', color='red',
        fontsize=10, va='bottom')
# k=3 统计最优标注
idx3 = np.argmax(sils)
ax.annotate('统计最优', xy=(ks[idx3], sils[idx3]),
            xytext=(ks[idx3] - 0.35, sils[idx3] + 0.03),
            fontsize=10, color='green',
            arrowprops=dict(arrowstyle='->', color='green'))
ax.scatter([ks[idx3]], [sils[idx3]], s=120, facecolors='none', edgecolors='green', linewidths=2)
ax.set_xlabel('聚类数目 k')
ax.set_ylabel('轮廓系数')
ax.set_title('不同聚类数目的轮廓系数')
ax.set_xticks(ks)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{DIR_FIGURES}\\fig13_silhouette_by_k.png', dpi=150, bbox_inches='tight')
plt.close()
print("   已保存 fig13_silhouette_by_k.png")

# ============ 图2: 新标签 vs 银行标签混淆矩阵 ============
print("\n2. 绘制 fig14_label_comparison.png ...")
df_lab = pd.read_csv(f'{DIR_RESULTS}\\new_labels_123.csv')
bank_order = ['A', 'B', 'C', 'D']
new_order = ['A*', 'B*', 'C*', 'D*']
y_bank = df_lab['信誉评级'].values
y_new = df_lab['新标签'].values
# 手动计算混淆矩阵(银行标签A/B/C/D 对齐 新标签A*/B*/C*/D*)
cm = np.zeros((4, 4), dtype=int)
for i, b in enumerate(bank_order):
    for j, nv in enumerate(new_order):
        cm[i, j] = int(((y_bank == b) & (y_new == nv)).sum())
print("   混淆矩阵:")
print(cm)

# 自定义着色: 非对角线用Blues, 对角线用深蓝突出
norm = cm / cm.max()
blues = plt.get_cmap('Blues')
img = np.zeros((4, 4, 3))
dark_blue = np.array([0.03, 0.19, 0.42])  # 深蓝
for i in range(4):
    for j in range(4):
        if i == j:
            img[i, j] = dark_blue
        else:
            # 非对角线: 浅蓝缩放(避免太深), 0.1~0.7区间
            val = 0.1 + 0.6 * norm[i, j]
            img[i, j] = blues(val)[:3]

fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(img, aspect='equal')
# 标注数字: 对角线白字, 非对角线黑字
for i in range(4):
    for j in range(4):
        color = 'white' if i == j else 'black'
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                fontsize=14, fontweight='bold', color=color)
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(new_order)
ax.set_yticklabels(bank_order)
ax.set_xlabel('新标签（熵权加权K-Means）')
ax.set_ylabel('银行原标签')
ax.set_title('新标签与银行标签对比（ARI=0.039）')
# 网格线
for i in range(5):
    ax.axhline(i - 0.5, color='white', linewidth=1)
    ax.axvline(i - 0.5, color='white', linewidth=1)
plt.tight_layout()
plt.savefig(f'{DIR_FIGURES}\\fig14_label_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   已保存 fig14_label_comparison.png")

# ============ 图3: 302家预测置信度分布 ============
print("\n3. 绘制 fig15_confidence_dist.png ...")
df_pred = pd.read_csv(f'{DIR_RESULTS}\\p2_prediction_302.csv')
conf = df_pred['置信度'].values
mean_c = conf.mean()
median_c = np.median(conf)
print(f"   置信度: 均值={mean_c:.3f}, 中位数={median_c:.3f}")

fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(0, 1, 21)  # 20 bins
counts, edges, patches = ax.hist(conf, bins=bins, color='lightblue',
                                  alpha=0.7, edgecolor='gray', label='企业数量')
# KDE缩放到计数尺度
bin_width = edges[1] - edges[0]
kde = gaussian_kde(conf)
x_kde = np.linspace(0, 1, 300)
y_kde = kde(x_kde) * len(conf) * bin_width
ax.plot(x_kde, y_kde, color='darkblue', linewidth=2, label='核密度曲线')
# 阈值线
ax.axvline(x=0.7, color='red', linestyle='--', linewidth=1.5)
ax.text(0.7, ax.get_ylim()[1] * 0.92, '高置信阈值', color='red', fontsize=10, ha='center')
ax.axvline(x=0.5, color='orange', linestyle='--', linewidth=1.5)
ax.text(0.5, ax.get_ylim()[1] * 0.82, '低置信阈值', color='orange', fontsize=10, ha='center')
# 均值/中位数标注
ax.text(0.03, 0.95, f'均值={mean_c:.3f}\n中位数={median_c:.3f}',
        transform=ax.transAxes, fontsize=11, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.set_xlabel('预测置信度（最大类别概率）')
ax.set_ylabel('企业数量')
ax.set_title('302家企业预测置信度分布')
ax.legend(loc='upper center')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{DIR_FIGURES}\\fig15_confidence_dist.png', dpi=150, bbox_inches='tight')
plt.close()
print("   已保存 fig15_confidence_dist.png")

print("\n" + "=" * 60)
print("补充绘图完成！")
