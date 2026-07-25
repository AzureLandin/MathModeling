# -*- coding: utf-8 -*-
"""
问题一 - XGBoost辅助验证 (修订版, 动态适配特征数)
输入: results/features_123.csv
输出: results/xgb_confusion_matrix.csv, results/xgb_feature_importance.csv
      figures/fig5_xgb_confusion.png, figures/fig6_xgb_importance.png
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============ 路径配置 ============
DIR_RESULTS = r'D:\建模\MathModeling\模块三练习\results'
DIR_FIGURES = r'D:\建模\MathModeling\模块三练习\figures'

# ============ 1. 读取特征 ============
print("=" * 60)
print("1. 读取特征数据...")
df = pd.read_csv(f'{DIR_RESULTS}\\features_123.csv')
print(f"   样本数: {len(df)}, 列: {list(df.columns)}")

feature_cols = [c for c in df.columns if c not in ['企业代号', '信誉评级']]
print(f"   入模特征({len(feature_cols)}个): {feature_cols}")

LABEL_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
LABEL_NAMES = ['A', 'B', 'C', 'D']

X = df[feature_cols].values
y = df['信誉评级'].map(LABEL_MAP).values

# ============ 2. XGBoost 5折交叉验证 ============
print("\n2. XGBoost 5折交叉验证...")
clf = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    eval_metric='mlogloss',
    random_state=42,
    use_label_encoder=False
)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(clf, X, y, cv=skf)

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

cm_df.to_csv(f'{DIR_RESULTS}\\xgb_confusion_matrix.csv', encoding='utf-8-sig')
print("\n   已保存 xgb_confusion_matrix.csv")

# ============ 3. 特征重要性 ============
print("\n3. 训练全量模型获取特征重要性...")
clf.fit(X, y)
importance = clf.feature_importances_
df_imp = pd.DataFrame({
    '特征': feature_cols,
    '重要性': importance
}).sort_values('重要性', ascending=False).reset_index(drop=True)
df_imp['排名'] = df_imp.index + 1
df_imp.to_csv(f'{DIR_RESULTS}\\xgb_feature_importance.csv', index=False, encoding='utf-8-sig')
print("   已保存 xgb_feature_importance.csv")
print("\n   特征重要性排序:")
for _, row in df_imp.iterrows():
    print(f"     #{int(row['排名'])} {row['特征']}: {row['重要性']:.4f}")

# ============ 4. 可视化 ============
print("\n4. 生成可视化...")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 图5: XGBoost混淆矩阵
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_df.astype(int), annot=True, fmt='d', cmap='Oranges', ax=ax,
            linewidths=0.5, cbar_kws={'label': '样本数'})
ax.set_title(f'XGBoost混淆矩阵 (Acc={acc:.2%}, Kappa={kappa:.3f})')
ax.set_xlabel('预测评级')
ax.set_ylabel('实际评级')
plt.tight_layout()
plt.savefig(f'{DIR_FIGURES}\\fig5_xgb_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("   已保存 fig5_xgb_confusion.png")

# 图6: 特征重要性条形图
fig, ax = plt.subplots(figsize=(7, 5))
colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(df_imp)))
bars = ax.barh(df_imp['特征'], df_imp['重要性'], color=colors)
ax.set_xlabel('重要性 (gain)')
ax.set_title('XGBoost特征重要性排序')
ax.invert_yaxis()
for bar, val in zip(bars, df_imp['重要性']):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(f'{DIR_FIGURES}\\fig6_xgb_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("   已保存 fig6_xgb_importance.png")

print("\n" + "=" * 60)
print("XGBoost辅助验证完成！")
