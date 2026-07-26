# -*- coding: utf-8 -*-
"""
问题二 - 步骤三: 有监督分类预测 (Logit vs XGBoost, 方案A/B对比, 预测302家)
输入: results/new_labels_123.csv, features_123_raw/full.csv, features_302_raw/full.csv
      results/p2_std_params.csv
输出: results/p2_classifier_cv.csv, results/p2_feature_importance.csv
      results/p2_prediction_302.csv
      figures/fig9_p2_importance.png, figures/fig10_p2_prediction.png
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

DIR_RESULTS = r'E:\MathModeling\模块三练习\results'
DIR_FIGURES = r'E:\MathModeling\模块三练习\figures'

feature_8 = ['F1', 'F2', 'F3', 'F4', 'F7', 'F8', 'F9', 'F10']
cols_19 = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8',
           'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16',
           'x17', 'x18', 'x19']
feature_23 = cols_19 + ['F7', 'F8', 'F9', 'F10']
LABELS = ['A*', 'B*', 'C*', 'D*']

# ============ 1. 读取数据 ============
print("=" * 60)
print("1. 读取数据...")
df_123 = pd.read_csv(f'{DIR_RESULTS}\\new_labels_123.csv')
df_123_full = pd.read_csv(f'{DIR_RESULTS}\\features_123_full.csv')
df_302_raw = pd.read_csv(f'{DIR_RESULTS}\\features_302_raw.csv')
df_302_full = pd.read_csv(f'{DIR_RESULTS}\\features_302_full.csv')

y = df_123['新标签'].values
# XGBoost需要数值标签
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder().fit(LABELS)
y_num = le.transform(y)
print(f"   123家新标签分布: {dict(zip(*np.unique(y, return_counts=True)))}")

# ============ 2. 特征标准化(用123家参数) ============
print("\n2. 特征标准化...")
# 方案A: 8特征用z-score(StandardScaler, 123家拟合)
scaler_A = StandardScaler().fit(df_123[feature_8].values)
X_A_123 = scaler_A.transform(df_123[feature_8].values)
X_A_302 = scaler_A.transform(df_302_raw[feature_8].values)

# 方案B: 23特征用z-score
scaler_B = StandardScaler().fit(df_123_full[feature_23].values)
X_B_123 = scaler_B.transform(df_123_full[feature_23].values)
X_B_302 = scaler_B.transform(df_302_full[feature_23].values)
print("   方案A(8维)和方案B(23维)已用123家参数标准化")

# ============ 3. 四组合5折CV对比 ============
print("\n3. 四组合5折交叉验证对比...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

configs = {
    'Logit-A(8维)': (LogisticRegression(multi_class='multinomial', class_weight='balanced',
                                         max_iter=2000, random_state=42), X_A_123, feature_8, X_A_302, y),
    'Logit-B(23维)': (LogisticRegression(multi_class='multinomial', class_weight='balanced',
                                          max_iter=2000, random_state=42), X_B_123, feature_23, X_B_302, y),
    'XGBoost-A(8维)': (XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                      eval_metric='mlogloss', random_state=42,
                                      use_label_encoder=False), X_A_123, feature_8, X_A_302, y_num),
    'XGBoost-B(23维)': (XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                       eval_metric='mlogloss', random_state=42,
                                       use_label_encoder=False), X_B_123, feature_23, X_B_302, y_num),
}

results = []
for name, (clf, X, feats, X302, y_use) in configs.items():
    y_pred = cross_val_predict(clf, X, y_use, cv=skf)
    # XGBoost预测为数值, 映射回字符串以便统一计算
    if y_use is y_num:
        y_pred_cmp = le.inverse_transform(y_pred)
        y_true_cmp = y
    else:
        y_pred_cmp = y_pred
        y_true_cmp = y
    acc = accuracy_score(y_true_cmp, y_pred_cmp)
    kappa = cohen_kappa_score(y_true_cmp, y_pred_cmp)
    results.append({'模型': name, '准确率': acc, 'Kappa': kappa})
    print(f"   {name}: 准确率={acc:.4f}, Kappa={kappa:.4f}")

df_cv = pd.DataFrame(results).sort_values('准确率', ascending=False).reset_index(drop=True)
df_cv.to_csv(f'{DIR_RESULTS}\\p2_classifier_cv.csv', index=False, encoding='utf-8-sig')
print("\n   已保存 p2_classifier_cv.csv")
print(df_cv.to_string(index=False))

# 选最优
best_name = df_cv.iloc[0]['模型']
best_clf, X_best_123, best_feats, X_best_302, y_best = configs[best_name]
is_xgb = (y_best is y_num)
print(f"\n   >>> 最优模型: {best_name}")

# ============ 4. 训练最优模型 + 特征重要性 ============
print(f"\n4. 训练最优模型并提取重要性...")
best_clf.fit(X_best_123, y_best)

if hasattr(best_clf, 'feature_importances_'):
    imp = best_clf.feature_importances_
else:
    # Logit: 用系数绝对值均值作为重要性
    imp = np.abs(best_clf.coef_).mean(axis=0)
    imp = imp / imp.sum()

df_imp = pd.DataFrame({'特征': best_feats, '重要性': imp})
df_imp = df_imp.sort_values('重要性', ascending=False).reset_index(drop=True)
df_imp['排名'] = df_imp.index + 1
df_imp.to_csv(f'{DIR_RESULTS}\\p2_feature_importance.csv', index=False, encoding='utf-8-sig')
print("   已保存 p2_feature_importance.csv")
print("\n   特征重要性Top10:")
print(df_imp.head(10).to_string(index=False))

# ============ 5. 预测302家 ============
print("\n5. 预测302家企业...")
probs_302 = best_clf.predict_proba(X_best_302)
pred_302_raw = best_clf.predict(X_best_302)
# XGBoost输出数值, 映射回字符串
if is_xgb:
    pred_302 = le.inverse_transform(pred_302_raw)
    class_labels = le.classes_
else:
    pred_302 = pred_302_raw
    class_labels = best_clf.classes_
confidence = probs_302.max(axis=1)

df_pred = df_302_full[['企业代号']].copy()
df_pred['信用评级'] = pred_302
df_pred['置信度'] = confidence
for i, cls in enumerate(class_labels):
    df_pred[f'P({cls})'] = probs_302[:, i]
df_pred.to_csv(f'{DIR_RESULTS}\\p2_prediction_302.csv', index=False, encoding='utf-8-sig')
print("   已保存 p2_prediction_302.csv")

print(f"\n   302家评级分布:")
dist = pd.Series(pred_302).value_counts().reindex(LABELS).fillna(0).astype(int)
print(dist.to_string())
print(f"\n   置信度: 均值={confidence.mean():.3f}, 中位数={np.median(confidence):.3f}")
print(f"   高置信(>0.7): {(confidence>0.7).sum()} 家, 低置信(<0.5): {(confidence<0.5).sum()} 家")

# 画像(用8维原始特征)
print("\n   各等级特征画像(8维原始均值):")
df_prof = pd.concat([pd.DataFrame({'信用评级': pred_302}),
                     df_302_raw[feature_8].reset_index(drop=True)], axis=1)
print(df_prof.groupby('信用评级')[feature_8].mean().reindex(LABELS).round(3).to_string())

# ============ 6. 可视化 ============
print("\n6. 生成可视化...")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 图9: 特征重要性
fig, ax = plt.subplots(figsize=(8, 6))
top_n = min(12, len(df_imp))
df_plot = df_imp.head(top_n)
colors = plt.cm.YlGn(np.linspace(0.3, 0.9, top_n))
bars = ax.barh(df_plot['特征'], df_plot['重要性'], color=colors)
ax.set_xlabel('重要性')
ax.set_title(f'特征重要性 ({best_name})')
ax.invert_yaxis()
for bar, val in zip(bars, df_plot['重要性']):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{DIR_FIGURES}\\fig9_p2_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("   已保存 fig9_p2_importance.png")

# 图10: 302家评级分布
fig, ax = plt.subplots(figsize=(7, 5))
colors_rating = {'A*': '#2ecc71', 'B*': '#3498db', 'C*': '#f39c12', 'D*': '#e74c3c'}
counts = [int(dist.get(r, 0)) for r in LABELS]
bars = ax.bar(LABELS, counts, color=[colors_rating[r] for r in LABELS])
ax.set_xlabel('信用评级')
ax.set_ylabel('企业数量')
ax.set_title(f'302家企业信用评级预测结果 ({best_name})')
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(cnt), ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{DIR_FIGURES}\\fig10_p2_prediction.png', dpi=150, bbox_inches='tight')
plt.close()
print("   已保存 fig10_p2_prediction.png")

print("\n" + "=" * 60)
print("问题二分类预测完成！")
