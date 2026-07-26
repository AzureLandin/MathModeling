# -*- coding: utf-8 -*-
"""比较TOPSIS有序断点方案的均衡性、稳定性与可学习性。"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

BASE = r'D:\建模\MathModeling\模块三练习'
DIR_RESULTS = f'{BASE}\\results'
FEATURE_COLS = ['F1', 'F2', 'F3', 'F4', 'F7', 'F8', 'F9', 'F10']
N_BOOTSTRAP = 300
RANDOM_STATE = 42


def quantile_thresholds(scores):
    q25, q50, q75 = np.percentile(scores, [25, 50, 75])
    return np.array([q75, q50, q25])


def equal_width_thresholds(scores):
    lower, upper = scores.min(), scores.max()
    width = (upper - lower) / 4
    return np.array([lower + 3 * width, lower + 2 * width, lower + width])


def kmeans_thresholds(scores):
    model = KMeans(n_clusters=4, n_init=20, random_state=RANDOM_STATE)
    model.fit(scores.reshape(-1, 1))
    centers = np.sort(model.cluster_centers_.ravel())
    boundaries = (centers[:-1] + centers[1:]) / 2
    return boundaries[::-1]


def assign(scores, thresholds):
    high, middle, low = thresholds
    return np.select(
        [scores >= high, scores >= middle, scores >= low],
        [0, 1, 2],
        default=3,
    ).astype(int)


def balance_metrics(labels):
    counts = np.bincount(labels, minlength=4)
    proportions = counts / counts.sum()
    entropy = -np.sum(proportions * np.log(proportions + 1e-12)) / np.log(4)
    target = len(labels) / 4
    squared_deviation = np.sum((counts - target) ** 2)
    return counts, counts.max() / counts.min(), entropy, squared_deviation


def rf_metrics(X, labels):
    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    predictions = cross_val_predict(model, X, labels, cv=cv)
    recalls = recall_score(labels, predictions, labels=[0, 1, 2, 3], average=None)
    return {
        'RF准确率': accuracy_score(labels, predictions),
        'RF平衡准确率': balanced_accuracy_score(labels, predictions),
        'RF宏F1': f1_score(labels, predictions, average='macro'),
        'RF_Kappa': cohen_kappa_score(labels, predictions),
        'RF最低类别召回率': recalls.min(),
    }


df_features = pd.read_csv(f'{DIR_RESULTS}\\features_123_raw.csv')
df_scores = pd.read_csv(f'{DIR_RESULTS}\\topsis_scores.csv')
scores = df_features[['企业代号']].merge(
    df_scores[['企业代号', 'TOPSIS得分']],
    on='企业代号',
    how='left',
    validate='one_to_one',
)['TOPSIS得分'].to_numpy()
X = df_features[FEATURE_COLS].to_numpy()

methods = {
    '分位数': quantile_thresholds,
    '一维K-Means': kmeans_thresholds,
    '等距区间': equal_width_thresholds,
}

summary_rows = []
baseline = {}
for name, threshold_fn in methods.items():
    thresholds = threshold_fn(scores)
    labels = assign(scores, thresholds)
    baseline[name] = (thresholds, labels)
    counts, imbalance, entropy, squared_deviation = balance_metrics(labels)
    row = {
        '方法': name,
        'A/B断点': thresholds[0],
        'B/C断点': thresholds[1],
        'C/D断点': thresholds[2],
        'A类数': counts[0],
        'B类数': counts[1],
        'C类数': counts[2],
        'D类数': counts[3],
        '最大最小类比': imbalance,
        '标准化类别熵': entropy,
        '类别数平方偏差': squared_deviation,
    }
    row.update(rf_metrics(X, labels))
    summary_rows.append(row)

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(
    f'{DIR_RESULTS}\\threshold_method_comparison.csv',
    index=False,
    encoding='utf-8-sig',
)

rng = np.random.default_rng(RANDOM_STATE)
bootstrap = {name: {'thresholds': [], 'ari': []} for name in methods}
for _ in range(N_BOOTSTRAP):
    sample = scores[rng.integers(0, len(scores), len(scores))]
    for name, threshold_fn in methods.items():
        thresholds = threshold_fn(sample)
        labels = assign(scores, thresholds)
        bootstrap[name]['thresholds'].append(thresholds)
        bootstrap[name]['ari'].append(adjusted_rand_score(baseline[name][1], labels))

stability_rows = []
for name, values in bootstrap.items():
    thresholds = np.asarray(values['thresholds'])
    ari_values = np.asarray(values['ari'])
    stability_rows.append({
        '方法': name,
        'A/B断点标准差': thresholds[:, 0].std(ddof=1),
        'B/C断点标准差': thresholds[:, 1].std(ddof=1),
        'C/D断点标准差': thresholds[:, 2].std(ddof=1),
        '平均断点标准差': thresholds.std(axis=0, ddof=1).mean(),
        '重抽样ARI均值': ari_values.mean(),
        '重抽样ARI标准差': ari_values.std(ddof=1),
        '重抽样ARI_5%分位': np.percentile(ari_values, 5),
    })

df_stability = pd.DataFrame(stability_rows)
df_stability.to_csv(
    f'{DIR_RESULTS}\\threshold_bootstrap_stability.csv',
    index=False,
    encoding='utf-8-sig',
)

print('断点方法比较:')
print(df_summary.round(4).to_string(index=False))
print('\n300次Bootstrap稳定性:')
print(df_stability.round(4).to_string(index=False))
