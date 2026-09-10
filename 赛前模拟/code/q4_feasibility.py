"""问题 4：女胎 AB 标签是否可由附件特征识别的可行性 Baseline。

输入：results/female_prepared.csv。
输出：results/q4_feasibility_summary.json、results/q4_group_cv_scores.csv，及
      results/q4_threshold_sensitivity.csv、results/q4_full_model_coefficients.csv，及
      figures/q4_label_separability.png。
标签：AB 列非空记为 1，空白记为 0；这表示附件中的“检出非整倍体”，并非出生健康结局。
模型：标准化 Logistic 回归。比较 Z 值核心模型与加入 X、GC、读段质量和 BMI 的扩展模型。
验证：以孕妇代码分组的五折交叉验证，避免同一孕妇重复检测泄漏到训练和测试两侧。
限制：本脚本检验“可预测性”，而不将 AB 非空直接解释为临床真值。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "results" / "female_prepared.csv"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
RANDOM_SEED = 20250908
N_SPLITS = 5
Z_FEATURES = ["13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值"]
EXTENDED_FEATURES = Z_FEATURES + [
    "X染色体的Z值",
    "X染色体浓度",
    "GC含量",
    "13号染色体的GC含量",
    "18号染色体的GC含量",
    "21号染色体的GC含量",
    "原始读段数",
    "在参考基因组上比对的比例",
    "重复读段的比例",
    "唯一比对的读段数",
    "被过滤掉读段数的比例",
    "孕妇BMI",
]


def grouped_cv(data: pd.DataFrame, features: list[str], model_name: str) -> tuple[pd.DataFrame, np.ndarray]:
    complete = data.dropna(subset=features).copy()
    y = complete["abnormal"].astype(int).to_numpy()
    groups = complete["孕妇代码"].to_numpy()
    x = complete[features]
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    probabilities = np.full(len(complete), np.nan)
    records: list[dict[str, object]] = []
    for fold, (train_index, test_index) in enumerate(splitter.split(x, y, groups), start=1):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
        )
        model.fit(x.iloc[train_index], y[train_index])
        prediction = model.predict_proba(x.iloc[test_index])[:, 1]
        probabilities[test_index] = prediction
        records.append(
            {
                "model": model_name,
                "fold": fold,
                "test_rows": int(len(test_index)),
                "test_abnormal_rows": int(y[test_index].sum()),
                "test_mothers": int(pd.Series(groups[test_index]).nunique()),
                "roc_auc": float(roc_auc_score(y[test_index], prediction)),
                "average_precision": float(average_precision_score(y[test_index], prediction)),
                "brier_score": float(brier_score_loss(y[test_index], prediction)),
            }
        )
    return pd.DataFrame(records), probabilities


def make_figure(y: np.ndarray, probabilities: dict[str, np.ndarray]) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)
    for name, values in probabilities.items():
        fpr, tpr, _ = roc_curve(y, values)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc_score(y, values):.3f})")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="random")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Grouped-CV label separability", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    fig.savefig(FIGURES_DIR / "q4_label_separability.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def threshold_sensitivity(y: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for threshold in np.arange(0.05, 0.51, 0.05):
        predicted = probability >= threshold
        tp = int(np.sum((predicted == 1) & (y == 1)))
        fp = int(np.sum((predicted == 1) & (y == 0)))
        tn = int(np.sum((predicted == 0) & (y == 0)))
        fn = int(np.sum((predicted == 0) & (y == 1)))
        sensitivity = tp / (tp + fn) if tp + fn else np.nan
        specificity = tn / (tn + fp) if tn + fp else np.nan
        precision = tp / (tp + fp) if tp + fp else np.nan
        rows.append(
            {
                "threshold": float(threshold),
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "precision": precision,
                "youden_j": sensitivity + specificity - 1,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(INPUT_PATH)
    data["abnormal"] = data["染色体的非整倍体"].notna().astype(int)
    data = data.dropna(subset=EXTENDED_FEATURES).copy()
    data["max_abs_autosomal_z"] = data[Z_FEATURES].abs().max(axis=1)
    y = data["abnormal"].to_numpy()
    z_scores, z_probability = grouped_cv(data, Z_FEATURES, "autosomal_z")
    extended_scores, extended_probability = grouped_cv(data, EXTENDED_FEATURES, "extended")
    scores = pd.concat([z_scores, extended_scores], ignore_index=True)
    scores.to_csv(RESULTS_DIR / "q4_group_cv_scores.csv", index=False, encoding="utf-8-sig")
    thresholds = threshold_sensitivity(y, extended_probability)
    thresholds.to_csv(RESULTS_DIR / "q4_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")

    full_model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
    ).fit(data[EXTENDED_FEATURES], y)
    pd.DataFrame(
        {
            "feature": EXTENDED_FEATURES,
            "standardized_logistic_coefficient": full_model[-1].coef_[0],
        }
    ).to_csv(RESULTS_DIR / "q4_full_model_coefficients.csv", index=False, encoding="utf-8-sig")

    mixed_label_mothers = int((data.groupby("孕妇代码")["abnormal"].nunique() > 1).sum())
    rule_positive = data["max_abs_autosomal_z"] >= 3
    summary = {
        "label_definition": "AB non-empty = 1; AB blank = 0",
        "rows": int(len(data)),
        "abnormal_rows": int(data["abnormal"].sum()),
        "abnormal_mothers": int(data.loc[data["abnormal"] == 1, "孕妇代码"].nunique()),
        "mothers_with_mixed_row_labels": mixed_label_mothers,
        "cross_validation": {
            "method": "5-fold StratifiedGroupKFold by pregnant-woman code",
            "models": {
                name: {
                    "mean_roc_auc": float(frame["roc_auc"].mean()),
                    "mean_average_precision": float(frame["average_precision"].mean()),
                    "mean_brier_score": float(frame["brier_score"].mean()),
                }
                for name, frame in {"autosomal_z": z_scores, "extended": extended_scores}.items()
            },
        },
        "threshold_note": "Threshold performance is calculated from out-of-fold probabilities, but selecting a threshold on the same attachment remains exploratory.",
        "best_grid_youden_threshold": float(
            thresholds.loc[thresholds["youden_j"].idxmax(), "threshold"]
        ),
        "standard_z_rule_max_abs_z_ge_3": {
            "true_positive_rate": float(rule_positive[y == 1].mean()),
            "false_positive_rate": float(rule_positive[y == 0].mean()),
        },
        "interpretation_limit": "Performance assesses consistency with AB labels in this attachment, not clinical diagnostic sensitivity or specificity.",
    }
    (RESULTS_DIR / "q4_feasibility_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_figure(y, {"autosomal Z": z_probability, "extended": extended_probability})
    print("Question 4 feasibility outputs written to results/ and figures/.")


if __name__ == "__main__":
    main()
