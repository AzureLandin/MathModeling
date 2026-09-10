"""问题 1：生理特征 XGBoost 回归与分组特征重要性验证。

输入：results/male_prepared.csv。
输出：results/q1_xgboost_cv_metrics.csv、results/q1_xgboost_oof_predictions.csv、
      results/q1_xgboost_feature_importance.csv、results/q1_xgboost_summary.json，
      figures/q1_xgboost_feature_importance.png。
目标：预测 Y 染色体浓度（百分点）。特征仅包括孕周、BMI、年龄、身高、IVF 方式、
      怀孕次数和生产次数；不同时纳入体重与 BMI，不纳入测序质量变量。
验证：以孕妇代码 GroupKFold 五折分组，避免同一孕妇重复检测泄漏；训练折内拟合模型，
      测试折评估 R²、RMSE、MAE。重要性以测试折置换后 RMSE 的上升为主，树的 gain 为辅。
限制：重要性是预测贡献，不表示因果效应；XGBoost 为解释力比较模型，非最终机制模型。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "results" / "male_prepared.csv"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
RANDOM_SEED = 20250908
N_SPLITS = 5
NUMERIC_FEATURES = ["gestational_week", "孕妇BMI", "年龄", "身高"]
CATEGORICAL_FEATURES = ["IVF妊娠", "怀孕次数", "生产次数"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
DISPLAY_NAMES = {
    "gestational_week": "Gestational week",
    "孕妇BMI": "BMI",
    "年龄": "Maternal age",
    "身高": "Height",
    "IVF妊娠": "Conception method",
    "怀孕次数": "Gravidity",
    "生产次数": "Parity",
}
MODEL_PARAMETERS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 2,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "objective": "reg:squarederror",
    "importance_type": "gain",
    "n_jobs": 1,
    "random_state": RANDOM_SEED,
}


def make_pipeline() -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [("preprocessor", preprocessing), ("model", XGBRegressor(**MODEL_PARAMETERS))]
    )


def aggregate_gain(pipeline: Pipeline) -> dict[str, float]:
    processed_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    raw_gain = pipeline.named_steps["model"].feature_importances_
    gain = {feature: 0.0 for feature in FEATURES}
    for name, value in zip(processed_names, raw_gain, strict=True):
        source = next((feature for feature in CATEGORICAL_FEATURES if name.startswith(f"{feature}_")), name)
        gain[source] += float(value)
    return gain


def make_importance_figure(importance: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    plot_data = importance.sort_values("permutation_rmse_increase_mean").copy()
    plot_data["display_name"] = plot_data["feature"].map(DISPLAY_NAMES)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.barh(
        plot_data["display_name"], plot_data["permutation_rmse_increase_mean"],
        xerr=plot_data["permutation_rmse_increase_std"], color="#2563eb", alpha=0.85,
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.set(
        xlabel="Increase in held-out RMSE after permutation (percentage points)",
        ylabel="Feature",
        title="Group-CV permutation importance for XGBoost regression",
    )
    fig.savefig(FIGURES_DIR / "q1_xgboost_feature_importance.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(INPUT_PATH)
    data["y_pct"] = 100 * data["Y染色体浓度"]
    data = data.dropna(subset=FEATURES + ["y_pct"]).copy()
    x = data[FEATURES]
    y = data["y_pct"].to_numpy()
    groups = data["孕妇代码"].to_numpy()
    splitter = GroupKFold(n_splits=N_SPLITS)
    oof_prediction = np.full(len(data), np.nan)
    fold_records: list[dict[str, object]] = []
    fold_importance: list[pd.DataFrame] = []

    for fold, (train_index, test_index) in enumerate(splitter.split(x, y, groups), start=1):
        pipeline = make_pipeline()
        pipeline.fit(x.iloc[train_index], y[train_index])
        prediction = pipeline.predict(x.iloc[test_index])
        oof_prediction[test_index] = prediction
        fold_records.append(
            {
                "fold": fold,
                "train_rows": int(len(train_index)),
                "test_rows": int(len(test_index)),
                "train_mothers": int(pd.Series(groups[train_index]).nunique()),
                "test_mothers": int(pd.Series(groups[test_index]).nunique()),
                "r_squared": float(r2_score(y[test_index], prediction)),
                "rmse": float(np.sqrt(mean_squared_error(y[test_index], prediction))),
                "mae": float(mean_absolute_error(y[test_index], prediction)),
            }
        )
        permutation = permutation_importance(
            pipeline,
            x.iloc[test_index],
            y[test_index],
            scoring="neg_root_mean_squared_error",
            n_repeats=30,
            random_state=RANDOM_SEED + fold,
            n_jobs=1,
        )
        fold_importance.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "feature": FEATURES,
                    "permutation_rmse_increase": permutation.importances_mean,
                    "permutation_rmse_increase_std": permutation.importances_std,
                }
            )
        )

    cv_metrics = pd.DataFrame(fold_records)
    cv_metrics.to_csv(RESULTS_DIR / "q1_xgboost_cv_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "row_index": data.index,
            "mother_id": data["孕妇代码"],
            "actual_y_pct": y,
            "oof_predicted_y_pct": oof_prediction,
            "residual": y - oof_prediction,
        }
    ).to_csv(RESULTS_DIR / "q1_xgboost_oof_predictions.csv", index=False, encoding="utf-8-sig")

    permutation_all = pd.concat(fold_importance, ignore_index=True)
    importance = (
        permutation_all.groupby("feature", as_index=False)
        .agg(
            permutation_rmse_increase_mean=("permutation_rmse_increase", "mean"),
            permutation_rmse_increase_std=("permutation_rmse_increase", "std"),
        )
    )
    full_model = make_pipeline()
    full_model.fit(x, y)
    gain = aggregate_gain(full_model)
    importance["full_data_tree_gain_share"] = importance["feature"].map(gain)
    importance = importance.sort_values("permutation_rmse_increase_mean", ascending=False)
    importance.to_csv(RESULTS_DIR / "q1_xgboost_feature_importance.csv", index=False, encoding="utf-8-sig")
    make_importance_figure(importance)

    summary = {
        "sample": {
            "rows": int(len(data)),
            "mothers": int(data["孕妇代码"].nunique()),
            "analysis_window": "11--29 weeks; all rows with complete selected features",
        },
        "features": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
        "model_parameters": MODEL_PARAMETERS,
        "validation": {
            "method": "5-fold GroupKFold by pregnant-woman code",
            "mean_r_squared": float(cv_metrics["r_squared"].mean()),
            "mean_rmse": float(cv_metrics["rmse"].mean()),
            "mean_mae": float(cv_metrics["mae"].mean()),
            "pooled_oof_r_squared": float(r2_score(y, oof_prediction)),
            "pooled_oof_rmse": float(np.sqrt(mean_squared_error(y, oof_prediction))),
            "pooled_oof_mae": float(mean_absolute_error(y, oof_prediction)),
        },
        "importance_interpretation": "Permutation importance is the held-out RMSE increase after shuffling one original feature. Gain share is calculated after refitting on all records and is secondary because correlated predictors can split gain.",
    }
    (RESULTS_DIR / "q1_xgboost_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("XGBoost group-CV outputs written to results/ and figures/.")


if __name__ == "__main__":
    main()
