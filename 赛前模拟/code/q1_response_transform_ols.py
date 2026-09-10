"""问题 1：比较原始、对数和 logit 响应尺度下的线性回归。

输入：results/male_prepared.csv。
输出：results/q1_response_transform_coefficients.csv、
      results/q1_response_transform_cv_metrics.csv、
      results/q1_response_transform_summary.json，
      figures/q1_response_transform_diagnostics.png。
共同自变量：孕周与 BMI；样本为全部 1082 条男胎记录。
模型：原始 Y_pct、log(Y) 和 logit(Y) 各自对孕周、BMI 拟合 OLS。
验证：按孕妇代码 GroupKFold 五折。所有预测反变换至原始百分点后再比较 R²、
      RMSE、MAE；log 模型使用训练折残差的 smearing 因子校正反变换偏差。
限制：变换可改善误差尺度，不会凭空补足未观测个体差异；本脚本不加入随机效应。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import expit, logit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from statsmodels.stats.diagnostic import het_breuschpagan


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "results" / "male_prepared.csv"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
N_SPLITS = 5
FORMULA_TEMPLATE = "{response} ~ gestational_week + Q('孕妇BMI')"
MODEL_SPECS = {
    "raw": "y_pct",
    "log": "log_y",
    "logit": "logit_y",
}


def raw_scale_prediction(name: str, model: object, test: pd.DataFrame) -> np.ndarray:
    predicted = model.predict(test).to_numpy()
    if name == "raw":
        return predicted
    if name == "log":
        smearing = float(np.mean(np.exp(model.resid)))
        return 100 * np.exp(predicted) * smearing
    return 100 * expit(predicted)


def diagnostics(model: object) -> dict[str, float]:
    bp_lm, bp_lm_pvalue, bp_f, bp_f_pvalue = het_breuschpagan(model.resid, model.model.exog)
    jb_stat, jb_pvalue, skewness, kurtosis = sm.stats.stattools.jarque_bera(model.resid)
    return {
        "r_squared_on_transformed_scale": float(model.rsquared),
        "adjusted_r_squared_on_transformed_scale": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "jarque_bera_statistic": float(jb_stat),
        "jarque_bera_pvalue": float(jb_pvalue),
        "residual_skewness": float(skewness),
        "residual_kurtosis": float(kurtosis),
        "breusch_pagan_lm_statistic": float(bp_lm),
        "breusch_pagan_lm_pvalue": float(bp_lm_pvalue),
        "breusch_pagan_f_statistic": float(bp_f),
        "breusch_pagan_f_pvalue": float(bp_f_pvalue),
    }


def make_figure(models: dict[str, object]) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    titles = {"raw": "Raw Y response", "log": "Log(Y) response", "logit": "Logit(Y) response"}
    for axis, (name, model) in zip(axes, models.items(), strict=True):
        axis.scatter(model.fittedvalues, model.resid, s=10, alpha=0.3, color="#2563eb")
        axis.axhline(0, color="crimson", lw=1, ls="--")
        axis.set(title=titles[name], xlabel="Fitted value on model scale", ylabel="Residual")
    fig.savefig(FIGURES_DIR / "q1_response_transform_diagnostics.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(INPUT_PATH)
    data = data.dropna(subset=["Y染色体浓度", "gestational_week", "孕妇BMI"]).copy()
    y = data["Y染色体浓度"].to_numpy()
    if np.any(y <= 0) or np.any(y >= 1):
        raise ValueError("log and logit comparisons require Y concentration strictly between 0 and 1.")
    data["y_pct"] = 100 * y
    data["log_y"] = np.log(y)
    data["logit_y"] = logit(y)
    groups = data["孕妇代码"].to_numpy()
    splitter = GroupKFold(n_splits=N_SPLITS)
    oof = {name: np.full(len(data), np.nan) for name in MODEL_SPECS}
    fold_records: list[dict[str, object]] = []

    for fold, (train_index, test_index) in enumerate(splitter.split(data, y, groups), start=1):
        train, test = data.iloc[train_index], data.iloc[test_index]
        for name, response in MODEL_SPECS.items():
            model = smf.ols(FORMULA_TEMPLATE.format(response=response), data=train).fit()
            prediction = raw_scale_prediction(name, model, test)
            oof[name][test_index] = prediction
            fold_records.append(
                {
                    "model": name,
                    "fold": fold,
                    "test_rows": int(len(test_index)),
                    "test_mothers": int(pd.Series(groups[test_index]).nunique()),
                    "r_squared_raw_scale": float(r2_score(data["y_pct"].iloc[test_index], prediction)),
                    "rmse_raw_scale": float(np.sqrt(mean_squared_error(data["y_pct"].iloc[test_index], prediction))),
                    "mae_raw_scale": float(mean_absolute_error(data["y_pct"].iloc[test_index], prediction)),
                }
            )

    full_models = {
        name: smf.ols(FORMULA_TEMPLATE.format(response=response), data=data).fit()
        for name, response in MODEL_SPECS.items()
    }
    coefficients = []
    for name, model in full_models.items():
        for term in model.params.index:
            coefficients.append(
                {
                    "model": name,
                    "term": term,
                    "estimate_on_model_scale": float(model.params[term]),
                    "standard_error": float(model.bse[term]),
                    "pvalue": float(model.pvalues[term]),
                }
            )
    pd.DataFrame(coefficients).to_csv(
        RESULTS_DIR / "q1_response_transform_coefficients.csv", index=False, encoding="utf-8-sig"
    )
    cv_metrics = pd.DataFrame(fold_records)
    cv_metrics.to_csv(RESULTS_DIR / "q1_response_transform_cv_metrics.csv", index=False, encoding="utf-8-sig")
    make_figure(full_models)

    summary = {
        "sample": {"rows": int(len(data)), "mothers": int(data["孕妇代码"].nunique()), "window": "11--29 weeks"},
        "formula_template": FORMULA_TEMPLATE,
        "transformation_notes": {
            "raw": "Y concentration expressed in percentage points",
            "log": "natural logarithm of proportion; predictions are back-transformed with a Duan smearing factor",
            "logit": "logit of proportion; predictions are back-transformed with inverse logit",
        },
        "full_sample_diagnostics": {name: diagnostics(model) for name, model in full_models.items()},
        "group_cv_raw_scale": {
            name: {
                "mean_r_squared": float(frame["r_squared_raw_scale"].mean()),
                "mean_rmse": float(frame["rmse_raw_scale"].mean()),
                "mean_mae": float(frame["mae_raw_scale"].mean()),
                "pooled_oof_r_squared": float(r2_score(data["y_pct"], oof[name])),
                "pooled_oof_rmse": float(np.sqrt(mean_squared_error(data["y_pct"], oof[name]))),
                "pooled_oof_mae": float(mean_absolute_error(data["y_pct"], oof[name])),
            }
            for name, frame in cv_metrics.groupby("model")
        },
        "scope_limit": "Cross-validation is grouped by pregnant woman, but the fitted regressions themselves retain the requested two predictors and do not include random effects.",
    }
    (RESULTS_DIR / "q1_response_transform_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Response-transform OLS comparison written to results/ and figures/.")


if __name__ == "__main__":
    main()
