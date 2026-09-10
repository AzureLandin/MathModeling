"""Question 1: Spearman tests and an additive cubic-spline regression.

Input: results/male_prepared.csv.
Outputs: Spearman correlations, spline coefficients, grouped CV metrics, a summary,
and partial-effect figures. All 1082 male records are included for this exploratory step.
The model is y_pct = alpha + f_week(week) + f_bmi(BMI) + error, with four
degrees of freedom for each natural cubic spline. It reports nominal OLS F and t tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import het_breuschpagan


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "results" / "male_prepared.csv"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
N_SPLITS = 5
SPLINE_DF = 4
LINEAR_FORMULA = "y_pct ~ gestational_week + Q('孕妇BMI')"
WEEK_SPLINE_FORMULA = "y_pct ~ cr(gestational_week, df=4) + Q('孕妇BMI')"
BMI_SPLINE_FORMULA = "y_pct ~ gestational_week + cr(Q('孕妇BMI'), df=4)"
GAM_FORMULA = "y_pct ~ cr(gestational_week, df=4) + cr(Q('孕妇BMI'), df=4)"
SPEARMAN_VARIABLES = {
    "gestational_week": "gestational_week",
    "BMI": "孕妇BMI",
    "age": "年龄",
    "height": "身高",
}


def model_metrics(model: object) -> dict[str, float]:
    return {
        "r_squared": float(model.rsquared),
        "adjusted_r_squared": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "rmse": float(np.sqrt(model.mse_resid)),
        "f_statistic": float(model.fvalue),
        "f_pvalue": float(model.f_pvalue),
    }


def nested_f(smaller: object, larger: object) -> dict[str, float | int]:
    comparison = anova_lm(smaller, larger).iloc[1]
    return {
        "f_statistic": float(comparison["F"]),
        "pvalue": float(comparison["Pr(>F)"]),
        "additional_degrees_of_freedom": int(comparison["df_diff"]),
        "additional_sum_squares": float(comparison["ss_diff"]),
    }


def make_partial_effect_figure(data: pd.DataFrame, model: object) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    week_grid = np.linspace(data["gestational_week"].min(), data["gestational_week"].max(), 200)
    bmi_grid = np.linspace(data["孕妇BMI"].min(), data["孕妇BMI"].max(), 200)
    bmi_ref = float(data["孕妇BMI"].median())
    week_ref = float(data["gestational_week"].median())
    week_prediction = model.predict(pd.DataFrame({"gestational_week": week_grid, "孕妇BMI": bmi_ref}))
    bmi_prediction = model.predict(pd.DataFrame({"gestational_week": week_ref, "孕妇BMI": bmi_grid}))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].scatter(data["gestational_week"], data["y_pct"], s=10, alpha=0.12, color="#64748b")
    axes[0].plot(week_grid, week_prediction, color="#dc2626", lw=2.5)
    axes[0].set(xlabel="Gestational week", ylabel="Predicted Y concentration (percentage points)", title=f"Week effect at BMI={bmi_ref:.2f}")
    axes[1].scatter(data["孕妇BMI"], data["y_pct"], s=10, alpha=0.12, color="#64748b")
    axes[1].plot(bmi_grid, bmi_prediction, color="#dc2626", lw=2.5)
    axes[1].set(xlabel="BMI (kg/m²)", ylabel="Predicted Y concentration (percentage points)", title=f"BMI effect at week={week_ref:.2f}")
    fig.savefig(FIGURES_DIR / "q1_spline_gam_partial_effects.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(INPUT_PATH)
    data = data.dropna(subset=["Y染色体浓度", "gestational_week", "孕妇BMI", "年龄", "身高"]).copy()
    data["y_pct"] = 100 * data["Y染色体浓度"]

    correlation_rows = []
    for label, column in SPEARMAN_VARIABLES.items():
        result = spearmanr(data["Y染色体浓度"], data[column], nan_policy="omit")
        correlation_rows.append(
            {"outcome": "Y_chromosome_concentration", "variable": label, "spearman_rho": float(result.statistic), "nominal_pvalue": float(result.pvalue), "rows": int(len(data))}
        )
    pd.DataFrame(correlation_rows).to_csv(RESULTS_DIR / "q1_spearman_correlations.csv", index=False, encoding="utf-8-sig")

    linear = smf.ols(LINEAR_FORMULA, data=data).fit()
    week_spline = smf.ols(WEEK_SPLINE_FORMULA, data=data).fit()
    bmi_spline = smf.ols(BMI_SPLINE_FORMULA, data=data).fit()
    gam = smf.ols(GAM_FORMULA, data=data).fit()
    bp_lm, bp_lm_pvalue, _, _ = het_breuschpagan(gam.resid, gam.model.exog)
    jb_stat, jb_pvalue, skewness, kurtosis = sm.stats.stattools.jarque_bera(gam.resid)

    pd.DataFrame(
        {
            "term": gam.params.index,
            "estimate": gam.params.values,
            "standard_error": gam.bse.values,
            "t_statistic": gam.tvalues.values,
            "pvalue": gam.pvalues.values,
            "ci_lower_95": gam.conf_int().iloc[:, 0].values,
            "ci_upper_95": gam.conf_int().iloc[:, 1].values,
        }
    ).to_csv(RESULTS_DIR / "q1_spline_gam_coefficients.csv", index=False, encoding="utf-8-sig")

    groups = data["孕妇代码"].to_numpy()
    splitter = GroupKFold(n_splits=N_SPLITS)
    oof_prediction = np.full(len(data), np.nan)
    fold_records = []
    for fold, (train_index, test_index) in enumerate(splitter.split(data, data["y_pct"], groups), start=1):
        train, test = data.iloc[train_index], data.iloc[test_index]
        fold_model = smf.ols(GAM_FORMULA, data=train).fit()
        prediction = fold_model.predict(test).to_numpy()
        oof_prediction[test_index] = prediction
        fold_records.append(
            {
                "fold": fold,
                "test_rows": int(len(test_index)),
                "test_mothers": int(pd.Series(groups[test_index]).nunique()),
                "r_squared": float(r2_score(test["y_pct"], prediction)),
                "rmse": float(np.sqrt(mean_squared_error(test["y_pct"], prediction))),
                "mae": float(mean_absolute_error(test["y_pct"], prediction)),
            }
        )
    cv_metrics = pd.DataFrame(fold_records)
    cv_metrics.to_csv(RESULTS_DIR / "q1_spline_gam_cv_metrics.csv", index=False, encoding="utf-8-sig")
    make_partial_effect_figure(data, gam)

    summary = {
        "sample": {"rows": int(len(data)), "mothers_ignored_in_nominal_tests": int(data["孕妇代码"].nunique()), "window": "11--29 weeks"},
        "spearman_note": "Nominal two-sided p-values treat all records as independent and are descriptive in this repeated-measurement dataset.",
        "model": {
            "name": "Gaussian additive natural-cubic-spline regression (GAM-style OLS)",
            "spline_df_per_predictor": SPLINE_DF,
            "formula": GAM_FORMULA,
            "metrics": model_metrics(gam),
            "residual_diagnostics": {
                "breusch_pagan_lm_statistic": float(bp_lm),
                "breusch_pagan_lm_pvalue": float(bp_lm_pvalue),
                "jarque_bera_statistic": float(jb_stat),
                "jarque_bera_pvalue": float(jb_pvalue),
                "residual_skewness": float(skewness),
                "residual_kurtosis": float(kurtosis),
            },
        },
        "model_comparison": {
            "linear": model_metrics(linear),
            "week_spline_only": model_metrics(week_spline),
            "bmi_spline_only": model_metrics(bmi_spline),
            "gam_additive_splines": model_metrics(gam),
            "linear_to_week_spline": nested_f(linear, week_spline),
            "linear_to_bmi_spline": nested_f(linear, bmi_spline),
            "linear_to_full_gam": nested_f(linear, gam),
        },
        "group_cv": {
            "method": "5-fold GroupKFold by pregnant-woman code",
            "mean_r_squared": float(cv_metrics["r_squared"].mean()),
            "mean_rmse": float(cv_metrics["rmse"].mean()),
            "mean_mae": float(cv_metrics["mae"].mean()),
            "pooled_oof_r_squared": float(r2_score(data["y_pct"], oof_prediction)),
            "pooled_oof_rmse": float(np.sqrt(mean_squared_error(data["y_pct"], oof_prediction))),
            "pooled_oof_mae": float(mean_absolute_error(data["y_pct"], oof_prediction)),
        },
        "inference_limit": "Individual spline-basis t-tests are algebraic basis tests, not separate clinical effects. Interpret smooth shapes and block F tests instead.",
    }
    (RESULTS_DIR / "q1_spline_gam_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Spearman and spline-GAM outputs written to results/ and figures/.")


if __name__ == "__main__":
    main()
