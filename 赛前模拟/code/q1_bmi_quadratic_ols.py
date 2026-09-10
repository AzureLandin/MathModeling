"""问题 1：仅保留 BMI 二次项的降阶合并 OLS 验证。

输入：results/male_prepared.csv。
输出：results/q1_bmi_quadratic_ols_coefficients.csv、
      results/q1_bmi_quadratic_ols_summary.json，
      figures/q1_bmi_quadratic_profile.png。
样本与假设：男胎全部检测记录均纳入，忽略同一孕妇的重复观测结构。
模型：Y_pct = beta0 + beta1*w + beta2*b + beta3*b² + error，
      w 和 b 分别是以全样本均值中心化的孕周、BMI。
验证：与线性模型和完整二次响应面比较调整 R²、AIC、BIC、RMSE；以嵌套 F 检验
      检验完整模型中删除孕周二次项和交互项是否造成显著信息损失。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "results" / "male_prepared.csv"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
LINEAR_FORMULA = "y_pct ~ week_c + bmi_c"
REDUCED_FORMULA = "y_pct ~ week_c + bmi_c + I(bmi_c ** 2)"
FULL_FORMULA = "y_pct ~ week_c + bmi_c + I(week_c ** 2) + I(bmi_c ** 2) + week_c:bmi_c"


def metrics(model: object) -> dict[str, float]:
    return {
        "r_squared": float(model.rsquared),
        "adjusted_r_squared": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "rmse": float(np.sqrt(model.mse_resid)),
    }


def make_profile(data: pd.DataFrame, model: object, week_mean: float) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    bmi_grid = np.linspace(data["孕妇BMI"].min(), data["孕妇BMI"].max(), 200)
    prediction_data = pd.DataFrame(
        {
            "week_c": np.zeros_like(bmi_grid),
            "bmi_c": bmi_grid - data["孕妇BMI"].mean(),
        }
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.7), constrained_layout=True)
    ax.scatter(data["孕妇BMI"], data["y_pct"], alpha=0.14, s=12, color="#64748b", label="observations")
    ax.plot(bmi_grid, model.predict(prediction_data), color="#dc2626", lw=2.5, label=f"predicted at {week_mean:.2f} weeks")
    ax.set(xlabel="BMI (kg/m²)", ylabel="Y concentration (percentage points)", title="Reduced quadratic BMI profile")
    ax.legend()
    fig.savefig(FIGURES_DIR / "q1_bmi_quadratic_profile.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(INPUT_PATH)
    data["y_pct"] = 100 * data["Y染色体浓度"]
    data = data.dropna(subset=["y_pct", "gestational_week", "孕妇BMI"]).copy()
    week_mean = float(data["gestational_week"].mean())
    bmi_mean = float(data["孕妇BMI"].mean())
    data["week_c"] = data["gestational_week"] - week_mean
    data["bmi_c"] = data["孕妇BMI"] - bmi_mean

    linear = smf.ols(LINEAR_FORMULA, data=data).fit()
    reduced = smf.ols(REDUCED_FORMULA, data=data).fit()
    full = smf.ols(FULL_FORMULA, data=data).fit()
    linear_to_reduced = anova_lm(linear, reduced).iloc[1]
    reduced_to_full = anova_lm(reduced, full).iloc[1]
    design = reduced.model.exog
    vif = {
        reduced.model.exog_names[index]: float(variance_inflation_factor(design, index))
        for index in range(1, design.shape[1])
    }
    bp_lm, bp_lm_pvalue, bp_f, bp_f_pvalue = het_breuschpagan(reduced.resid, design)

    pd.DataFrame(
        {
            "term": reduced.params.index,
            "estimate": reduced.params.values,
            "standard_error": reduced.bse.values,
            "t_statistic": reduced.tvalues.values,
            "pvalue": reduced.pvalues.values,
            "ci_lower_95": reduced.conf_int().iloc[:, 0].values,
            "ci_upper_95": reduced.conf_int().iloc[:, 1].values,
        }
    ).to_csv(RESULTS_DIR / "q1_bmi_quadratic_ols_coefficients.csv", index=False, encoding="utf-8-sig")
    make_profile(data, reduced, week_mean)

    summary = {
        "sample": {
            "rows": int(len(data)),
            "mothers_ignored_in_estimation": int(data["孕妇代码"].nunique()),
            "week_mean_for_centering": week_mean,
            "bmi_mean_for_centering": bmi_mean,
        },
        "formulas": {"linear": LINEAR_FORMULA, "reduced": REDUCED_FORMULA, "full_quadratic": FULL_FORMULA},
        "metrics": {"linear": metrics(linear), "reduced": metrics(reduced), "full_quadratic": metrics(full)},
        "nested_tests": {
            "linear_to_reduced": {
                "f_statistic": float(linear_to_reduced["F"]),
                "pvalue": float(linear_to_reduced["Pr(>F)"]),
                "additional_degrees_of_freedom": int(linear_to_reduced["df_diff"]),
            },
            "reduced_to_full": {
                "f_statistic": float(reduced_to_full["F"]),
                "pvalue": float(reduced_to_full["Pr(>F)"]),
                "additional_degrees_of_freedom": int(reduced_to_full["df_diff"]),
            },
        },
        "coefficients": {
            term: {"estimate": float(reduced.params[term]), "pvalue": float(reduced.pvalues[term])}
            for term in reduced.params.index
        },
        "diagnostics": {
            "vif": vif,
            "breusch_pagan_lm_statistic": float(bp_lm),
            "breusch_pagan_lm_pvalue": float(bp_lm_pvalue),
            "breusch_pagan_f_statistic": float(bp_f),
            "breusch_pagan_f_pvalue": float(bp_f_pvalue),
        },
        "scope_limit": "As requested for this model-development stage, all records are treated as independent; repeat-measurement dependence is not corrected.",
    }
    (RESULTS_DIR / "q1_bmi_quadratic_ols_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Reduced BMI-quadratic OLS outputs written to results/ and figures/.")


if __name__ == "__main__":
    main()
