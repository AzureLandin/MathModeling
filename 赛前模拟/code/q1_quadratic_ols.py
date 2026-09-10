"""问题 1：仅以孕周和 BMI 构造二次响应面，并与合并线性 OLS 比较。

输入：results/male_prepared.csv。
输出：results/q1_quadratic_ols_coefficients.csv、results/q1_quadratic_ols_summary.json，
      figures/q1_quadratic_response_surface.png。
样本：男胎全部 1082 条检测记录，按用户当前 Baseline 设定不区分孕妇重复观测。
模型：Y_pct = beta0 + beta1*w + beta2*b + beta3*w^2 + beta4*b^2 + beta5*w*b + error，
      其中 w=孕周-样本均值、b=BMI-样本均值。
验证：与仅含 w、b 的线性模型做嵌套 F 检验，并比较调整 R²、AIC、BIC、RMSE、
      VIF 和残差异方差检验。该比较仍不修正重复测量依赖性。
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
QUADRATIC_FORMULA = "y_pct ~ week_c + bmi_c + I(week_c ** 2) + I(bmi_c ** 2) + week_c:bmi_c"


def model_metrics(model: object) -> dict[str, float]:
    return {
        "r_squared": float(model.rsquared),
        "adjusted_r_squared": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "rmse": float(np.sqrt(model.mse_resid)),
    }


def make_surface_figure(data: pd.DataFrame, model: object) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    week_values = np.linspace(data["gestational_week"].min(), data["gestational_week"].max(), 80)
    bmi_values = np.linspace(data["孕妇BMI"].min(), data["孕妇BMI"].max(), 80)
    week_grid, bmi_grid = np.meshgrid(week_values, bmi_values)
    grid = pd.DataFrame(
        {
            "week_c": week_grid.ravel() - data["gestational_week"].mean(),
            "bmi_c": bmi_grid.ravel() - data["孕妇BMI"].mean(),
        }
    )
    prediction = model.predict(grid).to_numpy().reshape(week_grid.shape)
    fig, ax = plt.subplots(figsize=(8, 5.3), constrained_layout=True)
    contour = ax.contourf(week_grid, bmi_grid, prediction, levels=18, cmap="viridis")
    points = ax.scatter(
        data["gestational_week"], data["孕妇BMI"], c=data["y_pct"], s=10,
        cmap="viridis", alpha=0.28, edgecolors="none",
    )
    fig.colorbar(contour, ax=ax, label="Predicted Y concentration (percentage points)")
    ax.set(
        xlabel="Gestational week",
        ylabel="BMI (kg/m²)",
        title="Quadratic response surface: predicted Y concentration",
    )
    fig.savefig(FIGURES_DIR / "q1_quadratic_response_surface.png", dpi=240, bbox_inches="tight")
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
    quadratic = smf.ols(QUADRATIC_FORMULA, data=data).fit()
    comparison = anova_lm(linear, quadratic)
    design = quadratic.model.exog
    vif = {
        quadratic.model.exog_names[index]: float(variance_inflation_factor(design, index))
        for index in range(1, design.shape[1])
    }
    bp_lm, bp_lm_pvalue, bp_f, bp_f_pvalue = het_breuschpagan(quadratic.resid, design)

    coefficients = pd.DataFrame(
        {
            "term": quadratic.params.index,
            "estimate": quadratic.params.values,
            "standard_error": quadratic.bse.values,
            "t_statistic": quadratic.tvalues.values,
            "pvalue": quadratic.pvalues.values,
            "ci_lower_95": quadratic.conf_int().iloc[:, 0].values,
            "ci_upper_95": quadratic.conf_int().iloc[:, 1].values,
        }
    )
    coefficients.to_csv(RESULTS_DIR / "q1_quadratic_ols_coefficients.csv", index=False, encoding="utf-8-sig")
    make_surface_figure(data, quadratic)
    summary = {
        "sample": {
            "rows": int(len(data)),
            "mothers_ignored_in_estimation": int(data["孕妇代码"].nunique()),
            "gestational_week_mean_for_centering": week_mean,
            "bmi_mean_for_centering": bmi_mean,
        },
        "formulas": {"linear": LINEAR_FORMULA, "quadratic": QUADRATIC_FORMULA},
        "metrics": {"linear": model_metrics(linear), "quadratic": model_metrics(quadratic)},
        "nested_f_test": {
            "additional_sum_squares": float(comparison.iloc[1]["ss_diff"]),
            "f_statistic": float(comparison.iloc[1]["F"]),
            "pvalue": float(comparison.iloc[1]["Pr(>F)"]),
            "additional_degrees_of_freedom": int(comparison.iloc[1]["df_diff"]),
        },
        "quadratic_coefficients": {
            term: {
                "estimate": float(quadratic.params[term]),
                "pvalue": float(quadratic.pvalues[term]),
            }
            for term in quadratic.params.index
        },
        "quadratic_diagnostics": {
            "vif": vif,
            "breusch_pagan_lm_statistic": float(bp_lm),
            "breusch_pagan_lm_pvalue": float(bp_lm_pvalue),
            "breusch_pagan_f_statistic": float(bp_f),
            "breusch_pagan_f_pvalue": float(bp_f_pvalue),
        },
        "scope_limit": "All detection records are treated as independent by design; nominal p-values do not correct for repeated observations from the same pregnant woman.",
    }
    (RESULTS_DIR / "q1_quadratic_ols_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Quadratic response-surface outputs written to results/ and figures/.")


if __name__ == "__main__":
    main()
