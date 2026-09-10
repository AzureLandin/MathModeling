"""问题 1 的第一层 Baseline：忽略重复测量结构的合并线性回归。

输入：results/male_prepared.csv。
输出：results/q1_pooled_ols_coefficients.csv、results/q1_pooled_ols_summary.json，
      figures/q1_pooled_ols_diagnostics.png。
模型：对男胎全部检测记录（11--29 周）拟合
      Y concentration (percentage points) = beta0 + beta1*gestational week + beta2*BMI + error。
规则：每条检测记录均进入回归，不按孕妇代码去重、聚类或设随机效应；
      不加入年龄、交互项、非线性项或测序质量变量。
用途：这是按用户指定的合并 OLS 思路进行的可解释 Baseline，不适用于最终的重复测量推断。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "results" / "male_prepared.csv"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
FORMULA = "y_pct ~ gestational_week + Q('孕妇BMI')"


def make_diagnostic_figure(model: object, data: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    fitted = model.fittedvalues
    residuals = model.resid
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    axes[0].scatter(fitted, residuals, s=12, alpha=0.35)
    axes[0].axhline(0, color="crimson", ls="--", lw=1)
    axes[0].set(xlabel="Fitted Y concentration (percentage points)", ylabel="Residual (percentage points)", title="Residuals versus fitted values")
    sm.qqplot(residuals, line="45", ax=axes[1], markerfacecolor="#3b82f6", markeredgecolor="none", alpha=0.35)
    axes[1].set_title("Normal Q-Q plot of residuals")
    fig.savefig(FIGURES_DIR / "q1_pooled_ols_diagnostics.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(INPUT_PATH)
    data["y_pct"] = 100 * data["Y染色体浓度"]
    analysis = data.dropna(subset=["y_pct", "gestational_week", "孕妇BMI"]).copy()
    model = smf.ols(FORMULA, data=analysis).fit()

    design = model.model.exog
    vif = {
        model.model.exog_names[index]: float(variance_inflation_factor(design, index))
        for index in range(1, design.shape[1])
    }
    bp_lm, bp_lm_pvalue, bp_f, bp_f_pvalue = het_breuschpagan(model.resid, design)
    coefficients = pd.DataFrame(
        {
            "term": model.params.index,
            "estimate": model.params.values,
            "standard_error": model.bse.values,
            "t_statistic": model.tvalues.values,
            "pvalue": model.pvalues.values,
            "ci_lower_95": model.conf_int().iloc[:, 0].values,
            "ci_upper_95": model.conf_int().iloc[:, 1].values,
        }
    )
    coefficients.to_csv(RESULTS_DIR / "q1_pooled_ols_coefficients.csv", index=False, encoding="utf-8-sig")
    make_diagnostic_figure(model, analysis)

    summary = {
        "model": {
            "formula": FORMULA,
            "outcome_unit": "Y chromosome concentration in percentage points",
            "rows": int(len(analysis)),
            "mothers_ignored_in_estimation": int(analysis["孕妇代码"].nunique()),
            "assumption": "All records are treated as independent, as a first-pass pooled OLS baseline.",
        },
        "fit": {
            "r_squared": float(model.rsquared),
            "adjusted_r_squared": float(model.rsquared_adj),
            "rmse": float(np.sqrt(model.mse_resid)),
            "f_statistic": float(model.fvalue),
            "f_pvalue": float(model.f_pvalue),
        },
        "coefficients": {
            term: {
                "estimate": float(model.params[term]),
                "standard_error": float(model.bse[term]),
                "pvalue": float(model.pvalues[term]),
            }
            for term in model.params.index
        },
        "diagnostics": {
            "vif": vif,
            "durbin_watson": float(sm.stats.stattools.durbin_watson(model.resid)),
            "jarque_bera_statistic": float(sm.stats.stattools.jarque_bera(model.resid)[0]),
            "jarque_bera_pvalue": float(sm.stats.stattools.jarque_bera(model.resid)[1]),
            "breusch_pagan_lm_statistic": float(bp_lm),
            "breusch_pagan_lm_pvalue": float(bp_lm_pvalue),
            "breusch_pagan_f_statistic": float(bp_f),
            "breusch_pagan_f_pvalue": float(bp_f_pvalue),
        },
        "scope_limit": "Nominal OLS standard errors and p-values can be anti-conservative because repeated records from the same pregnant woman are not independent; this is intentionally not corrected in this baseline.",
    }
    (RESULTS_DIR / "q1_pooled_ols_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Pooled OLS baseline outputs written to results/ and figures/.")


if __name__ == "__main__":
    main()
