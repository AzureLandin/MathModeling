"""问题 1：男胎 Y 染色体浓度的可解释 Baseline 分析。

输入：results/male_prepared.csv。
输出：results/q1_baseline_coefficients.csv、results/q1_baseline_summary.json、
      results/q1_binned_rates.csv、results/q1_within_mother_coefficients.csv，
      以及 figures/q1_y_concentration_trends.png。
模型：在线性尺度（Y 浓度的百分点）拟合加性随机截距混合模型
      y_ij = beta0 + beta1 w_ij + beta2 b_ij + beta3 a_i + b_i + epsilon_ij，
      其中 w=孕周-16、b=BMI-32、a=年龄-29，b_i 为孕妇随机截距。
验证：以按孕妇聚类的稳健 OLS 复核固定效应显著性；以似然比检验检验孕周×BMI
      交互项的必要性；以孕妇固定效应模型检验同一孕妇内的时间效应；以 11--29 周
      全样本复核题目窗口（11--25 周）限制下的结论方向。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "results" / "male_prepared.csv"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
BASELINE_FORMULA = "y_pct ~ week_c + bmi_c + age_c"
INTERACTION_FORMULA = "y_pct ~ week_c * bmi_c + age_c"
WEEK_REFERENCE = 16.0
BMI_REFERENCE = 32.0
AGE_REFERENCE = 29.0
BMI_BINS = [20, 28, 32, 36, 40, np.inf]
BMI_LABELS = ["20-<28", "28-<32", "32-<36", "36-<40", "40+"]


def fit_models(data: pd.DataFrame, formula: str) -> tuple[object, object]:
    """拟合混合模型和按孕妇聚类的 OLS 稳健性复核模型。"""
    mixed = smf.mixedlm(formula, data=data, groups=data["孕妇代码"]).fit(
        reml=False, method="lbfgs", maxiter=500
    )
    clustered_ols = smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["孕妇代码"]}
    )
    return mixed, clustered_ols


def add_centered_predictors(data: pd.DataFrame) -> pd.DataFrame:
    """以典型孕妇的值中心化，使交互模型的主效应可解释。"""
    model_data = data.copy()
    model_data["week_c"] = model_data["gestational_week"] - WEEK_REFERENCE
    model_data["bmi_c"] = model_data["孕妇BMI"] - BMI_REFERENCE
    model_data["age_c"] = model_data["年龄"] - AGE_REFERENCE
    model_data["mother_id"] = model_data["孕妇代码"].astype("category")
    return model_data


def fit_within_mother_model(data: pd.DataFrame) -> object:
    """用孕妇固定效应估计同一孕妇内的孕周、BMI 变化效应。"""
    return smf.ols("y_pct ~ week_c + bmi_c + C(mother_id)", data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["孕妇代码"]}
    )


def coefficient_table(mixed: object, ols: object) -> pd.DataFrame:
    rows = []
    for term in mixed.fe_params.index:
        rows.append(
            {
                "term": term,
                "mixed_effect_estimate": float(mixed.fe_params[term]),
                "mixed_effect_se": float(mixed.bse_fe[term]),
                "mixed_effect_pvalue": float(mixed.pvalues[term]),
                "clustered_ols_estimate": float(ols.params[term]),
                "clustered_ols_se": float(ols.bse[term]),
                "clustered_ols_pvalue": float(ols.pvalues[term]),
            }
        )
    return pd.DataFrame(rows)


def make_binned_table(data: pd.DataFrame) -> pd.DataFrame:
    binned = data.copy()
    binned["bmi_group"] = pd.cut(
        binned["孕妇BMI"], bins=BMI_BINS, labels=BMI_LABELS, right=False
    )
    binned["week_integer"] = np.floor(binned["gestational_week"]).astype(int)
    return (
        binned.groupby(["bmi_group", "week_integer"], observed=False)
        .agg(
            n=("y_pct", "size"),
            y_pct_mean=("y_pct", "mean"),
            y_pct_median=("y_pct", "median"),
            pass_rate=("y_pass_4pct", "mean"),
        )
        .reset_index()
    )


def make_figure(data: pd.DataFrame, model: object) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    plot_data = data.copy()
    plot_data["bmi_group"] = pd.cut(
        plot_data["孕妇BMI"], bins=BMI_BINS, labels=BMI_LABELS, right=False
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.88, len(BMI_LABELS)))
    age_ref = float(data["年龄"].median())
    week_grid = np.linspace(11, 25, 141)

    for group, color in zip(BMI_LABELS, colors, strict=True):
        group_data = plot_data.loc[plot_data["bmi_group"].astype(str) == group]
        if group_data.empty:
            continue
        bmi_ref = float(group_data["孕妇BMI"].median())
        axes[0].scatter(
            group_data["gestational_week"], group_data["y_pct"],
            color=color, alpha=0.16, s=12,
        )
        predict_data = pd.DataFrame(
            {
                "gestational_week": week_grid,
                "孕妇BMI": bmi_ref,
                "年龄": age_ref,
                "week_c": week_grid - WEEK_REFERENCE,
                "bmi_c": bmi_ref - BMI_REFERENCE,
                "age_c": age_ref - AGE_REFERENCE,
            }
        )
        axes[0].plot(week_grid, model.predict(predict_data), color=color, lw=2, label=group)

        rates = (
            group_data.assign(week_integer=np.floor(group_data["gestational_week"]).astype(int))
            .groupby("week_integer")["y_pass_4pct"].mean()
        )
        axes[1].plot(rates.index, rates.values, marker="o", ms=3.5, color=color, label=group)

    axes[0].axhline(4, color="crimson", ls="--", lw=1.2, label="4% threshold")
    axes[0].set(xlabel="Gestational week", ylabel="Y concentration (percentage points)", xlim=(11, 25))
    axes[0].set_title("Observed Y concentration and mixed-model trend")
    axes[0].legend(title="BMI group", fontsize=8)
    axes[1].axhline(0.95, color="crimson", ls="--", lw=1.2, label="95% reference")
    axes[1].set(xlabel="Gestational week (floor)", ylabel="Observed pass rate", ylim=(0, 1.04), xlim=(11, 25))
    axes[1].set_title("Rate of Y concentration at least 4%")
    axes[1].legend(title="BMI group", fontsize=8)
    fig.savefig(FIGURES_DIR / "q1_y_concentration_trends.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(INPUT_PATH)
    data["y_pct"] = 100 * data["Y染色体浓度"]
    primary = add_centered_predictors(
        data.loc[data["gestational_week"].between(11, 25)].copy()
    )
    full = add_centered_predictors(
        data.loc[data["gestational_week"].between(11, 29)].copy()
    )
    mixed_primary, ols_primary = fit_models(primary, BASELINE_FORMULA)
    mixed_full, _ = fit_models(full, BASELINE_FORMULA)
    mixed_interaction, _ = fit_models(primary, INTERACTION_FORMULA)
    within_mother = fit_within_mother_model(primary)

    coefficients = coefficient_table(mixed_primary, ols_primary)
    coefficients.to_csv(RESULTS_DIR / "q1_baseline_coefficients.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "term": term,
                "estimate": float(within_mother.params[term]),
                "standard_error": float(within_mother.bse[term]),
                "pvalue": float(within_mother.pvalues[term]),
            }
            for term in ["week_c", "bmi_c"]
        ]
    ).to_csv(RESULTS_DIR / "q1_within_mother_coefficients.csv", index=False, encoding="utf-8-sig")
    binned = make_binned_table(primary)
    binned.to_csv(RESULTS_DIR / "q1_binned_rates.csv", index=False, encoding="utf-8-sig")
    make_figure(primary, mixed_primary)

    summary = {
        "primary_window": {"weeks": [11, 25], "rows": int(len(primary)), "mothers": int(primary["孕妇代码"].nunique())},
        "sensitivity_window": {"weeks": [11, 29], "rows": int(len(full)), "mothers": int(full["孕妇代码"].nunique())},
        "mixed_model": {
            "formula": BASELINE_FORMULA,
            "centering_references": {"gestational_week": WEEK_REFERENCE, "BMI": BMI_REFERENCE, "age": AGE_REFERENCE},
            "random_effect": "pregnant woman random intercept",
            "primary_aic": float(mixed_primary.aic),
            "primary_bic": float(mixed_primary.bic),
            "random_intercept_variance": float(mixed_primary.cov_re.iloc[0, 0]),
            "residual_variance": float(mixed_primary.scale),
            "converged": bool(mixed_primary.converged),
        },
        "interaction_test": {
            "candidate_formula": INTERACTION_FORMULA,
            "likelihood_ratio": float(2 * (mixed_interaction.llf - mixed_primary.llf)),
            "degrees_of_freedom": int(len(mixed_interaction.params) - len(mixed_primary.params)),
            "pvalue": float(
                chi2.sf(
                    2 * (mixed_interaction.llf - mixed_primary.llf),
                    len(mixed_interaction.params) - len(mixed_primary.params),
                )
            ),
            "baseline_aic": float(mixed_primary.aic),
            "interaction_aic": float(mixed_interaction.aic),
        },
        "sensitivity_fixed_effects": {
            term: {"primary": float(mixed_primary.fe_params[term]), "full": float(mixed_full.fe_params[term])}
            for term in mixed_primary.fe_params.index
        },
        "clustered_ols_r_squared": float(ols_primary.rsquared),
        "within_mother_model": {
            "formula": "y_pct ~ week_c + bmi_c + pregnant-woman fixed effects",
            "identification_note": "Age is constant within each pregnant woman and is therefore not identifiable in this model.",
            "coefficients": {
                term: {
                    "estimate": float(within_mother.params[term]),
                    "standard_error": float(within_mother.bse[term]),
                    "pvalue": float(within_mother.pvalues[term]),
                }
                for term in ["week_c", "bmi_c"]
            },
        },
    }
    (RESULTS_DIR / "q1_baseline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Question 1 baseline outputs written to results/ and figures/.")


if __name__ == "__main__":
    main()
