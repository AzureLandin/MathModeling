"""问题 2：基于 BMI 分组的 NIPT 时点选择 Baseline。

输入：results/male_prepared.csv。
输出：results/q2_baseline_summary.json、results/q2_time_sensitivity.csv、
      results/q2_bootstrap_times.csv、results/q2_group_summary.csv，及
      figures/q2_pass_probability_curves.png。
模型：GEE Logistic，logit P(Y>=4%) = beta0 + beta1*(week-16)+beta2*(BMI-32)，
      以孕妇代码聚类并使用 exchangeable 工作相关结构。
决策：对每个临床参考 BMI 区间，在该组最高观测 BMI 处取使达标概率不低于 q 的
      最早连续孕周，并向上取整到完整孕周；q=0.90 为演示性主阈值，另报告
      q=0.80、0.85、0.95 的敏感性结果。题目未给定 q，故 0.90 不是题设事实。
验证：按孕妇整体重采样的 Bootstrap（固定随机种子）量化时点估计的不确定性。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit, logit
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "results" / "male_prepared.csv"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
FORMULA = "y_pass_4pct ~ week_c + bmi_c"
WEEK_REFERENCE = 16.0
BMI_REFERENCE = 32.0
WINDOW = (11.0, 25.0)
PRIMARY_Q = 0.90
SENSITIVITY_Q = (0.80, 0.85, 0.90, 0.95)
N_BOOTSTRAP = 300
RANDOM_SEED = 20250908
GROUPS = (
    ("20-<28", 20.0, 28.0),
    ("28-<32", 28.0, 32.0),
    ("32-<36", 32.0, 36.0),
    ("36-<40", 36.0, 40.0),
    ("40+", 40.0, np.inf),
)


def add_predictors(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["week_c"] = data["gestational_week"] - WEEK_REFERENCE
    data["bmi_c"] = data["孕妇BMI"] - BMI_REFERENCE
    return data


def fit_gee(data: pd.DataFrame, group_column: str = "孕妇代码") -> object:
    return sm.GEE.from_formula(
        FORMULA,
        groups=group_column,
        data=data,
        family=Binomial(),
        cov_struct=Exchangeable(),
    ).fit()


def group_table(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, lower, upper in GROUPS:
        mask = data["孕妇BMI"].ge(lower) & (data["孕妇BMI"].lt(upper) if np.isfinite(upper) else True)
        subset = data.loc[mask]
        rows.append(
            {
                "bmi_group": name,
                "bmi_lower": lower,
                "bmi_upper_definition": None if not np.isfinite(upper) else upper,
                "bmi_upper_observed": float(subset["孕妇BMI"].max()),
                "rows": int(len(subset)),
                "mothers": int(subset["孕妇代码"].nunique()),
                "observed_pass_rate": float(subset["y_pass_4pct"].mean()),
            }
        )
    return pd.DataFrame(rows)


def threshold_time(params: pd.Series | np.ndarray, bmi: float, q: float) -> float:
    """返回模型达标概率为 q 的连续孕周；时间斜率非正时返回 NaN。"""
    beta0, beta_week, beta_bmi = params[0], params[1], params[2]
    if beta_week <= 0:
        return float("nan")
    return float(WEEK_REFERENCE + (logit(q) - beta0 - beta_bmi * (bmi - BMI_REFERENCE)) / beta_week)


def recommendation_row(params: pd.Series | np.ndarray, group: pd.Series, q: float) -> dict[str, object]:
    continuous_week = threshold_time(params, float(group["bmi_upper_observed"]), q)
    if not np.isfinite(continuous_week):
        status = "time_slope_nonpositive"
        recommended_week = np.nan
    elif continuous_week > WINDOW[1]:
        status = "beyond_25_week_window"
        recommended_week = np.nan
    else:
        status = "within_window"
        recommended_week = float(max(WINDOW[0], np.ceil(continuous_week)))
    return {
        "bmi_group": group["bmi_group"],
        "q": q,
        "bmi_used_for_conservative_group_decision": float(group["bmi_upper_observed"]),
        "continuous_threshold_week": continuous_week,
        "recommended_complete_week": recommended_week,
        "status": status,
    }


def bootstrap_times(data: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    mother_ids = data["孕妇代码"].unique()
    pieces = {mother: frame.copy() for mother, frame in data.groupby("孕妇代码", sort=False)}
    records: list[dict[str, object]] = []
    for replicate in range(N_BOOTSTRAP):
        selected = rng.choice(mother_ids, size=len(mother_ids), replace=True)
        sampled_frames = []
        for draw_id, mother in enumerate(selected):
            frame = pieces[mother].copy()
            frame["bootstrap_cluster"] = draw_id
            sampled_frames.append(frame)
        sample = pd.concat(sampled_frames, ignore_index=True)
        try:
            fitted = fit_gee(sample, "bootstrap_cluster")
            params = fitted.params.to_numpy()
            for _, group in groups.iterrows():
                row = recommendation_row(params, group, PRIMARY_Q)
                row["bootstrap_replicate"] = replicate
                records.append(row)
        except (np.linalg.LinAlgError, ValueError):
            continue
    return pd.DataFrame(records)


def make_figure(fitted: object, groups: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    weeks = np.linspace(*WINDOW, 141)
    fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.88, len(groups)))
    for (_, group), color in zip(groups.iterrows(), colors, strict=True):
        bmi = float(group["bmi_upper_observed"])
        linear_predictor = (
            fitted.params["Intercept"]
            + fitted.params["week_c"] * (weeks - WEEK_REFERENCE)
            + fitted.params["bmi_c"] * (bmi - BMI_REFERENCE)
        )
        ax.plot(weeks, expit(linear_predictor), color=color, lw=2, label=group["bmi_group"])
    ax.axhline(PRIMARY_Q, color="crimson", ls="--", lw=1.2, label=f"q = {PRIMARY_Q:.2f}")
    ax.axvspan(11, 12, color="#b6e3b6", alpha=0.3, label="early discovery window")
    ax.set(
        xlim=WINDOW,
        ylim=(0, 1.03),
        xlabel="Gestational week",
        ylabel="Estimated P(Y concentration at least 4%)",
        title="Population-average pass probability at conservative BMI by group",
    )
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(FIGURES_DIR / "q2_pass_probability_curves.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(INPUT_PATH)
    data = add_predictors(data.loc[data["gestational_week"].between(*WINDOW)].copy())
    groups = group_table(data)
    groups.to_csv(RESULTS_DIR / "q2_group_summary.csv", index=False, encoding="utf-8-sig")
    fitted = fit_gee(data)

    recommendations = [
        recommendation_row(fitted.params.to_numpy(), group, q)
        for q in SENSITIVITY_Q
        for _, group in groups.iterrows()
    ]
    sensitivity = pd.DataFrame(recommendations)
    sensitivity.to_csv(RESULTS_DIR / "q2_time_sensitivity.csv", index=False, encoding="utf-8-sig")

    bootstrap = bootstrap_times(data, groups)
    bootstrap.to_csv(RESULTS_DIR / "q2_bootstrap_times.csv", index=False, encoding="utf-8-sig")
    bootstrap_summary = (
        bootstrap.groupby("bmi_group", dropna=False)["continuous_threshold_week"]
        .agg(
            bootstrap_n="count",
            ci_lower=lambda x: x.quantile(0.025),
            ci_upper=lambda x: x.quantile(0.975),
            within_window_rate=lambda x: float((x <= WINDOW[1]).mean()),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    make_figure(fitted, groups)

    summary = {
        "analysis_window_weeks": list(WINDOW),
        "model": {
            "formula": FORMULA,
            "method": "GEE binomial-logit with exchangeable working correlation and pregnant-woman clusters",
            "parameters": {name: float(value) for name, value in fitted.params.items()},
            "robust_standard_errors": {name: float(value) for name, value in fitted.bse.items()},
            "pvalues": {name: float(value) for name, value in fitted.pvalues.items()},
        },
        "decision_rule": {
            "primary_q": PRIMARY_Q,
            "sensitivity_q": list(SENSITIVITY_Q),
            "rule": "At each group's maximum observed BMI, choose the earliest time with model pass probability at least q, rounded upward to a full week.",
            "caveat": "q is an analyst-defined reliability target because the problem provides no numeric loss function or required pass probability.",
        },
        "bootstrap": {
            "replicates_requested": N_BOOTSTRAP,
            "replicates_retained": int(bootstrap["bootstrap_replicate"].nunique()),
            "seed": RANDOM_SEED,
            "summaries_at_primary_q": bootstrap_summary,
        },
    }
    (RESULTS_DIR / "q2_baseline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Question 2 baseline outputs written to results/ and figures/.")


if __name__ == "__main__":
    main()
