"""Question 1: comprehensive Spearman feature screen for Y concentration.

Input: results/male_prepared.csv.
Outputs: results/q1_spearman_feature_screen.csv, results/q1_spearman_feature_summary.json,
and figures/q1_spearman_feature_screen.png.
Method: compute Spearman rho, two-sided nominal p values, and Benjamini-Hochberg FDR
for numerical and ordinal variables. Maternal/clinical, sequencing-quality, and
same-assay chromosome-derived variables are labelled separately.
Limit: repeated records are retained as requested, so p values are descriptive and not
cluster-robust. Same-assay chromosome variables are analytical correlates, not causes.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "results" / "male_prepared.csv"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
FEATURES = [
    ("maternal_clinical", "gestational_week", "Gestational week"),
    ("maternal_clinical", "孕妇BMI", "BMI"),
    ("maternal_clinical", "年龄", "Maternal age"),
    ("maternal_clinical", "身高", "Height"),
    ("maternal_clinical", "体重", "Weight"),
    ("maternal_clinical", "检测抽血次数", "Draw count"),
    ("maternal_clinical", "gravidity_ordinal", "Gravidity (ordinal)"),
    ("maternal_clinical", "parity_ordinal", "Parity (ordinal)"),
    ("sequencing_quality", "原始读段数", "Raw read count"),
    ("sequencing_quality", "在参考基因组上比对的比例", "Alignment proportion"),
    ("sequencing_quality", "重复读段的比例", "Duplicate-read proportion"),
    ("sequencing_quality", "唯一比对的读段数", "Unique mapped reads"),
    ("sequencing_quality", "GC含量", "Overall GC content"),
    ("sequencing_quality", "13号染色体的GC含量", "Chr13 GC content"),
    ("sequencing_quality", "18号染色体的GC含量", "Chr18 GC content"),
    ("sequencing_quality", "21号染色体的GC含量", "Chr21 GC content"),
    ("sequencing_quality", "被过滤掉读段数的比例", "Filtered-read proportion"),
    ("same_assay_analytical", "13号染色体的Z值", "Chr13 Z-score"),
    ("same_assay_analytical", "18号染色体的Z值", "Chr18 Z-score"),
    ("same_assay_analytical", "21号染色体的Z值", "Chr21 Z-score"),
    ("same_assay_analytical", "X染色体的Z值", "X Z-score"),
    ("same_assay_analytical", "Y染色体的Z值", "Y Z-score"),
    ("same_assay_analytical", "X染色体浓度", "X concentration"),
]


def ordinal_rank(series: pd.Series) -> pd.Series:
    """Preserve ordered bins such as 0, 1, 2, >=3 without inventing an exact count."""
    text = series.astype(str).str.strip().str.replace("≥", ">=", regex=False)
    output = pd.to_numeric(text, errors="coerce")
    output = output.fillna(text.str.extract(r">=\s*(\d+)", expand=False).astype(float))
    return output


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(INPUT_PATH)
    data["gravidity_ordinal"] = ordinal_rank(data["怀孕次数"])
    data["parity_ordinal"] = ordinal_rank(data["生产次数"])
    outcome = data["Y染色体浓度"]
    records: list[dict[str, object]] = []
    for feature_group, column, display_name in FEATURES:
        pair = pd.DataFrame({"y": outcome, "x": data[column]}).dropna()
        result = spearmanr(pair["y"], pair["x"])
        records.append(
            {
                "feature_group": feature_group,
                "feature": column,
                "display_name": display_name,
                "rows": int(len(pair)),
                "spearman_rho": float(result.statistic),
                "nominal_pvalue": float(result.pvalue),
            }
        )
    screen = pd.DataFrame(records)
    screen["fdr_bh_pvalue"] = multipletests(screen["nominal_pvalue"], method="fdr_bh")[1]
    screen["fdr_bh_significant_0_05"] = screen["fdr_bh_pvalue"] < 0.05
    screen["absolute_rho"] = screen["spearman_rho"].abs()
    screen = screen.sort_values("absolute_rho", ascending=False)
    screen.to_csv(RESULTS_DIR / "q1_spearman_feature_screen.csv", index=False, encoding="utf-8-sig")

    FIGURES_DIR.mkdir(exist_ok=True)
    plot = screen.sort_values("spearman_rho").copy()
    colors = plot["feature_group"].map(
        {
            "maternal_clinical": "#2563eb",
            "sequencing_quality": "#059669",
            "same_assay_analytical": "#d97706",
        }
    )
    fig, ax = plt.subplots(figsize=(8.4, 7.2), constrained_layout=True)
    ax.barh(plot["display_name"], plot["spearman_rho"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set(xlabel="Spearman correlation with Y concentration", ylabel="Feature", title="Spearman feature screen (all male test records)")
    fig.savefig(FIGURES_DIR / "q1_spearman_feature_screen.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "outcome": "Y chromosome concentration",
        "rows": int(len(data)),
        "mothers": int(data["孕妇代码"].nunique()),
        "method": "Spearman rank correlation with two-sided nominal p values and Benjamini-Hochberg FDR adjustment",
        "excluded_nonordinal_fields": ["sample serial number", "mother code", "last menstrual period", "test date", "IVF conception type", "aneuploidy label", "fetal health"],
        "interpretation_limits": [
            "Repeated observations are retained, so p values do not account for within-mother dependence.",
            "Correlation is unadjusted and cannot establish an independent or causal effect.",
            "Same-assay chromosome Z scores and X concentration are analytical correlates and should not be interpreted as maternal determinants.",
            "Gravidity and parity bins are used only as ordinal ranks; >=3 is not treated as an exact count.",
        ],
    }
    (RESULTS_DIR / "q1_spearman_feature_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Comprehensive Spearman feature screen written to results/ and figures/.")


if __name__ == "__main__":
    main()
