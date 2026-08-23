from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

import Q1_spatial_clustering as q1
from All_common_utilities import sha256
from All_data_processing import annualize_2025, load_inputs
from All_figure_generation import make_figures
from Q2_constrained_ridge import drivers_and_model
from Q3_scenario_forecast import forecast_scenarios
from Q4_policy_mapping import build_q4_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all carbon-emission modelling questions.")
    parser.add_argument("--project-root", required=True, help="Project root containing data, results and figures folders.")
    parser.add_argument("--input-root", required=True, help="Folder containing the two original attachments.")
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--minimal", action="store_true", help="Fit models without rewriting output files or figures.")
    args = parser.parse_args()

    q1.set_seed(args.seed)
    project = Path(args.project_root)
    input_root = Path(args.input_root)
    results = project / "results"
    figures = project / "figures"
    data_dir = project / "data"

    daily, province, max_err = load_inputs(input_root)
    annualized_2025, jan_sep_fraction, jan_sep_fraction_sd = annualize_2025(daily)
    annual = daily.groupby(["year", "Sector"], as_index=False)["CO2 (Mt)"].sum().rename(columns={"Sector": "sector", "CO2 (Mt)": "total_mt"})
    annual = annual[annual.year <= 2024].copy()
    annual = pd.concat([annual, pd.DataFrame([{"year": 2025, "sector": "Total", "total_mt": annualized_2025}])], ignore_index=True)

    spatial, kruskal, _ = q1.spatial_tests(province)
    province, pca, _, _, class_names, _, stability, cluster_decision = q1.classify(province)
    driver_data, _, model, scaler, coefficients, backtest, model_metrics = drivers_and_model(annual, project / "data" / "全国年度驱动变量_来源数据.csv")
    forecast, calibration = forecast_scenarios(model, scaler, annualized_2025, annual, model_metrics["residual_sd_Mt"], results)
    q4_province, q4_class, q4_phase, q4_sector, q4_thresholds = build_q4_outputs(province, annual, forecast)

    if args.minimal:
        print(json.dumps({"rows_input": len(daily), "provinces": len(province), "annualized_2025_mt": annualized_2025, "ridge_lambda": model_metrics["ridge_lambda"], "expanding_backtest_mae_mt": model_metrics["expanding_MAE_Mt"], "forecast_rows": len(forecast)}, ensure_ascii=False))
        return

    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    for grayscale in figures.glob("*_grayscale.png"):
        grayscale.unlink()

    q4_province.to_csv(results / "问题4_省级政策映射.csv", index=False, encoding="utf-8-sig")
    q4_class.to_csv(results / "问题4_类别政策覆盖率.csv", index=False, encoding="utf-8-sig")
    q4_phase.to_csv(results / "问题4_阶段情景指标.csv", index=False, encoding="utf-8-sig")
    q4_sector.to_csv(results / "问题4_部门优先级.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(results / "问题1_聚类决策.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(data_dir / "附件1_清洗后.csv", index=False, encoding="utf-8-sig")
    province.to_csv(results / "问题1_省级指标与分类.csv", index=False, encoding="utf-8-sig")
    spatial.to_csv(results / "问题1_Moran检验.csv", index=False, encoding="utf-8-sig")
    kruskal.to_csv(results / "问题1_分组差异检验.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"component": ["PC1", "PC2"], "explained_variance_ratio": pca.explained_variance_ratio_}).to_csv(results / "问题1_PCA.csv", index=False)
    stability.to_csv(results / "问题1_聚类稳定性.csv", index=False)
    coefficients.to_csv(results / "问题2_驱动因素系数.csv", index=False, encoding="utf-8-sig")
    backtest.to_csv(results / "问题2_回测.csv", index=False, encoding="utf-8-sig")
    driver_data.to_csv(data_dir / "external_driver_data.csv", index=False, encoding="utf-8-sig")
    forecast.to_csv(results / "问题3_2026_2045_三情景预测.csv", index=False, encoding="utf-8-sig")
    forecast.groupby("scenario").agg(path_valid=("path_valid", "all"), min_coal=("coal_share", "min"), max_clean=("clean_share", "max"), max_energy_sum=("coal_share", lambda x: float((x + forecast.loc[x.index, "clean_share"]).max()))).reset_index().to_csv(results / "问题3_情景约束检查.csv", index=False, encoding="utf-8-sig")

    metrics = {
        "input_rows": len(daily), "unique_dates": int(daily.Date.nunique()),
        "units": {"emissions": "Mt CO2", "gdp": "trillion yuan", "forecast_intensity": "Mt CO2 per trillion yuan GDP", "q1_intensity": "t CO2 per 10,000 yuan GDP", "energy_shares": "fraction in [0,1]"},
        "total_consistency_max_abs_error": max_err, "annualized_2025_mt": annualized_2025,
        "jan_sep_fraction_mean": jan_sep_fraction, "jan_sep_fraction_sd": jan_sep_fraction_sd,
        "spatial_weight_type": "land-border plus island bridge (Hainan-Guangdong/Guangxi)",
        "spatial_moran": spatial.to_dict("records"), "kruskal": kruskal.to_dict("records"),
        "pca_explained": pca.explained_variance_ratio_.tolist(), "cluster_silhouette": stability.to_dict("records"),
        "cluster_decision": cluster_decision, "cluster_names": class_names,
        "q4_policy_model": {"thresholds": q4_thresholds, "province_policy_rows": int(len(q4_province)), "class_policy_rows": int(len(q4_class)), "phase_rows": int(len(q4_phase)), "sector_rows": int(len(q4_sector))},
        "model_selection": "STIRPAT-ridge retained as the energy-structure model; trend ridge is a pure-error baseline and is not used for q3",
        "model_metrics": model_metrics, "calibration_factor_2025": calibration,
        "peak_by_scenario": {scenario: {"peak_year": int(group.loc[group.pred_mt.idxmax(), "year"]), "peak_mt": float(group.pred_mt.max()), "intensity_2045": float(group.loc[group.year == 2045, "intensity_mt_per_trillion_yuan"].iloc[0])} for scenario, group in forecast.groupby("scenario")},
    }
    (results / "关键指标.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    make_figures(annual, province, spatial, coefficients, backtest, forecast, q4_class, q4_phase, q4_sector, figures)
    runner_path = Path(__file__).resolve()
    manifest = {
        "seed": args.seed, "created_at": datetime.now().isoformat(), "python": sys.version,
        "input_files": {str(path): sha256(path) for path in [input_root / "附件1-中国2019年-2025年碳排放数据.csv", input_root / "附件2-2022年30个省份排放清单.xlsx", project / "data" / "全国年度驱动变量_来源数据.csv"]},
        "parameters": {"moran_permutations": 999, "cluster_k": cluster_decision["selected_k"], "cluster_selection_rule": cluster_decision["rule"], "scenario_defs": "see Q3_scenario_forecast.py", "annualization": "mean 2019-2024 Jan-Sep share"},
        "command": f'"E:\\Anaconda\\envs\\math_modeling\\python.exe" "{runner_path}" --project-root "{project}" --input-root "{input_root}" --seed {args.seed}',
    }
    (results / "复现清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "input_rows": len(daily), "provinces": len(province), "annualized_2025_mt": annualized_2025, "backtest": model_metrics, "peaks": metrics["peak_by_scenario"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
