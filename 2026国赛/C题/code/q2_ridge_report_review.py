"""Reconcile reported Q2 ridge results; this is not a full forecast audit."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/q2_ridge_forecast"
OUTPUT = ROOT / "results/q2_ridge_report_review"


def slot_correlations(errors, lag):
    x = errors[:-lag] - errors[:-lag].mean(axis=0)
    y = errors[lag:] - errors[lag:].mean(axis=0)
    scale = np.sqrt((x * x).sum(axis=0) * (y * y).sum(axis=0))
    valid = scale > 1e-12
    correlations = (x * y).sum(axis=0)[valid] / scale[valid]
    return {"mean": float(correlations.mean()),
            "median": float(np.median(correlations)),
            "valid_slots": int(valid.sum())}


def main():
    paths = [SOURCE / "comparison.csv", SOURCE / "selection_log.csv"]
    comparison = pd.read_csv(paths[0]).set_index("strategy_id")
    selection = pd.read_csv(paths[1])
    result = {"scope": "Independent read-only reconciliation of existing dispatch and logs; "
                       "no refit, MILP replay, source-data or information-boundary audit",
              "evaluation": "2025-02-01..2025-12-31",
              "correlation_scope": "Descriptive same-year, no significance or prediction claim",
              "selection": {}, "groups": {}, "source_sha256": {}}
    for target, rows in selection.groupby("target"):
        evaluation = rows[rows.date.between("2025-02-01", "2025-12-31")]
        assert len(evaluation) == 334 and evaluation.date.nunique() == 334
        result["selection"][target] = {
            "all_rows": len(rows), "evaluation_rows": len(evaluation),
            "all_counts": rows.chosen_lambda.value_counts().sort_index().to_dict(),
            "evaluation_counts": evaluation.chosen_lambda.value_counts().sort_index().to_dict(),
        }
    baseline_cost = None
    for strategy, expected in comparison.iterrows():
        path = SOURCE / strategy / "dispatch.csv"
        paths.append(path)
        frame = pd.read_csv(path).sort_values(["date", "slot"])
        assert len(frame) == 48096 and not frame[["date", "slot"]].duplicated().any()
        assert np.array_equal(frame.slot.to_numpy(), np.tile(np.arange(144), 334))
        assert frame.date.nunique() == 334
        q, c, d, e, w, start, end, price, load, pv = [frame[column].to_numpy() for column in
            ("planned_kWh", "charge_kWh", "discharge_kWh", "emergency_kWh", "unused_kWh",
             "state_start_kWh", "state_end_kWh", "price_yuan_kWh", "load_kW", "pv_kW")]
        net = (load - pv) / 6
        loss = 0.1 * c + (1 / 0.9 - 1) * d
        cash = price * q + 5 * price * e
        if baseline_cost is None:
            assert strategy == "R0_naive"
            baseline_cost = cash.copy()
        emergency = e > 1e-6
        event_start = emergency & np.r_[True, ~emergency[:-1] |
            (frame.date.to_numpy()[1:] != frame.date.to_numpy()[:-1])]
        monthly_delta = pd.Series(cash - baseline_cost, index=frame.date.str[:7]).groupby(level=0).sum()
        metrics, correlations = {}, {}
        for target, actual, forecast in (
            ("load", load, frame.load_forecast_kW.to_numpy()),
            ("pv", pv, frame.pv_forecast_kW.to_numpy()),
            ("net", load - pv, (frame.load_forecast_kW - frame.pv_forecast_kW).to_numpy()),
        ):
            error = actual - forecast
            metrics[target] = {"mae_kW": float(abs(error).mean()),
                               "rmse_kW": float(np.sqrt(np.mean(error ** 2))),
                               "bias_kW": float(error.mean())}
            matrix = error.reshape(334, 144)
            correlations[target] = {lag: slot_correlations(matrix, lag) for lag in (1, 2, 7)}
            if strategy == "R0_naive" and target == "net":
                curve_correlations = [np.corrcoef(matrix[i], matrix[i + 1])[0, 1]
                                      for i in range(333)]
                result["naive_adjacent_day_curve_correlation_mean"] = float(np.mean(curve_correlations))
        record = {"total_yuan": float(cash.sum()),
                  "delta_yuan": float((cash - baseline_cost).sum()),
                  "summary_difference_yuan": float(cash.sum() - expected.total_cost_yuan),
                  "emergency_days": int(frame.loc[emergency, "date"].nunique()),
                  "emergency_events": int(event_start.sum()), "emergency_slots": int(emergency.sum()),
                  "emergency_cost_yuan": float(np.sum(5 * price * e)),
                  "max_balance_kWh": float(np.max(abs(q + e + d - net - c - w))),
                  "max_state_equation_kWh": float(np.max(abs(end - start - .9 * c + d / .9))),
                  "max_continuity_kWh": float(np.max(abs(start[1:] - end[:-1]))),
                  "energy_identity_kWh": float(np.sum(q + e - net - w - loss) - (end[-1] - start[0])),
                  "improved_months": int((monthly_delta < -1e-6).sum()),
                  "worse_months_yuan": monthly_delta[monthly_delta > 1e-6].to_dict(),
                  "metrics": metrics, "same_slot_day_lag_correlations": correlations}
        assert abs(record["summary_difference_yuan"]) < 1e-4
        assert abs(record["energy_identity_kWh"]) < 1e-4
        assert max(record[key] for key in ("max_balance_kWh", "max_state_equation_kWh",
                                          "max_continuity_kWh")) < 1e-6
        result["groups"][strategy] = record
    paths.append(Path(__file__).resolve())
    for path in paths:
        result["source_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    OUTPUT.mkdir(exist_ok=True)
    destination = OUTPUT / "review.json"
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(destination), "selection": result["selection"],
                      "groups": {key: {field: row[field] for field in
                                 ("total_yuan", "emergency_days", "emergency_events", "improved_months")}
                                 for key, row in result["groups"].items()},
                      "naive_curve_corr": result["naive_adjacent_day_curve_correlation_mean"],
                      "naive_same_slot_day1_corr": result["groups"]["R0_naive"]
                      ["same_slot_day_lag_correlations"]["net"][1]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
