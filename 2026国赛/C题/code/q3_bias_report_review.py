"""Read saved Q3 bias diagnostics; do not regenerate forecasts or dispatch."""

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/q3_bias_correction_diagnostic"
OUTPUT = ROOT / "results/q3_bias_report_review/review.json"
HOURS = (0, 6, 12, 18)
STARTS = (1, 36, 72, 108)
DT = 1 / 6


def metrics(a, mask):
    truth_pv = np.broadcast_to(a["truth_pv"][:, None, :], mask.shape)[mask]
    truth_net = np.broadcast_to(
        ((a["truth_load"] - a["truth_pv"]) * DT)[:, None, :], mask.shape
    )[mask]
    result = {"n": int(mask.sum())}
    for name, pv_key, suffix in (("C0", "pv_linear", "c0"), ("C1", "pv_corrected", "c1")):
        pv_error = a[pv_key][mask] - truth_pv
        net_error = truth_net - a[f"net_{suffix}"][mask]
        error = truth_net - a[f"protected_{suffix}"][mask]
        result[name] = {
            "pv_mae_kW": float(np.abs(pv_error).mean()),
            "pv_rmse_kW": float(np.sqrt(np.mean(pv_error ** 2))),
            "pv_bias_kW": float(pv_error.mean()),
            "net_mae_kWh": float(np.abs(net_error).mean()),
            "net_rmse_kWh": float(np.sqrt(np.mean(net_error ** 2))),
            "shortfall_kWh": float(np.maximum(error, 0).mean()),
            "surplus_kWh": float(np.maximum(-error, 0).mean()),
        }
    delta = (a["protected_c1"] - a["protected_c0"])[mask]
    result.update(
        mean_delta_protected_kWh=float(delta.mean()),
        mean_abs_delta_protected_kWh=float(np.abs(delta).mean()),
        max_abs_delta_protected_kWh=float(np.abs(delta).max()),
        mean_delta_pv_kW=float(a["delta_v_kW"][mask].mean()),
        mean_abs_delta_pv_kW=float(np.abs(a["delta_v_kW"][mask]).mean()),
    )
    return result


def main():
    assert Path(sys.prefix).name == "math_modeling", sys.prefix
    path = SOURCE / "bias_forecast_archive.npz"
    with np.load(path, allow_pickle=False) as archive:
        a = {key: archive[key] for key in archive.files}
    mask = np.zeros(a["pv_linear"].shape, dtype=bool)
    for v, h in enumerate(STARTS):
        mask[31:365, v, h:145] = True
    result = {"scope": "saved-array review and four target history checks; no training, MILP or dispatch",
              "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "overall": metrics(a, mask), "versions": {}}
    comparison = pd.read_csv(SOURCE / "comparison.csv")
    mapping = {"pv_mae_kW": "pv_mae", "pv_rmse_kW": "pv_rmse",
               "net_mae_kWh": "net_mae", "net_rmse_kWh": "net_rmse",
               "shortfall_kWh": "mean_shortfall_kWh", "surplus_kWh": "mean_surplus_kWh"}
    differences = []
    for v, hour in enumerate(HOURS):
        selected = mask.copy()
        selected[:, np.arange(4) != v] = False
        m = metrics(a, selected)
        result["versions"][str(hour)] = m
        row = comparison[(comparison.group_type == "publication_hour") & (comparison.v == v)].iloc[0]
        for scheme, prefix in (("C0", "a"), ("C1", "b")):
            for key, column in mapping.items():
                differences.append(abs(m[scheme][key] - row[f"{prefix}_{column}"]))
    result["max_metric_difference_from_csv"] = max(differences)
    assert max(differences) < 1e-8
    result["clipped_subset"] = metrics(a, mask & a["truncated"])
    result["unclipped_subset"] = metrics(a, mask & ~a["truncated"])

    # Reconstruct only four histories using interval-end timestamps, not the saved availability mask.
    k = (pd.Timestamp("2025-06-21") - pd.Timestamp("2025-01-01")).days
    checks = []
    for v, h in ((0, 0), (0, 1), (0, 144), (1, 36)):
        days = [j for j in range(max(0, k - 28), k)
                if j * 144 + h + 1 <= k * 144 + HOURS[v] * 6]
        errors = [a["truth_pv"][j, h] - a["pv_linear"][j, v, h] for j in days]
        errors = np.array(errors)
        errors = errors[np.isfinite(errors)]
        bias = float(errors.mean()) if errors.size >= 7 else 0.0
        residuals = [((a["truth_load"][j, h] - a["truth_pv"][j, h]) -
                      (a["issued_load"][j, h] - a["pv_corrected"][j, v, h])) * DT for j in days]
        residuals = np.sort(np.array(residuals)[np.isfinite(residuals)])
        rho = float(residuals[math.ceil(0.8 * len(residuals)) - 1]) if len(residuals) >= 7 else 0.0
        item = dict(hour=HOURS[v], h=h, m_bias=int(errors.size), m_net=int(residuals.size),
                    bias_difference_kW=abs(bias - a["bias_kW"][k, v, h]),
                    rho_difference_kWh=abs(rho - a["rho_c1"][k, v, h]))
        assert item["bias_difference_kW"] < 1e-8 and item["rho_difference_kWh"] < 1e-8
        checks.append(item)
    result["history_spotchecks"] = checks
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
