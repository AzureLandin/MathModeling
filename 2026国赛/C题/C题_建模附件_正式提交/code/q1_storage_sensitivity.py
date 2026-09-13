"""Scan Q1 storage capacity and charge/discharge power sensitivity.

The scan keeps the Q1 baseline inputs, price, efficiency and 10-minute
discretization unchanged. For nominal capacity C, the physical state bounds
are 0.1*C and 0.9*C, with initial and terminal state fixed at 0.5*C. This
keeps the baseline's 1200/10800/6000 kWh proportions comparable across C. The
grid is moderately refined around the baseline rather than densely sampled.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "q1_sensitivity"
DATA = ROOT / "results" / "audit" / "q1_normalized.csv"


def load_baseline_module():
    spec = importlib.util.spec_from_file_location("q1_baseline", ROOT / "code" / "02_q1_baseline.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_case(solver, load, pv, price, capacity, power, scan_axis):
    state_min = 0.1 * capacity
    state_max = 0.9 * capacity
    initial = 0.5 * capacity
    _, result = solver(
        load, pv, price, power=power, initial_kWh=initial,
        terminal_kWh=initial, state_min_kWh=state_min,
        state_max_kWh=state_max,
    )
    return {
        "capacity_kWh": capacity,
        "power_max_kW": power,
        "state_min_kWh": state_min,
        "state_max_kWh": state_max,
        "initial_terminal_kWh": initial,
        "cost_yuan": result["cost_yuan"],
        "grid_kWh": result["grid_kWh"],
        "charge_kWh": result["charge_kWh"],
        "discharge_kWh": result["discharge_kWh"],
        "state_min_result_kWh": result["state_min_kWh"],
        "state_max_result_kWh": result["state_max_kWh"],
        "solver": result["solver"],
        "elapsed_seconds": result["elapsed_seconds"],
        "max_check": max(result["checks"].values()),
        "scan_axis": scan_axis,
    }


def main():
    solver_module = load_baseline_module()
    data = pd.read_csv(DATA)
    load, pv, price = [data[c].to_numpy() for c in ["load_kW", "pv_kW", "price_yuan_kWh"]]

    # Moderate refinement: keep the broad range and add resolution near the
    # baseline, without turning the sensitivity figure into a dense scan.
    capacities = [6000.0, 8000.0, 10000.0, 12000.0, 14000.0, 16000.0, 18000.0]
    powers = [2000.0, 3000.0, 4000.0, 4500.0, 5000.0, 5500.0, 6000.0, 7000.0]
    rows = []
    for capacity in capacities:
        rows.append(run_case(solver_module.solve, load, pv, price, capacity, 5000.0, "capacity"))
    for power in powers:
        rows.append(run_case(solver_module.solve, load, pv, price, 12000.0, power, "power"))

    result = pd.DataFrame(rows)
    baseline = float(result.loc[(result.capacity_kWh == 12000.0) & (result.power_max_kW == 5000.0), "cost_yuan"].iloc[0])
    result["cost_change_yuan"] = result["cost_yuan"] - baseline
    result["cost_change_pct"] = 100 * result["cost_change_yuan"] / baseline
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "capacity_power_sensitivity.csv", index=False, encoding="utf-8-sig")
    capacity_result = result[(result.power_max_kW == 5000.0)].copy()
    power_result = result[(result.capacity_kWh == 12000.0)].copy()
    capacity_result.to_csv(OUT / "capacity_sensitivity.csv", index=False, encoding="utf-8-sig")
    power_result.to_csv(OUT / "power_sensitivity.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "baseline_cost_yuan": baseline,
        "capacity_grid_kWh": capacities,
        "power_grid_kW": powers,
        "state_bound_rule": "state_min=0.1*C, state_max=0.9*C, initial=terminal=0.5*C",
        "input": str(DATA),
        "solver": "Q1 baseline HiGHS LP",
        "max_check": float(result["max_check"].max()),
    }
    (OUT / "scan_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result[["scan_axis", "capacity_kWh", "power_max_kW", "cost_yuan", "cost_change_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
