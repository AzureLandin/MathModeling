"""Review persisted Q3 factorial results without training or dispatch execution."""

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/q3_bias_quantile_cost"
BASELINE = ROOT / "results/q3_rolling_baseline"
OUT = ROOT / "results/q3_bias_quantile_cost_review/review.json"
ENERGY_TOL = 1e-6
CASH_TOL = 1e-4


def maximum(values):
    values = np.asarray(values, dtype=float)
    assert np.isfinite(values).all()
    return float(np.abs(values).max())


def matched_difference(left, right):
    assert left.shape == right.shape
    assert np.array_equal(np.isnan(left), np.isnan(right))
    valid = ~np.isnan(left)
    return maximum(left[valid] - right[valid])


def main():
    assert Path(sys.prefix).name == "math_modeling"
    manifest = json.loads((SOURCE / "run_manifest.json").read_text(encoding="utf-8"))
    registration = json.loads((SOURCE / "registration.json").read_text(encoding="utf-8"))
    hashes = {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == expected
              for name, expected in manifest["outputs"].items()}
    assert all(hashes.values())
    assert manifest["signature"] == registration["signature"]
    result = {"scope": "saved ledgers, feedback identities, summaries and six history targets; no solver",
              "signature": manifest["signature"], "manifest_hash_checks": hashes,
              "wall_seconds_from_manifest": manifest["wall_seconds"], "groups": {}}
    with np.load(ROOT / "results/q3_bias_correction_diagnostic/bias_forecast_archive.npz") as z:
        bias = {key: z[key] for key in z.files}
    with np.load(SOURCE / "protection_q75.npz") as z:
        protection = {key: z[key] for key in z.files}
    summary = pd.read_csv(SOURCE / "summary.csv").set_index("strategy_id")
    dates = pd.date_range("2025-02-01", "2026-01-01", freq="10min")
    k = (dates.normalize() - pd.Timestamp("2025-01-01")).days.to_numpy().copy()
    h = (dates.hour * 6 + dates.minute // 10).to_numpy().copy()
    k[-1], h[-1] = 364, 144
    truth = (bias["truth_load"][k, h] - bias["truth_pv"][k, h]) / 6
    daily, frames = {}, {}
    for sid in ("L80", "L75", "C80", "C75"):
        folder = BASELINE if sid == "L80" else SOURCE / sid
        prefix = "B2_" if sid == "L80" else ""
        f = pd.read_csv(folder / f"{prefix}dispatch.csv", parse_dates=["interval_start"])
        assert f.interval_start.tolist() == dates.tolist()
        assert len(f) == 48097
        for col in ("q0_kWh", "q_eff_kWh", "charge_kWh", "discharge_kWh", "emergency_kWh", "unused_kWh"):
            assert np.isfinite(f[col]).all() and f[col].min() >= -ENERGY_TOL
        state = f.state_start_kWh.to_numpy()
        assert f[["state_start_kWh", "state_end_kWh"]].min().min() >= 1200 - ENERGY_TOL
        assert f[["state_start_kWh", "state_end_kWh"]].max().max() <= 10800 + ENERGY_TOL
        assert f[["charge_kWh", "discharge_kWh"]].max().max() <= 5000 / 6 + ENERGY_TOL
        surplus = f.q_eff_kWh.to_numpy() - truth
        charge = np.minimum.reduce([np.maximum(surplus, 0), np.full(len(f), 5000 / 6), (10800 - state) / 0.9])
        discharge = np.minimum.reduce([np.maximum(-surplus, 0), np.full(len(f), 5000 / 6), (state - 1200) * 0.9])
        emergency = np.maximum(-surplus - discharge, 0)
        unused = np.maximum(surplus - charge, 0)
        errors = {
            "truth_kWh": maximum(f.net_kWh - truth),
            "charge_feedback_kWh": maximum(f.charge_kWh - charge),
            "discharge_feedback_kWh": maximum(f.discharge_kWh - discharge),
            "emergency_feedback_kWh": maximum(f.emergency_kWh - emergency),
            "unused_feedback_kWh": maximum(f.unused_kWh - unused),
            "soc_recurrence_kWh": maximum(f.state_end_kWh - state - .9 * f.charge_kWh + f.discharge_kWh / .9),
            "soc_continuity_kWh": maximum(state[1:] - f.state_end_kWh.to_numpy()[:-1]),
        }
        assert max(errors.values()) < ENERGY_TOL
        p = f.price_yuan_kWh.to_numpy()
        parts = {"ordinary_cost_yuan": p * f.q_eff_kWh.to_numpy(),
                 "adjustment_cost_yuan": .5 * p * np.abs(f.q_eff_kWh - f.q0_kWh).to_numpy(),
                 "emergency_cost_yuan": 5 * p * f.emergency_kWh.to_numpy()}
        cash = sum(parts.values())
        for col, values in parts.items():
            assert maximum(f[col] - values) < CASH_TOL
        values = {col: float(array[:-1].sum()) for col, array in parts.items()}
        natural = f.iloc[:-1]
        active = natural.emergency_kWh.to_numpy() > ENERGY_TOL
        day = natural.interval_start.dt.normalize()
        new_day = np.r_[True, day.to_numpy()[1:] != day.to_numpy()[:-1]]
        values.update(natural_total_yuan=float(cash[:-1].sum()), template_total_yuan=float(cash[1:].sum()),
                      emergency_kWh=float(natural.emergency_kWh.sum()),
                      emergency_intervals=int(active.sum()), emergency_days=int(day[active].nunique()),
                      emergency_events=int(np.sum(active & (new_day | ~np.r_[False, active[:-1]]))),
                      final_natural_state_kWh=float(natural.state_end_kWh.iloc[-1]),
                      final_template_state_kWh=float(f.state_end_kWh.iloc[-1]),
                      initial_state_kWh=float(state[0]))
        assert maximum([value - summary.loc[sid, col] for col, value in values.items()]) < CASH_TOL
        assert abs(state[0] - 6075.795025925926) < ENERGY_TOL
        assert abs(f.q_eff_kWh.iloc[0]) < ENERGY_TOL and abs(f.q0_kWh.iloc[0]) < ENERGY_TOL
        dec = pd.read_csv(folder / f"{prefix}revision_decisions.csv")
        expected = dec.new_predicted_total_yuan < dec.old_predicted_total_yuan - CASH_TOL
        assert np.array_equal(dec.accepted, expected)
        rejected = dec[~dec.accepted]
        largest = rejected.loc[rejected.revision_kWh.idxmax()]
        values["rejected"] = {"n": len(rejected), "hours": sorted(rejected.update_hour.unique().tolist()),
                              "largest_total_kWh": float(largest.revision_kWh),
                              "largest_window_segments": int(largest.remaining_intervals),
                              "largest_date": largest.date,
                              "candidate_minus_accepted_kWh": float(rejected.revision_kWh.sum())}
        values["errors"] = errors
        result["groups"][sid] = values
        daily[sid] = pd.Series(cash[:-1], index=natural.interval_start).resample("D").sum()
        frames[sid] = f
    contrasts = {}
    for name, a, b in (("quantile_L", "L75", "L80"), ("bias_80", "C80", "L80"),
                       ("quantile_C", "C75", "C80"), ("bias_75", "C75", "L75")):
        delta = daily[a] - daily[b]
        top = delta.loc[delta.abs().nlargest(5).index]
        contrasts[name] = dict(total_yuan=float(delta.sum()), better_days=int((delta < -CASH_TOL).sum()),
                               better_months=int((delta.resample("MS").sum() < -CASH_TOL).sum()),
                               top_five=[dict(date=str(date.date()), delta_yuan=float(value)) for date, value in top.items()],
                               top_five_sum_yuan=float(top.sum()), other_days_yuan=float(delta.sum() - top.sum()))
    result["contrasts"] = contrasts
    result["interaction_yuan"] = contrasts["quantile_C"]["total_yuan"] - contrasts["quantile_L"]["total_yuan"]
    result["q80_alignment"] = {}
    history = []
    for label, suffix in (("linear", "c0"), ("corrected", "c1")):
        for key in ("rho", "protected"):
            diff = matched_difference(protection[f"{label}_{key}_q80"], bias[f"{key}_{suffix}"])
            assert diff < 1e-8
            result["q80_alignment"][f"{label}_{key}"] = diff
        for target_day, v, hour, target in (("2025-06-21", 0, 0, 0), ("2025-06-21", 0, 0, 144),
                                           ("2025-06-21", 1, 6, 36)):
            current = (pd.Timestamp(target_day) - pd.Timestamp("2025-01-01")).days
            days = np.array([j for j in range(current - 28, current)
                             if 144 * j + target + 1 <= 144 * current + 6 * hour])
            eps = ((bias["truth_load"][days, target] - bias["truth_pv"][days, target]) / 6
                   - bias[f"net_{suffix}"][days, v, target])
            eps = np.sort(eps[np.isfinite(eps)])
            position = math.ceil(.75 * len(eps))
            value = eps[position - 1] if len(eps) >= 7 else 0.0
            diff = abs(value - protection[f"{label}_rho_q75"][current, v, target])
            assert diff < 1e-8
            history.append(dict(predictor=label, h=target, hour=hour, n=len(eps), position=position, difference_kWh=float(diff)))
    result["q75_history_spotchecks"] = history
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "manifest_hash_checks"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
