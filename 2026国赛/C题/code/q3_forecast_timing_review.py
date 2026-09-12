"""Read saved timing diagnostics and ledgers; no forecasting or dispatch."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/q3_forecast_timing_diagnostic"
OUT = ROOT / "results/q3_forecast_timing_review/review.json"


def main():
    assert Path(sys.prefix).name == "math_modeling"
    manifest = json.loads((SOURCE / "run_manifest.json").read_text(encoding="utf-8"))
    hashes = {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == expected
              for name, expected in manifest["outputs"].items()}
    assert all(hashes.values())
    f = pd.read_csv(SOURCE / "interval_diagnostic.csv", parse_dates=["interval_start"])
    hourly = pd.read_csv(SOURCE / "hourly_summary.csv")
    result = {"signature": manifest["signature"], "hash_checks": hashes, "groups": {}}
    for sid, path in (("L75", ROOT / "results/q3_bias_quantile_cost/L75/dispatch.csv"),
                      ("L80", ROOT / "results/q3_rolling_baseline/B2_dispatch.csv")):
        a = f[f.strategy_id == sid].sort_values("interval_start").reset_index(drop=True)
        raw = pd.read_csv(path, parse_dates=["interval_start"]).iloc[:-1].reset_index(drop=True)
        assert len(a) == 48096 and a.interval_start.equals(raw.interval_start)
        for col in ("actual_pv_kW", "actual_load_kW", "emergency_kWh", "emergency_cost_yuan", "state_start_kWh"):
            assert np.allclose(a[col], raw[col], rtol=0, atol=1e-8)
        assert int(a.is_midnight.sum()) == 334
        paired = a[~a.is_midnight]
        assert len(paired) == 47762
        assert (paired.version == paired.clock_hour // 6).all()
        assert np.allclose(paired.age_hours, (paired.target_h - 36 * paired.version) / 6, rtol=0, atol=1e-8)
        groups = paired.groupby(["publication_hour", "age_bin"]).clock_hour.nunique()
        assert len(groups) == 24 and (groups == 1).all()
        total = float(a.emergency_cost_yuan.sum())
        hours = {}
        for hour, block in a.groupby("clock_hour"):
            em = block[block.emergency_kWh > 1e-6]
            em_valid = em[~em.is_midnight]
            daily = em.groupby(em.interval_start.dt.date).emergency_cost_yuan.sum()
            raw_em = raw.loc[em.index]
            item = dict(cost_yuan=float(block.emergency_cost_yuan.sum()),
                        energy_kWh=float(block.emergency_kWh.sum()), emergency_intervals=len(em),
                        months=int(em.interval_start.dt.month.nunique()),
                        actual_pv_positive=int((block.actual_pv_kW > 0).sum()),
                        actual_pv_max_kW=float(block.actual_pv_kW.max()),
                        actual_pv_energy_kWh=float(block.actual_pv_kW.sum() / 6),
                        emergency_actual_pv_positive=int((em.actual_pv_kW > 0).sum()),
                        top_three_share=float(daily.nlargest(3).sum() / daily.sum()) if len(em) else None,
                        median_start_kWh=float(em.state_start_kWh.median()) if len(em) else None,
                        median_end_kWh=float(raw_em.state_end_kWh.median()) if len(em) else None,
                        start_above_minimum=int((em.state_start_kWh > 1200 + 1e-6).sum()),
                        mean_g_kWh=float(em_valid.g_kWh.mean()) if len(em_valid) else None,
                        pv_over_share=float((em_valid.a_pv_kWh > 0).mean()) if len(em_valid) else None)
            hours[str(hour)] = item
        bands = {}
        for name, selected in (("morning_06_10", range(6, 11)), ("evening_19_23", range(19, 24))):
            block = a[a.clock_hour.isin(selected)]
            bands[name] = dict(cost_yuan=float(block.emergency_cost_yuan.sum()),
                               cost_share=float(block.emergency_cost_yuan.sum() / total),
                               energy_kWh=float(block.emergency_kWh.sum()),
                               actual_pv_positive=int((block.actual_pv_kW > 0).sum()),
                               actual_pv_max_kW=float(block.actual_pv_kW.max()),
                               actual_pv_energy_kWh=float(block.actual_pv_kW.sum() / 6))
        result["groups"][sid] = dict(total_cost_yuan=total, total_emergency_kWh=float(a.emergency_kWh.sum()),
                                    zero_emergency_hours=[int(hour) for hour, item in hours.items() if item["emergency_intervals"] == 0],
                                    bands=bands, hours=hours)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"hash_count": len(hashes), "L75_bands": result["groups"]["L75"]["bands"],
                      "zero_hours": result["groups"]["L75"]["zero_emergency_hours"],
                      "selected_hours": {hour: result["groups"]["L75"]["hours"][hour]
                                         for hour in ("9", "10", "19", "20", "21")}}, indent=2))


if __name__ == "__main__":
    main()
