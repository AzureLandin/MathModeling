"""Convert issued hourly PV forecasts to interval averages; use math_modeling."""
from datetime import datetime, time
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import openpyxl
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "附件" / "附件3.xlsx"
OUT = ROOT / "results" / "q3_pv_time_conversion"
STEP_MINUTES = 10
TOL = 1e-8


def read_versions(path):
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = iter(workbook.worksheets[0].values)
        header = next(rows)
        assert list(header[2:]) == [f"预报{i}小时" for i in range(1, 25)]
        versions = []
        day = None
        for source_row, row in enumerate(rows, 2):
            if row[0] not in (None, ""):
                day = pd.Timestamp(row[0]).normalize()
            assert day is not None
            value = row[1]
            clock = value if isinstance(value, time) else datetime.strptime(str(value), "%H:%M").time()
            assert clock.hour in (0, 6, 12, 18) and clock.minute == 0
            issued = day + pd.Timedelta(hours=clock.hour)
            power = np.asarray(row[2:], dtype=float)
            assert power.shape == (24,) and np.isfinite(power).all() and (power >= 0).all()
            versions.append((issued, power, source_row))
        expected = pd.date_range("2025-01-01", "2025-12-31 18:00", freq="6h")
        assert pd.DatetimeIndex([v[0] for v in versions]).equals(expected)
        return versions
    finally:
        workbook.close()


def convert(versions):
    nodes, intervals = [], []
    max_hour_error = 0.0
    max_node_error = 0.0
    previous = None
    for issued, power, source_row in versions:
        if previous is None:
            anchor = power[0]
            anchor_issue = issued
            anchor_kind = "initial_first_forecast_hold"
        else:
            old_issue, old_power, _ = previous
            lead = (issued - old_issue) / pd.Timedelta(hours=1)
            assert lead == int(lead) and 1 <= lead <= 24
            anchor = old_power[int(lead) - 1]
            anchor_issue = old_issue
            anchor_kind = "previous_issue_at_current_time"
            assert anchor_issue < issued
        values = np.r_[anchor, power]
        for hour, value in enumerate(values):
            nodes.append(dict(issued_at=issued, target_at=issued + pd.Timedelta(hours=hour),
                              lead_hours=hour, pv_kW=float(value), source_row=source_row,
                              kind=anchor_kind if hour == 0 else "source_hourly_forecast",
                              value_source_issued_at=anchor_issue if hour == 0 else issued))
        # The extra interval is needed only by the midnight template commitment.
        count = 144 + int(issued.hour == 0)
        minutes = np.arange(count + 1) * STEP_MINUTES
        endpoints = np.interp(minutes, np.arange(25) * 60, values)
        means = (endpoints[:-1] + endpoints[1:]) / 2
        energy = means * STEP_MINUTES / 60
        expected_energy = (values[:-1] + values[1:]) / 2
        max_hour_error = max(max_hour_error, float(np.max(np.abs(
            energy[:144].reshape(24, 6).sum(axis=1) - expected_energy))))
        max_node_error = max(max_node_error, float(np.max(np.abs(endpoints[:145:6] - values))))
        assert np.isfinite(energy).all() and (energy >= 0).all()
        if count == 145:
            assert abs(energy[-1] - power[-1] / 6) < TOL
        for j in range(count):
            start = issued + pd.Timedelta(minutes=int(minutes[j]))
            kind = anchor_kind if j < 6 else ("tail_constant" if j == 144 else "linear")
            intervals.append(dict(issued_at=issued, interval_start=start,
                                  interval_end=start + pd.Timedelta(minutes=STEP_MINUTES),
                                  lead_start_minutes=int(minutes[j]),
                                  pv_start_kW=float(endpoints[j]), pv_end_kW=float(endpoints[j + 1]),
                                  pv_mean_kW=float(means[j]), pv_energy_kWh=float(energy[j]),
                                  boundary_kind=kind, anchor_source_issued_at=anchor_issue,
                                  source_row=source_row,
                                  in_midnight_template=issued.hour == 0 and 1 <= j <= 144))
        previous = (issued, power, source_row)
    assert max_hour_error < TOL and max_node_error < TOL
    return pd.DataFrame(nodes), pd.DataFrame(intervals), dict(
        max_hour_energy_error_kWh=max_hour_error, max_hourly_node_error_kW=max_node_error)


def main():
    assert Path(sys.prefix).name == "math_modeling", sys.prefix
    versions = read_versions(SOURCE)
    nodes, intervals, checks = convert(versions)
    assert len(nodes) == 1460 * 25
    assert len(intervals) == 1460 * 144 + 365
    assert int(intervals.in_midnight_template.sum()) == 365 * 144
    assert not intervals.duplicated(["issued_at", "interval_start"]).any()
    # A shortened issue history must produce the same already-issued forecasts.
    _, prefix, _ = convert(versions[:5])
    pd.testing.assert_frame_equal(prefix, intervals.iloc[:len(prefix)].reset_index(drop=True))
    assert ((3100 + 3200) / 2) / 6 == 525
    OUT.mkdir(parents=True, exist_ok=True)
    node_path = OUT / "pv_hourly_nodes.csv"
    interval_path = OUT / "pv_10min_forecasts.csv"
    nodes.to_csv(node_path, index=False, encoding="utf-8-sig")
    intervals.to_csv(interval_path, index=False, encoding="utf-8-sig")
    saved = pd.read_csv(interval_path)
    assert len(saved) == len(intervals)
    np.testing.assert_allclose(saved.pv_energy_kWh, intervals.pv_energy_kWh, rtol=0, atol=TOL)
    checks.update(status="passed", issue_count=len(versions), hourly_node_count=len(nodes),
                  interval_count=len(intervals), midnight_template_count=int(intervals.in_midnight_template.sum()),
                  boundary_counts=intervals.boundary_kind.value_counts().to_dict(),
                  historical_prefix_unchanged=True, csv_readback_passed=True,
                  source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  python=sys.version, executable=sys.executable,
                  dependencies={"numpy": np.__version__, "pandas": pd.__version__, "openpyxl": openpyxl.__version__},
                  step_minutes=STEP_MINUTES, tolerance=TOL,
                  timestamps="Asia/Shanghai local time; naive timestamps",
                  scope="All issued forecasts, not a dispatch schedule or observed energy; no actual data used.",
                  outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (node_path, interval_path)})
    (OUT / "validation.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: checks[k] for k in ("status", "issue_count", "interval_count", "max_hour_energy_error_kWh")}, indent=2))


if __name__ == "__main__":
    main()
