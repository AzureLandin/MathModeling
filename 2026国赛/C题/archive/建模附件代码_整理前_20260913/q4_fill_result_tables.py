#!/usr/bin/env python
"""问题四 正式结果表填报与回读核查 —— 填报层（只读账本，写副本）。

按 ``reports/问题四/问题四_最终收敛与交付方案.md`` 与用户本轮指令：

* 4-2 用 **Q42**、4-3 用 **Q43_S2**；Q43_S0 只作对照，不填报；
* 从 ``results/q4_price_transfer/`` 读取已核验账本，**不重新优化、不重建保护、不重训**；
* 只使用自然日正式评价期 2025-02-01—2025-12-31（334 天），1 月公共初始化不计入正式费用；
* 4-3 同时保留**原始普通购电计划**与**最终有效普通购电计划**；
* **写副本**：``results/q4_delivery/result4-2_filled.xlsx`` 与 ``result4-3_filled.xlsx``，
  附件5 原始模板只读并另存备份，绝不覆盖。

若账本字段不足以填表，脚本先报缺口并中止，不猜测、不补造。

运行::

    conda run -n math_modeling python code/q4_fill_result_tables.py
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code/q4_fill_result_tables.py"
LEDGER_DIR = ROOT / "results/q4_price_transfer"
OUT = ROOT / "results/q4_delivery"
BACKUP = OUT / "template_backup"
TEMPLATES = {name: ROOT / f"附件/附件5/{name}.xlsx" for name in ("result4-2", "result4-3")}

FIRST_DAY = pd.Timestamp("2025-02-01")
DAYS = 334
SLOTS = 144
TAIL = pd.Timestamp("2026-01-01 00:00")
ENERGY_TOL = 1e-6
CASH_TOL = 1e-4

# result4-2: sheet 0 = the 0:00 plan (Q42 has no revision, so it is also the effective plan);
# result4-3: sheet 0 = the original plan, sheet 1 = the final effective plan.
ASSIGNMENT = {
    "result4-2": dict(strategy="Q42", plan_sheets=[("计划购电量", "q0_kWh")]),
    "result4-3": dict(strategy="Q43_S2", plan_sheets=[("计划购电量", "q0_kWh"),
                                                     ("调整购电量", "q_eff_kWh")]),
}
EXPECTED_SHEETS = {"result4-2": ["计划购电量", "充放电量", "紧急购电量"],
                   "result4-3": ["计划购电量", "调整购电量", "充放电量", "紧急购电量"]}
COST_ITEMS = ("ordinary_cost_yuan", "adjustment_cost_yuan", "emergency_cost_yuan")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (Path, pd.Timestamp, datetime)):
        return str(obj)
    return obj


def save_json(path, obj):
    Path(path).write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=2, default=str),
                          encoding="utf-8")


def read_ledger(strategy):
    path = LEDGER_DIR / strategy / "dispatch.csv"
    frame = pd.read_csv(path, parse_dates=["interval_start", "interval_end"])
    frame = frame.sort_values("interval_start").reset_index(drop=True)
    required = ["interval_start", "interval_end", "q0_kWh", "q_eff_kWh", "charge_kWh",
                "discharge_kWh", "emergency_kWh", "unused_kWh", "price_yuan_kWh",
                "state_start_kWh", "state_end_kWh", *COST_ITEMS]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise SystemExit(f"GAP: {strategy} ledger lacks {missing}")
    nonfinite = int((~np.isfinite(frame[[c for c in required if c != 'interval_start'
                                         and c != 'interval_end']].to_numpy(dtype=float))).sum())
    if nonfinite:
        raise SystemExit(f"GAP: {strategy} ledger has {nonfinite} non-finite cell(s)")
    if len(frame) != DAYS * SLOTS + 1:
        raise SystemExit(f"GAP: {strategy} ledger has {len(frame)} rows, expected "
                         f"{DAYS * SLOTS + 1}")
    natural = frame[frame.interval_start < TAIL].copy()
    tail = frame[frame.interval_start == TAIL]
    if len(natural) != DAYS * SLOTS or len(tail) != 1:
        raise SystemExit(f"GAP: {strategy} split gives {len(natural)} natural rows and "
                         f"{len(tail)} tail rows")
    by_day = {day: block for day, block in
              natural.groupby(natural.interval_start.dt.normalize(), sort=True)}
    if len(by_day) != DAYS:
        raise SystemExit(f"GAP: {strategy} covers {len(by_day)} natural days, expected {DAYS}")
    for day, block in by_day.items():
        if len(block) != SLOTS:
            raise SystemExit(f"GAP: {strategy} {day.date()} has {len(block)} segments")
        if block.interval_start.diff().dropna().ne(pd.Timedelta(minutes=10)).any():
            raise SystemExit(f"GAP: {strategy} {day.date()} is not 10-minute contiguous")
    return dict(frame=frame, natural=natural, tail=tail.iloc[0], by_day=by_day,
                source=str(path.relative_to(ROOT)), sha256=digest(path))


def plan_row(ledger, day_index, column):
    """The 144 published slots of day ``day_index``: its own 00:10..23:50 plus the next midnight."""
    day = FIRST_DAY + pd.Timedelta(days=day_index)
    block = ledger["by_day"][day].sort_values("interval_start")
    values = block.iloc[1:][column].astype(float).tolist()
    next_day = day + pd.Timedelta(days=1)
    if next_day in ledger["by_day"]:
        values.append(float(ledger["by_day"][next_day].sort_values("interval_start")
                            .iloc[0][column]))
        price = float(ledger["by_day"][next_day].sort_values("interval_start")
                      .iloc[0]["price_yuan_kWh"])
    else:
        values.append(float(ledger["tail"][column]))
        price = float(ledger["tail"]["price_yuan_kWh"])
    prices = block.iloc[1:]["price_yuan_kWh"].astype(float).tolist() + [price]
    assert len(values) == SLOTS and len(prices) == SLOTS
    return values, prices


def emergency_ranges(ledger):
    """Contiguous positive-emergency ranges inside one natural day, plus clean-day rows."""
    rows = []
    for day in sorted(ledger["by_day"]):
        block = ledger["by_day"][day].sort_values("interval_start")
        mask = block.emergency_kWh.to_numpy() > ENERGY_TOL
        if not mask.any():
            rows.append(dict(date=day, period="无紧急购电", kWh=0.0, intervals=0))
            continue
        index = 0
        values = block.emergency_kWh.to_numpy()
        starts = block.interval_start.to_numpy()
        ends = block.interval_end.to_numpy()
        while index < len(mask):
            if not mask[index]:
                index += 1
                continue
            stop = index
            while stop + 1 < len(mask) and mask[stop + 1]:
                stop += 1
            rows.append(dict(
                date=day, kWh=float(values[index:stop + 1].sum()), intervals=int(stop - index + 1),
                period=f"{pd.Timestamp(starts[index]):%H:%M}-{pd.Timestamp(ends[stop]):%H:%M}"))
            index = stop + 1
    return rows


def fill_sheet_plan(ws, ledger, column, record):
    header = [ws.cell(1, c).value for c in range(1, SLOTS + 4)]
    assert header[-2] == "全天购电量" and header[-1] == "全天购电费", header[-2:]
    assert len([h for h in header[1:SLOTS + 1] if h]) == SLOTS
    worst_total = worst_cost = 0.0
    for day_index in range(DAYS):
        day = FIRST_DAY + pd.Timedelta(days=day_index)
        values, prices = plan_row(ledger, day_index, column)
        row = 2 + day_index
        cell = ws.cell(row, 1)
        if cell.value is None or pd.Timestamp(cell.value).normalize() != day:
            raise SystemExit(f"GAP: template date row {row} is {cell.value}, expected {day}")
        cell.number_format = "yyyy/mm/dd"
        for offset, value in enumerate(values):
            target = ws.cell(row, 2 + offset)
            target.value = float(value)
            target.number_format = "0.000000"
        total = float(sum(values))
        cost = float(sum(v * p for v, p in zip(values, prices)))
        ws.cell(row, SLOTS + 2).value = total
        ws.cell(row, SLOTS + 2).number_format = "0.000000"
        ws.cell(row, SLOTS + 3).value = cost
        ws.cell(row, SLOTS + 3).number_format = "0.000000"
        worst_total = max(worst_total, abs(total - sum(values)))
        worst_cost = max(worst_cost, abs(cost - sum(v * p for v, p in zip(values, prices))))
    record[ws.title] = dict(rows=DAYS, columns=SLOTS, values=DAYS * SLOTS,
                            max_abs_total_recompute_kWh=worst_total,
                            max_abs_cost_recompute_yuan=worst_cost)


def fill_sheet_storage(ws, ledger, record):
    needed = DAYS * 6
    if ws.max_row < 1 + needed:                      # grow in one call: row-by-row is very slow
        ws.insert_rows(ws.max_row + 1, 1 + needed - ws.max_row)
    worst_charge = worst_discharge = worst_state = 0.0
    for day_index in range(DAYS):
        day = FIRST_DAY + pd.Timedelta(days=day_index)
        block = ledger["by_day"][day].sort_values("interval_start")
        for index in range(6):
            row = 2 + day_index * 6 + index
            segment = block.iloc[index * 24:(index + 1) * 24]
            charge = float(segment.charge_kWh.sum())
            discharge = float(segment.discharge_kWh.sum())
            first = ws.cell(row, 1)
            first.value = day.to_pydatetime() if index == 0 else None
            if index == 0:
                first.number_format = "yyyy/mm/dd"
            ws.cell(row, 2).value = f"{index * 4:02d}:00-{(index + 1) * 4:02d}:00"
            ws.cell(row, 3).value = charge
            ws.cell(row, 4).value = discharge
            ws.cell(row, 5).value = "0:00" if index == 0 else ("24:00" if index == 1 else None)
            state = (float(block.state_start_kWh.iloc[0]) if index == 0
                     else (float(block.state_end_kWh.iloc[-1]) if index == 1 else None))
            ws.cell(row, 6).value = state
            for column in (3, 4, 6):
                ws.cell(row, column).number_format = "0.000000"
            worst_charge = max(worst_charge, abs(charge - float(segment.charge_kWh.sum())))
            worst_discharge = max(worst_discharge, abs(discharge - float(segment.discharge_kWh.sum())))
            if state is not None:
                worst_state = max(worst_state, 0.0)
    record[ws.title] = dict(rows=needed, blocks_per_day=6,
                            max_abs_charge_recompute_kWh=worst_charge,
                            max_abs_discharge_recompute_kWh=worst_discharge,
                            state_column="day-start state in block 0 and day-end state in block 1")


def fill_sheet_emergency(ws, ledger, record):
    rows = emergency_ranges(ledger)
    for row in range(2, ws.max_row + 1):
        for column in range(1, 4):
            ws.cell(row, column).value = None
    if ws.max_row < 1 + len(rows):
        ws.insert_rows(ws.max_row + 1, 1 + len(rows) - ws.max_row)
    for index, entry in enumerate(rows, start=2):
        cell = ws.cell(index, 1)
        cell.value = entry["date"].date()
        cell.number_format = "yyyy/mm/dd"
        ws.cell(index, 2).value = entry["period"]
        amount = ws.cell(index, 3)
        amount.value = entry["kWh"]
        amount.number_format = "0.000000"
    record[ws.title] = dict(
        rows=len(rows), clean_day_rows=sum(1 for r in rows if r["intervals"] == 0),
        range_rows=sum(1 for r in rows if r["intervals"] > 0),
        covered_intervals=sum(r["intervals"] for r in rows),
        total_kWh=float(sum(r["kWh"] for r in rows)),
        convention="contiguous positive-emergency ranges inside one natural day, one row per day "
                   "even when there is none (same convention as the filled result2.xlsx)")


def build_copy(name, ledger):
    template = TEMPLATES[name]
    if not template.exists():
        raise SystemExit(f"GAP: template {template} is missing")
    OUT.mkdir(parents=True, exist_ok=True)
    BACKUP.mkdir(parents=True, exist_ok=True)
    backup = BACKUP / f"{name}_before_fill.xlsx"
    if not backup.exists():
        shutil.copy2(template, backup)
    target = OUT / f"{name}_filled.xlsx"
    shutil.copy2(template, target)
    wb = load_workbook(target)
    if wb.sheetnames != EXPECTED_SHEETS[name]:
        raise SystemExit(f"GAP: {name} sheets are {wb.sheetnames}, expected "
                         f"{EXPECTED_SHEETS[name]}")
    record = {}
    for sheet_name, column in ASSIGNMENT[name]["plan_sheets"]:
        fill_sheet_plan(wb[sheet_name], ledger, column, record)
    fill_sheet_storage(wb["充放电量"], ledger, record)
    fill_sheet_emergency(wb["紧急购电量"], ledger, record)
    page = wb["计划购电量"]
    page.freeze_panes = page.freeze_panes or "B2"
    wb.save(target)
    return target, backup, record


def read_back(name, ledger, record):
    """Re-read the written copy and compare it against the ledger, cell by cell."""
    target = OUT / f"{name}_filled.xlsx"
    wb = load_workbook(target, data_only=True)
    checks = {"sheet_names": wb.sheetnames, "plan": {}, "storage": {}, "emergency": {}}
    for sheet_name, column in ASSIGNMENT[name]["plan_sheets"]:
        ws = wb[sheet_name]
        header = [ws.cell(1, c).value for c in range(1, SLOTS + 4)]
        assert header[0] == "日期\\时间" and header[-2:] == ["全天购电量", "全天购电费"]
        labels = [h for h in header[1:SLOTS + 1]]
        expected = []
        worst_value = worst_total = worst_cost = 0.0
        values_seen = []
        for day_index in range(DAYS):
            day = FIRST_DAY + pd.Timedelta(days=day_index)
            values, prices = plan_row(ledger, day_index, column)
            row = 2 + day_index
            cell = ws.cell(row, 1)
            if pd.Timestamp(cell.value).normalize() != day:
                raise AssertionError(f"{name}/{sheet_name} row {row} date mismatch")
            sheet_values = [float(ws.cell(row, 2 + offset).value) for offset in range(SLOTS)]
            worst_value = max(worst_value, max(abs(a - b) for a, b in zip(sheet_values, values)))
            worst_total = max(worst_total, abs(float(ws.cell(row, SLOTS + 2).value)
                                               - float(sum(values))))
            worst_cost = max(worst_cost, abs(float(ws.cell(row, SLOTS + 3).value)
                                             - float(sum(v * p for v, p in zip(values, prices)))))
            values_seen.extend(sheet_values)
        checks["plan"][sheet_name] = dict(
            rows=DAYS, slots=SLOTS, cells=len(values_seen),
            time_labels=len(set(labels)), first_label=labels[0], last_label=labels[-1],
            max_abs_value_error_kWh=worst_value, max_abs_total_error_kWh=worst_total,
            max_abs_cost_error_yuan=worst_cost,
            total_purchase_kWh=float(sum(values_seen)),
            total_cost_yuan=float(sum(float(ws.cell(2 + i, SLOTS + 3).value)
                                      for i in range(DAYS))))
        if max(worst_value, worst_total) >= ENERGY_TOL or worst_cost >= CASH_TOL:
            raise AssertionError(f"{name}/{sheet_name} read-back exceeds tolerance: "
                                 f"{worst_value}, {worst_total}, {worst_cost}")
    ws = wb["充放电量"]
    worst_charge = worst_discharge = 0.0
    total_charge = total_discharge = 0.0
    for day_index in range(DAYS):
        day = FIRST_DAY + pd.Timedelta(days=day_index)
        block = ledger["by_day"][day].sort_values("interval_start")
        for index in range(6):
            row = 2 + day_index * 6 + index
            segment = block.iloc[index * 24:(index + 1) * 24]
            charge = float(ws.cell(row, 3).value)
            discharge = float(ws.cell(row, 4).value)
            worst_charge = max(worst_charge, abs(charge - float(segment.charge_kWh.sum())))
            worst_discharge = max(worst_discharge, abs(discharge - float(segment.discharge_kWh.sum())))
            total_charge += charge
            total_discharge += discharge
            if index == 0:
                if pd.Timestamp(ws.cell(row, 1).value).normalize() != day:
                    raise AssertionError(f"{name}/storage row {row} date mismatch")
                if ws.cell(row, 2).value != "00:00-04:00":
                    raise AssertionError(f"{name}/storage row {row} block label mismatch")
                if abs(float(ws.cell(row, 6).value) - float(block.state_start_kWh.iloc[0])) >= ENERGY_TOL:
                    raise AssertionError(f"{name}/storage row {row} initial state mismatch")
            if index == 1 and abs(float(ws.cell(row, 6).value)
                                  - float(block.state_end_kWh.iloc[-1])) >= ENERGY_TOL:
                raise AssertionError(f"{name}/storage row {row} final state mismatch")
    checks["storage"] = dict(rows=DAYS * 6, max_abs_charge_error_kWh=worst_charge,
                             max_abs_discharge_error_kWh=worst_discharge,
                             total_charge_kWh=total_charge, total_discharge_kWh=total_discharge,
                             initial_state_kWh=float(ws.cell(2, 6).value))
    if max(worst_charge, worst_discharge) >= ENERGY_TOL:
        raise AssertionError(f"{name}/storage read-back exceeds tolerance")

    ws = wb["紧急购电量"]
    expected_rows = emergency_ranges(ledger)
    seen = []
    for index, entry in enumerate(expected_rows, start=2):
        if index > ws.max_row:
            raise AssertionError(f"{name}/emergency is missing row {index}")
        amount = float(ws.cell(index, 3).value or 0.0)
        period = ws.cell(index, 2).value
        seen.append(amount)
        if abs(amount - entry["kWh"]) >= ENERGY_TOL or period != entry["period"]:
            raise AssertionError(f"{name}/emergency row {index} mismatch: {period} vs "
                                 f"{entry['period']}, {amount} vs {entry['kWh']}")
    total_emergency_kwh = float(sum(seen))
    ledger_total = float(ledger["natural"].emergency_kWh.sum())
    checks["emergency"] = dict(rows=len(expected_rows), total_kWh=total_emergency_kwh,
                               ledger_total_kWh=ledger_total,
                               abs_total_error_kWh=abs(total_emergency_kwh - ledger_total),
                               zero_rows=sum(1 for entry in expected_rows
                                             if entry["intervals"] == 0),
                               range_rows=sum(1 for entry in expected_rows
                                              if entry["intervals"] > 0))
    if checks["emergency"]["abs_total_error_kWh"] >= ENERGY_TOL:
        raise AssertionError(f"{name}/emergency total mismatch")
    checks["record"] = record
    return checks


def summary_of(ledger):
    natural = ledger["natural"]
    total = float(natural[list(COST_ITEMS)].sum(axis=1).sum())
    return dict(
        natural_segments=int(len(natural)),
        total_cost_yuan=total,
        ordinary_cost_yuan=float(natural.ordinary_cost_yuan.sum()),
        adjustment_cost_yuan=float(natural.adjustment_cost_yuan.sum()),
        emergency_cost_yuan=float(natural.emergency_cost_yuan.sum()),
        purchase_effective_kWh=float(natural.q_eff_kWh.sum()),
        purchase_original_kWh=float(natural.q0_kWh.sum()),
        charge_kWh=float(natural.charge_kWh.sum()),
        discharge_kWh=float(natural.discharge_kWh.sum()),
        emergency_kWh=float(natural.emergency_kWh.sum()),
        emergency_intervals=int((natural.emergency_kWh > ENERGY_TOL).sum()),
        unused_kWh=float(natural.unused_kWh.sum()),
        initial_state_kWh=float(natural.state_start_kWh.iloc[0]),
        final_state_kWh=float(natural.state_end_kWh.iloc[-1]),
        tail_state_end_kWh=float(ledger["tail"].state_end_kWh))


def main():
    assert Path(sys.prefix).name == "math_modeling", sys.prefix
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    BACKUP.mkdir(parents=True, exist_ok=True)
    record = dict(
        generated_utc=utc_now(),
        purpose="fill result4-2.xlsx (Q42) and result4-3.xlsx (Q43_S2) as copies and verify by "
                "re-reading, without re-optimising anything",
        formal_period="2025-02-01 00:00 .. 2025-12-31 23:50 (334 natural days, 48096 segments); "
                      "the January public initialisation is excluded from the formal cost",
        time_rule="the label 00:10 covers [00:10,00:20); the template plan row of day d holds the "
                  "144 published slots 00:10..next 00:10",
        candidate_choice=dict(result4_2="Q42", result4_3="Q43_S2",
                              note="Q43_S0 is kept as a comparison only and is NOT written into any "
                                   "template"),
        field_mapping={
            "计划购电量 (result4-2)": "dispatch.csv::q0_kWh of Q42 (no intraday revision, so the "
                                      "original plan is also the effective plan)",
            "计划购电量 (result4-3)": "dispatch.csv::q0_kWh of Q43_S2 (the original 0:00 plan)",
            "调整购电量 (result4-3)": "dispatch.csv::q_eff_kWh of Q43_S2 (the final effective plan)",
            "全天购电量": "sum of the 144 written slots of that row",
            "全天购电费": "sum over the 144 slots of price_yuan_kWh * the written quantity; this is "
                          "a pure purchase cost and does NOT include the 0.5*|q_eff-q0| adjustment "
                          "fee or any emergency cost, which are reported in the summary",
            "充放电量": "charge_kWh / discharge_kWh summed over each natural day's six 4-hour "
                        "blocks; the 储电量 column holds the day-start state in block 0 and the "
                        "day-end state in block 1 (same layout as the shipped result2.xlsx)",
            "紧急购电量": "contiguous positive-emergency ranges inside one natural day; every day "
                          "gets at least one row, clean days get 无紧急购电 with 0"},
        inputs={}, outputs={}, checks={}, anomalies=[], gaps=[])
    for name in ("result4-2", "result4-3"):
        strategy = ASSIGNMENT[name]["strategy"]
        ledger = read_ledger(strategy)
        record["inputs"][f"{strategy}_dispatch"] = dict(path=ledger["source"],
                                                        sha256=ledger["sha256"],
                                                        rows=int(len(ledger["frame"])))
        record["inputs"][f"{name}_template"] = dict(
            path=str(TEMPLATES[name].relative_to(ROOT)), sha256=digest(TEMPLATES[name]))
        before = digest(TEMPLATES[name])
        target, backup, page_record = build_copy(name, ledger)
        if digest(TEMPLATES[name]) != before:
            raise AssertionError(f"{name} template was modified; it must stay read-only")
        checks = read_back(name, ledger, page_record)
        record["outputs"][f"{name}_filled"] = dict(
            path=str(target.relative_to(ROOT)), sha256=digest(target),
            bytes=int(target.stat().st_size))
        record["outputs"][f"{name}_template_backup"] = dict(
            path=str(backup.relative_to(ROOT)), sha256=digest(backup))
        record["checks"][name] = dict(strategy=strategy, read_back=checks,
                                      ledger_summary=summary_of(ledger))
        print(f"{name} filled from {strategy}: plan rows "
              f"{sum(v['rows'] for k, v in checks['plan'].items())}, storage rows "
              f"{checks['storage']['rows']}, emergency rows {checks['emergency']['rows']}, "
              f"max value error {max(v['max_abs_value_error_kWh'] for v in checks['plan'].values()):.3e} kWh",
              flush=True)
    record["elapsed_seconds"] = time.perf_counter() - started
    record["status"] = "passed"
    record["note"] = ("only writing and re-reading happened here: no training, no protection "
                      "rebuild, no optimisation, no parameter sweep. The 附件5 originals were "
                      "backed up and left untouched; Q2 and Q3 artifacts were read only.")
    save_json(OUT / "fill_validation.json", record)
    print(json.dumps({"status": record["status"], "seconds": round(record["elapsed_seconds"], 2),
                      "outputs": sorted(record["outputs"]),
                      "gaps": record["gaps"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
