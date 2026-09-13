#!/usr/bin/env python
"""问题三 新增预报时刻必要性轻量分析 —— 只读诊断层。

任务书 ``reports/问题三/问题三_新增预报时刻必要性轻量分析方案.md``。回答三件事：

1. 实际应急电量与费用集中在哪些时钟区间，是否随“距最近一次可得预报发布的时间”变化；
2. 应急时段是否伴随光伏高估、负载低估、保护后净需求低估或储能放电受限；
3. 哪些风险时段值得优先获取新增光伏预报，哪些更应先关注负载预测、保护程度或储能安排。

职责边界
--------
**纯只读统计**：模型训练 0、MILP/LP/DP 求解 0、储能回放 0、公共 1 月重跑 0、
预测/偏差/q75/q80 重建 0、更新时刻消融 0、参数扫描 0。不导入任何有求解副作用的实验入口，
不合成新时刻预报，不用后续版本或真值冒充当时可得的预报。本脚本只写 CSV/JSON；
报告与图表由 ``code/q3_forecast_timing_report.py`` 只读渲染。

关键口径
--------
* 正式现金区间 48096 段（2025-02-01 00:00 .. 2025-12-31 23:50），末模板尾段
  （2026-01-01 00:00—00:10）排除，不把自然日与模板口径混用；
* 每个实际区间每策略只计一次费用与应急，不把四版本的预测记录展开后重复累计；
* 非午夜区间：``v_t = floor(小时/6)``、``h = 6*小时 + 分钟/10``、
  ``tau_t = (h - 6 v_t)/6`` 小时（距最近一次发布的时长）；用 ``pv_linear[k, v_t, h]``、
  ``issued_load[k, h]`` 与该组自己的 ``rho[k, v_t, h]``（L75 用 q75、L80 用 q80）；
* 334 个午夜继承段保留现金与应急在全期小时/月份统计内，但从“当日最新预报—当期计划”与
  ``tau`` 主诊断中排除，单列 ``midnight_carry``；主预测配对为 47762 段。

列名约定：所有分组表用 ``{指标}_{标签}``，标签 ``L75``/``L80`` 为策略口径，
``all``/``em``/``emsoc`` 分别为全样本、应急子集、应急子集伴随库存状态，避免相互覆盖。

运行::

    E:/Anaconda/envs/math_modeling/python.exe code/q3_forecast_timing_diagnostic.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ======================================================================================
# §0  冻结规格
# ======================================================================================
ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code/q3_forecast_timing_diagnostic.py"
OUT = ROOT / "results/q3_forecast_timing_diagnostic"
FIG = ROOT / "figures/q3_forecast_timing_diagnostic"
REPORT_MD = ROOT / "reports/问题三/问题三_新增预报时刻必要性分析结果.md"
PLAN_MD = ROOT / "reports/问题三/问题三_新增预报时刻必要性轻量分析方案.md"
REVIEW_MD = ROOT / "reports/问题三/问题三_分位费用实验接收核查.md"
COST_DIR = ROOT / "results/q3_bias_quantile_cost"
L75_DISPATCH = COST_DIR / "L75/dispatch.csv"
L80_DISPATCH = ROOT / "results/q3_rolling_baseline/B2_dispatch.csv"
PROTECTION = COST_DIR / "protection_q75.npz"
BIAS_NPZ = ROOT / "results/q3_bias_correction_diagnostic/bias_forecast_archive.npz"
COST_SUMMARY = COST_DIR / "summary.csv"
COST_MANIFEST = COST_DIR / "run_manifest.json"

BASE = pd.Timestamp("2025-01-01")
DT = 1.0 / 6.0
CAP_KWH = 5000.0 / 6.0            # M: max charge/discharge energy per 10-minute slot
STATE_MIN_KWH = 1200.0
ETA = 0.9
UPDATE_HOURS = (0, 6, 12, 18)
ENERGY_TOL = 1e-8                 # energy/power alignment tolerance, own units, rtol = 0
CASH_TOL = 1e-4                   # cash aggregation tolerance, yuan
EMERGENCY_TOL = 1e-6              # kWh
NUM_INTERVALS = 48096             # natural cash segments
NUM_MIDNIGHT = 334                # midnight carry segments inside that range
NUM_PAIRED = 47762                # non-midnight forecast pairings
AGE_BINS = 6                      # tau in [0,1) .. [5,6) hours
RISK_HOURS = 3
TOP_DATES = 3
# Descriptive thresholds for the candidate table. They were introduced in this round, are NOT
# pre-registered, and are used only to structure the table: no hour is excluded as a possible
# new publication time by them (see 问题三_新增预报时刻分析接收核查.md finding 1).
PV_OVER_SHARE_MIN = 0.5
MONTHS_WITH_EMERGENCY_MIN = 6
STRATEGIES = ("L75", "L80")
RHO_KEY = {"L75": "linear_rho_q75", "L80": "linear_rho_q80"}

NON_COMPUTED_ARTIFACTS = frozenset({"run_manifest.json", "figure_integrity.json", "run.log"})

PARAMETERS = dict(
    experiment="Q3 forecast-timing necessity diagnostic (read-only)",
    specification="reports/问题三/问题三_新增预报时刻必要性轻量分析方案.md",
    subject="以 L75 为主要观察对象、L80 为对照，判断是否需要引入其他时刻的光伏预报",
    time_version="start_time_v1",
    range="2025-02-01 00:00 .. 2025-12-31 23:50, 48096 natural cash segments; the "
          "2026-01-01 00:00-00:10 template tail is excluded",
    clock_rule="non-midnight: v_t = floor(hour/6), h = 6*hour + minute/10, "
               "tau_t = (h - 6*v_t)/6 hours",
    midnight_rule="the 334 midnight carry segments keep their cash and emergency in the hourly and "
                  "monthly totals but are excluded from the latest-forecast pairing and tau "
                  "diagnostic; main pairing is 47762 segments",
    forecast_source="pv_linear[k, v_t, h] and issued_load[k, h] from the frozen archives; the "
                    "group's own rho (q75 for L75, q80 for L80). The latest published forecast is "
                    "used even when it did not lead to an accepted plan revision",
    decomposition="a_L = (L - Lhat)*dt, a_V = (Vhat - V)*dt, eps = a_L + a_V = n - nhat, "
                  "g = eps - rho; positive a_L = load under-forecast, positive a_V = PV "
                  "over-forecast, positive g = the protected curve still under-estimates",
    storage="Dmax_t = min(M, 0.9*(E_t - 1200)), A_t = (n_t - q_eff_t)^+, e_t = (A_t - Dmax_t)^+ "
            "under the existing greedy feedback; the classes inventory_limited / power_limited / "
            "close are companion state, not isolated causes",
    candidate_rule="top-3 risk hours by L75 hourly emergency cost; the candidate publication clock "
                   "(H-1) mod 24 is an unverified engineering convention for lead time, not an "
                   "estimated optimum; at most two new windows, none if the evidence does not "
                   "support them",
    descriptive_thresholds=dict(pv_over_share_min=PV_OVER_SHARE_MIN,
                                months_with_emergency_min=MONTHS_WITH_EMERGENCY_MIN,
                                note="introduced in this round, NOT pre-registered; used only to "
                                     "structure the candidate table. Every hour that satisfies them "
                                     "is reported even when its cost rank is below the top 3, and no "
                                     "publication time is excluded by them. No significance test and "
                                     "no optimality claim."),
    solver="none: no training, no MILP/LP/DP, no storage replay, no prediction or quantile rebuild",
    validation="three light checks only: sample/time mapping, one date and six boundaries, "
               "aggregation identities",
)


# ======================================================================================
# §1  基础设施
# ======================================================================================
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
    if isinstance(obj, (Path, pd.Timestamp, datetime, timedelta)):
        return str(obj)
    return obj


def save_json(path, obj):
    Path(path).write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=2, default=str),
                          encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def frame_to_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def known_max_abs(values, label):
    """``max |value|`` that treats any NaN/inf as a failure instead of silently skipping it."""
    arr = np.asarray(values, dtype=float)
    bad = ~np.isfinite(arr)
    if bad.any():
        raise AssertionError(f"{label}: {int(bad.sum())} non-finite value(s) of {arr.size}")
    return float(np.abs(arr).max())


def nonfinite_count(frame, columns, label):
    values = frame[list(columns)].to_numpy(dtype=float)
    bad = int((~np.isfinite(values)).sum())
    if bad:
        raise AssertionError(f"{label}: {bad} non-finite cell(s) out of {values.size}")
    return bad


# ======================================================================================
# §2  输入与区间表
# ======================================================================================
def read_inputs():
    with np.load(PROTECTION) as archive:
        rho = {sid: archive[key] for sid, key in RHO_KEY.items()}
    with np.load(BIAS_NPZ) as archive:
        truth_load = archive["truth_load"]
        truth_pv = archive["truth_pv"]
        issued_load = archive["issued_load"]
        pv_linear = archive["pv_linear"]
    summary = pd.read_csv(COST_SUMMARY)
    dispatch = {}
    for sid, path in (("L75", L75_DISPATCH), ("L80", L80_DISPATCH)):
        frame = pd.read_csv(path, parse_dates=["interval_start", "interval_end"])
        frame = frame.rename(columns={"group": "strategy_id"})
        assert len(frame) == NUM_INTERVALS + 1, (sid, len(frame))
        dispatch[sid] = frame
    return dict(rho=rho, truth_load=truth_load, truth_pv=truth_pv, issued_load=issued_load,
                pv_linear=pv_linear, summary=summary, dispatch=dispatch)


def build_intervals(sid, dispatch, data):
    """One row per real 10-minute interval for one strategy (48096 natural rows + tail)."""
    frame = dispatch.sort_values("interval_start").reset_index(drop=True)
    tail = frame.iloc[-1]
    natural = frame.iloc[:-1].copy()
    assert len(natural) == NUM_INTERVALS, len(natural)

    starts = natural.interval_start
    assert starts.is_monotonic_increasing and not starts.duplicated().any()
    assert starts.diff().dropna().eq(pd.Timedelta(minutes=10)).all()
    hour = starts.dt.hour.to_numpy()
    minute = starts.dt.minute.to_numpy()
    is_midnight = (hour == 0) & (minute == 0)
    target_h = 6 * hour + minute // 10
    version = np.where(is_midnight, -1, hour // 6)
    # publication slot is 36*v (= 6 * the publication clock hour); tau is (h - 36 v)/6 hours
    age = np.where(is_midnight, np.nan, (target_h - 36 * np.maximum(version, 0)) / 6.0)
    natural = natural.assign(
        k=(starts.dt.normalize() - BASE).dt.days.to_numpy(),
        clock_hour=hour, clock_minute=minute, target_h=target_h,
        is_midnight=is_midnight, version=version,
        publication_hour=np.where(is_midnight, -1, 6 * (hour // 6)),
        age_hours=age,
        age_bin=[f"[{int(a)},{int(a) + 1})" if np.isfinite(a) else "" for a in age])

    k = natural.k.to_numpy()
    h = natural.target_h.to_numpy()
    idx = np.where(~is_midnight)[0]
    f_load = np.full(len(natural), np.nan)
    f_pv = np.full(len(natural), np.nan)
    rho_col = np.full(len(natural), np.nan)
    f_load[idx] = data["issued_load"][k[idx], h[idx]]
    f_pv[idx] = data["pv_linear"][k[idx], version[idx], h[idx]]
    rho_col[idx] = data["rho"][sid][k[idx], version[idx], h[idx]]
    natural["forecast_load_kW"] = f_load
    natural["forecast_pv_kW"] = f_pv
    natural["rho_kWh"] = rho_col
    natural["net_forecast_kWh"] = (f_load - f_pv) * DT

    actual_load = natural.actual_load_kW.to_numpy(dtype=float)
    actual_pv = natural.actual_pv_kW.to_numpy(dtype=float)
    net = natural.net_kWh.to_numpy(dtype=float)
    natural["a_load_kWh"] = (actual_load - f_load) * DT
    natural["a_pv_kWh"] = (f_pv - actual_pv) * DT
    natural["eps_kWh"] = net - natural.net_forecast_kWh.to_numpy()
    natural["g_kWh"] = natural.eps_kWh.to_numpy() - rho_col
    natural["pv_prominent"] = ((natural.a_pv_kWh > 0)
                               & (natural.a_pv_kWh > natural.a_load_kWh.clip(lower=0.0)))
    natural["truth_recompute_residual_kWh"] = net - (actual_load - actual_pv) * DT

    state = natural.state_start_kWh.to_numpy(dtype=float)
    q_eff = natural.q_eff_kWh.to_numpy(dtype=float)
    cap = ETA * (state - STATE_MIN_KWH)
    need = np.clip(net - q_eff, 0.0, None)
    emergency = natural.emergency_kWh.to_numpy(dtype=float)
    natural["discharge_cap_kWh"] = cap
    natural["need_before_battery_kWh"] = need
    natural["expected_emergency_kWh"] = np.clip(need - np.minimum(CAP_KWH, cap), 0.0, None)
    natural["discharge_limit_class"] = np.where(
        emergency <= EMERGENCY_TOL, "not_emergency",
        np.where(cap < CAP_KWH - EMERGENCY_TOL, "inventory_limited",
                 np.where(cap > CAP_KWH + EMERGENCY_TOL, "power_limited", "close")))
    natural["emergency_cost_yuan"] = natural.emergency_cost_yuan.astype(float)
    natural["strategy_id"] = sid
    return natural.reset_index(drop=True), tail


# ======================================================================================
# §3  分组统计积木（列名一律 {指标}_{标签}，避免相互覆盖）
# ======================================================================================
def rate_block(frame, tag):
    """Emergency energy, cost, occurrence rate and share denominators for one subset."""
    total = len(frame)
    if total == 0:
        return {f"{field}_{tag}": None for field in
                ("n_intervals", "n_emergency", "days_with_emergency", "emergency_kWh",
                 "emergency_cost_yuan", "emergency_rate", "mean_kWh_per_interval")}
    flag = frame.emergency_kWh.to_numpy() > EMERGENCY_TOL
    energy = float(frame.emergency_kWh.sum())
    return {
        f"n_intervals_{tag}": int(total),
        f"n_emergency_{tag}": int(flag.sum()),
        f"days_with_emergency_{tag}": int(
            frame.loc[flag, "interval_start"].dt.strftime("%Y-%m-%d").nunique()),
        f"emergency_kWh_{tag}": energy,
        f"emergency_cost_yuan_{tag}": float(frame.emergency_cost_yuan.sum()),
        f"emergency_rate_{tag}": float(flag.mean()),
        f"mean_kWh_per_interval_{tag}": energy / total}


def error_block(frame, tag):
    """Error direction and magnitude over the paired (non-midnight) rows of a subset."""
    block = frame[~frame.is_midnight]
    fields = ("n_paired", "pv_mae_kW", "net_mae_kWh", "mean_a_load_kWh", "mean_a_pv_kWh",
              "mean_eps_kWh", "mean_g_kWh", "mean_pos_a_load_kWh", "mean_pos_a_pv_kWh",
              "share_load_under", "share_pv_over", "share_g_positive", "pv_prominent_share")
    if len(block) == 0:
        return {f"{field}_{tag}": None for field in fields}
    a_load = block.a_load_kWh.to_numpy()
    a_pv = block.a_pv_kWh.to_numpy()
    g = block.g_kWh.to_numpy()
    return {
        f"n_paired_{tag}": int(len(block)),
        f"pv_mae_kW_{tag}": float(np.abs(block.forecast_pv_kW - block.actual_pv_kW).mean()),
        f"net_mae_kWh_{tag}": float(np.abs(block.eps_kWh).mean()),
        f"mean_a_load_kWh_{tag}": float(a_load.mean()),
        f"mean_a_pv_kWh_{tag}": float(a_pv.mean()),
        f"mean_eps_kWh_{tag}": float(block.eps_kWh.mean()),
        f"mean_g_kWh_{tag}": float(g.mean()),
        f"mean_pos_a_load_kWh_{tag}": float(np.clip(a_load, 0.0, None).mean()),
        f"mean_pos_a_pv_kWh_{tag}": float(np.clip(a_pv, 0.0, None).mean()),
        f"share_load_under_{tag}": float((a_load > 0).mean()),
        f"share_pv_over_{tag}": float((a_pv > 0).mean()),
        f"share_g_positive_{tag}": float((g > 0).mean()),
        f"pv_prominent_share_{tag}": float(block.pv_prominent.mean())}


def soc_block(frame, tag):
    """Companion storage-state evidence on the emergency rows of a subset.

    Both the interval-start and interval-end internal storage medians are reported, because the
    limit class is defined from the START state; a median at the floor does not mean every
    interval starts at the floor.
    """
    block = frame[frame.emergency_kWh.to_numpy() > EMERGENCY_TOL]
    fields = ("n_em", "mean_state_start_kWh", "median_state_start_kWh", "median_state_end_kWh",
              "mean_state_end_kWh", "intervals_start_at_floor", "intervals_start_above_floor",
              "inventory_limited_share", "power_limited_share", "close_share",
              "mean_need_kWh", "mean_cap_kWh")
    if len(block) == 0:
        return {f"{field}_{tag}": None for field in fields}
    classes = block.discharge_limit_class
    return {
        f"n_em_{tag}": int(len(block)),
        f"mean_state_start_kWh_{tag}": float(block.state_start_kWh.mean()),
        f"median_state_start_kWh_{tag}": float(block.state_start_kWh.median()),
        f"median_state_end_kWh_{tag}": float(block.state_end_kWh.median()),
        f"mean_state_end_kWh_{tag}": float(block.state_end_kWh.mean()),
        f"intervals_start_at_floor_{tag}": int(
            (block.state_start_kWh <= STATE_MIN_KWH + EMERGENCY_TOL).sum()),
        f"intervals_start_above_floor_{tag}": int(
            (block.state_start_kWh > STATE_MIN_KWH + EMERGENCY_TOL).sum()),
        f"inventory_limited_share_{tag}": float((classes == "inventory_limited").mean()),
        f"power_limited_share_{tag}": float((classes == "power_limited").mean()),
        f"close_share_{tag}": float((classes == "close").mean()),
        f"mean_need_kWh_{tag}": float(block.need_before_battery_kWh.mean()),
        f"mean_cap_kWh_{tag}": float(block.discharge_cap_kWh.mean())}


def subset_block(frame_l75, frame_l80, tag_l75, tag_l80):
    """Cash/emergency block for both strategies over the same real intervals."""
    out = rate_block(frame_l75, tag_l75)
    out.update(rate_block(frame_l80, tag_l80))
    out[f"d_emergency_kWh_{tag_l75}_minus_{tag_l80}"] = (
        (out[f"emergency_kWh_{tag_l75}"] or 0.0) - (out[f"emergency_kWh_{tag_l80}"] or 0.0))
    out[f"d_emergency_cost_yuan_{tag_l75}_minus_{tag_l80}"] = (
        (out[f"emergency_cost_yuan_{tag_l75}"] or 0.0)
        - (out[f"emergency_cost_yuan_{tag_l80}"] or 0.0))
    return out


def with_shares(entry, totals):
    for sid in STRATEGIES:
        entry[f"energy_share_pct_{sid}"] = (
            100.0 * (entry[f"emergency_kWh_{sid}"] or 0.0) / totals[sid]["emergency_kWh"])
        entry[f"cost_share_pct_{sid}"] = (
            100.0 * (entry[f"emergency_cost_yuan_{sid}"] or 0.0)
            / totals[sid]["emergency_cost_yuan"])
    return entry


def pv_presence_block(frame):
    """Measured-PV presence for one clock hour: how much the hour is really dark."""
    positive = frame.actual_pv_kW.to_numpy() > 0.0
    return dict(pv_positive_intervals=int(positive.sum()),
                max_actual_pv_kW=float(frame.actual_pv_kW.max()),
                actual_pv_energy_kWh=float(frame.actual_pv_kW.sum() * DT))


def build_hourly(frames, totals):
    """24 clock hours: cash from all 48096 rows, error evidence from the 47762 pairings."""
    rows = []
    for hour in range(24):
        block = frames["L75"][frames["L75"].clock_hour == hour]
        block_l80 = frames["L80"][frames["L80"].clock_hour == hour]
        entry = dict(clock_hour=hour, n_intervals=int(len(block)),
                     n_midnight=int(block.is_midnight.sum()))
        entry.update(pv_presence_block(block))
        entry.update(subset_block(block, block_l80, "L75", "L80"))
        with_shares(entry, totals)
        entry.update(error_block(block, "all"))
        entry.update(error_block(block[block.emergency_kWh > EMERGENCY_TOL], "em"))
        entry.update(soc_block(block, "emsoc"))
        rows.append(entry)
    return pd.DataFrame(rows)


def build_monthly_hourly(frames):
    l75 = frames["L75"].assign(month=frames["L75"].interval_start.dt.strftime("%Y-%m"))
    l80 = frames["L80"].assign(month=frames["L80"].interval_start.dt.strftime("%Y-%m"))
    rows = []
    for (month, hour), block in l75.groupby(["month", "clock_hour"], sort=True):
        block_l80 = l80[(l80.month == month) & (l80.clock_hour == hour)]
        entry = dict(month=month, clock_hour=int(hour), n_intervals=int(len(block)),
                     n_midnight=int(block.is_midnight.sum()))
        entry.update(subset_block(block, block_l80, "L75", "L80"))
        entry.update(error_block(block, "all"))
        entry.update(error_block(block[block.emergency_kWh > EMERGENCY_TOL], "em"))
        rows.append(entry)
    return pd.DataFrame(rows)


def build_age_summary(frames, totals, midnight_rows):
    """Per publication node x age bin, plus the merged-across-nodes view and midnight carry.

    With a 6-hourly publication grid, (node, age) is a bijection with the clock hour, so the merged
    age view averages four specific clock hours and cannot separate age from hour-of-day.
    """
    rows = []
    for sid in STRATEGIES:
        other = "L80" if sid == "L75" else "L75"
        frame = frames[sid]
        for node in UPDATE_HOURS:
            for index in range(AGE_BINS):
                label = f"[{index},{index + 1})"
                block = frame[(frame.version == node // 6) & (frame.age_bin == label)]
                block_other = frames[other][(frames[other].version == node // 6)
                                            & (frames[other].age_bin == label)]
                entry = dict(group_type="node_age", strategy_id=sid, publication_hour=node,
                             version=node // 6, age_bin=label, age_lower_h=index,
                             age_upper_h=index + 1,
                             clock_hours=str(6 * (node // 6) + index))
                entry.update(rate_block(block, "self"))
                entry.update(rate_block(block_other, other))
                entry[f"d_emergency_kWh_self_minus_{other}"] = (
                    (entry["emergency_kWh_self"] or 0.0) - (entry[f"emergency_kWh_{other}"] or 0.0))
                entry.update(error_block(block, "all"))
                entry.update(error_block(block[block.emergency_kWh > EMERGENCY_TOL], "em"))
                entry.update(soc_block(block, "emsoc"))
                entry["energy_share_pct"] = (100.0 * entry["emergency_kWh_self"]
                                            / totals[sid]["emergency_kWh"])
                entry["cost_share_pct"] = (100.0 * entry["emergency_cost_yuan_self"]
                                          / totals[sid]["emergency_cost_yuan"])
                rows.append(entry)
        for index in range(AGE_BINS):
            label = f"[{index},{index + 1})"
            block = frame[frame.age_bin == label]
            block_other = frames[other][frames[other].age_bin == label]
            entry = dict(group_type="merged_age", strategy_id=sid, publication_hour=-1, version=-1,
                         age_bin=label, age_lower_h=index, age_upper_h=index + 1,
                         clock_hours=",".join(str(h) for h in range(index, 24, 6)))
            entry.update(rate_block(block, "self"))
            entry.update(rate_block(block_other, other))
            entry[f"d_emergency_kWh_self_minus_{other}"] = (
                (entry["emergency_kWh_self"] or 0.0) - (entry[f"emergency_kWh_{other}"] or 0.0))
            entry.update(error_block(block, "all"))
            entry.update(error_block(block[block.emergency_kWh > EMERGENCY_TOL], "em"))
            entry.update(soc_block(block, "emsoc"))
            entry["energy_share_pct"] = (100.0 * entry["emergency_kWh_self"]
                                        / totals[sid]["emergency_kWh"])
            entry["cost_share_pct"] = (100.0 * entry["emergency_cost_yuan_self"]
                                      / totals[sid]["emergency_cost_yuan"])
            rows.append(entry)
        block = midnight_rows[sid]
        entry = dict(group_type="midnight_carry", strategy_id=sid, publication_hour=-1, version=-1,
                     age_bin="", age_lower_h=np.nan, age_upper_h=np.nan, clock_hours="0")
        entry.update(rate_block(block, "self"))
        entry[f"d_emergency_kWh_self_minus_{other}"] = np.nan
        entry.update(soc_block(block, "emsoc"))
        entry["energy_share_pct"] = 100.0 * entry["emergency_kWh_self"] / totals[sid]["emergency_kWh"]
        entry["cost_share_pct"] = (100.0 * entry["emergency_cost_yuan_self"]
                                  / totals[sid]["emergency_cost_yuan"])
        rows.append(entry)
    return pd.DataFrame(rows)


# ======================================================================================
# §4  风险时窗与候选时窗
# ======================================================================================
def rule_satisfying_hours(frames):
    """Hours whose evidence satisfies the descriptive rule, whatever their cost rank.

    Finding 1 of the reception review: the top-3-by-cost cutoff must not be read as excluding
    other publication times, so every hour that satisfies the rule is reported alongside the
    top 3 and no hour is excluded. Uses the same emergency-subset quantities as ``error_block``
    (photo-voltaic over-forecast share, mean protected gap, months with an emergency event).
    """
    out = []
    for hour, block in frames["L75"].groupby("clock_hour"):
        emergency = block[block.emergency_kWh > EMERGENCY_TOL]
        if len(emergency) == 0:
            continue
        pv_over = float((emergency.a_pv_kWh.to_numpy() > 0).mean())
        mean_g = float(emergency.g_kWh.mean())
        months = int(emergency.interval_start.dt.strftime("%Y-%m").nunique())
        if pv_over >= PV_OVER_SHARE_MIN and mean_g > 0 and months >= MONTHS_WITH_EMERGENCY_MIN:
            out.append(int(hour))
    return sorted(out)


def build_risk_windows(frames, hourly, midnight_rows):
    """Top 3 risk hours by L75 cost, plus every remaining hour that satisfies the rule."""
    order = hourly.sort_values("emergency_cost_yuan_L75", ascending=False).reset_index(drop=True)
    top3 = [int(h) for h in order.clock_hour.head(RISK_HOURS)]
    month_labels = sorted(frames["L75"].interval_start.dt.strftime("%Y-%m").unique())
    selected = [(h, "top3_by_cost") for h in top3]
    for hour in rule_satisfying_hours(frames):
        if hour not in top3:
            selected.append((hour, "rule_satisfying_beyond_top3"))
    rows = []
    for rank, (hour, selection) in enumerate(selected, start=1):
        block = frames["L75"][frames["L75"].clock_hour == hour]
        block_l80 = frames["L80"][frames["L80"].clock_hour == hour]
        emergency = block[block.emergency_kWh > EMERGENCY_TOL].copy()
        candidate_clock = (hour - 1) % 24
        by_date = (emergency.assign(date=emergency.interval_start.dt.strftime("%Y-%m-%d"))
                   .groupby("date", as_index=False).emergency_cost_yuan.sum()
                   .sort_values("emergency_cost_yuan", ascending=False))
        top = by_date.head(TOP_DATES)
        total_cost = float(block.emergency_cost_yuan.sum())
        midnight = midnight_rows["L75"]
        midnight_hour = midnight[midnight.clock_hour == hour]
        entry = dict(
            rank=rank, selection=selection, clock_hour=hour,
            cost_rank=int(order.index[order.clock_hour == hour][0]) + 1,
            candidate_publication_clock=candidate_clock,
            candidate_is_existing_node=bool(candidate_clock in UPDATE_HOURS),
            candidate_note=("距风险窗起点提前 1 小时的工程约定，未估计最优提前量"
                            if candidate_clock not in UPDATE_HOURS else "该时刻已是现有发布节点"),
            n_intervals=int(len(block)), n_midnight=int(block.is_midnight.sum()),
            days_with_emergency_L75=int(by_date.shape[0]),
            days_with_emergency_L80=int(
                block_l80.loc[block_l80.emergency_kWh > EMERGENCY_TOL, "interval_start"]
                .dt.strftime("%Y-%m-%d").nunique()),
            months_with_emergency_L75=int(emergency.interval_start.dt.strftime("%Y-%m").nunique()),
            top_dates=";".join(f"{d}({c:,.0f})" for d, c in zip(top.date, top.emergency_cost_yuan)),
            top_dates_cost_sum=float(top.emergency_cost_yuan.sum()),
            top_dates_share_pct=(100.0 * float(top.emergency_cost_yuan.sum()) / total_cost
                                 if total_cost > 0 else None),
            midnight_emergency_kWh_L75=float(midnight_hour.emergency_kWh.sum()),
            midnight_emergency_cost_L75=float(midnight_hour.emergency_cost_yuan.sum()))
        entry.update(pv_presence_block(block))
        entry.update(subset_block(block, block_l80, "L75", "L80"))
        entry.update(error_block(block, "all"))
        entry.update(error_block(emergency, "em"))
        entry.update(soc_block(block, "emsoc"))
        for label in month_labels:
            subset = block[block.interval_start.dt.strftime("%Y-%m") == label]
            entry[f"em_cost_{label}"] = float(subset.emergency_cost_yuan.sum())
        rows.append(entry)
    return top3, pd.DataFrame(rows)


def build_candidates(risk):
    """Descriptive verdicts. No hour is excluded as a possible new publication time.

    Finding 1 of the reception review: the previous wording turned a table-structuring rule into a
    rejection of research directions. The verdicts now only describe which companion evidence the
    hour shows; whether a new publication time would help is left unverified for every hour.
    """
    rows = []
    for row in risk.itertuples():
        existing = bool(row.candidate_is_existing_node)
        pv_over = row.share_pv_over_em
        mean_g = row.mean_g_kWh_em
        months = int(row.months_with_emergency_L75)
        has_signature = (pv_over is not None and np.isfinite(pv_over)
                         and pv_over >= PV_OVER_SHARE_MIN and mean_g is not None
                         and np.isfinite(mean_g) and mean_g > 0)
        if existing:
            verdict = "existing_node_review"
            reason = ("候选发布时刻已是现有 0/6/12/18 点节点：应先讨论该节点的信息质量与计划安排，"
                      "不必移动时刻制造新增节点")
        elif has_signature and months >= MONTHS_WITH_EMERGENCY_MIN:
            verdict = "candidate_direction_for_verification"
            reason = (f"应急子集光伏高估占比 {pv_over:.1%}、保护后净需求平均仍低估 {mean_g:.3f} kWh、"
                      f"且 {months} 个月重复出现：可列为**待验证的数据补充方向**，"
                      "但其必要性与节费效果本轮未验证")
        elif has_signature:
            verdict = "direction_with_limited_month_coverage"
            reason = (f"应急子集光伏高估占比 {pv_over:.1%}、保护后净需求平均仍低估 {mean_g:.3f} kWh，"
                      f"但只出现在 {months} 个月、且费用集中于少数日期："
                      "存在跨月重复但月份覆盖有限，仍属待验证方向")
        else:
            verdict = "companion_evidence_not_pv"
            reason = (f"该小时当期光伏误差很小（应急子集光伏高估占比 "
                      f"{'—' if pv_over is None or not np.isfinite(pv_over) else f'{pv_over:.1%}'}）："
                      "负载低估、保护后低估与储能状态是明显的伴随现象；"
                      "本轮没有隔离因果贡献，也不排除改善预测后有助益")
        rows.append(dict(
            rank=int(row.rank), selection=row.selection,
            risk_clock_hour=int(row.clock_hour), cost_rank=int(row.cost_rank),
            risk_emergency_cost_yuan_L75=row.emergency_cost_yuan_L75,
            risk_emergency_kWh_L75=row.emergency_kWh_L75,
            candidate_publication_clock=int(row.candidate_publication_clock),
            candidate_is_existing_node=existing, lead_hours=1,
            months_with_emergency=months,
            emergency_subset_pv_over_share=pv_over,
            emergency_subset_mean_g_kWh=mean_g,
            emergency_subset_power_limited_share=row.power_limited_share_emsoc,
            emergency_subset_inventory_limited_share=row.inventory_limited_share_emsoc,
            excludes_new_publication_time=False,
            verdict=verdict, reason=reason,
            verification_status="未验证：提前 1 小时是工程约定，本轮无该时刻真实预报与匹配调度",
            limits=("不能把该风险窗应急费当作新增预报的节费上界：提前调整还会改变普通费、调整费、"
                    "库存与其他时段费用；未做更新时刻消融或参数扫描")))
    return pd.DataFrame(rows)


# ======================================================================================
# §5  三类精简检查
# ======================================================================================
def check_samples(frames, tails):
    out = {}
    forecast_cols = ["forecast_load_kW", "forecast_pv_kW", "rho_kWh", "net_forecast_kWh"]
    for sid in STRATEGIES:
        frame = frames[sid]
        paired = frame[~frame.is_midnight]
        midnight = frame[frame.is_midnight]
        assert len(frame) == NUM_INTERVALS and len(paired) == NUM_PAIRED
        assert len(midnight) == NUM_MIDNIGHT, len(midnight)
        assert int(frame.clock_hour.min()) == 0 and int(frame.clock_hour.max()) == 23
        assert str(tails[sid].interval_start) == "2026-01-01 00:00:00"
        out[sid] = dict(
            natural_intervals=int(len(frame)), paired_non_midnight=int(len(paired)),
            midnight_carry=int(len(midnight)),
            template_tail_excluded=str(tails[sid].interval_start),
            paired_forecast_finite_cells=int(
                np.isfinite(paired[forecast_cols].to_numpy(dtype=float)).sum()),
            paired_forecast_expected_cells=int(4 * NUM_PAIRED),
            midnight_forecast_null_cells=int(
                midnight[["forecast_load_kW", "forecast_pv_kW", "rho_kWh"]].isna().to_numpy().sum()),
            midnight_forecast_expected_null=int(3 * NUM_MIDNIGHT))
        assert out[sid]["paired_forecast_finite_cells"] == out[sid]["paired_forecast_expected_cells"]
        assert out[sid]["midnight_forecast_null_cells"] == out[sid]["midnight_forecast_expected_null"]
        assert int((paired.age_hours >= 0).sum()) == NUM_PAIRED
        assert int(((paired.age_hours >= 0) & (paired.age_hours < 6)).sum()) == NUM_PAIRED
        assert (paired.version >= 0).all() and (midnight.version == -1).all()
        nonfinite_count(frame, ["price_yuan_kWh", "actual_load_kW", "actual_pv_kW", "net_kWh",
                                "emergency_kWh", "emergency_cost_yuan", "q_eff_kWh",
                                "state_start_kWh"], f"{sid} ledger columns")
    a, b = frames["L75"], frames["L80"]
    assert a.interval_start.equals(b.interval_start)
    for column in ("net_kWh", "price_yuan_kWh", "actual_load_kW", "actual_pv_kW"):
        value = known_max_abs(a[column].to_numpy() - b[column].to_numpy(),
                              f"{column} differs between the two ledgers")
        out[f"ledger_delta_{column}_max_abs"] = value
        if value > ENERGY_TOL:
            raise AssertionError(f"{column} differs by {value:.3e}")
    out["ledger_alignment_tolerance"] = dict(atol=ENERGY_TOL, rtol=0)
    out["scope"] = ("L75/L80 natural ledgers: 48096 natural segments, 47762 non-midnight pairings, "
                    "334 midnight carries; the template tail is excluded")
    return out


def check_mapping(frames, data):
    day = "2025-06-21"
    k = (pd.Timestamp(day) - BASE).days
    probes = [("05:50", 0), ("06:00", 1), ("11:50", 1), ("12:00", 2), ("17:50", 2), ("18:00", 3)]
    frame = frames["L75"]
    rows = []
    for clock, expected_version in probes:
        stamp = pd.Timestamp(f"{day} {clock}")
        match = frame[frame.interval_start == stamp]
        assert len(match) == 1, (clock, len(match))
        row = match.iloc[0]
        target_h = 6 * stamp.hour + stamp.minute // 10
        publication = pd.Timestamp(f"{day} 00:00") + pd.Timedelta(hours=6 * expected_version)
        assert int(row.version) == expected_version, (clock, int(row.version))
        assert int(row.target_h) == target_h, (clock, int(row.target_h))
        assert publication <= stamp, (clock, publication)
        expected_pv = float(data["pv_linear"][k, expected_version, target_h])
        expected_load = float(data["issued_load"][k, target_h])
        assert abs(float(row.forecast_pv_kW) - expected_pv) <= ENERGY_TOL
        assert abs(float(row.forecast_load_kW) - expected_load) <= ENERGY_TOL
        assert abs(float(row.age_hours) - (target_h - 36 * expected_version) / 6.0) <= ENERGY_TOL
        rows.append(dict(day=day, clock=clock, target_h=int(target_h), version=int(row.version),
                         publication_clock=f"{6 * expected_version:02d}:00",
                         publication_not_after_interval=bool(publication <= stamp),
                         age_hours=float(row.age_hours), age_bin=row.age_bin,
                         forecast_pv_kW=float(row.forecast_pv_kW), archive_pv_kW=expected_pv,
                         forecast_load_kW=float(row.forecast_load_kW),
                         archive_load_kW=expected_load, expected_version=expected_version))
    midnight = frame[frame.interval_start == pd.Timestamp(f"{day} 00:00")]
    assert len(midnight) == 1 and bool(midnight.iloc[0].is_midnight)
    assert not np.isfinite(float(midnight.iloc[0].forecast_pv_kW))
    return dict(probes=rows, tolerance=dict(atol=ENERGY_TOL, rtol=0),
                midnight_isolated=dict(
                    interval_start=f"{day} 00:00:00", is_midnight=True,
                    forecast_columns_null=bool(
                        midnight[["forecast_load_kW", "forecast_pv_kW", "rho_kWh"]].isna()
                        .all(axis=None)),
                    note="午夜继承段保留现金与应急，但从最新预报配对与 tau 诊断中排除"))


def check_aggregates(frames, hourly, age, totals, midnight_rows, summary):
    out = {}
    reference = summary.set_index("strategy_id")
    for sid in STRATEGIES:
        energy = float(hourly[f"emergency_kWh_{sid}"].sum())
        cost = float(hourly[f"emergency_cost_yuan_{sid}"].sum())
        out[f"{sid}_hourly_energy_delta_vs_total_kWh"] = abs(energy - totals[sid]["emergency_kWh"])
        out[f"{sid}_hourly_cost_delta_vs_total_yuan"] = abs(cost - totals[sid]["emergency_cost_yuan"])
        if max(out[f"{sid}_hourly_energy_delta_vs_total_kWh"],
               out[f"{sid}_hourly_cost_delta_vs_total_yuan"]) > CASH_TOL:
            raise AssertionError(f"{sid} hourly sums do not match the verified totals")
        out[f"{sid}_total_delta_vs_summary_kWh"] = abs(totals[sid]["emergency_kWh"]
                                                       - float(reference.loc[sid, "emergency_kWh"]))
        out[f"{sid}_total_delta_vs_summary_yuan"] = abs(
            totals[sid]["emergency_cost_yuan"] - float(reference.loc[sid, "emergency_cost_yuan"]))
        if max(out[f"{sid}_total_delta_vs_summary_kWh"],
               out[f"{sid}_total_delta_vs_summary_yuan"]) > CASH_TOL:
            raise AssertionError(f"{sid} totals disagree with summary.csv")
        # node_age is the partition of the 47762 pairings; merged_age re-aggregates the same
        # segments, so it must be checked separately and never added to node_age
        for group_type in ("node_age", "merged_age"):
            grouped = age[(age.strategy_id == sid) & (age.group_type == group_type)]
            covered = float(grouped.emergency_kWh_self.sum()) \
                + float(midnight_rows[sid].emergency_kWh.sum())
            out[f"{sid}_{group_type}_plus_midnight_residual_kWh"] = abs(
                covered - totals[sid]["emergency_kWh"])
            if out[f"{sid}_{group_type}_plus_midnight_residual_kWh"] > CASH_TOL:
                raise AssertionError(f"{sid} {group_type} + midnight does not cover the period")

        frame = frames[sid]
        paired = frame[~frame.is_midnight]
        checks = {
            "error_identity_max_abs_kWh": known_max_abs(
                paired.eps_kWh.to_numpy() - paired.a_load_kWh.to_numpy()
                - paired.a_pv_kWh.to_numpy(), f"{sid} eps = a_load + a_pv"),
            "g_identity_max_abs_kWh": known_max_abs(
                paired.g_kWh.to_numpy() - paired.eps_kWh.to_numpy() + paired.rho_kWh.to_numpy(),
                f"{sid} g = eps - rho"),
            "truth_identity_max_abs_kWh": known_max_abs(
                frame.truth_recompute_residual_kWh, f"{sid} (L - V)*dt == net"),
            "net_forecast_identity_max_abs_kWh": known_max_abs(
                paired.net_forecast_kWh.to_numpy()
                - (paired.forecast_load_kW.to_numpy() - paired.forecast_pv_kW.to_numpy()) * DT,
                f"{sid} net_forecast = (Lhat - Vhat)*dt"),
            "emergency_identity_max_abs_kWh": known_max_abs(
                frame.expected_emergency_kWh.to_numpy() - frame.emergency_kWh.to_numpy(),
                f"{sid} e = (A - Dmax)^+")}
        for key, value in checks.items():
            out[f"{sid}_{key}"] = value
            if value > ENERGY_TOL:
                raise AssertionError(f"{sid} {key} = {value:.3e}")
        emergency = frame[frame.emergency_kWh > EMERGENCY_TOL]
        counts = {k: int(v) for k, v in emergency.discharge_limit_class.value_counts().items()}
        assert set(counts) <= {"inventory_limited", "power_limited", "close"}, counts
        assert sum(counts.values()) == len(emergency)
        assert (frame.loc[frame.emergency_kWh <= EMERGENCY_TOL, "discharge_limit_class"]
                == "not_emergency").all()
        out[f"{sid}_emergency_class_counts"] = counts
        out[f"{sid}_emergency_intervals"] = int(len(emergency))
    out["tolerances"] = dict(energy_atol=ENERGY_TOL, cash_atol=CASH_TOL, rtol=0)
    out["scope"] = ("hourly sums, age groups + midnight, the error/g identities and the "
                    "storage-class partition; no physics re-audit and no storage replay")
    return out


# ======================================================================================
# §6  登记与 main
# ======================================================================================
def register(parameters, inputs, dependency, amend_reason):
    hashes = {rel: digest(ROOT / rel) for rel in inputs}
    signature = hashlib.sha256(json.dumps(
        dict(parameters=parameters, dependency=dependency, input_sha256=hashes,
             code=digest(CODE)), sort_keys=True).encode()).hexdigest()
    path = OUT / "registration.json"
    record = dict(registered_utc=utc_now(), specification=str(PLAN_MD.relative_to(ROOT)),
                  time_version="start_time_v1", parameters=parameters, dependency=dependency,
                  input_sha256=hashes, code_sha256=digest(CODE), executable=sys.executable,
                  python=sys.version, numpy=np.__version__, pandas=pd.__version__)
    if path.exists():
        existing = load_json(path)
        if existing["signature"] != signature:
            assert amend_reason, "inputs or code changed since registration; pass --amend-reason"
            amendments = existing.get("amendments", [])
            amendments.append(dict(amended_utc=utc_now(),
                                   previous_signature=existing["signature"],
                                   previous_code_sha256=existing.get("code_sha256"),
                                   new_signature=signature, new_code_sha256=digest(CODE),
                                   reason=amend_reason))
            record["amendments"] = amendments
    record["signature"] = signature
    save_json(path, record)
    snapshot = OUT / "source_archive"
    (snapshot / "code").mkdir(parents=True, exist_ok=True)
    for rel in inputs:
        if rel.startswith("code/"):
            shutil.copy2(ROOT / rel, snapshot / "code" / Path(rel).name)
    shutil.copy2(PLAN_MD, snapshot / "experiment_plan.md")
    return signature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["register", "full"], default="full")
    parser.add_argument("--amend-reason", default=None)
    args = parser.parse_args()
    assert Path(sys.prefix).name == "math_modeling", sys.prefix
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    started_utc = utc_now()
    timings = {}

    inputs = ["results/q3_bias_quantile_cost/L75/dispatch.csv",
              "results/q3_rolling_baseline/B2_dispatch.csv",
              "results/q3_bias_correction_diagnostic/bias_forecast_archive.npz",
              "results/q3_bias_quantile_cost/protection_q75.npz",
              "results/q3_bias_quantile_cost/summary.csv",
              "results/q3_bias_quantile_cost/run_manifest.json",
              "reports/问题三/问题三_分位费用实验接收核查.md",
              "reports/问题三/问题三_新增预报时刻必要性轻量分析方案.md"]
    for rel in inputs:
        assert (ROOT / rel).exists(), rel
    dependency = dict(cost_registration=load_json(COST_DIR / "registration.json")["signature"],
                      cost_manifest_status=load_json(COST_MANIFEST)["status"],
                      zero_budget=dict(training=0, milp_lp_dp=0, storage_replay=0,
                                       public_january_reruns=0, prediction_rebuild=0,
                                       quantile_rebuild=0, update_time_ablation=0,
                                       parameter_sweeps=0))
    signature = register(PARAMETERS, inputs, dependency, args.amend_reason)
    print(f"registered signature={signature[:16]}", flush=True)
    if args.mode == "register":
        return

    tick = time.perf_counter()
    data = read_inputs()
    frames, tails, midnight_rows = {}, {}, {}
    for sid in STRATEGIES:
        frame, tail = build_intervals(sid, data["dispatch"][sid], data)
        frames[sid], tails[sid] = frame, tail
        midnight_rows[sid] = frame[frame.is_midnight]
    totals = {sid: dict(emergency_kWh=float(frames[sid].emergency_kWh.sum()),
                        emergency_cost_yuan=float(frames[sid].emergency_cost_yuan.sum()))
              for sid in STRATEGIES}
    timings["inputs_seconds"] = time.perf_counter() - tick
    for sid in STRATEGIES:
        print(f"{sid}: {NUM_INTERVALS} intervals, emergency {totals[sid]['emergency_kWh']:,.3f} kWh, "
              f"cost {totals[sid]['emergency_cost_yuan']:,.3f} yuan", flush=True)

    tick = time.perf_counter()
    checks = dict(samples=check_samples(frames, tails), mapping=check_mapping(frames, data))
    timings["checks_early_seconds"] = time.perf_counter() - tick
    print("sample and mapping checks passed", flush=True)

    tick = time.perf_counter()
    hourly = build_hourly(frames, totals)
    monthly_hourly = build_monthly_hourly(frames)
    age = build_age_summary(frames, totals, midnight_rows)
    top3_hours, risk = build_risk_windows(frames, hourly, midnight_rows)
    candidates = build_candidates(risk)
    timings["aggregates_seconds"] = time.perf_counter() - tick
    print(f"top-{RISK_HOURS} risk hours by L75 emergency cost: {top3_hours}; "
          f"plus rule-satisfying hours beyond the top 3: "
          f"{sorted(set(risk.clock_hour) - set(top3_hours))}", flush=True)

    tick = time.perf_counter()
    checks["aggregates"] = check_aggregates(frames, hourly, age, totals, midnight_rows,
                                            data["summary"])
    timings["checks_aggregate_seconds"] = time.perf_counter() - tick
    print("aggregation checks passed", flush=True)

    interval_cols = [
        "strategy_id", "interval_start", "interval_end", "k", "clock_hour", "clock_minute",
        "target_h", "version", "publication_hour", "age_hours", "age_bin", "is_midnight",
        "price_yuan_kWh", "actual_load_kW", "actual_pv_kW", "net_kWh",
        "forecast_load_kW", "forecast_pv_kW", "net_forecast_kWh", "a_load_kWh", "a_pv_kWh",
        "eps_kWh", "rho_kWh", "g_kWh", "pv_prominent", "q_eff_kWh", "state_start_kWh",
        "emergency_kWh", "emergency_cost_yuan", "need_before_battery_kWh", "discharge_cap_kWh",
        "state_end_kWh", "discharge_limit_class"]
    frame_to_csv(pd.concat([frames[sid][interval_cols] for sid in STRATEGIES], ignore_index=True),
                 OUT / "interval_diagnostic.csv")
    frame_to_csv(hourly, OUT / "hourly_summary.csv")
    frame_to_csv(monthly_hourly, OUT / "monthly_hourly.csv")
    frame_to_csv(age, OUT / "age_summary.csv")
    frame_to_csv(risk, OUT / "risk_window_evidence.csv")
    frame_to_csv(candidates, OUT / "candidate_windows.csv")

    validation = dict(
        status="passed", signature=signature, samples=checks["samples"],
        mapping=checks["mapping"], aggregates=checks["aggregates"], totals=totals,
        top3_risk_hours=top3_hours,
        rule_satisfying_hours=[int(h) for h in sorted(set(risk.clock_hour) - set(top3_hours))],
        reported_risk_hours=[int(h) for h in risk.clock_hour],
        excluded=["the 2026-01-01 00:00-00:10 template tail (one row per ledger)",
                  "the 334 midnight carry segments, from the latest-forecast pairing and tau "
                  "diagnostic only; their cash and emergency stay in the hourly/monthly totals"],
        zero_budget=dependency["zero_budget"],
        limits=["descriptive only: no new MILP, no storage replay, no update-time ablation and no "
                "parameter sweep, so no saving from a new publication time can be computed",
                "emergency cost inside a subset is not a causal cost contribution and not a "
                "recoverable saving",
                "the candidate publication clock is derived from the risk hour by a fixed one-hour "
                "engineering convention and is not an estimated optimum",
                "with a 6-hourly publication grid, (publication node, forecast age) is in one-to-one "
                "correspondence with the clock hour, so age cannot be separated from hour-of-day",
                "2025 was already used for method design: this describes one developed year, not an "
                "independent blind test",
                "no figure was inspected visually (the agent cannot view images)",
                "the two descriptive thresholds were introduced in this round, are not "
                "pre-registered, and only structure the candidate table: no publication time is "
                "excluded by them",
                "the emergency-subset cost of an hour is companion evidence, not an isolated "
                "causal contribution; storage couples across periods, so no hour is ruled out"])
    save_json(OUT / "validation.json", validation)
    manifest = dict(status="complete", started_utc=started_utc, finished_utc=utc_now(),
                    wall_seconds=time.perf_counter() - started, timings=timings,
                    signature=signature, executable=sys.executable, python=sys.version,
                    zero_solves=dict(training=0, milp_lp_dp=0, storage_replay=0,
                                     prediction_rebuild=0, quantile_rebuild=0),
                    inputs_read=sorted(inputs),
                    outputs={p.relative_to(OUT).as_posix(): digest(p) for p in sorted(OUT.rglob("*"))
                             if p.is_file() and p.name not in NON_COMPUTED_ARTIFACTS})
    save_json(OUT / "run_manifest.json", manifest)
    print(json.dumps({"status": "complete", "wall_seconds": manifest["wall_seconds"],
                      "top3_risk_hours": top3_hours,
                      "reported_risk_hours": [int(h) for h in risk.clock_hour],
                      "verdicts": candidates[["risk_clock_hour", "candidate_publication_clock",
                                              "verdict"]].to_dict("records")}, indent=2,
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
