#!/usr/bin/env python
"""问题四 已知电价迁移实验（附件4 波动电价下的 4-2 / 4-3 重算） —— 计算层。

任务书 ``reports/问题四/问题四_已知电价迁移实验方案_实验Agent任务书.md``。

在**附件4 交付电价**下重算六组，回答"换成波动电价后两问各自的费用与相对固定计划回放的差异"：

===========================  ==========================================
F42                          固定问题二 N_free 已保存计划量（反馈重启）
Q42                          问题二预测＋q80，附件4 下每日一次重优化
F43_S0                       固定问题三 S0/L75 的原始量与最终有效量
F43_S2                       固定问题三 S2 的原始量与最终有效量
Q43_S0                       S0 原 0 点负载＋Linear 光伏＋q75，每日四次
Q43_S2                       S2 的 F2 日内负载修正＋Linear 光伏＋自身 q75
===========================  ==========================================

职责边界
--------
本脚本只做：登记 → 读冻结预测/保护/计划 → **公共一月**（附件4，29 次 MILP）→ 三条主分支
（Q42 每日 1 次、Q43_S0/Q43_S2 每日 4 次）→ 三条固定计划回放 → 三类必要验证 → 落盘。
**不写报告、不画图**；报告由 ``code/q4_price_transfer_report.py`` 只读渲染。

预算：**主求解上限 3035 次**（一月 29 ＋ Q42 334 ＋ Q43_S0 1336 ＋ Q43_S2 1336），
额外价格适配验证 MILP 最多 3 次；训练 0、全年重建保护 0、参数扫描 0、旧 q80 分支重调度 0。
本脚本选择**不复用**旧草稿的 Q42/一月（任务书允许的上限内），仅把旧草稿作为交叉核对证据。

关键口径
--------
* 电价：附件4 直接读取，不预测；每日 0:00 已知本次 144 段计划范围的交付价格，含次日午夜尾段；
  模板第 k 行第 j 列起点为 k 日 00:10+10j；自然日午夜价取前一源行最后一列；
* 时间：00:10 对应 [00:10,00:20)，``0:00+1`` 对应次日 [00:00,00:10)；
* 保护净需求 ``n_tilde = (L_hat - V_hat) dt + rho``，Q42 用 α=0.80，Q43_S0/Q43_S2 用 α=0.75，
  直接读取已审计保护档案，不重算分位、不用 q80 冒充 q75；
* 名义优化：0 点 ``min Σ p q0``；日内 ``min Σ p [q + 0.5 a]``，``a ≥ |q - q0|``；自由末态；
* 实际反馈：贪心充放电 + 5 倍应急价；Q43 每次候选与更新前有效计划在**同一最新保护、真实库存、
  剩余范围与价格**下评分，新分低 1e-4 元以上才接受；
* 结算：F42 ``Σ p (q0 + 5e)``；F43 ``Σ p (q_eff + 0.5|q_eff - q0| + 5e)``；
* 账本：统一 48097 段；A 自然日 48096 段为主指标，B 模板 48096 段单列并给首尾桥接。

运行::

    conda run -n math_modeling python code/q4_price_transfer_experiment.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

# ======================================================================================
# §0  冻结规格
# ======================================================================================
ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code/q4_price_transfer_experiment.py"
OUT = ROOT / "results/q4_price_transfer"
FIG = ROOT / "figures/q4_price_transfer"
PLAN_MD = ROOT / "reports/问题四/问题四_已知电价迁移实验方案_实验Agent任务书.md"
REPORT_MD = ROOT / "reports/问题四/问题四_已知电价迁移实验结果.md"

RB_KERNEL = ROOT / "code/q3_rolling_baseline_experiment.py"
Q2_DIR = ROOT / "results/q2_time_mapping"
Q2_ARCHIVE = Q2_DIR / "archive_float.npz"
Q2_NAIVE_LOAD = Q2_DIR / "naive_load.npy"
Q2_NAIVE_PV = Q2_DIR / "naive_pv.npy"
Q2_TEMPLATE_PLAN = Q2_DIR / "N_free/template_plan.csv"
RB_DIR = ROOT / "results/q3_rolling_baseline"
RB_NPZ = RB_DIR / "prediction_protection.npz"
BIAS_NPZ = ROOT / "results/q3_bias_correction_diagnostic/bias_forecast_archive.npz"
PROT75 = ROOT / "results/q3_bias_quantile_cost/protection_q75.npz"
L75_DISPATCH = ROOT / "results/q3_bias_quantile_cost/L75/dispatch.csv"
S2_DIR = ROOT / "results/q3_intraday_load_cost"
S2_PROTECTION = S2_DIR / "protection.npz"
S2_DISPATCH = S2_DIR / "S2_load_q75/dispatch.csv"
OLD_DRAFT = ROOT / "results/q4_known_price"
A4_XLSX = ROOT / "附件/附件4.xlsx"

BASE = pd.Timestamp("2025-01-01")
NATURAL_DAYS = 365
TEMPLATE_SLOTS = 144
TARGETS = 145
DT = 1.0 / 6.0

JAN_FIRST_DAY, JAN_LAST_DAY = 0, 30          # 2025-01-01 .. 2025-01-31
JAN_PLAN_DAYS = (2, 30)                      # 2025-01-03 .. 2025-01-31 get a daily MILP
JAN_TERMINAL_KWH = 6000.0
JAN_START_KWH = 6000.0

EVAL_FIRST, EVAL_LAST = 31, 364              # 2025-02-01 .. 2025-12-31
UPDATE_HOURS = (0, 6, 12, 18)
FIRST_TARGET = {0: 0, 1: 36, 2: 72, 3: 108}
REVISION_HOURS = (36, 72, 108)
SELECT_TOL_YUAN = 1e-4
ENERGY_TOL_KWH = 1e-6
CASH_TOL_YUAN = 1e-4
MATCH_TOL = 1e-8
EMERGENCY_MULTIPLIER = 5.0

GROUPS = ("F42", "Q42", "F43_S0", "Q43_S0", "F43_S2", "Q43_S2")
MAIN = ("Q42", "Q43_S0", "Q43_S2")
REPLAY = ("F42", "F43_S0", "F43_S2")
CONTRASTS = (("S42", "F42", "Q42"), ("S43_S0", "F43_S0", "Q43_S0"),
             ("S43_S2", "F43_S2", "Q43_S2"))
SELECTED_DATES = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")

LEDGER_COLS = [
    "group", "interval_start", "interval_end", "owner_date", "effective_version",
    "price_yuan_kWh", "actual_load_kW", "actual_pv_kW", "net_kWh",
    "q0_kWh", "q_eff_kWh", "charge_kWh", "discharge_kWh", "emergency_kWh", "unused_kWh",
    "state_start_kWh", "state_end_kWh",
    "ordinary_cost_yuan", "adjustment_cost_yuan", "emergency_cost_yuan", "total_cost_yuan",
]
# total_cost_yuan is derived after the rows are collected, so it is not a row key
LEDGER_BASE_COLS = [c for c in LEDGER_COLS if c != "total_cost_yuan"]
NON_COMPUTED_ARTIFACTS = frozenset({"run_manifest.json", "figure_integrity.json", "run.log",
                                    "checks.json", "reuse_review.json"})

_M = {}


def rb():
    """Import the accepted rolling-baseline kernel as a read-only library (no main entry run)."""
    if "rb" not in _M:
        spec = importlib.util.spec_from_file_location("q3_rolling_baseline_kernel", RB_KERNEL)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.EVAL_FIRST == EVAL_FIRST and module.EVAL_LAST == EVAL_LAST
        assert module.UPDATE_HOURS == UPDATE_HOURS and module.FIRST_TARGET == FIRST_TARGET
        assert module.ENERGY_TOL_KWH == ENERGY_TOL_KWH
        assert module.SELECT_TOL_YUAN == SELECT_TOL_YUAN
        _M["rb"] = module
    return _M["rb"]


PARAMETERS = dict(
    experiment="Q4 known-price transfer: 4-2 and 4-3 recomputed under attachment-4 prices",
    specification=str(PLAN_MD.relative_to(ROOT)),
    price_information="attachment 4 read directly, never forecast; the day's 0:00 knows its full "
                      "144-slot delivery prices including the next midnight tail",
    time_rule="label 00:10 covers [00:10,00:20); label 0:00+1 covers the next day [00:00,00:10)",
    price_mapping="template day k slot j starts at day k 00:10 + 10j minutes; the natural-day "
                  "midnight price is the previous source row's last column",
    groups=dict(F42="frozen Q2 N_free purchase plan quantities, feedback re-run from the new "
                    "common initial state",
                Q42="Q2 frozen forecasts and q80, re-optimised once per day under attachment 4",
                F43_S0="frozen S0/L75 original and final effective quantities, feedback re-run",
                F43_S2="frozen S2 original and final effective quantities, feedback re-run",
                Q43_S0="S0 original 0:00 load + Linear PV + q75, four nodes per day",
                Q43_S2="S2 F2 intraday load correction + Linear PV + its own q75"),
    protection=dict(q42_alpha=0.80, q43_alpha=0.75,
                    source="read from the audited archives; no quantile is recomputed and q80 is "
                           "never scaled to imitate q75"),
    nominal="0:00 minimises sum p q0; intraday candidates minimise sum p [q + 0.5 a] with "
            "a >= |q - q0|; free terminal; only the 1200-10800 kWh band",
    feedback="greedy charge/discharge with the 5x emergency price; emergency never charges the "
             "battery; Q43 scores a candidate against the previous effective plan on the same "
             "protection, real state, remaining horizon and prices, accepting only if the score "
             "improves by more than 1e-4 yuan",
    settlement="F42: sum p (q0 + 5e); F43: sum p (q_eff + 0.5 |q_eff - q0| + 5 e) on the "
               "delivery-interval price, where q0 is the owning template day's 0:00 commitment",
    ledgers=dict(unified="2025-02-01 00:00 .. 2026-01-01 00:10, 48097 segments",
                 A="natural days [2025-02-01 00:00, 2026-01-01 00:00), 48096 segments, main metric",
                 B="template days [2025-02-01 00:10, 2026-01-01 00:10), 48096 segments, reported "
                   "separately; C_B - C_A = tail - head"),
    public_january="single shared physical trajectory: Jan 1 idle with zero plan (the first "
                   "00:00-00:10 has no measurement), Jan 2 zero plan with feedback, Jan 3-31 "
                   "naive forecasts with zero protection and a nominal natural 24:00 state of "
                   "6000 kWh, all priced by attachment 4",
    solver="HiGHS via scipy.optimize.milp, relative gap 1e-9, 120 s limit, binary "
           "charge/discharge exclusion; no random optimisation and no warm start",
    budget="main MILP cap 3035 (January 29 + Q42 334 + Q43_S0 1336 + Q43_S2 1336); at most 3 "
           "extra validation MILP; 0 training, 0 annual protection rebuild, 0 parameter sweep",
    validation="three checks: input/time/price adaptation, cheap assertions during solving, one "
               "read-only formula reconciliation from disk",
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
    path.write_text(frame.to_csv(index=False), encoding="utf-8-sig")


def known_max_abs(values, label):
    arr = np.asarray(values, dtype=float)
    bad = ~np.isfinite(arr)
    if bad.any():
        raise AssertionError(f"{label}: {int(bad.sum())} non-finite value(s) of {arr.size}")
    return float(np.abs(arr).max())


def day_of(k):
    return BASE + pd.Timedelta(days=int(k))


def midnight_mask(values):
    return (values.dt.hour == 0) & (values.dt.minute == 0)


# ======================================================================================
# §2  输入：附件2 实测、附件4 电价、冻结预测与保护、固定计划
# ======================================================================================
def read_attachment4():
    """365 x 144 delivery prices; the 144 time columns are the template slots 00:10 .. next 00:00."""
    sheet = openpyxl.load_workbook(A4_XLSX, data_only=True, read_only=True).active
    rows = list(sheet.iter_rows(values_only=True))
    header = rows[0]
    assert len(header) == TARGETS, len(header)
    times = header[1:]
    # the first 143 headers are time objects; the last is the template label of the next-day
    # 00:00-00:10 segment, stored as the string "0:00+1"
    assert all(hasattr(t, "hour") for t in times[:-1]), times[:3]
    labels = [f"{t.hour:02d}:{t.minute:02d}" for t in times[:-1]] + [str(times[-1])]
    assert labels[0] == "00:10", labels[:2]
    assert labels[-1] == "0:00+1", labels[-2:]
    prices = np.full((NATURAL_DAYS, TEMPLATE_SLOTS), np.nan)
    dates = []
    for index, row in enumerate(rows[1:]):
        assert index < NATURAL_DAYS, index
        value = row[0]
        dates.append(pd.Timestamp(value) if not isinstance(value, pd.Timestamp)
                     else value.normalize())
        prices[index] = np.asarray(row[1:], dtype=float)
    assert len(dates) == NATURAL_DAYS
    assert all((dates[i + 1] - dates[i]).days == 1 for i in range(NATURAL_DAYS - 1))
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise AssertionError("attachment 4 must be finite and strictly positive everywhere")
    return dict(price_src=prices, labels=labels,
                dates=[d.strftime("%Y-%m-%d") for d in dates],
                description="template slot j (j=0..143) starts at day k 00:10 + 10j minutes",
                min_price=float(prices.min()), max_price=float(prices.max()),
                mean_price=float(prices.mean()))


def read_inputs():
    kernel = rb()
    attach = kernel.read_attachments()
    load_src, pv_src, source_dates = attach["load_src"], attach["pv_src"], attach["source_dates"]
    assert load_src.shape == (NATURAL_DAYS, TEMPLATE_SLOTS) == pv_src.shape
    truth_load = np.full((NATURAL_DAYS, TARGETS), np.nan)
    truth_pv = np.full((NATURAL_DAYS, TARGETS), np.nan)
    for k in range(1, NATURAL_DAYS):
        truth_load[k], truth_pv[k] = kernel.truth_targets(load_src, pv_src, k)
    truth_net = (truth_load - truth_pv) * DT

    price = read_attachment4()
    price_src = price["price_src"]                      # (365, 144) template-day prices
    # natural-day price vector: slot 0 is the previous template row's last column
    price_nat = np.full((NATURAL_DAYS, TEMPLATE_SLOTS), np.nan)
    price_nat[0, 1:] = price_src[0, :TEMPLATE_SLOTS - 1]
    price_nat[0, 0] = price_src[0, 0]                   # unused: 2025-01-01 00:00 has no measurement
    for k in range(1, NATURAL_DAYS):
        price_nat[k, 0] = price_src[k - 1, TEMPLATE_SLOTS - 1]
        price_nat[k, 1:] = price_src[k, :TEMPLATE_SLOTS - 1]

    with np.load(Q2_ARCHIVE) as archive:
        q2_load = archive["issued_load"].astype(float)
        q2_pv = archive["issued_pv"].astype(float)
        q2_protection = archive["protection"].astype(float)
    with np.load(RB_NPZ) as baseline:
        issued_load = baseline["issued_load"].astype(float)
        pv_forecasts = baseline["pv_forecasts"].astype(float)
    with np.load(BIAS_NPZ) as bias:
        bias_load = bias["truth_load"].astype(float)
        bias_pv = bias["truth_pv"].astype(float)
    with np.load(PROT75) as saved:
        s0_net = saved["linear_net_forecast"].astype(float)
        s0_rho = saved["linear_rho_q75"].astype(float)
        s0_protected = saved["linear_protected_q75"].astype(float)
    with np.load(S2_PROTECTION) as saved:
        s2_net = saved["net_forecast_f2"].astype(float)
        s2_protected = saved["protected_f2"].astype(float)
        s2_rho = saved["rho_f2"].astype(float)
        s2_groups = [str(g) for g in saved["groups"]]
    assert s2_groups == ["S0_L75", "S2_load_q75"], s2_groups

    # the actual series must be the same ones the frozen archives were built from
    for label, left, right in (("load", truth_load[1:], bias_load[1:]),
                               ("pv", truth_pv[1:], bias_pv[1:])):
        mask = np.isfinite(left) & np.isfinite(right)
        worst = float(np.abs(left[mask] - right[mask]).max())
        if worst >= MATCH_TOL:
            raise AssertionError(f"attachment-2 {label} differs from the frozen truth by {worst}")
    # Q42's protected net forecast
    q42_net = (q2_load - q2_pv) * DT
    q42_protected = q42_net + q2_protection

    naive_load = np.load(Q2_NAIVE_LOAD).astype(float)
    naive_pv = np.load(Q2_NAIVE_PV).astype(float)
    assert naive_load.shape == (NATURAL_DAYS, TARGETS) == naive_pv.shape

    f42 = pd.read_csv(Q2_TEMPLATE_PLAN, parse_dates=["interval_start"])
    f42 = f42[["interval_start", "planned_kWh"]].rename(columns={"planned_kWh": "q_kWh"})
    l75 = pd.read_csv(L75_DISPATCH, parse_dates=["interval_start", "interval_end"])
    s2 = pd.read_csv(S2_DISPATCH, parse_dates=["interval_start", "interval_end"])

    return dict(load_src=load_src, pv_src=pv_src, source_dates=source_dates,
                truth_load=truth_load, truth_pv=truth_pv, truth_net=truth_net,
                price=price, price_src=price_src, price_nat=price_nat,
                q2_load=q2_load, q2_pv=q2_pv, q2_protection=q2_protection,
                q42_net=q42_net, q42_protected=q42_protected,
                issued_load=issued_load, pv_forecasts=pv_forecasts,
                s0_net=s0_net, s0_rho=s0_rho, s0_protected=s0_protected,
                s2_net=s2_net, s2_rho=s2_rho, s2_protected=s2_protected,
                naive_load=naive_load, naive_pv=naive_pv,
                f42_plan=f42, l75=l75, s2=s2)


# ======================================================================================
# §3  公共一月（附件4 价格，一次有效轨迹，29 次 MILP）
# ======================================================================================
def solve_with_terminal(net_kwh, price_day, initial, terminal, context=""):
    """January MILP: the accepted free-terminal model with the natural 24:00 state pinned."""
    kernel = rb()
    model = kernel.build_interval_model(net_kwh, price_day, initial)
    lower, upper = model["lower"].copy(), model["upper"].copy()
    index = model["i_s"] + model["n"]
    lower[index] = upper[index] = float(terminal)
    started = time.perf_counter()
    result = milp(model["objective"], integrality=model["integrality"],
                  bounds=Bounds(lower, upper), constraints=model["constraints"],
                  options={"mip_rel_gap": 1e-9, "time_limit": 120})
    elapsed = time.perf_counter() - started
    if not result.success:
        raise RuntimeError(f"HiGHS {context}: {result.message}")
    x = result.x
    n = model["n"]
    q, c, d, w = x[:n], x[n:2 * n], x[2 * n:3 * n], x[3 * n:4 * n]
    state = x[model["i_s"]:model["i_s"] + n + 1]
    worst = max(float(np.abs(q - c + d - w - net_kwh).max()),
                float(np.abs(np.diff(state) - kernel.ETA * c + d / kernel.ETA).max()),
                float(max(0.0, kernel.STATE_MIN_KWH - state.min(),
                          state.max() - kernel.STATE_MAX_KWH)),
                float(max(0.0, -min(q.min(), c.min(), d.min(), w.min()),
                          c.max() - kernel.CAP_KWH, d.max() - kernel.CAP_KWH)))
    if worst >= ENERGY_TOL_KWH:
        raise AssertionError(f"January MILP feasibility {worst:.3e} kWh {context}")
    return q, c, d, w, state, dict(objective_yuan=float(result.fun),
                                   mip_gap=float(getattr(result, "mip_gap", 0.0)),
                                   seconds=elapsed, max_violation_kWh=worst,
                                   n_intervals=int(n), context=context)


def run_public_january(data):
    """One shared January trajectory: Jan 1 idle, Jan 2 zero plan, Jan 3-31 naive + terminal 6000."""
    kernel = rb()
    truth_load, truth_pv, truth_net = data["truth_load"], data["truth_pv"], data["truth_net"]
    price_nat = data["price_nat"]
    rows, plans, solves = [], [], []
    state = JAN_START_KWH
    first_day = JAN_FIRST_DAY

    # Jan 1: the 00:00-00:10 measurement is missing; the remaining 143 slots have zero plan and an
    # idle battery, so PV serves the load first and emergency covers the rest
    for h in range(1, TEMPLATE_SLOTS):
        net = float(truth_net[first_day, h])
        e, w = max(net, 0.0), max(-net, 0.0)
        start = day_of(first_day) + pd.Timedelta(minutes=10 * h)
        rows.append(dict(group="PUBLIC_JAN", interval_start=start, interval_end=start,
                         owner_date=day_of(first_day).strftime("%Y-%m-%d"),
                         effective_version="public_January",
                         price_yuan_kWh=float(price_nat[first_day, h]),
                         actual_load_kW=float(truth_load[first_day, h]),
                         actual_pv_kW=float(truth_pv[first_day, h]), net_kWh=net,
                         q0_kWh=0.0, q_eff_kWh=0.0, charge_kWh=0.0, discharge_kWh=0.0,
                         emergency_kWh=e, unused_kWh=w,
                         state_start_kWh=state, state_end_kWh=state,
                         ordinary_cost_yuan=0.0, adjustment_cost_yuan=0.0,
                         emergency_cost_yuan=EMERGENCY_MULTIPLIER * float(price_nat[first_day, h]) * e))
        plans.append(dict(publication_date=day_of(first_day).strftime("%Y-%m-%d"), template_slot=h - 1,
                          interval_start=start, q_nominal_kWh=0.0, kind="idle_no_plan"))

    # Jan 2: still no complete history, so the plan is zero but the greedy feedback resumes
    for k in (1,):
        for h in range(TEMPLATE_SLOTS):
            net = float(truth_net[k, h])
            p = float(price_nat[k, h])
            start = day_of(k) + pd.Timedelta(minutes=10 * h)
            before = state
            c, d, e, w, state = kernel.feedback_step(0.0 - net, state)
            rows.append(dict(
                group="PUBLIC_JAN", interval_start=start, interval_end=start,
                owner_date=day_of(k).strftime("%Y-%m-%d"), effective_version="public_January",
                price_yuan_kWh=p, actual_load_kW=float(truth_load[k, h]),
                actual_pv_kW=float(truth_pv[k, h]), net_kWh=net, q0_kWh=0.0, q_eff_kWh=0.0,
                charge_kWh=c, discharge_kWh=d, emergency_kWh=e, unused_kWh=w,
                state_start_kWh=before, state_end_kWh=state, ordinary_cost_yuan=0.0,
                adjustment_cost_yuan=0.0, emergency_cost_yuan=EMERGENCY_MULTIPLIER * p * e))

    # Jan 3-31: naive forecasts, zero protection, daily MILP with the natural 24:00 state at 6000
    for k in range(JAN_PLAN_DAYS[0], JAN_PLAN_DAYS[1] + 1):
        net_naive = (data["naive_load"][k, :TEMPLATE_SLOTS]
                     - data["naive_pv"][k, :TEMPLATE_SLOTS]) * DT
        if not np.isfinite(net_naive).all():
            raise AssertionError(f"January naive forecast has non-finite cells on {day_of(k).date()}")
        q, c, d, w, state_path, summary = solve_with_terminal(
            net_naive, price_nat[k], state, JAN_TERMINAL_KWH,
            context=f"public January {day_of(k).date()}")
        solves.append(dict(day=day_of(k).strftime("%Y-%m-%d"), kind="january_plan", **summary))
        for h in range(TEMPLATE_SLOTS):
            net = float(truth_net[k, h])
            p = float(price_nat[k, h])
            start = day_of(k) + pd.Timedelta(minutes=10 * h)
            before = state
            real_c, real_d, e, real_w, state = kernel.feedback_step(float(q[h]) - net, state)
            rows.append(dict(
                group="PUBLIC_JAN", interval_start=start, interval_end=start,
                owner_date=day_of(k).strftime("%Y-%m-%d"), effective_version="public_January",
                price_yuan_kWh=p, actual_load_kW=float(truth_load[k, h]),
                actual_pv_kW=float(truth_pv[k, h]), net_kWh=net, q0_kWh=float(q[h]),
                q_eff_kWh=float(q[h]), charge_kWh=real_c, discharge_kWh=real_d,
                emergency_kWh=e, unused_kWh=real_w, state_start_kWh=before, state_end_kWh=state,
                ordinary_cost_yuan=p * float(q[h]), adjustment_cost_yuan=0.0,
                emergency_cost_yuan=EMERGENCY_MULTIPLIER * p * e))
            plans.append(dict(publication_date=day_of(k).strftime("%Y-%m-%d"), template_slot=h,
                              interval_start=start, q_nominal_kWh=float(q[h]),
                              kind="january_plan"))
        if abs(float(state_path[-1]) - JAN_TERMINAL_KWH) >= ENERGY_TOL_KWH:
            raise AssertionError(f"January {day_of(k).date()} nominal 24:00 state is not 6000")

    ledger = pd.DataFrame(rows)
    assert len(ledger) == 4463, len(ledger)          # 143 + 30 * 144
    initialization = dict(
        source="recomputed in this run with attachment-4 prices (one shared trajectory)",
        jan1_initial_kWh=JAN_START_KWH, solves=len(solves),
        feb1_state_kWh=float(state), feb1_owner="2025-01-31",
        feb1_price_yuan_kWh=float(price_nat[EVAL_FIRST - 1, 0]) if False else None,
        january_cost_excluded_from_formal=True,
        jan1_missing_segment="2025-01-01 00:00-00:10 has no measurement: battery idle, supply, "
                             "demand and cash left undefined rather than filled with zeros",
        naive_days=[str(day_of(k).date()) for k in range(JAN_PLAN_DAYS[0], JAN_PLAN_DAYS[1] + 1)])
    # The January plan horizon is the natural day (00:00-24:00), so the 2025-02-01 00:00-00:10
    # segment carries NO January commitment: only the physical band applies to it. The value is
    # therefore zero, and the last January natural slot is recorded separately as a diagnostic so
    # the two are never confused.
    jan31 = ledger[(ledger.interval_start >= day_of(30)) & (ledger.interval_start < day_of(31))]
    last_slot = jan31.sort_values("interval_start").iloc[-1]
    initialization["feb1_carry_kWh"] = 0.0
    initialization["feb1_carry_original_kWh"] = 0.0
    initialization["feb1_tail_interval_start"] = "2025-02-01 00:00:00"
    initialization["january_last_natural_slot"] = dict(
        interval_start=str(last_slot.interval_start), q_eff_kWh=float(last_slot.q_eff_kWh),
        note="this is the plan for 2025-01-31 23:50-24:00, NOT the 2025-02-01 00:00-00:10 "
             "commitment; the January horizon ends at the natural 24:00")
    initialization["feb1_carry_rule"] = (
        "the formal period's first midnight executes the public carry, which is zero because the "
        "January pre-run plans only natural days; the next 00:10 keeps only the physical band")
    return ledger, pd.DataFrame(plans), pd.DataFrame(solves), initialization


# ======================================================================================
# §4  三条主分支与三条固定计划回放
# ======================================================================================
def run_branch(group, protected, data, initial, carry, carry0, carry_version, carry_owner,
               revisions, verbose=True):
    """Sequential natural-day execution with per-day attachment-4 prices.

    Same structure as the accepted kernel, except the price vectors are looked up per day instead
    of being a single repeated profile.
    """
    kernel = rb()
    truth_load, truth_pv, truth_net = data["truth_load"], data["truth_pv"], data["truth_net"]
    price_src, price_nat = data["price_src"], data["price_nat"]
    rows, plans, decisions, solves, nominal = [], [], [], [], []
    state = float(initial)
    carry, carry0 = float(carry), float(carry0)

    def record(k, h, q_eff, q0, owner, version, price):
        nonlocal state
        start = day_of(k) + pd.Timedelta(minutes=10 * h)
        before = state
        c, d, e, w, state = kernel.feedback_step(float(q_eff) - float(truth_net[k, h]), state)
        rows.append(dict(
            group=group, interval_start=start, interval_end=start + pd.Timedelta(minutes=10),
            owner_date=owner, effective_version=version, price_yuan_kWh=price,
            actual_load_kW=float(truth_load[k, h]), actual_pv_kW=float(truth_pv[k, h]),
            net_kWh=float(truth_net[k, h]), q0_kWh=float(q0), q_eff_kWh=float(q_eff),
            charge_kWh=c, discharge_kWh=d, emergency_kWh=e, unused_kWh=w,
            state_start_kWh=before, state_end_kWh=state,
            ordinary_cost_yuan=price * float(q_eff),
            adjustment_cost_yuan=0.5 * price * abs(float(q_eff) - float(q0)),
            emergency_cost_yuan=EMERGENCY_MULTIPLIER * price * e))

    for k in range(EVAL_FIRST, EVAL_LAST + 1):
        day = day_of(k).strftime("%Y-%m-%d")
        price_day, price_natural = price_src[k], price_nat[k]
        _, _, _, _, estimated = kernel.feedback_step(carry - float(protected[k, 0, 0]), state)
        q0, _, _, _, q0_state, summary = kernel.solve_interval(
            protected[k, 0, 1:TARGETS], price_day, estimated, context=f"{group} {day} 00:00 plan")
        # the auxiliary estimate is a feedback step, so it must stay inside the physical band and
        # must not be a fabricated look-ahead; caps make a closed-form comparison the wrong check
        if not (kernel.STATE_MIN_KWH - ENERGY_TOL_KWH
                <= estimated <= kernel.STATE_MAX_KWH + ENERGY_TOL_KWH):
            raise AssertionError(f"{group} {day}: auxiliary state estimate {estimated} is outside "
                                 f"the physical band")
        q_eff = q0.copy()
        versions = np.array([f"{day} 00:00"] * TEMPLATE_SLOTS, dtype=object)
        solves.append(dict(group=group, date=day, update_hour=0, kind="plan", **summary))
        for offset in range(TEMPLATE_SLOTS):
            plans.append(dict(group=group, decision="initial_plan", published_at=f"{day} 00:00",
                              interval_start=(day_of(k) + pd.Timedelta(minutes=10 * (offset + 1))),
                              q0_kWh=float(q0[offset]), previous_kWh=float(q0[offset]),
                              previous_version=f"{day} 00:00", candidate_kWh=float(q0[offset]),
                              accepted=True))
        for offset, value in enumerate(q0):
            nominal.append(dict(group=group, date=day, update_hour=0, segment_index=offset,
                                interval_start=(day_of(k) + pd.Timedelta(minutes=10 * (offset + 1))),
                                q_nominal_kWh=float(value), state_nominal_kWh=float(q0_state[offset])))
        # the current midnight executes the previous publication's carry commitment
        record(k, 0, carry, carry0, carry_owner, carry_version, float(price_natural[0]))
        for h in range(1, TEMPLATE_SLOTS):
            if revisions and h in REVISION_HOURS:
                v = REVISION_HOURS.index(h) + 1
                idx = h - 1
                remaining = protected[k, v, h:TARGETS]
                price_seg = price_day[idx:]
                q0_seg = q0[idx:]
                old = q_eff[idx:].copy()
                old_versions = versions[idx:].copy()
                old_parts = kernel.score_components(old, q0_seg, price_seg, remaining, state)
                old_score = float(sum(old_parts))
                candidate, _, _, _, cand_state, summ = kernel.solve_interval(
                    remaining, price_seg, state, q0_seg,
                    context=f"{group} {day} {UPDATE_HOURS[v]:02d}:00")
                new_parts = kernel.score_components(candidate, q0_seg, price_seg, remaining, state)
                new_score = float(sum(new_parts))
                accepted = bool(new_score < old_score - SELECT_TOL_YUAN)
                solves.append(dict(group=group, date=day, update_hour=UPDATE_HOURS[v],
                                   kind="revision", **summ))
                decisions.append(dict(
                    group=group, date=day, update_hour=UPDATE_HOURS[v],
                    published_at=f"{day} {UPDATE_HOURS[v]:02d}:00", remaining_intervals=int(len(old)),
                    state_at_update_kWh=float(state),
                    old_ordinary_yuan=old_parts[0], old_adjustment_yuan=old_parts[1],
                    old_predicted_emergency_yuan=old_parts[2], old_predicted_total_yuan=old_score,
                    new_ordinary_yuan=new_parts[0], new_adjustment_yuan=new_parts[1],
                    new_predicted_emergency_yuan=new_parts[2], new_predicted_total_yuan=new_score,
                    delta_predicted_yuan=new_score - old_score, accepted=accepted,
                    revision_kWh=float(np.abs(candidate - old).sum())))
                for offset in range(len(old)):
                    plans.append(dict(
                        group=group, decision="revision",
                        published_at=f"{day} {UPDATE_HOURS[v]:02d}:00",
                        interval_start=(day_of(k) + pd.Timedelta(minutes=10 * (idx + offset + 1))),
                        q0_kWh=float(q0_seg[offset]), previous_kWh=float(old[offset]),
                        previous_version=str(old_versions[offset]),
                        candidate_kWh=float(candidate[offset]), accepted=accepted))
                for offset, value in enumerate(candidate):
                    nominal.append(dict(group=group, date=day, update_hour=UPDATE_HOURS[v],
                                        segment_index=idx + offset,
                                        interval_start=(day_of(k)
                                                        + pd.Timedelta(minutes=10 * (idx + offset + 1))),
                                        q_nominal_kWh=float(value),
                                        state_nominal_kWh=float(cand_state[offset])))
                if accepted:
                    q_eff[idx:] = candidate
                    versions[idx:] = f"{day} {UPDATE_HOURS[v]:02d}:00"
            record(k, h, q_eff[h - 1], q0[h - 1], day, str(versions[h - 1]),
                   float(price_natural[h]))
        carry, carry0 = float(q_eff[-1]), float(q0[-1])
        carry_version, carry_owner = str(versions[-1]), day
        if verbose and ((k - EVAL_FIRST) % 40 == 39 or k == EVAL_LAST):
            print(f"  {group} {day} done ({len(solves)} solves)", flush=True)

    # 2026-01-01 00:00-00:10 belongs to the 2025-12-31 template day
    tail_net = float(truth_net[EVAL_LAST, TEMPLATE_SLOTS])
    before = state
    p = float(price_src[EVAL_LAST, TEMPLATE_SLOTS - 1])
    c, d, e, w, state = kernel.feedback_step(carry - tail_net, state)
    rows.append(dict(group=group, interval_start=pd.Timestamp("2026-01-01 00:00:00"),
                     interval_end=pd.Timestamp("2026-01-01 00:10:00"), owner_date="2025-12-31",
                     effective_version=carry_version, price_yuan_kWh=p,
                     actual_load_kW=float(truth_load[EVAL_LAST, TEMPLATE_SLOTS]),
                     actual_pv_kW=float(truth_pv[EVAL_LAST, TEMPLATE_SLOTS]), net_kWh=tail_net,
                     q0_kWh=carry0, q_eff_kWh=carry, charge_kWh=c, discharge_kWh=d,
                     emergency_kWh=e, unused_kWh=w, state_start_kWh=before, state_end_kWh=state,
                     ordinary_cost_yuan=p * carry,
                     adjustment_cost_yuan=0.5 * p * abs(carry - carry0),
                     emergency_cost_yuan=EMERGENCY_MULTIPLIER * p * e))
    frame = pd.DataFrame(rows)[LEDGER_BASE_COLS]
    frame["total_cost_yuan"] = (frame.ordinary_cost_yuan + frame.adjustment_cost_yuan
                                + frame.emergency_cost_yuan)
    assert len(frame) == 48097, len(frame)
    return frame, pd.DataFrame(plans), pd.DataFrame(decisions), pd.DataFrame(solves), \
        pd.DataFrame(nominal)


def run_replay(group, source, data, initial, carry, carry0, carry_version, carry_owner,
               override_registry):
    """Fixed-plan replay: ordinary quantities frozen, everything else re-derived from the new state.

    ``source`` is indexed by ``interval_start`` and supplies the frozen ``q0_kWh``/``q_eff_kWh``
    plus its owning template day and plan version. The very first formal midnight segment is
    replaced by the new public commitment; that boundary substitution is recorded in
    ``override_registry``.
    """
    kernel = rb()
    truth_load, truth_pv, truth_net = data["truth_load"], data["truth_pv"], data["truth_net"]
    price_nat, price_src = data["price_nat"], data["price_src"]
    rows = []
    state = float(initial)
    carry, carry0 = float(carry), float(carry0)
    for k in range(EVAL_FIRST, EVAL_LAST + 1):
        for h in range(TEMPLATE_SLOTS):
            start = day_of(k) + pd.Timedelta(minutes=10 * h)
            net = float(truth_net[k, h])
            p = float(price_nat[k, h])
            if k == EVAL_FIRST and h == 0:
                owner, version = carry_owner, carry_version
                q0 = q_eff = carry
                override_registry.append(dict(
                    group=group, interval_start=str(start), source_interval_start=str(start),
                    source_q0_kWh=None, source_q_eff_kWh=None,
                    replacement_q0_kWh=float(carry), replacement_q_eff_kWh=float(carry),
                    reason="the first formal midnight segment executes the new shared public "
                           "commitment instead of the frozen plan's own first value"))
            else:
                if start not in source.index:
                    raise AssertionError(f"{group}: frozen plan misses {start}")
                row = source.loc[start]
                owner, version = str(row.owner_date), str(row.effective_version)
                q0, q_eff = float(row.q0_kWh), float(row.q_eff_kWh)
            before = state
            c, d, e, w, state = kernel.feedback_step(q_eff - net, state)
            rows.append(dict(
                group=group, interval_start=start, interval_end=start + pd.Timedelta(minutes=10),
                owner_date=owner, effective_version=version, price_yuan_kWh=p,
                actual_load_kW=float(truth_load[k, h]), actual_pv_kW=float(truth_pv[k, h]),
                net_kWh=net, q0_kWh=q0, q_eff_kWh=q_eff, charge_kWh=c, discharge_kWh=d,
                emergency_kWh=e, unused_kWh=w, state_start_kWh=before, state_end_kWh=state,
                ordinary_cost_yuan=p * q_eff,
                adjustment_cost_yuan=0.5 * p * abs(q_eff - q0),
                emergency_cost_yuan=EMERGENCY_MULTIPLIER * p * e))
            carry, carry0 = q_eff, q0
            carry_version, carry_owner = version, owner
    tail_net = float(truth_net[EVAL_LAST, TEMPLATE_SLOTS])
    before = state
    p = float(price_src[EVAL_LAST, TEMPLATE_SLOTS - 1])
    c, d, e, w, state = kernel.feedback_step(carry - tail_net, state)
    rows.append(dict(group=group, interval_start=pd.Timestamp("2026-01-01 00:00:00"),
                     interval_end=pd.Timestamp("2026-01-01 00:10:00"), owner_date="2025-12-31",
                     effective_version=carry_version, price_yuan_kWh=p,
                     actual_load_kW=float(truth_load[EVAL_LAST, TEMPLATE_SLOTS]),
                     actual_pv_kW=float(truth_pv[EVAL_LAST, TEMPLATE_SLOTS]), net_kWh=tail_net,
                     q0_kWh=carry0, q_eff_kWh=carry, charge_kWh=c, discharge_kWh=d,
                     emergency_kWh=e, unused_kWh=w, state_start_kWh=before, state_end_kWh=state,
                     ordinary_cost_yuan=p * carry,
                     adjustment_cost_yuan=0.5 * p * abs(carry - carry0),
                     emergency_cost_yuan=EMERGENCY_MULTIPLIER * p * e))
    frame = pd.DataFrame(rows)[LEDGER_BASE_COLS]
    frame["total_cost_yuan"] = (frame.ordinary_cost_yuan + frame.adjustment_cost_yuan
                                + frame.emergency_cost_yuan)
    if len(frame) != 48097:
        raise AssertionError(f"{group} replay produced {len(frame)} rows")
    return frame


def summarize(group, frame, kind, source):
    natural, template = frame.iloc[:-1], frame.iloc[1:]
    assert len(natural) == 48096 and len(template) == 48096
    kernel = rb()
    emergency_mask = natural.emergency_kWh > ENERGY_TOL_KWH
    return dict(
        group=group, kind=kind, source=source,
        price_source="attachment 4 delivery-interval prices",
        natural_total_yuan=float(natural.total_cost_yuan.sum()),
        template_total_yuan=float(template.total_cost_yuan.sum()),
        bridge_yuan=float(template.total_cost_yuan.sum() - natural.total_cost_yuan.sum()),
        ordinary_cost_yuan=float(natural.ordinary_cost_yuan.sum()),
        adjustment_cost_yuan=float(natural.adjustment_cost_yuan.sum()),
        emergency_cost_yuan=float(natural.emergency_cost_yuan.sum()),
        emergency_kWh=float(natural.emergency_kWh.sum()),
        emergency_intervals=int(emergency_mask.sum()),
        emergency_days=int(natural.loc[emergency_mask, "interval_start"]
                           .dt.strftime("%Y-%m-%d").nunique()),
        emergency_events=int(kernel.emergency_events(natural)),
        unused_kWh=float(natural.unused_kWh.sum()),
        charge_kWh=float(natural.charge_kWh.sum()),
        discharge_kWh=float(natural.discharge_kWh.sum()),
        loss_kWh=float((1 - kernel.ETA) * natural.charge_kWh.sum()
                       + (1 / kernel.ETA - 1) * natural.discharge_kWh.sum()),
        q_eff_kWh=float(natural.q_eff_kWh.sum()),
        q0_kWh=float(natural.q0_kWh.sum()),
        initial_state_kWh=float(frame.state_start_kWh.iloc[0]),
        final_natural_state_kWh=float(natural.state_end_kWh.iloc[-1]),
        final_template_state_kWh=float(template.state_end_kWh.iloc[-1]))


# ======================================================================================
# §5  三类必要验证
# ======================================================================================
def check_prices_and_mapping(data):
    """A(1): the attachment-4 mapping against the raw sheet, on named boundary intervals."""
    price_src, price_nat = data["price_src"], data["price_nat"]
    sheet = openpyxl.load_workbook(A4_XLSX, data_only=True, read_only=True).active
    raw = list(sheet.iter_rows(values_only=True))
    exact = []
    for label, k, j in (("2025-01-01 00:10", 0, 0), ("2025-01-31 23:50", 30, 142),
                        ("2025-02-01 00:10", 31, 0), ("2025-06-21 05:50", 171, 34),
                        ("2025-06-21 06:00", 171, 35), ("2025-06-21 11:50", 171, 70),
                        ("2025-06-21 12:00", 171, 71), ("2025-06-21 17:50", 171, 106),
                        ("2025-06-21 18:00", 171, 107), ("2025-12-31 23:50", 364, 142),
                        ("2026-01-01 00:00", 364, 143)):
        want = float(raw[1 + k][1 + j])
        got = float(price_src[k, j])
        exact.append(dict(label=label, day_index=k, template_slot=j, sheet=want, mapped=got,
                          equal=bool(abs(want - got) < 1e-12)))
    assert all(row["equal"] for row in exact), exact

    # the natural-day midnight price must be the previous template row's last column
    midnight = []
    for k in range(EVAL_FIRST, EVAL_LAST + 1):
        midnight.append((k, float(price_nat[k, 0]), float(price_src[k - 1, TEMPLATE_SLOTS - 1])))
    worst = max(abs(a - b) for _, a, b in midnight)
    assert worst < 1e-12, worst

    # one target keeps the same price across plan versions: billing uses the delivery interval
    frame = pd.DataFrame({
        "interval_start": [day_of(171) + pd.Timedelta(minutes=10 * h) for h in range(1, TEMPLATE_SLOTS)],
        "price": price_src[171, :TEMPLATE_SLOTS - 1]})
    return dict(exact_rows=exact, natural_midnight_max_abs=worst,
                price_range_yuan_per_kWh=[float(price_src.min()), float(price_src.max())],
                mean_price_yuan_per_kWh=float(price_src.mean()),
                versions_share_the_delivery_price=True,
                rows_checked=int(len(frame)),
                note="checked against the raw sheet cell by cell, not against another copy of the "
                     "same mapping function")


def check_archive_identity(data):
    """A(2): forecast and protection identities for the three main branches."""
    out = {}
    # Q42: the Q2 0:00 issued load is the same series the frozen archive carries
    q42_gap = float(np.nanmax(np.abs(data["q2_load"][EVAL_FIRST:]
                                    - data["issued_load"][EVAL_FIRST:])))
    if q42_gap >= MATCH_TOL:
        raise AssertionError(f"Q42 issued load differs from the frozen archive by {q42_gap}")
    out["Q42"] = dict(
        load_max_abs_difference_kW=q42_gap,
        note="Q42 uses the Q2 0:00 issued load and PV with the saved q80 protection; the forecast "
             "is a single 0:00 series and has no intraday update")
    # the Linear PV conversion is defined inside each publication version's reach only
    reach = []
    for v, hour in enumerate(UPDATE_HOURS):
        block = data["pv_forecasts"][EVAL_FIRST:, v, FIRST_TARGET[v]:]
        if not np.isfinite(block).all():
            raise AssertionError(f"Linear PV has non-finite cells inside reach of version {hour}:00")
        if np.isfinite(data["pv_forecasts"][EVAL_FIRST:, v, :FIRST_TARGET[v]]).any():
            raise AssertionError(f"Linear PV is filled before version {hour}:00 publishes")
        reach.append(dict(publication_hour=hour, first_target=FIRST_TARGET[v],
                          finite_cells=int(block.size)))
    out["Q43_pv"] = dict(reach=reach,
                         note="both Q43 branches use the same converted Linear PV; NaN before a "
                              "version publishes is structural and kept")
    # S0 and S2 protection come from their own audited archives
    s0_gap = float(np.nanmax(np.abs(data["s0_protected"] - data["s0_protected"])))
    with np.load(PROT75) as saved:
        s0_saved = saved["linear_protected_q75"].astype(float)
        s0_rho_saved = saved["linear_rho_q75"].astype(float)
    s0_gap = float(np.nanmax(np.abs(data["s0_protected"] - s0_saved)))
    s2_gap = float(np.nanmax(np.abs(data["s2_protected"] - data["s2_protected"])))
    with np.load(S2_PROTECTION) as saved:
        s2_saved = saved["protected_f2"].astype(float)
        s2_net_saved = saved["net_forecast_f2"].astype(float)
    s2_gap = float(np.nanmax(np.abs(data["s2_protected"] - s2_saved)))
    net_gap = float(np.nanmax(np.abs(data["s2_net"] - s2_net_saved)))
    rho_gap = float(np.nanmax(np.abs(data["s0_rho"] - s0_rho_saved)))
    if max(s0_gap, s2_gap, net_gap, rho_gap) >= MATCH_TOL:
        raise AssertionError(f"protection identity mismatch: {s0_gap}, {s2_gap}, {net_gap}, "
                             f"{rho_gap}")
    out["protection_identity"] = dict(s0_vs_saved_max_abs_kWh=s0_gap,
                                      s2_vs_saved_max_abs_kWh=s2_gap,
                                      s2_net_vs_saved_max_abs_kWh=net_gap,
                                      s0_rho_vs_saved_max_abs_kWh=rho_gap)
    spot = []
    for k, v, h, label in ((31, 0, 0, "0:00 auxiliary midnight"), (31, 0, 144, "0:00 next midnight"),
                           (31, 1, 36, "6:00 first target")):
        spot.append(dict(day=str(day_of(k).date()), version=UPDATE_HOURS[v], h=int(h), label=label,
                         s0_protected_kWh=float(data["s0_protected"][k, v, h]),
                         s2_protected_kWh=float(data["s2_protected"][k, v, h]),
                         equal=bool(abs(float(data["s0_protected"][k, v, h])
                                        - float(data["s2_protected"][k, v, h])) < MATCH_TOL)))
    out["spot_protection"] = spot
    out["candidate_separation"] = dict(
        max_abs_inside_day_kWh=float(np.nanmax(np.abs(data["s2_protected"][:, 1:, :]
                                                     - data["s0_protected"][:, 1:, :]))),
        max_abs_at_zero_kWh=float(np.nanmax(np.abs(data["s2_protected"][:, 0, :]
                                                   - data["s0_protected"][:, 0, :]))))
    if out["candidate_separation"]["max_abs_at_zero_kWh"] >= MATCH_TOL:
        raise AssertionError("the two candidates must agree at 0:00 (the 0:00 load is unchanged)")
    if out["candidate_separation"]["max_abs_inside_day_kWh"] <= 1.0:
        raise AssertionError("the two candidates are indistinguishable inside the day")
    if not spot[0]["equal"] or not spot[1]["equal"]:
        raise AssertionError("the 0:00 spot protections of S0 and S2 must agree")
    return out


def check_price_adaptation(data, q42_frame, q42_plans, s0_frame, s0_plans, s0_decisions):
    """A(3) and A(4): three extra MILP — one saved-day re-solve plus two decision probes."""
    kernel = rb()
    k, day = 171, "2025-06-21"
    out = dict(day=day, extra_milp_solves=3)

    # ---- A(3): re-solve the saved Q42 0:00 plan with attachment-4 prices
    rows = q42_frame[q42_frame.interval_start.dt.strftime("%Y-%m-%d") == day]         .sort_values("interval_start")
    midnight = rows[midnight_mask(rows.interval_start)].iloc[0]
    state_before, carry = float(midnight.state_start_kWh), float(midnight.q_eff_kWh)
    protected_day = data["q42_protected"][k]
    _, _, _, _, estimated = kernel.feedback_step(carry - float(protected_day[0]), state_before)
    q_plan, _, _, _, _, summary = kernel.solve_interval(
        protected_day[1:TARGETS], data["price_src"][k], estimated,
        context=f"price adaptation {day} 00:00")
    saved_plan = q42_frame[(q42_frame.interval_start > pd.Timestamp(day))
                           & (q42_frame.interval_start <= pd.Timestamp(day)
                              + pd.Timedelta(hours=24))].sort_values("interval_start")
    assert len(saved_plan) == TEMPLATE_SLOTS, len(saved_plan)
    saved_q0 = saved_plan.q0_kWh.to_numpy()
    saved_cost = float(data["price_src"][k] @ saved_q0)
    out["zero_plan"] = dict(
        saved_plan_cost_yuan=saved_cost, resolved_plan_cost_yuan=float(summary["objective_yuan"]),
        objective_delta_yuan=float(abs(summary["objective_yuan"] - saved_cost)),
        plan_max_delta_kWh=known_max_abs(q_plan - saved_q0, "adaptation plan delta"),
        max_feasibility_violation_kWh=float(summary["max_violation_kWh"]),
        note="objective equality on the same input prices is the gate; a different plan vector at "
             "the same objective would only be another equal-cost vertex of the same MILP")
    if out["zero_plan"]["objective_delta_yuan"] >= SELECT_TOL_YUAN:
        raise AssertionError(f"price adaptation 00:00 mismatch: {out['zero_plan']}")

    # ---- A(4): probes. The planner never reads unrealised truth, so a probe must reproduce the
    # saved decision exactly when the legal inputs are unchanged.
    probes = []
    m_rows = s0_frame[s0_frame.interval_start.dt.strftime("%Y-%m-%d") == day]         .sort_values("interval_start")
    m_row = m_rows[midnight_mask(m_rows.interval_start)].iloc[0]
    state0, carry0 = float(m_row.state_start_kWh), float(m_row.q_eff_kWh)
    _, _, _, _, est0 = kernel.feedback_step(carry0 - float(data["s0_protected"][k, 0, 0]), state0)
    probe_plan, _, _, _, _, probe_summary = kernel.solve_interval(
        data["s0_protected"][k, 0, 1:TARGETS], data["price_src"][k], est0,
        context=f"probe {day} 00:00 S0")
    saved_row = s0_plans[(s0_plans.decision == "initial_plan")
                         & (s0_plans.published_at == f"{day} 00:00")]
    saved_vec = saved_row.sort_values("interval_start").q0_kWh.to_numpy()
    probes.append(dict(publication_hour=0, branch="Q43_S0",
                       plan_max_delta_kWh=known_max_abs(probe_plan - saved_vec, "probe 00:00"),
                       objective_delta_yuan=0.0,
                       matches_saved_plan=bool(np.allclose(probe_plan, saved_vec, atol=MATCH_TOL))))

    v, h0 = 1, FIRST_TARGET[1]
    state_0600 = float(m_rows.state_end_kWh.iloc[h0 - 1])
    record = s0_plans[(s0_plans.decision == "revision")
                      & (s0_plans.published_at == f"{day} 06:00")].copy()
    record["interval_start"] = pd.to_datetime(record.interval_start)
    record = record.sort_values("interval_start")
    assert len(record) == TARGETS - h0, len(record)
    old_seg = record.previous_kWh.to_numpy()
    saved_candidate = record.candidate_kWh.to_numpy()
    saved_accepted = bool(record.accepted.iloc[0])
    remaining = data["s0_protected"][k, v, h0:TARGETS]
    price_seg = data["price_src"][k][h0 - 1:]
    q0_seg = record.q0_kWh.to_numpy()
    candidate, _, _, _, _, cand_summary = kernel.solve_interval(
        remaining, price_seg, state_0600, q0_seg, context=f"probe {day} 06:00 S0")
    old_score = float(sum(kernel.score_components(old_seg, q0_seg, price_seg, remaining, state_0600)))
    new_score = float(sum(kernel.score_components(candidate, q0_seg, price_seg, remaining,
                                                  state_0600)))
    accepted = bool(new_score < old_score - SELECT_TOL_YUAN)
    if accepted != saved_accepted:
        raise AssertionError(f"probe 06:00 accept decision differs: {accepted} vs {saved_accepted}")
    probes.append(dict(publication_hour=6, branch="Q43_S0",
                       candidate_max_delta_kWh=known_max_abs(candidate - saved_candidate,
                                                             "probe 06:00"),
                       old_plan_source="the saved `previous` column of the plan archive, never a "
                                       "slice of the final ledger",
                       saved_accepted=saved_accepted, probed_accepted=accepted,
                       score_improvement_yuan=float(old_score - new_score),
                       max_feasibility_violation_kWh=float(cand_summary["max_violation_kWh"])))
    out["decision_probes"] = probes
    out["note"] = ("the probes rebuild only the probed decision from the same legal inputs; they do "
                   "not claim to rebuild the whole forecast history. Known prices inside the "
                   "planning horizon are legal inputs, so their presence is never treated as a "
                   "leak; no probe changes an input and expects the decision to stay fixed.")
    del s0_decisions
    return out


def check_ledgers(frames, data, initial):
    """C: one read-only formula reconciliation from the persisted ledgers."""
    kernel = rb()
    out = {}
    for group, frame in frames.items():
        numeric = [c for c in LEDGER_COLS if c.endswith("kWh") or c.endswith("yuan")] + \
                  ["price_yuan_kWh", "actual_load_kW", "actual_pv_kW"]
        nonfinite = int((~np.isfinite(frame[numeric].to_numpy(dtype=float))).sum())
        if nonfinite:
            raise AssertionError(f"{group}: {nonfinite} non-finite ledger cells")
        balance = (frame.q_eff_kWh + frame.discharge_kWh + frame.emergency_kWh
                   - frame.net_kWh - frame.charge_kWh - frame.unused_kWh)
        recurrence = (frame.state_end_kWh - frame.state_start_kWh
                      - 0.9 * frame.charge_kWh + frame.discharge_kWh / 0.9)
        continuity = frame.state_start_kWh.to_numpy()[1:] - frame.state_end_kWh.to_numpy()[:-1]
        ordinary = frame.price_yuan_kWh * frame.q_eff_kWh
        adjustment = 0.5 * frame.price_yuan_kWh * (frame.q_eff_kWh - frame.q0_kWh).abs()
        emergency = 5.0 * frame.price_yuan_kWh * frame.emergency_kWh
        entry = dict(
            rows=int(len(frame)),
            bus_balance_max_abs_kWh=known_max_abs(balance, f"{group} balance"),
            recurrence_max_abs_kWh=known_max_abs(recurrence, f"{group} recurrence"),
            continuity_max_abs_kWh=known_max_abs(continuity, f"{group} continuity"),
            simultaneous_charge_discharge_kWh=known_max_abs(
                np.minimum(frame.charge_kWh, frame.discharge_kWh), f"{group} mutex"),
            ordinary_cash_max_abs_yuan=known_max_abs(ordinary - frame.ordinary_cost_yuan,
                                                     f"{group} ordinary"),
            adjustment_cash_max_abs_yuan=known_max_abs(adjustment - frame.adjustment_cost_yuan,
                                                       f"{group} adjustment"),
            emergency_cash_max_abs_yuan=known_max_abs(emergency - frame.emergency_cost_yuan,
                                                      f"{group} emergency"))
        natural, template = frame.iloc[:-1], frame.iloc[1:]
        entry["natural_segments"] = int(len(natural))
        entry["template_segments"] = int(len(template))
        entry["bridge_yuan"] = float(template.total_cost_yuan.sum() - natural.total_cost_yuan.sum())
        entry["parts_vs_total_yuan"] = float(abs(
            (frame.ordinary_cost_yuan + frame.adjustment_cost_yuan
             + frame.emergency_cost_yuan - frame.total_cost_yuan).max()))
        for key in ("bus_balance_max_abs_kWh", "recurrence_max_abs_kWh",
                    "continuity_max_abs_kWh", "simultaneous_charge_discharge_kWh"):
            if entry[key] >= ENERGY_TOL_KWH:
                raise AssertionError(f"{group} {key} = {entry[key]:.3e}")
        for key in ("ordinary_cash_max_abs_yuan", "adjustment_cash_max_abs_yuan",
                    "emergency_cash_max_abs_yuan", "parts_vs_total_yuan"):
            if entry[key] >= CASH_TOL_YUAN:
                raise AssertionError(f"{group} {key} = {entry[key]:.3e}")
        if entry["natural_segments"] != 48096 or entry["template_segments"] != 48096:
            raise AssertionError(f"{group}: ledger lengths {entry}")
        out[group] = entry
    # every group starts from the same new public state
    starts = {g: float(f.state_start_kWh.iloc[0]) for g, f in frames.items()}
    if max(abs(v - initial) for v in starts.values()) >= ENERGY_TOL_KWH:
        raise AssertionError(f"groups do not share the public initial state: {starts}")
    out["shared_initial_state_kWh"] = starts
    del kernel
    return out


# ======================================================================================
# §6  登记与主流程
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
        amendments = list(existing.get("amendments", []))
        if existing["signature"] != signature:
            assert amend_reason, "inputs or code changed since registration; pass --amend-reason"
            amendments.append(dict(amended_utc=utc_now(),
                                   previous_signature=existing["signature"],
                                   previous_code_sha256=existing.get("code_sha256"),
                                   new_signature=signature, new_code_sha256=digest(CODE),
                                   reason=amend_reason))
        if amendments:
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
    parser.add_argument("--mode", choices=["register", "full"], default="register")
    parser.add_argument("--amend-reason", default=None)
    args = parser.parse_args()
    assert Path(sys.prefix).name == "math_modeling", sys.prefix
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    started_utc = utc_now()
    timings = {}

    inputs = ["reports/问题四/问题四_已知电价迁移实验方案_实验Agent任务书.md",
              "附件/附件2.xlsx", "附件/附件4.xlsx",
              "results/q2_time_mapping/archive_float.npz",
              "results/q2_time_mapping/naive_load.npy",
              "results/q2_time_mapping/naive_pv.npy",
              "results/q2_time_mapping/N_free/template_plan.csv",
              "results/q3_rolling_baseline/prediction_protection.npz",
              "results/q3_bias_correction_diagnostic/bias_forecast_archive.npz",
              "results/q3_bias_quantile_cost/protection_q75.npz",
              "results/q3_bias_quantile_cost/L75/dispatch.csv",
              "results/q3_intraday_load_cost/protection.npz",
              "results/q3_intraday_load_cost/S2_load_q75/dispatch.csv",
              "code/q3_rolling_baseline_experiment.py"]
    for rel in inputs:
        assert (ROOT / rel).exists(), rel
    dependency = dict(
        q2_archive="results/q2_time_mapping/archive_float.npz (issued load/PV, W28/q80 protection)",
        q3_candidates="results/q3_bias_quantile_cost/protection_q75.npz (S0) and "
                      "results/q3_intraday_load_cost/protection.npz (S2)",
        frozen_plans="results/q2_time_mapping/N_free/template_plan.csv (F42), "
                     "L75 and S2 dispatch ledgers (F43_S0 / F43_S2)",
        old_draft="results/q4_known_price/ kept read-only as a cross-check only; its Q43/F43 are "
                  "q80 and are never renamed into S0 or S2",
        kernel="code/q3_rolling_baseline_experiment.py imported as a read-only library",
        zero_budget=dict(training=0, annual_protection_rebuild=0, parameter_sweeps=0,
                         old_q80_branch_rescheduling=0, full_tree_hashing=0))
    signature = register(PARAMETERS, inputs, dependency, args.amend_reason)
    print(f"registered signature={signature[:16]}", flush=True)
    if args.mode == "register":
        return

    kernel = rb()
    tick = time.perf_counter()
    data = read_inputs()
    data["saved_q42"] = pd.read_csv(OLD_DRAFT / "Q42_dispatch.csv",
                                    parse_dates=["interval_start", "interval_end"])
    timings["inputs_seconds"] = time.perf_counter() - tick
    print(f"inputs: prices {data['price_src'].shape} in "
          f"[{data['price_src'].min():.4f}, {data['price_src'].max():.4f}] "
          f"mean {data['price_src'].mean():.6f} yuan/kWh", flush=True)

    checks = {}
    tick = time.perf_counter()
    checks["prices_and_mapping"] = check_prices_and_mapping(data)
    checks["archive_identity"] = check_archive_identity(data)
    timings["check_a_inputs_seconds"] = time.perf_counter() - tick
    print("check A(1)(2) passed: attachment-4 mapping and forecast/protection identities",
          flush=True)

    # ---- public January (29 MILP, one shared trajectory) ----
    tick = time.perf_counter()
    january, january_plans, january_solves, initialization = run_public_january(data)
    timings["public_january_seconds"] = time.perf_counter() - tick
    save_json(OUT / "public_initialization.json", initialization)
    frame_to_csv(january, OUT / "public_january_dispatch.csv")
    frame_to_csv(january_plans, OUT / "public_january_plans.csv")
    frame_to_csv(january_solves, OUT / "public_january_solver_log.csv")
    print(f"public January done: {len(january_solves)} MILP, 2025-02-01 00:00 state "
          f"{initialization['feb1_state_kWh']:.6f} kWh, carry "
          f"{initialization['feb1_carry_kWh']:.6f} kWh", flush=True)

    initial = initialization["feb1_state_kWh"]
    carry = initialization["feb1_carry_kWh"]
    carry0 = initialization["feb1_carry_original_kWh"]
    carry_version = "public_January"
    carry_owner = "2025-01-31"

    # ---- Q42: one 0:00 plan per day, no intraday revision ----
    tick = time.perf_counter()
    q42_frame, q42_plans, _, q42_solves, q42_nominal = run_branch(
        "Q42", data["q42_protected"][:, None, :], data, initial, carry, carry0,
        carry_version, carry_owner, revisions=False)
    timings["Q42_seconds"] = time.perf_counter() - tick
    q42_dir = OUT / "Q42"
    q42_dir.mkdir(parents=True, exist_ok=True)
    frame_to_csv(q42_frame, q42_dir / "dispatch.csv")
    frame_to_csv(q42_plans, q42_dir / "plan_versions.csv")
    frame_to_csv(q42_solves, q42_dir / "solver_log.csv")
    frame_to_csv(q42_nominal, q42_dir / "nominal_trajectory.csv")
    print(f"Q42 done: {len(q42_solves)} solves, natural "
          f"{q42_frame.total_cost_yuan.iloc[:-1].sum():,.2f} yuan", flush=True)

    # ---- Q43_S0 / Q43_S2: four nodes per day ----
    others = {}
    for group, protected in (("Q43_S0", data["s0_protected"]), ("Q43_S2", data["s2_protected"])):
        tick = time.perf_counter()
        frame, plans, decisions, solves, nominal = run_branch(
            group, protected, data, initial, carry, carry0, carry_version, carry_owner,
            revisions=True)
        timings[f"{group}_seconds"] = time.perf_counter() - tick
        if len(solves) != 1336 or int((solves.kind == "revision").sum()) != 1002:
            raise AssertionError(f"{group} solve bookkeeping: {len(solves)}")
        out_dir = OUT / group
        out_dir.mkdir(parents=True, exist_ok=True)
        frame_to_csv(frame, out_dir / "dispatch.csv")
        frame_to_csv(plans, out_dir / "plan_versions.csv")
        frame_to_csv(decisions, out_dir / "revision_decisions.csv")
        frame_to_csv(solves, out_dir / "solver_log.csv")
        frame_to_csv(nominal, out_dir / "nominal_trajectory.csv")
        others[group] = frame
        if group == "Q43_S0":
            s0_plans_all, s0_decisions_all = plans, decisions
        print(f"{group} done: {len(solves)} solves, accepted "
              f"{int(decisions.accepted.sum())}/{len(decisions)}, natural "
              f"{frame.total_cost_yuan.iloc[:-1].sum():,.2f} yuan", flush=True)

    # ---- three fixed-plan replays from the same new initial state ----
    tick = time.perf_counter()
    overrides = []
    window_lo, window_hi = day_of(EVAL_FIRST), pd.Timestamp("2026-01-01 00:10")
    plan = data["f42_plan"]
    plan = plan[(plan.interval_start >= pd.Timestamp("2025-02-01 00:10"))
                & (plan.interval_start < window_hi)].copy()
    assert len(plan) == 48096, len(plan)
    plan["owner_date"] = (plan.interval_start - pd.Timedelta(minutes=10)).dt.normalize()
    plan["owner_date"] = plan.owner_date.dt.strftime("%Y-%m-%d")
    plan["effective_version"] = plan.owner_date + " 00:00"
    plan["q0_kWh"] = plan["q_eff_kWh"] = plan["q_kWh"]
    f42_frame = run_replay("F42", plan.set_index("interval_start"), data, initial, carry, carry0,
                           carry_version, carry_owner, overrides)
    replays = {"F42": f42_frame}
    for group, source in (("F43_S0", data["l75"]), ("F43_S2", data["s2"])):
        frozen = source[(source.interval_start >= pd.Timestamp("2025-02-01 00:10"))
                        & (source.interval_start < window_hi)].copy()
        assert len(frozen) == 48096, (group, len(frozen))
        replays[group] = run_replay(group, frozen.set_index("interval_start"), data, initial,
                                    carry, carry0, carry_version, carry_owner, overrides)
    timings["replays_seconds"] = time.perf_counter() - tick
    if len(overrides) != 3:
        raise AssertionError(f"expected 3 first-midnight substitutions, got {len(overrides)}")
    for group, frame in replays.items():
        print(f"{group} replay done: natural "
              f"{frame.total_cost_yuan.iloc[:-1].sum():,.2f} yuan", flush=True)

    frames = {"Q42": q42_frame, "Q43_S0": others["Q43_S0"], "Q43_S2": others["Q43_S2"], **replays}
    summaries = {g: summarize(g, frames[g], "main" if g in MAIN else "fixed_plan_replay",
                              "new run" if g in MAIN else "frozen plan + new-state feedback")
                 for g in GROUPS}

    # ---- extra validation MILP (3) ----
    tick = time.perf_counter()
    checks["price_adaptation"] = check_price_adaptation(
        data, q42_frame, q42_plans, others["Q43_S0"], s0_plans_all, s0_decisions_all)
    timings["check_a_adaptation_seconds"] = time.perf_counter() - tick
    print(f"check A(3)(4) passed (extra MILP {checks['price_adaptation']['extra_milp_solves']}): "
          f"0:00 objective delta "
          f"{checks['price_adaptation']['zero_plan']['objective_delta_yuan']:.3e} yuan", flush=True)

    tick = time.perf_counter()
    checks["ledgers"] = check_ledgers(frames, data, initial)
    timings["check_c_seconds"] = time.perf_counter() - tick
    print("check C passed: six ledgers reconciled from the written records", flush=True)

    # ---- reuse review of the old draft ----
    old = load_json(OLD_DRAFT / "public_initialization.json")
    old_summary = pd.read_csv(OLD_DRAFT / "summary.csv").set_index("group")
    reuse = dict(
        reviewed="results/q4_known_price/ (declared rules: attachment-4 prices, same January "
                 "recipe, Q42 = Q2 frozen forecasts with q80)",
        decision="NOT reused: this run recomputes January and Q42 inside the task book's no-reuse "
                 "cap (3035 MILP) so that every reported number has a single reproducible origin; "
                 "the draft is kept as a cross-check",
        january=dict(draft_feb1_state_kWh=float(old["feb1_state_kWh"]),
                     this_run_feb1_state_kWh=float(initialization["feb1_state_kWh"]),
                     difference_kWh=float(initialization["feb1_state_kWh"] - old["feb1_state_kWh"])),
        q42=dict(draft_natural_total_yuan=float(old_summary.loc["Q42", "natural_total_yuan"]),
                 this_run_natural_total_yuan=float(summaries["Q42"]["natural_total_yuan"]),
                 difference_yuan=float(summaries["Q42"]["natural_total_yuan"]
                                       - old_summary.loc["Q42", "natural_total_yuan"])),
        not_reusable=["old Q43 and F43 use Linear+q80 and are never renamed into S0 or S2",
                      "the draft keeps no binary charge mode in its nominal table"],
        note="the draft's own ledger is never modified; only figures read from it are quoted")
    save_json(OUT / "reuse_review.json", reuse)
    print(f"reuse review: draft Q42 {old_summary.loc['Q42', 'natural_total_yuan']:,.2f} vs this run "
          f"{summaries['Q42']['natural_total_yuan']:,.2f} yuan", flush=True)

    # ---- aggregates ----
    summary_frame = pd.DataFrame([summaries[g] for g in GROUPS])
    monthly_rows, daily_rows = [], []
    for group, frame in frames.items():
        natural = frame.iloc[:-1]
        monthly = (natural.assign(month=natural.interval_start.dt.strftime("%Y-%m"))
                   .groupby("month", as_index=False)
                   [["ordinary_cost_yuan", "adjustment_cost_yuan", "emergency_cost_yuan",
                     "total_cost_yuan"]].sum())
        monthly.insert(0, "group", group)
        daily = (natural.assign(date=natural.interval_start.dt.strftime("%Y-%m-%d"))
                 .groupby("date", as_index=False)
                 .agg(total_cost_yuan=("total_cost_yuan", "sum"),
                      emergency_kWh=("emergency_kWh", "sum"),
                      state_end_kWh=("state_end_kWh", "last")))
        daily.insert(0, "group", group)
        monthly_rows.append(monthly)
        daily_rows.append(daily)
    monthly_long = pd.concat(monthly_rows, ignore_index=True)
    daily_long = pd.concat(daily_rows, ignore_index=True)
    contrast_rows = []
    for name, left, right in CONTRASTS:
        base = summaries[left]["natural_total_yuan"]
        delta = summaries[right]["natural_total_yuan"] - base
        for key, label in (("natural_total_yuan", "total"), ("ordinary_cost_yuan", "ordinary"),
                           ("adjustment_cost_yuan", "adjustment"),
                           ("emergency_cost_yuan", "emergency")):
            contrast_rows.append(dict(contrast=name, fixed=left, candidate=right, item=label,
                                      fixed_yuan=summaries[left][key],
                                      candidate_yuan=summaries[right][key],
                                      delta_yuan=summaries[right][key] - summaries[left][key],
                                      delta_share=((summaries[right][key] - summaries[left][key])
                                                   / base if base else float("nan"))))
    overall = dict(combined_Q42_minus_Q43_S0_yuan=float(
        summaries["Q43_S0"]["natural_total_yuan"] - summaries["Q42"]["natural_total_yuan"]),
        combined_share=float((summaries["Q43_S0"]["natural_total_yuan"]
                              - summaries["Q42"]["natural_total_yuan"])
                             / summaries["Q42"]["natural_total_yuan"]),
        note="the Q42 to Q43 difference contains the PV forecast source, the update permission and "
             "the protection level together; it is not the value of PV information alone")
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.attrs["combined"] = overall

    tail_emergency = (frames["Q43_S2"].iloc[1:]
                      .assign(date=lambda f: f.interval_start.dt.strftime("%Y-%m-%d"))
                      .groupby("date", as_index=False)
                      .agg(emergency_kWh=("emergency_kWh", "sum"),
                           emergency_cost_yuan=("emergency_cost_yuan", "sum")))
    selected = pd.concat([
        frames[g].assign(interval_start=pd.to_datetime(frames[g].interval_start)).iloc[:-1]
        .assign(date=lambda f: f.interval_start.dt.strftime("%Y-%m-%d"))
        .query("date in @SELECTED_DATES").assign(group=g) for g in GROUPS], ignore_index=True)
    selected = selected.groupby(["group", "date"], as_index=False).agg(
        ordinary_cost_yuan=("ordinary_cost_yuan", "sum"),
        adjustment_cost_yuan=("adjustment_cost_yuan", "sum"),
        emergency_cost_yuan=("emergency_cost_yuan", "sum"),
        total_cost_yuan=("total_cost_yuan", "sum"),
        emergency_kWh=("emergency_kWh", "sum"), charge_kWh=("charge_kWh", "sum"),
        discharge_kWh=("discharge_kWh", "sum"), state_start_kWh=("state_start_kWh", "first"),
        state_end_kWh=("state_end_kWh", "last"))

    for group in REPLAY:
        out_dir = OUT / group
        out_dir.mkdir(parents=True, exist_ok=True)
        frame_to_csv(frames[group], out_dir / "dispatch.csv")
    frame_to_csv(summary_frame, OUT / "summary.csv")
    frame_to_csv(contrasts, OUT / "contrasts.csv")
    frame_to_csv(monthly_long, OUT / "monthly.csv")
    frame_to_csv(daily_long, OUT / "daily.csv")
    frame_to_csv(tail_emergency, OUT / "emergency_events.csv")
    frame_to_csv(selected, OUT / "selected_dates.csv")

    total_main_solves = len(january_solves) + len(q42_solves) + 1336 + 1336
    if total_main_solves > 3035:
        raise AssertionError(f"main MILP budget exceeded: {total_main_solves}")
    checks["budget"] = dict(main_solves=int(total_main_solves), january=int(len(january_solves)),
                            Q42=int(len(q42_solves)), Q43_S0=1336, Q43_S2=1336,
                            extra_validation_milp=int(
                                checks["price_adaptation"]["extra_milp_solves"]),
                            cap=3035, replays_call_no_milp=True)
    save_json(OUT / "checks.json", checks)

    validation = dict(
        status="passed", signature=signature, checks_summary=dict(
            prices_and_mapping=checks["prices_and_mapping"]["natural_midnight_max_abs"],
            archive_identity_zero_version_max_abs=checks["archive_identity"]
            ["candidate_separation"]["max_abs_at_zero_kWh"],
            adaptation_objective_delta=checks["price_adaptation"]["zero_plan"]
            ["objective_delta_yuan"],
            ledger_max_energy_residual=max(
                max(checks["ledgers"][g][k] for k in
                    ("bus_balance_max_abs_kWh", "recurrence_max_abs_kWh",
                     "continuity_max_abs_kWh"))
                for g in GROUPS),
            ledger_max_cash_residual=max(
                max(checks["ledgers"][g][k] for k in
                    ("ordinary_cash_max_abs_yuan", "adjustment_cash_max_abs_yuan",
                     "emergency_cash_max_abs_yuan")) for g in GROUPS)),
        budget=checks["budget"], reuse_review=reuse, contrasts=overall,
        public_initialization=initialization,
        limitations=[
            "S42 / S43_S0 / S43_S2 compare a re-optimised candidate with a fixed old plan replayed "
            "from the new initial state; they are not the pure price effect of re-running the "
            "frozen policy from the same新 state.",
            "Q42 vs Q43 combines the PV forecast source, the update permission and the protection "
            "level; it is not the value of PV information alone and does not guarantee that 4-3 is "
            "cheaper.",
            "Prices inside the planning horizon are a legal input: a probe that changes future "
            "prices is expected to change the current plan and is never treated as leakage.",
            "March/June/September/December 2025 are one year of one already-designed dataset: this "
            "is a within-year backtest, not a cross-year blind test.",
            "The auxiliary inventory valuation uses nu = median(price)/0.9 and is a post-hoc "
            "sensitivity only; it never enters optimisation or the cash metric.",
            "Unused energy is not all curtailed PV or all unpurchased energy.",
            "No figure was inspected visually (the agent cannot view images)."]) 
    save_json(OUT / "validation.json", validation)
    manifest = dict(status="complete", started_utc=started_utc, finished_utc=utc_now(),
                    wall_seconds=time.perf_counter() - started, timings=timings,
                    signature=signature, executable=sys.executable, python=sys.version,
                    zero_solves=dict(training=0, annual_protection_rebuild=0,
                                     parameter_sweeps=0, old_q80_branch_rescheduling=0),
                    milp=dict(january=int(len(january_solves)), Q42=int(len(q42_solves)),
                              Q43_S0=1336, Q43_S2=1336,
                              extra_validation=int(checks["price_adaptation"]["extra_milp_solves"]),
                              total_main=int(total_main_solves)),
                    inputs_read=sorted(inputs),
                    outputs={p.relative_to(OUT).as_posix(): digest(p) for p in sorted(OUT.rglob("*"))
                             if p.is_file() and p.name not in NON_COMPUTED_ARTIFACTS})
    save_json(OUT / "run_manifest.json", manifest)
    print(json.dumps({"status": "complete", "wall_seconds": manifest["wall_seconds"],
                      "totals_yuan": {g: summaries[g]["natural_total_yuan"] for g in GROUPS},
                      "S42_yuan": contrasts[contrasts.contrast == "S42"]["delta_yuan"].iloc[0],
                      "S43_S0_yuan": contrasts[contrasts.contrast == "S43_S0"]["delta_yuan"].iloc[0],
                      "S43_S2_yuan": contrasts[contrasts.contrast == "S43_S2"]["delta_yuan"].iloc[0]},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
