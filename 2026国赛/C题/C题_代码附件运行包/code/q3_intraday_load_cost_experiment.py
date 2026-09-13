#!/usr/bin/env python
"""问题三 日内负载修正与 q75 费用对照实验 —— 计算层。

任务书 ``reports/问题三/问题三_日内负载修正与q75费用对照实验方案.md``。

只回答一件事：**已保存的 F2 日内负载修正，在匹配自身 q75 保护及原调度规则后，
能否降低实际总现金费用。**

两组（固定，不搜索）
--------------------
* ``S0_L75``：前向档案 F0 负载（= 原 0:00 预测）＋原官方 Linear 光伏 ＋ 已保存的自身 W28/q75；
  **只读复用已核账本，全年重跑 0 次**；
* ``S2_load_q75``：前向档案 F2 负载（0 点原预测、6/12/18 点修正）＋同一 Linear 光伏 ＋
  **依据 F2 自身发布误差重建的 W28/q75**；本轮唯一新增的正式调度组（1336 次 MILP）。

所得差异度量的是「F2 负载修正 ＋ 相应残差保护 ＋ 调度响应」的组合效果，
**不能拆称纯当天信息价值，也不能判断 F2 一定比 F1 更节费**。

职责边界
--------
本脚本只做：登记 → 读冻结输入 → 纯数组重建 F0/F2 保护 → 三类精简检查 →
S2 全年四节点滚动调度 → 汇总与核账 → 落盘 CSV/NPZ/JSON。**不写报告、不画图**；
报告与图表由 ``code/q3_intraday_load_cost_report.py`` 只读渲染。

预算：**S2 正式 1336 次 MILP（334 个 0 点 ＋ 1002 个日内节点）＋ 适配检查最多 2 次**；
S0 全年重跑 0 次、LightGBM 训练 0 次、一元回归重拟合 0 次、光伏插值重建 0 次、
公共 1 月调度 0 次、参数扫描 0 次。保护重建是纯数组运算，**如实登记、不称 0 分位重建**。

关键口径
--------
* ``n̂^(g)[k,v,h] = (L̂^(g)[k,v,h] - V̂^Linear[k,v,h]) * Δt``，``n[k,h] = (L - V) * Δt``，
  ``ε = n - n̂``（kWh，不能重复乘 Δt）；
* 历史资格：``max(0,k-28) <= d < k`` 且 ``144d + h + 1 <= 144k + 6r_v``（目标区间已结束）；
* ``ρ = ε_(ceil(0.75m))``（升序次序统计量），``m < 7`` 时 ρ = 0，允许负保护量；
* F2 最少 14 日的**权重拟合**与 q75 最少 7 日的**保护历史**是两套机制，不得混用；
* 0 点计划覆盖当日 00:10 到次日 00:10；当前 00:00—00:10 执行前日承诺；
  6/12/18 点从当前实际库存更新、含同刻起始区间，剩余 109/73/37 段；
* 结算 ``C = Σ [p_t q_t^eff + 0.5 p_t |q_t^eff - q_t^0| + 5 p_t e_t]``，
  多次覆盖不累计重复调整费，午夜继承段用前一所属日原始承诺；
* 主账本 = 自然范围 48096 段，另保存 48097 段轨迹（含 2026-01-01 00:00—00:10 尾段），模板账本单列。

运行::

    E:/Anaconda/envs/math_modeling/python.exe code/q3_intraday_load_cost_experiment.py
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
import pandas as pd

# ======================================================================================
# §0  冻结规格
# ======================================================================================
ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code/q3_intraday_load_cost_experiment.py"
OUT = ROOT / "results/q3_intraday_load_cost"
FIG = ROOT / "figures/q3_intraday_load_cost"
PLAN_MD = ROOT / "reports/问题三/问题三_日内负载修正与q75费用对照实验方案.md"
REPORT_MD = ROOT / "reports/问题三/问题三_日内负载修正与q75费用对照实验结果.md"

RB_KERNEL = ROOT / "code/q3_rolling_baseline_experiment.py"
RB_DIR = ROOT / "results/q3_rolling_baseline"
RB_NPZ = RB_DIR / "prediction_protection.npz"
BIAS_DIR = ROOT / "results/q3_bias_correction_diagnostic"
BIAS_NPZ = BIAS_DIR / "bias_forecast_archive.npz"
FWD_DIR = ROOT / "results/q3_intraday_load_forward"
FWD_NPZ = FWD_DIR / "forecast_archive.npz"
FWD_REG = FWD_DIR / "registration.json"
FWD_MAN = FWD_DIR / "run_manifest.json"
COST_DIR = ROOT / "results/q3_bias_quantile_cost"
PROT75 = COST_DIR / "protection_q75.npz"
L75_DIR = COST_DIR / "L75"
COST_SUMMARY = COST_DIR / "summary.csv"
COST_REG = COST_DIR / "registration.json"
COST_MAN = COST_DIR / "run_manifest.json"

BASE = pd.Timestamp("2025-01-01")
NATURAL_DAYS = 365
TEMPLATE_SLOTS = 144
TARGETS = 145
DT = 1.0 / 6.0

ALPHA = 0.75                     # this round's single protection level
REFERENCE_ALPHA = 0.80           # the rolling-baseline kernel's frozen reference level
WINDOW = 28
MIN_SAMPLES = 7
UPDATE_HOURS = (0, 6, 12, 18)
FIRST_TARGET = {0: 0, 1: 36, 2: 72, 3: 108}
EVAL_FIRST, EVAL_LAST = 31, 364

SELECT_TOL_YUAN = 1e-4
ENERGY_TOL_KWH = 1e-6
CASH_TOL_YUAN = 1e-4
PROT_TOL = 1e-8                  # prediction / protection absolute tolerance, own units

PUBLIC_INITIAL_KWH = 6075.795025925926
PUBLIC_CARRY_KWH = 0.0

S0_REFERENCE = dict(
    natural_total_yuan=13353822.038919853, ordinary_cost_yuan=12771312.063726118,
    adjustment_cost_yuan=396263.274196023, emergency_cost_yuan=186246.700997711,
    emergency_kWh=36526.874321672)
NU = 0.8970555555555555          # the original cost comparison's inventory valuation

GROUPS = ("S0_L75", "S2_load_q75")
F0_KEY, F2_KEY = 0, 2            # indices into the forward archive's group axis
SPOT_DAY = 171                   # 2025-06-21, the shared boundary day
SELECTED_DATES = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")

STRATEGY_COLS = [
    "strategy_id", "interval_start", "interval_end", "owner_date", "effective_version",
    "price_yuan_kWh", "actual_load_kW", "actual_pv_kW", "net_kWh",
    "q0_kWh", "q_eff_kWh", "charge_kWh", "discharge_kWh", "emergency_kWh", "unused_kWh",
    "state_start_kWh", "state_end_kWh",
    "ordinary_cost_yuan", "adjustment_cost_yuan", "emergency_cost_yuan", "total_cost_yuan",
]
SCORE_COLS = ["ordinary_cost_yuan", "adjustment_cost_yuan", "emergency_cost_yuan"]

NON_COMPUTED_ARTIFACTS = frozenset({"run_manifest.json", "figure_integrity.json", "run.log"})

_M = {}


def rb():
    """Import the accepted rolling-baseline kernel as a read-only library (no main entry run)."""
    if "rb" not in _M:
        spec = importlib.util.spec_from_file_location("q3_rolling_baseline_kernel", RB_KERNEL)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.EVAL_FIRST == EVAL_FIRST and module.EVAL_LAST == EVAL_LAST
        assert module.WINDOW == WINDOW and module.MIN_SAMPLES == MIN_SAMPLES
        assert module.ALPHA == REFERENCE_ALPHA and module.UPDATE_HOURS == UPDATE_HOURS
        assert module.FIRST_TARGET == FIRST_TARGET and module.ENERGY_TOL_KWH == ENERGY_TOL_KWH
        assert module.SELECT_TOL_YUAN == SELECT_TOL_YUAN
        _M["rb"] = module
    return _M["rb"]


PARAMETERS = dict(
    experiment="Q3 intraday load correction with its own q75 protection: S0_L75 vs S2_load_q75",
    specification=str(PLAN_MD.relative_to(ROOT)),
    groups=dict(S0_L75="forward-archive F0 load (the original 0:00 forecast) + attachment-3 "
                       "Linear PV + the saved own W28/q75; read-only reuse of the audited ledger",
                S2_load_q75="forward-archive F2 load (0:00 original, revised at 6/12/18) + the "
                            "same Linear PV + a W28/q75 rebuilt from F2's own published errors"),
    identification="the difference measures the combined effect of the F2 load correction, its "
                   "matching residual protection and the dispatch response. It is NOT the isolated "
                   "value of same-day information and says nothing about F1.",
    net_forecast="n_hat[k,v,h] = (load_hat[k,v,h] - pv_linear[k,v,h]) * dt; n = (L - V) * dt; "
                 "eps = n - n_hat, all in kWh",
    history=dict(window_days=WINDOW, min_samples=MIN_SAMPLES,
                 rule="d in [max(0,k-28), k-1] with 144d + h + 1 <= 144k + 6*r_v, that "
                      "publication's prediction and the target truth valid",
                 quantile="ascending order statistic at position ceil(0.75 m); zero when m < 7; "
                          "negative protection retained",
                 separation="the 14-day weight fit of the F2 forecast and the 7-day q75 history "
                            "are different mechanisms and are never mixed; F2 uses its own stored "
                            "published values, never F0/F1 errors or current parameters backfilled"),
    engine="unchanged rolling kernel: power cap 5000 kW (5000/6 kWh per slot), state band "
           "1200-10800 kWh, charge/discharge efficiency 0.9 each, free terminal, no selling and no "
           "new penalty; the 0:00 plan minimises sum p q0, intraday revisions minimise "
           "sum [p q + 0.5 p |q - q0|]; accept a candidate only if its greedy score beats the "
           "previous plan by more than 1e-4 yuan",
    settlement="C = sum over the final effective purchase of [p q_eff + 0.5 p |q_eff - q0| + 5 p e] "
               "on the delivery-interval price; no repeated adjustment fee across overwrites; the "
               "midnight carry keeps the original commitment of its owning day",
    ledgers=dict(A=f"natural [2025-02-01 00:00, 2026-01-01 00:00), {48096} segments, main metric",
                 B="full 48097-segment trajectory including the 2026-01-01 00:00-00:10 tail, "
                   "not part of the main cost",
                 template="[2025-02-01 00:10, 2026-01-01 00:10), reported separately and bridged "
                          "by the head/tail difference; the two ledgers are never summed up as one"),
    selected_dates=list(SELECTED_DATES),
    inventory_valuation_yuan_per_kWh=NU,
    gate=(f"delta C = C_S2 - C_S0 on the natural ledger; |delta| <= {SELECT_TOL_YUAN:g} yuan is "
          "numerically tied; lower cash supports only 'cheaper in this one-year fixed "
          "configuration'. No automatic model replacement and no result3.xlsx write."),
    solver="HiGHS via scipy.optimize.milp, the same binary mutual-exclusion formulation and "
           "tolerances as the accepted kernel",
    validation="three light checks: input and protection boundary (0 MILP), two adaptation solves "
               "(2 MILP), persisted-ledger reconciliation (0 MILP)",
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


def known_max(values, label):
    arr = np.asarray(values, dtype=float)
    bad = ~np.isfinite(arr)
    if bad.any():
        raise AssertionError(f"{label}: {int(bad.sum())} non-finite value(s) of {arr.size}")
    return float(arr.max())


def comparable_max_abs(left, right, label):
    """NaN masks must agree exactly before comparing, so a missing cell cannot pass silently."""
    lp, rp = np.isnan(left), np.isnan(right)
    if not np.array_equal(lp, rp):
        raise AssertionError(f"{label}: {int((lp != rp).sum())} cell(s) with differing NaN mask")
    cells = int(np.isfinite(left).sum())
    if cells == 0:
        return 0.0, 0
    return float(np.abs(left[~lp] - right[~rp]).max()), cells


# ======================================================================================
# §2  输入：F0/F2 负载预测与自身 q75 保护
# ======================================================================================
def availability_mask():
    """``ok[s, v, h]``: lag ``s`` of target ``h`` is realised by publication (k, hour r_v).

    ``144d + h + 1 <= 144k + 6 r_v`` with ``s = k - d`` reduces to ``h + 1 - 6 r_v <= 144 s``,
    which depends only on ``(s, v, h)``. Identical construction to the accepted diagnostics.
    """
    h_grid = np.arange(TARGETS)
    ok = np.zeros((WINDOW + 1, len(UPDATE_HOURS), TARGETS), dtype=bool)
    for s in range(1, WINDOW + 1):
        for v, hour in enumerate(UPDATE_HOURS):
            ok[s, v] = (h_grid + 1 - 6 * hour) <= TEMPLATE_SLOTS * s
    return ok


def grouped_order_statistics(values, ok, alphas):
    """Grouped order statistic over lags 1..WINDOW in one pass, for several alpha levels."""
    n = values.shape[0]
    lagged = np.full((WINDOW, n) + values.shape[1:], np.nan)
    counts = np.zeros(values.shape, dtype=np.int64)
    for s in range(1, WINDOW + 1):
        if s >= n:
            break
        block = np.where(ok[s][None, :, :], values[:n - s], np.nan)
        lagged[s - 1, s:] = block
        counts[s:] += np.isfinite(block)
    ordered = np.sort(lagged, axis=0)
    del lagged
    out = {}
    for alpha in alphas:
        position = np.ceil(alpha * counts).astype(int)
        index = np.clip(position - 1, 0, WINDOW - 1)
        picked = np.take_along_axis(ordered, index[None, :, :, :], axis=0)[0]
        out[float(alpha)] = dict(rho=np.where(counts >= MIN_SAMPLES, picked, 0.0),
                                 position=position)
    return out, counts


def read_inputs():
    """Load the frozen forward-archive loads, Linear PV, truth and the saved L75 ledger."""
    with np.load(FWD_NPZ) as archive:
        forecast = archive["forecast"].astype(float)
        valid = archive["valid"]
        groups = [str(g) for g in archive["groups"]]
        versions = [int(v) for v in archive["versions"]]
    assert forecast.shape == (3, NATURAL_DAYS, 4, TARGETS), forecast.shape
    assert groups == ["F0_original", "F1_hist_mean", "F2_intraday_linear"], groups
    assert versions == [0, 6, 12, 18], versions
    with np.load(RB_NPZ) as baseline:
        pv_forecasts = baseline["pv_forecasts"].astype(float)
    with np.load(BIAS_NPZ) as bias:
        truth_load = bias["truth_load"].astype(float)
        truth_pv = bias["truth_pv"].astype(float)
        net_c0 = bias["net_c0"].astype(float)
    with np.load(PROT75) as saved:
        saved_protected_q75 = saved["linear_protected_q75"].astype(float)
        saved_rho_q75 = saved["linear_rho_q75"].astype(float)
        saved_counts = saved["linear_counts"].astype(np.int64)
        saved_alpha = float(saved["alpha_low"])
        saved_window = int(saved["window_days"])
        saved_min_samples = int(saved["min_samples"])
    assert abs(saved_alpha - ALPHA) < 1e-12 and saved_window == WINDOW
    assert saved_min_samples == MIN_SAMPLES

    truth_net = (truth_load - truth_pv) * DT
    load_f0 = forecast[F0_KEY]
    load_f2 = forecast[F2_KEY]
    net_f0 = (load_f0 - pv_forecasts) * DT
    net_f2 = (load_f2 - pv_forecasts) * DT

    # the rebuilt F0 net forecast must be the frozen Linear net forecast used by the saved ledgers
    f0_vs_saved = comparable_max_abs(net_f0, net_c0, "net_f0 vs bias-archive net_c0")
    # and the forward archive's F0 must be the frozen 0:00 load archive itself
    with np.load(RB_NPZ) as baseline:
        issued_load = baseline["issued_load"].astype(float)
    f0_vs_issued = comparable_max_abs(forecast[F0_KEY, :, 0, :], issued_load, "F0 vs issued_load")

    public = pd.read_csv(COST_DIR / "L75/dispatch.csv",
                         parse_dates=["interval_start", "interval_end"])
    assert len(public) == 48097, len(public)
    initial = float(public.state_start_kWh.iloc[0])
    carry = float(public.q_eff_kWh.iloc[0])
    carry_version = str(public.effective_version.iloc[0])
    carry_owner = str(public.owner_date.iloc[0])
    reference = pd.read_csv(COST_SUMMARY).set_index("strategy_id").loc["L75"]
    return dict(forecast=forecast, valid=valid, pv_forecasts=pv_forecasts,
                truth_load=truth_load, truth_pv=truth_pv, truth_net=truth_net,
                issued_load=issued_load, load_f0=load_f0, load_f2=load_f2,
                net_f0=net_f0, net_f2=net_f2,
                saved_protected_q75=saved_protected_q75, saved_rho_q75=saved_rho_q75,
                saved_counts=saved_counts,
                identity=dict(net_f0_vs_frozen_net_c0_max_abs_kWh=f0_vs_saved[0],
                              net_f0_compared_cells=f0_vs_saved[1],
                              f0_vs_issued_load_max_abs_kW=f0_vs_issued[0],
                              f0_vs_issued_compared_cells=f0_vs_issued[1]),
                public=public, initial=initial, carry=carry,
                carry_version=carry_version, carry_owner=carry_owner,
                reference_row=reference.to_dict())


def build_protection(data):
    """Rebuild the F0 and F2 q75 protections from the frozen arrays (pure array arithmetic)."""
    ok = availability_mask()
    eps = {"S0_L75": data["truth_net"][:, None, :] - data["net_f0"],
           "S2_load_q75": data["truth_net"][:, None, :] - data["net_f2"]}
    stats, counts, rho, protected = {}, {}, {}, {}
    for sid in GROUPS:
        stats[sid], counts[sid] = grouped_order_statistics(eps[sid], ok, (ALPHA,))
        rho[sid] = stats[sid][ALPHA]["rho"]
        net = data["net_f0"] if sid == "S0_L75" else data["net_f2"]
        protected[sid] = net + rho[sid]
    return dict(ok=ok, eps=eps, stats=stats, counts=counts, rho=rho, protected=protected)


# ======================================================================================
# §3  调度执行与统一账本
# ======================================================================================
def truth_reproduction(frame, truth_net):
    starts = pd.to_datetime(frame.interval_start)
    is_midnight = rb().midnight_mask(starts)
    k_pub = np.where(is_midnight, ((starts.dt.normalize() - BASE).dt.days).to_numpy() - 1,
                     ((starts.dt.normalize() - BASE).dt.days).to_numpy())
    h_pub = np.where(is_midnight, TEMPLATE_SLOTS,
                     ((starts - starts.dt.normalize()).dt.total_seconds() / 600).astype(int))
    expected = truth_net[k_pub, h_pub]
    actual = frame.net_kWh.to_numpy()
    defined = np.isfinite(expected) & np.isfinite(actual)
    value = float(np.abs(actual[defined] - expected[defined]).max()) if defined.any() else 0.0
    return value, int(defined.sum()), int((~defined).sum())


def require_truth_reproduction(frame, truth_net, label):
    value, compared, missing = truth_reproduction(frame, truth_net)
    if missing or compared != len(frame):
        raise AssertionError(f"{label}: {missing} non-finite truth cells, {compared} compared of "
                             f"{len(frame)}")
    if value >= ENERGY_TOL_KWH:
        raise AssertionError(f"{label}: stored net demand is not reproducible: {value:.3e}")
    return dict(max_abs_kWh=value, compared_cells=compared, nonfinite_cells=missing)


def finalize_group(sid, frame, truth_net, out_dir=None):
    """B-class assertions on the unified frame, plus the per-strategy ledger and summaries."""
    kernel = rb()
    frame = frame.copy()
    frame["strategy_id"] = sid
    for col in ("interval_start", "interval_end"):
        frame[col] = pd.to_datetime(frame[col])
    frame = frame.sort_values("interval_start").reset_index(drop=True)
    assert len(frame) == 48097, len(frame)
    assert not frame.interval_start.duplicated().any()
    assert frame.interval_start.diff().dropna().eq(pd.Timedelta(minutes=10)).all()
    frame["total_cost_yuan"] = (frame.ordinary_cost_yuan + frame.adjustment_cost_yuan
                               + frame.emergency_cost_yuan)
    settlement = kernel.settlement_residual(frame)
    if settlement >= SELECT_TOL_YUAN:
        raise AssertionError(f"{sid} settlement identity residual {settlement:.3e} yuan")
    physics = kernel.physics_residual(frame)
    truth_check = require_truth_reproduction(frame, truth_net, sid)

    natural, template = frame.iloc[:-1], frame.iloc[1:]
    assert len(natural) == 48096 and len(template) == 48096
    monthly = (natural.assign(month=natural.interval_start.dt.strftime("%Y-%m"))
               .groupby("month", as_index=False)
               [["ordinary_cost_yuan", "adjustment_cost_yuan", "emergency_cost_yuan",
                 "total_cost_yuan"]].sum())
    monthly.insert(0, "strategy_id", sid)
    daily = (natural.assign(date=natural.interval_start.dt.strftime("%Y-%m-%d"))
             .groupby("date", as_index=False)
             .agg(total_cost_yuan=("total_cost_yuan", "sum"),
                  ordinary_cost_yuan=("ordinary_cost_yuan", "sum"),
                  adjustment_cost_yuan=("adjustment_cost_yuan", "sum"),
                  emergency_cost_yuan=("emergency_cost_yuan", "sum"),
                  emergency_kWh=("emergency_kWh", "sum"),
                  unused_kWh=("unused_kWh", "sum"),
                  state_start_kWh=("state_start_kWh", "first"),
                  state_end_kWh=("state_end_kWh", "last")))
    daily.insert(0, "strategy_id", sid)

    summary = dict(
        strategy_id=sid,
        source=("read-only reuse of results/q3_bias_quantile_cost/L75"
                if sid == "S0_L75" else "new 334-day rolling run with F2 load and its own q75"),
        natural_total_yuan=float(natural.total_cost_yuan.sum()),
        template_total_yuan=float(template.total_cost_yuan.sum()),
        ordinary_cost_yuan=float(natural.ordinary_cost_yuan.sum()),
        adjustment_cost_yuan=float(natural.adjustment_cost_yuan.sum()),
        emergency_cost_yuan=float(natural.emergency_cost_yuan.sum()),
        emergency_kWh=float(natural.emergency_kWh.sum()),
        emergency_intervals=int((natural.emergency_kWh > ENERGY_TOL_KWH).sum()),
        emergency_days=int(natural.loc[natural.emergency_kWh > ENERGY_TOL_KWH, "interval_start"]
                           .dt.strftime("%Y-%m-%d").nunique()),
        emergency_events=int(kernel.emergency_events(natural)),
        unused_kWh=float(natural.unused_kWh.sum()),
        charge_kWh=float(natural.charge_kWh.sum()),
        discharge_kWh=float(natural.discharge_kWh.sum()),
        loss_kWh=float((1 - kernel.ETA) * natural.charge_kWh.sum()
                       + (1 / kernel.ETA - 1) * natural.discharge_kWh.sum()),
        initial_state_kWh=float(frame.state_start_kWh.iloc[0]),
        final_natural_state_kWh=float(natural.state_end_kWh.iloc[-1]),
        final_template_state_kWh=float(template.state_end_kWh.iloc[-1]),
        settlement_residual_yuan=settlement, physics_residual_kWh=physics,
        truth_residual_kWh=truth_check["max_abs_kWh"],
        truth_compared_cells=truth_check["compared_cells"])
    if out_dir is not None:
        frame_to_csv(frame[STRATEGY_COLS], Path(out_dir) / "dispatch.csv")
    return summary, monthly, daily, frame


def revision_summary(decisions, solver_log):
    accepted = decisions[decisions.accepted]
    out = dict(solves_total=int(len(solver_log)),
               solves_plan=int((solver_log.kind == "plan").sum()),
               solves_revision=int((solver_log.kind == "revision").sum()),
               revision_decisions=int(len(decisions)),
               revisions_accepted=int(len(accepted)),
               revisions_rejected=int(len(decisions) - len(accepted)),
               candidate_revision_kWh=float(decisions.revision_kWh.sum()),
               accepted_revision_kWh=float(accepted.revision_kWh.sum()),
               max_solver_gap=float(solver_log.mip_gap.max()),
               solver_seconds_total=float(solver_log.seconds.sum()),
               max_feasibility_violation_kWh=float(solver_log.max_violation_kWh.max()))
    for hour in UPDATE_HOURS[1:]:
        out[f"accepted_{hour:02d}"] = int((accepted.update_hour == hour).sum())
        out[f"decisions_{hour:02d}"] = int((decisions.update_hour == hour).sum())
    return out


# ======================================================================================
# §4  三类精简检查
# ======================================================================================
def check_inputs_and_protection(data, prot, l75_frame):
    """Class 1 (0 MILP): input identity, F0 rebuild vs the saved archive, F2's 0:00 equality."""
    out = {"identity": data["identity"]}
    assert out["identity"]["net_f0_vs_frozen_net_c0_max_abs_kWh"] <= PROT_TOL
    assert out["identity"]["f0_vs_issued_load_max_abs_kW"] <= PROT_TOL
    assert abs(data["initial"] - PUBLIC_INITIAL_KWH) < ENERGY_TOL_KWH, data["initial"]
    assert abs(data["carry"] - PUBLIC_CARRY_KWH) < ENERGY_TOL_KWH, data["carry"]
    out["public_initial"] = dict(initial_kWh=data["initial"], carry_kWh=data["carry"],
                                 carry_version=data["carry_version"],
                                 carry_owner=data["carry_owner"])

    # F0 rebuilt protection must equal the saved q75 archive cell by cell
    value, cells = comparable_max_abs(prot["protected"]["S0_L75"], data["saved_protected_q75"],
                                      "F0 rebuilt protected vs saved q75")
    rho_value, rho_cells = comparable_max_abs(prot["rho"]["S0_L75"], data["saved_rho_q75"],
                                              "F0 rebuilt rho vs saved q75")
    counts_equal = bool(np.array_equal(prot["counts"]["S0_L75"], data["saved_counts"]))
    if value > PROT_TOL or rho_value > PROT_TOL or not counts_equal:
        raise AssertionError(f"F0 protection rebuild mismatch: {value:.3e}, {rho_value:.3e}, "
                             f"counts_equal={counts_equal}")
    out["f0_protection_rebuild"] = dict(max_abs_protected_kWh=value,
                                        max_abs_rho_kWh=rho_value, cells=cells,
                                        counts_identical=counts_equal)

    # F2's 0:00 protection equals F0's: the 0:00 load forecast is unchanged
    value0, cells0 = comparable_max_abs(prot["protected"]["S2_load_q75"][:, 0, :],
                                       prot["protected"]["S0_L75"][:, 0, :], "F2 vs F0 at 0:00")
    if value0 > PROT_TOL:
        raise AssertionError(f"F2 0:00 protection differs from F0 by {value0:.3e}")
    out["f2_zero_version_equals_f0"] = dict(max_abs_kWh=value0, cells=cells0)
    for v in (1, 2, 3):
        reach = FIRST_TARGET[v]
        changed = float(np.nanmax(np.abs(prot["protected"]["S2_load_q75"][:, v, reach:]
                                         - prot["protected"]["S0_L75"][:, v, reach:])))
        out[f"intraday_version_{UPDATE_HOURS[v]:02d}_protection_max_abs_change_kWh"] = changed

    # a few saved history groups, picked independently of the rebuild
    spot = []
    for k, v, h in ((31, 1, 36), (SPOT_DAY, 0, 144), (SPOT_DAY, 3, 108)):
        history = []
        for d in range(max(0, k - WINDOW), k):
            if not (144 * d + h + 1 <= 144 * k + 6 * UPDATE_HOURS[v]):
                continue
            err = data["truth_net"][d, h] - data["net_f0"][d, v, h]
            if np.isfinite(err):
                history.append(err)
        m = len(history)
        position = int(math.ceil(ALPHA * m)) if m else 0
        expected = sorted(history)[position - 1] if m >= MIN_SAMPLES else 0.0
        got = float(prot["rho"]["S0_L75"][k, v, h])
        spot.append(dict(k=int(k), date=(BASE + timedelta(days=int(k))).strftime("%Y-%m-%d"),
                         publication_hour=UPDATE_HOURS[v], h=int(h), m=int(m),
                         position=position,
                         saved_position=int(prot["stats"]["S0_L75"][ALPHA]["position"][k, v, h]),
                         rho_kWh=float(expected), rebuilt_rho_kWh=got,
                         abs_difference_kWh=abs(expected - got)))
    worst = max(row["abs_difference_kWh"] for row in spot)
    if worst > PROT_TOL:
        raise AssertionError(f"hand-checked protection groups disagree: {worst:.3e}")
    out["spot_protection_groups"] = spot

    # 2025-06-21: perturbing not-yet-finished targets cannot move that instant's protection
    boundary = []
    for v, hour in ((0, 0), (1, 6)):
        k = SPOT_DAY
        shifted_truth = data["truth_net"].copy()
        for h in range(TARGETS):
            if 144 * k + h + 1 > 144 * k + 6 * hour:          # target still open at this instant
                shifted_truth[k, h] += 25.0
        shifted_eps = shifted_truth[:, None, :] - data["net_f0"]
        rebuilt, _ = grouped_order_statistics(shifted_eps, prot["ok"], (ALPHA,))
        after = data["net_f0"][k, v, :] + rebuilt[ALPHA]["rho"][k, v, :]
        before = prot["protected"]["S0_L75"][k, v, :]
        finite = np.isfinite(before)
        boundary.append(dict(publication_hour=hour,
                             max_abs_change_kWh=float(np.abs(after[finite] - before[finite]).max())))
    worst_boundary = max(row["max_abs_change_kWh"] for row in boundary)
    if worst_boundary > PROT_TOL:
        raise AssertionError(f"protection boundary broken: {worst_boundary:.3e}")
    out["boundary_perturbation"] = dict(
        day="2025-06-21", perturb_kWh=25.0, rows=boundary,
        max_abs_change_kWh=worst_boundary,
        rule="only targets whose interval has NOT ended at the publication instant are shifted; "
             "that instant's protection must not move. The forecast causality itself was verified "
             "in the previous round and is not refitted here.")
    del l75_frame
    return out


def check_adaptation(price_src, protected_s0, l75_frame, plan_versions):
    """Class 2 (2 MILP): replay the saved L75 2025-06-21 decisions through the same code path."""
    kernel = rb()
    k, day = SPOT_DAY, (BASE + timedelta(days=SPOT_DAY)).strftime("%Y-%m-%d")
    out = dict(day=day, k=int(k), extra_milp_solves=2)
    rows = l75_frame[l75_frame.interval_start.dt.strftime("%Y-%m-%d") == day].sort_values(
        "interval_start")
    midnight = rows[kernel.midnight_mask(rows.interval_start)].iloc[0]
    state_before, carry = float(midnight.state_start_kWh), float(midnight.q_eff_kWh)

    _, _, _, _, estimated = kernel.feedback_step(carry - float(protected_s0[k, 0, 0]), state_before)
    q_plan, _, _, _, _, summary = kernel.solve_interval(
        protected_s0[k, 0, 1:TARGETS], price_src, estimated, context=f"adaptation {day} 0:00")
    saved_plan = l75_frame[(l75_frame.interval_start > pd.Timestamp(day))
                           & (l75_frame.interval_start <= pd.Timestamp(day) + pd.Timedelta(hours=24))
                           ].sort_values("interval_start")
    assert len(saved_plan) == TEMPLATE_SLOTS, len(saved_plan)
    saved_q0 = saved_plan.q0_kWh.to_numpy()
    saved_plan_cost = float(price_src @ saved_q0)
    out["midnight"] = dict(
        auxiliary_state_estimate_kWh=float(estimated), saved_plan_cost_yuan=saved_plan_cost,
        resolved_plan_cost_yuan=float(summary["objective_yuan"]),
        objective_delta_yuan=float(abs(summary["objective_yuan"] - saved_plan_cost)),
        plan_max_delta_kWh=known_max_abs(q_plan - saved_q0, "adaptation 0:00 plan delta"),
        max_feasibility_violation_kWh=float(summary["max_violation_kWh"]),
        note="objective equality is the gate; a nonzero plan delta would only reflect another "
             "equal-cost vertex of the same MILP")
    if out["midnight"]["objective_delta_yuan"] >= SELECT_TOL_YUAN:
        raise AssertionError(f"adaptation 0:00 objective mismatch: {out['midnight']}")

    v, h0 = 1, FIRST_TARGET[1]
    state_0600 = float(rows.state_end_kWh.iloc[h0 - 1])
    record = plan_versions[(plan_versions.decision == "revision")
                           & (plan_versions.published_at == f"{day} 06:00")].copy()
    record["interval_start"] = pd.to_datetime(record.interval_start)
    record = record.sort_values("interval_start")
    assert len(record) == TARGETS - h0, len(record)
    assert (record.previous_version == f"{day} 00:00").all()
    q0_seg = record.q0_kWh.to_numpy()
    old_seg = record.previous_kWh.to_numpy()
    saved_candidate = record.candidate_kWh.to_numpy()
    saved_accepted = bool(record.accepted.iloc[0])
    price_seg, remaining = price_src[h0 - 1:], protected_s0[k, v, h0:TARGETS]
    candidate, _, _, _, _, cand_summary = kernel.solve_interval(
        remaining, price_seg, state_0600, q0_seg, context=f"adaptation {day} 06:00")
    old_score = float(sum(kernel.score_components(old_seg, q0_seg, price_seg, remaining, state_0600)))
    new_score = float(sum(kernel.score_components(candidate, q0_seg, price_seg, remaining, state_0600)))
    accepted = bool(new_score < old_score - SELECT_TOL_YUAN)
    out["six"] = dict(
        update_instant_state_kWh=state_0600,
        old_score_yuan=old_score, new_score_yuan=new_score,
        score_improvement_yuan=float(old_score - new_score),
        saved_candidate_score_yuan=float(
            sum(kernel.score_components(saved_candidate, q0_seg, price_seg, remaining, state_0600))),
        saved_accepted=saved_accepted, replayed_accepted=accepted,
        candidate_max_delta_kWh=known_max_abs(candidate - saved_candidate,
                                              "adaptation 06:00 candidate delta"),
        max_feasibility_violation_kWh=float(cand_summary["max_violation_kWh"]),
        note="the old effective plan is the saved `previous` column of the plan archive, never a "
             "slice of the final ledger")
    if accepted != saved_accepted:
        raise AssertionError(f"adaptation 06:00 accept decision differs: {out['six']}")
    return out


def check_saved_results(paths, price_src, truth_net, decisions_by_group):
    """Class 3 (0 MILP): one read-only reconciliation of the persisted artifacts."""
    out = dict(note="read back from the persisted CSV artifacts, not from the in-memory frames")
    for sid in GROUPS:
        dispatch = pd.read_csv(paths[sid]["dispatch"], parse_dates=["interval_start", "interval_end"])
        solver_log = pd.read_csv(paths[sid]["solver_log"])
        decisions = decisions_by_group[sid]
        check = dict(rows=int(len(dispatch)))
        assert len(dispatch) == 48097, len(dispatch)
        assert dispatch.interval_start.diff().dropna().eq(pd.Timedelta(minutes=10)).all()
        numeric = ["q0_kWh", "q_eff_kWh", "charge_kWh", "discharge_kWh", "emergency_kWh",
                   "unused_kWh", "state_start_kWh", "state_end_kWh", "price_yuan_kWh", "net_kWh",
                   "ordinary_cost_yuan", "adjustment_cost_yuan", "emergency_cost_yuan",
                   "total_cost_yuan"]
        check["nonfinite_dispatch_cells"] = int(
            (~np.isfinite(dispatch[numeric].to_numpy(dtype=float))).sum())
        if check["nonfinite_dispatch_cells"]:
            raise AssertionError(f"{sid}: {check['nonfinite_dispatch_cells']} non-finite ledger "
                                 "cells; pandas min/max would have skipped them silently")
        for col, low, high in (("q_eff_kWh", 0.0, None), ("charge_kWh", 0.0, 5000 / 6),
                               ("discharge_kWh", 0.0, 5000 / 6), ("emergency_kWh", 0.0, None),
                               ("unused_kWh", 0.0, None), ("state_start_kWh", 1200.0, 10800.0),
                               ("state_end_kWh", 1200.0, 10800.0)):
            values = dispatch[col]
            assert values.min() >= low - ENERGY_TOL_KWH, (sid, col, float(values.min()))
            if high is not None:
                assert values.max() <= high + ENERGY_TOL_KWH, (sid, col, float(values.max()))
        balance = (dispatch.q_eff_kWh + dispatch.discharge_kWh + dispatch.emergency_kWh
                   - dispatch.net_kWh - dispatch.charge_kWh - dispatch.unused_kWh)
        recurrence = (dispatch.state_end_kWh - dispatch.state_start_kWh
                      - 0.9 * dispatch.charge_kWh + dispatch.discharge_kWh / 0.9)
        continuity = (dispatch.state_start_kWh.to_numpy()[1:]
                      - dispatch.state_end_kWh.to_numpy()[:-1])
        truth_check = require_truth_reproduction(dispatch, truth_net, sid)
        check["simultaneous_charge_discharge_kWh"] = known_max_abs(
            np.minimum(dispatch.charge_kWh, dispatch.discharge_kWh),
            f"{sid} simultaneous charge/discharge")
        check["bus_balance_max_abs_kWh"] = known_max_abs(balance, f"{sid} bus balance")
        check["battery_recurrence_max_abs_kWh"] = known_max_abs(recurrence, f"{sid} recurrence")
        check["state_continuity_max_abs_kWh"] = known_max_abs(continuity, f"{sid} continuity")
        check["truth_reproduction_max_abs_kWh"] = truth_check["max_abs_kWh"]
        check["truth_compared_cells"] = truth_check["compared_cells"]
        if max(check["simultaneous_charge_discharge_kWh"], check["state_continuity_max_abs_kWh"],
               check["truth_reproduction_max_abs_kWh"]) >= ENERGY_TOL_KWH:
            raise AssertionError(f"{sid} persisted-ledger energy checks failed: {check}")
        # independent re-computation of the three cash items from the stated rules
        price = dispatch.price_yuan_kWh.to_numpy()
        rebuilt = {
            "ordinary": price * dispatch.q_eff_kWh.to_numpy(),
            "adjustment": 0.5 * price * np.abs(dispatch.q_eff_kWh.to_numpy()
                                               - dispatch.q0_kWh.to_numpy()),
            "emergency": 5.0 * price * dispatch.emergency_kWh.to_numpy()}
        for side, values in rebuilt.items():
            worst_cash = known_max_abs(values - dispatch[f"{side}_cost_yuan"].to_numpy(),
                                       f"{sid} {side} cash")
            check[f"{side}_cash_max_abs_yuan"] = worst_cash
            assert worst_cash < CASH_TOL_YUAN, (sid, side, worst_cash)
        total_rebuilt = float(dispatch.total_cost_yuan.sum())
        parts = float((dispatch.ordinary_cost_yuan + dispatch.adjustment_cost_yuan
                       + dispatch.emergency_cost_yuan).sum())
        check["cash_parts_vs_total_yuan"] = abs(total_rebuilt - parts)
        assert check["cash_parts_vs_total_yuan"] < CASH_TOL_YUAN
        natural, template = dispatch.iloc[:-1], dispatch.iloc[1:]
        check["natural_total_yuan"] = float(natural.total_cost_yuan.sum())
        check["template_total_yuan"] = float(template.total_cost_yuan.sum())
        check["bridge_yuan"] = float(template.total_cost_yuan.sum()
                                     - natural.total_cost_yuan.sum())
        check["max_solver_gap"] = float(solver_log.mip_gap.max())
        # solve_interval raises on a failed HiGHS call, so completeness is the check here: one row
        # per solve with a finite objective and a finite non-negative gap
        check["solver_rows"] = int(len(solver_log))
        check["nonfinite_objectives"] = int((~np.isfinite(solver_log.objective_yuan)).sum())
        check["negative_gaps"] = int((solver_log.mip_gap < 0).sum())
        if check["solver_rows"] != 1336 or check["nonfinite_objectives"] or check["negative_gaps"]:
            raise AssertionError(f"{sid} solver log incomplete: {check}")
        # the accept rule is `new score < old score - 1e-4`, i.e. delta_predicted_yuan < -1e-4
        violations = decisions.accepted & (decisions.delta_predicted_yuan >= -SELECT_TOL_YUAN)
        misses = (~decisions.accepted) & (decisions.delta_predicted_yuan < -SELECT_TOL_YUAN)
        check["accept_rule_violations"] = int(violations.sum())
        check["accept_rule_misses"] = int(misses.sum())
        check["min_improvement_yuan_of_accepted"] = float(
            (-decisions.loc[decisions.accepted, "delta_predicted_yuan"]).min())
        check["max_improvement_yuan_of_rejected"] = float(
            (-decisions.loc[~decisions.accepted, "delta_predicted_yuan"]).max())
        assert check["accept_rule_violations"] == 0 and check["accept_rule_misses"] == 0, check
        out[sid] = check
    del price_src
    return out


# ======================================================================================
# §5  汇总
# ======================================================================================
def build_contrasts(summaries, monthly, daily):
    s0, s2 = summaries["S0_L75"], summaries["S2_load_q75"]
    rows = []
    for label, key in (("total", "natural_total_yuan"), ("ordinary", "ordinary_cost_yuan"),
                       ("adjustment", "adjustment_cost_yuan"),
                       ("emergency", "emergency_cost_yuan")):
        delta = float(s2[key] - s0[key])
        rows.append(dict(item=label, s0_yuan=float(s0[key]), s2_yuan=float(s2[key]),
                         delta_yuan=delta,
                         delta_share_of_total=delta / s0["natural_total_yuan"]))
    # the three components must add up to the total difference (rows[1:4], not rows[:3])
    parts = sum(row["delta_yuan"] for row in rows[1:4])
    if abs(parts - rows[0]["delta_yuan"]) >= CASH_TOL_YUAN:
        raise AssertionError(f"cash components do not sum to the total: {parts} vs "
                             f"{rows[0]['delta_yuan']}")
    base = float(s0["natural_total_yuan"])
    out = pd.DataFrame(rows)
    out.attrs["components_sum_to_total_yuan"] = abs(parts - rows[0]["delta_yuan"])
    monthly_wide = monthly.pivot_table(index="month", columns="strategy_id",
                                       values="total_cost_yuan")
    monthly_wide["delta_yuan"] = monthly_wide["S2_load_q75"] - monthly_wide["S0_L75"]
    monthly_wide = monthly_wide.reset_index()
    daily_wide = daily.pivot_table(index="date", columns="strategy_id", values="total_cost_yuan")
    daily_wide["delta_yuan"] = daily_wide["S2_load_q75"] - daily_wide["S0_L75"]
    daily_wide = daily_wide.reset_index()
    daily_wide["emergency_kWh_S2"] = daily.set_index(["date", "strategy_id"]).emergency_kWh.xs(
        "S2_load_q75", level="strategy_id").reindex(daily_wide.date).to_numpy()
    ranked = daily_wide.reindex(daily_wide.delta_yuan.abs().sort_values(ascending=False).index)
    worst = daily_wide.sort_values("delta_yuan", ascending=False)
    out.attrs["days"] = dict(
        refreshed=int((daily_wide.delta_yuan < -SELECT_TOL_YUAN).sum()),
        worsened=int((daily_wide.delta_yuan > SELECT_TOL_YUAN).sum()),
        tied=int((daily_wide.delta_yuan.abs() <= SELECT_TOL_YUAN).sum()),
        months_refreshed=int((monthly_wide.delta_yuan < -SELECT_TOL_YUAN).sum()),
        months_worsened=int((monthly_wide.delta_yuan > SELECT_TOL_YUAN).sum()),
        largest_absolute=[(r.date, float(r.delta_yuan)) for r in ranked.head(3).itertuples()],
        largest_increase=[(r.date, float(r.delta_yuan)) for r in worst.head(3).itertuples()],
        best_day=f"{daily_wide.loc[daily_wide.delta_yuan.idxmin(), 'date']}:"
                 f"{daily_wide.delta_yuan.min():+.6f}",
        delta_median_yuan=float(daily_wide.delta_yuan.median()),
        baseline_total_yuan=base)
    return out, monthly_wide, daily_wide


def build_protection_change(prot, data):
    """Signed/absolute changes of net forecast, rho and protected, per publication version."""
    rows = []
    for quantity, key in (("net_forecast", "net"), ("rho", "rho"), ("protected", "protected")):
        if key == "net":
            left, right = data["net_f2"], data["net_f0"]
        else:
            left, right = prot[key]["S2_load_q75"], prot[key]["S0_L75"]
        for v, hour in enumerate(UPDATE_HOURS):
            reach = FIRST_TARGET[v]
            for scope, sl in (("reach", slice(reach, TARGETS)),
                              ("tail", slice(TARGETS - 1, TARGETS))):
                a = left[:, v, sl]
                b = right[:, v, sl]
                ok = np.isfinite(a) & np.isfinite(b)
                diff = (a - b)[ok]
                rows.append(dict(quantity=quantity, publication_hour=hour, scope=scope,
                                 n=int(diff.size),
                                 signed_mean=float(diff.mean()) if diff.size else float("nan"),
                                 mean_abs=float(np.abs(diff).mean()) if diff.size else float("nan"),
                                 max_abs=float(np.abs(diff).max()) if diff.size else float("nan"),
                                 mean_left=float(a[ok].mean()) if diff.size else float("nan"),
                                 mean_right=float(b[ok].mean()) if diff.size else float("nan")))
    return pd.DataFrame(rows)


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
        # carry the amendment trail forward even on an idempotent re-run
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

    inputs = ["reports/问题三/问题三_日内负载修正与q75费用对照实验方案.md",
              "results/q3_intraday_load_forward/forecast_archive.npz",
              "results/q3_intraday_load_forward/registration.json",
              "results/q3_intraday_load_forward/run_manifest.json",
              "results/q3_rolling_baseline/prediction_protection.npz",
              "results/q3_bias_correction_diagnostic/bias_forecast_archive.npz",
              "results/q3_bias_quantile_cost/protection_q75.npz",
              "results/q3_bias_quantile_cost/summary.csv",
              "results/q3_bias_quantile_cost/registration.json",
              "results/q3_bias_quantile_cost/run_manifest.json",
              "results/q3_bias_quantile_cost/L75/dispatch.csv",
              "results/q3_bias_quantile_cost/L75/plan_versions.csv",
              "results/q3_bias_quantile_cost/L75/revision_decisions.csv",
              "results/q3_bias_quantile_cost/L75/solver_log.csv",
              "results/q3_bias_quantile_cost/L75/nominal_trajectory.csv",
              "code/q3_rolling_baseline_experiment.py"]
    for rel in inputs:
        assert (ROOT / rel).exists(), rel
    dependency = dict(
        forward_archive_signature=load_json(FWD_REG)["signature"],
        forward_archive_status=load_json(FWD_MAN)["status"],
        cost_experiment_signature=load_json(COST_REG)["signature"],
        cost_experiment_status=load_json(COST_MAN)["status"],
        kernel="code/q3_rolling_baseline_experiment.py imported as a read-only library",
        zero_budget=dict(s0_full_year_reruns=0, lightgbm_training=0, weight_refits=0,
                         pv_rebuild=0, public_january_dispatch=0, parameter_sweeps=0),
        note="the protection rebuild is pure array arithmetic and is NOT reported as zero work")
    signature = register(PARAMETERS, inputs, dependency, args.amend_reason)
    print(f"registered signature={signature[:16]}", flush=True)
    if args.mode == "register":
        return

    kernel = rb()
    tick = time.perf_counter()
    data = read_inputs()
    price_src = None
    attachments = kernel.read_attachments()
    price_src = attachments["price_src"]
    l75_frame = pd.read_csv(L75_DIR / "dispatch.csv",
                            parse_dates=["interval_start", "interval_end"])
    timings["inputs_seconds"] = time.perf_counter() - tick
    print(f"inputs: L75 {len(l75_frame)} rows, initial {data['initial']:.6f} kWh, carry "
          f"{data['carry']:.6f} kWh, F0 net identity "
          f"{data['identity']['net_f0_vs_frozen_net_c0_max_abs_kWh']:.3e} kWh", flush=True)

    tick = time.perf_counter()
    prot = build_protection(data)
    np.savez_compressed(
        OUT / "protection.npz",
        groups=np.array(GROUPS), alpha=ALPHA, window_days=WINDOW, min_samples=MIN_SAMPLES,
        availability=prot["ok"],
        net_forecast_f0=data["net_f0"], net_forecast_f2=data["net_f2"],
        error_f0=prot["eps"]["S0_L75"], error_f2=prot["eps"]["S2_load_q75"],
        rho_f0=prot["rho"]["S0_L75"], rho_f2=prot["rho"]["S2_load_q75"],
        counts_f0=prot["counts"]["S0_L75"], counts_f2=prot["counts"]["S2_load_q75"],
        protected_f0=prot["protected"]["S0_L75"], protected_f2=prot["protected"]["S2_load_q75"],
        position_f0=prot["stats"]["S0_L75"][ALPHA]["position"],
        position_f2=prot["stats"]["S2_load_q75"][ALPHA]["position"],
        semantics=np.array([
            "axis0 = publication version 0:00/6:00/12:00/18:00; axis1 = day index k; "
            "axis2 = target h = 0..144; NaN outside a version's reach or where invalid; "
            "rho = ascending order statistic at ceil(0.75 m), zero when m < 7"]))
    timings["protection_seconds"] = time.perf_counter() - tick
    print("F0 / F2 protection rebuilt and saved", flush=True)

    tick = time.perf_counter()
    checks = {"inputs_and_protection": check_inputs_and_protection(data, prot, l75_frame)}
    timings["check_class1_seconds"] = time.perf_counter() - tick
    print("class-1 input / protection checks passed (0 MILP)", flush=True)

    tick = time.perf_counter()
    l75_plans = pd.read_csv(L75_DIR / "plan_versions.csv", parse_dates=["interval_start"])
    checks["adaptation"] = check_adaptation(price_src, prot["protected"]["S0_L75"], l75_frame,
                                            l75_plans)
    timings["check_class2_seconds"] = time.perf_counter() - tick
    print(f"class-2 adaptation checks passed (2 MILP): 0:00 objective delta "
          f"{checks['adaptation']['midnight']['objective_delta_yuan']:.3e} yuan, 06:00 accept "
          f"decision {checks['adaptation']['six']['replayed_accepted']}", flush=True)

    # ---- S0: read-only reconciliation of the saved L75 ledger ----
    s0_summary, s0_monthly, s0_daily, s0_frame = finalize_group("S0_L75", l75_frame, data["truth_net"])
    for key, expected in S0_REFERENCE.items():
        delta = abs(float(s0_summary[key]) - expected)
        if delta >= max(CASH_TOL_YUAN, ENERGY_TOL_KWH):
            raise AssertionError(f"S0 {key} differs from the registered reference by {delta:.3e}")
    print(f"S0 reconciled with the registered L75 ledger: "
          f"{s0_summary['natural_total_yuan']:,.6f} yuan", flush=True)

    # ---- S2: the only new formal scheduling group ----
    tick = time.perf_counter()
    s2_dir = OUT / "S2_load_q75"
    s2_dir.mkdir(parents=True, exist_ok=True)
    frame, plans, decisions, solves, nominal = kernel.run_group(
        "B2", prot["protected"]["S2_load_q75"], data["truth_load"], data["truth_pv"],
        data["truth_net"], price_src, kernel.natural_price(price_src),
        data["initial"], data["carry"], PUBLIC_CARRY_KWH, data["carry_version"], data["carry_owner"])
    s2_summary, s2_monthly, s2_daily, s2_frame = finalize_group("S2_load_q75", frame,
                                                                data["truth_net"], s2_dir)
    frame_to_csv(plans, s2_dir / "plan_versions.csv")
    frame_to_csv(decisions, s2_dir / "revision_decisions.csv")
    frame_to_csv(solves, s2_dir / "solver_log.csv")
    frame_to_csv(nominal, s2_dir / "nominal_trajectory.csv")
    revision = revision_summary(decisions, solves)
    if revision["solves_total"] != 1336 or revision["solves_revision"] != 1002:
        raise AssertionError(f"S2 solve bookkeeping: {revision}")
    timings["S2_seconds"] = time.perf_counter() - tick
    print(f"S2 finished: natural {s2_summary['natural_total_yuan']:,.2f} yuan, emergency "
          f"{s2_summary['emergency_kWh']:,.1f} kWh, accepted "
          f"{revision['revisions_accepted']}/{len(decisions)} revisions, "
          f"{timings['S2_seconds']:.1f}s", flush=True)

    paths = {"S0_L75": dict(dispatch=L75_DIR / "dispatch.csv",
                            solver_log=L75_DIR / "solver_log.csv",
                            plan_versions=L75_DIR / "plan_versions.csv"),
             "S2_load_q75": dict(dispatch=s2_dir / "dispatch.csv",
                                 solver_log=s2_dir / "solver_log.csv",
                                 plan_versions=s2_dir / "plan_versions.csv")}
    s0_decisions = pd.read_csv(L75_DIR / "revision_decisions.csv")
    decisions_by_group = {"S0_L75": s0_decisions, "S2_load_q75": decisions}

    tick = time.perf_counter()
    checks["saved_results"] = check_saved_results(paths, price_src, data["truth_net"],
                                                 decisions_by_group)
    timings["check_class3_seconds"] = time.perf_counter() - tick
    print("class-3 persisted-ledger reconciliation passed (0 MILP)", flush=True)

    summaries = {"S0_L75": s0_summary, "S2_load_q75": s2_summary}
    for sid, revision_info in (("S0_L75", revision_summary(s0_decisions,
                                                           pd.read_csv(L75_DIR / "solver_log.csv"))),
                               ("S2_load_q75", revision)):
        summaries[sid].update(revision_info)
        summaries[sid]["inventory_value_coefficient_yuan_per_kWh"] = NU
        summaries[sid]["cost_net_of_inventory_yuan"] = (
            summaries[sid]["natural_total_yuan"]
            - NU * (summaries[sid]["final_natural_state_kWh"]
                    - summaries[sid]["initial_state_kWh"]))
    summary_frame = pd.DataFrame([summaries[sid] for sid in GROUPS])
    monthly_long = pd.concat([s0_monthly, s2_monthly], ignore_index=True)
    daily_long = pd.concat([s0_daily, s2_daily], ignore_index=True)
    contrasts, monthly_wide, daily_wide = build_contrasts(summaries, monthly_long, daily_long)
    protection_change = build_protection_change(prot, data)

    selected = pd.concat([
        (s0_frame if sid == "S0_L75" else s2_frame)
        .assign(strategy_id=sid, date=lambda f: f.interval_start.dt.strftime("%Y-%m-%d"))
        .query("date in @SELECTED_DATES") for sid in GROUPS], ignore_index=True)
    selected = selected.groupby(["strategy_id", "date"], as_index=False).agg(
        ordinary_cost_yuan=("ordinary_cost_yuan", "sum"),
        adjustment_cost_yuan=("adjustment_cost_yuan", "sum"),
        emergency_cost_yuan=("emergency_cost_yuan", "sum"),
        total_cost_yuan=("total_cost_yuan", "sum"),
        emergency_kWh=("emergency_kWh", "sum"), charge_kWh=("charge_kWh", "sum"),
        discharge_kWh=("discharge_kWh", "sum"))

    frame_to_csv(summary_frame, OUT / "summary.csv")
    frame_to_csv(contrasts, OUT / "contrasts.csv")
    frame_to_csv(monthly_wide, OUT / "monthly.csv")
    frame_to_csv(daily_wide, OUT / "daily.csv")
    frame_to_csv(protection_change, OUT / "protection_change.csv")
    frame_to_csv(selected, OUT / "selected_dates.csv")

    delta_total = float(s2_summary["natural_total_yuan"] - s0_summary["natural_total_yuan"])
    if abs(delta_total) <= SELECT_TOL_YUAN:
        verdict = "两组自然账本总费在数值上持平，不支持 F2 组合在费用上更优"
    elif delta_total < 0:
        verdict = (f"S2 比 S0 低 {-delta_total:,.2f} 元：在本年度此固定配置下费用更低。"
                   f"这是「F2 负载修正＋相应残差保护＋调度响应」的组合效果，"
                   f"不能拆称纯当天信息价值，也不称最优")
    else:
        verdict = (f"S2 比 S0 高 {delta_total:,.2f} 元：预测 MAE 的改善没有转化为现金改善，"
                   f"保留原 L75，结束本轮")

    validation = dict(
        status="passed", signature=signature, checks=checks,
        solves=dict(S2_formal=1336, adaptation=2, S0_full_year_reruns=0),
        protection_rebuild="pure array arithmetic over the frozen archives; counted as 2 group "
                           "rebuilds, not as zero work",
        delta_total_yuan=delta_total, verdict=verdict,
        days=contrasts.attrs["days"],
        zero_budget=dependency["zero_budget"],
        limitations=[
            "所得差异是「F2 负载修正＋相应残差保护＋调度响应」的组合效果，不是纯当天信息价值，"
            "也不能判断 F2 一定比 F1 更节费。",
            "点预测 MAE 改善不代表费用改善；本轮只报现金与风险指标，不反过来由 MAE 推费用。",
            "2025 年已参与方法设计，这是同年后验回测，不是跨年盲测，也不是独立样本外验证。",
            "应急风险与现金收益必须并列披露；现金已含 5 倍应急费，不额外发明题面没有的风险否决约束。",
            "末期库存的 ν=0.8970555555555555 元/kWh 只是沿用原费用对照的辅助估值敏感性，"
            "不是真实终端市场价值，也不改现金主指标。",
            "1,002 条午夜尾槽记录对应 334 个不同实际午夜区间、被三个版本各预测一次，"
            "不能当成三次实际发生；保留 S2 全年版本预测，不按本轮诊断事后把尾槽换回 F0。",
            "没有任何图件经过目视核查（本 Agent 无法查看图像），只做程序化完整性检查。"])
    save_json(OUT / "validation.json", validation)
    manifest = dict(status="complete", started_utc=started_utc, finished_utc=utc_now(),
                    wall_seconds=time.perf_counter() - started, timings=timings,
                    signature=signature, executable=sys.executable, python=sys.version,
                    zero_solves=dict(s0_full_year_reruns=0, lightgbm_training=0, weight_refits=0,
                                     pv_rebuild=0, public_january_dispatch=0, parameter_sweeps=0),
                    milp=dict(S2_formal=1336, adaptation=2),
                    inputs_read=sorted(inputs),
                    outputs={p.relative_to(OUT).as_posix(): digest(p) for p in sorted(OUT.rglob("*"))
                             if p.is_file() and p.name not in NON_COMPUTED_ARTIFACTS})
    save_json(OUT / "run_manifest.json", manifest)
    print(json.dumps({"status": "complete", "wall_seconds": manifest["wall_seconds"],
                      "S0_natural_yuan": s0_summary["natural_total_yuan"],
                      "S2_natural_yuan": s2_summary["natural_total_yuan"],
                      "delta_total_yuan": delta_total,
                      "delta_share": delta_total / s0_summary["natural_total_yuan"],
                      "verdict": verdict}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
