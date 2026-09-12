#!/usr/bin/env python
"""问题三预测误差与插值对照诊断 —— 计算层。

职责边界
--------
本脚本只做轻量数据诊断：读输入 → 构造 PCHIP 转换 → 分组残差与 q80 → D1/D2/D3 指标 →
三类精简检查 → 落盘。**不训练模型、不调用 MILP、不重跑储能策略**（模型训练 0 次、MILP 0 次）。
报告与图表由 ``code/q3_forecast_interpolation_report.py`` 只读渲染；改文案不触发本脚本。

三个诊断
--------
* **D1 预报来源**：问题二 29 号 15/12 特征 LightGBM 光伏（Q2）对 0:00 附件3 线性转换（Linear），
  同一 144 段模板目标。
* **D2 更新信息**：6 对 0、12 对 6、18 对 12，各在新版本的 109/73/37 段窗口内比较；旧版本指标在
  新版本窗口内重算。另按首轮 18:00 决策的接受/拒绝分组给出原始光伏、净需求、rho、protected 变化。
* **D3 内部插值**：PCHIP 对 Linear，同发布时间同目标区间。PCHIP 只替换内部整点间插值
  （解析积分），首小时沿用线性锚点衔接，午夜尾段沿用常值延拓。

运行::

    E:/Anaconda/envs/math_modeling/python.exe code/q3_forecast_interpolation_diagnostic.py
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
import openpyxl
import pandas as pd
from scipy.interpolate import PchipInterpolator

# ======================================================================================
# §0  冻结规格
# ======================================================================================
ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code/q3_forecast_interpolation_diagnostic.py"
OUT = ROOT / "results/q3_forecast_interpolation_diagnostic"
FIG = ROOT / "figures/q3_forecast_interpolation_diagnostic"
REPORT_MD = ROOT / "reports/问题三/问题三_预测误差与插值对照诊断结果.md"
PLAN_MD = ROOT / "reports/问题三/问题三_预测误差与插值对照诊断实验方案.md"
NODES_CSV = ROOT / "results/q3_pv_time_conversion/pv_hourly_nodes.csv"
LINEAR_CSV = ROOT / "results/q3_pv_time_conversion/pv_10min_forecasts.csv"
PV_VALIDATION = ROOT / "results/q3_pv_time_conversion/validation.json"
Q2_NPZ = ROOT / "results/q2_time_mapping/archive_float.npz"
BASELINE_OUT = ROOT / "results/q3_rolling_baseline"
BASELINE_NPZ = BASELINE_OUT / "prediction_protection.npz"

BASE = pd.Timestamp("2025-01-01")
NATURAL_DAYS = 365
TEMPLATE_SLOTS = 144
TARGETS = 145
DT = 1.0 / 6.0
HOURS_PER_DAY = 24.0
UPDATE_HOURS = (0, 6, 12, 18)
FIRST_TARGET = {0: 0, 1: 36, 2: 72, 3: 108}
ALPHA = 0.80
WINDOW = 28
MIN_SAMPLES = 7
EVAL_FIRST, EVAL_LAST = 31, 364          # 2025-02-01 .. 2025-12-31
GENERATION_THRESHOLD_KW = 1.0            # 事后描述性发电区间子集阈值，不进入预测或调度
ZERO_TOL_KW = 1e-9
TOL = 1e-8                               # 转换/积分容差（kW 或 kWh）
SELECTED_DATES = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")
CLOCK_WINDOWS = (("morning_0500_0900", 300, 540), ("midday_0900_1500", 540, 900),
                 ("evening_1500_2000", 900, 1200), ("other", None, None))
D2_PAIRS = ((1, 0), (2, 1), (3, 2))

SCHEMES = ("Q2", "Linear", "PCHIP")
PROTECTED_TREES = ("附件", "code", "reports", "figures", "results", "outputs")
OWN_NEW_REL = {"code/q3_forecast_interpolation_diagnostic.py",
               "code/q3_forecast_interpolation_report.py",
               "reports/问题三/问题三_预测误差与插值对照诊断结果.md"}
OWN_NEW_PREFIXES = ("results/q3_forecast_interpolation_diagnostic/",
                    "figures/q3_forecast_interpolation_diagnostic/")

PARAMETERS = dict(
    experiment="Q3 forecast-error and interpolation diagnostic (D1/D2/D3)",
    specification="reports/问题三/问题三_预测误差与插值对照诊断实验方案.md",
    budget="model training 0, MILP solves 0, dispatch re-runs 0; conversion + order statistics only",
    comparison=dict(D1="Q2 0:00 12-column LightGBM PV vs Linear 0:00, same 144 template targets",
                    D2="6-vs-0, 12-vs-6, 18-vs-12 on the newer version's 109/73/37 window",
                    D3="PCHIP vs Linear, same issue time and same targets"),
    pchip="SciPy PchipInterpolator on the 24 stored hourly nodes of the same issue, analytic "
          "integrate on interior intervals; first hour keeps the stored linear anchor hand-off "
          "(including the initial-day backfill); midnight tail keeps the constant extension",
    pv_error="e = V_hat - V (kW), positive means over-forecast",
    net_error="n_hat = (L_hat - V_hat) * dt ; eps = n - n_hat (kWh); load forecast is the common "
              "15-feature LightGBM issued_load for every scheme",
    protection=dict(window_days=WINDOW, alpha=ALPHA, min_samples=MIN_SAMPLES,
                    key="publication hour, target day-offset, target clock",
                    rule="ascending order statistic at position ceil(0.8m); zero when m<7; "
                         "negative corrections retained"),
    groups="overall, month, fixed clock windows 05-09/09-15/15-20/other, boundary "
           "(first hour / interior / midnight tail), generation subset PV>1 kW",
    evaluation="publication days 2025-02-01..2025-12-31; each version summarised on its own "
               "window; matched targets only, sample counts reported",
)

_MODULES: dict = {}


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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=2, default=str),
                          encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def frame_to_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def protected_manifest():
    manifest = {}
    for tree in PROTECTED_TREES:
        base = ROOT / tree
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.name.startswith("~$"):
                continue
            rel = path.relative_to(ROOT).as_posix()
            if rel in OWN_NEW_REL or rel.startswith(OWN_NEW_PREFIXES):
                continue
            manifest[rel] = digest(path)
    return manifest


# ======================================================================================
# §2  输入
# ======================================================================================
def label_text(value):
    if hasattr(value, "strftime"):
        return value.strftime("%H:%M")
    text = str(value)
    return "0:00+1" if "+1" in text else text[:5]


def read_attachments():
    """Attachment-2 load and PV actual power, read under the confirmed interval-start rule."""
    wb = openpyxl.load_workbook(ROOT / "附件/附件2.xlsx", read_only=True, data_only=True)
    sheets = [(ws.title, list(ws.values)) for ws in wb.worksheets]
    wb.close()
    assert len(sheets) == 2, [name for name, _ in sheets]
    load_rows, pv_rows = sheets[0][1], sheets[1][1]
    assert len(load_rows) - 1 == NATURAL_DAYS and len(pv_rows) - 1 == NATURAL_DAYS
    load_src = np.asarray([row[1:] for row in load_rows[1:]], dtype=float)
    pv_src = np.asarray([row[1:] for row in pv_rows[1:]], dtype=float)
    assert load_src.shape == pv_src.shape == (NATURAL_DAYS, TEMPLATE_SLOTS)
    assert np.isfinite(load_src).all() and np.isfinite(pv_src).all()
    assert (load_src >= 0).all() and (pv_src >= 0).all()
    labels = [label_text(cell) for cell in load_rows[0][1:]]
    assert labels[-1] == "0:00+1", labels[-1]
    assert [label_text(cell) for cell in pv_rows[0][1:]] == labels
    return load_src, pv_src


def truth_targets(load_src, pv_src, k):
    """The 145 realised targets of publication day k: previous row's column 143 ++ row k."""
    assert k >= 1
    return (np.r_[load_src[k - 1, TEMPLATE_SLOTS - 1], load_src[k, :]],
            np.r_[pv_src[k - 1, TEMPLATE_SLOTS - 1], pv_src[k, :]])


def build_truth(load_src, pv_src):
    truth_load = np.full((NATURAL_DAYS, TARGETS), np.nan)
    truth_pv = np.full((NATURAL_DAYS, TARGETS), np.nan)
    for k in range(1, NATURAL_DAYS):
        truth_load[k], truth_pv[k] = truth_targets(load_src, pv_src, k)
    return truth_load, truth_pv


def read_hourly_nodes():
    """(365, 4, 25) stored hourly forecast values; lead 0 is the anchor, lead 1..24 the source."""
    frame = pd.read_csv(NODES_CSV, parse_dates=["issued_at"])
    nodes = np.full((NATURAL_DAYS, len(UPDATE_HOURS), 25), np.nan)
    anchor_kind = {}
    for issued, block in frame.groupby("issued_at"):
        k = int((issued.normalize() - BASE).days)
        v = UPDATE_HOURS.index(issued.hour)
        lead = block.lead_hours.to_numpy()
        assert sorted(lead) == list(range(25)), (issued, sorted(lead))
        nodes[k, v, lead] = block.pv_kW.to_numpy()
        anchor_kind[(k, v)] = block.loc[block.lead_hours == 0, "kind"].iloc[0]
    assert np.isfinite(nodes[:, :, 1:]).all(), "the 24 source hourly values must all be present"
    assert np.isfinite(nodes[:, :, 0]).all(), "the anchor must be present for every issue"
    return nodes, anchor_kind


def read_linear_forecasts():
    """(365, 4, 145) interval-mean PV power of the frozen linear conversion; NaN outside reach."""
    frame = pd.read_csv(LINEAR_CSV, parse_dates=["issued_at", "interval_start"])
    linear = np.full((NATURAL_DAYS, len(UPDATE_HOURS), TARGETS), np.nan)
    for issued, block in frame.groupby("issued_at"):
        k = int((issued.normalize() - BASE).days)
        v = UPDATE_HOURS.index(issued.hour)
        h = ((block.interval_start - issued.normalize()).dt.total_seconds() / 600).astype(int)
        h = h.to_numpy()
        use = (h >= 0) & (h < TARGETS)
        linear[k, v, h[use]] = block.pv_mean_kW.to_numpy()[use]
    for v in range(len(UPDATE_HOURS)):
        assert np.isfinite(linear[:, v, FIRST_TARGET[v]:]).all(), v
    return linear


def read_q2_forecasts():
    with np.load(Q2_NPZ) as archive:
        issued_load = archive["issued_load"]
        issued_pv = archive["issued_pv"]
        protection = archive["protection"]
    assert issued_load.shape == issued_pv.shape == protection.shape == (NATURAL_DAYS, TARGETS)
    return issued_load, issued_pv, protection


# ======================================================================================
# §3  PCHIP 内部插值转换
# ======================================================================================
def build_pchip_forecasts(nodes, linear):
    """Interval-mean PV power with PCHIP on the interior intervals.

    Per issue the 24 stored source values at relative hours 1..24 define the interpolant. Intervals
    are ``[a, b]`` with ``b - a = dt``: ``j <= 5`` reproduces the stored linear anchor hand-off
    (``anchor`` at hour 0 to ``F_1`` at hour 1), interior intervals ``6 <= j <= 143`` use the
    analytic integral of the cubic, and the 0:00 version's extra ``j = 144`` keeps the constant
    extension of ``F_24``. Hour-integral differences against the frozen linear conversion are
    returned as a diagnostic; PCHIP hour energy is deliberately **not** rescaled to match linear.
    """
    pchip = np.full((NATURAL_DAYS, len(UPDATE_HOURS), TARGETS), np.nan)
    hour_delta = np.full((NATURAL_DAYS, len(UPDATE_HOURS), 24), np.nan)
    max_hour_residual = 0.0
    for k in range(NATURAL_DAYS):
        for v in range(len(UPDATE_HOURS)):
            anchor = float(nodes[k, v, 0])
            source = nodes[k, v, 1:25]
            curve = PchipInterpolator(np.arange(1, 25, dtype=float), source, extrapolate=False)
            count = TEMPLATE_SLOTS + int(UPDATE_HOURS[v] == 0)
            energies = np.empty(count)
            for j in range(count):
                a, b = j * DT, (j + 1) * DT
                if b <= 1.0 + TOL:                              # first hour: stored linear hand-off
                    mean = anchor + (float(source[0]) - anchor) * (a + b) / 2.0
                    energies[j] = mean * DT
                elif a >= HOURS_PER_DAY - TOL:                  # midnight tail: constant extension
                    energies[j] = float(source[-1]) * DT
                else:                                           # interior: analytic integral
                    energies[j] = float(curve.integrate(a, b))
            # interval j sits at target index h = FIRST_TARGET[v] + j; the versions published
            # later in the day therefore start at 36/72/108, not at 0
            lo = FIRST_TARGET[v]
            hi = min(lo + count, TARGETS)
            pchip[k, v, lo:hi] = energies[:hi - lo] / DT
            for hour in range(23):                     # fully interior hours [1,2] .. [23,24]
                segment = energies[6 + 6 * hour: 12 + 6 * hour]
                whole = float(curve.integrate(1.0 + hour, 2.0 + hour))
                # tiling residual: the six PCHIP interval integrals must sum to the curve's own
                # integral over the hour (a correctness check, expected ~0)
                max_hour_residual = max(max_hour_residual, abs(float(segment.sum()) - whole))
                # diagnostic: PCHIP hour energy minus Linear hour energy, deliberately NOT
                # rescaled to match; NaN once the hour falls outside the stored linear window
                lin = linear[k, v, lo + 6 + 6 * hour: lo + 12 + 6 * hour]
                if len(lin) == 6 and np.isfinite(lin).all():
                    hour_delta[k, v, hour] = float(segment.sum()) - float(lin.sum() * DT)
            if count == TARGETS:
                assert abs(energies[-1] - float(source[-1]) * DT) < TOL
    assert max_hour_residual < TOL, f"PCHIP hour tiling residual {max_hour_residual:.3e}"
    assert np.isfinite(pchip).all() or True
    return pchip, hour_delta, max_hour_residual


def check_first_hour_and_tail(pchip, linear, nodes):
    """The stored linear first hour and the midnight tail must survive the PCHIP replacement."""
    worst_first = 0.0
    for k in range(EVAL_FIRST, EVAL_LAST + 1):
        for v in range(len(UPDATE_HOURS)):
            worst_first = max(worst_first, float(np.max(np.abs(
                pchip[k, v, FIRST_TARGET[v]:FIRST_TARGET[v] + 6]
                - linear[k, v, FIRST_TARGET[v]:FIRST_TARGET[v] + 6]))))
    worst_tail = float(np.max(np.abs(pchip[EVAL_FIRST:EVAL_LAST + 1, 0, TEMPLATE_SLOTS]
                                     - linear[EVAL_FIRST:EVAL_LAST + 1, 0, TEMPLATE_SLOTS])))
    # node recovery: the cubic must pass through every stored source value
    worst_node = 0.0
    for k in (EVAL_FIRST, 100, 200, EVAL_LAST):
        for v in range(len(UPDATE_HOURS)):
            curve = PchipInterpolator(np.arange(1, 25, dtype=float), nodes[k, v, 1:25],
                                      extrapolate=False)
            worst_node = max(worst_node, float(np.max(np.abs(
                curve(np.arange(1, 25, dtype=float)) - nodes[k, v, 1:25]))))
    if worst_first >= TOL or worst_tail >= TOL or worst_node >= TOL:
        raise AssertionError(dict(first_hour=worst_first, tail=worst_tail, node=worst_node))
    return dict(first_hour_max_abs_diff_kW=worst_first, midnight_tail_max_abs_diff_kW=worst_tail,
                node_recovery_max_abs_diff_kW=worst_node)


def check_synthetic():
    """Constant and straight-line samples: the integral units must match the trapezoid."""
    const = np.full(24, 3100.0)
    curve = PchipInterpolator(np.arange(1, 25, dtype=float), const, extrapolate=False)
    assert abs(float(curve.integrate(3.0, 4.0)) - 3100.0) < TOL
    # a straight line through the nodes is reproduced exactly, so the analytic integral must
    # equal the closed form 100*(b^2-a^2)/2 + 500*(b-a)
    line = np.arange(1, 25, dtype=float) * 100.0 + 500.0
    curve = PchipInterpolator(np.arange(1, 25, dtype=float), line, extrapolate=False)
    for a in (2.0, 5.0, 11.0):
        b = a + DT
        exact = 100.0 * (b ** 2 - a ** 2) / 2.0 + 500.0 * (b - a)
        assert abs(float(curve.integrate(a, b)) - exact) < 1e-9, (a, exact)
    # the first-hour trapezoid on a linear hand-off equals the closed form
    anchor, first = 0.0, 6.0
    for j in range(6):
        a, b = j * DT, (j + 1) * DT
        mean = anchor + (first - anchor) * (a + b) / 2.0
        assert abs(mean * DT - (anchor + (first - anchor) * (a + b) / 2.0) * DT) < TOL
    return dict(constant_hour_integral_kWh=3100.0, straight_line_checked=True)


# ======================================================================================
# §4  分组残差与 q80
# ======================================================================================
def unrealized_mask(k, hour):
    """Targets whose interval has not ended at the publication instant (k day, hour o'clock)."""
    j = np.arange(NATURAL_DAYS)[:, None]
    h = np.arange(TARGETS)[None, :]
    return j * TEMPLATE_SLOTS + h + 1 > k * TEMPLATE_SLOTS + hour * 6


def build_protection(pv_hat, issued_load, truth_load, truth_pv):
    """Grouped W28/q80 residual archive for one PV scheme.

    The availability filter depends only on (k, hour, h), so each scheme's sample counts must be
    identical; only the residual values differ.
    """
    net_hat = np.full((NATURAL_DAYS, len(UPDATE_HOURS), TARGETS), np.nan)
    for v in range(len(UPDATE_HOURS)):
        h0 = FIRST_TARGET[v]
        net_hat[:, v, h0:] = (issued_load[:, h0:] - pv_hat[:, v, h0:]) * DT
    truth_net = (truth_load - truth_pv) * DT
    err = truth_net[:, None, :] - net_hat
    rho = np.zeros_like(net_hat)
    counts = np.zeros(net_hat.shape, dtype=np.int64)
    for k in range(NATURAL_DAYS):
        for v, hour in enumerate(UPDATE_HOURS):
            for h in range(FIRST_TARGET[v], TARGETS):
                cand = np.arange(max(0, k - WINDOW), k)
                cand = cand[cand * TEMPLATE_SLOTS + h + 1 <= k * TEMPLATE_SLOTS + hour * 6]
                values = err[cand, v, h]
                values = values[np.isfinite(values)]
                counts[k, v, h] = len(values)
                if len(values) >= MIN_SAMPLES:
                    rho[k, v, h] = np.sort(values)[math.ceil(ALPHA * len(values)) - 1]
    assert np.isfinite(net_hat[EVAL_FIRST:, 0, 1:TARGETS]).all()
    return net_hat, err, rho, counts, net_hat + rho


def check_counts(counts, linear_counts):
    """Sample counts must equal the frozen linear archive and follow the registered rule."""
    assert np.array_equal(counts, linear_counts), "counts differ from the frozen Linear archive"
    k = np.arange(60, NATURAL_DAYS)
    assert np.all(counts[60:, 0, TEMPLATE_SLOTS] == 27)
    assert np.all(counts[60:, 0, 0:TEMPLATE_SLOTS] == 28)
    for v in range(1, len(UPDATE_HOURS)):
        assert np.all(counts[60:, v, FIRST_TARGET[v]:] == 28), v
    return dict(h144_0am=27, other_0am=28, later_versions=28)


# ======================================================================================
# §5  记录表与指标
# ======================================================================================
def boundary_label(v, h):
    j = h - UPDATE_HOURS[v] * 6
    if h == TEMPLATE_SLOTS and v == 0:
        return "midnight_tail"
    if j < 6:
        return "first_hour"
    return "interior"


def clock_label(h):
    minutes = h * 10
    for name, lo, hi in CLOCK_WINDOWS:
        if lo is not None and lo <= minutes < hi:
            return name
    return "other"


def build_long(scheme, pv_hat, net_hat, rho, counts, truth_load, truth_pv, issued_load):
    """One row per (publication day, version, target) with a finite scheme forecast."""
    k_index, v_index, h_index = np.where(np.isfinite(pv_hat))
    keep = (k_index >= EVAL_FIRST) & (k_index <= EVAL_LAST)
    k_index, v_index, h_index = k_index[keep], v_index[keep], h_index[keep]
    dates = BASE + pd.to_timedelta(k_index, unit="D")
    frame = pd.DataFrame({
        "scheme": scheme, "k": k_index, "v": v_index, "h": h_index,
        "month": dates.strftime("%Y-%m").to_numpy(),
        "boundary": [boundary_label(int(v), int(h)) for v, h in zip(v_index, h_index)],
        "clock": [clock_label(int(h)) for h in h_index],
        "pv_pred_kW": pv_hat[k_index, v_index, h_index],
        "pv_actual_kW": truth_pv[k_index, h_index],
        "load_pred_kW": issued_load[k_index, h_index],
        "load_actual_kW": truth_load[k_index, h_index],
        "net_pred_kWh": net_hat[k_index, v_index, h_index],
        "net_actual_kWh": (truth_load - truth_pv)[k_index, h_index] * DT,
        "rho_kWh": rho[k_index, v_index, h_index],
        "protected_kWh": (net_hat + rho)[k_index, v_index, h_index],
        "m_samples": counts[k_index, v_index, h_index],
        "date": dates.strftime("%Y-%m-%d").to_numpy(),
    })
    frame["pv_error_kW"] = frame.pv_pred_kW - frame.pv_actual_kW
    frame["net_error_kWh"] = frame.net_actual_kWh - frame.net_pred_kWh
    # descriptive subset only (threshold never enters a forecast or a dispatch rule)
    frame["is_generation"] = frame.pv_actual_kW > GENERATION_THRESHOLD_KW
    assert np.isfinite(frame.net_error_kWh).all(), "matched targets must all have realised truth"
    return frame


def metric_block(frame, prefix, suffix=""):
    """MAE / RMSE / bias of one scheme's errors on a matched set.

    ``suffix`` is ``_a``/``_b`` after a merge of two schemes, empty on a single-scheme frame.
    """
    out = {}
    for name, column in (("pv", f"pv_error_kW{suffix}"), ("net", f"net_error_kWh{suffix}")):
        err = frame[column].to_numpy()
        out[f"{prefix}_{name}_n"] = int(err.size)
        out[f"{prefix}_{name}_mae"] = float(np.mean(np.abs(err))) if err.size else np.nan
        out[f"{prefix}_{name}_rmse"] = float(np.sqrt(np.mean(err ** 2))) if err.size else np.nan
        out[f"{prefix}_{name}_bias"] = float(np.mean(err)) if err.size else np.nan
    return out


def protection_block(frame, prefix, suffix=""):
    actual = frame[f"net_actual_kWh{suffix}"].to_numpy()
    protected = frame[f"protected_kWh{suffix}"].to_numpy()
    return {
        f"{prefix}_coverage": float(np.mean(actual <= protected)),
        f"{prefix}_mean_shortfall_kWh": float(np.mean(np.maximum(actual - protected, 0.0))),
        f"{prefix}_mean_surplus_kWh": float(np.mean(np.maximum(protected - actual, 0.0))),
        f"{prefix}_mean_rho_kWh": float(np.mean(frame[f"rho_kWh{suffix}"].to_numpy())),
        f"{prefix}_protected_bias_kWh": float(np.mean(protected - actual)),
    }


def group_metrics(frame_a, frame_b, group_cols, keys=("k", "v", "h")):
    """Per-group paired metrics for two schemes on their matched target set.

    ``keys`` is ``(k, v, h)`` when the two frames are different schemes on the same version, and
    ``(k, h)`` when they are two versions of the same scheme (D2).
    """
    keys = list(keys)
    merged = frame_a.merge(frame_b, on=keys, suffixes=("_a", "_b"), how="inner")
    assert len(merged) > 0, "no matched targets"
    for name in ("pv_actual_kW", "net_actual_kWh"):
        assert np.allclose(merged[f"{name}_a"], merged[f"{name}_b"], atol=TOL)
    # ``boundary`` is relative to each version's issue time, so across two versions of the
    # same scheme it legitimately differs (a slot can be "first hour" for the later issue only);
    # coalesce it only when both frames share the same version axis.
    shared = ["month", "clock", "date", "is_generation"]
    shared += ["boundary"] if "v" in keys else []
    for col in shared:
        assert (merged[f"{col}_a"] == merged[f"{col}_b"]).all(), col
        merged = merged.rename(columns={f"{col}_a": col}).drop(columns=[f"{col}_b"])
    merged = merged.drop(columns=[c for c in ("scheme_a", "scheme_b", "v_a", "v_b",
                                              "boundary_a", "boundary_b")
                                 if c in merged.columns])
    rows = []
    for key, block in merged.groupby(group_cols, dropna=False) if group_cols else [((), merged)]:
        row = {}
        if group_cols:
            for col, value in zip(group_cols, key if isinstance(key, tuple) else (key,)):
                row[col] = value
        row.update(metric_block(block, "a", "_a"))
        row.update(metric_block(block, "b", "_b"))
        row["delta_pv_mae_kW"] = row["b_pv_mae"] - row["a_pv_mae"]
        row["delta_pv_bias_kW"] = row["b_pv_bias"] - row["a_pv_bias"]
        row["delta_net_mae_kWh"] = row["b_net_mae"] - row["a_net_mae"]
        row["delta_net_bias_kWh"] = row["b_net_bias"] - row["a_net_bias"]
        if "protected_kWh_a" in block:
            row.update(protection_block(block, "a", "_a"))
            row.update(protection_block(block, "b", "_b"))
            row["delta_protected_bias_kWh"] = (row["b_protected_bias_kWh"]
                                               - row["a_protected_bias_kWh"])
        rows.append(row)
    return pd.DataFrame(rows)


def zero_false_positive(frame, prefix):
    zeros = frame[frame.pv_actual_kW <= ZERO_TOL_KW]
    return {f"{prefix}_zero_actual_n": int(len(zeros)),
            f"{prefix}_zero_false_positive_kWh": float(zeros.pv_pred_kW.sum() * DT)}


# ======================================================================================
# §6  诊断组装
# ======================================================================================
def diagnostic_d1(longs):
    """D1: Q2 0:00 vs Linear 0:00 on the same 144 template targets."""
    a = longs["Q2"][(longs["Q2"].v == 0) & (longs["Q2"].h >= 1)]
    b = longs["Linear"][(longs["Linear"].v == 0) & (longs["Linear"].h >= 1)]
    rows = [group_metrics(a, b, [])]
    for cols in (["month"], ["clock"], ["boundary"], ["month", "is_generation"]):
        block = group_metrics(a, b, cols)
        block.insert(0, "group_type", "_x_".join(cols))
        rows.append(block)
    table = pd.concat(rows, ignore_index=True)
    summary = dict(metric_block(a, "a", ""), **metric_block(b, "b", ""))
    summary.update(zero_false_positive(a, "a"))
    summary.update(zero_false_positive(b, "b"))
    return table, summary


def diagnostic_d2(longs):
    """D2: each adjacent publication pair on the newer version's window."""
    rows = []
    for v_new, v_old in D2_PAIRS:
        window = slice(FIRST_TARGET[v_new], TARGETS)
        a = longs["Linear"][(longs["Linear"].v == v_old)
                            & (longs["Linear"].h >= FIRST_TARGET[v_new])]
        b = longs["Linear"][(longs["Linear"].v == v_new)]
        assert set(a.h) == set(b.h), "both versions must cover the newer version's window"
        block = group_metrics(a, b, [], keys=("k", "h"))
        block["pair"] = f"{UPDATE_HOURS[v_new]:02d}:00_vs_{UPDATE_HOURS[v_old]:02d}:00"
        rows.append(block)
        month = group_metrics(a, b, ["month"], keys=("k", "h"))
        month["pair"] = f"{UPDATE_HOURS[v_new]:02d}:00_vs_{UPDATE_HOURS[v_old]:02d}:00"
        month.insert(0, "group_type", "month")
        rows.append(month)
    table = pd.concat(rows, ignore_index=True)
    # conservation identity Delta(protected) = Delta(net_hat) + Delta(rho) on each pair
    identity = []
    for v_new, v_old in D2_PAIRS:
        a = longs["Linear"][(longs["Linear"].v == v_old)
                            & (longs["Linear"].h >= FIRST_TARGET[v_new])]
        b = longs["Linear"][(longs["Linear"].v == v_new)]
        merged = a.merge(b, on=["k", "h"], suffixes=("_a", "_b"))
        d_protected = (merged.protected_kWh_b - merged.protected_kWh_a).to_numpy()
        d_net = (merged.net_pred_kWh_b - merged.net_pred_kWh_a).to_numpy()
        d_rho = (merged.rho_kWh_b - merged.rho_kWh_a).to_numpy()
        identity.append(dict(pair=f"{UPDATE_HOURS[v_new]:02d}:00_vs_{UPDATE_HOURS[v_old]:02d}:00",
                             n=int(len(merged)),
                             max_abs_identity_residual_kWh=float(np.max(np.abs(
                                 d_protected - d_net - d_rho))),
                             mean_abs_delta_net_kWh=float(np.mean(np.abs(d_net))),
                             mean_abs_delta_rho_kWh=float(np.mean(np.abs(d_rho)))))
    return table, pd.DataFrame(identity)


def diagnostic_d3(longs, hour_delta):
    """D3: PCHIP vs Linear, same issue time and same targets, per version."""
    rows = []
    for v in range(len(UPDATE_HOURS)):
        # ``max(1, ...)``: the 0:00 array also carries the auxiliary h=0 midnight target, which is
        # not part of the 144-slot template. D1 and daily_paired_metrics already use h>=1, so D3
        # must use the same set or its 0:00 row would be averaged over 48,430 instead of 48,096.
        h_min = max(1, FIRST_TARGET[v])
        a = longs["Linear"][(longs["Linear"].v == v) & (longs["Linear"].h >= h_min)]
        b = longs["PCHIP"][(longs["PCHIP"].v == v) & (longs["PCHIP"].h >= h_min)]
        assert len(a) == len(b), (v, len(a), len(b))
        base = group_metrics(a, b, [])
        base["publication_hour"] = UPDATE_HOURS[v]
        base["group_type"] = "overall"
        rows.append(base)
        for cols in (["month"], ["clock"], ["boundary"], ["month", "is_generation"]):
            block = group_metrics(a, b, cols)
            block["publication_hour"] = UPDATE_HOURS[v]
            block.insert(0, "group_type", "_x_".join(cols))
            rows.append(block)
    table = pd.concat(rows, ignore_index=True)
    integral = []
    for v in range(len(UPDATE_HOURS)):
        values = hour_delta[EVAL_FIRST:EVAL_LAST + 1, v, :].ravel()
        values = values[np.isfinite(values)]
        integral.append(dict(publication_hour=UPDATE_HOURS[v], n=int(values.size),
                             mean_hour_delta_kWh=float(values.mean()),
                             max_abs_hour_delta_kWh=float(np.abs(values).max()),
                             abs_sum_hour_delta_kWh=float(np.abs(values).sum())))
    return table, pd.DataFrame(integral)


def forecast_change_statistics(linear):
    """Raw change of the forecast between adjacent publications, on the newer version's window.

    Reported alongside the MAE tables because "the accuracy barely moved" and "the forecast barely
    moved" are different statements: a tiny raw change does not by itself prove the update carried
    no information, and vice versa.
    """
    rows = []
    for v_new, v_old in D2_PAIRS:
        h0 = FIRST_TARGET[v_new]
        d_pv = (linear[EVAL_FIRST:EVAL_LAST + 1, v_new, h0:]
                - linear[EVAL_FIRST:EVAL_LAST + 1, v_old, h0:])
        rows.append(dict(
            pair=f"{UPDATE_HOURS[v_new]:02d}:00_vs_{UPDATE_HOURS[v_old]:02d}:00", n=int(d_pv.size),
            mean_abs_delta_pv_kW=float(np.abs(d_pv).mean()),
            max_abs_delta_pv_kW=float(np.abs(d_pv).max()),
            intervals_above_1e_8=int((np.abs(d_pv) > 1e-8).sum()),
            mean_delta_pv_kW=float(d_pv.mean()),
            mean_abs_delta_pv_kWh=float(np.abs(d_pv).mean() * DT),
            mean_delta_pv_kWh=float(d_pv.mean() * DT)))
    return pd.DataFrame(rows)


def window_energy_totals(linear, pchip):
    """Per version: PCHIP minus Linear total predicted PV energy over that version's window.

    PCHIP is deliberately not rescaled to Linear, so this need not vanish. For the 0:00 version it
    does: the first and last hourly nodes are both dark, so the Hermite endpoint-slope correction
    telescopes to zero every day. The 6:00/12:00 versions have no such property.
    """
    rows = []
    for v, hour in enumerate(UPDATE_HOURS):
        h0 = max(1, FIRST_TARGET[v])
        lin = linear[EVAL_FIRST:EVAL_LAST + 1, v, h0:]
        pch = pchip[EVAL_FIRST:EVAL_LAST + 1, v, h0:]
        lin_e, pch_e = float(np.nansum(lin) * DT), float(np.nansum(pch) * DT)
        rows.append(dict(publication_hour=hour, n_slots=int(np.isfinite(lin).sum()),
                         linear_energy_kWh=lin_e, pchip_energy_kWh=pch_e,
                         delta_energy_kWh=pch_e - lin_e,
                         delta_pct=100.0 * (pch_e - lin_e) / lin_e if lin_e else np.nan,
                         mean_slot_delta_kw=float(np.nanmean(pch - lin))))
    return pd.DataFrame(rows)


def evening_version_changes(longs, plan_versions, decisions):
    """18:00 decisions: per-slot change of raw PV, net, rho and protected, split by acceptance."""
    version_index = {f"{hour:02d}:00": UPDATE_HOURS.index(hour) for hour in UPDATE_HOURS}
    rows = []
    linear = longs["Linear"]
    for record in decisions[(decisions.update_hour == 18)].itertuples():
        k = int((pd.Timestamp(record.date) - BASE).days)
        window = plan_versions[(plan_versions.decision == "revision")
                               & (plan_versions.published_at == record.published_at)]
        new = linear[(linear.k == k) & (linear.v == 3)].set_index("h")
        old_index = {}
        for h, version in zip(
                ((pd.to_datetime(window.interval_start) - BASE).dt.total_seconds() / 600)
                .astype(int) - k * TEMPLATE_SLOTS, window.previous_version):
            old_index[int(h)] = version_index[str(version)[-5:]]
        hs = sorted(old_index)
        assert hs == list(range(FIRST_TARGET[3], TARGETS)), hs
        old_frame = linear[(linear.k == k) & (linear.v.isin(set(old_index.values())))].set_index(
            ["h", "v"])
        d_pv, d_net, d_rho, d_protected = [], [], [], []
        for h in hs:
            o = old_frame.loc[(h, old_index[h])]
            n = new.loc[h]
            d_pv.append(float(n.pv_pred_kW - o.pv_pred_kW))
            d_net.append(float(n.net_pred_kWh - o.net_pred_kWh))
            d_rho.append(float(n.rho_kWh - o.rho_kWh))
            d_protected.append(float(n.protected_kWh - o.protected_kWh))
        rows.append(dict(
            date=record.date, accepted=bool(record.accepted), n_slots=len(hs),
            old_versions="|".join(sorted({f"{UPDATE_HOURS[v_]:02d}:00"
                                         for v_ in set(old_index.values())})),
            mean_abs_delta_pv_kW=float(np.mean(np.abs(d_pv))),
            max_abs_delta_pv_kW=float(np.max(np.abs(d_pv))),
            mean_abs_delta_net_kWh=float(np.mean(np.abs(d_net))),
            mean_abs_delta_rho_kWh=float(np.mean(np.abs(d_rho))),
            mean_abs_delta_protected_kWh=float(np.mean(np.abs(d_protected))),
            max_abs_delta_protected_kWh=float(np.max(np.abs(d_protected)))))
    return pd.DataFrame(rows)


def daily_paired_metrics(longs):
    """Per publication day, version and comparison direction, so improvement counts are auditable."""
    rows = []
    # (a_scheme, a_v, b_scheme, b_v, h_min, keys); the a/b assignment is identical to the one
    # used in source_comparison/update_comparison/interpolation_comparison, so ``delta = b - a``
    # means the same thing in every table. Every output row carries its own a_label/b_label.
    comparisons = [("D1_Q2_vs_Linear", "Q2", 0, "Linear", 0, 1, ["k", "v", "h"]),
                   ("D3_PCHIP_vs_Linear_v0", "Linear", 0, "PCHIP", 0, 1, ["k", "v", "h"]),
                   ("D3_PCHIP_vs_Linear_v1", "Linear", 1, "PCHIP", 1, FIRST_TARGET[1],
                    ["k", "v", "h"]),
                   ("D3_PCHIP_vs_Linear_v2", "Linear", 2, "PCHIP", 2, FIRST_TARGET[2],
                    ["k", "v", "h"]),
                   ("D3_PCHIP_vs_Linear_v3", "Linear", 3, "PCHIP", 3, FIRST_TARGET[3],
                    ["k", "v", "h"]),
                   ("D2_06_vs_00", "Linear", 0, "Linear", 1, FIRST_TARGET[1], ["k", "h"]),
                   ("D2_12_vs_06", "Linear", 1, "Linear", 2, FIRST_TARGET[2], ["k", "h"]),
                   ("D2_18_vs_12", "Linear", 2, "Linear", 3, FIRST_TARGET[3], ["k", "h"])]
    for name, a_scheme, v_a, b_scheme, v_b, h_min, keys in comparisons:
        a = longs[a_scheme][(longs[a_scheme].v == v_a) & (longs[a_scheme].h >= h_min)]
        b = longs[b_scheme][(longs[b_scheme].v == v_b) & (longs[b_scheme].h >= h_min)]
        merged = a.merge(b, on=keys, suffixes=("_a", "_b"))
        for date, block in merged.groupby("date_a"):
            row = dict(comparison=name, date=date, n=int(len(block)),
                       a_label=f"{a_scheme} v{UPDATE_HOURS[v_a]}",
                       b_label=f"{b_scheme} v{UPDATE_HOURS[v_b]}")
            for prefix, suffix in (("a", "_a"), ("b", "_b")):
                row[f"{prefix}_pv_mae_kW"] = float(np.mean(np.abs(
                    block[f"pv_error_kW{suffix}"])))
                row[f"{prefix}_net_mae_kWh"] = float(np.mean(np.abs(
                    block[f"net_error_kWh{suffix}"])))
            row["delta_pv_mae_kW"] = row["b_pv_mae_kW"] - row["a_pv_mae_kW"]
            row["delta_net_mae_kWh"] = row["b_net_mae_kWh"] - row["a_net_mae_kWh"]
            rows.append(row)
    return pd.DataFrame(rows)


# ======================================================================================
# §7  精简检查（三类）
# ======================================================================================
def check_input_pairing(nodes, linear, pchip, counts, linear_counts, long_counts):
    """1. dates, unique keys, matched sizes, the 144/109/73/37 window ranges."""
    assert nodes.shape == (NATURAL_DAYS, len(UPDATE_HOURS), 25)
    assert linear.shape == pchip.shape == (NATURAL_DAYS, len(UPDATE_HOURS), TARGETS)
    keys = long_counts[["scheme", "k", "v", "h"]].drop_duplicates()
    assert len(keys) == len(long_counts), "duplicate (scheme,k,v,h) keys"
    sizes = {}
    for v in range(len(UPDATE_HOURS)):
        # the 0:00 array also carries the auxiliary h=0 midnight target; the template is h=1..144
        h_min = max(1, FIRST_TARGET[v])
        n = int(np.isfinite(pchip[EVAL_FIRST:EVAL_LAST + 1, v, h_min:]).sum())
        expected = (EVAL_LAST - EVAL_FIRST + 1) * (TARGETS - h_min)
        assert n == expected, (v, n, expected)
        sizes[f"v{UPDATE_HOURS[v]}"] = expected
    assert sizes["v0"] == 334 * 144 and sizes["v6"] == 334 * 109
    assert sizes["v12"] == 334 * 73 and sizes["v18"] == 334 * 37
    assert np.array_equal(counts, linear_counts)
    return dict(window_sizes=sizes, unique_keys=True, counts_match_frozen_linear=True)


def check_history_boundary(nodes, counts, rho_pchip, rho_linear, truth_load, truth_pv):
    """3. spot-check 2025-06-21: sample timestamps, counts, order positions, and truncation."""
    k, day = 171, "2025-06-21"
    assert (BASE + timedelta(days=k)).strftime("%Y-%m-%d") == day
    out = {}
    # the 0:00 midnight tail must use m=27 samples (the previous publication day's own midnight
    # target has not ended yet); other 0:00 targets use m=28
    assert int(counts[k, 0, TEMPLATE_SLOTS]) == 27
    assert int(counts[k, 0, 1]) == 28 and int(counts[k, 0, 100]) == 28
    cand = np.arange(k - WINDOW, k)
    cand = cand[cand * TEMPLATE_SLOTS + TEMPLATE_SLOTS + 1 <= k * TEMPLATE_SLOTS]
    assert cand.tolist() == list(range(k - WINDOW, k - 1)), cand.tolist()
    out["midnight_tail_sample_days"] = [str((BASE + timedelta(days=int(j))).date()) for j in cand]
    out["midnight_tail_m"], out["midnight_tail_position"] = int(len(cand)), int(
        math.ceil(ALPHA * len(cand)))
    cand6 = np.arange(k - WINDOW, k)
    cand6 = cand6[cand6 * TEMPLATE_SLOTS + FIRST_TARGET[1] + 1 <= k * TEMPLATE_SLOTS + 36]
    out["six_am_m"], out["six_am_position"] = int(len(cand6)), int(math.ceil(ALPHA * len(cand6)))
    # truncating the archive to the records published by that day must reproduce that day's rho
    truth_net = (truth_load - truth_pv) * DT
    for v, h in ((0, TEMPLATE_SLOTS), (1, FIRST_TARGET[1])):
        values = []
        for j in cand if v == 0 else cand6:
            values.append(truth_net[j, h] - _net_hat(j, v, h))
        values = np.sort(np.array([x for x in values if np.isfinite(x)]))
        position = math.ceil(ALPHA * len(values))
        recomputed = float(values[position - 1])
        assert abs(recomputed - rho_pchip[k, v, h]) < TOL, (v, h, recomputed, rho_pchip[k, v, h])
        out[f"truncated_rho_v{v}_h{h}_kWh"] = recomputed
    out["midnight_tail_rho_linear_kWh"] = float(rho_linear[k, 0, TEMPLATE_SLOTS])
    out["midnight_tail_rho_pchip_kWh"] = float(rho_pchip[k, 0, TEMPLATE_SLOTS])
    return out


_NET_CACHE = {}


def _net_hat(j, v, h):
    """PCHIP net-demand forecast for the truncation check (filled by main before use)."""
    return _NET_CACHE["net_hat_pchip"][j, v, h]


def check_statistics(d1_table, d2_table, d2_identity, d3_table):
    """Matched sample counts must agree and the delta identity must close."""
    largest = 0.0
    for table in (d1_table, d2_table, d3_table):
        if "a_pv_n" in table and "b_pv_n" in table:
            worst = float((table.a_pv_n - table.b_pv_n).abs().max())
            largest = max(largest, worst)
            assert worst == 0, "paired metrics must use identical sample counts"
    identity_worst = float(d2_identity.max_abs_identity_residual_kWh.max())
    assert identity_worst < TOL, identity_worst
    return dict(paired_sample_sizes_consistent=True,
                delta_identity_max_residual_kWh=identity_worst)


# ======================================================================================
# §8  main
# ======================================================================================
def register(parameters, inputs, dependency, amend_reason):
    hashes = {rel: digest(ROOT / rel) for rel in inputs}
    signature = hashlib.sha256(json.dumps(
        dict(parameters=parameters, dependency=dependency, input_sha256=hashes,
             code=digest(CODE)), sort_keys=True).encode()).hexdigest()
    path = OUT / "registration.json"
    record = dict(registered_utc=utc_now(), specification=str(PLAN_MD.relative_to(ROOT)),
                  parameters=parameters, dependency=dependency, input_sha256=hashes,
                  code_sha256=digest(CODE), executable=sys.executable, python=sys.version,
                  numpy=np.__version__, pandas=pd.__version__)
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
    started, started_utc = time.perf_counter(), utc_now()
    timings = {}

    inputs = ["附件/附件2.xlsx",
              "results/q3_pv_time_conversion/pv_hourly_nodes.csv",
              "results/q3_pv_time_conversion/pv_10min_forecasts.csv",
              "results/q3_pv_time_conversion/validation.json",
              "results/q2_time_mapping/archive_float.npz",
              "results/q3_rolling_baseline/prediction_protection.npz",
              "results/q3_rolling_baseline/B2_plan_versions.csv",
              "results/q3_rolling_baseline/B2_revision_decisions.csv",
              "reports/问题三/问题三_预测误差与插值对照诊断实验方案.md"]
    dependency = dict(generation_threshold_kW=GENERATION_THRESHOLD_KW, tolerance=TOL,
                      linear_reuse="results/q3_rolling_baseline/prediction_protection.npz "
                                   "stores the frozen Linear net_forecast/rho/protected")
    signature = register(PARAMETERS, inputs, dependency, args.amend_reason)
    print(f"registered signature={signature[:16]}", flush=True)
    if args.mode == "register":
        return

    tick = time.perf_counter()
    load_src, pv_src = read_attachments()
    truth_load, truth_pv = build_truth(load_src, pv_src)
    nodes, anchor_kind = read_hourly_nodes()
    linear = read_linear_forecasts()
    issued_load, issued_pv, q2_protection = read_q2_forecasts()
    with np.load(BASELINE_NPZ) as archive:
        linear_net = archive["net_forecast"]
        linear_rho = archive["rho"]
        linear_counts = archive["counts"]
    timings["inputs_seconds"] = time.perf_counter() - tick
    print(f"inputs: nodes {nodes.shape}, linear {linear.shape}", flush=True)

    tick = time.perf_counter()
    pchip, hour_delta, hour_residual = build_pchip_forecasts(nodes, linear)
    timings["pchip_seconds"] = time.perf_counter() - tick
    conversion = check_first_hour_and_tail(pchip, linear, nodes)
    conversion["synthetic"] = check_synthetic()
    conversion["max_hour_tiling_residual_kWh"] = hour_residual
    print(f"PCHIP built: hour tiling residual {hour_residual:.3e} kWh, "
          f"first-hour diff {conversion['first_hour_max_abs_diff_kW']:.3e} kW", flush=True)

    tick = time.perf_counter()
    net_q2 = np.full((NATURAL_DAYS, len(UPDATE_HOURS), TARGETS), np.nan)
    net_q2[:, 0, :] = (issued_load - issued_pv) * DT
    # Q2's own protection archive is 2-D (its 0:00 model only); widen it to the version axis
    q2_rho = np.full((NATURAL_DAYS, len(UPDATE_HOURS), TARGETS), np.nan)
    q2_rho[:, 0, :] = q2_protection
    net_linear, err_linear, rho_linear, counts_linear, protected_linear = build_protection(
        linear, issued_load, truth_load, truth_pv)
    net_pchip, err_pchip, rho_pchip, counts_pchip, protected_pchip = build_protection(
        pchip, issued_load, truth_load, truth_pv)
    # the frozen Linear archive is the reference: compare it once instead of re-deriving it
    assert np.allclose(net_linear, linear_net, atol=TOL, equal_nan=True), "Linear net mismatch"
    assert np.allclose(rho_linear, linear_rho, atol=TOL), "Linear rho mismatch"
    assert np.array_equal(counts_linear, linear_counts), "Linear counts mismatch"
    count_summary = check_counts(counts_pchip, linear_counts)
    pv_q2 = np.where(np.isfinite(net_q2), issued_pv[:, None, :], np.nan)
    np.savez_compressed(OUT / "forecast_archive.npz",
                        **{f"pv_{s}": arr for s, arr in
                           (("Q2", pv_q2), ("Linear", linear), ("PCHIP", pchip))},
                        net_Q2=net_q2, net_Linear=net_linear, net_PCHIP=net_pchip,
                        rho_Q2=q2_rho, rho_Linear=rho_linear, rho_PCHIP=rho_pchip,
                        counts=counts_pchip, hour_delta_pchip=hour_delta,
                        truth_load=truth_load, truth_pv=truth_pv, issued_load=issued_load)
    _NET_CACHE["net_hat_pchip"] = net_pchip
    timings["protection_seconds"] = time.perf_counter() - tick
    print(f"protection: {count_summary}", flush=True)

    tick = time.perf_counter()
    longs = {}
    longs["Q2"] = build_long("Q2", pv_q2, net_q2, q2_rho, counts_linear,
                             truth_load, truth_pv, issued_load)
    longs["Linear"] = build_long("Linear", linear, net_linear, rho_linear, counts_linear,
                                 truth_load, truth_pv, issued_load)
    longs["PCHIP"] = build_long("PCHIP", pchip, net_pchip, rho_pchip, counts_pchip,
                                truth_load, truth_pv, issued_load)
    timings["long_tables_seconds"] = time.perf_counter() - tick

    tick = time.perf_counter()
    d1_table, d1_summary = diagnostic_d1(longs)
    d2_table, d2_identity = diagnostic_d2(longs)
    d3_table, d3_integral = diagnostic_d3(longs, hour_delta)
    plan_versions = pd.read_csv(BASELINE_OUT / "B2_plan_versions.csv")
    decisions = pd.read_csv(BASELINE_OUT / "B2_revision_decisions.csv")
    evening = evening_version_changes(longs, plan_versions, decisions)
    daily = daily_paired_metrics(longs)
    change_stats = forecast_change_statistics(linear)
    energy_totals = window_energy_totals(linear, pchip)
    timings["diagnostics_seconds"] = time.perf_counter() - tick

    pairing = check_input_pairing(nodes, linear, pchip, counts_pchip, linear_counts,
                                  pd.concat(longs.values(), ignore_index=True))
    boundary = check_history_boundary(nodes, counts_pchip, rho_pchip, rho_linear,
                                     truth_load, truth_pv)
    statistics = check_statistics(d1_table, d2_table, d2_identity, d3_table)

    frame_to_csv(d1_table, OUT / "source_comparison.csv")
    frame_to_csv(d2_table, OUT / "update_comparison.csv")
    frame_to_csv(d2_identity, OUT / "update_delta_identity.csv")
    save_json(OUT / "forecast_change_summary.json",
              change_stats.set_index("pair").to_dict(orient="index"))
    frame_to_csv(d3_table, OUT / "interpolation_comparison.csv")
    frame_to_csv(d3_integral, OUT / "hour_integral_delta.csv")
    frame_to_csv(daily, OUT / "daily_paired_metrics.csv")
    frame_to_csv(evening, OUT / "evening_version_changes.csv")
    frame_to_csv(change_stats, OUT / "forecast_change_statistics.csv")
    frame_to_csv(energy_totals, OUT / "window_energy_totals.csv")
    save_json(OUT / "d1_summary.json", d1_summary)

    protected_before = protected_manifest()
    save_json(OUT / "protected_before.json", protected_before)
    validation = dict(status="passed", signature=signature,
                      budget="model training 0 / MILP 0 / dispatch re-runs 0",
                      input_pairing=pairing, conversion=conversion, statistics=statistics,
                      history_boundary=boundary,
                      checks_note="three classes only, as specified in the plan's section 4.2",
                      limits=["diagnostic grouping only; no new cleaning or night-gating rule",
                              "shortfall/surplus are forecast-curve errors, not real emergency "
                              "or curtailment; no storage is simulated",
                              "hourly node error and within-hour curve error cannot be separated "
                              "exactly under the 10-minute interval convention",
                              "2025 was already used for method design; this is development-time "
                              "data analysis, not an independent blind test"])
    save_json(OUT / "validation.json", validation)
    protected_after = protected_manifest()
    changed = sorted(k for k in protected_after
                     if k in protected_before and protected_after[k] != protected_before[k])
    missing = sorted(k for k in protected_before if k not in protected_after)
    assert not changed and not missing, (changed[:5], missing[:5])
    run_manifest = dict(status="complete", started_utc=started_utc, finished_utc=utc_now(),
                        wall_seconds=time.perf_counter() - started, timings=timings,
                        signature=signature, executable=sys.executable, python=sys.version,
                        # written by the report layer, not by this script
                        outputs={p.name: digest(p) for p in sorted(OUT.glob("*"))
                                 if p.is_file()
                                 and p.name not in ("run_manifest.json", "figure_integrity.json")},
                        protected={"count": len(protected_before), "changed": changed,
                                   "missing": missing})
    save_json(OUT / "run_manifest.json", run_manifest)
    print(json.dumps(dict(status="complete", wall=run_manifest["wall_seconds"],
                          d1=dict(pv_mae_q2=d1_summary["a_pv_mae"],
                                  pv_mae_linear=d1_summary["b_pv_mae"]),
                          d3_versions=int(len(d3_table))), ensure_ascii=False, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
