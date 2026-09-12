#!/usr/bin/env python
r"""问题三历史偏差校正轻量诊断 —— 计算层（C0_Linear 对 C1_Bias28）。

职责边界
--------
只做：读输入 → 前向生成历史偏差与校正预测 → 重建 q80 → 匹配指标与变化分解 →
三类精简检查 → 落盘。**训练新预测器 0 次、MILP 0 次、储能策略回测 0 次**。
报告与图表由 ``code/q3_bias_correction_report.py`` 只读渲染。

两组
----
* **C0_Linear**：冻结的原 Linear 区间平均光伏预测与冻结 W28/q80（只读复用）。
* **C1_Bias28**：原 Linear 加"同发布时间、同目标时段、前 28 日"的平均光伏偏差，非负截断；
  再用**当时已发布**的校正预测重建自己的 W28/q80。

关键因果约束（方案 3.3 节）
--------------------------
偏差 $b$ 始终由**原 Linear** 的历史 PV 误差估计，**不递归**使用校正后误差；
历史校正预测 $\widehat V^{(1)}_{j}$ 必须用**其自身当时**的 $b_j$ 前向生成并冻结，
不得用当前 $b_k$ 回填历史。

运行::

    E:/Anaconda/envs/math_modeling/python.exe code/q3_bias_correction_diagnostic.py
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
CODE = ROOT / "code/q3_bias_correction_diagnostic.py"
OUT = ROOT / "results/q3_bias_correction_diagnostic"
FIG = ROOT / "figures/q3_bias_correction_diagnostic"
REPORT_MD = ROOT / "reports/问题三/问题三_历史偏差校正诊断结果.md"
PLAN_MD = ROOT / "reports/问题三/问题三_历史偏差校正轻量诊断实验方案.md"
DIAG_NPZ = ROOT / "results/q3_forecast_interpolation_diagnostic/forecast_archive.npz"
BASELINE_NPZ = ROOT / "results/q3_rolling_baseline/prediction_protection.npz"

BASE = pd.Timestamp("2025-01-01")
NATURAL_DAYS = 365
TEMPLATE_SLOTS = 144
TARGETS = 145
DT = 1.0 / 6.0
UPDATE_HOURS = (0, 6, 12, 18)
FIRST_TARGET = {0: 0, 1: 36, 2: 72, 3: 108}
ALPHA = 0.80
WINDOW = 28
MIN_SAMPLES = 7
EVAL_FIRST, EVAL_LAST = 31, 364
GENERATION_THRESHOLD_KW = 1.0
ZERO_TOL_KW = 1e-9
TOL = 1e-8
SELECTED_DATES = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")
CLOCK_WINDOWS = (("morning_0500_0900", 300, 540), ("midday_0900_1500", 540, 900),
                 ("evening_1500_2000", 900, 1200), ("other", None, None))
SCHEMES = ("C0_Linear", "C1_Bias28")
PERTURB_DAY = 171        # 2025-06-21 prefix check

OWN_NEW_REL = {"code/q3_bias_correction_diagnostic.py", "code/q3_bias_correction_report.py",
               "reports/问题三/问题三_历史偏差校正诊断结果.md"}
OWN_NEW_PREFIXES = ("results/q3_bias_correction_diagnostic/",
                    "figures/q3_bias_correction_diagnostic/")

PARAMETERS = dict(
    experiment="Q3 historical PV bias-correction diagnostic (C0_Linear vs C1_Bias28)",
    specification="reports/问题三/问题三_历史偏差校正轻量诊断实验方案.md",
    budget="new model training 0, MILP 0, dispatch re-runs 0",
    groups=dict(C0_Linear="frozen Linear interval-mean PV and frozen W28/q80",
                C1_Bias28="Linear + 28-day same-(publication hour, target) mean PV bias, "
                          "non-negative truncation, own W28/q80 rebuilt from the historically "
                          "issued corrected forecasts"),
    bias="$u_{j,v,h}=V_{j,h}-\\hat V^{(0)}_{j,v,h}$ (positive = the original forecast was too "
         "low); $b_{k,v,h}$ = arithmetic mean over the legal 28-day window, 0 when m<7; no "
         "trimming, no decay, no shrinkage, no full-year constant",
    correction="$\\hat V^{(1)}=\\max(0,\\hat V^{(0)}+b)$; applied to the converted 10-minute "
               "interval-mean power; hourly nodes, interpolator and anchors unchanged",
    causality="b is always estimated from ORIGINAL Linear errors and never recursively from "
              "corrected errors; every historical corrected forecast uses its own b_j",
    protection=dict(window_days=WINDOW, alpha=ALPHA, min_samples=MIN_SAMPLES,
                    key="publication hour, target day-offset, target clock",
                    rule="ascending order statistic at position ceil(0.8m); zero when m<7"),
    signs=dict(pv_error="e^V = V_hat - V (kW), positive = over-forecast",
               net_error="eps = n - n_hat (kWh)", bias_estimate="u = V - V_hat (opposite sign)"),
    evaluation="publication days 2025-02-01..2025-12-31; 0:00 h=1..144, 6/12/18 h=36/72/108..144; "
               "the 0:00 auxiliary h=0 is generated but excluded from the main metrics",
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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=2, default=str),
                          encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def frame_to_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


# ======================================================================================
# §2  输入
# ======================================================================================
def availability_mask():
    """ok[s, v, h]: lag s (1..WINDOW) of target h is realised by publication (k, hour r_v).

    interval_end(j, h) <= k day + r_v hours  <=>  144(k-s) + h + 1 <= 144k + 6 r_v
                                                <=>  h + 1 - 6 r_v <= 144 s
    Only (s, v, h) matter, so the mask needs no k axis.
    """
    h_grid = np.arange(TARGETS)
    ok = np.zeros((WINDOW + 1, len(UPDATE_HOURS), TARGETS), dtype=bool)
    for s in range(1, WINDOW + 1):
        for v, hour in enumerate(UPDATE_HOURS):
            ok[s, v] = (h_grid + 1 - 6 * hour) <= TEMPLATE_SLOTS * s
    return ok


def read_inputs():
    with np.load(DIAG_NPZ) as archive:
        pv_linear = archive["pv_Linear"]
        truth_pv = archive["truth_pv"]
        truth_load = archive["truth_load"]
        issued_load = archive["issued_load"]
    with np.load(BASELINE_NPZ) as archive:
        frozen_net = archive["net_forecast"]
        frozen_rho = archive["rho"]
        frozen_counts = archive["counts"]
        frozen_protected = archive["protected"]
    assert pv_linear.shape == (NATURAL_DAYS, len(UPDATE_HOURS), TARGETS)
    assert truth_pv.shape == truth_load.shape == issued_load.shape == (NATURAL_DAYS, TARGETS)
    for v in range(len(UPDATE_HOURS)):
        assert np.isfinite(pv_linear[EVAL_FIRST:, v, FIRST_TARGET[v]:]).all(), v
    return dict(pv_linear=pv_linear, truth_pv=truth_pv, truth_load=truth_load,
                issued_load=issued_load, frozen_net=frozen_net, frozen_rho=frozen_rho,
                frozen_counts=frozen_counts, frozen_protected=frozen_protected)


# ======================================================================================
# §3  历史偏差估计与校正预测（前向生成）
# ======================================================================================
def window_mean(values, ok, valid):
    """Arithmetic mean over lags 1..WINDOW of ``values`` where ``valid``; 0 when m<MIN_SAMPLES.

    Lag ``s`` adds the value at source row ``k-s`` to target row ``k``, so the validity mask and
    the value slice must both be taken from the SOURCE rows ``0..n-s-1`` (``valid[s:]`` would pair
    a target-row mask with a source-row value and silently mis-count the samples).
    """
    n = values.shape[0]
    counts = np.zeros(values.shape, dtype=np.int64)
    total = np.zeros(values.shape, dtype=float)
    for s in range(1, WINDOW + 1):
        if s >= n:
            break
        source_valid = valid[:n - s] & ok[s][None, :, :]
        counts[s:] += source_valid
        total[s:] += np.where(source_valid, values[:n - s], 0.0)
    mean = np.where(counts >= MIN_SAMPLES, total / np.maximum(counts, 1), 0.0)
    return mean, counts


def window_order_statistic(values, ok):
    """Grouped order statistic at position ceil(0.8m) over lags 1..WINDOW; 0 when m<MIN_SAMPLES."""
    n = values.shape[0]
    lagged = np.full((WINDOW, n) + values.shape[1:], np.nan)
    counts = np.zeros(values.shape, dtype=np.int64)
    for s in range(1, WINDOW + 1):
        if s >= n:
            break
        block = np.where(ok[s][None, :, :], values[:n - s], np.nan)
        lagged[s - 1, s:] = block
        counts[s:] += np.isfinite(block)
    ordered = np.sort(lagged, axis=0)                      # NaN sorts last
    position = np.ceil(ALPHA * counts).astype(int)
    index = np.clip(position - 1, 0, WINDOW - 1)
    picked = np.take_along_axis(ordered, index[None, :, :, :], axis=0)[0]
    return np.where(counts >= MIN_SAMPLES, picked, 0.0), counts


def build_bias(pv_linear, truth_pv, ok):
    """b_{k,v,h}: mean of u = V - V_hat over the legal past 28 days; counts reported."""
    u = truth_pv[:, None, :] - pv_linear
    bias, m_v = window_mean(u, ok, np.isfinite(u))
    return bias, m_v, u


def correct_forecast(pv_linear, bias):
    """max(0, V_hat + b); truncation mask and the applied delta are kept for reporting."""
    raw = pv_linear + bias
    corrected = np.where(np.isfinite(pv_linear), np.maximum(0.0, raw), np.nan)
    truncated = np.isfinite(pv_linear) & (raw < 0.0)
    delta = np.where(np.isfinite(pv_linear), corrected - pv_linear, np.nan)
    return corrected, delta, truncated


def net_forecast(pv, issued_load):
    net = np.full((NATURAL_DAYS, len(UPDATE_HOURS), TARGETS), np.nan)
    for v in range(len(UPDATE_HOURS)):
        h0 = FIRST_TARGET[v] if v else 0
        net[:, v, h0:] = (issued_load[:, h0:] - pv[:, v, h0:]) * DT
    return net


# ======================================================================================
# §4  分组 q80
# ======================================================================================
def protection_archive(net_hat, truth_load, truth_pv, ok):
    """Grouped W28/q80 residual correction rho, plus the residual and sample-count archives."""
    truth_net = (truth_load - truth_pv) * DT
    eps = truth_net[:, None, :] - net_hat
    rho, m_n = window_order_statistic(eps, ok)
    return eps, rho, m_n, net_hat + rho


# ======================================================================================
# §5  记录表与指标
# ======================================================================================
def boundary_label(v, h):
    if h == TEMPLATE_SLOTS and v == 0:
        return "midnight_tail"
    if h - UPDATE_HOURS[v] * 6 < 6:
        return "first_hour"
    return "interior"


def clock_label(h):
    minutes = h * 10
    for name, lo, hi in CLOCK_WINDOWS:
        if lo is not None and lo <= minutes < hi:
            return name
    return "other"


def build_long(scheme, pv, net_hat, rho, protected, m_v, m_n, delta_v, truncated,
               truth_load, truth_pv, bias=None):
    """One row per (publication day, version, target) inside that version's evaluation window."""
    rows = []
    for v in range(len(UPDATE_HOURS)):
        h0 = max(1, FIRST_TARGET[v])           # the 0:00 auxiliary h=0 is excluded here
        for k in range(EVAL_FIRST, EVAL_LAST + 1):
            hs = np.arange(h0, TARGETS)
            finite = np.isfinite(pv[k, v, hs])
            hs = hs[finite]
            if hs.size == 0:
                continue
            date = BASE + pd.Timedelta(days=k)
            rows.append(pd.DataFrame({
                "scheme": scheme, "k": k, "v": v, "h": hs,
                "date": date.strftime("%Y-%m-%d"), "month": date.strftime("%Y-%m"),
                "boundary": [boundary_label(v, int(h)) for h in hs],
                "clock": [clock_label(int(h)) for h in hs],
                "pv_pred_kW": pv[k, v, hs],
                "pv_actual_kW": truth_pv[k, hs],
                "net_pred_kWh": net_hat[k, v, hs],
                "net_actual_kWh": (truth_load[k, hs] - truth_pv[k, hs]) * DT,
                "rho_kWh": rho[k, v, hs],
                "protected_kWh": protected[k, v, hs],
                "m_bias_samples": m_v[k, v, hs],
                "m_net_samples": m_n[k, v, hs],
                "delta_v_kW": delta_v[k, v, hs],
                "truncated": truncated[k, v, hs],
                "bias_kW": bias[k, v, hs] if bias is not None else 0.0,
            }))
    frame = pd.concat(rows, ignore_index=True)
    frame["pv_error_kW"] = frame.pv_pred_kW - frame.pv_actual_kW       # positive = over-forecast
    frame["net_error_kWh"] = frame.net_actual_kWh - frame.net_pred_kWh
    frame["is_generation"] = frame.pv_actual_kW > GENERATION_THRESHOLD_KW
    assert np.isfinite(frame.pv_pred_kW).all() and np.isfinite(frame.net_actual_kWh).all()
    assert (frame.pv_pred_kW >= -TOL).all(), "corrected PV must be non-negative"
    return frame


def metric_block(frame, prefix, suffix=""):
    out = {}
    for name, column in (("pv", f"pv_error_kW{suffix}"), ("net", f"net_error_kWh{suffix}")):
        err = frame[column].to_numpy()
        out[f"{prefix}_{name}_n"] = int(err.size)
        out[f"{prefix}_{name}_mae"] = float(np.mean(np.abs(err)))
        out[f"{prefix}_{name}_rmse"] = float(np.sqrt(np.mean(err ** 2)))
        out[f"{prefix}_{name}_bias"] = float(np.mean(err))
    return out


def protection_block(frame, prefix, suffix=""):
    actual = frame[f"net_actual_kWh{suffix}"].to_numpy()
    protected = frame[f"protected_kWh{suffix}"].to_numpy()
    return {f"{prefix}_coverage": float(np.mean(actual <= protected)),
            f"{prefix}_mean_shortfall_kWh": float(np.mean(np.maximum(actual - protected, 0.0))),
            f"{prefix}_mean_surplus_kWh": float(np.mean(np.maximum(protected - actual, 0.0))),
            f"{prefix}_mean_rho_kWh": float(np.mean(frame[f"rho_kWh{suffix}"].to_numpy())),
            f"{prefix}_rho_p10_kWh": float(np.percentile(frame[f"rho_kWh{suffix}"], 10)),
            f"{prefix}_rho_p90_kWh": float(np.percentile(frame[f"rho_kWh{suffix}"], 90))}


def compare_groups(a, b, group_cols):
    """Paired metrics for C0 (a) and C1 (b) on their identical matched target set."""
    keys = ["k", "v", "h"]
    merged = a.merge(b, on=keys, suffixes=("_a", "_b"), how="inner")
    assert len(merged) > 0, "no matched targets"
    for name in ("pv_actual_kW", "net_actual_kWh"):
        assert np.allclose(merged[f"{name}_a"], merged[f"{name}_b"], atol=TOL), name
    # both schemes share the same (k, v, h) axis, so the per-target labels are identical
    for col in ("date", "month", "clock", "boundary", "is_generation"):
        assert (merged[f"{col}_a"] == merged[f"{col}_b"]).all(), col
        merged = merged.rename(columns={f"{col}_a": col}).drop(columns=[f"{col}_b"])
    merged = merged.drop(columns=[c for c in ("scheme_a", "scheme_b") if c in merged.columns])
    groups = merged.groupby(group_cols, dropna=False) if group_cols else [((), merged)]
    rows = []
    for key, block in groups:
        row = {}
        if group_cols:
            for col, value in zip(group_cols, key if isinstance(key, tuple) else (key,)):
                row[col] = value
        row.update(metric_block(block, "a", "_a"))
        row.update(metric_block(block, "b", "_b"))
        row.update(protection_block(block, "a", "_a"))
        row.update(protection_block(block, "b", "_b"))
        row["delta_pv_mae_kW"] = row["b_pv_mae"] - row["a_pv_mae"]
        row["delta_pv_rmse_kW"] = row["b_pv_rmse"] - row["a_pv_rmse"]
        row["delta_pv_bias_kW"] = row["b_pv_bias"] - row["a_pv_bias"]
        row["delta_net_mae_kWh"] = row["b_net_mae"] - row["a_net_mae"]
        row["delta_net_bias_kWh"] = row["b_net_bias"] - row["a_net_bias"]
        row["delta_shortfall_kWh"] = (row["b_mean_shortfall_kWh"] - row["a_mean_shortfall_kWh"])
        row["delta_surplus_kWh"] = row["b_mean_surplus_kWh"] - row["a_mean_surplus_kWh"]
        rows.append(row)
    return pd.DataFrame(rows)


def decomposition(c0, c1):
    """Delta(n_hat), Delta(rho) and Delta(n_tilde) statistics, overall and by subset."""
    merged = c0.merge(c1, on=["k", "v", "h"], suffixes=("_a", "_b"))
    merged = merged.rename(columns={"month_a": "month", "clock_a": "clock",
                                    "is_generation_a": "is_generation"})
    merged["d_net"] = merged.net_pred_kWh_b - merged.net_pred_kWh_a
    merged["d_rho"] = merged.rho_kWh_b - merged.rho_kWh_a
    merged["d_protected"] = merged.protected_kWh_b - merged.protected_kWh_a
    merged["boundary"] = [boundary_label(int(v), int(h))
                          for v, h in zip(merged.v, merged.h)]
    rows = []

    def add(label, block):
        rows.append(dict(
            subset=label, n=int(len(block)),
            mean_d_net_kWh=float(block.d_net.mean()), mean_abs_d_net_kWh=float(block.d_net.abs().mean()),
            mean_d_rho_kWh=float(block.d_rho.mean()), mean_abs_d_rho_kWh=float(block.d_rho.abs().mean()),
            mean_d_protected_kWh=float(block.d_protected.mean()),
            mean_abs_d_protected_kWh=float(block.d_protected.abs().mean()),
            p10_d_protected_kWh=float(block.d_protected.quantile(0.10)),
            p90_d_protected_kWh=float(block.d_protected.quantile(0.90)),
            max_abs_d_protected_kWh=float(block.d_protected.abs().max()),
            max_abs_identity_residual_kWh=float(
                (block.d_protected - block.d_net - block.d_rho).abs().max()),
            truncated_share=float(block.truncated_b.mean())))

    add("overall", merged)
    for v, hour in enumerate(UPDATE_HOURS):
        add(f"publication_{hour:02d}", merged[merged.v == v])
    for name in ("first_hour", "interior", "midnight_tail"):
        add(f"boundary_{name}", merged[merged.boundary == name])
    add("truncated_rows", merged[merged.truncated_b])
    add("non_truncated_rows", merged[~merged.truncated_b])
    add("generation_subset", merged[merged.is_generation])

    # effect of the truncation, isolated on the rows where it actually binds
    hit = merged[merged.truncated_b]
    rows.append(dict(subset="truncation_binding_only", n=int(len(hit)),
                     mean_d_net_kWh=float(hit.d_net.mean()) if len(hit) else np.nan,
                     mean_abs_d_net_kWh=float(hit.d_net.abs().mean()) if len(hit) else np.nan,
                     mean_d_rho_kWh=float(hit.d_rho.mean()) if len(hit) else np.nan,
                     mean_abs_d_rho_kWh=float(hit.d_rho.abs().mean()) if len(hit) else np.nan,
                     mean_d_protected_kWh=float(hit.d_protected.mean()) if len(hit) else np.nan,
                     mean_abs_d_protected_kWh=float(hit.d_protected.abs().mean()) if len(hit) else np.nan,
                     p10_d_protected_kWh=np.nan, p90_d_protected_kWh=np.nan,
                     max_abs_d_protected_kWh=(float(hit.d_protected.abs().max())
                                              if len(hit) else np.nan),
                     max_abs_identity_residual_kWh=(float(
                         (hit.d_protected - hit.d_net - hit.d_rho).abs().max()) if len(hit) else np.nan),
                     truncated_share=1.0 if len(hit) else np.nan))
    return pd.DataFrame(rows)


def bias_statistics(bias, delta_v, truncated, m_v):
    """Per publication hour: size of the applied bias/correction and the truncation incidence.

    Reported because "the protected net change is near zero" and "the correction is near zero" are
    different statements; the second is false for every version.
    """
    rows = []
    for v, hour in enumerate(UPDATE_HOURS):
        sel = np.zeros_like(bias, dtype=bool)
        sel[EVAL_FIRST:EVAL_LAST + 1, v, max(1, FIRST_TARGET[v]):] = True
        sel &= np.isfinite(bias)
        rows.append(dict(
            publication_hour=hour, n=int(sel.sum()),
            bias_mean_kW=float(bias[sel].mean()), bias_mean_abs_kW=float(np.abs(bias[sel]).mean()),
            bias_max_abs_kW=float(np.abs(bias[sel]).max()),
            delta_v_mean_kW=float(delta_v[sel].mean()),
            delta_v_mean_abs_kW=float(np.abs(delta_v[sel]).mean()),
            delta_v_max_abs_kW=float(np.abs(delta_v[sel]).max()),
            delta_v_min_kW=float(delta_v[sel].min()),
            delta_v_max_kW=float(delta_v[sel].max()),
            delta_v_mean_abs_kWh=float(np.abs(delta_v[sel]).mean() * DT),
            truncated_rows=int(truncated[sel].sum()),
            truncated_share=float(truncated[sel].mean()),
            zero_history_rows=int((m_v[sel] < MIN_SAMPLES).sum())))
    frame = pd.DataFrame(rows)
    hit = np.zeros_like(bias, dtype=bool)
    hit[EVAL_FIRST:EVAL_LAST + 1] = truncated[EVAL_FIRST:EVAL_LAST + 1]
    hit &= np.isfinite(bias)
    frame = pd.concat([frame, pd.DataFrame([dict(
        publication_hour="truncated_subset", n=int(hit.sum()),
        bias_mean_kW=float(bias[hit].mean()), bias_mean_abs_kW=float(np.abs(bias[hit]).mean()),
        bias_max_abs_kW=float(np.abs(bias[hit]).max()),
        delta_v_mean_kW=float(delta_v[hit].mean()),
        delta_v_mean_abs_kW=float(np.abs(delta_v[hit]).mean()),
        delta_v_max_abs_kW=float(np.abs(delta_v[hit]).max()),
        delta_v_min_kW=float(delta_v[hit].min()), delta_v_max_kW=float(delta_v[hit].max()),
        delta_v_mean_abs_kWh=float(np.abs(delta_v[hit]).mean() * DT),
        truncated_rows=int(hit.sum()), truncated_share=1.0, zero_history_rows=0)])],
        ignore_index=True)
    return frame


def daily_paired(c0, c1):
    merged = c0.merge(c1, on=["k", "v", "h"], suffixes=("_a", "_b"))
    rows = []
    for (date, v), block in merged.groupby(["date_a", "v"]):
        row = dict(date=date, publication_hour=UPDATE_HOURS[int(v)], n=int(len(block)))
        for prefix, suffix in (("a", "_a"), ("b", "_b")):
            row[f"{prefix}_pv_mae_kW"] = float(block[f"pv_error_kW{suffix}"].abs().mean())
            row[f"{prefix}_pv_bias_kW"] = float(block[f"pv_error_kW{suffix}"].mean())
            row[f"{prefix}_net_mae_kWh"] = float(block[f"net_error_kWh{suffix}"].abs().mean())
            row[f"{prefix}_shortfall_kWh"] = float(np.maximum(
                block[f"net_actual_kWh{suffix}"] - block[f"protected_kWh{suffix}"], 0.0).mean())
        row["delta_pv_mae_kW"] = row["b_pv_mae_kW"] - row["a_pv_mae_kW"]
        row["delta_pv_bias_kW"] = row["b_pv_bias_kW"] - row["a_pv_bias_kW"]
        row["delta_net_mae_kWh"] = row["b_net_mae_kWh"] - row["a_net_mae_kWh"]
        row["delta_shortfall_kWh"] = row["b_shortfall_kWh"] - row["a_shortfall_kWh"]
        rows.append(row)
    return pd.DataFrame(rows)


# ======================================================================================
# §6  三类验证
# ======================================================================================
def check_synthetic():
    """1. Two hand-checkable examples: exact q80 translation cancellation, and truncation."""
    out = {}
    # (a) constant shift +c over the whole legal window and histories, no truncation:
    #     d(n_hat) = -c*dt everywhere, so every historical residual rises by exactly c*dt and the
    #     order statistic shifts by c*dt -> d(rho) = +c*dt -> d(protected) = 0 exactly.
    rng = np.random.default_rng(20250912)
    n_days, n_slots = 40, 6
    base_forecast = rng.uniform(50.0, 400.0, size=(n_days, n_slots))
    actual = base_forecast + rng.normal(0.0, 30.0, size=(n_days, n_slots))
    shift = 17.0
    # net residual as the production code defines it: eps = n - n_hat = (V_hat - V)*dt when the
    # load forecast is exact, so shifting the PV forecast UP by `shift` raises eps by shift*dt
    eps_a = (base_forecast - actual) * DT
    eps_b = (base_forecast + shift - actual) * DT
    rho_a, m_a = window_order_statistic(eps_a[:, None, :], ok_dummy(n_slots))
    rho_b, m_b = window_order_statistic(eps_b[:, None, :], ok_dummy(n_slots))
    assert np.array_equal(m_a, m_b)
    finite = m_a[:, 0, :] >= MIN_SAMPLES
    d_rho = (rho_b - rho_a)[:, 0, :][finite]
    d_net = np.full(d_rho.shape, -shift * DT)
    assert np.allclose(d_rho, shift * DT, atol=1e-12), d_rho[:3]
    assert np.allclose(d_net + d_rho, 0.0, atol=1e-12), "q80 must cancel a uniform shift"
    out["uniform_shift_cancellation_max_abs_residual_kWh"] = float(np.max(np.abs(d_net + d_rho)))
    out["uniform_shift_d_net_kWh"] = float(-shift * DT)
    out["uniform_shift_d_rho_kWh"] = float(d_rho.mean())
    # (b) insufficient history and non-negative truncation
    small = np.full((3, 1, 2), 5.0, dtype=float)                 # only 2 lags available
    rho_small, m_small = window_order_statistic(small, ok_dummy(2))
    assert int(m_small.max()) < MIN_SAMPLES and np.all(rho_small == 0.0)
    out["insufficient_history_m_max"] = int(m_small.max())
    out["insufficient_history_rho"] = float(rho_small.max())
    v0 = np.array([[1.0, 100.0, 0.0]])
    bias = np.array([[-10.0, -10.0, 5.0]])
    corrected, delta, truncated = correct_forecast(v0[:, None, :], bias[:, None, :])
    assert corrected[0, 0, 0] == 0.0 and truncated[0, 0, 0] and delta[0, 0, 0] == -1.0
    assert corrected[0, 0, 1] == 90.0 and not truncated[0, 0, 1]
    assert corrected[0, 0, 2] == 5.0 and not truncated[0, 0, 2]
    out["truncation_example"] = dict(corrected=corrected[0, 0, :].tolist(),
                                     truncated=truncated[0, 0, :].tolist())
    return out


def ok_dummy(n_slots):
    """All-true availability mask shaped (WINDOW+1, 1, n_slots) for the synthetic examples."""
    return np.ones((WINDOW + 1, 1, n_slots), dtype=bool)


def check_prefix(bias, m_v, rho_c1, m_n, eps_c1, pv_linear, truth_pv, ok):
    """2. 2025-06-21 prefix check: recompute that day's b and rho from the legal history only.

    Nothing is taken from the forward-generated archive except the comparison target: the candidate
    set, the bias mean and the order statistic are rebuilt from the raw inputs.
    """
    k, day = PERTURB_DAY, "2025-06-21"
    assert (BASE + timedelta(days=k)).strftime("%Y-%m-%d") == day
    rows, out = [], {}
    for v, h in ((0, TEMPLATE_SLOTS), (0, 1), (1, FIRST_TARGET[1])):
        cand = [j for j in range(max(0, k - WINDOW), k)
                if ok[k - j, v, h] and np.isfinite(truth_pv[j, h])
                and np.isfinite(pv_linear[j, v, h])]
        m = len(cand)
        position = math.ceil(ALPHA * m) if m else 0
        u_values = np.array([truth_pv[j, h] - pv_linear[j, v, h] for j in cand])
        recomputed_bias = float(u_values.mean()) if m >= MIN_SAMPLES else 0.0
        eps_values = np.sort(np.array([eps_c1[j, v, h] for j in cand
                                       if np.isfinite(eps_c1[j, v, h])]))
        m_net = len(eps_values)
        recomputed_rho = (float(eps_values[math.ceil(ALPHA * m_net) - 1])
                          if m_net >= MIN_SAMPLES else 0.0)
        assert int(m_v[k, v, h]) == m, (v, h, int(m_v[k, v, h]), m)
        assert abs(float(bias[k, v, h]) - recomputed_bias) < TOL, (v, h, recomputed_bias)
        assert int(m_n[k, v, h]) == m_net, (v, h, int(m_n[k, v, h]), m_net)
        assert abs(float(rho_c1[k, v, h]) - recomputed_rho) < TOL, (v, h, recomputed_rho)
        rows.append(dict(date=day, publication_hour=UPDATE_HOURS[v], target_h=int(h),
                         m_bias_samples=int(m), m_net_samples=int(m_net),
                         quantile_position=int(position),
                         recomputed_bias_kW=recomputed_bias, stored_bias_kW=float(bias[k, v, h]),
                         recomputed_rho_kWh=recomputed_rho, stored_rho_kWh=float(rho_c1[k, v, h]),
                         first_sample_day=str((BASE + timedelta(days=cand[0])).date()) if cand else "",
                         last_sample_day=str((BASE + timedelta(days=cand[-1])).date()) if cand else ""))
        out[f"v{UPDATE_HOURS[v]}_h{h}"] = dict(m_bias=m, m_net=m_net, position=position)
    return pd.DataFrame(rows), out



def check_saved(c0, c1, inputs, rho_c0, m_n_c0, ok):
    """3. Saved-result check: sample counts, common samples, units/signs, change identities."""
    out = {}
    sizes = {}
    for scheme, frame in (("C0_Linear", c0), ("C1_Bias28", c1)):
        per_version = {}
        for v, hour in enumerate(UPDATE_HOURS):
            h0 = max(1, FIRST_TARGET[v])
            expected = (EVAL_LAST - EVAL_FIRST + 1) * (TARGETS - h0)
            got = int(((frame.v == v)).sum())
            assert got == expected, (scheme, hour, got, expected)
            per_version[f"v{hour}"] = got
        sizes[scheme] = per_version
    assert sizes["C0_Linear"] == sizes["C1_Bias28"]
    out["window_sizes"] = sizes["C0_Linear"]
    merged = c0.merge(c1, on=["k", "v", "h"], how="inner")
    assert len(merged) == len(c0) == len(c1), "the two schemes must share an identical target set"
    out["common_samples"] = int(len(merged))
    assert np.isfinite(c0.pv_pred_kW).all() and np.isfinite(c1.pv_pred_kW).all()
    assert (c0.pv_pred_kW >= -TOL).all() and (c1.pv_pred_kW >= -TOL).all()
    assert np.isfinite(c0.protected_kWh).all() and np.isfinite(c1.protected_kWh).all()
    # C0 must still reproduce the frozen archive, and the recomputed sample counts must match
    ordered = c0.sort_values(["k", "v", "h"])
    frozen = inputs["frozen_net"]
    reference = frozen[ordered.k.to_numpy(), ordered.v.to_numpy(), ordered.h.to_numpy()]
    out["c0_matches_frozen_net_max_abs_kWh"] = float(
        np.max(np.abs(ordered.net_pred_kWh.to_numpy() - reference)))
    assert out["c0_matches_frozen_net_max_abs_kWh"] < TOL
    frozen_counts = inputs["frozen_counts"]
    assert np.array_equal(m_n_c0, frozen_counts), "recomputed C0 sample counts differ from frozen"
    out["c0_counts_match_frozen"] = True
    assert np.allclose(rho_c0, inputs["frozen_rho"], atol=TOL), "recomputed C0 rho differs"
    out["c0_rho_matches_frozen"] = True
    # the availability rule itself: 0:00 h=144 has m=27 position 22, everything else m=28/23
    late = slice(60, NATURAL_DAYS)
    assert np.all(m_n_c0[late, 0, TEMPLATE_SLOTS] == 27)
    assert np.all(m_n_c0[late, 0, 1:TEMPLATE_SLOTS] == 28)
    for v in range(1, len(UPDATE_HOURS)):
        assert np.all(m_n_c0[late, v, FIRST_TARGET[v]:] == 28), v
    out["quantile_positions"] = dict(h144_0am=22, other_0am=23, later_versions=23)
    return out


# ======================================================================================
# §7  main
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

    inputs = ["results/q3_pv_time_conversion/pv_10min_forecasts.csv",
              "results/q3_forecast_interpolation_diagnostic/forecast_archive.npz",
              "results/q3_rolling_baseline/prediction_protection.npz",
              "reports/问题三/问题三_历史偏差校正轻量诊断实验方案.md"]
    dependency = dict(window_days=WINDOW, min_samples=MIN_SAMPLES, alpha=ALPHA,
                      tolerance=TOL, generation_threshold_kW=GENERATION_THRESHOLD_KW,
                      truth_source="results/q3_forecast_interpolation_diagnostic/"
                                   "forecast_archive.npz (attachment-2 truth under the accepted "
                                   "interval-start mapping)")
    signature = register(PARAMETERS, inputs, dependency, args.amend_reason)
    print(f"registered signature={signature[:16]}", flush=True)
    if args.mode == "register":
        return

    tick = time.perf_counter()
    data = read_inputs()
    ok = availability_mask()
    timings["inputs_seconds"] = time.perf_counter() - tick
    print(f"inputs: pv_linear {data['pv_linear'].shape}", flush=True)

    tick = time.perf_counter()
    bias, m_v, u = build_bias(data["pv_linear"], data["truth_pv"], ok)
    pv_c1, delta_v, truncated = correct_forecast(data["pv_linear"], bias)
    net_c1 = net_forecast(pv_c1, data["issued_load"])
    eps_c1, rho_c1, m_n_c1, protected_c1 = protection_archive(
        net_c1, data["truth_load"], data["truth_pv"], ok)
    # C0 side is recomputed with the same code path purely to cross-check the frozen archive
    net_c0 = data["frozen_net"]
    truth_net = (data["truth_load"] - data["truth_pv"]) * DT
    eps_c0 = truth_net[:, None, :] - net_c0
    rho_c0, m_n_c0 = window_order_statistic(eps_c0, ok)
    protected_c0 = net_c0 + rho_c0
    timings["correction_seconds"] = time.perf_counter() - tick
    print(f"bias built: |b| mean {np.abs(bias).mean():.6f} kW, max {np.abs(bias).max():.3f} kW, "
          f"truncated rows {int(truncated.sum()):,}", flush=True)

    np.savez_compressed(OUT / "bias_forecast_archive.npz",
                        pv_linear=data["pv_linear"], pv_corrected=pv_c1, bias_kW=bias,
                        delta_v_kW=delta_v, truncated=truncated, m_bias_samples=m_v,
                        m_net_samples_c1=m_n_c1, m_net_samples_c0=m_n_c0,
                        net_c0=net_c0, net_c1=net_c1, rho_c0=rho_c0, rho_c1=rho_c1,
                        protected_c0=protected_c0, protected_c1=protected_c1,
                        truth_pv=data["truth_pv"], truth_load=data["truth_load"],
                        issued_load=data["issued_load"], availability=ok)

    tick = time.perf_counter()
    c0 = build_long("C0_Linear", data["pv_linear"], net_c0, rho_c0, protected_c0, m_v, m_n_c0,
                    np.zeros_like(delta_v), np.zeros_like(truncated, dtype=bool),
                    data["truth_load"], data["truth_pv"])
    c1 = build_long("C1_Bias28", pv_c1, net_c1, rho_c1, protected_c1, m_v, m_n_c1,
                    delta_v, truncated, data["truth_load"], data["truth_pv"], bias=bias)
    timings["long_tables_seconds"] = time.perf_counter() - tick

    tick = time.perf_counter()
    overall = compare_groups(c0, c1, [])
    by_version = compare_groups(c0, c1, ["v"])
    by_month = compare_groups(c0, c1, ["v", "month"])
    by_clock = compare_groups(c0, c1, ["v", "clock"])
    by_boundary = compare_groups(c0, c1, ["v", "boundary"])
    by_generation = compare_groups(c0, c1, ["v", "is_generation"])
    truncated_keys = c1.loc[c1.truncated, ["k", "v", "h"]]
    plain_keys = c1.loc[~c1.truncated, ["k", "v", "h"]]
    by_truncated = compare_groups(c0.merge(truncated_keys, on=["k", "v", "h"]),
                                  c1.merge(truncated_keys, on=["k", "v", "h"]), [])
    by_plain = compare_groups(c0.merge(plain_keys, on=["k", "v", "h"]),
                              c1.merge(plain_keys, on=["k", "v", "h"]), [])
    bias_table = bias_statistics(bias, delta_v, truncated, m_v)
    comparison = pd.concat([
        overall.assign(group_type="overall"), by_version.assign(group_type="publication_hour"),
        by_month.assign(group_type="publication_hour_x_month"),
        by_clock.assign(group_type="publication_hour_x_clock"),
        by_boundary.assign(group_type="publication_hour_x_boundary"),
        by_generation.assign(group_type="publication_hour_x_generation"),
        by_truncated.assign(group_type="truncated_subset"),
        by_plain.assign(group_type="non_truncated_subset"),
    ], ignore_index=True)
    monthly = pd.concat([
        by_month.assign(group_type="month"), by_clock.assign(group_type="clock"),
        by_boundary.assign(group_type="boundary"),
        by_generation.assign(group_type="generation")], ignore_index=True)
    for frame in (comparison, monthly):
        if "v" in frame.columns:
            frame["publication_hour"] = frame.v.map(
                lambda x: UPDATE_HOURS[int(x)] if pd.notna(x) else np.nan)
    daily = daily_paired(c0, c1)
    decomp = decomposition(c0, c1)
    timings["metrics_seconds"] = time.perf_counter() - tick

    synthetic = check_synthetic()
    spotcheck, prefix = check_prefix(bias, m_v, rho_c1, m_n_c1, eps_c1, data["pv_linear"],
                                     data["truth_pv"], ok)
    saved = check_saved(c0, c1, data, rho_c0, m_n_c0, ok)

    frame_to_csv(comparison, OUT / "comparison.csv")
    frame_to_csv(monthly, OUT / "monthly.csv")
    frame_to_csv(daily, OUT / "daily_paired_metrics.csv")
    frame_to_csv(decomp, OUT / "correction_decomposition.csv")
    frame_to_csv(spotcheck, OUT / "history_spotcheck.csv")
    frame_to_csv(bias_table, OUT / "bias_statistics.csv")

    summary = dict(
        signature=signature,
        bias_abs_mean_kW=float(np.abs(bias).mean()), bias_max_kW=float(bias.max()),
        bias_min_kW=float(bias.min()), truncated_rows=int(truncated.sum()),
        # shares use the evaluation window as the denominator (121,242 rows), not the raw array,
        # which also contains NaN cells and versions' out-of-window targets
        truncated_share=float(c1.truncated.mean()),
        overall=overall.iloc[0].to_dict(),
        per_version=by_version.to_dict(orient="records"),
        zero_history_rows=int((c1.m_bias_samples < MIN_SAMPLES).sum()),
        c1_rho_mean_kWh=float(rho_c1[EVAL_FIRST:].mean()),
        c0_rho_mean_kWh=float(rho_c0[EVAL_FIRST:].mean()),
        bias_by_version=bias_table.to_dict(orient="records"))
    save_json(OUT / "summary.json", summary)
    validation = dict(status="passed", signature=signature,
                      budget="new model training 0 / MILP 0 / dispatch re-runs 0",
                      synthetic=synthetic, prefix_check=prefix, saved_check=saved,
                      checks_note="three classes only, as specified in the plan's section 4.3",
                      limits=["shortfall/surplus are forecast-curve errors, not real emergency or "
                              "curtailment; no storage is simulated and no cost is computed",
                              "the bias is a rolling 28-day mean; short-term bias stability is a "
                              "hypothesis under test, not an established property",
                              "all results come from the same 2025 development data; this is a "
                              "rolling causal diagnostic, not an independent blind test",
                              "coverage rising is not automatically better: it can come from "
                              "excess surplus"])
    save_json(OUT / "validation.json", validation)
    run_manifest = dict(status="complete", started_utc=started_utc, finished_utc=utc_now(),
                        wall_seconds=time.perf_counter() - started, timings=timings,
                        signature=signature, executable=sys.executable, python=sys.version,
                        # figure_integrity.json is written by the report layer: listing it
                        # here would make a later report render invalidate a certified hash
                        outputs={p.name: digest(p) for p in sorted(OUT.glob("*"))
                                 if p.is_file() and p.name not in ("run_manifest.json", "figure_integrity.json")})
    save_json(OUT / "run_manifest.json", run_manifest)
    print(json.dumps({"status": "complete", "wall_seconds": run_manifest["wall_seconds"],
                      "pv_mae_c0": overall.iloc[0]["a_pv_mae"],
                      "pv_mae_c1": overall.iloc[0]["b_pv_mae"],
                      "pv_bias_c0": overall.iloc[0]["a_pv_bias"],
                      "pv_bias_c1": overall.iloc[0]["b_pv_bias"],
                      "net_mae_c0": overall.iloc[0]["a_net_mae"],
                      "net_mae_c1": overall.iloc[0]["b_net_mae"],
                      "shortfall_c0": overall.iloc[0]["a_mean_shortfall_kWh"],
                      "shortfall_c1": overall.iloc[0]["b_mean_shortfall_kWh"]},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
