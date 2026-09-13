#!/usr/bin/env python
"""问题三 日内负载修正前向预测对照实验 —— 计算层。

任务书 ``reports/问题三/问题三_日内负载修正前向预测对照实验方案.md``。

只回答一件事：**当天已经观测到的负载偏差，能否在真实信息边界下改善未来 10 分钟级负载预测，
并把「一般历史偏差校正」与「当天新增信息」的作用分开。**

三组（固定，不搜索）
--------------------
* ``F0_original``        完全复用 0:00 原预测（Baseline）；
* ``F1_hist_mean``      原预测 + 历史估计截距 ``ȳ``（仅历史偏差校正）；
* ``F2_intraday_linear`` 原预测 + 历史估计截距 + 历史估计斜率 × 当天近期偏差（加入当天信息）。

主对比 F2−F0 与 F2−F1，另报 F1−F0。**F2 优于 F0 但不优于 F1，就不能认为当天新增信息带来额外改进。**

职责边界
--------
本脚本只做：登记 → 读冻结预测与真值 → 形成原残差与 b/y → 逐日闭式滚动参数 → 生成三组全年
发布预测 → 正式期配对评分 → 三类精简检查 → 落盘 CSV/NPZ/JSON。**不写报告、不画图**；
报告与图表由 ``code/q3_intraday_load_forward_report.py`` 只读渲染，改文案永远不会触发本脚本。

预算：LightGBM 训练 0、调度 MILP/LP/DP 求解 0、储能回放 0、公共 1 月调度重跑 0、光伏重建 0、
q75 重建 0、参数网格搜索 0。但**本轮不是“0 拟合”**：确实做了少量滚动一元最小二乘闭式估计，
次数与回退次数单独统计并落盘（``fit_counts``）。

关键口径
--------
* ``e[k,h] = truth_load[k,h] - issued_load[k,h]``（kW，正=负载被低估）；
* 唯一近期特征 ``b[k,r] = mean(e[k, s_r-6 .. s_r-1])``，6 段必须全有效，不足不外扩；
* 未来标签 ``y[k,r,j] = mean(e[k, s_r+6j .. s_r+6j+5])``（j=0 是发布后第一个小时），
  午夜尾槽 ``y_tail[k,r] = e[k,144]`` 单列，共 36 个小时参数组 + 3 个尾槽参数组；
* 历史集合 ``H = {d : max(0,k-28) <= d < k，b 与 y 有效，且所需真实区间已全部结束}``，
  结束时间以当天 r 时为界并显式断言；**不取当天后续标签、不向更早补足样本**；
* F1 与 F2 使用**完全相同**的历史日集合；标签始终是当时冻结的 0:00 原预测误差，
  不用本组新预测残差作回归标签；
* ``m >= 14`` 且历史 b 方差 ``> 1e-10 kW^2`` 才估计 ``β``、``a``；``m < 14`` 时 F1/F2 修正为 0
  回退 F0；当前 b 无效时 F2 回退 F1；方差退化时 ``β=0``、``a=ȳ``（F2 等于 F1）；
* 每小时修正统一作用于该小时六个 10 分钟槽，最终功率非负截断，截断前后分别记录；
* 每个节点都相对**原 0:00 预测**修正，不在上一个节点的校正值上累加；0:00 版本保持原预测。

运行::

    E:/Anaconda/envs/math_modeling/python.exe code/q3_intraday_load_forward_experiment.py
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
CODE = ROOT / "code/q3_intraday_load_forward_experiment.py"
OUT = ROOT / "results/q3_intraday_load_forward"
FIG = ROOT / "figures/q3_intraday_load_forward"
PLAN_MD = ROOT / "reports/问题三/问题三_日内负载修正前向预测对照实验方案.md"
REPORT_MD = ROOT / "reports/问题三/问题三_日内负载修正前向预测对照实验结果.md"
BIAS_NPZ = ROOT / "results/q3_bias_correction_diagnostic/bias_forecast_archive.npz"
Q2_NPZ = ROOT / "results/q2_time_mapping/archive_float.npz"
PREV_CODE = ROOT / "code/q3_intraday_load_residual_diagnostic.py"
PREV_PAIRS = ROOT / "results/q3_intraday_load_residual_diagnostic/hourly_pairs.csv"

BASE = pd.Timestamp("2025-01-01")
NATURAL_DAYS = 365
TARGETS = 145
MINUTES_PER_SLOT = 10

EVAL_FIRST, EVAL_LAST = 31, 364
EVAL_DAYS = EVAL_LAST - EVAL_FIRST + 1            # 334

GROUPS = ("F0_original", "F1_hist_mean", "F2_intraday_linear")
COLUMN_OF = {"F0_original": "f0_kW", "F1_hist_mean": "f1_kW", "F2_intraday_linear": "f2_kW"}
PUB_HOURS = (6, 12, 18)
SLOT_OF = {6: 36, 12: 72, 18: 108}
VERSION_OF = {6: 1, 12: 2, 18: 3}
MAX_FUTURE_HOURS = {6: 18, 12: 12, 18: 6}
A_SLOTS_PER_NODE = 36

RECENT_SLOTS = 6
HISTORY_DAYS = 28
MIN_SAMPLES = 14
VAR_TOL = 1e-10
ABS_TOL = 1e-8

PERTURB_DAY = 171                                 # 2025-06-21
PERTURB_DELTA_KW = 25.0

N_A_TARGETS = EVAL_DAYS * len(PUB_HOURS) * A_SLOTS_PER_NODE                  # 36072
N_B_RECORDS = EVAL_DAYS * sum(TARGETS - SLOT_OF[r] for r in PUB_HOURS)       # 73146
N_TAIL_RECORDS = EVAL_DAYS * len(PUB_HOURS)                                  # 1002
N_PARAM_GROUPS_PER_DAY = sum(MAX_FUTURE_HOURS[r] + 1 for r in PUB_HOURS)     # 39

NON_COMPUTED_ARTIFACTS = frozenset({"run_manifest.json", "figure_integrity.json", "run.log"})

PARAMETERS = dict(
    experiment="Q3 intraday load-correction forward forecast comparison (F0/F1/F2)",
    specification=str(PLAN_MD.relative_to(ROOT)),
    groups=dict(F0_original="frozen 0:00 forecast reused unchanged",
                F1_hist_mean="0:00 forecast + rolling historical mean residual ybar",
                F2_intraday_linear="0:00 forecast + rolling historical intercept a + rolling "
                                   "historical slope beta times the same-day recent one-hour bias b"),
    target_rule="h=0 current 00:00-00:10; h=1..143 current 00:10..23:50; h=144 next 00:00-00:10",
    residual="e[k,h] = truth_load[k,h] - issued_load[k,h] (kW); positive = load under-forecast",
    recent_feature="b[k,r] = mean(e[k, s_r-6 .. s_r-1]), s_r = 6r; all six slots must be valid",
    labels="y[k,r,j] = mean(e[k, s_r+6j .. s_r+6j+5]) for j = 0..23-r; the midnight tail "
           "y_tail[k,r] = e[k,144] is a separate 10-minute slot; 36 hour groups + 3 tail groups",
    history=dict(window_days=HISTORY_DAYS, min_samples=MIN_SAMPLES,
                 definition="d in [max(0,k-28), k-1], b and y valid for that group, and the "
                            "required true intervals already ended at the day-k r:00 instant",
                 rule="F1 and F2 share exactly the same history set, filtered on b validity too; "
                      "the label is always the frozen 0:00 forecast error",
                 no_backfill="early invalid dates are never replaced to reach 28 samples; one "
                             "history date contributes exactly one (b, y) point"),
    estimation="closed-form univariate least squares with intercept, no regularisation and no grid "
               "search; beta is not constrained to be positive, below one, or decaying",
    fallback=dict(m_below_min="F1 and F2 corrections are both 0 and fall back to F0",
                  b_invalid="F2 falls back to F1",
                  degenerate_variance=f"var(b) <= {VAR_TOL:g} kW^2 -> beta = 0, a = ybar, F2 = F1"),
    application="one correction per full hour applied to its six 10-minute slots; final power "
                "clipped at zero; every node corrects the original 0:00 forecast without "
                "accumulating earlier node corrections; the 0:00 version keeps the original "
                "forecast including the midnight auxiliary slot and the template tail",
    evaluation=dict(main_set="A = {(k,r,h): formal 334 days, r in {6,12,18}, s_r <= h < s_r+36}, "
                             f"{N_A_TARGETS} unique realised 10-minute targets, 06:00-24:00 only",
                    secondary_set="B = each publication's full future window (109/73/37 slots), "
                                  f"{N_B_RECORDS} records with overlapping targets: diagnostic only",
                    tail=f"{N_TAIL_RECORDS} midnight-tail records over {EVAL_DAYS} distinct "
                         "realised midnight intervals predicted by three versions",
                    metrics="MAE / RMSE / Bias / mean positive residual / mean negative residual "
                            "magnitude, all with the same N as denominator; the primary metric is "
                            "the A-set 10-minute MAE and is not switched after the fact",
                    hourly="six clipped published predictions are averaged first, then compared "
                           "with the realised hourly mean",
                    paired="group differences are paired record by record"),
    gate=(f"primary gate is the A-set 10-minute MAE; differences <= {ABS_TOL:g} kW are numerically "
          "tied and are not called an improvement. F2 must beat both F0 and F1 to be called better "
          "on the primary metric in this one-year forward backtest; no automatic model upgrade."),
    solver="closed-form arithmetic only: 0 LightGBM fits, 0 MILP/LP/DP, 0 storage replay, "
           "0 public-January rerun, 0 PV rebuild, 0 q75 rebuild, 0 grid search",
    validation="three light checks only: input/history-eligibility/boundary, local closed-form and "
               "fallback, archive and scoring read-back",
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


def masked_max_abs(left, right, label):
    """``max |left-right|`` over cells published in both, after proving the masks agree.

    Legacy NaN cells are legitimate (a 6/12/18 publication does not reach its past slots), so a
    plain subtraction would return NaN and a plain ``np.max`` would silently pass the check.
    """
    lp, rp = np.isnan(left), np.isnan(right)
    if not np.array_equal(lp, rp):
        raise AssertionError(f"{label}: {int((lp != rp).sum())} cell(s) with differing "
                             f"applicability mask")
    sel = ~lp
    if not sel.any():
        return 0.0
    return float(np.abs(left[sel] - right[sel]).max())


def same_nan_mask(left, right, label):
    lp, rp = np.isnan(left), np.isnan(right)
    if not np.array_equal(lp, rp):
        raise AssertionError(f"{label}: {int((lp != rp).sum())} cell(s) with differing NaN mask")
    return int(np.isfinite(left).sum())


def day_of(k):
    return BASE + pd.Timedelta(days=int(k))


def slot_end(k, slot_exclusive):
    return day_of(k) + pd.Timedelta(minutes=MINUTES_PER_SLOT * int(slot_exclusive))


def publication_instant(k, r):
    return day_of(k) + pd.Timedelta(minutes=MINUTES_PER_SLOT * SLOT_OF[r])


def label_end_slot(r, j):
    """Exclusive end slot of a label; the tail is the single slot h=144 -> end slot 145."""
    return TARGETS if j is None else SLOT_OF[r] + 6 * j + 6


# ======================================================================================
# §2  输入、原残差与特征
# ======================================================================================
def read_inputs():
    with np.load(BIAS_NPZ) as archive:
        issued = archive["issued_load"].astype(float)
        truth = archive["truth_load"].astype(float)
    with np.load(Q2_NPZ) as archive:
        issued_ref = archive["issued_load"].astype(float)
    assert issued.shape == (NATURAL_DAYS, TARGETS) == truth.shape == issued_ref.shape, issued.shape

    cells = same_nan_mask(issued, issued_ref, "issued_load vs Q2 kernel archive")
    diff = np.abs(np.where(np.isnan(issued), 0.0, issued)
                  - np.where(np.isnan(issued_ref), 0.0, issued_ref))
    max_diff = float(diff.max())
    assert max_diff <= ABS_TOL, f"issued_load archives disagree by {max_diff}"

    issued_eval = issued[EVAL_FIRST:EVAL_LAST + 1]
    truth_eval = truth[EVAL_FIRST:EVAL_LAST + 1]
    if not np.isfinite(issued_eval).all():
        raise AssertionError(f"issued_load: {int((~np.isfinite(issued_eval)).sum())} non-finite "
                             f"cell(s) inside the formal window")
    if not np.isfinite(truth_eval).all():
        raise AssertionError(f"truth_load: {int((~np.isfinite(truth_eval)).sum())} non-finite "
                             f"cell(s) inside the formal window")
    if np.nanmin(issued) < 0 or np.nanmin(truth) < 0:
        raise AssertionError("negative load found in the frozen archive")

    return dict(issued_load=issued, truth_load=truth, residual=truth - issued,
                cross_check=dict(cells=int(issued.size), agreeing_finite_cells=cells,
                                 max_abs_difference_kW=max_diff,
                                 formal_cells=int(issued_eval.size)))


def build_features(residual):
    """b (365,3), y (365,3,18), y_tail (365,3); NaN wherever a slot is not usable."""
    b = np.full((NATURAL_DAYS, len(PUB_HOURS)), np.nan)
    y = np.full((NATURAL_DAYS, len(PUB_HOURS), max(MAX_FUTURE_HOURS.values())), np.nan)
    y_tail = np.full((NATURAL_DAYS, len(PUB_HOURS)), np.nan)
    for ri, r in enumerate(PUB_HOURS):
        s = SLOT_OF[r]
        block = residual[:, s - RECENT_SLOTS:s]
        ok = np.isfinite(block).all(axis=1)
        b[ok, ri] = block[ok].mean(axis=1)
        for j in range(MAX_FUTURE_HOURS[r]):
            cells = residual[:, s + 6 * j:s + 6 * j + 6]
            good = np.isfinite(cells).all(axis=1)
            y[good, ri, j] = cells[good].mean(axis=1)
        tail = residual[:, TARGETS - 1]
        finite = np.isfinite(tail)
        y_tail[finite, ri] = tail[finite]
    return dict(b=b, y=y, y_tail=y_tail)


def group_keys():
    """(r, kind, j) for the 36 hour groups and 3 tail groups; j=None marks the tail slot."""
    keys = {r: [] for r in PUB_HOURS}
    for r in PUB_HOURS:
        for j in range(MAX_FUTURE_HOURS[r]):
            keys[r].append((r, "hour", j))
        keys[r].append((r, "tail", None))
    return keys


# ======================================================================================
# §3  滚动闭式参数
# ======================================================================================
def history_for(features, k, ri, j):
    dates, bs, ys = [], [], []
    for d in range(max(0, k - HISTORY_DAYS), k):
        bv = features["b"][d, ri]
        yv = features["y_tail"][d, ri] if j is None else features["y"][d, ri, j]
        if not (np.isfinite(bv) and np.isfinite(yv)):
            continue
        dates.append(d)
        bs.append(float(bv))
        ys.append(float(yv))
    return dates, np.asarray(bs), np.asarray(ys)


def fit_group(k, r, ri, j, features):
    """One closed-form rolling univariate OLS for a (day, publication node, future hour|tail) group."""
    dates, b_hist, y_hist = history_for(features, k, ri, j)
    m = len(dates)
    b_now = features["b"][k, ri]
    b_ok = bool(np.isfinite(b_now))
    record = dict(k=k, date=day_of(k).strftime("%Y-%m-%d"), publication_hour=r,
                  kind="tail" if j is None else "hour", j=-1 if j is None else int(j), m=m,
                  history_dates="|".join(day_of(d).strftime("%Y-%m-%d") for d in dates),
                  history_first=day_of(dates[0]).strftime("%Y-%m-%d") if dates else "",
                  history_last=day_of(dates[-1]).strftime("%Y-%m-%d") if dates else "",
                  latest_label_end=(slot_end(dates[-1], label_end_slot(r, j)).strftime(
                      "%Y-%m-%d %H:%M") if dates else ""),
                  b_now_kW=float(b_now) if b_ok else float("nan"))
    mean_b = float(b_hist.mean()) if m else float("nan")
    mean_y = float(y_hist.mean()) if m else float("nan")
    var_b = float(b_hist.var()) if m else float("nan")
    beta = a = float("nan")
    fitted = False
    if m >= MIN_SAMPLES and math.isfinite(var_b) and var_b > VAR_TOL:
        centred = b_hist - mean_b
        beta = float(centred @ (y_hist - mean_y) / (centred @ centred))
        a = mean_y - beta * mean_b
        # the same-day feature being unusable sends F2 back to F1 instead of producing NaN
        delta1 = mean_y
        delta2 = (a + beta * float(b_now)) if b_ok else mean_y
        fitted = True
        fallback = "" if b_ok else "b_invalid"
    elif m >= MIN_SAMPLES:
        beta, a = 0.0, mean_y
        delta1 = delta2 = mean_y
        fallback = "degenerate_variance" if b_ok else "degenerate_variance+b_invalid"
    else:
        delta1 = delta2 = 0.0
        fallback = "m_below_min" if b_ok else "m_below_min+b_invalid"
    if fitted and not (math.isfinite(beta) and math.isfinite(a) and math.isfinite(delta2)):
        raise AssertionError(f"non-finite parameter at k={k} r={r} j={j}")
    record.update(mean_b_kW=mean_b, var_b_kW2=var_b, mean_y_kW=mean_y, beta=beta,
                  intercept_a_kW=a, delta1_kW=delta1, delta2_kW=delta2,
                  fitted=fitted, fallback_reason=fallback)
    return record


def build_forecasts(issued, features, keys):
    """Three publication archives plus the raw corrections and the fit bookkeeping."""
    forecast = np.full((3, NATURAL_DAYS, 4, TARGETS), np.nan)
    correction = np.full((2, NATURAL_DAYS, 4, TARGETS), np.nan)   # raw F1 / F2 corrections
    valid = np.zeros((NATURAL_DAYS, 4, TARGETS), dtype=bool)

    forecast[:, :, 0, :] = issued[None, :, :]
    correction[:, :, 0, :] = 0.0
    valid[:, 0, :] = np.isfinite(issued)

    log_rows = []
    fits = dict(groups=int(NATURAL_DAYS * N_PARAM_GROUPS_PER_DAY), fitted=0,
                fallback_m_below_min=0, fallback_degenerate_variance=0, fallback_b_invalid=0)
    for k in range(NATURAL_DAYS):
        for r in PUB_HOURS:
            ri, vi, s = PUB_HOURS.index(r), VERSION_OF[r], SLOT_OF[r]
            for (_, kind, j) in keys[r]:
                record = fit_group(k, r, ri, j, features)
                log_rows.append(record)
                if record["fitted"]:
                    fits["fitted"] += 1
                elif record["fallback_reason"].startswith("m_below_min"):
                    fits["fallback_m_below_min"] += 1
                else:
                    fits["fallback_degenerate_variance"] += 1
                if "b_invalid" in record["fallback_reason"]:
                    fits["fallback_b_invalid"] += 1
                slots = [TARGETS - 1] if j is None else list(range(s + 6 * j, s + 6 * j + 6))
                for h in slots:
                    correction[0, k, vi, h] = record["delta1_kW"]
                    correction[1, k, vi, h] = record["delta2_kW"]
            reach = np.arange(s, TARGETS)
            forecast[0, k, vi, reach] = issued[k, reach]
            forecast[1, k, vi, reach] = np.maximum(0.0, issued[k, reach] + correction[0, k, vi, reach])
            forecast[2, k, vi, reach] = np.maximum(0.0, issued[k, reach] + correction[1, k, vi, reach])
            valid[k, vi, reach] = np.isfinite(forecast[0, k, vi, reach]) & \
                np.isfinite(forecast[1, k, vi, reach]) & np.isfinite(forecast[2, k, vi, reach])
    return forecast, correction, valid, pd.DataFrame(log_rows), fits


# ======================================================================================
# §4  评价
# ======================================================================================
def build_pairs(forecast, correction, truth):
    """Set B: every publication-target record of the formal window, overlapping targets kept."""
    rows = []
    for k in range(EVAL_FIRST, EVAL_LAST + 1):
        for r in PUB_HOURS:
            vi, s = VERSION_OF[r], SLOT_OF[r]
            for h in range(s, TARGETS):
                j = None if h == TARGETS - 1 else (h - s) // 6
                raw1 = float(correction[0, k, vi, h])
                raw2 = float(correction[1, k, vi, h])
                f0 = float(forecast[0, k, vi, h])
                f1 = float(forecast[1, k, vi, h])
                f2 = float(forecast[2, k, vi, h])
                short1 = max(0.0, -raw1) if f1 == 0.0 else 0.0
                short2 = max(0.0, -raw2) if f2 == 0.0 else 0.0
                rows.append(dict(
                    k=k, date=day_of(k).strftime("%Y-%m-%d"), month=day_of(k).strftime("%Y-%m"),
                    publication_hour=r, version=vi,
                    kind="tail" if j is None else "hour", j=-1 if j is None else int(j),
                    h=h, lead_hours_start=(h - s) / 6.0,
                    hour_index=None if j is None else int(j) + 1,
                    in_A=bool(h < s + A_SLOTS_PER_NODE), is_tail=bool(j is None),
                    actual_kW=float(truth[k, h]),
                    f0_kW=f0, f1_kW=f1, f2_kW=f2,
                    delta1_raw_kW=raw1, delta2_raw_kW=raw2,
                    delta1_eff_kW=f1 - f0, delta2_eff_kW=f2 - f0,
                    truncated_f1=bool(short1 > 0.0), truncated_f2=bool(short2 > 0.0),
                    truncation_shortfall_f1_kW=short1, truncation_shortfall_f2_kW=short2))
    return pd.DataFrame(rows)


def metrics(actual, predicted):
    residual = actual - predicted
    return dict(n=int(residual.size), mae_kW=float(np.abs(residual).mean()),
                rmse_kW=float(np.sqrt((residual ** 2).mean())), bias_kW=float(residual.mean()),
                mean_under_kW=float(np.clip(residual, 0, None).mean()),
                mean_over_kW=float(np.clip(-residual, 0, None).mean()))


def build_metrics(pairs):
    sets = {"A": pairs[pairs.in_A], "B": pairs, "tail": pairs[pairs.is_tail],
            "B_nontail": pairs[~pairs.is_tail]}
    summary_rows, grouped_rows = [], []
    for set_name, block in sets.items():
        for group in GROUPS:
            summary_rows.append(dict(set=set_name, group=group,
                                     **metrics(block.actual_kW.to_numpy(),
                                               block[COLUMN_OF[group]].to_numpy())))
            for label, key in (("node", "publication_hour"), ("month", "month"), ("day", "date")):
                for value, sub in block.groupby(key, sort=True):
                    residual = sub.actual_kW.to_numpy() - sub[COLUMN_OF[group]].to_numpy()
                    grouped_rows.append(dict(
                        set=set_name, group=group, group_type=label, group_value=str(value),
                        n=int(len(sub)), mae_kW=float(np.abs(residual).mean()),
                        rmse_kW=float(np.sqrt((residual ** 2).mean())),
                        bias_kW=float(residual.mean())))
    a = pairs[pairs.in_A]
    hourly = []
    for (k, r, j), block in a.groupby(["k", "publication_hour", "j"], sort=True):
        actual_mean = float(block.actual_kW.mean())
        for group in GROUPS:
            predicted_mean = float(block[COLUMN_OF[group]].mean())
            hourly.append(dict(set="A_hourly", group=group, group_type="hour",
                               group_value=f"{k}_{r}_{j}", n=int(len(block)),
                               mae_kW=abs(actual_mean - predicted_mean),
                               rmse_kW=abs(actual_mean - predicted_mean),
                               bias_kW=actual_mean - predicted_mean))
    return (pd.DataFrame(summary_rows),
            pd.concat([pd.DataFrame(grouped_rows), pd.DataFrame(hourly)], ignore_index=True))


def build_contrasts(pairs):
    a = pairs[pairs.in_A]
    actual = a.actual_kW.to_numpy()
    diff = {g: actual - a[COLUMN_OF[g]].to_numpy() for g in GROUPS}
    rows = []
    for name, left, right in (("F2_minus_F0", "F2_intraday_linear", "F0_original"),
                              ("F2_minus_F1", "F2_intraday_linear", "F1_hist_mean"),
                              ("F1_minus_F0", "F1_hist_mean", "F0_original")):
        l_abs, r_abs = np.abs(diff[left]), np.abs(diff[right])
        improved = int((l_abs < r_abs - ABS_TOL).sum())
        worsened = int((l_abs > r_abs + ABS_TOL).sum())
        work = a.assign(_l=l_abs, _r=r_abs)
        daily = work.groupby("date", sort=True).apply(
            lambda b: b._l.mean() - b._r.mean(), include_groups=False)
        monthly = work.groupby("month", sort=True).apply(
            lambda b: b._l.mean() - b._r.mean(), include_groups=False)
        worst = daily.sort_values(ascending=False).head(3)
        rows.append(dict(
            contrast=name, left=left, right=right, n=int(len(a)),
            delta_mae_kW=float(l_abs.mean() - r_abs.mean()),
            delta_rmse_kW=float(np.sqrt((diff[left] ** 2).mean())
                                - np.sqrt((diff[right] ** 2).mean())),
            delta_bias_kW=float(diff[left].mean() - diff[right].mean()),
            targets_improved=improved, targets_worsened=worsened,
            targets_tied=int(len(a) - improved - worsened),
            days_improved=int((daily < -ABS_TOL).sum()), days_worsened=int((daily > ABS_TOL).sum()),
            months_improved=int((monthly < -ABS_TOL).sum()),
            months_worsened=int((monthly > ABS_TOL).sum()),
            worst_days="|".join(f"{d}:{v:+.6f}" for d, v in worst.items()),
            best_day=f"{daily.idxmin()}:{daily.min():+.6f}"))
    return pd.DataFrame(rows)


def build_parameter_summary(log, pairs):
    """Per (publication hour, hour|tail) parameter diagnostics plus two whole-run rows."""
    rows = []
    fitted = log[log.fitted]
    for (r, kind), block in log.groupby(["publication_hour", "kind"], sort=True):
        sub = block[block.fitted]
        rows.append(dict(
            publication_hour=int(r), kind=kind, groups=int(len(block)),
            fitted=int(block.fitted.sum()),
            fallback_m_below_min=int(block.fallback_reason.str.startswith("m_below_min").sum()),
            fallback_degenerate_variance=int(
                block.fallback_reason.str.startswith("degenerate_variance").sum()),
            fallback_b_invalid=int(block.fallback_reason.str.contains("b_invalid").sum()),
            m_min=int(block.m.min()), m_median=float(block.m.median()), m_max=int(block.m.max()),
            beta_mean=float(sub.beta.mean()) if len(sub) else float("nan"),
            beta_min=float(sub.beta.min()) if len(sub) else float("nan"),
            beta_max=float(sub.beta.max()) if len(sub) else float("nan"),
            beta_negative_share=float((sub.beta < 0).mean()) if len(sub) else float("nan"),
            delta1_mean_kW=float(block.delta1_kW.mean()),
            delta1_max_abs_kW=float(block.delta1_kW.abs().max()),
            delta2_mean_kW=float(block.delta2_kW.mean()),
            delta2_min_kW=float(block.delta2_kW.min()),
            delta2_max_kW=float(block.delta2_kW.max()),
            delta2_max_abs_date=str(block.loc[block.delta2_kW.abs().idxmax(), "date"]),
            truncation_hits_f1="", truncation_hits_f2="",
            truncation_max_shortfall_kW_f1="", truncation_max_shortfall_kW_f2=""))
    def whole(kind, blocks, truncation=None):
        row = dict(
            publication_hour=-1, kind=kind, groups=int(len(blocks)),
            fitted=int(blocks.fitted.sum()),
            fallback_m_below_min=int(blocks.fallback_reason.str.startswith("m_below_min").sum()),
            fallback_degenerate_variance=int(
                blocks.fallback_reason.str.startswith("degenerate_variance").sum()),
            fallback_b_invalid=int(blocks.fallback_reason.str.contains("b_invalid").sum()),
            m_min=int(blocks.m.min()), m_median=float(blocks.m.median()), m_max=int(blocks.m.max()),
            beta_mean=float(blocks[blocks.fitted].beta.mean()),
            beta_min=float(blocks[blocks.fitted].beta.min()),
            beta_max=float(blocks[blocks.fitted].beta.max()),
            beta_negative_share=float((blocks[blocks.fitted].beta < 0).mean()),
            delta1_mean_kW=float(blocks.delta1_kW.mean()),
            delta1_max_abs_kW=float(blocks.delta1_kW.abs().max()),
            delta2_mean_kW=float(blocks.delta2_kW.mean()),
            delta2_min_kW=float(blocks.delta2_kW.min()),
            delta2_max_kW=float(blocks.delta2_kW.max()),
            delta2_max_abs_date=str(blocks.loc[blocks.delta2_kW.abs().idxmax(), "date"]),
            truncation_hits_f1="", truncation_hits_f2="",
            truncation_max_shortfall_kW_f1="", truncation_max_shortfall_kW_f2="")
        if truncation is not None:
            row.update(truncation_hits_f1=int(truncation.truncated_f1.sum()),
                       truncation_hits_f2=int(truncation.truncated_f2.sum()),
                       truncation_max_shortfall_kW_f1=float(truncation.truncation_shortfall_f1_kW.max()),
                       truncation_max_shortfall_kW_f2=float(truncation.truncation_shortfall_f2_kW.max()))
        return row
    rows.append(whole("all_groups", log))
    rows.append(whole("A_set_truncation", log, pairs[pairs.in_A]))
    frame = pd.DataFrame(rows)
    return frame, frame[frame.kind == "A_set_truncation"].iloc[0].to_dict()


# ======================================================================================
# §5  三类精简验证
# ======================================================================================
def check_inputs_and_history(data, features, log):
    checks = {"cross_check": data["cross_check"]}
    assert (EVAL_FIRST, EVAL_LAST, EVAL_DAYS) == (31, 364, 334)
    checks["formal_window"] = dict(first=day_of(EVAL_FIRST).strftime("%Y-%m-%d"),
                                   last=day_of(EVAL_LAST).strftime("%Y-%m-%d"), days=EVAL_DAYS)
    checks["residual_identity"] = dict(max_abs_difference_kW=known_max_abs(
        (data["truth_load"] - data["issued_load"])[EVAL_FIRST:EVAL_LAST + 1]
        - data["residual"][EVAL_FIRST:EVAL_LAST + 1], "e identity"))

    kept = log[log.m > 0].copy()
    kept["last_day"] = kept.history_last.map(lambda s: (pd.Timestamp(s) - BASE).days)
    gap = int((kept.k - kept.last_day).min())
    assert gap >= 1, "history must exclude the publication day"
    nonempty = log[log.latest_label_end != ""]
    publication = nonempty.apply(lambda row: publication_instant(row.k, row.publication_hour),
                                 axis=1)
    late = int((pd.to_datetime(nonempty.latest_label_end) > publication).sum())
    assert late == 0, f"{late} history label(s) end after the publication instant"
    checks["history_eligibility"] = dict(
        groups=int(len(log)), max_history_samples=int(log.m.max()),
        min_history_samples=int(log.m.min()), min_gap_days_to_publication=gap,
        labels_ending_after_publication=late, groups_with_empty_history=int((log.m == 0).sum()),
        rule="every history day d satisfies max(0,k-28) <= d < k and its whole label interval has "
             "ended at the day-k r:00 publication instant; F1 and F2 share the same set")

    previous = pd.read_csv(PREV_PAIRS)
    previous = previous[previous.kind != "midnight_tail"]
    worst_b = worst_y = 0.0
    for (k, r, j), block in previous.groupby(["k", "publication_hour", "j"], sort=True):
        ri = PUB_HOURS.index(int(r))
        worst_b = max(worst_b, abs(float(block.b_kW.iloc[0]) - float(features["b"][int(k), ri])))
        worst_y = max(worst_y, abs(float(block.y_kW.iloc[0])
                                   - float(features["y"][int(k), ri, int(j)])))
    assert worst_b <= ABS_TOL and worst_y <= ABS_TOL, (worst_b, worst_y)
    checks["feature_definition_vs_previous_diagnostic"] = dict(
        max_abs_b_difference_kW=worst_b, max_abs_y_difference_kW=worst_y,
        note="the rolling window and the label definition are unchanged from the reviewed "
             "diagnostic; this round changes only what is estimated from them")
    return checks


def check_boundary(data, features, forecast, keys):
    """2025-06-21: perturbing the truth at/after a node's instant must not move that node's output."""
    k = PERTURB_DAY
    assert day_of(k).strftime("%Y-%m-%d") == "2025-06-21"
    rows, published, downstream = [], [], []
    for r in PUB_HOURS:
        ri, vi, s = PUB_HOURS.index(r), VERSION_OF[r], SLOT_OF[r]
        # fresh copy per node: a shared array would carry one node's shift into the next
        shifted = data["truth_load"].copy()
        shifted[k, s:] += PERTURB_DELTA_KW
        shifted_features = build_features(shifted - data["issued_load"])
        param_worst = 0.0
        for j in [jj for (_, _, jj) in keys[r]]:
            before = fit_group(k, r, ri, j, features)
            after = fit_group(k, r, ri, j, shifted_features)
            for key in ("beta", "intercept_a_kW", "delta1_kW", "delta2_kW"):
                param_worst = max(param_worst, abs(float(before[key]) - float(after[key])))
        shifted_forecast, _, _, _, _ = build_forecasts(data["issued_load"], shifted_features, keys)
        # only THIS node's own publication must stay fixed. Perturbing the truth from the 06:00
        # instant also moves the 12:00 node's same-day feature, which is a legitimate downstream
        # consequence of the information boundary, not a leak into the perturbed node.
        published.append(masked_max_abs(shifted_forecast[:, k, vi, s:],
                                        forecast[:, k, vi, s:], "published"))
        downstream.append(masked_max_abs(shifted_forecast[:, k, vi + 1:, :],
                                         forecast[:, k, vi + 1:, :], "downstream")
                          if vi + 1 <= 3 else 0.0)
        rows.append(dict(
            publication_hour=r, s_r=s,
            b_before_kW=float(features["b"][k, ri]),
            b_after_shift_kW=float(shifted_features["b"][k, ri]),
            b_abs_change_kW=abs(float(shifted_features["b"][k, ri]) - float(features["b"][k, ri])),
            parameter_max_abs_change=param_worst,
            published_forecast_max_abs_change=float(published[-1]),
            downstream_publication_max_abs_change=float(downstream[-1]),
            y_first_hour_abs_change_kW=abs(float(shifted_features["y"][k, ri, 0])
                                           - float(features["y"][k, ri, 0])),
            tail_abs_change_kW=abs(float(shifted_features["y_tail"][k, ri])
                                   - float(features["y_tail"][k, ri]))))
    frame = pd.DataFrame(rows)
    assert (frame.b_abs_change_kW <= ABS_TOL).all(), frame
    assert (frame.parameter_max_abs_change <= ABS_TOL).all(), frame
    assert (frame.published_forecast_max_abs_change <= ABS_TOL).all(), frame
    assert np.allclose(frame.y_first_hour_abs_change_kW, PERTURB_DELTA_KW, atol=ABS_TOL), frame
    assert np.allclose(frame.tail_abs_change_kW, PERTURB_DELTA_KW, atol=ABS_TOL), frame
    return dict(day="2025-06-21", k=k, perturb_kW=PERTURB_DELTA_KW,
                rule="for each node the truth at and after its publication instant is shifted in "
                     "memory on a fresh copy; that node's b, its estimated parameters and its "
                     "published forecasts must not move, while the evaluation labels do",
                rows=frame.to_dict("records"),
                max_abs_b_change_kW=known_max_abs(frame.b_abs_change_kW, "b change"),
                max_abs_parameter_change=known_max_abs(frame.parameter_max_abs_change, "parameter"),
                max_abs_published_forecast_change=known_max_abs(
                    frame.published_forecast_max_abs_change, "published"),
                all_labels_moved=bool(np.allclose(frame.y_first_hour_abs_change_kW,
                                                  PERTURB_DELTA_KW, atol=ABS_TOL)),
                note="only the perturbed node's own publication is required to stay fixed; a later "
                     "node's same-day feature legitimately moves when an earlier instant is "
                     "perturbed, which the downstream column records")


def check_closed_form_and_fallback(features, log):
    recomputed, worst = [], 0.0
    for r, j in ((6, 0), (18, 0)):
        row = log[(log.k == PERTURB_DAY) & (log.publication_hour == r) & (log.j == j)].iloc[0]
        dates = [(pd.Timestamp(s) - BASE).days for s in row.history_dates.split("|")]
        ri = PUB_HOURS.index(r)
        b_hist = np.array([features["b"][d, ri] for d in dates], dtype=float)
        y_hist = np.array([features["y"][d, ri, j] for d in dates], dtype=float)
        design = np.column_stack([np.ones(len(b_hist)), b_hist])
        coefficients, *_ = np.linalg.lstsq(design, y_hist, rcond=None)   # a different algebra path
        a_hat, beta_hat = float(coefficients[0]), float(coefficients[1])
        delta2_hat = a_hat + beta_hat * float(features["b"][PERTURB_DAY, ri])
        for got, want in ((row.intercept_a_kW, a_hat), (row.beta, beta_hat),
                          (row.delta2_kW, delta2_hat)):
            worst = max(worst, abs(float(got) - want))
        recomputed.append(dict(publication_hour=r, j=int(j), m=int(row.m), a_kW=a_hat,
                               beta=beta_hat, delta2_kW=delta2_hat,
                               saved_delta2_kW=float(row.delta2_kW)))
    assert worst <= 1e-9, worst

    flat = np.full(HISTORY_DAYS, 3.0)
    rising = np.arange(HISTORY_DAYS, dtype=float)
    assert float(flat.var()) <= VAR_TOL
    centred = rising - rising.mean()
    beta_rising = float(centred @ centred / (centred @ centred))
    assert abs(beta_rising - 1.0) <= 1e-12
    assert MIN_SAMPLES == 14 and (MIN_SAMPLES - 1) < MIN_SAMPLES
    return dict(recomputed_groups=recomputed, max_abs_closed_form_difference=worst,
                degenerate_b=dict(var_b_kW2=float(flat.var()), beta=0.0,
                                  a_kW=float(rising.mean()), f2_equals_f1=True),
                perfect_positive_slope=beta_rising,
                min_samples_gate=dict(below=int(MIN_SAMPLES - 1), at=int(MIN_SAMPLES),
                                      rule="m < 14 -> both corrections are 0 and both fall back"),
                non_negative_clip="max(0, forecast + correction) keeps the published power >= 0")


def check_readback():
    with np.load(OUT / "forecast_archive.npz") as archive:
        forecast, valid = archive["forecast"], archive["valid"]
        correction, issued = archive["correction_raw"], archive["issued_load"]
    pairs = pd.read_csv(OUT / "prediction_pairs.csv")
    summary = pd.read_csv(OUT / "summary.csv")
    contrasts = pd.read_csv(OUT / "contrasts.csv")
    checks = {}

    checks["version0_max_abs_difference_from_baseline_kW"] = masked_max_abs(
        forecast[:, :, 0, :], np.broadcast_to(issued[None], forecast[:, :, 0, :].shape), "v0")
    assert checks["version0_max_abs_difference_from_baseline_kW"] <= ABS_TOL

    # each publication publishes only its reach; below it nothing is published
    for r in PUB_HOURS:
        s, vi = SLOT_OF[r], VERSION_OF[r]
        assert not valid[:, vi, :s].any(), f"version {r}: past slots must be not applicable"
        rebuilt = np.maximum(0.0, issued[None, :, :] + correction[:, :, vi, :])
        checks[f"version{r}_max_abs_difference_kW"] = masked_max_abs(
            forecast[1:, :, vi, s:], rebuilt[:, :, s:], f"v{r} rebuild")
        assert checks[f"version{r}_max_abs_difference_kW"] <= ABS_TOL

    published = np.concatenate([forecast[0][:, :, 1:].ravel(), forecast[1][:, :, 1:].ravel(),
                                forecast[2][:, :, 1:].ravel()])
    assert float(np.nanmin(published)) >= 0.0
    checks["counts"] = dict(
        A_targets=int(pairs.in_A.sum()), B_records=int(len(pairs)),
        tail_records=int(pairs.is_tail.sum()),
        unique_A_targets=int(pairs[pairs.in_A].drop_duplicates(["k", "h"]).shape[0]),
        distinct_tail_intervals=int(pairs[pairs.is_tail].drop_duplicates(["k", "h"]).shape[0]),
        duplicate_pair_rows=int(pairs.duplicated(["k", "publication_hour", "h"]).sum()))
    assert checks["counts"]["A_targets"] == N_A_TARGETS, checks["counts"]
    assert checks["counts"]["B_records"] == N_B_RECORDS, checks["counts"]
    assert checks["counts"]["tail_records"] == N_TAIL_RECORDS, checks["counts"]
    assert checks["counts"]["unique_A_targets"] == N_A_TARGETS, checks["counts"]
    assert checks["counts"]["distinct_tail_intervals"] == EVAL_DAYS, checks["counts"]
    assert checks["counts"]["duplicate_pair_rows"] == 0, checks["counts"]

    a = pairs[pairs.in_A]
    actual = a.actual_kW.to_numpy()
    worst = 0.0
    for group in GROUPS:
        stat = metrics(actual, a[COLUMN_OF[group]].to_numpy())
        saved = summary[(summary.set == "A") & (summary.group == group)].iloc[0]
        for key in ("mae_kW", "rmse_kW", "bias_kW", "mean_under_kW", "mean_over_kW"):
            worst = max(worst, abs(float(saved[key]) - stat[key]))
    left = np.abs(actual - a.f2_kW.to_numpy())
    right = np.abs(actual - a.f0_kW.to_numpy())
    saved = contrasts[contrasts.contrast == "F2_minus_F0"].iloc[0]
    worst = max(worst, abs(float(saved.delta_mae_kW) - float(left.mean() - right.mean())))
    assert worst <= 1e-9, worst
    checks["A_metric_readback_max_abs_difference"] = worst

    hourly = pd.read_csv(OUT / "grouped_metrics.csv")
    hourly = hourly[hourly.set == "A_hourly"]
    saved_hourly = {(row.group, row.group_value): float(row.mae_kW)
                    for row in hourly.itertuples(index=False)}
    worst_hour = 0.0
    for (k, r, j), block in a.groupby(["k", "publication_hour", "j"], sort=True):
        for group in GROUPS:
            want = abs(float(block.actual_kW.mean()) - float(block[COLUMN_OF[group]].mean()))
            worst_hour = max(worst_hour, abs(saved_hourly[(group, f"{k}_{r}_{j}")] - want))
    assert worst_hour <= 1e-9, worst_hour
    checks["hourly_metric_max_abs_difference"] = worst_hour
    checks["note"] = ("recomputed only from the written tables; no solver or LightGBM call is "
                      "involved, and no previous full-tree hashes are re-checked")
    return checks


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
        # carry the amendment trail forward even on an idempotent re-run: the defect the earlier
        # read-only diagnostics have is deliberately not copied into this script
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


def verdict_for(mae):
    f0, f1, f2 = mae["F0_original"], mae["F1_hist_mean"], mae["F2_intraday_linear"]
    if f2 < f0 - ABS_TOL and f2 < f1 - ABS_TOL:
        return ("F2 在本年度前向回测中同时优于 F0 与 F1（主指标 A 集 10 分钟 MAE）："
                "可称当天偏差模型比这两个基准的主指标更好，但需再结合 RMSE、各节点、月份"
                "与极端日判断是否值得设计费用对照，不自动升级模型")
    if f2 < f0 - ABS_TOL:
        return "F2 优于 F0 但不优于 F1：只能支持校正组合的作用，不能确认当天特征带来额外收益"
    if f1 < f0 - ABS_TOL:
        return "F1 优于 F0 而 F2 未进一步改善：保留历史偏差校正作为单独线索，不默认启用 F2"
    return "F1/F2 均无优势：保留 F0，结束本轮；不在同一任务追加调参、换窗口或节点筛选"


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

    inputs = ["reports/问题三/问题三_日内负载修正前向预测对照实验方案.md",
              "results/q3_bias_correction_diagnostic/bias_forecast_archive.npz",
              "results/q2_time_mapping/archive_float.npz",
              "code/q3_intraday_load_residual_diagnostic.py",
              "results/q3_intraday_load_residual_diagnostic/hourly_pairs.csv"]
    for rel in inputs:
        assert (ROOT / rel).exists(), rel
    dependency = dict(previous_diagnostic="results/q3_intraday_load_residual_diagnostic/"
                                         "hourly_pairs.csv (feature-definition cross-check only)",
                      zero_budget=dict(lightgbm_training=0, milp_lp_dp=0, storage_replay=0,
                                       public_january_dispatch_reruns=0, pv_rebuild=0,
                                       quantile_rebuild=0, parameter_grid_search=0),
                      note="closed-form univariate rolling fits are NOT zero and are counted "
                           "separately in fit_counts")
    signature = register(PARAMETERS, inputs, dependency, args.amend_reason)
    print(f"registered signature={signature[:16]}", flush=True)
    if args.mode == "register":
        return

    tick = time.perf_counter()
    data = read_inputs()
    features = build_features(data["residual"])
    keys = group_keys()
    timings["inputs_seconds"] = time.perf_counter() - tick
    print(f"inputs: issued/truth {data['issued_load'].shape}; groups "
          f"{sum(len(v) for v in keys.values())}", flush=True)

    tick = time.perf_counter()
    forecast, correction, valid, log, fits = build_forecasts(data["issued_load"], features, keys)
    timings["forecast_seconds"] = time.perf_counter() - tick
    print(f"forecasts built: closed-form fits {fits['fitted']}, m<14 {fits['fallback_m_below_min']}, "
          f"degenerate {fits['fallback_degenerate_variance']}, b invalid "
          f"{fits['fallback_b_invalid']}", flush=True)

    tick = time.perf_counter()
    pairs = build_pairs(forecast, correction, data["truth_load"])
    summary, grouped = build_metrics(pairs)
    contrasts = build_contrasts(pairs)
    param_rows, param_wide = build_parameter_summary(log, pairs)
    timings["scoring_seconds"] = time.perf_counter() - tick

    a = pairs[pairs.in_A]
    actual = a.actual_kW.to_numpy()
    mae = {g: float(np.abs(actual - a[COLUMN_OF[g]].to_numpy()).mean()) for g in GROUPS}
    verdict = verdict_for(mae)
    print(f"A-set MAE: {mae}", flush=True)

    tick = time.perf_counter()
    checks = dict(inputs_and_history=check_inputs_and_history(data, features, log),
                  boundary_date=check_boundary(data, features, forecast, keys),
                  closed_form_and_fallback=check_closed_form_and_fallback(features, log))
    timings["checks_seconds"] = time.perf_counter() - tick
    print("input/history, boundary and closed-form checks passed", flush=True)

    np.savez_compressed(
        OUT / "forecast_archive.npz", forecast=forecast, valid=valid, correction_raw=correction,
        issued_load=data["issued_load"], truth_load=data["truth_load"], residual=data["residual"],
        groups=np.array(GROUPS), publication_hours=np.array(PUB_HOURS),
        versions=np.array([0, *PUB_HOURS]), first_target=np.array([0, *[SLOT_OF[r] for r in PUB_HOURS]]),
        semantics=np.array([
            "forecast/correction axis0 = group F0/F1/F2 and F1/F2; "
            "axis1 = day index k (natural day 2025-01-01 + k); "
            "axis2 = publication version 0:00/6:00/12:00/18:00; "
            "axis3 = target slot h = 0..144 (h=144 is next-day 00:00-00:10); "
            "NaN = not applicable (past slot of a 6/12/18 publication) or invalid; "
            "forecast[g,k,v,h] = max(0, issued_load[k,h] + correction_raw[g-1,k,v,h]) for g>0; "
            "correction_raw is the pre-clip correction and is NOT the effective change"]))
    frame_to_csv(log, OUT / "parameter_log.csv")
    frame_to_csv(pairs, OUT / "prediction_pairs.csv")
    frame_to_csv(summary, OUT / "summary.csv")
    frame_to_csv(contrasts, OUT / "contrasts.csv")
    frame_to_csv(grouped, OUT / "grouped_metrics.csv")
    frame_to_csv(param_rows, OUT / "parameter_summary.csv")

    checks["readback"] = check_readback()
    print("archive and scoring read-back passed", flush=True)

    validation = dict(
        status="passed", signature=signature, checks=checks, fit_counts=fits,
        parameter_summary_A=param_wide,
        primary_metric=dict(set="A (36 slots per publication, 06:00-24:00 only, 36072 targets)",
                            mae_kW=mae, gate_tolerance_kW=ABS_TOL),
        verdict=verdict, zero_budget=dependency["zero_budget"],
        limitations=[
            "点预测误差不是费用：本轮没有调度、没有储能回放、没有 q75 重建，MAE 改善不能读成节费，"
            "也不能自动升级为正式模型。",
            "主指标是 A 集逐段 10 分钟 MAE，事前固定、不得事后更换；小时均值只能作补充，"
            "不能用它代替逐段 MAE。",
            "回归以平方误差估计、评价以 MAE 为主，二者不同；结果不佳也不能临时改主指标掩盖。",
            "28 天参数窗口、14 日下限、按发布小时与未来小时分组都是工程设定，未证最优；"
            "本轮不调窗口、不加岭回归、不加门控、不按表现挑节点。",
            "2025 年此前已参与方法设计，本轮仍不是跨年盲测，也不是独立样本外验证。",
            "B 集 73,146 条记录含重叠目标，只作发布版本质量诊断，不能当独立区间或替代 A 集。",
            "1,002 条午夜尾槽对应 334 个不同实际午夜区间、被三个版本各预测一次，"
            "不能当成三次实际发生。",
            "每小时内统一平移可能造成小时边界跳变或局部恶化，本轮只记录不修正。",
            "截断前修正量与截断后实际变化分开记录：截断命中时两者不等，不能混称一个修正量。",
            "没有任何图件经过目视核查（本 Agent 无法查看图像），只做程序化完整性检查。"])
    save_json(OUT / "validation.json", validation)
    manifest = dict(status="complete", started_utc=started_utc, finished_utc=utc_now(),
                    wall_seconds=time.perf_counter() - started, timings=timings,
                    signature=signature, executable=sys.executable, python=sys.version,
                    zero_solves=dict(lightgbm_training=0, milp_lp_dp=0, storage_replay=0,
                                     pv_rebuild=0, quantile_rebuild=0, parameter_grid_search=0),
                    closed_form_fits=dict(estimated=fits["fitted"], groups=fits["groups"],
                                          fallback_m_below_min=fits["fallback_m_below_min"],
                                          fallback_degenerate_variance=(
                                              fits["fallback_degenerate_variance"]),
                                          fallback_b_invalid=fits["fallback_b_invalid"]),
                    inputs_read=sorted(inputs),
                    outputs={p.relative_to(OUT).as_posix(): digest(p) for p in sorted(OUT.rglob("*"))
                             if p.is_file() and p.name not in NON_COMPUTED_ARTIFACTS})
    save_json(OUT / "run_manifest.json", manifest)
    print(json.dumps({"status": "complete", "wall_seconds": manifest["wall_seconds"],
                      "mae_kW": mae, "fit_counts": fits, "verdict": verdict},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
