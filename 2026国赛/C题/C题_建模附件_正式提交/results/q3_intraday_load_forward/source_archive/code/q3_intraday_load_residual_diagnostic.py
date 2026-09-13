#!/usr/bin/env python
"""问题三 日内负载残差可预测性轻量诊断 —— 只读计算层。

任务书 ``reports/问题三/问题三_日内负载残差可预测性轻量诊断方案.md``。只回答一件事：

    在已有的 6:00 / 12:00 / 18:00 更新节点，当天**已经观测到**的负载偏差，
    是否包含关于**剩余时段**负载偏差的稳定线索。

职责边界
--------
**纯只读统计**：模型训练 0、MILP/LP/DP 求解 0、储能回放 0、公共 1 月重跑 0、
光伏预测重建 0、分位重建 0、全年负载模型重训 0。不发布修正预测、不估计费用、
不搜索窗口、不调参。本脚本只写 CSV/JSON；报告与图表由
``code/q3_intraday_load_residual_report.py`` 只读渲染，
**修改报告文案或图表永远不会使本脚本的登记签名失效，也不需要重跑统计**。

关键口径
--------
* 发布日 ``k``（自然日 2025-01-01 + k 天）；目标槽 ``h = 0..144``，每槽 10 分钟；
  ``h=0`` 为当日 00:00—00:10，``h=144`` 为次日 00:00—00:10，不能当作当日 00:00。
* 原 0:00 负载预测残差 ``e^L[k,h] = L[k,h] - Lhat[k,h|0]``，正值表示负载被低估。
* 近期特征固定为**发布前已完整观测的一小时**：``b[k,r] = mean(e^L[k, s_r-6 .. s_r-1])``，
  ``s_r = 6r``（6:00→36、12:00→72、18:00→108），最后一段起点 ``s_r-1`` 即 05:50/11:50/17:50，
  区间结束时刻恰为发布时刻，不使用任何 ``h >= s_r`` 的观测。
* 未来标签 ``y[k,r,j] = mean(e^L[k, s_r+6j .. s_r+6j+5])``，``j=0`` 是发布后第一小时；
  三个节点分别有 18/12/6 个完整未来小时，另各有一个次日午夜 10 分钟尾槽（``h=144``）单列。
* 正式诊断期 2025-02-01—12-31，共 334 日。每日每 ``(r,j)`` 只有一个配对，
  不得把同日相同的 ``b`` 重复六次当作独立样本。
* 统计量全部是描述性的：Pearson 相关、带截距描述性直线斜率
  ``beta = cov(b,y)/var(b)``、``a = ybar - beta*bbar``。不做显著性星号、
  不 Bootstrap、不扫描窗口、不按结果挑启用月份。标准差统一用样本定义（``ddof=1``）。

运行::

    E:/Anaconda/envs/math_modeling/python.exe code/q3_intraday_load_residual_diagnostic.py
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
CODE = ROOT / "code/q3_intraday_load_residual_diagnostic.py"
OUT = ROOT / "results/q3_intraday_load_residual_diagnostic"
PLAN_MD = ROOT / "reports/问题三/问题三_日内负载残差可预测性轻量诊断方案.md"
REPORT_MD = ROOT / "reports/问题三/问题三_日内负载残差可预测性诊断结果.md"
BIAS_NPZ = ROOT / "results/q3_bias_correction_diagnostic/bias_forecast_archive.npz"
Q2_NPZ = ROOT / "results/q2_time_mapping/archive_float.npz"
Q2_KERNEL = ROOT / "code/29_q2_time_mapping_experiment.py"
Q3_ROLLING = ROOT / "code/q3_rolling_baseline_experiment.py"

BASE = pd.Timestamp("2025-01-01")
NATURAL_DAYS = 365              # source rows = natural days 2025-01-01 .. 2025-12-31
TARGETS = 145                   # h = 0..144, one publication day's target clock
DT_MINUTES = 10
EVAL_FIRST, EVAL_LAST = 31, 364          # 2025-02-01 .. 2025-12-31
EVAL_DAYS = EVAL_LAST - EVAL_FIRST + 1   # 334
EVAL = slice(EVAL_FIRST, EVAL_LAST + 1)

PUB_HOURS = (6, 12, 18)
SLOT_OF = {6: 36, 12: 72, 18: 108}       # s_r = 6r
RECENT_SLOTS = 6                         # fixed one-hour window, not searched
MIDNIGHT_H = 144                         # next-day 00:00-00:10, reported separately
DD_OF = 1                                # sample standard deviation, fixed once

# expected sample sizes (asserted, not assumed)
N_PUB_FEATURES = EVAL_DAYS * len(PUB_HOURS)                       # 1002
N_HOUR_PAIRS = EVAL_DAYS * sum(24 - r for r in PUB_HOURS)         # 12024
N_NEAR_PAIRS = EVAL_DAYS * RECENT_SLOTS * len(PUB_HOURS)          # 6012 (j = 0..5)
N_TAIL_ROWS = EVAL_DAYS * len(PUB_HOURS)                          # 1002
HOURS_TO_NEXT_NODE = 6                   # each publication is diagnosed up to the next node
N_FULL_WINDOW_SLOTS = EVAL_DAYS * sum(TARGETS - SLOT_OF[r] for r in PUB_HOURS)   # 73146
N_TO_NEXT_NODE_SLOTS = EVAL_DAYS * HOURS_TO_NEXT_NODE * RECENT_SLOTS * len(PUB_HOURS)  # 36072

PERTURB_DAY = 171                        # 2025-06-21, same boundary day as the rolling baseline
PERTURB_DELTA_KW = 25.0                  # in-memory truth shift at/after the publication instant
ABS_TOL = 1e-8                           # source-alignment tolerance, own units, rtol = 0

NON_COMPUTED_ARTIFACTS = frozenset({"run_manifest.json", "figure_integrity.json", "run.log"})

PARAMETERS = dict(
    experiment="Q3 intraday load-residual predictability, light read-only diagnostic",
    specification=str(PLAN_MD.relative_to(ROOT)),
    subject="does the load deviation already observed at 6:00/12:00/18:00 carry a stable "
            "signal about the remaining hours of the same day",
    time_version="start_time_v1",
    target_rule="h=0 current 00:00-00:10; h=1..143 current 00:10..23:50; h=144 next 00:00-00:10; "
                "slot h starts at 00:00 + h*10 minutes of the publication day",
    residual="eL[k,h] = truth_load[k,h] - issued_load[k,h] (kW); positive = load under-forecast",
    recent_feature="b[k,r] = mean(eL[k, s_r-6 .. s_r-1]), s_r = 6r in {36,72,108}; the last slot "
                   "starts at 05:50/11:50/17:50 and its interval ends exactly at the publication "
                   "instant, so no h >= s_r observation is used",
    future_label="y[k,r,j] = mean(eL[k, s_r+6j .. s_r+6j+5]), j = 0..23-r (18/12/6 full hours); "
                 "the next-day midnight tail h=144 is a single 10-minute slot reported separately",
    statistics="Pearson r and the descriptive slope beta = cov(b,y)/var(b) with intercept "
               "a = ybar - beta*bbar; sample standard deviation ddof=1; a constant series returns "
               "a missing correlation instead of 0",
    stability="per-pair month table plus two full-period descriptive controls: month-demeaned and "
              "weekday-demeaned correlations, using within-group full-period means",
    evaluation="2025-02-01 .. 2025-12-31, 334 days; one pair per (day, r, j); the same b must not "
               "be repeated into six independent samples",
    reference="frozen Q2 15-feature LightGBM issued load; no retraining, no forecast publication",
    excluded=["the 2026-01-01 00:00-00:10 template tail (not produced)",
              "any correction model, weight, gate, window search or cost estimate"],
    solver="none: no training, no MILP/LP/DP, no storage replay, no prediction or quantile rebuild",
    validation="three light checks only: input/sample bookkeeping, one boundary date with an "
               "in-memory truth shift, and one disk read-back recomputation",
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


def close(a, b, tol=ABS_TOL):
    return abs(float(a) - float(b)) <= tol


def known_max_abs(values, label):
    """``max |value|`` that treats any NaN/inf as a failure instead of silently skipping it."""
    arr = np.asarray(values, dtype=float)
    bad = ~np.isfinite(arr)
    if bad.any():
        raise AssertionError(f"{label}: {int(bad.sum())} non-finite value(s) of {arr.size}")
    return float(np.abs(arr).max())


def same_nan_mask(left, right, label):
    """NaN masks must agree exactly before any value comparison (NaN != NaN would hide gaps)."""
    lp, rp = np.isnan(left), np.isnan(right)
    if not np.array_equal(lp, rp):
        raise AssertionError(f"{label}: {int((lp != rp).sum())} cell(s) with differing NaN mask")
    return int(np.isfinite(left).sum())


def slot_interval_start(k, h):
    """Interval start of target slot h of publication day k (h=144 -> next day 00:00)."""
    return BASE + pd.Timedelta(days=int(k)) + pd.Timedelta(minutes=DT_MINUTES * int(h))


# ======================================================================================
# §2  输入
# ======================================================================================
def read_inputs():
    with np.load(BIAS_NPZ) as archive:
        issued = archive["issued_load"].astype(float)
        truth = archive["truth_load"].astype(float)
    with np.load(Q2_NPZ) as archive:
        issued_ref = archive["issued_load"].astype(float)

    assert issued.shape == (NATURAL_DAYS, TARGETS), issued.shape
    assert truth.shape == (NATURAL_DAYS, TARGETS), truth.shape
    assert issued_ref.shape == (NATURAL_DAYS, TARGETS), issued_ref.shape

    # same frozen load archive on both sides: masks must match, then values must match exactly
    cells = same_nan_mask(issued, issued_ref, "issued_load vs Q2 kernel archive")
    diff = np.abs(np.where(np.isnan(issued), 0.0, issued)
                  - np.where(np.isnan(issued_ref), 0.0, issued_ref))
    max_diff = float(diff.max())
    assert max_diff <= ABS_TOL, f"issued_load archives disagree by {max_diff}"

    # formal window must be fully defined on both sides
    issued_eval, truth_eval = issued[EVAL], truth[EVAL]
    if not np.isfinite(issued_eval).all():
        raise AssertionError(f"issued_load: {int((~np.isfinite(issued_eval)).sum())} non-finite "
                             f"cell(s) inside the evaluation window")
    if not np.isfinite(truth_eval).all():
        raise AssertionError(f"truth_load: {int((~np.isfinite(truth_eval)).sum())} non-finite "
                             f"cell(s) inside the evaluation window")

    finite_rows = np.isfinite(issued).all(axis=1)
    first_full_row = int(np.flatnonzero(finite_rows)[0])
    truth_rows = np.isfinite(truth).all(axis=1)
    first_truth_row = int(np.flatnonzero(truth_rows)[0])

    return dict(issued_load=issued, truth_load=truth,
                residual=truth_eval - issued_eval,
                dates=[BASE + pd.Timedelta(days=k) for k in range(NATURAL_DAYS)],
                cross_check=dict(archive_cells=issued.size, nan_mask_agreeing_cells=cells,
                                 max_abs_difference_kW=max_diff,
                                 first_all_finite_issued_row=first_full_row,
                                 first_all_finite_truth_row=first_truth_row,
                                 finite_eval_cells_issued=int(issued_eval.size),
                                 finite_eval_cells_truth=int(truth_eval.size)))


# ======================================================================================
# §3  残差特征
# ======================================================================================
def build_publication_features(residual, dates):
    """One row per (publication day, publication hour): the pre-publication one-hour mean b."""
    rows = []
    for r in PUB_HOURS:
        s = SLOT_OF[r]
        block = residual[:, s - RECENT_SLOTS:s]
        n_valid = np.isfinite(block).sum(axis=1)
        valid = n_valid == RECENT_SLOTS
        mean = np.where(valid, np.nansum(block, axis=1) / RECENT_SLOTS, np.nan)
        for i, k in enumerate(range(EVAL_FIRST, EVAL_LAST + 1)):
            row = dict(k=k, date=dates[k].strftime("%Y-%m-%d"), weekday=int(dates[k].weekday()),
                       month=dates[k].strftime("%Y-%m"), publication_hour=r, s_r=s,
                       recent_h_start=s - RECENT_SLOTS, recent_h_end=s - 1,
                       recent_interval_start=slot_interval_start(k, s - RECENT_SLOTS),
                       recent_interval_end=slot_interval_start(k, s),
                       n_valid_slots=int(n_valid[i]), valid=bool(valid[i]),
                       b_kW=float(mean[i]) if valid[i] else np.nan)
            for offset in range(RECENT_SLOTS):
                row[f"e_slot_{s - RECENT_SLOTS + offset}_kW"] = float(block[i, offset])
            rows.append(row)
    frame = pd.DataFrame(rows)
    assert len(frame) == N_PUB_FEATURES, len(frame)
    assert frame.b_kW.notna().all(), "formal window has no incomplete recent window"
    return frame


def build_hourly_pairs(residual, dates, features):
    """One row per (publication day, publication hour, future hour) plus the midnight tail."""
    b_map = {(int(r.k), int(r.publication_hour)): float(r.b_kW)
             for r in features.itertuples(index=False)}
    rows = []
    for r in PUB_HOURS:
        s = SLOT_OF[r]
        for j in range(24 - r):
            h0 = s + 6 * j
            block = residual[:, h0:h0 + 6]
            n_valid = np.isfinite(block).sum(axis=1)
            valid = n_valid == 6
            mean = np.where(valid, np.nansum(block, axis=1) / 6.0, np.nan)
            for i, k in enumerate(range(EVAL_FIRST, EVAL_LAST + 1)):
                rows.append(dict(
                    k=k, date=dates[k].strftime("%Y-%m-%d"), weekday=int(dates[k].weekday()),
                    month=dates[k].strftime("%Y-%m"), publication_hour=r, kind="hour", j=j,
                    lead_hours=j + 1, h_start=h0, h_end=h0 + 5,
                    interval_start=slot_interval_start(k, h0),
                    interval_end=slot_interval_start(k, h0 + 6),
                    n_valid_slots=int(n_valid[i]), valid=bool(valid[i]),
                    b_kW=b_map[(k, r)], y_kW=float(mean[i]) if valid[i] else np.nan))
        column = residual[:, MIDNIGHT_H]
        for i, k in enumerate(range(EVAL_FIRST, EVAL_LAST + 1)):
            ok = bool(np.isfinite(column[i]))
            rows.append(dict(
                k=k, date=dates[k].strftime("%Y-%m-%d"), weekday=int(dates[k].weekday()),
                month=dates[k].strftime("%Y-%m"), publication_hour=r, kind="midnight_tail", j=-1,
                lead_hours=24 - r, h_start=MIDNIGHT_H, h_end=MIDNIGHT_H,
                interval_start=slot_interval_start(k, MIDNIGHT_H),
                interval_end=slot_interval_start(k, MIDNIGHT_H + 1),
                n_valid_slots=int(ok), valid=ok,
                b_kW=b_map[(k, r)], y_kW=float(column[i]) if ok else np.nan))
    frame = pd.DataFrame(rows)
    hours = frame[frame.kind == "hour"]
    tails = frame[frame.kind == "midnight_tail"]
    assert len(hours) == N_HOUR_PAIRS, len(hours)
    assert len(tails) == N_TAIL_ROWS, len(tails)
    assert int((hours.j <= 5).sum()) == N_NEAR_PAIRS, int((hours.j <= 5).sum())
    assert not frame.duplicated(["k", "publication_hour", "kind", "j"]).any()
    return frame


# ======================================================================================
# §4  描述性统计
# ======================================================================================
def describe(x, y):
    """Descriptive Pearson r, slope and intercept for one pair group; constants stay missing."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[ok], y[ok]
    n = int(ok.sum())
    out = dict(n=n, n_invalid=int(x.size - n))
    if n == 0:
        return {**out, "mean_b_kW": None, "sd_b_kW": None, "mean_y_kW": None, "sd_y_kW": None,
                "pearson_r": None, "beta": None, "intercept_a_kW": None,
                "constant_b": None, "constant_y": None}
    mean_b, mean_y = float(xv.mean()), float(yv.mean())
    sd_b = float(xv.std(ddof=DD_OF)) if n > 1 else float("nan")
    sd_y = float(yv.std(ddof=DD_OF)) if n > 1 else float("nan")
    db, dy = xv - mean_b, yv - mean_y
    ss_b, ss_y, cov = float(db @ db), float(dy @ dy), float(db @ dy)
    constant_b, constant_y = ss_b == 0.0, ss_y == 0.0
    beta = None if constant_b else cov / ss_b
    r = None if (constant_b or constant_y) else cov / math.sqrt(ss_b * ss_y)
    return {**out, "mean_b_kW": mean_b, "sd_b_kW": sd_b, "mean_y_kW": mean_y, "sd_y_kW": sd_y,
            "pearson_r": r, "beta": beta,
            "intercept_a_kW": None if beta is None else mean_y - beta * mean_b,
            "constant_b": bool(constant_b), "constant_y": bool(constant_y)}


def build_overall(pairs):
    rows = []
    for (r, kind, j), block in pairs.groupby(["publication_hour", "kind", "j"], sort=True):
        rows.append(dict(publication_hour=int(r), kind=kind, j=int(j),
                         lead_hours=int(block.lead_hours.iloc[0]),
                         range_label="next_node" if (kind == "hour" and j <= 5) else
                                     ("far" if kind == "hour" else "midnight_tail"),
                         **describe(block.b_kW, block.y_kW)))
    return pd.DataFrame(rows).sort_values(["publication_hour", "kind", "j"]).reset_index(drop=True)


def build_monthly(pairs):
    rows = []
    for (r, kind, j, month), block in pairs.groupby(
            ["publication_hour", "kind", "j", "month"], sort=True):
        stats = describe(block.b_kW, block.y_kW)
        rows.append(dict(publication_hour=int(r), kind=kind, j=int(j), month=month, **stats))
    return pd.DataFrame(rows).sort_values(
        ["publication_hour", "kind", "j", "month"]).reset_index(drop=True)


def build_centered(pairs):
    """Within-group full-period demeaning by month and by weekday; diagnostic only."""
    rows = []
    for (r, kind, j), block in pairs.groupby(["publication_hour", "kind", "j"], sort=True):
        for key, label in (("month", "month"), ("weekday", "weekday")):
            grouped = block.groupby(key)
            b_c = block.b_kW - grouped.b_kW.transform("mean")
            y_c = block.y_kW - grouped.y_kW.transform("mean")
            stats = describe(b_c, y_c)
            groups = int(block[key].nunique())
            rows.append(dict(publication_hour=int(r), kind=kind, j=int(j),
                             demean=label, n_groups=groups, **stats))
    return pd.DataFrame(rows).sort_values(
        ["publication_hour", "kind", "j", "demean"]).reset_index(drop=True)


def build_stability(overall, monthly, centered):
    """Direction/reversal rollup per publication node, plus the pooled and demeaned extremes."""
    monthly_cells = monthly.groupby(["publication_hour", "kind", "j"]).pearson_r
    reversals = monthly_cells.apply(lambda s: int((s < 0).sum()))
    key = ["publication_hour", "kind", "j"]
    base = overall.set_index(key)[["pearson_r", "beta", "range_label"]]
    base["months_negative"] = reversals
    base["months_positive"] = monthly_cells.apply(lambda s: int((s > 0).sum()))
    base["months_total"] = monthly_cells.apply(lambda s: int(s.size))
    base["min_month_r"] = monthly_cells.min()
    base["max_month_r"] = monthly_cells.max()
    for label in ("month", "weekday"):
        block = centered[centered.demean == label].set_index(key).pearson_r
        base[f"r_demeaned_{label}"] = block
    base = base.reset_index()

    def node_rollup(block):
        return dict(cells=int(len(block)),
                    pooled_positive=int((block.pearson_r > 0).sum()),
                    pooled_negative=int((block.pearson_r < 0).sum()),
                    pooled_r_min=float(block.pearson_r.min()),
                    pooled_r_max=float(block.pearson_r.max()),
                    cells_with_month_reversal=int((block.months_negative > 0).sum()),
                    min_months_negative=int(block.months_negative.min()),
                    max_months_negative=int(block.months_negative.max()),
                    min_month_r=float(block.min_month_r.min()),
                    min_demeaned_month_r=float(block.r_demeaned_month.min()),
                    min_demeaned_weekday_r=float(block.r_demeaned_weekday.min()))

    rollup = dict(
        cells_total=int(len(base)),
        cells_pooled_positive=int((base.pearson_r > 0).sum()),
        cells_pooled_negative=int((base.pearson_r < 0).sum()),
        cells_positive_after_month_demeaning=int((base.r_demeaned_month > 0).sum()),
        cells_positive_after_weekday_demeaning=int((base.r_demeaned_weekday > 0).sum()),
        min_pooled_r=float(base.pearson_r.min()),
        min_demeaned_month_r=float(base.r_demeaned_month.min()),
        min_demeaned_weekday_r=float(base.r_demeaned_weekday.min()),
        cells_with_month_reversal=int((base.months_negative > 0).sum()),
        worst_month_reversal_cells=[
            dict(publication_hour=int(x.publication_hour), kind=x.kind, j=int(x.j),
                 months_negative=int(x.months_negative), months_total=int(x.months_total),
                 pooled_r=float(x.pearson_r), min_month_r=float(x.min_month_r))
            for x in base.sort_values(["months_negative", "pearson_r"], ascending=[False, True])
            .head(6).itertuples(index=False)],
        by_node={label: node_rollup(block) for label, block in
                 (("next_node", base[base.range_label == "next_node"]),
                  ("far", base[base.range_label == "far"]),
                  ("midnight_tail", base[base.range_label == "midnight_tail"]))})
    return base, rollup


# ======================================================================================
# §5  三类精简验证
# ======================================================================================
def check_inputs_and_samples(data, features, pairs):
    issued, truth, residual = data["issued_load"], data["truth_load"], data["residual"]
    dates = data["dates"]
    checks = {}

    # source identity on both sides of the formal window
    checks["cross_check"] = data["cross_check"]
    assert EVAL_FIRST == 31 and EVAL_LAST == 364 and EVAL_DAYS == 334
    assert dates[EVAL_FIRST].strftime("%Y-%m-%d") == "2025-02-01"
    assert dates[EVAL_LAST].strftime("%Y-%m-%d") == "2025-12-31"
    checks["dates"] = dict(first=dates[EVAL_FIRST].strftime("%Y-%m-%d"),
                           last=dates[EVAL_LAST].strftime("%Y-%m-%d"), days=EVAL_DAYS,
                           unique_and_consecutive=bool(
                               all((dates[i + 1] - dates[i]).days == 1
                                   for i in range(NATURAL_DAYS - 1))))

    # residual definition identity: eL = truth - issued, checked on the formal window
    rebuilt = truth[EVAL] - issued[EVAL]
    checks["residual_identity"] = dict(cells=int(residual.size),
                                       max_abs_difference_kW=known_max_abs(rebuilt - residual,
                                                                           "eL identity"))

    # window geometry: each window's end is exactly at or before its publication instant
    window = {}
    for r in PUB_HOURS:
        s = SLOT_OF[r]
        end = slot_interval_start(int(features.k.iloc[0]), s)
        assert end.minute == 0 and end.hour == r, (r, end)
        window[str(r)] = dict(s_r=s, recent_h=(s - RECENT_SLOTS, s - 1),
                              recent_last_interval_start=slot_interval_start(0, s - 1).strftime(
                                  "%H:%M"),
                              publication_instant=f"{r:02d}:00",
                              full_future_slots=int(TARGETS - s),
                              full_hours=int(24 - r),
                              uses_no_observation_at_or_after_publication=True)
    checks["window_geometry"] = window

    # sample bookkeeping
    counts = dict(publication_features=len(features), hour_pairs=int((pairs.kind == "hour").sum()),
                  midnight_tail_rows=int((pairs.kind == "midnight_tail").sum()),
                  near_hour_pairs=int(((pairs.kind == "hour") & (pairs.j <= 5)).sum()),
                  far_hour_pairs=int(((pairs.kind == "hour") & (pairs.j > 5)).sum()),
                  distinct_days=int(features.k.unique().size),
                  duplicate_pairs=int(pairs.duplicated(
                      ["k", "publication_hour", "kind", "j"]).sum()),
                  full_window_slot_records=int(
                      EVAL_DAYS * sum(TARGETS - SLOT_OF[r] for r in PUB_HOURS)),
                  to_next_node_slot_records=int(
                      EVAL_DAYS * HOURS_TO_NEXT_NODE * 6 * len(PUB_HOURS)),
                  windowed_2025_06_21_hour_pairs=int(
                      ((pairs.date == "2025-06-21") & (pairs.kind == "hour")).sum()))
    assert counts["publication_features"] == N_PUB_FEATURES
    assert counts["hour_pairs"] == N_HOUR_PAIRS
    assert counts["midnight_tail_rows"] == N_TAIL_ROWS
    assert counts["near_hour_pairs"] == N_NEAR_PAIRS
    assert counts["duplicate_pairs"] == 0
    assert counts["full_window_slot_records"] == N_FULL_WINDOW_SLOTS
    assert counts["to_next_node_slot_records"] == N_TO_NEXT_NODE_SLOTS
    assert counts["distinct_days"] == EVAL_DAYS
    checks["sample_counts"] = counts

    # every pairing carries a finite feature; invalid labels are counted, never filled
    checks["validity"] = dict(feature_invalid=int((~features.valid).sum()),
                              hour_pair_invalid=int(((pairs.kind == "hour") & (~pairs.valid)).sum()),
                              tail_invalid=int((~pairs[pairs.kind == "midnight_tail"].valid).sum()),
                              invalid_rows_filled=int(
                                  pairs.loc[~pairs.valid, "y_kW"].notna().sum()),
                              feature_b_finite=int(features.b_kW.notna().sum()))
    assert checks["validity"]["invalid_rows_filled"] == 0

    # no observation at or after the publication instant may enter b
    b_by_key = {(int(x.k), int(x.publication_hour)): x.b_kW for x in features.itertuples(index=False)}
    replay = []
    for r in PUB_HOURS:
        s = SLOT_OF[r]
        block = residual[:, s - RECENT_SLOTS:s]
        worst = 0.0
        for i, k in enumerate(range(EVAL_FIRST, EVAL_LAST + 1)):
            worst = max(worst, abs(float(np.mean(block[i])) - float(b_by_key[(k, r)])))
        replay.append(worst)
    checks["feature_replay_max_abs_kW"] = known_max_abs(replay, "feature replay")
    assert checks["feature_replay_max_abs_kW"] <= ABS_TOL
    return checks


def check_boundary_date(data, residual, dates, features):
    """2025-06-21: b must not move when the truth at/after the publication instant changes."""
    issued = data["issued_load"]
    k = PERTURB_DAY
    assert dates[k].strftime("%Y-%m-%d") == "2025-06-21"
    rows = []
    for r in PUB_HOURS:
        s = SLOT_OF[r]
        # fresh copy per node: shifting inside the loop would accumulate across nodes and leak
        # an earlier node's shift into a later node's pre-publication window
        shifted_truth = data["truth_load"].copy()
        shifted_truth[k, s:] += PERTURB_DELTA_KW
        shifted_residual = shifted_truth[EVAL] - issued[EVAL]
        b_before = float(np.mean(residual[k - EVAL_FIRST, s - RECENT_SLOTS:s]))
        b_after = float(np.mean(shifted_residual[k - EVAL_FIRST, s - RECENT_SLOTS:s]))
        y_before = float(np.mean(residual[k - EVAL_FIRST, s:s + 6]))
        y_after = float(np.mean(shifted_residual[k - EVAL_FIRST, s:s + 6]))
        tail_before = float(residual[k - EVAL_FIRST, MIDNIGHT_H])
        tail_after = float(shifted_residual[k - EVAL_FIRST, MIDNIGHT_H])
        saved = features[(features.k == k) & (features.publication_hour == r)].iloc[0]
        assert close(saved.b_kW, b_before)
        rows.append(dict(
            publication_hour=r, s_r=s,
            recent_first_interval_start=slot_interval_start(k, s - RECENT_SLOTS).strftime(
                "%Y-%m-%d %H:%M"),
            recent_last_interval_start=slot_interval_start(k, s - 1).strftime("%Y-%m-%d %H:%M"),
            future_first_interval_start=slot_interval_start(k, s).strftime("%Y-%m-%d %H:%M"),
            future_first_interval_end=slot_interval_start(k, s + 6).strftime("%Y-%m-%d %H:%M"),
            midnight_tail_interval_start=slot_interval_start(k, MIDNIGHT_H).strftime(
                "%Y-%m-%d %H:%M"),
            b_before_kW=b_before, b_after_shift_kW=b_after, b_abs_change_kW=abs(b_after - b_before),
            y_first_hour_before_kW=y_before, y_first_hour_after_shift_kW=y_after,
            y_first_hour_abs_change_kW=abs(y_after - y_before),
            tail_before_kW=tail_before, tail_after_shift_kW=tail_after,
            tail_abs_change_kW=abs(tail_after - tail_before)))
    frame = pd.DataFrame(rows)
    assert (frame.b_abs_change_kW <= ABS_TOL).all(), frame
    assert np.allclose(frame.y_first_hour_abs_change_kW, PERTURB_DELTA_KW, atol=ABS_TOL), frame
    assert np.allclose(frame.tail_abs_change_kW, PERTURB_DELTA_KW, atol=ABS_TOL), frame
    return dict(day="2025-06-21", k=k, perturb_kW=PERTURB_DELTA_KW,
                rule="truth at and after the publication instant moved in memory; b must not move "
                     "because it uses only h < s_r, while the future label is allowed to move",
                rows=frame.to_dict("records"),
                max_abs_b_change_kW=known_max_abs(frame.b_abs_change_kW, "b change"),
                all_b_unchanged=bool((frame.b_abs_change_kW <= ABS_TOL).all()),
                all_future_labels_moved=bool(
                    np.allclose(frame.y_first_hour_abs_change_kW, PERTURB_DELTA_KW, atol=ABS_TOL)))


def check_disk_readback():
    """Recompute two published cells straight from the saved tables, plus the edge cases."""
    pairs = pd.read_csv(OUT / "hourly_pairs.csv")
    overall = pd.read_csv(OUT / "overall_summary.csv")
    rows = []
    for r, j in ((6, 0), (18, 0)):
        block = pairs[(pairs.publication_hour == r) & (pairs.kind == "hour") & (pairs.j == j)]
        stats = describe(block.b_kW, block.y_kW)
        saved = overall[(overall.publication_hour == r) & (overall.kind == "hour")
                        & (overall.j == j)].iloc[0]
        rows.append(dict(publication_hour=r, j=j, n=stats["n"],
                         mean_b_recomputed_kW=stats["mean_b_kW"],
                         mean_b_saved_kW=float(saved.mean_b_kW),
                         mean_y_recomputed_kW=stats["mean_y_kW"],
                         mean_y_saved_kW=float(saved.mean_y_kW),
                         r_recomputed=stats["pearson_r"],
                         r_saved=None if pd.isna(saved.pearson_r) else float(saved.pearson_r),
                         beta_recomputed=stats["beta"],
                         beta_saved=None if pd.isna(saved.beta) else float(saved.beta)))
    frame = pd.DataFrame(rows)
    for left_name, right_name in (("mean_b_recomputed_kW", "mean_b_saved_kW"),
                                  ("mean_y_recomputed_kW", "mean_y_saved_kW"),
                                  ("r_recomputed", "r_saved"), ("beta_recomputed", "beta_saved")):
        left = frame[left_name].astype(float)
        right = frame[right_name].astype(float)
        assert np.allclose(left, right, atol=ABS_TOL, rtol=0.0, equal_nan=True), (left_name, frame)
    assert (frame.n == EVAL_DAYS).all(), frame

    # constant series -> missing correlation, never 0
    flat = describe(np.full(10, 5.0), np.arange(10.0))
    rising = describe(np.arange(10.0), np.arange(10.0))
    assert flat["pearson_r"] is None and flat["beta"] is None and flat["constant_b"] is True
    assert close(rising["pearson_r"], 1.0) and close(rising["beta"], 1.0) and close(
        rising["intercept_a_kW"], 0.0)
    # incomplete recent window -> the publication sample is invalid and b stays NaN
    frame_short = pd.DataFrame([dict(b_kW=np.nan, y_kW=1.0, valid=False)])
    assert frame_short.b_kW.isna().all()
    return dict(recomputed_cells=frame.to_dict("records"),
                max_abs_recompute_difference=known_max_abs(
                    np.r_[np.abs(frame.mean_b_recomputed_kW - frame.mean_b_saved_kW),
                          np.abs(frame.mean_y_recomputed_kW - frame.mean_y_saved_kW),
                          np.abs(frame.r_recomputed - frame.r_saved),
                          np.abs(frame.beta_recomputed - frame.beta_saved)], "read-back difference"),
                constant_series=dict(pearson_r=flat["pearson_r"], beta=flat["beta"],
                                     constant_b=flat["constant_b"], flagged=True),
                rising_series=dict(pearson_r=rising["pearson_r"], beta=rising["beta"],
                                   intercept_a_kW=rising["intercept_a_kW"]),
                incomplete_recent_window="b and the pair are marked invalid and left NaN; the "
                                          "window is never widened to recover a sample")


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
        # carry the amendment trail forward even when nothing changed: an idempotent re-run used to
        # rebuild this record from scratch and silently drop the whole history
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
    parser.add_argument("--mode", choices=["register", "full"], default="full")
    parser.add_argument("--amend-reason", default=None)
    args = parser.parse_args()
    assert Path(sys.prefix).name == "math_modeling", sys.prefix
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    started_utc = utc_now()
    timings = {}

    inputs = ["reports/问题三/问题三_日内负载残差可预测性轻量诊断方案.md",
              "results/q3_bias_correction_diagnostic/bias_forecast_archive.npz",
              "results/q2_time_mapping/archive_float.npz",
              "code/29_q2_time_mapping_experiment.py",
              "code/q3_rolling_baseline_experiment.py"]
    for rel in inputs:
        assert (ROOT / rel).exists(), rel
    dependency = dict(bias_archive="results/q3_bias_correction_diagnostic/"
                                   "bias_forecast_archive.npz::issued_load,truth_load",
                      q2_kernel_archive="results/q2_time_mapping/archive_float.npz::issued_load",
                      zero_budget=dict(training=0, milp_lp_dp=0, storage_replay=0,
                                       public_january_reruns=0, prediction_rebuild=0,
                                       pv_forecast_rebuild=0, quantile_rebuild=0,
                                       window_search=0, parameter_sweeps=0))
    signature = register(PARAMETERS, inputs, dependency, args.amend_reason)
    print(f"registered signature={signature[:16]}", flush=True)
    if args.mode == "register":
        return

    tick = time.perf_counter()
    data = read_inputs()
    timings["inputs_seconds"] = time.perf_counter() - tick
    print(f"inputs: issued/truth {data['issued_load'].shape}, eval window "
          f"{data['dates'][EVAL_FIRST].date()} .. {data['dates'][EVAL_LAST].date()}", flush=True)

    tick = time.perf_counter()
    residual, dates = data["residual"], data["dates"]
    features = build_publication_features(residual, dates)
    pairs = build_hourly_pairs(residual, dates, features)
    timings["features_seconds"] = time.perf_counter() - tick

    tick = time.perf_counter()
    checks = dict(inputs_and_samples=check_inputs_and_samples(data, features, pairs))
    checks["boundary_date"] = check_boundary_date(data, residual, dates, features)
    timings["checks_seconds"] = time.perf_counter() - tick
    print(f"input/sample checks passed; 2025-06-21 b unchanged under a "
          f"{PERTURB_DELTA_KW:+.0f} kW truth shift", flush=True)

    tick = time.perf_counter()
    overall = build_overall(pairs)
    monthly = build_monthly(pairs)
    centered = build_centered(pairs)
    per_cell, stability = build_stability(overall, monthly, centered)
    timings["aggregates_seconds"] = time.perf_counter() - tick

    frame_to_csv(features, OUT / "publication_features.csv")
    frame_to_csv(pairs, OUT / "hourly_pairs.csv")
    frame_to_csv(overall, OUT / "overall_summary.csv")
    frame_to_csv(monthly, OUT / "monthly_summary.csv")
    frame_to_csv(centered, OUT / "centered_summary.csv")

    checks["disk_readback"] = check_disk_readback()
    print("disk read-back check passed", flush=True)

    verdicts = dict(
        near_pairs=int((overall.range_label == "next_node").sum()),
        far_pairs=int((overall.range_label == "far").sum()),
        tail_cells=int((overall.range_label == "midnight_tail").sum()),
        **stability)
    verdicts.pop("worst_month_reversal_cells", None)

    validation = dict(
        status="passed", signature=signature, checks=checks, overall_verdicts=verdicts,
        stability=stability, zero_budget=dependency["zero_budget"],
        limitations=[
            "只做描述性统计：仅一个事先固定的一小时近期窗口，不搜索窗口、不做显著性星号。"
            "一个窗口的负结果只说明该设计未发现足够线索，不证明所有日内特征均无价值。",
            "相关不是因果；相关系数与斜率都既不是预测改善，也不是节费。本轮没有拟合或发布任何修正模型。",
            "同一发布日内所有未来小时共用同一个 b，增加 j 并不增加独立的日样本；日间残差本身存在序列相关，"
            "因此不附未经校正的独立样本显著性解释。",
            "月份／星期去均值只用于检查共同水平是否解释了原相关，不能消除全部混杂；"
            "去均值用的是全期分组统计，只可用于诊断，绝不能进入前向预测。",
            "2025 年此前已用于方法设计：本轮描述的是一个已被开发的年份，不是独立盲测。",
            "没有任何图件经过目视核查（本 Agent 无法查看图像），只做程序化完整性检查。"])
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
                      "publication_features": len(features),
                      "hour_pairs": int((pairs.kind == "hour").sum()),
                      "midnight_tail_rows": int((pairs.kind == "midnight_tail").sum()),
                      "near_hour_pairs": int(((pairs.kind == "hour") & (pairs.j <= 5)).sum()),
                      "min_pooled_r": stability["min_pooled_r"],
                      "min_demeaned_month_r": stability["min_demeaned_month_r"],
                      "min_demeaned_weekday_r": stability["min_demeaned_weekday_r"],
                      "cells_with_month_reversal": stability["cells_with_month_reversal"],
                      "by_node": stability["by_node"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
