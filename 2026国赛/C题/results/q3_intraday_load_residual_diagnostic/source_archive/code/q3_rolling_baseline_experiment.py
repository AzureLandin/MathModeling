#!/usr/bin/env python
"""问题三首轮 B0/B1/B2 滚动 Baseline 对照实验 —— 计算层。

职责边界
--------
本脚本只做：登记 → 读输入 → 构建预测/残差/保护档案 → B1/B2 调度求解 →
三类精简验证（A 小样例与信息边界、B 运行中断言、C 完成后只读核账）→ 落盘 CSV/JSON/NPZ。
**不写报告、不画图**；报告与图表由 ``code/q3_rolling_baseline_report.py`` 只读渲染。
修改报告文案或图表永远不会使本脚本的登记签名失效，也不需要重跑任何求解。

固定口径（任务书 ``reports/问题三/问题三_首轮Baseline对照实验方案.md``）
--------------------------------------------------------------------
* 三组：B0（问题二新时间口径原 12 特征光伏 LightGBM，0:00 计划后不调整，只读复用）、
  B1（附件3 的 0:00 光伏预报，全天不调整）、B2（B1 + 6/12/18 点带费修订并择优）。
* 自由名义末态；15 特征负载 LightGBM 发布档案；公共 1 月初始化；
  同发布小时-同目标时段 W28/q80；即时日内更新至次日 00:10；最终一次交付价结算。
* 不重训预测器、不重跑公共 1 月、不新增参数扫描、不填正式 Excel。

复用与独立性
------------
* B0 直接从 ``results/q2_time_mapping/N_free/`` 已保存账本只读载入，不重解。
* 为保证与 B0 的求解器行为一致，29 号已验证内核中的原语
  （``read_attachments`` / ``natural_arrays`` / ``natural_price`` / ``truth_targets`` /
  ``feedback_step`` / ``feedback``）在本脚本内逐字转写；A 类检查再用 29 号 ``solve_day``
  与本地自由末态求解器在同一输入上对拍。

运行::

    E:/Anaconda/envs/math_modeling/python.exe code/q3_rolling_baseline_experiment.py
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
from scipy.sparse import lil_matrix

# ======================================================================================
# §0  冻结规格
# ======================================================================================
ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code/q3_rolling_baseline_experiment.py"
OUT = ROOT / "results/q3_rolling_baseline"
FIG = ROOT / "figures/q3_rolling_baseline"
Q2_DIR = ROOT / "results/q2_time_mapping"
Q2_KERNEL = ROOT / "code/29_q2_time_mapping_experiment.py"
PV_CSV = ROOT / "results/q3_pv_time_conversion/pv_10min_forecasts.csv"
PLAN_MD = ROOT / "reports/问题三/问题三_首轮Baseline对照实验方案.md"
REPORT_MD = ROOT / "reports/问题三/问题三_首轮滚动实验结果.md"

BASE = pd.Timestamp("2025-01-01")
NATURAL_DAYS = 365            # source rows = natural days 2025-01-01 .. 2025-12-31
TEMPLATE_SLOTS = 144          # 10-minute slots per source row / template day
TARGETS = 145                 # h = 0..144 targets of one publication day
DT = 1.0 / 6.0                # hours per slot
ETA = 0.9
CAP_KWH = 5000.0 / 6.0        # max charge/discharge energy per slot (kWh)
STATE_MIN_KWH, STATE_MAX_KWH = 1200.0, 10800.0
ALPHA = 0.80
WINDOW = 28                   # residual window in calendar publication days
MIN_SAMPLES = 7               # fewer valid samples -> zero correction
EVAL_FIRST, EVAL_LAST = 31, 364   # 2025-02-01 .. 2025-12-31
UPDATE_HOURS = (0, 6, 12, 18)
FIRST_TARGET = {0: 0, 1: 36, 2: 72, 3: 108}   # first target h covered by each version
REVISION_HOURS = (36, 72, 108)                # 10-minute slots h where B2 may revise
EMERGENCY_MULTIPLIER = 5.0
SELECT_TOL_YUAN = 1e-4
ENERGY_TOL_KWH = 1e-6

PUBLIC_INITIAL_KWH = 6075.795025925926   # reference; asserted against the saved ledger
PUBLIC_CARRY_KWH = 0.0                   # reference midnight commitment of 2025-02-01
B0_REFERENCE_YUAN = 13699337.684431653   # N_free natural ledger A reference
PERTURB_DAY = 171                        # 2025-06-21 information-boundary day
PERTURB_TRUTH_SCALE = 1.05               # registered perturbation: scale realised net demand
PERTURB_PV_ADD_KW = 50.0                 # registered perturbation: shift future PV forecasts
SELECTED_DATES = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")

GROUPS = ("B0", "B1", "B2")

DISPATCH_COLS = [
    "group", "interval_start", "interval_end", "owner_date", "effective_version",
    "price_yuan_kWh", "actual_load_kW", "actual_pv_kW", "net_kWh",
    "q0_kWh", "q_eff_kWh", "charge_kWh", "discharge_kWh", "emergency_kWh", "unused_kWh",
    "state_start_kWh", "state_end_kWh",
    "ordinary_cost_yuan", "adjustment_cost_yuan", "emergency_cost_yuan",
]

PROTECTED_TREES = ("附件", "code", "reports", "figures", "results", "outputs")
OWN_NEW_REL = {
    "code/q3_rolling_baseline_experiment.py",
    "code/q3_rolling_baseline_report.py",
    "reports/问题三/问题三_首轮滚动实验结果.md",
}
OWN_NEW_PREFIXES = ("results/q3_rolling_baseline/", "figures/q3_rolling_baseline/")
# Written by the report layer, not by this script: they must stay out of run_manifest.outputs,
# otherwise re-rendering the report would invalidate a hash the manifest claims to certify.
REPORT_LAYER_ARTIFACTS = frozenset({"run_manifest.json", "figure_integrity.json",
                                    "revision_information_diagnostic.csv"})
SHARED_APPEND_REL = {"建模上下文记忆.md", "reports/项目进度.md"}

PARAMETERS = dict(
    experiment="Q3 first-round rolling baseline B0/B1/B2",
    specification="reports/问题三/问题三_首轮Baseline对照实验方案.md",
    time_version="start_time_v1",
    interval_rule="label 00:10 covers 00:10-00:20; label 0:00+1 covers the next day 00:00-00:10",
    natural_day_rule="natural day d = previous source row column 143 ++ current row columns 0..142",
    template_rule="k-day 0:00 publishes 144 slots covering [k 00:10, (k+1) 00:10); the current "
                  "midnight 00:00-00:10 executes the previous publication's carry commitment",
    targets="h=0 current 00:00-00:10; h=1..143 current 00:10..23:50; h=144 next 00:00-00:10; "
            "template slot j = h-1",
    price_rule="template slot j is billed at price_src[j]; natural slot t at "
               "[price_src[143], price_src[:143]][t]; delivered-interval price, independent of "
               "the revision clock",
    groups=dict(B0="Q2 N_free ledger, 12-column PV LightGBM, no intraday revision",
                B1="attachment-3 0:00 PV forecast, no intraday revision",
                B2="B1 plus 6/12/18 revision with cost-based accept/reject"),
    terminal="free nominal state; both true natural 24:00 and the template tail keep only the "
             "1200-10800 kWh band",
    load_forecast="Q2 15-feature LightGBM issued_load archive, unchanged within the day",
    protection=dict(window_days=WINDOW, alpha=ALPHA, min_samples=MIN_SAMPLES,
                    key="publication hour, target day-offset, target slot clock",
                    rule="ascending order statistic at position ceil(0.8m); zero when m<7; "
                         "negative corrections retained; h=144 at 0:00 has m=27 position 22"),
    adjustment_rule="p*q_eff + 0.5*p*|q_eff - q0| on the delivered interval price",
    emergency_rule="5*p*e, no extra ordinary charge on the emergency energy",
    settlement="final effective version against the 0:00 original plan, once only",
    ledgers=dict(A="natural days 2025-02-01..2025-12-31, 334 days, 48096 segments",
                 B="template publications 2025-02-01..2025-12-31, 48096 segments",
                 bridge="C_B - C_A = cost(2026-01-01 00:00-00:10) - cost(2025-02-01 "
                        "00:00-00:10)"),
    solver="SciPy milp/HiGHS, relative MIP gap 1e-9, time limit 120 s, free terminal",
    reuse=dict(B0="results/q2_time_mapping/N_free/ (read only)",
               load="results/q2_time_mapping/archive_float.npz::issued_load",
               pv="results/q3_pv_time_conversion/pv_10min_forecasts.csv"),
    validation="three classes only: A small examples + one information-boundary day, "
               "B in-run assertions, C one read-only ledger check; no full re-run",
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
    Path(path).write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=2, default=str),
                          encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def frame_to_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def midnight_mask(index):
    """True only for the true 00:00 slot; ``hour == 0`` alone also catches 00:10..00:50."""
    index = pd.DatetimeIndex(index)
    return np.asarray((index.hour == 0) & (index.minute == 0), dtype=bool)


def q2_kernel():
    """Import the verified 29 kernel only for the cross-kernel A-class check."""
    if "q2" not in _MODULES:
        spec = importlib.util.spec_from_file_location("q2_kernel", Q2_KERNEL)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULES["q2"] = module
    return _MODULES["q2"]


def protected_manifest():
    manifest = {}
    for tree in PROTECTED_TREES:
        base = ROOT / tree
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.name.startswith("~$"):
                continue
            relative = path.relative_to(ROOT).as_posix()
            if relative in OWN_NEW_REL or relative in SHARED_APPEND_REL:
                continue
            if relative.startswith(OWN_NEW_PREFIXES):
                continue
            manifest[relative] = digest(path)
    return manifest


# ======================================================================================
# §2  输入（read_attachments / natural_arrays / natural_price / truth_targets 逐字转写自 29 号）
# ======================================================================================
def label_text(value):
    if hasattr(value, "strftime"):
        return value.strftime("%H:%M")
    text = str(value)
    return "0:00+1" if "+1" in text else text[:5]


def read_attachments():
    wb = openpyxl.load_workbook(ROOT / "附件/附件1.xlsx", read_only=True, data_only=True)
    raw1 = list(wb.worksheets[0].values)
    wb.close()
    assert len(raw1) - 1 == TEMPLATE_SLOTS, len(raw1)
    price_src = np.array([row[1] for row in raw1[1:]], dtype=float)
    wb = openpyxl.load_workbook(ROOT / "附件/附件2.xlsx", read_only=True, data_only=True)
    sheets = [(ws.title, list(ws.values)) for ws in wb.worksheets]
    wb.close()
    assert len(sheets) == 2, [name for name, _ in sheets]
    load_rows, pv_rows = sheets[0][1], sheets[1][1]
    assert len(load_rows) - 1 == NATURAL_DAYS and len(pv_rows) - 1 == NATURAL_DAYS
    source_dates = [row[0] for row in load_rows[1:]]
    assert [row[0] for row in pv_rows[1:]] == source_dates
    load_src = np.asarray([row[1:] for row in load_rows[1:]], dtype=float)
    pv_src = np.asarray([row[1:] for row in pv_rows[1:]], dtype=float)
    assert load_src.shape == pv_src.shape == (NATURAL_DAYS, TEMPLATE_SLOTS)
    assert price_src.shape == (TEMPLATE_SLOTS,)
    assert np.isfinite(load_src).all() and np.isfinite(pv_src).all() and np.isfinite(price_src).all()
    assert (load_src >= 0).all() and (pv_src >= 0).all() and (price_src > 0).all()
    labels = [label_text(cell) for cell in load_rows[0][1:]]
    assert len(labels) == TEMPLATE_SLOTS and labels[-1] == "0:00+1", labels[-1]
    assert [label_text(cell) for cell in pv_rows[0][1:]] == labels, "load and pv headers differ"
    return dict(price_src=price_src, load_src=load_src, pv_src=pv_src,
                source_dates=source_dates, labels=labels,
                sheets=[name for name, _ in sheets])


def natural_arrays(load_src, pv_src):
    """Natural day d slot 0 = previous row's column 143; slots 1..143 = row d columns 0..142."""
    nat_load = np.full((NATURAL_DAYS + 1, TEMPLATE_SLOTS), np.nan)
    nat_pv = np.full((NATURAL_DAYS + 1, TEMPLATE_SLOTS), np.nan)
    for d in range(NATURAL_DAYS):
        nat_load[d, 1:] = load_src[d, 0:TEMPLATE_SLOTS - 1]
        nat_pv[d, 1:] = pv_src[d, 0:TEMPLATE_SLOTS - 1]
        nat_load[d + 1, 0] = load_src[d, TEMPLATE_SLOTS - 1]
        nat_pv[d + 1, 0] = pv_src[d, TEMPLATE_SLOTS - 1]
    nat_load[0, 0] = np.nan      # natural day 0 has no observed 00:00-00:10: the declared gap
    nat_pv[0, 0] = np.nan
    return nat_load, nat_pv


def natural_price(price_src):
    """Natural-slot price: the source's last value covers 00:00-00:10, then source rows 0..142."""
    return np.r_[price_src[TEMPLATE_SLOTS - 1], price_src[:TEMPLATE_SLOTS - 1]]


def truth_targets(load_src, pv_src, k):
    """The 145 realised targets of publication day k: previous row's column 143 ++ row k."""
    assert k >= 1, "day 0 has no previous source row"
    return (np.r_[load_src[k - 1, TEMPLATE_SLOTS - 1], load_src[k, :]],
            np.r_[pv_src[k - 1, TEMPLATE_SLOTS - 1], pv_src[k, :]])


def build_truth(load_src, pv_src):
    truth_load = np.full((NATURAL_DAYS, TARGETS), np.nan)
    truth_pv = np.full((NATURAL_DAYS, TARGETS), np.nan)
    for k in range(1, NATURAL_DAYS):
        truth_load[k], truth_pv[k] = truth_targets(load_src, pv_src, k)
    return truth_load, truth_pv


def read_pv_forecasts():
    """Attachment-3 conversion archive -> (365, 4, 145) interval-mean PV power (kW).

    Version v covers targets h = FIRST_TARGET[v] .. 144; everything else stays NaN because the
    publication does not reach it within the template horizon.
    """
    frame = pd.read_csv(PV_CSV, parse_dates=["issued_at", "interval_start"])
    forecasts = np.full((NATURAL_DAYS, len(UPDATE_HOURS), TARGETS), np.nan)
    seen = np.zeros((NATURAL_DAYS, len(UPDATE_HOURS)), dtype=bool)
    for issued, block in frame.groupby("issued_at"):
        k = int((issued.normalize() - BASE).days)
        v = UPDATE_HOURS.index(issued.hour)
        h = ((block.interval_start - issued.normalize()).dt.total_seconds() / 600).astype(int)
        h = h.to_numpy()
        use = (h >= 0) & (h < TARGETS)
        forecasts[k, v, h[use]] = block.pv_mean_kW.to_numpy()[use]
        seen[k, v] = True
    assert seen.all(), "every publication must be present in the conversion archive"
    assert np.isfinite(forecasts[:, 0, :]).all(), "0:00 version must cover h=0..144"
    for v in range(1, len(UPDATE_HOURS)):
        assert np.isfinite(forecasts[:, v, FIRST_TARGET[v]:]).all()
    return forecasts


def read_q2_archive():
    with np.load(Q2_DIR / "archive_float.npz") as archive:
        issued_load = archive["issued_load"]
        issued_pv = archive["issued_pv"]
    assert issued_load.shape == (NATURAL_DAYS, TARGETS)
    assert np.isfinite(issued_load[2:, :]).all(), "issued load must be finite from 2025-01-03"
    return issued_load, issued_pv


# ======================================================================================
# §3  时间映射、净需求预测档案与分组 q80 保护
# ======================================================================================
def build_forecast_archive(issued_load, pv_forecasts):
    """n_hat[k, v, h] = (issued load - attachment-3 PV) * dt, NaN outside each version's reach."""
    net_fc = np.full((NATURAL_DAYS, len(UPDATE_HOURS), TARGETS), np.nan)
    for v in range(len(UPDATE_HOURS)):
        h0 = FIRST_TARGET[v]
        net_fc[:, v, h0:] = (issued_load[:, h0:] - pv_forecasts[:, v, h0:]) * DT
    return net_fc


def build_protection(net_fc, truth_net):
    """Grouped W28/q80 residual archive keyed by (publication hour, target offset, slot clock).

    Target h itself already carries the day offset (h=0 current midnight, h=1..143 same day,
    h=144 next midnight) and the interval clock, so (v, h) is the full grouping key. A sample is
    usable only if its interval ended at or before the current publication instant.
    """
    err = truth_net[:, None, :] - net_fc
    rho = np.zeros_like(net_fc)
    counts = np.zeros(net_fc.shape, dtype=np.int64)
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
    return net_fc, err, rho, counts, net_fc + rho


def protection_count_checks(counts):
    """Sample counts must match the registered rule for every late-enough publication day."""
    k = np.arange(60, NATURAL_DAYS)
    assert np.all(counts[60:, 0, 144] == 27), counts[60:, 0, 144]
    assert np.all(counts[60:, 0, 0:TEMPLATE_SLOTS] == 28)
    for v in range(1, len(UPDATE_HOURS)):
        assert np.all(counts[60:, v, FIRST_TARGET[v]:] == 28), v
    # explicit order-statistic spot check on two groups
    return dict(h144_0am=27, other_0am=28, later_versions=28)


# ======================================================================================
# §4  求解内核：自由末态名义 MILP（0:00 计划与日内修订共用）
# ======================================================================================
def build_interval_model(net_kwh, price_day, initial, original=None):
    """Nominal free-terminal MILP over one contiguous horizon of n 10-minute intervals.

    Variable order: ``q[n], c[n], d[n], w[n], S[0..n], z[n], a[n]`` (``a`` only with ``original``).

    ``min sum_t p_t q_t + 0.5 p_t |q_t - q0_t|`` subject to
    ``q - c + d - w = n``, ``S_{t+1} = S_t + eta c_t - d_t/eta``,
    ``0 <= c <= M z``, ``0 <= d <= M(1-z)``, ``z`` binary,
    ``S in [1200, 10800]`` with ``S_0 = initial`` and ``S_n`` free inside the band,
    ``q_t - a_t <= q0_t``, ``-q_t - a_t <= -q0_t``, ``a_t >= 0``.
    """
    net = np.asarray(net_kwh, dtype=float)
    price = np.asarray(price_day, dtype=float)
    n = int(net.size)
    assert n >= 1 and net.shape == price.shape and np.isfinite(net).all()
    assert np.isfinite(price).all() and (price > 0).all()
    q0 = None if original is None else np.asarray(original, dtype=float)
    if q0 is not None:
        assert q0.shape == (n,)
    i_c, i_d, i_w, i_s = n, 2 * n, 3 * n, 4 * n
    i_z = 5 * n + 1
    i_a = i_z + n
    n_vars = i_a + (n if q0 is not None else 0)

    objective = np.zeros(n_vars)
    objective[:n] = price
    if q0 is not None:
        objective[i_a:i_a + n] = 0.5 * price

    n_rows = 4 * n + (2 * n if q0 is not None else 0)
    matrix = lil_matrix((n_rows, n_vars))
    lower_rows = np.full(n_rows, -np.inf)
    upper_rows = np.zeros(n_rows)
    for t in range(n):
        matrix[t, t], matrix[t, i_c + t] = 1.0, -1.0
        matrix[t, i_d + t], matrix[t, i_w + t] = 1.0, -1.0
        lower_rows[t] = upper_rows[t] = float(net[t])
        matrix[n + t, i_s + t + 1], matrix[n + t, i_s + t] = 1.0, -1.0
        matrix[n + t, i_c + t], matrix[n + t, i_d + t] = -ETA, 1.0 / ETA
        lower_rows[n + t] = upper_rows[n + t] = 0.0
        matrix[2 * n + t, i_c + t], matrix[2 * n + t, i_z + t] = 1.0, -CAP_KWH
        upper_rows[2 * n + t] = 0.0
        matrix[3 * n + t, i_d + t], matrix[3 * n + t, i_z + t] = 1.0, CAP_KWH
        upper_rows[3 * n + t] = CAP_KWH
        if q0 is not None:
            matrix[4 * n + t, t], matrix[4 * n + t, i_a + t] = 1.0, -1.0
            upper_rows[4 * n + t] = float(q0[t])
            matrix[5 * n + t, t], matrix[5 * n + t, i_a + t] = -1.0, -1.0
            upper_rows[5 * n + t] = -float(q0[t])

    lower = np.zeros(n_vars)
    upper = np.full(n_vars, np.inf)
    upper[i_c:i_d] = CAP_KWH
    upper[i_d:i_w] = CAP_KWH
    lower[i_s:i_z] = STATE_MIN_KWH
    upper[i_s:i_z] = STATE_MAX_KWH
    lower[i_s] = upper[i_s] = float(initial)
    lower[i_z:i_a] = 0.0
    upper[i_z:i_a] = 1.0
    integrality = np.zeros(n_vars)
    integrality[i_z:i_a] = 1.0
    return dict(n=n, n_vars=n_vars, n_rows=n_rows, i_c=i_c, i_d=i_d, i_w=i_w, i_s=i_s,
                i_z=i_z, i_a=i_a, has_adjustment=q0 is not None, initial=float(initial),
                objective=objective, integrality=integrality,
                bounds=Bounds(lower, upper),
                constraints=LinearConstraint(matrix.tocsr(), lower_rows, upper_rows),
                lower=lower, upper=upper)


def solve_interval(net_kwh, price_day, initial, original=None, context=""):
    """Solve one nominal MILP, assert feasibility, and return (q, c, d, w, S, summary)."""
    model = build_interval_model(net_kwh, price_day, initial, original)
    n = model["n"]
    started = time.perf_counter()
    result = milp(model["objective"], integrality=model["integrality"], bounds=model["bounds"],
                  constraints=model["constraints"], options={"mip_rel_gap": 1e-9, "time_limit": 120})
    elapsed = time.perf_counter() - started
    if not result.success:
        raise RuntimeError(f"HiGHS {context}: {result.message}")
    x = result.x
    q, c, d, w = x[:n], x[n:2 * n], x[2 * n:3 * n], x[3 * n:4 * n]
    state = x[4 * n:5 * n + 1]
    z = x[model["i_z"]:model["i_a"]]
    net = np.asarray(net_kwh, dtype=float)
    checks = dict(
        bus_balance_max_abs_kWh=float(np.max(np.abs(q - c + d - w - net))),
        battery_balance_max_abs_kWh=float(np.max(np.abs(np.diff(state) - ETA * c + d / ETA))),
        first_state_error_kWh=float(abs(state[0] - float(initial))),
        state_bound_violation_kWh=float(max(0.0, STATE_MIN_KWH - state.min(),
                                            state.max() - STATE_MAX_KWH)),
        flow_bound_violation_kWh=float(max(0.0, -min(q.min(), c.min(), d.min(), w.min()),
                                           c.max() - CAP_KWH, d.max() - CAP_KWH)),
        simultaneous_charge_discharge_kWh=float(np.minimum(c, d).max()),
        binary_integrality_error=float(np.max(np.abs(z - np.rint(z)))),
        binary_gate_violation_kWh=float(max(0.0, np.max(c - CAP_KWH * z),
                                            np.max(d - CAP_KWH * (1 - z)))))
    worst = max(checks.values())
    if worst >= ENERGY_TOL_KWH:
        raise AssertionError(f"nominal MILP feasibility {worst:.3e} kWh {context}")
    if original is not None:
        q0 = np.asarray(original, dtype=float)
        expected = float(price_day @ q + 0.5 * np.asarray(price_day) @ np.abs(q - q0))
        if abs(float(result.fun) - expected) >= SELECT_TOL_YUAN:
            raise AssertionError(f"revision objective mismatch {context}")
    summary = dict(objective_yuan=float(result.fun), mip_gap=float(getattr(result, "mip_gap", 0.0)),
                   seconds=elapsed, max_violation_kWh=worst, n_intervals=int(n),
                   context=context)
    return q, c, d, w, state, summary


# ======================================================================================
# §5  实际反馈（feedback_step / feedback 逐字转写自 29 号）
# ======================================================================================
def feedback_step(u, state):
    """Surplus charges, deficit discharges, the rest is emergency; never charges from emergency."""
    if u >= 0:
        charge = min(u, CAP_KWH, max(0.0, (STATE_MAX_KWH - state) / ETA))
        return charge, 0.0, 0.0, u - charge, state + ETA * charge
    discharge = min(-u, CAP_KWH, max(0.0, (state - STATE_MIN_KWH) * ETA))
    emergency = -u - discharge
    return 0.0, discharge, emergency, 0.0, state - discharge / ETA


def feedback(plan_kwh, net_kwh, initial):
    plan = np.asarray(plan_kwh, dtype=float)
    net = np.asarray(net_kwh, dtype=float)
    c, d, e, w = [np.zeros(len(plan)) for _ in range(4)]
    states = np.zeros(len(plan) + 1)
    states[0] = float(initial)
    for t in range(len(plan)):
        c[t], d[t], e[t], w[t], states[t + 1] = feedback_step(plan[t] - net[t], states[t])
    return c, d, e, w, states


def score_components(q, q0_seg, price_seg, net_seg, state):
    """Three components of one candidate's predicted cost, in yuan.

    ``ordinary + adjustment + 5 x predicted emergency``. The emergency term is the greedy
    feedback's predicted shortfall under the same protection used for the decision, so it is a
    deterministic score, not an expectation.
    """
    price = np.asarray(price_seg, dtype=float)
    _, _, emergency, _, _ = feedback(q, net_seg, state)
    ordinary = float(np.sum(price * np.asarray(q)))
    adjustment = float(np.sum(0.5 * price * np.abs(np.asarray(q) - np.asarray(q0_seg))))
    predicted_emergency = float(np.sum(EMERGENCY_MULTIPLIER * price * emergency))
    return ordinary, adjustment, predicted_emergency


def score_plan(q, q0_seg, price_seg, net_seg, state):
    """Predicted total cost: ordinary + adjustment + 5x predicted emergency."""
    return float(sum(score_components(q, q0_seg, price_seg, net_seg, state)))


# ======================================================================================
# §6  调度：B1 / B2 单组滚动执行
# ======================================================================================
def run_group(group, protected, truth_load, truth_pv, truth_net, price_src, price_nat,
              initial, carry, carry0, carry_version, carry_owner,
              k_first=EVAL_FIRST, k_last=EVAL_LAST, verbose=True):
    """Sequential natural-day execution with the midnight carry; builds the unified ledger.

    ``k_first``/``k_last`` exist for staged debugging runs; the registered experiment uses the
    full evaluation period.
    """
    assert group in ("B1", "B2")
    rows, plans, decisions, solves, nominal = [], [], [], [], []
    state = float(initial)
    carry, carry0 = float(carry), float(carry0)

    def plan_archive(k, published_at, idx, q0_seg, previous, previous_versions, candidate,
                     accepted, decision):
        """One row per planned 10-minute slot: original plan, prior effective plan, candidate."""
        for offset in range(len(q0_seg)):
            plans.append(dict(
                group=group, decision=decision, published_at=published_at,
                interval_start=(BASE + timedelta(days=k, minutes=10 * (idx + offset + 1))
                                ).isoformat(sep=" "),
                q0_kWh=float(q0_seg[offset]),
                previous_kWh=float(previous[offset]) if previous is not None else float(q0_seg[offset]),
                previous_version=(previous_versions[offset] if previous_versions is not None
                                  else published_at),
                candidate_kWh=float(candidate[offset]), accepted=bool(accepted)))

    def nominal_rows(k, update_hour, idx, q_nominal, state_nominal):
        """Nominal MILP quantities and states, per segment, for the delivery log."""
        for offset, value in enumerate(q_nominal):
            nominal.append(dict(
                group=group, date=(BASE + timedelta(days=k)).strftime("%Y-%m-%d"),
                update_hour=update_hour, segment_index=offset,
                interval_start=(BASE + timedelta(days=k, minutes=10 * (idx + offset + 1))
                                ).isoformat(sep=" "),
                q_nominal_kWh=float(value), state_nominal_kWh=float(state_nominal[offset])))

    def record(k, h, q_eff, q0, owner, version):
        nonlocal state
        start = BASE + timedelta(days=k, minutes=10 * h)
        before = state
        c, d, e, w, state = feedback_step(float(q_eff) - truth_net[k, h], state)
        p = float(price_nat[h])
        rows.append(dict(
            group=group, interval_start=start.isoformat(sep=" "),
            interval_end=(start + timedelta(minutes=10)).isoformat(sep=" "),
            owner_date=owner, effective_version=version, price_yuan_kWh=p,
            actual_load_kW=float(truth_load[k, h]), actual_pv_kW=float(truth_pv[k, h]),
            net_kWh=float(truth_net[k, h]), q0_kWh=float(q0), q_eff_kWh=float(q_eff),
            charge_kWh=c, discharge_kWh=d, emergency_kWh=e, unused_kWh=w,
            state_start_kWh=before, state_end_kWh=state,
            ordinary_cost_yuan=p * float(q_eff),
            adjustment_cost_yuan=0.5 * p * abs(float(q_eff) - float(q0)),
            emergency_cost_yuan=EMERGENCY_MULTIPLIER * p * e))

    for k in range(k_first, k_last + 1):
        day = (BASE + timedelta(days=k)).strftime("%Y-%m-%d")
        # 0:00 plan from the causal midnight estimate
        _, _, _, _, estimated = feedback_step(carry - float(protected[k, 0, 0]), state)
        q0, _, _, _, q0_state, summary = solve_interval(protected[k, 0, 1:TARGETS], price_src,
                                                       estimated,
                                                       context=f"{group} {day} 00:00 plan")
        q_eff = q0.copy()
        versions = np.array([f"{day} 00:00"] * TEMPLATE_SLOTS, dtype=object)
        solves.append(dict(group=group, date=day, update_hour=0, kind="plan", **summary))
        plan_archive(k, f"{day} 00:00", 0, q0, None, None, q0, True, "initial_plan")
        nominal_rows(k, 0, 0, q0, q0_state)
        # midnight 00:00-00:10 inherits the previous publication's carry commitment
        record(k, 0, carry, carry0, carry_owner, carry_version)
        for h in range(1, TEMPLATE_SLOTS):
            if group == "B2" and h in REVISION_HOURS:
                v = REVISION_HOURS.index(h) + 1
                idx = h - 1
                remaining = protected[k, v, h:TARGETS]
                price_seg = price_src[idx:]
                q0_seg = q0[idx:]
                old = q_eff[idx:].copy()
                old_versions = versions[idx:].copy()
                old_parts = score_components(old, q0_seg, price_seg, remaining, state)
                old_score = float(sum(old_parts))
                candidate, _, _, _, cand_state, summ = solve_interval(
                    remaining, price_seg, state, q0_seg,
                    context=f"{group} {day} {UPDATE_HOURS[v]:02d}:00")
                new_parts = score_components(candidate, q0_seg, price_seg, remaining, state)
                new_score = float(sum(new_parts))
                accepted = bool(new_score < old_score - SELECT_TOL_YUAN)
                solves.append(dict(group=group, date=day, update_hour=UPDATE_HOURS[v],
                                   kind="revision", **summ))
                decisions.append(dict(
                    group=group, date=day, update_hour=UPDATE_HOURS[v],
                    published_at=f"{day} {UPDATE_HOURS[v]:02d}:00",
                    remaining_intervals=int(len(old)), state_at_update_kWh=float(state),
                    old_ordinary_yuan=old_parts[0], old_adjustment_yuan=old_parts[1],
                    old_predicted_emergency_yuan=old_parts[2], old_predicted_total_yuan=old_score,
                    new_ordinary_yuan=new_parts[0], new_adjustment_yuan=new_parts[1],
                    new_predicted_emergency_yuan=new_parts[2], new_predicted_total_yuan=new_score,
                    delta_predicted_yuan=new_score - old_score, accepted=accepted,
                    revision_kWh=float(np.abs(candidate - old).sum())))
                plan_archive(k, f"{day} {UPDATE_HOURS[v]:02d}:00", idx, q0_seg, old, old_versions,
                             candidate, accepted, "revision")
                nominal_rows(k, UPDATE_HOURS[v], idx, candidate, cand_state)
                if accepted:
                    q_eff[idx:] = candidate
                    versions[idx:] = f"{day} {UPDATE_HOURS[v]:02d}:00"
            record(k, h, q_eff[h - 1], q0[h - 1], day, versions[h - 1])
        carry, carry0 = float(q_eff[-1]), float(q0[-1])
        carry_version, carry_owner = str(versions[-1]), day
        if verbose and ((k - k_first) % 30 == 29 or k == k_last):
            print(f"  {group} {day} done ({len(solves)} solves)", flush=True)

    # 2026-01-01 00:00-00:10 belongs to the 2025-12-31 template; billed from its 0:00 plan
    tail_net = float(truth_net[EVAL_LAST, TEMPLATE_SLOTS])
    before = state
    c, d, e, w, state = feedback_step(carry - tail_net, state)
    p = float(price_src[TEMPLATE_SLOTS - 1])
    rows.append(dict(
        group=group, interval_start="2026-01-01 00:00:00", interval_end="2026-01-01 00:10:00",
        owner_date="2025-12-31", effective_version=carry_version, price_yuan_kWh=p,
        actual_load_kW=float(truth_load[EVAL_LAST, TEMPLATE_SLOTS]),
        actual_pv_kW=float(truth_pv[EVAL_LAST, TEMPLATE_SLOTS]), net_kWh=tail_net,
        q0_kWh=carry0, q_eff_kWh=carry, charge_kWh=c, discharge_kWh=d, emergency_kWh=e,
        unused_kWh=w, state_start_kWh=before, state_end_kWh=state,
        ordinary_cost_yuan=p * carry,
        adjustment_cost_yuan=0.5 * p * abs(carry - carry0),
        emergency_cost_yuan=EMERGENCY_MULTIPLIER * p * e))
    return (pd.DataFrame(rows)[DISPATCH_COLS], pd.DataFrame(plans), pd.DataFrame(decisions),
            pd.DataFrame(solves), pd.DataFrame(nominal))


# ======================================================================================
# §7  B0 只读载入：问题二新时间口径自由末态账本
# ======================================================================================
def load_b0(truth_load, truth_pv, truth_net):
    """Read the saved N_free ledger and rebuild the unified 48097-row frame; nothing is solved."""
    natural = pd.read_csv(Q2_DIR / "N_free/natural_dispatch.csv",
                          parse_dates=["interval_start", "interval_end"])
    natural = natural[natural.date >= "2025-02-01"].copy()
    assert len(natural) == 48096, len(natural)
    starts = natural["interval_start"]
    is_midnight = midnight_mask(starts)
    owner = starts.dt.normalize() - pd.to_timedelta(np.where(is_midnight, 1, 0), unit="D")
    frame = pd.DataFrame({
        "group": "B0", "interval_start": starts.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "interval_end": natural["interval_end"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "owner_date": owner.dt.strftime("%Y-%m-%d"),
        "effective_version": np.where(is_midnight,
                                      (owner.dt.strftime("%Y-%m-%d") + " 00:00"),
                                      (starts.dt.normalize().dt.strftime("%Y-%m-%d") + " 00:00")),
        "price_yuan_kWh": natural.price_yuan_kWh, "actual_load_kW": natural.load_kW,
        "actual_pv_kW": natural.pv_kW, "net_kWh": natural.net_kWh,
        "q0_kWh": natural.plan_kWh, "q_eff_kWh": natural.plan_kWh,
        "charge_kWh": natural.charge_kWh, "discharge_kWh": natural.discharge_kWh,
        "emergency_kWh": natural.emergency_kWh, "unused_kWh": natural.unused_kWh,
        "state_start_kWh": natural.state_start_kWh, "state_end_kWh": natural.state_end_kWh,
        "ordinary_cost_yuan": natural.planned_cost_yuan, "adjustment_cost_yuan": 0.0,
        "emergency_cost_yuan": natural.emergency_cost_yuan})
    tail = pd.read_csv(Q2_DIR / "tail_interval.csv")
    tail = tail[tail.strategy_id == "N_free"].iloc[0]
    k_tail = EVAL_LAST
    tail_row = dict(
        group="B0", interval_start="2026-01-01 00:00:00", interval_end="2026-01-01 00:10:00",
        owner_date="2025-12-31", effective_version="2025-12-31 00:00",
        price_yuan_kWh=float(tail["price"]), actual_load_kW=float(truth_load[k_tail, TEMPLATE_SLOTS]),
        actual_pv_kW=float(truth_pv[k_tail, TEMPLATE_SLOTS]),
        net_kWh=float(truth_net[k_tail, TEMPLATE_SLOTS]), q0_kWh=float(tail["planned_kWh"]),
        q_eff_kWh=float(tail["planned_kWh"]), charge_kWh=float(tail["charge_kWh"]),
        discharge_kWh=float(tail["discharge_kWh"]), emergency_kWh=float(tail["emergency_kWh"]),
        unused_kWh=float(tail["unused_kWh"]), state_start_kWh=float(tail["state_start_kWh"]),
        state_end_kWh=float(tail["state_end_kWh"]),
        ordinary_cost_yuan=float(tail["planned_cost_yuan"]), adjustment_cost_yuan=0.0,
        emergency_cost_yuan=float(tail["emergency_cost_yuan"]))
    frame = pd.concat([frame, pd.DataFrame([tail_row])], ignore_index=True)
    frame = frame[DISPATCH_COLS]
    initial = float(frame.state_start_kWh.iloc[0])
    carry = float(frame.q_eff_kWh.iloc[0])
    ledger = load_json(Q2_DIR / "N_free/ledger.json")
    assert abs(initial - ledger["ledger_A"]["initial_kWh"]) < ENERGY_TOL_KWH
    assert abs(initial - PUBLIC_INITIAL_KWH) < ENERGY_TOL_KWH, initial
    assert abs(carry - PUBLIC_CARRY_KWH) < ENERGY_TOL_KWH, carry
    meta = dict(initial_kWh=initial, carry_kWh=carry,
                reference_total_yuan=ledger["ledger_A"]["total_cost_yuan"])
    return frame, meta


# ======================================================================================
# §8  账本、汇总与派生输出
# ======================================================================================
def settlement_residual(frame):
    expected = (frame.price_yuan_kWh * frame.q_eff_kWh
                + 0.5 * frame.price_yuan_kWh * (frame.q_eff_kWh - frame.q0_kWh).abs()
                + EMERGENCY_MULTIPLIER * frame.price_yuan_kWh * frame.emergency_kWh)
    actual = (frame.ordinary_cost_yuan + frame.adjustment_cost_yuan
              + frame.emergency_cost_yuan)
    return float((expected - actual).abs().max())


def physics_residual(frame):
    balance = (frame.q_eff_kWh + frame.discharge_kWh + frame.emergency_kWh
               - frame.net_kWh - frame.charge_kWh - frame.unused_kWh)
    recurrence = (frame.state_end_kWh - frame.state_start_kWh
                  - ETA * frame.charge_kWh + frame.discharge_kWh / ETA)
    continuity = frame.state_start_kWh.to_numpy()[1:] - frame.state_end_kWh.to_numpy()[:-1]
    worst = max(float(balance.abs().max()), float(recurrence.abs().max()),
                float(np.abs(continuity).max()))
    if worst >= ENERGY_TOL_KWH:
        raise AssertionError(f"actual physics residual {worst:.3e} kWh")
    assert frame[["state_start_kWh", "state_end_kWh"]].min().min() >= STATE_MIN_KWH - ENERGY_TOL_KWH
    assert frame[["state_start_kWh", "state_end_kWh"]].max().max() <= STATE_MAX_KWH + ENERGY_TOL_KWH
    assert frame[["charge_kWh", "discharge_kWh"]].max().max() <= CAP_KWH + ENERGY_TOL_KWH
    assert frame[["q_eff_kWh", "charge_kWh", "discharge_kWh", "emergency_kWh",
                  "unused_kWh"]].min().min() >= -ENERGY_TOL_KWH
    assert float(np.minimum(frame.charge_kWh, frame.discharge_kWh).max()) < ENERGY_TOL_KWH
    assert float(np.minimum(frame.emergency_kWh,
                            frame.charge_kWh + frame.unused_kWh).max()) < ENERGY_TOL_KWH
    return worst


def emergency_events(natural):
    """Consecutive emergency slots merged into events, split at natural-day boundaries."""
    flags = (natural.emergency_kWh > ENERGY_TOL_KWH).to_numpy()
    dates = natural.interval_start.dt.strftime("%Y-%m-%d").to_numpy()
    count = 0
    for i in range(len(flags)):
        if flags[i] and (i == 0 or dates[i] != dates[i - 1] or not flags[i - 1]):
            count += 1
    return count


def finalize(group, frame, truth_net):
    """B-class assertions on the unified frame and persist the per-group ledgers."""
    frame = frame.copy()
    for col in ("interval_start", "interval_end"):
        frame[col] = pd.to_datetime(frame[col])
    frame = frame.sort_values("interval_start").reset_index(drop=True)
    assert len(frame) == 48097, len(frame)
    assert not frame.interval_start.duplicated().any()
    assert frame.interval_start.diff().dropna().eq(pd.Timedelta(minutes=10)).all()
    assert frame.state_start_kWh.iloc[0] > 0
    frame["total_cost_yuan"] = (frame.ordinary_cost_yuan + frame.adjustment_cost_yuan
                                + frame.emergency_cost_yuan)
    settlement = settlement_residual(frame)
    if settlement >= SELECT_TOL_YUAN:
        raise AssertionError(f"{group} settlement identity residual {settlement:.3e} yuan")
    physics = physics_residual(frame)
    # the stored true net demand must be reproducible from the shared truth arrays
    starts = frame.interval_start
    is_midnight = midnight_mask(starts)
    k_pub = np.where(is_midnight,
                     ((starts.dt.normalize() - BASE).dt.days).to_numpy() - 1,
                     ((starts.dt.normalize() - BASE).dt.days).to_numpy())
    h_pub = np.where(is_midnight, TEMPLATE_SLOTS,
                     ((starts - starts.dt.normalize()).dt.total_seconds() / 600).astype(int).to_numpy())
    assert np.allclose(frame.net_kWh.to_numpy(), truth_net[k_pub, h_pub], atol=ENERGY_TOL_KWH)
    natural, template = frame.iloc[:-1], frame.iloc[1:]
    assert len(natural) == 48096 and len(template) == 48096
    monthly = (natural.assign(month=natural.interval_start.dt.strftime("%Y-%m"))
               .groupby("month", as_index=False)
               [["ordinary_cost_yuan", "adjustment_cost_yuan", "emergency_cost_yuan",
                 "total_cost_yuan"]].sum())
    monthly.insert(0, "group", group)
    daily = (natural.assign(date=natural.interval_start.dt.strftime("%Y-%m-%d"))
             .groupby("date", as_index=False)
             .agg(total_cost_yuan=("total_cost_yuan", "sum"),
                  emergency_kWh=("emergency_kWh", "sum"),
                  state_start_kWh=("state_start_kWh", "first"),
                  state_end_kWh=("state_end_kWh", "last")))
    daily.insert(0, "group", group)
    frame_to_csv(frame, OUT / f"{group}_dispatch.csv")
    summary = dict(
        group=group, natural_total_yuan=float(natural.total_cost_yuan.sum()),
        template_total_yuan=float(template.total_cost_yuan.sum()),
        ordinary_cost_yuan=float(natural.ordinary_cost_yuan.sum()),
        adjustment_cost_yuan=float(natural.adjustment_cost_yuan.sum()),
        emergency_cost_yuan=float(natural.emergency_cost_yuan.sum()),
        emergency_kWh=float(natural.emergency_kWh.sum()),
        emergency_intervals=int((natural.emergency_kWh > ENERGY_TOL_KWH).sum()),
        emergency_days=int(natural.loc[natural.emergency_kWh > ENERGY_TOL_KWH, "interval_start"]
                           .dt.strftime("%Y-%m-%d").nunique()),
        emergency_events=int(emergency_events(natural)),
        unused_kWh=float(natural.unused_kWh.sum()),
        charge_kWh=float(natural.charge_kWh.sum()),
        discharge_kWh=float(natural.discharge_kWh.sum()),
        loss_kWh=float((1 - ETA) * natural.charge_kWh.sum()
                       + (1 / ETA - 1) * natural.discharge_kWh.sum()),
        initial_state_kWh=float(frame.state_start_kWh.iloc[0]),
        final_natural_state_kWh=float(natural.state_end_kWh.iloc[-1]),
        final_template_state_kWh=float(template.state_end_kWh.iloc[-1]),
        settlement_residual_yuan=settlement, physics_residual_kWh=physics)
    return summary, monthly, daily, frame


# ======================================================================================
# §9  三类精简验证
# ======================================================================================
def check_small_examples():
    """A1: pure-arithmetic settlement, order statistic, feedback boundary, event merge."""
    out = {}
    settle = lambda q0, q, p: p * q + 0.5 * p * abs(q - q0)
    cases = {0: 50.0, 80: 90.0, 100: 100.0, 120: 130.0}
    for q, expected in cases.items():
        assert abs(settle(100, q, 1.0) - expected) < 1e-12, (q, expected)
    out["settlement_examples_yuan"] = {str(k): v for k, v in cases.items()}
    out["settlement_all_cancelled_yuan"] = settle(100, 0, 1.0)
    out["settlement_chain_100_80_100_yuan"] = settle(100, 100, 1.0)   # final version bills once
    # integer order statistic: position ceil(0.8 m); m<7 falls back to zero; negatives kept
    positions = {}
    for m in (6, 7, 27, 28):
        arr = np.arange(1, m + 1, dtype=float)
        arr[0] = -5.0                                    # a negative residual must survive
        ordered = np.sort(arr)
        positions[m] = 0.0 if m < MIN_SAMPLES else float(ordered[math.ceil(ALPHA * m) - 1])
    assert positions[6] == 0.0, positions
    assert positions[7] == 6.0, positions
    assert positions[27] == 22.0 and positions[28] == 23.0, positions
    out["order_statistic"] = positions
    # feedback boundaries: empty, full, power saturation both directions
    full = feedback_step(+1.0, STATE_MAX_KWH)
    assert full[0] == 0.0 and abs(full[3] - 1.0) < 1e-12, full
    empty = feedback_step(-1.0, STATE_MIN_KWH)
    assert empty[1] == 0.0 and abs(empty[2] - 1.0) < 1e-12, empty
    sat_c = feedback_step(2000.0, 6000.0)
    assert abs(sat_c[0] - CAP_KWH) < 1e-12 and abs(sat_c[3] - (2000.0 - CAP_KWH)) < 1e-12, sat_c
    sat_d = feedback_step(-2000.0, 6000.0)
    assert abs(sat_d[1] - CAP_KWH) < 1e-12 and abs(sat_d[2] - (2000.0 - CAP_KWH)) < 1e-12, sat_d
    out["feedback_boundaries"] = dict(full_unused_kWh=full[3], empty_emergency_kWh=empty[2],
                                      saturated_charge_kWh=sat_c[0], saturated_discharge_kWh=sat_d[1])
    # event merge across a day boundary counts as two events
    probe = pd.DataFrame({"emergency_kWh": [1.0, 1.0, 0.0, 1.0, 1.0],
                          "interval_start": pd.to_datetime(
                              ["2025-03-01 23:40", "2025-03-01 23:50", "2025-03-02 00:00",
                               "2025-03-02 00:10", "2025-03-02 00:20"])})
    assert emergency_events(probe) == 2
    out["event_split_check"] = 2
    return out


def unrealized_mask(k, hour):
    """Targets whose interval has not yet ended at the publication instant (k day, ``hour``).

    This is the complement of the sample-availability filter in :func:`build_protection`, so
    perturbing exactly these cells probes the whole window/availability logic: any sample the
    filter wrongly admitted would carry an unrealised (perturbed) truth.
    """
    j = np.arange(NATURAL_DAYS)[:, None]
    h = np.arange(TARGETS)[None, :]
    return j * TEMPLATE_SLOTS + h + 1 > k * TEMPLATE_SLOTS + hour * 6


def perturb_arrays(truth_net, pv_forecasts, k, hour, v_min):
    """Perturb every truth not yet realised at the decision instant, plus the forecasts that are
    not published yet (``v >= v_min``).

    Already-published forecasts are deliberately left alone: a new forecast legitimately changes
    the decision, so perturbing it would test nothing about the information boundary.
    """
    mask = unrealized_mask(k, hour)
    p_net = truth_net.copy()
    p_net[mask] = p_net[mask] * PERTURB_TRUTH_SCALE
    pv_p = pv_forecasts.copy()
    for v in range(v_min, len(UPDATE_HOURS)):
        pv_p[k, v, :] = pv_p[k, v, :] + PERTURB_PV_ADD_KW
    return p_net, pv_p, int(mask.sum())


def check_information_boundary(protected, price_src, truth_net, pv_forecasts, issued_load,
                               b1_frame, b2_frame, plan_versions):
    """A2: one day (2025-06-21) with clean/perturbed inputs at the 0:00 and 06:00 decisions.

    Every unrealised truth cell is perturbed and the protection archive is rebuilt from the
    perturbed arrays, so a leak anywhere in the window/availability logic surfaces as a changed
    protection, auxiliary state, candidate plan or accept decision. Only the versions already
    published at that instant are held fixed (``v_min``).
    """
    k, day = PERTURB_DAY, "2025-06-21"
    out = dict(day=day, k=int(k), truth_scale=PERTURB_TRUTH_SCALE, pv_shift_kW=PERTURB_PV_ADD_KW)
    assert (BASE + timedelta(days=k)).strftime("%Y-%m-%d") == day

    # ---- 0:00: only the 0:00 forecast is published; the current midnight has not ended yet
    p_net, pv_p, n_pert = perturb_arrays(truth_net, pv_forecasts, k, 0, v_min=1)
    protected_p = build_protection(build_forecast_archive(issued_load, pv_p), p_net)[4]
    same_inputs = bool(np.allclose(protected[k, 0, 1:TARGETS], protected_p[k, 0, 1:TARGETS],
                                   rtol=0, atol=0))
    clean_row = b1_frame[(b1_frame.interval_start.dt.strftime("%Y-%m-%d") == day)
                         & midnight_mask(b1_frame.interval_start)]
    assert len(clean_row) == 1
    state_before = float(clean_row.state_start_kWh.iloc[0])
    carry = float(clean_row.q_eff_kWh.iloc[0])
    _, _, _, _, est_clean = feedback_step(carry - float(protected[k, 0, 0]), state_before)
    _, _, _, _, est_pert = feedback_step(carry - float(protected_p[k, 0, 0]), state_before)
    q_pert, _, _, _, _, summ = solve_interval(protected_p[k, 0, 1:TARGETS], price_src, est_pert,
                                              context=f"A2 {day} perturbed 0:00")
    clean_plan = b1_frame[(b1_frame.interval_start > pd.Timestamp(day))
                          & (b1_frame.interval_start <= pd.Timestamp(day) + pd.Timedelta(hours=24))]
    clean_q = clean_plan.sort_values("interval_start").q0_kWh.to_numpy()
    assert len(clean_q) == TEMPLATE_SLOTS
    delta = float(np.max(np.abs(q_pert - clean_q)))
    out["midnight_perturbed_truth_cells"] = n_pert
    out["midnight_inputs_identical"] = same_inputs
    out["midnight_aux_estimate_delta_kWh"] = float(abs(est_pert - est_clean))
    out["midnight_plan_max_delta_kWh"] = delta
    out["midnight_objective_yuan"] = float(summ["objective_yuan"])
    if not same_inputs or delta >= ENERGY_TOL_KWH or abs(est_pert - est_clean) >= ENERGY_TOL_KWH:
        raise AssertionError(f"A2 midnight leak: inputs_equal={same_inputs} delta={delta}")

    # ---- 06:00: the 0:00 and 06:00 forecasts are published; 12:00/18:00 are not
    p_net, pv_p, n_pert = perturb_arrays(truth_net, pv_forecasts, k, 6, v_min=2)
    protected_p = build_protection(build_forecast_archive(issued_load, pv_p), p_net)[4]
    v, h0 = 1, FIRST_TARGET[1]
    same_06 = bool(np.allclose(protected[k, v, h0:TARGETS], protected_p[k, v, h0:TARGETS],
                               rtol=0, atol=0))
    rows_b2 = b2_frame[b2_frame.interval_start.dt.strftime("%Y-%m-%d") == day].sort_values(
        "interval_start")
    state_0600 = float(rows_b2.state_end_kWh.iloc[h0 - 1])          # state after executing h=0..35
    # The real 06:00 decision comes from the saved revision record: template slots 35..143,
    # i.e. intervals 06:00 .. the next day 00:00 (109 segments). The saved ``previous`` column is
    # the plan effective before this revision; taking q_eff from the final ledger would instead
    # fold in the later 12:00/18:00 revisions and would also start one slot too early.
    record = plan_versions[(plan_versions.decision == "revision")
                           & (plan_versions.published_at == f"{day} {UPDATE_HOURS[v]:02d}:00")
                           ].copy()
    record["interval_start"] = pd.to_datetime(record.interval_start)
    record = record.sort_values("interval_start")
    assert len(record) == TARGETS - h0, len(record)
    assert record.interval_start.iloc[0] == pd.Timestamp(day) + pd.Timedelta(hours=6), \
        record.interval_start.iloc[0]
    assert record.interval_start.iloc[-1] == pd.Timestamp(day) + pd.Timedelta(hours=24), \
        record.interval_start.iloc[-1]
    q0_seg = record.q0_kWh.to_numpy()
    old_seg = record.previous_kWh.to_numpy()
    saved_candidate = record.candidate_kWh.to_numpy()
    saved_accepted = bool(record.accepted.iloc[0])
    assert (record.previous_version == f"{day} 00:00").all(), \
        "the first revision of the day must face the 0:00 original plan"
    cand_clean, _, _, _, _, _ = solve_interval(protected[k, v, h0:TARGETS], price_src[h0 - 1:],
                                               state_0600, q0_seg,
                                               context=f"A2 {day} clean 06:00")
    cand_pert, _, _, _, _, _ = solve_interval(protected_p[k, v, h0:TARGETS], price_src[h0 - 1:],
                                              state_0600, q0_seg,
                                              context=f"A2 {day} perturbed 06:00")
    old_score = score_plan(old_seg, q0_seg, price_src[h0 - 1:], protected[k, v, h0:TARGETS],
                           state_0600)
    acc_clean = bool(score_plan(cand_clean, q0_seg, price_src[h0 - 1:], protected[k, v, h0:TARGETS],
                                state_0600) < old_score - SELECT_TOL_YUAN)
    acc_pert = bool(score_plan(cand_pert, q0_seg, price_src[h0 - 1:], protected_p[k, v, h0:TARGETS],
                               state_0600) < old_score - SELECT_TOL_YUAN)
    out["six_perturbed_truth_cells"] = n_pert
    out["six_inputs_identical"] = same_06
    out["six_remaining_intervals"] = int(len(q0_seg))
    out["six_window"] = [str(record.interval_start.iloc[0]), str(record.interval_start.iloc[-1])]
    out["six_saved_decision_accepted"] = saved_accepted
    out["six_clean_reproduces_saved_max_delta_kWh"] = float(np.max(np.abs(cand_clean
                                                                          - saved_candidate)))
    out["six_candidate_max_delta_kWh"] = float(np.max(np.abs(cand_pert - cand_clean)))
    out["six_accept_clean"] = acc_clean
    out["six_accept_perturbed"] = acc_pert
    out["six_accept_matches_saved"] = bool(acc_clean == saved_accepted)
    if not same_06 or out["six_candidate_max_delta_kWh"] >= ENERGY_TOL_KWH:
        raise AssertionError(f"A2 06:00 leak: inputs_equal={same_06} "
                             f"delta={out['six_candidate_max_delta_kWh']}")
    if out["six_clean_reproduces_saved_max_delta_kWh"] >= ENERGY_TOL_KWH:
        raise AssertionError("A2 06:00 clean replay does not reproduce the saved decision: "
                             f"{out['six_clean_reproduces_saved_max_delta_kWh']}")
    if not (acc_clean == acc_pert == saved_accepted):
        raise AssertionError(f"A2 06:00 accept decision changed: clean={acc_clean} "
                             f"perturbed={acc_pert} saved={saved_accepted}")
    return out


def check_kernel_agreement(protected, price_src, b1_frame):
    """A3: the verified 29 kernel must reproduce the already-saved B1 plan for one day.

    One extra MILP only: the local side is the saved plan, so this costs a single solve and stays
    inside the task-book's four-solve validation budget. Objective equality is the gate; an
    identical objective with a different plan vector only reflects another equal-cost vertex.
    """
    k = PERTURB_DAY
    day = (BASE + timedelta(days=k)).strftime("%Y-%m-%d")
    rows = b1_frame[b1_frame.interval_start.dt.strftime("%Y-%m-%d") == day].sort_values(
        "interval_start")
    midnight = rows[midnight_mask(rows.interval_start)].iloc[0]
    state_before = float(midnight.state_start_kWh)
    carry = float(midnight.q_eff_kWh)
    _, _, _, _, estimated = feedback_step(carry - float(protected[k, 0, 0]), state_before)
    saved = b1_frame[(b1_frame.interval_start > pd.Timestamp(day))
                     & (b1_frame.interval_start <= pd.Timestamp(day) + pd.Timedelta(hours=24))
                     ].sort_values("interval_start")
    assert len(saved) == TEMPLATE_SLOTS
    saved_q = saved.q0_kWh.to_numpy()
    saved_cost = float(np.sum(price_src * saved_q))
    q2 = q2_kernel()
    qk, _, _, _, _, kernel_summary, _ = q2.solve_day(protected[k, 0, 1:TARGETS], price_src,
                                                     estimated, "free")
    kernel_cost = float(kernel_summary["cost_yuan"])
    delta_cost = abs(saved_cost - kernel_cost)
    delta_q = float(np.max(np.abs(saved_q - qk)))
    out = dict(day=day, saved_plan_objective_yuan=saved_cost, kernel_objective_yuan=kernel_cost,
               objective_delta_yuan=delta_cost, plan_max_delta_kWh=delta_q, extra_solves=1,
               note="one day, one publication; objective equality is the gate, and a nonzero plan "
                    "delta only reflects an alternative equal-cost vertex of the same MILP. This "
                    "does not by itself certify solver behaviour on every date; it checks that the "
                    "transcribed kernel and the one 29 uses give the same objective here.")
    if delta_cost >= SELECT_TOL_YUAN:
        raise AssertionError(f"A3 kernel disagreement: {out}")
    return out


def check_ledger(b0_frame, b0_summary, group_frames, group_summaries):
    """C: read-only reconciliation of both ledgers from the saved frames."""
    out = dict(reference_b0_total_yuan=B0_REFERENCE_YUAN)
    for group in GROUPS:
        frame = group_frames[group]
        natural, template = frame.iloc[:-1], frame.iloc[1:]
        assert len(natural) == 48096 and len(template) == 48096
        assert natural.interval_start.is_monotonic_increasing
        assert template.interval_start.is_monotonic_increasing
        head = float(frame.total_cost_yuan.iloc[0]) if "total_cost_yuan" in frame else 0.0
        out[f"{group}_natural_total_yuan"] = float(natural.total_cost_yuan.sum())
        out[f"{group}_template_total_yuan"] = float(template.total_cost_yuan.sum())
        out[f"{group}_bridge_residual_yuan"] = float(
            (template.total_cost_yuan.sum() - natural.total_cost_yuan.sum())
            - (float(frame.total_cost_yuan.iloc[-1]) - head))
        assert abs(out[f"{group}_bridge_residual_yuan"]) < SELECT_TOL_YUAN
    b0_total = out["B0_natural_total_yuan"]
    out["b0_reference_delta_yuan"] = abs(b0_total - B0_REFERENCE_YUAN)
    assert out["b0_reference_delta_yuan"] < SELECT_TOL_YUAN, out
    return out


# ======================================================================================
# §10  main
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

    inputs = ["附件/附件1.xlsx", "附件/附件2.xlsx",
              "results/q3_pv_time_conversion/pv_10min_forecasts.csv",
              "results/q3_pv_time_conversion/validation.json",
              "results/q2_time_mapping/archive_float.npz",
              "results/q2_time_mapping/N_free/natural_dispatch.csv",
              "results/q2_time_mapping/N_free/template_plan.csv",
              "results/q2_time_mapping/N_free/ledger.json",
              "results/q2_time_mapping/tail_interval.csv",
              "code/29_q2_time_mapping_experiment.py",
              "reports/问题三/问题三_首轮Baseline对照实验方案.md"]
    dependency = dict(PUBLIC_INITIAL_KWH=PUBLIC_INITIAL_KWH, PUBLIC_CARRY_KWH=PUBLIC_CARRY_KWH,
                      B0_REFERENCE_YUAN=B0_REFERENCE_YUAN,
                      perturbation=dict(day=PERTURB_DAY, truth_scale=PERTURB_TRUTH_SCALE,
                                       pv_shift_kW=PERTURB_PV_ADD_KW))
    signature = register(PARAMETERS, inputs, dependency, args.amend_reason)
    print(f"registered signature={signature[:16]}", flush=True)
    if args.mode == "register":
        return

    # ---- inputs ----
    tick = time.perf_counter()
    attachments = read_attachments()
    price_src = attachments["price_src"]
    price_nat = natural_price(price_src)
    load_src, pv_src = attachments["load_src"], attachments["pv_src"]
    truth_load, truth_pv = build_truth(load_src, pv_src)
    truth_net = (truth_load - truth_pv) * DT
    issued_load, issued_pv = read_q2_archive()
    pv_forecasts = read_pv_forecasts()
    timings["inputs_seconds"] = time.perf_counter() - tick

    tick = time.perf_counter()
    net_fc, err, rho, counts, protected = build_protection(
        build_forecast_archive(issued_load, pv_forecasts), truth_net)
    count_summary = protection_count_checks(counts)
    np.savez_compressed(OUT / "prediction_protection.npz", net_forecast=net_fc, error=err,
                        rho=rho, counts=counts, protected=protected, issued_load=issued_load,
                        pv_forecasts=pv_forecasts)
    save_json(OUT / "protection_summary.json", dict(
        count_summary=count_summary, window_days=WINDOW, alpha=ALPHA, min_samples=MIN_SAMPLES,
        finite_net_forecast_cells=int(np.isfinite(net_fc).sum()),
        zero_correction_cells=int((counts < MIN_SAMPLES).sum()),
        h144_m_values=sorted(set(int(v) for v in counts[EVAL_FIRST:, 0, TEMPLATE_SLOTS])),
        other_m_values=sorted(set(int(v) for v in counts[EVAL_FIRST:, 0, :TEMPLATE_SLOTS].ravel())),
        rho_min_kWh=float(np.nanmin(rho)), rho_max_kWh=float(np.nanmax(rho))))
    timings["protection_seconds"] = time.perf_counter() - tick
    print(f"protection archive: {count_summary}", flush=True)

    # ---- A1 ----
    checks = dict(small_examples=check_small_examples())
    print("A1 small examples passed", flush=True)

    # ---- B0 read-only ----
    protected_before = protected_manifest()
    save_json(OUT / "protected_before.json", protected_before)
    b0_frame, b0_meta = load_b0(truth_load, truth_pv, truth_net)
    b0_summary, b0_monthly, b0_daily, b0_frame = finalize("B0", b0_frame, truth_net)
    assert abs(b0_summary["natural_total_yuan"] - B0_REFERENCE_YUAN) < SELECT_TOL_YUAN
    assert abs(b0_meta["initial_kWh"] - PUBLIC_INITIAL_KWH) < ENERGY_TOL_KWH
    print(f"B0 loaded: natural {b0_summary['natural_total_yuan']:,.2f} yuan "
          f"(reference delta {abs(b0_summary['natural_total_yuan'] - B0_REFERENCE_YUAN):.2e})",
          flush=True)

    # ---- B1 / B2 ----
    frames, summaries, monthlies, dailies = {"B0": b0_frame}, {"B0": b0_summary}, {}, {}
    plan_archives = {}
    monthlies["B0"], dailies["B0"] = b0_monthly, b0_daily
    for group in ("B1", "B2"):
        tick = time.perf_counter()
        frame, plans, decisions, solves, nominal = run_group(
            group, protected, truth_load, truth_pv, truth_net, price_src, price_nat,
            b0_meta["initial_kWh"], b0_meta["carry_kWh"], PUBLIC_CARRY_KWH,
            "public_January", "2025-01-31")
        timings[f"{group}_seconds"] = time.perf_counter() - tick
        summary, monthly, daily, frame = finalize(group, frame, truth_net)
        frames[group], summaries[group] = frame, summary
        monthlies[group], dailies[group] = monthly, daily
        plan_archives[group] = plans
        frame_to_csv(plans, OUT / f"{group}_plan_versions.csv")
        if len(decisions):
            frame_to_csv(decisions, OUT / f"{group}_revision_decisions.csv")
        frame_to_csv(solves, OUT / f"{group}_solver_log.csv")
        frame_to_csv(nominal, OUT / f"{group}_nominal_trajectory.csv")
        print(f"{group} finished: natural {summary['natural_total_yuan']:,.2f} yuan, "
              f"emergency {summary['emergency_kWh']:,.1f} kWh, {timings[f'{group}_seconds']:.1f}s",
              flush=True)

    # ---- A2 / A3 ----
    tick = time.perf_counter()
    checks["information_boundary"] = check_information_boundary(
        protected, price_src, truth_net, pv_forecasts, issued_load,
        frames["B1"], frames["B2"], plan_archives["B2"])
    checks["kernel_agreement"] = check_kernel_agreement(protected, price_src, frames["B1"])
    timings["validation_seconds"] = time.perf_counter() - tick
    print(f"A2/A3 passed: midnight delta "
          f"{checks['information_boundary']['midnight_plan_max_delta_kWh']:.3e} kWh, kernel "
          f"objective delta {checks['kernel_agreement']['objective_delta_yuan']:.3e} yuan",
          flush=True)

    # ---- aggregate + C ----
    summary_frame = pd.DataFrame([summaries[g] for g in GROUPS])
    frame_to_csv(summary_frame, OUT / "summary.csv")
    frame_to_csv(pd.concat([monthlies[g] for g in GROUPS], ignore_index=True), OUT / "monthly.csv")
    frame_to_csv(pd.concat([dailies[g] for g in GROUPS], ignore_index=True), OUT / "daily.csv")
    contrast = []
    for treatment, baseline in (("B1", "B0"), ("B2", "B1")):
        c_t = summaries[treatment]["natural_total_yuan"]
        c_b = summaries[baseline]["natural_total_yuan"]
        contrast.append(dict(treatment=treatment, baseline=baseline,
                             treatment_total_yuan=c_t, baseline_total_yuan=c_b,
                             delta_yuan=c_t - c_b,
                             delta_pct=100.0 * (c_t - c_b) / c_b,
                             delta_ordinary_yuan=(summaries[treatment]["ordinary_cost_yuan"]
                                                  - summaries[baseline]["ordinary_cost_yuan"]),
                             delta_adjustment_yuan=(summaries[treatment]["adjustment_cost_yuan"]
                                                    - summaries[baseline]["adjustment_cost_yuan"]),
                             delta_emergency_yuan=(summaries[treatment]["emergency_cost_yuan"]
                                                   - summaries[baseline]["emergency_cost_yuan"])))
    frame_to_csv(pd.DataFrame(contrast), OUT / "contrast.csv")
    monthly_wide = pd.concat([monthlies[g] for g in GROUPS], ignore_index=True).pivot(
        index="month", columns="group", values="total_cost_yuan")
    monthly_wide["B1_minus_B0"] = monthly_wide["B1"] - monthly_wide["B0"]
    monthly_wide["B2_minus_B1"] = monthly_wide["B2"] - monthly_wide["B1"]
    frame_to_csv(monthly_wide.reset_index(), OUT / "monthly_contrasts.csv")
    daily_wide = pd.concat([dailies[g] for g in GROUPS], ignore_index=True).pivot(
        index="date", columns="group", values="total_cost_yuan")
    daily_wide["B1_minus_B0"] = daily_wide["B1"] - daily_wide["B0"]
    daily_wide["B2_minus_B1"] = daily_wide["B2"] - daily_wide["B1"]
    frame_to_csv(daily_wide.reset_index(), OUT / "daily_contrasts.csv")
    selected = pd.concat([
        frames[g].assign(date=frames[g].interval_start.dt.strftime("%Y-%m-%d")).query(
            "date in @SELECTED_DATES") for g in GROUPS], ignore_index=True)
    selected = selected.groupby(["group", "date"], as_index=False).agg(
        ordinary_cost_yuan=("ordinary_cost_yuan", "sum"),
        adjustment_cost_yuan=("adjustment_cost_yuan", "sum"),
        emergency_cost_yuan=("emergency_cost_yuan", "sum"),
        total_cost_yuan=("total_cost_yuan", "sum"),
        emergency_kWh=("emergency_kWh", "sum"), charge_kWh=("charge_kWh", "sum"),
        discharge_kWh=("discharge_kWh", "sum"),
        state_start_kWh=("state_start_kWh", "first"), state_end_kWh=("state_end_kWh", "last"))
    frame_to_csv(selected, OUT / "selected_dates.csv")
    checks["ledger"] = check_ledger(b0_frame, b0_summary, frames, summaries)
    print("C ledger check passed", flush=True)

    # ---- manifest ----
    protected_after = protected_manifest()
    changed = sorted(k for k in protected_after
                     if k in protected_before and protected_after[k] != protected_before[k])
    missing = sorted(k for k in protected_before if k not in protected_after)
    assert not changed and not missing, (changed[:5], missing[:5])
    save_json(OUT / "protected_after.json", protected_after)
    checks["protected_unchanged"] = dict(count=len(protected_before), changed=changed,
                                         missing=missing)
    save_json(OUT / "checks.json", checks)
    extra_solves = dict(A2=3, A3=int(checks["kernel_agreement"].get("extra_solves", 1)))
    extra_solves["total"] = extra_solves["A2"] + extra_solves["A3"]
    validation = dict(status="passed", signature=signature,
                      small_examples=checks["small_examples"],
                      information_boundary=checks["information_boundary"],
                      kernel_agreement=checks["kernel_agreement"], ledger=checks["ledger"],
                      validation_budget=dict(
                          extra_milp_solves=extra_solves, task_book_limit=4,
                          within_limit=bool(extra_solves["total"] <= 4),
                          protected_manifest_scope=(
                              "full protected-tree manifest "
                              f"({len(protected_before):,} files); the task book asks for a "
                              "dependency-only registration, so this is stricter than required")),
                      limits=["A2 is one day and one publication pair, not a full-period sweep",
                              "one perturbed q80 that does not move does not prove every wrongly "
                              "admitted sample would be caught",
                              "A3 checks one day and one publication, not every date",
                              "B0 is a read-only reuse of the Q2 N_free ledger",
                              "nominal MILPs may have equal-cost alternative vertices",
                              "the 2025 year was already used for method design; this is a "
                              "rolling causal backtest, not an independent blind test"])
    save_json(OUT / "validation.json", validation)
    run_manifest = dict(status="complete", started_utc=started_utc, finished_utc=utc_now(),
                        wall_seconds=time.perf_counter() - started, timings=timings,
                        signature=signature, executable=sys.executable, python=sys.version,
                        n_solves={g: (int(len(pd.read_csv(OUT / f'{g}_solver_log.csv')))
                                      if (OUT / f"{g}_solver_log.csv").exists() else 0)
                                  for g in GROUPS},
                        outputs={p.name: digest(p) for p in sorted(OUT.glob("*"))
                                 if p.is_file() and p.name not in REPORT_LAYER_ARTIFACTS})
    save_json(OUT / "run_manifest.json", run_manifest)
    print(json.dumps({"status": "complete", "wall_seconds": run_manifest["wall_seconds"],
                      "B0": summaries["B0"]["natural_total_yuan"],
                      "B1": summaries["B1"]["natural_total_yuan"],
                      "B2": summaries["B2"]["natural_total_yuan"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
