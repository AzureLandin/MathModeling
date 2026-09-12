#!/usr/bin/env python
"""问题三 偏差校正 × 分位水平 四组费用对照实验 —— 计算层。

任务书 ``reports/问题三/问题三_偏差校正与分位水平四组费用实验方案.md``。四组：

===========  ==========================  ======  ==========================================
组别          光伏预测                     分位    调度
===========  ==========================  ======  ==========================================
L80          冻结 Linear                 0.80    只读复用既有 B2 完整账本，不重跑全年
L75          与 L80 相同                  0.75    新增 334 日四节点滚动轨迹
C80          冻结 Bias28 校正预测          0.80    复用已重算 C1 q80，首次运行 334 日
C75          与 C80 相同                  0.75    新增 334 日四节点滚动轨迹
===========  ==========================  ======  ==========================================

职责边界
--------
本脚本只做：登记 → 读输入 → 生成 q75 保护档案 → 三类精简检查 → L75/C80/C75 三条 334 日
滚动求解 → 只读核账 → 落盘 CSV/JSON/NPZ。**不写报告、不画图**；报告与图表由
``code/q3_bias_quantile_cost_report.py`` 只读渲染。修改报告文案不影响本脚本的登记签名，
也不需要重跑任何求解。

复用与独立性
------------
已接收的 MILP、实际反馈、评分、结算与跨日执行原语全部从
``code/q3_rolling_baseline_experiment.py`` 以库的形式导入，不复制、不改写：

* ``read_attachments`` / ``natural_price`` / ``build_truth``：附件与时间口径；
* ``read_pv_forecasts`` / ``read_q2_archive``：Linear 预测与 15 列负载发布档案；
* ``solve_interval`` / ``feedback_step`` / ``score_components`` / ``score_plan``：求解与评分；
* ``run_group``：四节点滚动（三个新组都走同一 ``B2`` 分支，落盘层再赋唯一 strategy_id）；
* ``settlement_residual`` / ``physics_residual`` / ``emergency_events`` / ``midnight_mask``。

按任务书 4.1 节，**不调用旧脚本的 ``main``**，也不调用其全树保护与多项复跑。

运行::

    E:/Anaconda/envs/math_modeling/python.exe code/q3_bias_quantile_cost_experiment.py
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
CODE = ROOT / "code/q3_bias_quantile_cost_experiment.py"
OUT = ROOT / "results/q3_bias_quantile_cost"
FIG = ROOT / "figures/q3_bias_quantile_cost"
REPORT_MD = ROOT / "reports/问题三/问题三_偏差校正与分位水平费用实验结果.md"
PLAN_MD = ROOT / "reports/问题三/问题三_偏差校正与分位水平四组费用实验方案.md"
RB_KERNEL = ROOT / "code/q3_rolling_baseline_experiment.py"
RB_DIR = ROOT / "results/q3_rolling_baseline"
RB_NPZ = RB_DIR / "prediction_protection.npz"
BIAS_DIR = ROOT / "results/q3_bias_correction_diagnostic"
BIAS_NPZ = BIAS_DIR / "bias_forecast_archive.npz"
Q2_DIR = ROOT / "results/q2_time_mapping"

BASE = pd.Timestamp("2025-01-01")
NATURAL_DAYS = 365
TEMPLATE_SLOTS = 144
TARGETS = 145
DT = 1.0 / 6.0
ALPHA = 0.80                     # frozen protection level (reference, and L80/C80)
ALPHA_LOW = 0.75                 # this round's single lower-protection candidate
WINDOW = 28
MIN_SAMPLES = 7
EVAL_FIRST, EVAL_LAST = 31, 364  # 2025-02-01 .. 2025-12-31
UPDATE_HOURS = (0, 6, 12, 18)
FIRST_TARGET = {0: 0, 1: 36, 2: 72, 3: 108}
SELECT_TOL_YUAN = 1e-4
ENERGY_TOL_KWH = 1e-6
CASH_TOL_YUAN = 1e-4
PROT_TOL = 1e-8                  # prediction / protection absolute tolerance, own units

L80_REFERENCE_YUAN = 13492312.266087154
PUBLIC_INITIAL_KWH = 6075.795025925926
PUBLIC_CARRY_KWH = 0.0
PUBLIC_CARRY_VERSION = "public_January"
PUBLIC_CARRY_OWNER = "2025-01-31"

STRATEGIES = ("L80", "L75", "C80", "C75")
NEW_RUNS = ("L75", "C80", "C75")
PREDICTOR = {"L80": "linear", "L75": "linear", "C80": "corrected", "C75": "corrected"}
NET_KEY = {"linear": "c0", "corrected": "c1"}
ALPHA_OF = {"L80": ALPHA, "L75": ALPHA_LOW, "C80": ALPHA, "C75": ALPHA_LOW}
SELECTED_DATES = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")
SPOT_DAY = 171                   # 2025-06-21
SPOT_TARGETS = ((0, 0), (0, 144), (1, 36))
CONTRASTS = (
    ("bias_80", "C80", "L80"),
    ("bias_75", "C75", "L75"),
    ("quantile_L", "L75", "L80"),
    ("quantile_C", "C75", "C80"),
    ("combined_75_vs_80", "C75", "L80"),
)
INTERACTION = ("C75", "C80", "L75", "L80")

REVISION_WINDOWS = {1: (36, 109), 2: (72, 73), 3: (108, 37)}   # v -> (first h, count)
FINAL_PREFIX = 36                # slots of a revision window no later publication can overwrite

STRATEGY_COLS = [
    "strategy_id", "interval_start", "interval_end", "owner_date", "effective_version",
    "price_yuan_kWh", "actual_load_kW", "actual_pv_kW", "net_kWh",
    "q0_kWh", "q_eff_kWh", "charge_kWh", "discharge_kWh", "emergency_kWh", "unused_kWh",
    "state_start_kWh", "state_end_kWh",
    "ordinary_cost_yuan", "adjustment_cost_yuan", "emergency_cost_yuan", "total_cost_yuan",
]

# Not computed artifacts: the report layer rewrites figure_integrity.json, run_manifest.json
# certifies itself only after the fact, and run.log is appended by the shell while the run is in
# progress. Hashing any of them here would create a guaranteed mismatch.
NON_COMPUTED_ARTIFACTS = frozenset({"run_manifest.json", "figure_integrity.json", "run.log"})

PARAMETERS = dict(
    experiment="Q3 bias-correction x quantile-level four-group cost comparison",
    specification="reports/问题三/问题三_偏差校正与分位水平四组费用实验方案.md",
    groups=dict(L80="frozen Linear + q80, read-only reuse of the saved B2 ledger",
                L75="frozen Linear + q75, re-extracted from Linear residuals",
                C80="frozen Bias28 corrected forecast + its own q80, reused from the accepted "
                    "bias-correction diagnostic",
                C75="frozen Bias28 corrected forecast + q75, re-extracted from corrected "
                    "residuals"),
    time_version="start_time_v1",
    interval_rule="label 00:10 covers 00:10-00:20; label 0:00+1 covers the next day 00:00-00:10",
    natural_day_rule="natural day d = previous source row column 143 ++ current row columns 0..142",
    template_rule="k-day 0:00 publishes 144 slots covering [k 00:10, (k+1) 00:10); the current "
                  "midnight 00:00-00:10 executes the previous publication's carry commitment",
    targets="h=0 current 00:00-00:10; h=1..143 current 00:10..23:50; h=144 next 00:00-00:10; "
            "template slot j = h-1",
    protection=dict(window_days=WINDOW, alphas=[ALPHA_LOW, ALPHA], min_samples=MIN_SAMPLES,
                    key="publication hour, target day-offset, target slot clock",
                    rule="ascending order statistic at position ceil(alpha*m); zero when m<7; "
                         "negative corrections retained",
                    expected_positions={"q75": {"m=27": 21, "m=28": 21},
                                        "q80": {"m=27": 22, "m=28": 23}},
                    note="q75 is re-extracted from each predictor's own historical residuals; it is "
                         "not q80 times a coefficient, and C75 never uses Linear residuals"),
    schedule="four-node rolling: 0:00 plans 144 slots; 6/12/18 revise the remaining 109/73/37",
    scoring="0:00 minimises sum p*q0; intraday minimises sum p*q + 0.5 p|q - q0|; a candidate must "
            "lower the predicted total (ordinary + adjustment + 5x predicted emergency) by more "
            "than 1e-4 yuan to be accepted, ties keep the old plan",
    emergency_rule="5*p*e on the delivered-interval price",
    settlement="final effective version against the 0:00 original plan, once only",
    ledgers=dict(A="natural days 2025-02-01..2025-12-31, 334 days, 48096 segments",
                 B="template publications 2025-02-01..2025-12-31, 48096 segments",
                 bridge="C_B - C_A = cost(2026-01-01 00:00-00:10) - cost(2025-02-01 00:00-00:10)"),
    solver="SciPy milp/HiGHS, relative MIP gap 1e-9, time limit 120 s, free terminal",
    reuse=dict(kernel="code/q3_rolling_baseline_experiment.py imported as a library; its main "
                      "entry is NOT called",
               L80="results/q3_rolling_baseline/B2_dispatch.csv and summaries, read only",
               C80="results/q3_bias_correction_diagnostic/bias_forecast_archive.npz::protected_c1",
               load="results/q2_time_mapping/archive_float.npz::issued_load"),
    validation="three classes only: quantile-input checks (no MILP), one adaptation control on "
               "2025-06-21 (2 MILPs), one read-only reconciliation of the saved ledgers",
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
    """``max |value|`` that treats any NaN/inf as a failure instead of silently skipping it.

    ``np.max`` and ``pandas.min/max`` propagate or skip NaN, so a comparison built on them can
    pass without ever looking at a poisoned cell. Every checked quantity goes through here.
    """
    arr = np.asarray(values, dtype=float)
    bad = ~np.isfinite(arr)
    if bad.any():
        raise AssertionError(f"{label}: {int(bad.sum())} non-finite value(s) of {arr.size}")
    return float(np.abs(arr).max())


def known_max(values, label):
    """``max value`` (signed) that treats any NaN/inf as a failure instead of skipping it."""
    arr = np.asarray(values, dtype=float)
    bad = ~np.isfinite(arr)
    if bad.any():
        raise AssertionError(f"{label}: {int(bad.sum())} non-finite value(s) of {arr.size}")
    return float(arr.max())


def comparable_max_abs(left, right, label):
    """``max |left-right|`` over the cells where both are defined; NaN masks must agree exactly.

    Used where NaN is legitimate (targets outside a publication's reach): a mismatched NaN mask is
    itself an alignment failure, and the two sides must have the same finite cell count.
    """
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise AssertionError(f"{label}: shape mismatch {left.shape} vs {right.shape}")
    if not np.array_equal(np.isnan(left), np.isnan(right)):
        raise AssertionError(f"{label}: NaN masks differ "
                             f"({int(np.isnan(left).sum())} vs {int(np.isnan(right).sum())})")
    defined = np.isfinite(left)
    value = float(np.abs(left[defined] - right[defined]).max()) if defined.any() else 0.0
    return value, int(defined.sum())


_MODULES: dict = {}


def rb():
    """Import the accepted rolling-baseline kernel as a read-only library."""
    if "rb" not in _MODULES:
        spec = importlib.util.spec_from_file_location("q3_rolling_baseline_kernel", RB_KERNEL)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # the frozen conventions must be the same ones this experiment registers
        assert module.EVAL_FIRST == EVAL_FIRST and module.EVAL_LAST == EVAL_LAST
        assert module.WINDOW == WINDOW and module.MIN_SAMPLES == MIN_SAMPLES
        assert module.ALPHA == ALPHA and module.UPDATE_HOURS == UPDATE_HOURS
        assert module.FIRST_TARGET == FIRST_TARGET and module.ENERGY_TOL_KWH == ENERGY_TOL_KWH
        assert module.SELECT_TOL_YUAN == SELECT_TOL_YUAN
        _MODULES["rb"] = module
    return _MODULES["rb"]


# ======================================================================================
# §2  输入：分位档案、L80 只读复用、公共初态
# ======================================================================================
def availability_mask():
    """``ok[s, v, h]``: lag ``s`` of target ``h`` is realised by publication (k, hour r_v).

    ``interval_end(j, h) <= k day + r_v hours  <=>  h + 1 - 6 r_v <= 144 s``; only (s, v, h)
    matter, so the mask needs no k axis. Must equal the mask stored in the bias archive.
    """
    h_grid = np.arange(TARGETS)
    ok = np.zeros((WINDOW + 1, len(UPDATE_HOURS), TARGETS), dtype=bool)
    for s in range(1, WINDOW + 1):
        for v, hour in enumerate(UPDATE_HOURS):
            ok[s, v] = (h_grid + 1 - 6 * hour) <= TEMPLATE_SLOTS * s
    return ok


def grouped_order_statistics(values, ok, alphas):
    """Grouped order statistic over lags 1..WINDOW in one pass, for several alpha levels.

    Identical construction to the accepted bias diagnostic: lag ``s`` adds the value at source row
    ``k-s`` to target row ``k``, and the availability mask is applied to the source block. NaN sorts
    last, so a missing cell can never be picked as a quantile, and ``m < MIN_SAMPLES`` returns zero.
    """
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


def read_bias_archive():
    """Load the frozen bias-correction archive and cross-check it against the frozen Linear run."""
    with np.load(BIAS_NPZ) as archive:
        data = {key: archive[key] for key in archive.files}
    with np.load(RB_NPZ) as baseline:
        rb_net, rb_rho = baseline["net_forecast"], baseline["rho"]
        rb_counts, rb_protected = baseline["counts"], baseline["protected"]

    shape = (NATURAL_DAYS, len(UPDATE_HOURS), TARGETS)
    for key in ("pv_linear", "pv_corrected", "net_c0", "net_c1", "rho_c0", "rho_c1",
                "protected_c0", "protected_c1", "m_net_samples_c0", "m_net_samples_c1"):
        assert data[key].shape == shape, (key, data[key].shape)
    for key in ("truth_pv", "truth_load", "issued_load"):
        assert data[key].shape == (NATURAL_DAYS, TARGETS), (key, data[key].shape)
    assert np.isfinite(data["truth_pv"][1:, :]).all() and np.isfinite(data["truth_load"][1:, :]).all()

    net_value, net_cells = comparable_max_abs(data["net_c0"], rb_net, "net_c0 vs frozen net")
    rho_value, _ = comparable_max_abs(data["rho_c0"], rb_rho, "rho_c0 vs frozen rho")
    prot_value, _ = comparable_max_abs(data["protected_c0"], rb_protected,
                                       "protected_c0 vs frozen protected")
    count_delta = int(np.abs(data["m_net_samples_c0"] - rb_counts).max())
    alignment = dict(net_c0_vs_frozen_net_max_abs_kWh=net_value,
                     net_c0_compared_cells=net_cells,
                     rho_c0_vs_frozen_rho_max_abs_kWh=rho_value,
                     protected_c0_vs_frozen_protected_max_abs_kWh=prot_value,
                     counts_c0_vs_frozen_max_abs=count_delta,
                     availability_shape=list(data["availability"].shape))
    if max(net_value, rho_value, prot_value) >= PROT_TOL or count_delta != 0:
        raise AssertionError("the Linear archive does not match the frozen rolling baseline: "
                             f"{alignment}")
    return data, alignment


def load_l80(truth_net):
    """Read the saved B2 ledger and rebuild the unified 48097-row frame; nothing is solved."""
    frame = pd.read_csv(RB_DIR / "B2_dispatch.csv", parse_dates=["interval_start", "interval_end"])
    frame = frame.rename(columns={"group": "strategy_id"})
    assert len(frame) == 48097, len(frame)
    reference = pd.read_csv(RB_DIR / "summary.csv").set_index("group").loc["B2"]
    initial, carry = float(frame.state_start_kWh.iloc[0]), float(frame.q_eff_kWh.iloc[0])
    assert abs(initial - PUBLIC_INITIAL_KWH) < ENERGY_TOL_KWH, initial
    assert abs(carry - PUBLIC_CARRY_KWH) < ENERGY_TOL_KWH, carry
    head = frame.iloc[0]
    assert str(head["effective_version"]) == PUBLIC_CARRY_VERSION, head["effective_version"]
    assert str(head["owner_date"]) == PUBLIC_CARRY_OWNER, head["owner_date"]
    meta = dict(initial_kWh=initial, carry_kWh=carry, carry_version=PUBLIC_CARRY_VERSION,
                carry_owner=PUBLIC_CARRY_OWNER, reference_total_yuan=float(reference.natural_total_yuan),
                reference_row=reference.to_dict())
    return frame, meta


# ======================================================================================
# §3  统一核对与落盘
# ======================================================================================
def truth_reproduction(frame, truth_net):
    """Compare the stored actual net demand with the shared truth arrays.

    Returns ``(max_abs_residual_kWh, compared_cells, nonfinite_cells)``; the caller must require
    both a small residual and zero non-finite cells, so a missing truth cell cannot pass the check
    by turning the maximum into NaN.
    """
    starts = pd.to_datetime(frame.interval_start)
    is_midnight = rb().midnight_mask(starts)
    k_pub = np.where(is_midnight,
                     ((starts.dt.normalize() - BASE).dt.days).to_numpy() - 1,
                     ((starts.dt.normalize() - BASE).dt.days).to_numpy())
    h_pub = np.where(is_midnight, TEMPLATE_SLOTS,
                     ((starts - starts.dt.normalize()).dt.total_seconds() / 600).astype(int).to_numpy())
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
    assert frame.state_start_kWh.iloc[0] > 0
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
                  emergency_kWh=("emergency_kWh", "sum"),
                  state_start_kWh=("state_start_kWh", "first"),
                  state_end_kWh=("state_end_kWh", "last")))
    daily.insert(0, "strategy_id", sid)

    summary = dict(
        strategy_id=sid, predictor=PREDICTOR[sid], alpha=ALPHA_OF[sid],
        source=("read-only reuse of results/q3_rolling_baseline/B2_dispatch.csv"
                if sid == "L80" else "new 334-day rolling run"),
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
    """Per-strategy solve bookkeeping; candidate revisions and accepted revisions stay separate."""
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
def check_quantile_inputs(data, rho75, rho80_check, counts_check):
    """Class 1: quantile-input checks, no MILP."""
    out = {}
    # ---- 4.2(1a) small ascending arrays: fallback and the two positions
    examples = {}
    for m in (6, 7, 21, 27, 28):
        arr = np.arange(1, m + 1, dtype=float)
        arr[0] = -5.0                       # a negative residual must survive into the quantile
        ordered = np.sort(arr)
        row = {}
        for label, alpha in (("q75", ALPHA_LOW), ("q80", ALPHA)):
            if m < MIN_SAMPLES:
                row[label] = dict(position=0, value=0.0, fallback=True)
            else:
                pos = math.ceil(alpha * m)
                row[label] = dict(position=pos, value=float(ordered[pos - 1]), fallback=False)
        examples[f"m={m}"] = row
    assert examples["m=6"]["q75"]["fallback"] and examples["m=6"]["q80"]["fallback"]
    assert examples["m=7"]["q80"]["value"] == 6.0, examples["m=7"]
    assert examples["m=27"]["q75"]["position"] == 21
    assert examples["m=27"]["q80"]["position"] == 22
    assert examples["m=28"]["q75"]["position"] == 21
    assert examples["m=28"]["q80"]["position"] == 23
    out["order_statistic_examples"] = examples

    # ---- 4.2(1b) the two q80 must reproduce the frozen archive cell by cell
    alignment = {}
    for key in ("c0", "c1"):
        alignment[f"{key}_rho_max_abs_kWh"] = known_max_abs(
            rho80_check[key]["rho"] - data[f"rho_{key}"], f"recomputed q80 vs archive ({key})")
        alignment[f"{key}_counts_max_abs"] = int(
            np.abs(counts_check[key] - data[f"m_net_samples_{key}"]).max())
    out["q80_alignment"] = alignment
    if max(alignment.values()) >= PROT_TOL:
        raise AssertionError(f"the recomputed q80 does not reproduce the frozen archive: {alignment}")

    # ---- 4.2(1c) 2025-06-21 spot check: legal sample set, sample count, order position
    day = (BASE + timedelta(days=SPOT_DAY)).strftime("%Y-%m-%d")
    assert day == "2025-06-21", day
    ok = data["availability"]
    spot = []
    for v, h in SPOT_TARGETS:
        sources = [SPOT_DAY - s for s in range(1, WINDOW + 1)
                   if SPOT_DAY - s >= 0 and ok[s, v, h]]
        spot.append(dict(
            day=day, version=int(v), publication_hour=int(UPDATE_HOURS[v]), target_h=int(h),
            m=len(sources),
            legal_lags=[int(SPOT_DAY - j) for j in sources],
            legal_dates=[(BASE + timedelta(days=j)).strftime("%Y-%m-%d") for j in sources],
            q80_archive_value_kWh=float(data["rho_c0"][SPOT_DAY, v, h]),
            q80_recomputed_value_kWh=float(rho80_check["c0"]["rho"][SPOT_DAY, v, h]),
            q80_position=int(rho80_check["c0"]["position"][SPOT_DAY, v, h]),
            q75_linear_value_kWh=float(rho75["c0"]["rho"][SPOT_DAY, v, h]),
            q75_linear_position=int(rho75["c0"]["position"][SPOT_DAY, v, h]),
            q75_corrected_value_kWh=float(rho75["c1"]["rho"][SPOT_DAY, v, h]),
            q75_corrected_position=int(rho75["c1"]["position"][SPOT_DAY, v, h])))
    out["spot_check"] = spot
    expected = {(0, 0): (28, 21, 23), (0, 144): (27, 21, 22), (1, 36): (28, 21, 23)}
    for row in spot:
        key = (row["version"], row["target_h"])
        m, pos75, pos80 = expected[key]
        for field in ("q80_archive_value_kWh", "q80_recomputed_value_kWh",
                      "q75_linear_value_kWh", "q75_corrected_value_kWh"):
            if not math.isfinite(row[field]):
                raise AssertionError(f"spot check produced a non-finite quantile: {row}")
        assert row["m"] == m, row
        assert row["q75_linear_position"] == pos75, row
        assert row["q80_position"] == pos80, row
        assert row["q80_archive_value_kWh"] == row["q80_recomputed_value_kWh"], row
    out["spot_check_expected"] = {f"v={k[0]},h={k[1]}": dict(m=v[0], q75_position=v[1],
                                                            q80_position=v[2])
                                  for k, v in expected.items()}

    # ---- 4.2(1d) rho75 <= rho80 under identical samples; NOT extrapolated to cost monotonicity
    guard = {}
    for key in ("c0", "c1"):
        mask = counts_check[key] >= MIN_SAMPLES
        guard[f"{key}_cells_m_ge_{MIN_SAMPLES}"] = int(mask.sum())
        guard[f"{key}_max_q75_minus_q80_kWh"] = known_max(
            rho75[key]["rho"][mask] - rho80_check[key]["rho"][mask],
            f"signed q75-q80 spread ({key})")
        guard[f"{key}_cells_q75_strictly_below"] = int(
            (rho75[key]["rho"][mask] < rho80_check[key]["rho"][mask] - 1e-12).sum())
    out["monotonicity"] = guard
    if max(guard["c0_max_q75_minus_q80_kWh"], guard["c1_max_q75_minus_q80_kWh"]) > 1e-9:
        raise AssertionError(f"q75 exceeds q80 under identical samples: {guard}")
    out["monotonicity_note"] = ("q75 <= q80 holds cell by cell under identical samples; this does "
                                "not imply monotone cost, emergency or unused energy")

    # ---- predictor isolation: C75 must come from corrected residuals, never from Linear
    isolation = dict(
        max_abs_rho75_corrected_minus_linear_kWh=known_max_abs(
            rho75["c1"]["rho"] - rho75["c0"]["rho"], "corrected q75 minus Linear q75"),
        cells_rho75_corrected_differs_from_linear=int(
            (np.abs(rho75["c1"]["rho"] - rho75["c0"]["rho"]) > 1e-12).sum()),
        max_abs_rho75_corrected_minus_rho80_corrected_kWh=known_max_abs(
            rho75["c1"]["rho"] - rho80_check["c1"]["rho"], "corrected q75 minus corrected q80"),
        cells_rho75_corrected_differs_from_rho80=int(
            (np.abs(rho75["c1"]["rho"] - rho80_check["c1"]["rho"]) > 1e-12).sum()))
    out["predictor_isolation"] = isolation
    if isolation["cells_rho75_corrected_differs_from_linear"] == 0:
        raise AssertionError("the corrected q75 is identical to the Linear q75: sample sets wrong")
    if isolation["cells_rho75_corrected_differs_from_rho80"] == 0:
        raise AssertionError("the corrected q75 is identical to the corrected q80")
    out["scope"] = ("no MILP; only the newly extracted quantiles and their sample sets. The two q80 "
                    "are re-derived and compared cell by cell to the frozen diagnostic archive")
    return out


def check_adaptation(price_src, protected_l80, l80_frame, plan_versions, decisions):
    """Class 2: adaptation control, at most 2 MILPs.

    Reproduces the saved B2 (L80) decisions on 2025-06-21 through the same code path the three new
    runs use: one 0:00 plan and one 06:00 revision. The 06:00 candidate uses the saved update-instant
    state, the saved ``q0`` segment and the ``previous`` column of the saved plan archive (never the
    final ``q_eff``). The gate is objective equality; a different plan vector at the same objective
    is only another equal-cost vertex of the same MILP.
    """
    kernel = rb()
    k, day = SPOT_DAY, (BASE + timedelta(days=SPOT_DAY)).strftime("%Y-%m-%d")
    out = dict(day=day, k=int(k), extra_milp_solves=2)
    rows = l80_frame[l80_frame.interval_start.dt.strftime("%Y-%m-%d") == day].sort_values(
        "interval_start")
    midnight = rows[kernel.midnight_mask(rows.interval_start)].iloc[0]
    state_before, carry = float(midnight.state_start_kWh), float(midnight.q_eff_kWh)

    # ---- 0:00 plan (1st MILP)
    _, _, _, _, estimated = kernel.feedback_step(carry - float(protected_l80[k, 0, 0]), state_before)
    q_plan, _, _, _, _, summary = kernel.solve_interval(
        protected_l80[k, 0, 1:TARGETS], price_src, estimated, context=f"adaptation {day} 0:00")
    # h=144 sits at the next day's 00:00, so the plan window must be queried as a 24-hour range
    saved_plan = l80_frame[(l80_frame.interval_start > pd.Timestamp(day))
                           & (l80_frame.interval_start
                              <= pd.Timestamp(day) + pd.Timedelta(hours=24))
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

    # ---- 06:00 revision (2nd MILP)
    v, h0 = 1, FIRST_TARGET[1]
    state_0600 = float(rows.state_end_kWh.iloc[h0 - 1])
    record = plan_versions[(plan_versions.decision == "revision")
                           & (plan_versions.published_at == f"{day} {UPDATE_HOURS[v]:02d}:00")].copy()
    record["interval_start"] = pd.to_datetime(record.interval_start)
    record = record.sort_values("interval_start")
    assert len(record) == TARGETS - h0, len(record)
    assert record.interval_start.iloc[0] == pd.Timestamp(day) + pd.Timedelta(hours=6)
    assert record.interval_start.iloc[-1] == pd.Timestamp(day) + pd.Timedelta(hours=24)
    assert (record.previous_version == f"{day} 00:00").all()
    q0_seg = record.q0_kWh.to_numpy()
    old_seg = record.previous_kWh.to_numpy()
    saved_candidate = record.candidate_kWh.to_numpy()
    saved_accepted = bool(record.accepted.iloc[0])
    price_seg, remaining = price_src[h0 - 1:], protected_l80[k, v, h0:TARGETS]
    candidate, _, _, _, _, cand_summary = kernel.solve_interval(
        remaining, price_seg, state_0600, q0_seg, context=f"adaptation {day} 06:00")
    old_score = float(sum(kernel.score_components(old_seg, q0_seg, price_seg, remaining, state_0600)))
    new_score = float(sum(kernel.score_components(candidate, q0_seg, price_seg, remaining,
                                                  state_0600)))
    accepted = bool(new_score < old_score - SELECT_TOL_YUAN)
    saved_decision = decisions[(decisions.date == day) & (decisions.update_hour == UPDATE_HOURS[v])]
    assert len(saved_decision) == 1
    saved = saved_decision.iloc[0]
    out["six"] = dict(
        remaining_intervals=int(len(q0_seg)), saved_accepted=saved_accepted,
        replayed_accepted=accepted, accept_matches_saved=bool(accepted == saved_accepted),
        candidate_max_delta_kWh=known_max_abs(candidate - saved_candidate,
                                              "adaptation 06:00 candidate delta"),
        revision_objective_delta_yuan=float(abs(cand_summary["objective_yuan"]
                                                - (float(saved.new_ordinary_yuan)
                                                   + float(saved.new_adjustment_yuan)))),
        old_score_yuan=old_score, old_score_saved_yuan=float(saved.old_predicted_total_yuan),
        new_score_yuan=new_score, new_score_saved_yuan=float(saved.new_predicted_total_yuan),
        score_delta_yuan=float(abs(new_score - float(saved.new_predicted_total_yuan))),
        max_feasibility_violation_kWh=float(cand_summary["max_violation_kWh"]),
        note="the 06:00 old plan comes from plan_versions.previous, never the final q_eff; this "
             "checks function reuse and parameter passing, not an independent audit")
    if out["six"]["revision_objective_delta_yuan"] >= SELECT_TOL_YUAN:
        raise AssertionError(f"adaptation 06:00 objective mismatch: {out['six']}")
    if out["six"]["score_delta_yuan"] >= SELECT_TOL_YUAN:
        raise AssertionError(f"adaptation 06:00 predicted-score mismatch: {out['six']}")
    if not out["six"]["accept_matches_saved"]:
        raise AssertionError(f"adaptation 06:00 accept decision changed: {out['six']}")
    out["scope"] = ("one day (2025-06-21), one 0:00 plan and one 06:00 revision on the saved L80 "
                    "inputs; 2 MILPs, inside the task book's budget")
    return out


def check_saved_results(strategy_paths, price_src, truth_net, decisions_by_strategy):
    """Class 3: one read-only reconciliation of the persisted artifacts, no new solve."""
    out = dict(note="read back from the persisted CSV artifacts, not from the in-memory frames")
    for sid in STRATEGIES:
        dispatch = pd.read_csv(strategy_paths[sid]["dispatch"],
                               parse_dates=["interval_start", "interval_end"])
        solver_log = pd.read_csv(strategy_paths[sid]["solver_log"])
        decisions = decisions_by_strategy[sid]
        plans = pd.read_csv(strategy_paths[sid]["plan_versions"],
                            parse_dates=["interval_start"])
        check = dict(rows=int(len(dispatch)))
        assert len(dispatch) == 48097, len(dispatch)
        assert dispatch.interval_start.diff().dropna().eq(pd.Timedelta(minutes=10)).all()
        numeric = ["q0_kWh", "q_eff_kWh", "charge_kWh", "discharge_kWh", "emergency_kWh",
                   "unused_kWh", "state_start_kWh", "state_end_kWh", "price_yuan_kWh",
                   "net_kWh", "ordinary_cost_yuan", "adjustment_cost_yuan",
                   "emergency_cost_yuan", "total_cost_yuan"]
        check["nonfinite_dispatch_cells"] = int(
            (~np.isfinite(dispatch[numeric].to_numpy(dtype=float))).sum())
        if check["nonfinite_dispatch_cells"]:
            raise AssertionError(f"{sid}: {check['nonfinite_dispatch_cells']} non-finite ledger "
                                 "cells; pandas min/max would have skipped them silently")
        for col, low, high in (("q_eff_kWh", 0.0, None), ("charge_kWh", 0.0, 5000 / 6),
                               ("discharge_kWh", 0.0, 5000 / 6), ("emergency_kWh", 0.0, None),
                               ("unused_kWh", 0.0, None),
                               ("state_start_kWh", 1200.0, 10800.0),
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
        for key in ("bus_balance_max_abs_kWh", "battery_recurrence_max_abs_kWh",
                    "state_continuity_max_abs_kWh", "truth_reproduction_max_abs_kWh",
                    "simultaneous_charge_discharge_kWh"):
            if check[key] >= ENERGY_TOL_KWH:
                raise AssertionError(f"{sid} {key} = {check[key]:.3e}")

        price = dispatch.price_yuan_kWh
        check["ordinary_recompute_max_abs_yuan"] = known_max_abs(
            price * dispatch.q_eff_kWh - dispatch.ordinary_cost_yuan, f"{sid} ordinary recompute")
        check["adjustment_recompute_max_abs_yuan"] = known_max_abs(
            0.5 * price * (dispatch.q_eff_kWh - dispatch.q0_kWh).abs()
            - dispatch.adjustment_cost_yuan, f"{sid} adjustment recompute")
        check["emergency_recompute_max_abs_yuan"] = known_max_abs(
            5.0 * price * dispatch.emergency_kWh - dispatch.emergency_cost_yuan,
            f"{sid} emergency recompute")
        for key in ("ordinary_recompute_max_abs_yuan", "adjustment_recompute_max_abs_yuan",
                    "emergency_recompute_max_abs_yuan"):
            if check[key] >= CASH_TOL_YUAN:
                raise AssertionError(f"{sid} {key} = {check[key]:.3e}")

        natural, template = dispatch.iloc[:-1], dispatch.iloc[1:]
        check["bridge_residual_yuan"] = float(
            (template.total_cost_yuan.sum() - natural.total_cost_yuan.sum())
            - (float(dispatch.total_cost_yuan.iloc[-1])
               - float(dispatch.total_cost_yuan.iloc[0])))
        if abs(check["bridge_residual_yuan"]) >= CASH_TOL_YUAN:
            raise AssertionError(f"{sid} bridge residual {check['bridge_residual_yuan']:.3e}")

        check["solves_total"] = int(len(solver_log))
        check["solves_plan"] = int((solver_log.kind == "plan").sum())
        check["solves_revision"] = int((solver_log.kind == "revision").sum())
        assert check["solves_total"] == 1336 and check["solves_plan"] == 334
        assert check["solves_revision"] == 1002
        check["max_solver_gap"] = float(solver_log.mip_gap.max())
        check["max_solver_feasibility_violation_kWh"] = float(solver_log.max_violation_kWh.max())
        check["solver_seconds_total"] = float(solver_log.seconds.sum())
        check["revision_decisions"] = int(len(decisions))
        assert len(decisions) == 1002, len(decisions)
        for hour in UPDATE_HOURS[1:]:
            assert int((decisions.update_hour == hour).sum()) == 334, hour
        expected_accept = (decisions.old_predicted_total_yuan - decisions.new_predicted_total_yuan
                           - SELECT_TOL_YUAN) > 0
        check["accept_inequality_mismatches"] = int((decisions.accepted != expected_accept).sum())
        if check["accept_inequality_mismatches"]:
            raise AssertionError(f"{sid} accept inequality mismatches "
                                 f"{check['accept_inequality_mismatches']}")
        check["revisions_accepted"] = int(decisions.accepted.sum())

        # an accepted candidate must actually execute on the part of its window that no later
        # publication can overwrite (the first 36 slots of every revision window)
        q_eff_series = dispatch.set_index("interval_start").q_eff_kWh
        mismatch, compared = 0, 0
        for v, (first_h, count) in REVISION_WINDOWS.items():
            for date in decisions[decisions.update_hour == UPDATE_HOURS[v]].date.unique():
                record = plans[(plans.decision == "revision")
                               & (plans.published_at == f"{date} {UPDATE_HOURS[v]:02d}:00")]
                record = record.sort_values("interval_start")
                assert len(record) == count, (sid, date, len(record))
                row = decisions[(decisions.date == date)
                                & (decisions.update_hour == UPDATE_HOURS[v])]
                assert len(row) == 1
                take_candidate = bool(row.accepted.iloc[0])
                expected = (record.candidate_kWh.to_numpy()[:FINAL_PREFIX] if take_candidate
                            else record.previous_kWh.to_numpy()[:FINAL_PREFIX])
                starts = [pd.Timestamp(date) + pd.Timedelta(minutes=10 * h)
                          for h in range(first_h, first_h + FINAL_PREFIX)]
                actual = q_eff_series.reindex(starts).to_numpy()
                if not np.isfinite(actual).all():
                    raise AssertionError(f"{sid} {date} revision window is not contiguous")
                mismatch += int((np.abs(actual - expected) >= ENERGY_TOL_KWH).sum())
                compared += len(starts)
        check["accepted_candidate_slots_compared"] = int(compared)
        check["accepted_candidate_execution_mismatches"] = int(mismatch)
        if mismatch:
            raise AssertionError(f"{sid} accepted candidate is not what executes: {mismatch}")
        out[sid] = check
    out["scope"] = ("both ledgers, physical laws, the three cash components recomputed from the "
                    "saved columns, the head-tail bridge, the 1336-solve / 1002-decision "
                    "bookkeeping, the accept inequality and the executed accepted candidates for "
                    "all four strategies; no new solve")
    return out


# ======================================================================================
# §5  汇总与配对差额
# ======================================================================================
def build_wide(long_frame, value_col, key):
    """Wide per-strategy table plus the five registered differences."""
    wide = long_frame.pivot(index=key, columns="strategy_id", values=value_col)
    for name, treatment, baseline in CONTRASTS:
        wide[f"d_{name}"] = wide[treatment] - wide[baseline]
    return wide.reset_index()


def build_contrasts(summaries, nu):
    rows = []
    for name, treatment, baseline in CONTRASTS:
        a, b = summaries[treatment], summaries[baseline]
        delta = a["natural_total_yuan"] - b["natural_total_yuan"]
        rows.append(dict(
            contrast=name, treatment=treatment, baseline=baseline,
            treatment_total_yuan=a["natural_total_yuan"],
            baseline_total_yuan=b["natural_total_yuan"], delta_yuan=delta,
            delta_pct=100.0 * delta / b["natural_total_yuan"],
            delta_ordinary_yuan=a["ordinary_cost_yuan"] - b["ordinary_cost_yuan"],
            delta_adjustment_yuan=a["adjustment_cost_yuan"] - b["adjustment_cost_yuan"],
            delta_emergency_yuan=a["emergency_cost_yuan"] - b["emergency_cost_yuan"],
            delta_emergency_kWh=a["emergency_kWh"] - b["emergency_kWh"],
            delta_unused_kWh=a["unused_kWh"] - b["unused_kWh"],
            delta_charge_kWh=a["charge_kWh"] - b["charge_kWh"],
            delta_discharge_kWh=a["discharge_kWh"] - b["discharge_kWh"],
            delta_final_natural_state_kWh=(a["final_natural_state_kWh"]
                                           - b["final_natural_state_kWh"]),
            delta_final_template_state_kWh=(a["final_template_state_kWh"]
                                            - b["final_template_state_kWh"]),
            delta_inventory_adjusted_yuan=(
                a["natural_total_yuan"] - nu * a["final_natural_state_kWh"]
                - (b["natural_total_yuan"] - nu * b["final_natural_state_kWh"]))))
    t1, b1, t2, b2 = INTERACTION
    interaction = ((summaries[t1]["natural_total_yuan"] - summaries[b1]["natural_total_yuan"])
                   - (summaries[t2]["natural_total_yuan"] - summaries[b2]["natural_total_yuan"]))
    rows.append(dict(contrast="interaction_bias_x_quantile", treatment=f"{t1}-{b1}",
                     baseline=f"{t2}-{b2}",
                     treatment_total_yuan=(summaries[t1]["natural_total_yuan"]
                                           - summaries[b1]["natural_total_yuan"]),
                     baseline_total_yuan=(summaries[t2]["natural_total_yuan"]
                                          - summaries[b2]["natural_total_yuan"]),
                     delta_yuan=interaction, delta_pct=np.nan))
    return pd.DataFrame(rows), float(interaction)


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

    inputs = ["附件/附件1.xlsx", "附件/附件2.xlsx",
              "results/q3_bias_correction_diagnostic/bias_forecast_archive.npz",
              "results/q3_bias_correction_diagnostic/registration.json",
              "results/q3_bias_correction_diagnostic/run_manifest.json",
              "results/q3_rolling_baseline/prediction_protection.npz",
              "results/q3_rolling_baseline/B2_dispatch.csv",
              "results/q3_rolling_baseline/B2_plan_versions.csv",
              "results/q3_rolling_baseline/B2_revision_decisions.csv",
              "results/q3_rolling_baseline/B2_solver_log.csv",
              "results/q3_rolling_baseline/summary.csv",
              "results/q3_rolling_baseline/registration.json",
              "results/q3_rolling_baseline/run_manifest.json",
              "results/q2_time_mapping/N_free/natural_dispatch.csv",
              "results/q2_time_mapping/N_free/template_plan.csv",
              "code/q3_rolling_baseline_experiment.py",
              "reports/问题三/问题三_偏差校正与分位水平四组费用实验方案.md"]
    for rel in inputs:
        assert (ROOT / rel).exists(), rel
    dependency = dict(L80_REFERENCE_YUAN=L80_REFERENCE_YUAN, PUBLIC_INITIAL_KWH=PUBLIC_INITIAL_KWH,
                      PUBLIC_CARRY_KWH=PUBLIC_CARRY_KWH, kernel_sha256=digest(RB_KERNEL),
                      report_layer=dict(figures=str(FIG.relative_to(ROOT)),
                                        report=str(REPORT_MD.relative_to(ROOT))),
                      budgets=dict(new_training=0, public_january_reruns=0,
                                   l80_full_year_reruns=0, extra_validation_milp_limit=2,
                                   planned_formal_milp=4008))
    signature = register(PARAMETERS, inputs, dependency, args.amend_reason)
    print(f"registered signature={signature[:16]}", flush=True)
    if args.mode == "register":
        return

    kernel = rb()
    bias_registration = load_json(BIAS_DIR / "registration.json")
    bias_manifest = load_json(BIAS_DIR / "run_manifest.json")

    # ---- inputs and shared truth ----
    tick = time.perf_counter()
    attachments = kernel.read_attachments()
    price_src = attachments["price_src"]
    truth_load, truth_pv = kernel.build_truth(attachments["load_src"], attachments["pv_src"])
    truth_net = (truth_load - truth_pv) * DT
    data, alignment = read_bias_archive()
    if not (np.allclose(truth_pv[1:], data["truth_pv"][1:], rtol=0, atol=ENERGY_TOL_KWH)
            and np.allclose(truth_load[1:], data["truth_load"][1:], rtol=0, atol=ENERGY_TOL_KWH)):
        raise AssertionError("the attachment-2 truth and the frozen archive truth disagree")
    issued_load = data["issued_load"]
    net_c0, net_c1 = data["net_c0"], data["net_c1"]
    public = pd.read_csv(Q2_DIR / "N_free/natural_dispatch.csv")
    public = public[public.date >= "2025-02-01"]
    assert len(public) == 48096, len(public)
    assert abs(float(public.state_start_kWh.iloc[0]) - PUBLIC_INITIAL_KWH) < ENERGY_TOL_KWH
    assert abs(float(public.plan_kWh.iloc[0]) - PUBLIC_CARRY_KWH) < ENERGY_TOL_KWH
    l80_frame, l80_meta = load_l80(truth_net)
    assert abs(l80_meta["initial_kWh"] - float(public.state_start_kWh.iloc[0])) < ENERGY_TOL_KWH
    timings["inputs_seconds"] = time.perf_counter() - tick
    print(f"inputs: L80 initial {l80_meta['initial_kWh']:.6f} kWh, carry "
          f"{l80_meta['carry_kWh']:.6f} kWh, saved reference "
          f"{l80_meta['reference_total_yuan']:,.6f} yuan", flush=True)

    # ---- quantile archive ----
    tick = time.perf_counter()
    ok = data["availability"]
    assert np.array_equal(ok, availability_mask()), "the stored availability mask differs"
    eps_c0 = truth_net[:, None, :] - net_c0
    eps_c1 = truth_net[:, None, :] - net_c1
    stats_c0, counts_c0 = grouped_order_statistics(eps_c0, ok, (ALPHA, ALPHA_LOW))
    stats_c1, counts_c1 = grouped_order_statistics(eps_c1, ok, (ALPHA, ALPHA_LOW))
    rho80_check = {"c0": stats_c0[ALPHA], "c1": stats_c1[ALPHA]}
    rho75 = {"c0": stats_c0[ALPHA_LOW], "c1": stats_c1[ALPHA_LOW]}
    counts_check = {"c0": counts_c0, "c1": counts_c1}
    protected_of = {
        "L80": data["protected_c0"],
        "L75": net_c0 + rho75["c0"]["rho"],
        "C80": data["protected_c1"],
        "C75": net_c1 + rho75["c1"]["rho"]}
    np.savez_compressed(
        OUT / "protection_q75.npz",
        linear_net_forecast=net_c0, corrected_net_forecast=net_c1,
        linear_rho_q75=rho75["c0"]["rho"], linear_rho_q80=rho80_check["c0"]["rho"],
        corrected_rho_q75=rho75["c1"]["rho"], corrected_rho_q80=rho80_check["c1"]["rho"],
        linear_protected_q75=protected_of["L75"], linear_protected_q80=protected_of["L80"],
        corrected_protected_q75=protected_of["C75"], corrected_protected_q80=protected_of["C80"],
        linear_counts=counts_c0, corrected_counts=counts_c1,
        linear_position_q75=rho75["c0"]["position"],
        corrected_position_q75=rho75["c1"]["position"],
        linear_position_q80=rho80_check["c0"]["position"],
        corrected_position_q80=rho80_check["c1"]["position"],
        alpha_low=ALPHA_LOW, alpha_reference=ALPHA, window_days=WINDOW, min_samples=MIN_SAMPLES)
    timings["quantile_seconds"] = time.perf_counter() - tick
    print("q75 / q80 protection archives built", flush=True)

    checks = {}
    tick = time.perf_counter()
    checks["quantile_inputs"] = check_quantile_inputs(data, rho75, rho80_check, counts_check)
    timings["check_quantile_seconds"] = time.perf_counter() - tick
    print(f"class-1 quantile checks passed (0 MILP): q75<=q80 over "
          f"{checks['quantile_inputs']['monotonicity']['c0_cells_m_ge_7']:,} Linear cells", flush=True)

    # ---- adaptation control on the saved L80 inputs (2 MILPs) ----
    tick = time.perf_counter()
    l80_plans = pd.read_csv(RB_DIR / "B2_plan_versions.csv")
    l80_decisions = pd.read_csv(RB_DIR / "B2_revision_decisions.csv")
    checks["adaptation"] = check_adaptation(price_src, protected_of["L80"], l80_frame, l80_plans,
                                            l80_decisions)
    timings["check_adaptation_seconds"] = time.perf_counter() - tick
    print(f"class-2 adaptation checks passed (2 MILP): 0:00 objective delta "
          f"{checks['adaptation']['midnight']['objective_delta_yuan']:.3e} yuan, 06:00 "
          f"{checks['adaptation']['six']['revision_objective_delta_yuan']:.3e} yuan", flush=True)

    # ---- L80 summary (read-only) ----
    l80_summary, l80_monthly, l80_daily, l80_frame = finalize_group("L80", l80_frame, truth_net)
    reference = l80_meta["reference_row"]
    for key in ("natural_total_yuan", "ordinary_cost_yuan", "adjustment_cost_yuan",
                "emergency_cost_yuan", "emergency_kWh", "unused_kWh", "charge_kWh",
                "discharge_kWh", "final_natural_state_kWh", "final_template_state_kWh"):
        delta = abs(l80_summary[key] - float(reference[key]))
        if delta >= max(CASH_TOL_YUAN, ENERGY_TOL_KWH):
            raise AssertionError(f"L80 replay differs from the saved B2 summary: {key} {delta:.3e}")
    assert abs(l80_summary["natural_total_yuan"] - L80_REFERENCE_YUAN) < CASH_TOL_YUAN
    print(f"L80 reconciled with the saved B2 summary: "
          f"{l80_summary['natural_total_yuan']:,.6f} yuan", flush=True)

    # ---- three new 334-day rolling runs ----
    summaries = {"L80": l80_summary}
    monthlies = {"L80": l80_monthly}
    dailies = {"L80": l80_daily}
    frames = {"L80": l80_frame}
    paths, decisions_by_strategy, revision_info = {}, {}, {}
    for sid in NEW_RUNS:
        tick = time.perf_counter()
        group_frame, plans, decisions, solves, nominal = kernel.run_group(
            "B2", protected_of[sid], truth_load, truth_pv, truth_net, price_src,
            kernel.natural_price(price_src), l80_meta["initial_kWh"], l80_meta["carry_kWh"],
            PUBLIC_CARRY_KWH, l80_meta["carry_version"], l80_meta["carry_owner"])
        out_dir = OUT / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        summary, monthly, daily, group_frame = finalize_group(sid, group_frame, truth_net, out_dir)
        summaries[sid], monthlies[sid], dailies[sid] = summary, monthly, daily
        frames[sid] = group_frame
        frame_to_csv(plans, out_dir / "plan_versions.csv")
        frame_to_csv(decisions, out_dir / "revision_decisions.csv")
        frame_to_csv(solves, out_dir / "solver_log.csv")
        frame_to_csv(nominal, out_dir / "nominal_trajectory.csv")
        decisions_by_strategy[sid] = decisions
        revision_info[sid] = revision_summary(decisions, solves)
        if revision_info[sid]["solves_total"] != 1336 or revision_info[sid]["solves_revision"] != 1002:
            raise AssertionError(f"{sid} solve bookkeeping: {revision_info[sid]}")
        paths[sid] = dict(dispatch=out_dir / "dispatch.csv", solver_log=out_dir / "solver_log.csv",
                          revision_decisions=out_dir / "revision_decisions.csv",
                          plan_versions=out_dir / "plan_versions.csv")
        timings[f"{sid}_seconds"] = time.perf_counter() - tick
        print(f"{sid} finished: natural {summary['natural_total_yuan']:,.2f} yuan, emergency "
              f"{summary['emergency_kWh']:,.1f} kWh, unused {summary['unused_kWh']:,.1f} kWh, "
              f"accepted {revision_info[sid]['revisions_accepted']}/{len(decisions)} revisions, "
              f"{timings[f'{sid}_seconds']:.1f}s", flush=True)

    paths["L80"] = dict(dispatch=RB_DIR / "B2_dispatch.csv", solver_log=RB_DIR / "B2_solver_log.csv",
                        revision_decisions=RB_DIR / "B2_revision_decisions.csv",
                        plan_versions=RB_DIR / "B2_plan_versions.csv")
    decisions_by_strategy["L80"] = l80_decisions
    revision_info["L80"] = revision_summary(l80_decisions, pd.read_csv(paths["L80"]["solver_log"]))

    # ---- class-3 reconciliation, from disk ----
    tick = time.perf_counter()
    checks["saved_results"] = check_saved_results(paths, price_src, truth_net,
                                                  decisions_by_strategy)
    timings["check_saved_seconds"] = time.perf_counter() - tick
    print("class-3 saved-result reconciliation passed (0 MILP)", flush=True)

    # ---- aggregates ----
    nu = float(np.median(price_src) / kernel.ETA)
    for sid in STRATEGIES:
        summaries[sid]["inventory_value_coefficient_yuan_per_kWh"] = nu
        summaries[sid]["cost_net_of_inventory_yuan"] = (
            summaries[sid]["natural_total_yuan"]
            - nu * (summaries[sid]["final_natural_state_kWh"] - summaries[sid]["initial_state_kWh"]))
        summaries[sid].update(revision_info[sid])
    summary_frame = pd.DataFrame([summaries[sid] for sid in STRATEGIES])
    contrasts, interaction = build_contrasts(summaries, nu)
    monthly_long = pd.concat([monthlies[sid] for sid in STRATEGIES], ignore_index=True)
    daily_long = pd.concat([dailies[sid] for sid in STRATEGIES], ignore_index=True)
    monthly_emergency = (daily_long.assign(month=daily_long.date.str[:7])
                         .groupby(["month", "strategy_id"], as_index=False).emergency_kWh.sum())
    monthly_contrasts = build_wide(monthly_long, "total_cost_yuan", key="month").merge(
        build_wide(monthly_emergency, "emergency_kWh", key="month")
        .rename(columns={c: (c if c == "month" else f"em_{c}")
                         for c in build_wide(monthly_emergency, "emergency_kWh", key="month").columns}),
        on="month")
    daily_contrasts = build_wide(daily_long, "total_cost_yuan", key="date")
    selected = pd.concat([
        frames[sid].assign(date=frames[sid].interval_start.dt.strftime("%Y-%m-%d"))
        .query("date in @SELECTED_DATES") for sid in STRATEGIES], ignore_index=True)
    selected = selected.groupby(["strategy_id", "date"], as_index=False).agg(
        ordinary_cost_yuan=("ordinary_cost_yuan", "sum"),
        adjustment_cost_yuan=("adjustment_cost_yuan", "sum"),
        emergency_cost_yuan=("emergency_cost_yuan", "sum"),
        total_cost_yuan=("total_cost_yuan", "sum"),
        emergency_kWh=("emergency_kWh", "sum"), charge_kWh=("charge_kWh", "sum"),
        discharge_kWh=("discharge_kWh", "sum"),
        state_start_kWh=("state_start_kWh", "first"), state_end_kWh=("state_end_kWh", "last"))
    selected_contrasts = build_wide(selected, "total_cost_yuan", key="date")

    frame_to_csv(summary_frame, OUT / "summary.csv")
    frame_to_csv(contrasts, OUT / "contrasts.csv")
    frame_to_csv(monthly_long, OUT / "monthly.csv")
    frame_to_csv(monthly_contrasts, OUT / "monthly_contrasts.csv")
    frame_to_csv(daily_long, OUT / "daily.csv")
    frame_to_csv(daily_contrasts, OUT / "daily_contrasts.csv")
    frame_to_csv(selected, OUT / "selected_dates.csv")
    frame_to_csv(selected_contrasts, OUT / "selected_dates_contrasts.csv")

    extra_solves = checks["adaptation"]["extra_milp_solves"]
    checks["budget"] = dict(
        formal_milp_solves=int(sum(revision_info[sid]["solves_total"] for sid in NEW_RUNS)),
        planned_formal_milp=4008, extra_validation_milp=extra_solves,
        extra_validation_limit=2, within_limit=bool(extra_solves <= 2), new_training=0,
        public_january_reruns=0, l80_full_year_reruns=0,
        note="L80 is a read-only reuse; the three new groups are the 334-day rolling experiment, "
             "not repeat tests")
    validation = dict(status="passed", signature=signature,
                      quantile_inputs=checks["quantile_inputs"], adaptation=checks["adaptation"],
                      saved_results=checks["saved_results"], budget=checks["budget"],
                      archive_alignment=alignment,
                      bias_archive=dict(registration_signature=bias_registration.get("signature"),
                                        archive_sha256=digest(BIAS_NPZ),
                                        manifest_status=bias_manifest.get("status")),
                      limits=["the quantile check re-derives the same archive, so it cannot detect "
                              "a wrong-but-consistent rule written into that archive",
                              "the adaptation control covers one day and one revision, not every "
                              "date; it checks function reuse, it is not an audit",
                              "the reconciliation re-reads persisted artifacts; it is not a full "
                              "independent audit and no figure was inspected visually",
                              "C80 reuses the protected curve computed by the accepted bias "
                              "diagnostic instead of resolving it again here",
                              "2025 was already used for method design: this is a fixed four-group "
                              "rolling causal comparison, not an independent blind test",
                              "q75 is one fixed lower-protection candidate, not a known optimum"])
    save_json(OUT / "validation.json", validation)
    save_json(OUT / "checks.json", checks)
    manifest = dict(status="complete", started_utc=started_utc, finished_utc=utc_now(),
                    wall_seconds=time.perf_counter() - started, timings=timings,
                    signature=signature, executable=sys.executable, python=sys.version,
                    n_solves={sid: int(revision_info[sid]["solves_total"]) for sid in STRATEGIES},
                    n_solves_executed_by_this_script=int(
                        sum(revision_info[sid]["solves_total"] for sid in NEW_RUNS)),
                    n_solves_note="L80 counts the solves recorded in the reused B2 solver log; this "
                                  "script executed only L75/C80/C75 plus 2 validation MILPs",
                    outputs={p.relative_to(OUT).as_posix(): digest(p)
                             for p in sorted(OUT.rglob("*"))
                             if p.is_file() and p.name not in NON_COMPUTED_ARTIFACTS})
    save_json(OUT / "run_manifest.json", manifest)
    print(json.dumps({"status": "complete", "wall_seconds": manifest["wall_seconds"],
                      "totals": {sid: summaries[sid]["natural_total_yuan"] for sid in STRATEGIES},
                      "interaction_yuan": interaction}, indent=2), flush=True)


if __name__ == "__main__":
    main()
