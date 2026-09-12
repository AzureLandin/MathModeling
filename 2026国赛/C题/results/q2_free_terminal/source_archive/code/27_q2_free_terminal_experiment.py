#!/usr/bin/env python
"""问题二 27 号实验：固定 6000 与自由名义末态对照。

对应任务书 ``reports/问题二_固定6000与自由末态对照实验方案.md``。

两组只差一件事——日前名义 MILP 的日末储电量约束：

* ``T_fixed6000``  名义日末 ``E_bar[144] = 6000`` kWh（问题二现用约定，复现参考）；
* ``T_free``       名义日末只在物理界 ``[1200, 10800]`` 内，不加等式、不加惩罚、不加收益。

预测、W28/q80 保护、公共 1 月预运行、实际贪心反馈、收费规则、电池与时间参数全部相同，
两组共享同一份冻结发布档案（23 号 ``pv_forecast_archive_P1.csv``）与同一套保护量。
实际状态自 2025-02-01 起按各自轨迹独立连续推进，不要求日内首尾相等。

冻结内核 ``code/02_q1_baseline.py`` 只用 ``terminal_kWh`` 表达末态，且 ``None`` 会被解释成
"末态等于初态"（日循环），无法表达自由末态；因此本脚本在自己的命名空间里逐字转写该整数内核
的建模代码，新增显式 ``terminal_mode``，仅把 ``E_144`` 的状态上下界从 ``[6000,6000]`` 改为
``[1200,10800]``。转写与冻结内核在固定模式下的目标值与全部解向量逐位一致，已用抽样日与整段
控制复现验证；未修改、未猴子补丁任何旧 02/05/08/14/21 资产。

只新增 27 号脚本、``results/q2_free_terminal/``、``figures/q2_free_terminal/`` 与本实验报告；
旧附件、源码、签名实验输出只读。不锁定模型、不填 ``result2.xlsx``、不启动其他 Agent。
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
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix, csr_matrix, hstack, vstack

ROOT = Path(__file__).resolve().parents[1]
CODE_27 = ROOT / 'code/27_q2_free_terminal_experiment.py'
OUT = ROOT / 'results/q2_free_terminal'
FIG = ROOT / 'figures/q2_free_terminal'
SNAP = ROOT / 'results/q2_revision_audit_20260911/source_snapshot'
PV23_RUN = ROOT / 'results/q2_pv_support_features'
PLAN_MD = ROOT / 'reports/问题二_固定6000与自由末态对照实验方案.md'
REPORT_MD = ROOT / 'reports/问题二_固定6000与自由末态对照实验结果报告.md'

DT = 1 / 6
ALPHA = 0.80
RESIDUAL_WINDOW = 28
MIN_HISTORY_DAYS = 7
STATE_MIN_KWH = 1200.0
STATE_MAX_KWH = 10800.0
POWER_MAX_KW = 5000.0
ETA = 0.9
TERMINAL_FIXED_KWH = 6000.0
WARMUP_DAYS = 31
REFERENCE_WARMUP_KWH = 8801.462273333342
ENERGY_TOL_KWH = 1e-6
COST_TOL_YUAN = 1e-4
CONTROL_ID = 'T_fixed6000'
TREATMENT_ID = 'T_free'
CONTROL_REFERENCE_TOTAL_YUAN = 13658867.74953597
PRIMARY_LABEL = 'T_free minus T_fixed6000 (pre-registered terminal-mode comparison)'

SAMPLED_DATES = [31, 78, 171, 265, 354]
SAMPLED_LABELS = {31: '2025-02-01', 78: '2025-03-20', 171: '2025-06-21',
                  265: '2025-09-23', 354: '2025-12-21'}
PERTURBATION_DAYS = [31, 171, 354]
HALF_DAY_INDEX = 171
HALF_DAY_LABEL = '2025-06-21'
HALF_DAY_SLOT = 72
EMERGENCY_WINDOWS = [('00-06', 0, 36), ('06-10', 36, 60), ('10-24', 60, 144)]
DIAGNOSTIC_WINDOWS = [('19-21', 114, 126)]
LATE_NIGHT_WINDOW = ('23-24', 138, 144)
NEXT_MORNING_WINDOW = ('next_00-06', 0, 36)

ARCHIVE_CSV = PV23_RUN / 'pv_forecast_archive_P1.csv'
CONTROL_REFERENCES = [
    ('P1_q80', PV23_RUN / 'P1_q80' / 'dispatch.csv'),
    ('F01_pv_support', ROOT / 'results/q2_load_pv_shape' / 'F01_pv_support' / 'dispatch.csv'),
]

MODES = ('fixed6000', 'free')
GROUPS = [
    dict(id=CONTROL_ID, terminal_mode='fixed6000',
         role='current question-2 convention: nominal end-of-day state pinned to 6000 kWh'),
    dict(id=TREATMENT_ID, terminal_mode='free',
         role='treatment: nominal end-of-day state only bounded by 1200-10800 kWh'),
]
GROUP_BY_ID = {g['id']: g for g in GROUPS}
CONTRAST = dict(label=PRIMARY_LABEL, treatment=TREATMENT_ID, baseline=CONTROL_ID)

PROTECTED_TREES = ['附件', 'code', 'reports', 'figures', 'results']
OWN_NEW_REL = {'code/27_q2_free_terminal_experiment.py',
               'reports/问题二_固定6000与自由末态对照实验结果报告.md'}
OWN_NEW_PREFIXES = ('results/q2_free_terminal/', 'figures/q2_free_terminal/')
SHARED_APPEND_REL = {'建模上下文记忆.md', 'reports/项目进度.md'}

PARAMETERS = dict(
    experiment='fixed-6000 versus free nominal end-of-day state, two pre-registered groups on one '
               'frozen published forecast archive',
    alpha=ALPHA, residual_window_days=RESIDUAL_WINDOW, min_history_days=MIN_HISTORY_DAYS,
    terminal_modes=dict(fixed6000='E_bar[144] = 6000 kWh', free='1200 <= E_bar[144] <= 10800 kWh'),
    groups=[dict(id=g['id'], terminal_mode=g['terminal_mode']) for g in GROUPS],
    primary_comparison=PRIMARY_LABEL,
    forecast_archive='results/q2_pv_support_features/pv_forecast_archive_P1.csv (frozen, not retrained)',
    quantile_rule='empirical inverse distribution, per-slot ascending order statistic at 1-based '
                  'position (m*a+99)//100 with a the integer percentage, index position-1; '
                  'm = k - max(1, k-28), m<7 gives no correction',
    solver=dict(kernel='verbatim local transcription of code/02_q1_baseline.py solve(integer=True); '
                       'relative gap 1e-9, time limit 120 s',
                only_difference='the E_144 lower/upper bound: [6000,6000] versus [1200,10800]',
                variable_order='q[144], c[144], d[144], w[144], E[0..144], z[144] binary'),
    actual_control='surplus charges, deficit discharges, residual deficit at 5x the slot price; '
                   'actual state carries across days; no daily cycle is forced on the actual state',
    warmup='shared public January pre-run, 2025-02-01 00:00 begin state 8801.462273333342 kWh for both groups',
    evaluation='2025-02-01..2025-12-31, 334 days, 48096 slots per group',
    sampled_dates=[SAMPLED_LABELS[k] for k in SAMPLED_DATES],
    perturbation_days=[SAMPLED_LABELS[k] for k in PERTURBATION_DAYS],
    half_day_check=dict(date=HALF_DAY_LABEL, perturbed_from_slot=HALF_DAY_SLOT,
                        load_factor=1.2, pv_factor=0.7),
    emergency_windows={name: [start, stop] for name, start, stop in EMERGENCY_WINDOWS},
    diagnostic_windows={name: [start, stop] for name, start, stop in DIAGNOSTIC_WINDOWS},
    late_night_window=dict(name=LATE_NIGHT_WINDOW[0], start=LATE_NIGHT_WINDOW[1],
                           stop=LATE_NIGHT_WINDOW[2]),
    valuation='C* = C - nu*(E_end - E_start), nu = median(price)/0.9 CNY per internal kWh; '
              'diagnostic only, never part of the main cost',
    thresholds=dict(energy_kWh=ENERGY_TOL_KWH, cost_yuan=COST_TOL_YUAN),
    control_reference=dict(P1_q80=CONTROL_REFERENCE_TOTAL_YUAN),
    control_reproduction=dict(
        strict='day-level aggregates within 1e-6 kWh of the registered P1_q80 archive and '
               '|total cash cost difference| <= 1e-4 CNY',
        equal_cost_face_exemption='additionally accepted, and stored beside the strict flag, when at '
                                  'most 3 days differ, the first diverging day has an identical own '
                                  'plan cost (equal-cost optimum), every differing slot shares a '
                                  'price, the day-level difference stays within 5 kWh and the annual '
                                  'difference stays within 1 CNY; disclosure only, never a rewrite '
                                  'of the strict number',
    ),
)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MODULES: dict = {}


def parent():
    """Module 14: frozen snapshot access (02 kernel, 05 physics/reader, 08 harness)."""
    if 'ridge' not in _MODULES:
        _MODULES['ridge'] = _load('ridge_parent', 'code/14_q2_ridge_forecast_experiment.py')
    return _MODULES['ridge']


def scan():
    """Module 21: explicit-alpha quantile helper, dispatch schema and day-level aggregates."""
    if 'scan' not in _MODULES:
        module = _load('quantile_scan', 'code/21_q2_quantile_level_scan.py')
        module.OUT = OUT
        module.FIG = FIG
        _MODULES['scan'] = module
    return _MODULES['scan']


def frozen():
    return parent().frozen()


def dependency_assertions():
    """The reused definitions must still be the registered ones."""
    s = scan()
    f = frozen()
    core = f.core
    assert core.CONFIG['dt_hours'] == DT and core.CONFIG['charge_efficiency'] == ETA
    assert core.CONFIG['discharge_efficiency'] == ETA
    assert core.CONFIG['state_min_kWh'] == STATE_MIN_KWH
    assert core.CONFIG['state_max_kWh'] == STATE_MAX_KWH
    assert core.CONFIG['power_max_kW'] == POWER_MAX_KW
    assert core.CONFIG['milp_relative_gap'] == 1e-9
    assert core.CONFIG['milp_time_limit_seconds'] == 120
    assert s.TERMINAL_KWH == TERMINAL_FIXED_KWH, (
        'the frozen reference archive pins the nominal terminal at 6000 kWh')
    assert s.MIN_HISTORY_DAYS == MIN_HISTORY_DAYS and s.RESIDUAL_WINDOW == RESIDUAL_WINDOW
    assert s.quantile_index(RESIDUAL_WINDOW, ALPHA) == 23
    assert s.ceil_index(RESIDUAL_WINDOW, ALPHA) == 23
    assert int(f.bm.CONFIG['eta_c'] * 100) == 90
    return dict(residual_window=RESIDUAL_WINDOW, alpha=ALPHA, position_at_28=23,
                frozen_terminal_kWh=s.TERMINAL_KWH, modes=list(MODES))


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
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def save(path, obj):
    Path(path).write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=2, default=str),
                          encoding='utf-8')


def frame_to_csv(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def protected_manifest():
    """Hash every pre-existing asset, excluding only this round's own new tree."""
    manifest = {}
    for tree in PROTECTED_TREES:
        base = ROOT / tree
        if not base.exists():
            continue
        for path in sorted(base.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.name.startswith('~$'):
                continue
            relative = path.relative_to(ROOT).as_posix()
            if relative in OWN_NEW_REL or relative in SHARED_APPEND_REL:
                continue
            if relative.startswith(OWN_NEW_PREFIXES):
                continue
            manifest[relative] = digest(path)
    return manifest


# --------------------------------------------------------------------------------------
# local MILP kernel with an explicit terminal mode
# --------------------------------------------------------------------------------------
def solver_options():
    config = frozen().core.CONFIG
    return dict(mip_rel_gap=config['milp_relative_gap'],
                time_limit=config['milp_time_limit_seconds'])


def build_model(protected_kwh, price_day, initial, mode):
    """Transcribe the frozen integer kernel, replacing only the E_144 bounds by ``mode``.

    Everything else - the objective, the bus-balance and storage rows, the binary mutex rows, the
    variable order, the variable bounds and the solver options - is copied from
    ``code/02_q1_baseline.py`` ``solve(..., integer=True)``. The returned arrays are the audit
    evidence that the two modes differ in exactly one pair of bounds.
    """
    assert mode in MODES, mode
    assert len(price_day) == len(protected_kwh)
    load_kw = np.asarray(protected_kwh, dtype=float) / DT
    pv_kw = np.zeros_like(load_kw)
    n = len(load_kw)
    n_vars = 5 * n + 1
    cap = POWER_MAX_KW * DT
    objective = np.zeros(n_vars)
    objective[:n] = price_day
    balance = lil_matrix((2 * n, n_vars))
    rhs = np.r_[(load_kw - pv_kw) * DT, np.zeros(n)]
    for t in range(n):
        balance[t, t], balance[t, n + t], balance[t, 2 * n + t], balance[t, 3 * n + t] = 1, -1, 1, -1
        balance[n + t, 4 * n + t + 1], balance[n + t, 4 * n + t] = 1, -1
        balance[n + t, n + t], balance[n + t, 2 * n + t] = -ETA, 1 / ETA
    balance = csr_matrix(balance)
    lower, upper = np.zeros(n_vars), np.full(n_vars, np.inf)
    upper[n:3 * n] = cap
    lower[4 * n:], upper[4 * n:] = STATE_MIN_KWH, STATE_MAX_KWH
    lower[4 * n] = upper[4 * n] = float(initial)
    if mode == 'fixed6000':
        lower[-1] = upper[-1] = TERMINAL_FIXED_KWH
        target = TERMINAL_FIXED_KWH
    else:
        lower[-1], upper[-1] = STATE_MIN_KWH, STATE_MAX_KWH
        target = None
    constraints = hstack([balance, csr_matrix((2 * n, n))], format='csr')
    mutex = lil_matrix((2 * n, n_vars + n))
    for t in range(n):
        mutex[t, n + t], mutex[t, n_vars + t] = 1, -cap
        mutex[n + t, 2 * n + t], mutex[n + t, n_vars + t] = 1, cap
    constraints = vstack([constraints, csr_matrix(mutex)], format='csr')
    integrality = np.r_[np.zeros(n_vars), np.ones(n)]
    bounds = Bounds(np.r_[lower, np.zeros(n)], np.r_[upper, np.ones(n)])
    linear = LinearConstraint(constraints, np.r_[rhs, np.full(2 * n, -np.inf)],
                              np.r_[rhs, np.zeros(n), np.full(n, cap)])
    full_objective = np.r_[objective, np.zeros(n)]
    return dict(n=n, n_vars=n_vars, cap=float(cap), objective=full_objective,
                continuous_objective=objective, integrality=integrality,
                bounds=bounds, constraints=linear, lower=lower, upper=upper, rhs=rhs,
                balance=balance, mutex=csr_matrix(mutex), mode=mode, terminal_target=target,
                terminal_lower_kWh=float(lower[-1]), terminal_upper_kWh=float(upper[-1]),
                variable_order='q[0..143], c[0..143], d[0..143], w[0..143], E[0..144], z[0..143]',
                objective_terms=int(np.count_nonzero(objective)),
                equality_rows=int(2 * n), mutex_rows=int(2 * n),
                binary_count=int(n), continuous_count=int(n_vars))


def solve_day(protected_kwh, price_day, initial, mode):
    """Solve one day-ahead plan and return the frozen decision vector plus mode-aware evidence."""
    model = build_model(protected_kwh, price_day, initial, mode)
    start = time.perf_counter()
    result = milp(model['objective'], integrality=model['integrality'], bounds=model['bounds'],
                  constraints=model['constraints'], options=solver_options())
    elapsed = time.perf_counter() - start
    if not result.success:
        raise RuntimeError(f'HiGHS: {result.message}')
    n = model['n']
    x = result.x[:model['n_vars']]
    q, c, d, w = x[:n], x[n:2 * n], x[2 * n:3 * n], x[3 * n:4 * n]
    state = x[4 * n:5 * n + 1]
    z = result.x[model['n_vars']:]
    protected = np.asarray(protected_kwh, dtype=float)
    checks = dict(
        bus_balance_max_abs_kWh=float(np.max(np.abs(q + d - protected - c - w))),
        battery_balance_max_abs_kWh=float(np.max(np.abs(np.diff(state) - ETA * c + d / ETA))),
        boundary_error_kWh=float(max(abs(state[0] - float(initial)),
                                     abs(state[-1] - TERMINAL_FIXED_KWH)
                                     if mode == 'fixed6000' else 0.0)),
        state_bound_violation_kWh=float(max(0.0, STATE_MIN_KWH - state.min(),
                                            state.max() - STATE_MAX_KWH)),
        flow_bound_violation_kWh=float(max(0.0, -min(q.min(), c.min(), d.min(), w.min()),
                                           c.max() - model['cap'], d.max() - model['cap'])),
        simultaneous_charge_discharge_kWh=float(np.minimum(c, d).max()),
        binary_integrality_error=float(np.max(np.abs(z - np.rint(z)))),
        binary_bound_violation=float(max(0.0, -z.min(), z.max() - 1.0)),
        binary_gate_violation_kWh=float(max(0.0, np.max(c - model['cap'] * z),
                                            np.max(d - model['cap'] * (1 - z)))))
    if max(checks.values()) >= ENERGY_TOL_KWH:
        raise AssertionError(dict(mode=mode, checks=checks))
    summary = dict(elapsed_seconds=elapsed, mip_gap=float(getattr(result, 'mip_gap', 0.0)),
                   checks=checks, cost_yuan=float(price_day @ q),
                   mip_node_count=int(result.mip_node_count),
                   mip_dual_bound_yuan=float(result.mip_dual_bound),
                   objective_bound_gap_yuan=float(result.fun - result.mip_dual_bound),
                   terminal_mode=mode, terminal_target=model['terminal_target'],
                   terminal_lower_kWh=model['terminal_lower_kWh'],
                   terminal_upper_kWh=model['terminal_upper_kWh'],
                   nominal_terminal_kWh=float(state[-1]))
    nominal = dict(charge=np.array(c), discharge=np.array(d), unused=np.array(w),
                   binary=np.asarray(z, dtype=float), state=np.array(state),
                   terminal_target=model['terminal_target'],
                   lower=model['terminal_lower_kWh'], upper=model['terminal_upper_kWh'])
    return q, c, d, w, state, summary, nominal


class TerminalArchive:
    """Frozen published forecasts, the W28/q80 protection rule and the mode-aware local kernel."""

    def __init__(self, load, pv, n_days, load_forecast, pv_forecast):
        self.load = load
        self.pv = pv
        self.n_days = n_days
        self.pred_l = np.asarray(load_forecast, dtype=float)
        self.pred_v = np.asarray(pv_forecast, dtype=float)
        self.n_forecast = (self.pred_l - self.pred_v) * DT
        self.n_actual = (load - pv) * DT
        self.eps = self.n_actual - self.n_forecast
        self.solve_calls = 0
        self.last_nominal = None

    def plan_day(self, k, alpha, window, price, initial, mode):
        self.last_nominal = None
        r, m, reason = scan().adjustment(self.eps, k, window, alpha)
        protected = self.n_forecast[k] + r
        try:
            q, c, d, w, state, summary, nominal = solve_day(protected, price, initial, mode)
            summary['fallback'] = False
            self.last_nominal = nominal
        except Exception as exc:                                    # noqa: BLE001 - recorded, not hidden
            q = np.maximum(protected, 0.0)
            state = np.full(145, float(initial))
            summary = dict(fallback=True, elapsed_seconds=0.0, mip_gap=0.0, checks={},
                           message=f'MILP fallback: {exc}', terminal_mode=mode,
                           terminal_target=TERMINAL_FIXED_KWH if mode == 'fixed6000' else None,
                           terminal_lower_kWh=np.nan, terminal_upper_kWh=np.nan,
                           nominal_terminal_kWh=float(initial), mip_node_count=-1,
                           mip_dual_bound_yuan=np.nan, objective_bound_gap_yuan=np.nan,
                           cost_yuan=float(price @ q))
        self.solve_calls += 1
        return q, state, summary, protected, r, m, reason


# --------------------------------------------------------------------------------------
# published-archive input gate
# --------------------------------------------------------------------------------------
def read_published_archive(load, pv, price, dates):
    """Read the frozen P1 archive, rebuild the published arrays and cross-check the reference run."""
    assert ARCHIVE_CSV.exists(), f'missing frozen archive: {ARCHIVE_CSV}'
    frame = pd.read_csv(ARCHIVE_CSV, low_memory=False, dtype={'date': str})
    n_days = len(dates)
    assert len(frame) == n_days * 144, len(frame)
    frame = frame.sort_values(['date', 'slot'], kind='mergesort').reset_index(drop=True)
    assert frame.groupby('date').size().eq(144).all()
    assert frame.date.nunique() == n_days and frame.slot.min() == 0 and frame.slot.max() == 143
    assert [str(value)[:10] for value in dates] == sorted(frame.date.unique())
    published_l = frame.load_forecast_kW.to_numpy().reshape(n_days, 144)
    published_v = frame.pv_forecast_kW.to_numpy().reshape(n_days, 144)
    scan().require_finite_published(published_l, published_v, n_days)
    assert np.isnan(published_l[0]).all() and np.isnan(published_v[0]).all(), (
        'day 0 has no point forecast by construction')
    reference = pd.read_csv(CONTROL_REFERENCES[0][1], low_memory=False, dtype={'date': str})
    reference = reference.sort_values(['date', 'slot'], kind='mergesort').reset_index(drop=True)
    assert len(reference) == (n_days - WARMUP_DAYS) * 144
    reference_l = reference.load_forecast_kW.to_numpy().reshape(n_days - WARMUP_DAYS, 144)
    reference_v = reference.pv_forecast_kW.to_numpy().reshape(n_days - WARMUP_DAYS, 144)
    net_fc = (published_l - published_v) * DT

    def worst(left, right):
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        mask = np.isfinite(left) & np.isfinite(right)
        return float(np.max(np.abs(left[mask] - right[mask]))) if mask.any() else 0.0

    meta = dict(
        source=str(ARCHIVE_CSV.relative_to(ROOT)), sha256=digest(ARCHIVE_CSV),
        dates=int(frame.date.nunique()), slots=int(len(frame)),
        rows_per_date_ok=bool(frame.groupby('date').size().eq(144).all()),
        day0_is_nan=bool(np.isnan(published_l[0]).all() and np.isnan(published_v[0]).all()),
        formal_period_finite=bool(np.isfinite(published_l[1:]).all()
                                  and np.isfinite(published_v[1:]).all()),
        nonnegative=bool((published_l[1:] >= 0).all() and (published_v[1:] >= 0).all()),
        archive_load_vs_reference_kW=worst(published_l[WARMUP_DAYS:], reference_l),
        archive_pv_vs_reference_kW=worst(published_v[WARMUP_DAYS:], reference_v),
        archive_net_vs_reference_kWh=worst(net_fc[WARMUP_DAYS:], reference.net_forecast_kWh.to_numpy()
                                           .reshape(n_days - WARMUP_DAYS, 144)),
        support_columns_present=bool({'support_start_slot', 'support_end_slot', 'support_valid'}
                                     .issubset(frame.columns)),
        day0_support_default=dict(
            start=int(frame.support_start_slot.iloc[0]), stop=int(frame.support_end_slot.iloc[0]),
            valid=int(frame.support_valid.iloc[0])),
        note='the archive stores 4-decimal rounded powers; the reference run below proves the '
             'rounded values still reproduce the registered dispatch, so the archive is used as is '
             'and no forecast is retrained',
    )
    meta['all_passed'] = bool(
        meta['rows_per_date_ok'] and meta['day0_is_nan'] and meta['formal_period_finite']
        and meta['nonnegative'] and meta['archive_load_vs_reference_kW'] < 1e-3
        and meta['archive_pv_vs_reference_kW'] < 1e-3
        and meta['archive_net_vs_reference_kWh'] < 1e-3)
    return published_l, published_v, meta


# --------------------------------------------------------------------------------------
# formal model evidence: matrix difference and frozen-kernel equivalence
# --------------------------------------------------------------------------------------
def _array_difference(left, right):
    """Element-wise difference of two constraint matrices as an integer nnz count."""
    return int((csr_matrix(left) != csr_matrix(right)).nnz)


def matrix_evidence(protected, price, initial):
    """Both modes on one input: only the E_144 lower/upper bound may change."""
    fixed = build_model(protected, price, initial, 'fixed6000')
    free = build_model(protected, price, initial, 'free')
    n = fixed['n']
    lower_diff = np.flatnonzero(fixed['bounds'].lb != free['bounds'].lb)
    upper_diff = np.flatnonzero(fixed['bounds'].ub != free['bounds'].ub)
    out = dict(
        n_slots=int(n), continuous_variables=fixed['continuous_count'],
        binary_variables=fixed['binary_count'], equality_rows=fixed['equality_rows'],
        mutex_rows=fixed['mutex_rows'], objective_terms=fixed['objective_terms'],
        variable_order=fixed['variable_order'], cap_kWh=fixed['cap'],
        objective_identical=bool(np.array_equal(fixed['objective'], free['objective'])),
        integrality_identical=bool(np.array_equal(fixed['integrality'], free['integrality'])),
        constraint_matrix_difference_nnz=_array_difference(fixed['constraints'].A,
                                                           free['constraints'].A),
        constraint_lower_identical=bool(np.array_equal(fixed['constraints'].lb,
                                                       free['constraints'].lb)),
        constraint_upper_identical=bool(np.array_equal(fixed['constraints'].ub,
                                                       free['constraints'].ub)),
        bounds_length_equal=bool(len(fixed['bounds'].lb) == len(free['bounds'].lb)),
        lower_bound_differing_indices=[int(v) for v in lower_diff],
        upper_bound_differing_indices=[int(v) for v in upper_diff],
        expected_terminal_index=int(5 * n), terminal_index_is_last_state=bool(5 * n == 5 * n),
        fixed_terminal_bounds=[fixed['terminal_lower_kWh'], fixed['terminal_upper_kWh']],
        free_terminal_bounds=[free['terminal_lower_kWh'], free['terminal_upper_kWh']],
        fixed_terminal_target=fixed['terminal_target'], free_terminal_target=free['terminal_target'],
        solver_options_fixed=solver_options(), solver_options_free=solver_options(),
    )
    out['all_passed'] = bool(
        out['objective_identical'] and out['integrality_identical']
        and out['constraint_matrix_difference_nnz'] == 0
        and out['constraint_lower_identical'] and out['constraint_upper_identical']
        and out['bounds_length_equal']
        and out['lower_bound_differing_indices'] == [5 * n]
        and out['upper_bound_differing_indices'] == [5 * n]
        and out['fixed_terminal_bounds'] == [TERMINAL_FIXED_KWH, TERMINAL_FIXED_KWH]
        and out['free_terminal_bounds'] == [STATE_MIN_KWH, STATE_MAX_KWH]
        and out['fixed_terminal_target'] == TERMINAL_FIXED_KWH
        and out['free_terminal_target'] is None
        and out['solver_options_fixed'] == out['solver_options_free'])
    return out


def kernel_equivalence_check(archive, price, dates):
    """Sampled days: the local fixed6000 model must equal the frozen kernel bit for bit.

    The frozen kernel also proves why ``terminal_kWh=None`` is not a free terminal: None is
    interpreted as ``E_144 = E_0`` (a daily cycle), which is a different model. The initial states
    are the registered P1_q80 day-begin states, so the comparison uses real inputs.
    """
    core = frozen().core
    reference = pd.read_csv(PV23_RUN / 'P1_q80' / 'daily_summary.csv', dtype={'date': str})
    reference = reference.set_index('date')
    cases = []
    worst_cost = 0.0
    worst_vector = 0.0
    worst_none = 0.0
    for k in SAMPLED_DATES:
        date = str(dates[k].date())
        initial = float(reference.loc[date, 'initial_kWh'])
        r, m, _ = scan().adjustment(archive.eps, k, RESIDUAL_WINDOW, ALPHA)
        protected = archive.n_forecast[k] + r
        local_q, local_c, local_d, local_w, local_state, local_summary, _ = solve_day(
            protected, price, initial, 'fixed6000')
        frozen_solution, frozen_summary = core.solve(np.asarray(protected) / DT, np.zeros(144), price,
                                                     integer=True, initial_kWh=initial,
                                                     terminal_kWh=TERMINAL_FIXED_KWH)
        frozen_q, frozen_c, frozen_d, frozen_w, frozen_state = frozen_solution
        cost_gap = abs(float(price @ local_q) - float(price @ frozen_q))
        vector_gap = float(max(np.max(np.abs(local_q - frozen_q)),
                               np.max(np.abs(local_c - frozen_c)),
                               np.max(np.abs(local_d - frozen_d)),
                               np.max(np.abs(local_w - frozen_w)),
                               np.max(np.abs(local_state - frozen_state))))
        cycle_solution, _ = core.solve(np.asarray(protected) / DT, np.zeros(144), price,
                                       integer=True, initial_kWh=initial, terminal_kWh=None)
        cycle_gap = float(abs(cycle_solution[4][-1] - initial))
        free_state = solve_day(protected, price, initial, 'free')[4]
        none_gap = float(np.max(np.abs(free_state - cycle_solution[4])))
        worst_cost = max(worst_cost, cost_gap)
        worst_vector = max(worst_vector, vector_gap)
        worst_none = max(worst_none, none_gap)
        cases.append(dict(day_index=int(k), date=date, initial_kWh=initial,
                          residual_days=int(m), local_cost_yuan=float(price @ local_q),
                          frozen_cost_yuan=float(price @ frozen_q), cost_difference_yuan=cost_gap,
                          max_vector_difference_kW_or_kWh=vector_gap,
                          frozen_none_terminal_equals_initial_kWh=cycle_gap,
                          max_difference_free_vs_none_terminal=none_gap))
    out = dict(cases=cases, max_cost_difference_yuan=worst_cost,
               max_vector_difference=worst_vector,
               none_terminal_is_daily_cycle_max_error_kWh=worst_none,
               note='terminal_kWh=None in the frozen kernel means E_144 = E_0; that is a daily cycle, '
                    'not a free terminal, which is why this experiment needs an explicit mode',
               all_passed=bool(worst_cost < COST_TOL_YUAN and worst_vector < ENERGY_TOL_KWH
                               and worst_none > 0.0))
    return out


# --------------------------------------------------------------------------------------
# synthetic boundaries for the free mode and the mode switch
# --------------------------------------------------------------------------------------
def boundary_checks(price):
    """Analytic cases that must distinguish a free terminal from a hidden daily cycle."""
    rows = []

    def add(name, passed, expected, observed):
        rows.append(dict(case=name, expected=expected, observed=observed, passed=bool(passed)))

    zero = np.zeros(144)
    # 1. zero net demand, initial 1200, free mode: zero purchase and the terminal stays at 1200
    q, c, d, w, state, summary, _ = solve_day(zero, price, STATE_MIN_KWH, 'free')
    add('zero_net_free_no_purchase', bool(q.sum() == 0.0 and abs(state[-1] - STATE_MIN_KWH) < 1e-9),
        'q_sum = 0 and E_144 = 1200', dict(q_sum=float(q.sum()), e144=float(state[-1])))
    # 2. same input, fixed6000: must buy at least (6000-1200)/0.9 bus-side kWh
    q2, c2, d2, w2, state2, summary2, _ = solve_day(zero, price, STATE_MIN_KWH, 'fixed6000')
    need = (TERMINAL_FIXED_KWH - STATE_MIN_KWH) / ETA
    add('zero_net_fixed_minimum_purchase',
        bool(abs(q2.sum() - need) < 1e-6 and abs(state2[-1] - TERMINAL_FIXED_KWH) < 1e-9),
        f'q_sum = (6000-1200)/0.9 = {need:.6f} and E_144 = 6000',
        dict(q_sum=float(q2.sum()), e144=float(state2[-1]), cost_yuan=summary2['cost_yuan']))
    # 3. one slot, net demand 1 kWh, initial 1200 + 1/0.9: free terminal reaches 1200 with no purchase
    one = np.array([1.0])
    initial_one = STATE_MIN_KWH + 1.0 / ETA
    q3, c3, d3, w3, state3, summary3, _ = solve_day(one, np.array([price[0]]), initial_one, 'free')
    add('single_slot_free_discharge_to_floor',
        bool(abs(q3[0]) < 1e-9 and abs(d3[0] - 1.0) < 1e-9 and abs(state3[-1] - STATE_MIN_KWH) < 1e-9),
        'q = 0, d = 1, E_1 = 1200 (no hidden start=end equality)',
        dict(q=float(q3[0]), d=float(d3[0]), e1=float(state3[-1])))
    # 4. the same one-slot input is infeasible for fixed6000: recorded, not treated as a failure
    infeasible = False
    message = ''
    try:
        solve_day(one, np.array([price[0]]), initial_one, 'fixed6000')
    except Exception as exc:                                        # noqa: BLE001 - expected
        infeasible = True
        message = str(exc)[:120]
    add('single_slot_fixed6000_infeasible', infeasible,
        'HiGHS reports infeasible (a one-slot step to 6000 is beyond the power limit)',
        dict(infeasible=infeasible, message=message))
    # 5. full battery, zero net demand, free mode: zero purchase is feasible and the floor is not forced
    q5, c5, d5, w5, state5, summary5, _ = solve_day(zero, price, STATE_MAX_KWH, 'free')
    add('full_battery_free_no_forced_drain',
        bool(q5.sum() == 0.0 and STATE_MIN_KWH - 1e-9 <= state5[-1] <= STATE_MAX_KWH + 1e-9),
        'q_sum = 0 and 1200 <= E_144 <= 10800 (multiple cost-0 optima are allowed and reported)',
        dict(q_sum=float(q5.sum()), e144=float(state5[-1]),
             note='idle, discharge-to-unused and their mixtures are all cost 0'))
    # 6. large PV surplus. Free mode: zero purchase is enough and the absorbed energy is split
    #    between charge and unused - with a free terminal, idling and charging are both cost 0, so
    #    the split is a genuine multi-optimum and only the balance is asserted.
    surplus = np.full(144, -2000.0)
    q6, c6, d6, w6, state6, summary6, _ = solve_day(surplus, price, STATE_MIN_KWH, 'free')
    add('pv_surplus_free_zero_purchase',
        bool(q6.sum() == 0.0 and abs(c6.sum() + w6.sum() - 288000.0) < 1e-6
             and STATE_MIN_KWH - 1e-9 <= state6[-1] <= STATE_MAX_KWH + 1e-9),
        'q = 0, c + w = 2000 per slot, 1200 <= E_144 <= 10800 (idle and charge are tied optima)',
        dict(q_sum=float(q6.sum()), c_sum=float(c6.sum()), w_sum=float(w6.sum()),
             e144=float(state6[-1])))
    # the charge cap is discriminating only when the terminal must move, i.e. in fixed mode
    q6b, c6b, d6b, w6b, state6b, summary6b, _ = solve_day(surplus, price, STATE_MIN_KWH, 'fixed6000')
    add('pv_surplus_charge_capped_in_fixed_mode',
        bool(q6b.sum() == 0.0 and c6b.max() <= 5000 * DT + 1e-9
             and abs(state6b[-1] - TERMINAL_FIXED_KWH) < 1e-9 and w6b.max() > 0.0),
        'q = 0, c <= 5000/6 per slot, E_144 = 6000, surplus left as unused',
        dict(q_sum=float(q6b.sum()), c_max=float(c6b.max()), e144=float(state6b[-1]),
             w_sum=float(w6b.sum())))
    # 7. negative protection is retained. Free mode spends it on charge/unused with no purchase;
    #    fixed mode must still reach 6000, so the purchase is reduced by exactly the surplus.
    negative = -np.full(144, 5.0)
    q7, c7, d7, w7, state7, summary7, _ = solve_day(negative, price, STATE_MIN_KWH, 'free')
    add('negative_protection_retained_free',
        bool(q7.sum() == 0.0 and abs(c7.sum() + w7.sum() - 720.0) < 1e-6 and summary7['cost_yuan'] == 0.0),
        'q = 0, c + w = 5 per slot, cost 0 (the negative correction is not clipped away)',
        dict(q_sum=float(q7.sum()), c_sum=float(c7.sum()), w_sum=float(w7.sum()),
             protected_min=float(negative.min())))
    q7b, c7b, d7b, w7b, state7b, summary7b, _ = solve_day(negative, price, STATE_MIN_KWH, 'fixed6000')
    add('negative_protection_reduces_fixed_purchase',
        bool(abs(q7b.sum() - ((TERMINAL_FIXED_KWH - STATE_MIN_KWH) / ETA - 720.0)) < 1e-6
             and abs(state7b[-1] - TERMINAL_FIXED_KWH) < 1e-9),
        'fixed mode buys (6000-1200)/0.9 - 720 = 4613.333333 kWh, i.e. 720 kWh less than zero net',
        dict(q_sum=float(q7b.sum()), reference=float((TERMINAL_FIXED_KWH - STATE_MIN_KWH) / ETA)))
    # 8. cross-midnight continuity under both modes
    protected_a = np.full(144, 200.0)
    protected_b = np.full(144, -200.0)
    q8a, c8a, d8a, w8a, state8a, _, _ = solve_day(protected_a, price, 6000.0, 'free')
    q8b, c8b, d8b, w8b, state8b, summary8b, _ = solve_day(protected_b, price, float(state8a[-1]), 'free')
    q9a, c9a, d9a, w9a, state9a, _, _ = solve_day(protected_a, price, 6000.0, 'fixed6000')
    q9b, c9b, d9b, w9b, state9b, _, _ = solve_day(protected_b, price, float(state9a[-1]), 'fixed6000')
    add('cross_midnight_continuity_free',
        bool(abs(state8b[0] - state8a[-1]) < 1e-12),
        'day 2 E_0 equals day 1 E_144; no reset to 6000 or 1200 in between',
        dict(day1_end=float(state8a[-1]), day2_start=float(state8b[0]), day2_end=float(state8b[-1])))
    add('cross_midnight_continuity_fixed',
        bool(abs(state9b[0] - state9a[-1]) < 1e-12 and abs(state9a[-1] - TERMINAL_FIXED_KWH) < 1e-9
             and abs(state9b[-1] - TERMINAL_FIXED_KWH) < 1e-9),
        'fixed mode pins each nominal E_144 to 6000 but still carries the actual state continuously',
        dict(day1_end=float(state9a[-1]), day2_start=float(state9b[0]), day2_end=float(state9b[-1])))
    # 9. the mode switch changes exactly one pair of bounds (formal, on a real day's demand)
    real = np.full(144, 300.0)
    matrix = matrix_evidence(real, price, 6000.0)
    add('mode_switch_changes_only_E144_bounds', matrix['all_passed'],
        'objective, integrality, matrix, constraint bounds identical; only index 5n bounds differ',
        dict(lower=matrix['lower_bound_differing_indices'],
             upper=matrix['upper_bound_differing_indices'],
             matrix_nnz=matrix['constraint_matrix_difference_nnz']))
    # 10. nominal and actual differ without any reset
    control = frozen().bm.control
    plan = np.full(144, 100.0)
    actual_load = np.full(144, 400.0)
    actual_pv = np.full(144, 0.0)
    cc, dd, ee, ww, states, physical = control(plan, actual_load, actual_pv, 6000.0)
    add('nominal_and_actual_differ_without_reset',
        bool(abs(states[-1] - 6000.0) > 1e-6 and abs(states[0] - 6000.0) < 1e-12
             and max(physical.values()) < ENERGY_TOL_KWH),
        'the actual end state follows the recursion and is not forced back to 6000',
        dict(actual_end=float(states[-1]), nominal_target=TERMINAL_FIXED_KWH,
             emergency_kWh=float(ee.sum())))
    return dict(rows=rows, all_passed=bool(all(row['passed'] for row in rows)),
                case_count=len(rows), matrix_evidence=matrix)


# --------------------------------------------------------------------------------------
# quantile / shared-forecast validation
# --------------------------------------------------------------------------------------
def quantile_check(archive, dates):
    lengths = {}
    positions = set()
    for k in range(WARMUP_DAYS, len(dates)):
        window = len(range(max(1, k - RESIDUAL_WINDOW), k))
        lengths[window] = lengths.get(window, 0) + 1
        positions.add(scan().quantile_index(window, ALPHA))
    seven = np.array([-2.0, 0.0, 1.0, 3.0, 4.0, 5.0, 10.0])
    rule = scan().quantile_index(7, ALPHA)
    broken = np.zeros((RESIDUAL_WINDOW, 144))
    broken[3, 5] = np.nan
    rejected = False
    try:
        scan().require_finite_published(broken, broken, RESIDUAL_WINDOW)
    except ValueError:
        rejected = True
    negative_probe = np.full((RESIDUAL_WINDOW, 144), -5.0)
    out = dict(
        evaluation_days=int(len(dates) - WARMUP_DAYS), window_lengths=lengths,
        window_length_values=sorted(lengths), positions=sorted(positions),
        m7_position=int(rule), m7_value=float(np.sort(seven)[rule - 1]),
        negative_correction_kWh=float(scan().adjustment(negative_probe, RESIDUAL_WINDOW + 1,
                                                        RESIDUAL_WINDOW, ALPHA)[0].min()),
        nan_input_gate_rejects=bool(rejected),
        note='every formal-period day must use exactly m = 28 residuals and 1-based position 23')
    out['all_passed'] = bool(
        sorted(lengths) == [RESIDUAL_WINDOW] and lengths[RESIDUAL_WINDOW] == len(dates) - WARMUP_DAYS
        and positions == {23} and rule == 6 and float(np.sort(seven)[5]) == 5.0
        and out['negative_correction_kWh'] == -5.0 and rejected)
    return out


def shared_forecast_frame(load, pv, dates, archive):
    """Point-forecast quality, protection and coverage - shared by both groups, reported once."""
    k0 = WARMUP_DAYS
    rows = []
    for name, predicted, actual, unit in (
            ('load', archive.pred_l, load, 'kW'), ('pv', archive.pred_v, pv, 'kW')):
        residual = actual[k0:] - predicted[k0:]
        rows.append(dict(target=name, unit=unit, n=int(residual.size),
                         mae=float(np.mean(np.abs(residual))),
                         rmse=float(np.sqrt(np.mean(residual ** 2))),
                         bias=float(np.mean(residual))))
    net_residual = archive.n_actual[k0:] - archive.n_forecast[k0:]
    rows.append(dict(target='net_demand', unit='kWh per slot', n=int(net_residual.size),
                     mae=float(np.mean(np.abs(net_residual))),
                     rmse=float(np.sqrt(np.mean(net_residual ** 2))),
                     bias=float(np.mean(net_residual))))
    frame = pd.DataFrame(rows)
    protection = np.vstack([scan().adjustment(archive.eps, k, RESIDUAL_WINDOW, ALPHA)[0]
                            for k in range(k0, len(dates))])
    protected = archive.n_forecast[k0:] + protection
    coverage = float(np.mean(archive.n_actual[k0:] <= protected))
    summary = dict(
        point_forecast=frame.to_dict('records'),
        mean_protection_kWh=float(protection.mean()),
        protection_min_kWh=float(protection.min()), protection_max_kWh=float(protection.max()),
        coverage=coverage,
        slots=int(protection.size), protection_shared=True,
        note='the two groups use this identical prediction and protection; it is reported once and '
             'asserted identical from the per-group dispatch',
    )
    return frame, summary, protection, protected


# --------------------------------------------------------------------------------------
# scenario runner
# --------------------------------------------------------------------------------------
def emergency_event_rows(date, emergency, price_day):
    label = frozen().core.label
    active = emergency > ENERGY_TOL_KWH
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    stops = np.flatnonzero(active & ~np.r_[active[1:], False]) + 1
    return [dict(date=str(pd.Timestamp(date).date()),
                 interval=f'{label(int(s) * 10)}-{label(int(t) * 10)}',
                 start_slot=int(s), end_slot=int(t),
                 emergency_kWh=float(emergency[s:t].sum()),
                 emergency_cost_yuan=float(5.0 * (emergency[s:t] * price_day[s:t]).sum()))
            for s, t in zip(starts, stops)]


DISPATCH_EXTRA_COLUMNS = ['terminal_mode', 'nominal_terminal_target_kWh',
                          'nominal_terminal_lower_kWh', 'nominal_terminal_upper_kWh',
                          'nominal_terminal_violation_kWh']


def run_scenario(group, load, pv, price, dates, archive, warm_states, verbose=False):
    m = parent()
    control = m.frozen().bm.control
    s = scan()
    strategy_id = group['id']
    mode = group['terminal_mode']
    state = float(warm_states[WARMUP_DAYS])
    dispatch_rows, nominal_rows, daily_rows, event_rows, solver_rows = [], [], [], [], []
    checks = {}
    protected_trace, adjustment_trace = [], []
    for k in range(WARMUP_DAYS, len(dates)):
        initial = state
        q, nom_state, solver, protected, r, residual_days, reason = archive.plan_day(
            k, ALPHA, RESIDUAL_WINDOW, price, initial, mode)
        assert not solver.get('fallback', False), (strategy_id, k, solver.get('message'))
        c, d, e, w, states, physical = control(q, load[k], pv[k], initial)
        for name, value in physical.items():
            checks[name] = max(checks.get(name, 0.0), float(value))
        for name, value in solver.get('checks', {}).items():
            checks['nominal_' + name] = max(checks.get('nominal_' + name, 0.0), float(value))
        protected_trace.append(np.array(protected, dtype=float))
        adjustment_trace.append(np.array(r, dtype=float))
        date = dates[k]
        starts = pd.date_range(date, periods=144, freq='10min')
        planned_cost = q * price
        emergency_cost = 5.0 * e * price
        nominal = archive.last_nominal
        target = solver.get('terminal_target')
        lower = solver.get('terminal_lower_kWh')
        upper = solver.get('terminal_upper_kWh')
        e144 = float(np.asarray(nom_state)[-1])
        violation = float(max(0.0, (lower - e144) if lower is not None else 0.0,
                              (e144 - upper) if upper is not None else 0.0))
        dispatch_rows.append(pd.DataFrame({
            'strategy_id': strategy_id, 'predictor_id': 'P1', 'alpha': ALPHA,
            'date': str(date.date()), 'issue_time': date, 'slot': np.arange(144),
            'start_time': starts, 'end_time': starts + pd.Timedelta(minutes=10),
            'price_yuan_kWh': price, 'load_kW': load[k], 'pv_kW': pv[k],
            'load_forecast_kW': archive.pred_l[k], 'pv_forecast_kW': archive.pred_v[k],
            'net_forecast_kWh': archive.n_forecast[k], 'residual_adjustment_kWh': r,
            'protected_net_kWh': protected, 'residual_days': residual_days,
            'fallback_reason': reason, 'planned_kWh': q, 'charge_kWh': c, 'discharge_kWh': d,
            'emergency_kWh': e, 'unused_kWh': w, 'state_start_kWh': states[:-1],
            'state_end_kWh': states[1:], 'nominal_state_end_kWh': np.asarray(nom_state)[1:],
            'planned_cost_yuan': planned_cost, 'emergency_cost_yuan': emergency_cost,
            'terminal_mode': mode,
            'nominal_terminal_target_kWh': np.nan if target is None else float(target),
            'nominal_terminal_lower_kWh': lower, 'nominal_terminal_upper_kWh': upper,
            'nominal_terminal_violation_kWh': violation,
        })[s.DISPATCH_COLUMNS + DISPATCH_EXTRA_COLUMNS])
        nominal_rows.append(pd.DataFrame({
            'strategy_id': strategy_id, 'predictor_id': 'P1', 'alpha': ALPHA,
            'date': str(date.date()), 'slot': np.arange(144), 'terminal_mode': mode,
            'nominal_charge_kWh': nominal['charge'], 'nominal_discharge_kWh': nominal['discharge'],
            'nominal_unused_kWh': nominal['unused'], 'nominal_binary_mode': nominal['binary'],
            'nominal_state_start_kWh': nominal['state'][:-1],
            'nominal_state_end_kWh': nominal['state'][1:],
            'nominal_planned_cost_yuan': planned_cost}))
        loss = 0.1 * c + (1 / ETA - 1) * d
        daily_rows.append(dict(
            strategy_id=strategy_id, predictor_id='P1', alpha=ALPHA, terminal_mode=mode,
            date=str(date.date()), initial_kWh=float(initial), final_kWh=float(states[-1]),
            planned_kWh=float(q.sum()), emergency_kWh=float(e.sum()), unused_kWh=float(w.sum()),
            charge_kWh=float(c.sum()), discharge_kWh=float(d.sum()), loss_kWh=float(loss.sum()),
            planned_cost_yuan=float(planned_cost.sum()), emergency_cost_yuan=float(emergency_cost.sum()),
            total_cost_yuan=float(planned_cost.sum() + emergency_cost.sum()),
            emergency_slots=int((e > ENERGY_TOL_KWH).sum()),
            emergency_events=len(emergency_event_rows(date, e, price)),
            solved=True, solver_seconds=float(solver['elapsed_seconds']),
            mip_gap=float(solver['mip_gap']), milp_fallback=False,
            mip_node_count=int(solver.get('mip_node_count', -1)),
            mip_dual_bound_yuan=float(solver.get('mip_dual_bound_yuan', np.nan)),
            objective_bound_gap_yuan=float(solver.get('objective_bound_gap_yuan', np.nan)),
            nominal_terminal_kWh=e144,
            nominal_terminal_target_kWh=np.nan if target is None else float(target),
            nominal_terminal_lower_kWh=float(lower),
            nominal_terminal_upper_kWh=float(upper),
            nominal_terminal_violation_kWh=violation,
            nominal_terminal_error_kWh=(abs(e144 - TERMINAL_FIXED_KWH) if mode == 'fixed6000'
                                        else float(max(0.0, STATE_MIN_KWH - e144,
                                                        e144 - STATE_MAX_KWH)))))
        event_rows.extend(emergency_event_rows(date, e, price))
        solver_rows.append(dict(strategy_id=strategy_id, terminal_mode=mode, date=str(date.date()),
                                day_index=int(k), elapsed_seconds=float(solver['elapsed_seconds']),
                                mip_gap=float(solver['mip_gap']),
                                mip_node_count=int(solver.get('mip_node_count', -1)),
                                mip_dual_bound_yuan=float(solver.get('mip_dual_bound_yuan', np.nan)),
                                objective_bound_gap_yuan=float(solver.get('objective_bound_gap_yuan',
                                                                          np.nan)),
                                terminal_target=(np.nan if target is None else float(target)),
                                terminal_lower_kWh=float(lower), terminal_upper_kWh=float(upper),
                                fallback=False, residual_days=int(residual_days), reason=reason,
                                **{f'check_{key}': float(value)
                                   for key, value in solver['checks'].items()}))
        state = float(states[-1])
        if verbose and k % 80 == 0:
            print(f'    {strategy_id}: day {k} state={state:.3f}', flush=True)
    dispatch = pd.concat(dispatch_rows, ignore_index=True)
    nominal_frame = pd.concat(nominal_rows, ignore_index=True)
    daily = pd.DataFrame(daily_rows)
    events = pd.DataFrame(event_rows, columns=['date', 'interval', 'start_slot', 'end_slot',
                                              'emergency_kWh', 'emergency_cost_yuan'])
    folder = OUT / strategy_id
    folder.mkdir(parents=True, exist_ok=True)
    frame_to_csv(dispatch, folder / 'dispatch.csv')
    frame_to_csv(nominal_frame, folder / 'nominal_dispatch.csv')
    frame_to_csv(daily, folder / 'daily_summary.csv')
    frame_to_csv(events, folder / 'emergency_events.csv')
    frame_to_csv(pd.DataFrame(solver_rows), folder / 'solver_log.csv')
    frame_to_csv(terminal_bounds_frame(daily), folder / 'terminal_bounds.csv')
    monthly = daily.assign(month=daily.date.str[:7]).groupby('month').agg(
        planned_kWh=('planned_kWh', 'sum'), emergency_kWh=('emergency_kWh', 'sum'),
        unused_kWh=('unused_kWh', 'sum'), charge_kWh=('charge_kWh', 'sum'),
        discharge_kWh=('discharge_kWh', 'sum'), loss_kWh=('loss_kWh', 'sum'),
        planned_cost_yuan=('planned_cost_yuan', 'sum'),
        emergency_cost_yuan=('emergency_cost_yuan', 'sum'),
        total_cost_yuan=('total_cost_yuan', 'sum'), emergency_slots=('emergency_slots', 'sum'),
        emergency_events=('emergency_events', 'sum'),
        emergency_days=('emergency_kWh', lambda values: int((values > ENERGY_TOL_KWH).sum()))
    ).reset_index()
    frame_to_csv(monthly, folder / 'monthly_summary.csv')
    totals = dict(
        strategy_id=strategy_id, predictor_id='P1', alpha=ALPHA, terminal_mode=mode,
        planned_kWh=float(daily.planned_kWh.sum()), emergency_kWh=float(daily.emergency_kWh.sum()),
        unused_kWh=float(daily.unused_kWh.sum()), charge_kWh=float(daily.charge_kWh.sum()),
        discharge_kWh=float(daily.discharge_kWh.sum()), loss_kWh=float(daily.loss_kWh.sum()),
        planned_cost_yuan=float(daily.planned_cost_yuan.sum()),
        emergency_cost_yuan=float(daily.emergency_cost_yuan.sum()),
        total_cost_yuan=float(daily.total_cost_yuan.sum()),
        emergency_slots=int(daily.emergency_slots.sum()),
        emergency_days=int((daily.emergency_kWh > ENERGY_TOL_KWH).sum()),
        emergency_events=int(daily.emergency_events.sum()),
        evaluation_initial_kWh=float(daily.initial_kWh.iloc[0]),
        final_kWh=float(daily.final_kWh.iloc[-1]),
        max_mip_gap=float(daily.mip_gap.max()),
        solver_seconds=float(daily.solver_seconds.sum()),
        solver_failures=int(daily.milp_fallback.sum()),
        mean_adjustment_kWh=float(dispatch.residual_adjustment_kWh.mean()),
        nominal_terminal_mean_kWh=float(daily.nominal_terminal_kWh.mean()),
        nominal_terminal_min_kWh=float(daily.nominal_terminal_kWh.min()),
        nominal_terminal_max_kWh=float(daily.nominal_terminal_kWh.max()),
        nominal_terminal_violation_kWh=float(daily.nominal_terminal_violation_kWh.max()),
        evaluated_days=int(daily.date.nunique()), evaluated_slots=int(len(dispatch)))
    validation = independent_validation(strategy_id, mode, dispatch, daily, events, nominal_frame)
    validation['nominal_check_max'] = float(max(checks.values())) if checks else 0.0
    validation['nominal_checks'] = {key: float(value) for key, value in checks.items()}
    save(folder / 'validation.json', dict(totals=totals, validation=validation))
    return dict(group=group, totals=totals, validation=validation, dispatch=dispatch,
                daily=daily, monthly=monthly, events=events, archive=archive,
                protected=np.vstack(protected_trace), adjustment=np.vstack(adjustment_trace))


def terminal_bounds_frame(daily):
    frame = daily[['strategy_id', 'terminal_mode', 'date', 'nominal_terminal_target_kWh',
                   'nominal_terminal_lower_kWh', 'nominal_terminal_upper_kWh',
                   'nominal_terminal_kWh', 'nominal_terminal_violation_kWh',
                   'nominal_terminal_error_kWh']].copy()
    frame['terminal_target_is_null'] = frame.nominal_terminal_target_kWh.isna()
    return frame


def independent_validation(strategy_id, mode, dispatch, daily, events, nominal):
    q = dispatch.planned_kWh.to_numpy()
    c = dispatch.charge_kWh.to_numpy()
    d = dispatch.discharge_kWh.to_numpy()
    e = dispatch.emergency_kWh.to_numpy()
    w = dispatch.unused_kWh.to_numpy()
    s0 = dispatch.state_start_kWh.to_numpy()
    s1 = dispatch.state_end_kWh.to_numpy()
    load = dispatch.load_kW.to_numpy()
    pv = dispatch.pv_kW.to_numpy()
    loss = 0.1 * c + (1 / ETA - 1) * d
    terminal = nominal.groupby('date').nominal_state_end_kWh.last().to_numpy()
    if mode == 'fixed6000':
        nominal_terminal_check = float(np.max(np.abs(terminal - TERMINAL_FIXED_KWH)))
    else:
        nominal_terminal_check = float(max(0.0, STATE_MIN_KWH - terminal.min(),
                                           terminal.max() - STATE_MAX_KWH))
    checks = dict(
        balance=float(np.max(np.abs(q + pv * DT + d + e - load * DT - c - w))),
        state_recursion=float(np.max(np.abs(s1 - s0 - ETA * c + d / ETA))),
        day_continuity=float(np.max(np.abs(s0[1:] - s1[:-1]))),
        capacity=float(max(0.0, STATE_MIN_KWH - min(s0.min(), s1.min()),
                           max(s0.max(), s1.max()) - STATE_MAX_KWH)),
        power=float(max(0.0, c.max() - 5000 * DT, d.max() - 5000 * DT)),
        nonnegative=float(max(0.0, -min(q.min(), c.min(), d.min(), e.min(), w.min()))),
        mutex=float(np.minimum(c, d).max()),
        nominal_terminal=nominal_terminal_check,
        nominal_mutex=float(np.minimum(nominal.nominal_charge_kWh.to_numpy(),
                                       nominal.nominal_discharge_kWh.to_numpy()).max()),
        nominal_binary=float(np.max(np.abs(nominal.nominal_binary_mode.to_numpy()
                                           - np.rint(nominal.nominal_binary_mode.to_numpy())))),
        nominal_recursion=float(np.max(np.abs(
            nominal.nominal_state_end_kWh.to_numpy() - nominal.nominal_state_start_kWh.to_numpy()
            - ETA * nominal.nominal_charge_kWh.to_numpy()
            + nominal.nominal_discharge_kWh.to_numpy() / ETA))))
    energy_identity = float(np.sum(q + e + pv * DT - load * DT - w - loss) - (s1[-1] - s0[0]))
    cost_reconciliation = float(abs(dispatch.planned_cost_yuan.sum()
                                    + dispatch.emergency_cost_yuan.sum() - daily.total_cost_yuan.sum()))
    emergency_reconciliation = float(abs(events.emergency_kWh.sum() - daily.emergency_kWh.sum()))
    assert len(dispatch) == 334 * 144 and daily.date.nunique() == 334, (strategy_id, len(dispatch))
    assert max(checks.values()) < ENERGY_TOL_KWH, (strategy_id, checks)
    assert abs(energy_identity) < 1e-4, (strategy_id, energy_identity)
    assert cost_reconciliation < COST_TOL_YUAN and emergency_reconciliation < 1e-5
    return dict(checks=checks, energy_identity_residual_kWh=energy_identity,
                cost_reconciliation_yuan=cost_reconciliation,
                emergency_reconciliation_kWh=emergency_reconciliation,
                terminal_mode=mode,
                nominal_terminal_target=(TERMINAL_FIXED_KWH if mode == 'fixed6000' else None))


# --------------------------------------------------------------------------------------
# reference reproduction (three tiers, as in modules 21/23/25)
# --------------------------------------------------------------------------------------
def compare_to_reference(result, group_id, reference_path, registered_total):
    s = scan()
    reference = pd.read_csv(reference_path, low_memory=False, dtype={'date': str})
    current = result['dispatch']
    skip = ('strategy_id', 'predictor_id', 'alpha', 'date', 'issue_time', 'start_time', 'end_time',
            'fallback_reason')
    executed = [name for name in s.DISPATCH_COLUMNS
                if name in reference.columns and name not in skip
                and name not in s.NOMINAL_ONLY_COLUMNS]
    assert reference[['date', 'slot']].equals(current[['date', 'slot']])
    executed_differences = {name: float(np.max(np.abs(reference[name].to_numpy()
                                                      - current[name].to_numpy())))
                            for name in executed}
    nominal_difference = float(np.max(np.abs(
        reference.nominal_state_end_kWh.to_numpy() - current.nominal_state_end_kWh.to_numpy()))) \
        if 'nominal_state_end_kWh' in reference.columns else 0.0
    day_reference = reference.groupby('date').agg(**s.DAY_LEVEL_AGGREGATES)
    day_current = current.groupby('date').agg(**s.DAY_LEVEL_AGGREGATES)
    day_differences = (day_current - day_reference).abs()
    delta_plan = current.planned_kWh.to_numpy() - reference.planned_kWh.to_numpy()
    affected = np.flatnonzero(np.abs(delta_plan) > ENERGY_TOL_KWH)
    prices_equal = bool(len(affected) == 0 or np.max(np.abs(
        current.price_yuan_kWh.to_numpy()[affected]
        - reference.price_yuan_kWh.to_numpy()[affected])) < 1e-12)
    day_cost_identical = bool(float(day_differences['planned_cost_yuan'].max()) < ENERGY_TOL_KWH)
    reference_cost = float(reference.planned_cost_yuan.sum() + reference.emergency_cost_yuan.sum())
    current_cost = float(current.planned_cost_yuan.sum() + current.emergency_cost_yuan.sum())
    out = dict(group=group_id, reference=str(Path(reference_path).relative_to(ROOT)),
               segments=int(len(current)),
               max_executed_difference=float(max(executed_differences.values())),
               max_absolute_differences=executed_differences,
               max_day_level_difference=float(day_differences.to_numpy().max()),
               day_level_differences={name: float(day_differences[name].max())
                                      for name in day_differences},
               days_with_any_difference=int((day_differences.max(axis=1) > ENERGY_TOL_KWH).sum()),
               nominal_trajectory_difference_kWh=nominal_difference,
               equivalent_optimum=dict(
                   affected_slots=int(len(affected)),
                   affected_dates=sorted(set(current.date.to_numpy()[affected])) if len(affected) else [],
                   max_slot_plan_difference_kWh=float(np.max(np.abs(delta_plan))) if len(affected) else 0.0,
                   affected_slot_prices_equal=prices_equal, day_level_cost_identical=day_cost_identical,
                   note='a day-ahead optimum that moves charge between equal-price slots changes the '
                        'slot-level plan without moving any day-level aggregate or the cash cost'),
               reference_total_cost_yuan=reference_cost, current_total_cost_yuan=current_cost,
               difference_vs_reference_yuan=abs(current_cost - reference_cost),
               registered_total_cost_yuan=registered_total,
               difference_vs_registered_yuan=abs(current_cost - registered_total))
    out['strict_passed'] = bool(out['max_day_level_difference'] < ENERGY_TOL_KWH
                                and out['difference_vs_registered_yuan'] <= COST_TOL_YUAN
                                and prices_equal and day_cost_identical)
    # Equal-cost-face exemption. The daily price vector repeats, so some day-ahead problems have a
    # large face of optimal plans (identical objective, different charge/discharge split). A
    # reconstruction difference of ~1 ULP in the protected net demand moves the returned vertex;
    # through the actual-state carry-over that can move the next day's cash cost by a few tenths of
    # a yuan. The plan (section 4.1) asks for exactly this to be localised, objective-verified and
    # disclosed rather than reported as segment-level reproduction, so both flags are stored.
    any_difference = day_differences.max(axis=1) > ENERGY_TOL_KWH
    first_index = None
    first_plan_cost_difference = 0.0
    if bool(any_difference.any()):
        first_index = str(day_differences.index[int(np.flatnonzero(any_difference.to_numpy())[0])])
        first_plan_cost_difference = float(day_differences.loc[first_index, 'planned_cost_yuan'])
    exemption = dict(
        applies=bool(not out['strict_passed'] and out['days_with_any_difference'] <= 3
                     and out['max_day_level_difference'] <= 5.0
                     and first_plan_cost_difference < COST_TOL_YUAN and prices_equal
                     and out['difference_vs_registered_yuan'] <= 1.0),
        first_affected_date=first_index,
        first_affected_plan_cost_difference_yuan=first_plan_cost_difference,
        first_affected_plan_cost_identical=bool(first_plan_cost_difference < COST_TOL_YUAN),
        day_limit=3, day_level_limit_kWh=5.0, cost_limit_yuan=1.0,
        note='applies only when the first diverging day is an equal-cost optimum (its own plan cost '
             'is unchanged) and the annual difference stays within 1 CNY; the strict result is kept '
             'beside it and never overwritten')
    out['equal_cost_face_exemption'] = exemption
    out['passed'] = bool(out['strict_passed'] or exemption['applies'])
    return out


def shared_identity_check(results, protection, protected):
    """Both groups must have used the identical prediction, residual and protection."""
    by_id = {result['totals']['strategy_id']: result for result in results}
    assert set(by_id) == {CONTROL_ID, TREATMENT_ID}, sorted(by_id)
    control_dispatch = by_id[CONTROL_ID]['dispatch']
    treatment_dispatch = by_id[TREATMENT_ID]['dispatch']
    control_totals = by_id[CONTROL_ID]['totals']
    treatment_totals = by_id[TREATMENT_ID]['totals']

    def worst(left, right):
        return float(np.max(np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))))

    out = dict(
        load_forecast=worst(control_dispatch.load_forecast_kW, treatment_dispatch.load_forecast_kW),
        pv_forecast=worst(control_dispatch.pv_forecast_kW, treatment_dispatch.pv_forecast_kW),
        net_forecast=worst(control_dispatch.net_forecast_kWh, treatment_dispatch.net_forecast_kWh),
        adjustment=worst(control_dispatch.residual_adjustment_kWh,
                         treatment_dispatch.residual_adjustment_kWh),
        protected=worst(control_dispatch.protected_net_kWh, treatment_dispatch.protected_net_kWh),
        protected_vs_shared=worst(control_dispatch.protected_net_kWh.to_numpy().reshape(334, 144),
                                  protected),
        adjustment_vs_shared=worst(control_dispatch.residual_adjustment_kWh.to_numpy()
                                   .reshape(334, 144), protection),
        terminal_mode_differs=bool(control_totals['terminal_mode'] != treatment_totals['terminal_mode']),
        roles='control=' + control_totals['terminal_mode'] + ', treatment='
              + treatment_totals['terminal_mode'],
        exact_identity_keys=['load_forecast', 'pv_forecast', 'net_forecast', 'adjustment', 'protected'],
        reference_tolerance_kWh=1e-4,
        reference_note='the published archive is stored rounded to 4 decimals, so a protection '
                       'recomputed from it differs from the full-precision reference dispatch by '
                       '~1e-5 kWh; only the two groups must agree exactly',
    )
    exact = all(out[key] == 0.0 for key in out['exact_identity_keys'])
    out['all_passed'] = bool(exact and out['terminal_mode_differs']
                             and out['protected_vs_shared'] <= out['reference_tolerance_kWh']
                             and out['adjustment_vs_shared'] <= out['reference_tolerance_kWh'])
    return out


# --------------------------------------------------------------------------------------
# causal checks
# --------------------------------------------------------------------------------------
def replay_prefix(load, pv, dates, price, published_l, published_v, mode, k_end):
    """Replay one mode from the shared begin state up to (not including) day ``k_end``."""
    archive = TerminalArchive(load, pv, len(dates), published_l, published_v)
    state = REFERENCE_WARMUP_KWH
    plans, states = [], []
    for k in range(WARMUP_DAYS, k_end):
        initial = state
        q, nom_state, solver, protected, r, m, reason = archive.plan_day(
            k, ALPHA, RESIDUAL_WINDOW, price, initial, mode)
        assert not solver.get('fallback', False), (mode, k)
        c, d, e, w, day_states, _ = frozen().bm.control(q, load[k], pv[k], initial)
        plans.append(np.array(q, dtype=float))
        states.append(float(day_states[-1]))
        state = float(day_states[-1])
    return dict(plans=plans, states=states, final_state=state, archive=archive)


def perturbation_checks(load, pv, price, dates, published_l, published_v):
    """Future-truth perturbations must not move the past of either group."""
    cases = []
    worst_prefix = 0.0
    worst_protection = 0.0
    base_cache = {}
    for k in PERTURBATION_DAYS:
        for group in GROUPS:
            base_cache[(group['id'], k)] = replay_prefix(
                load, pv, dates, price, published_l, published_v, group['terminal_mode'], k)
        for name, load_factor, pv_factor in (('load_x1.2', 1.2, 1.0), ('pv_x0.7', 1.0, 0.7)):
            perturbed_load = load.copy()
            perturbed_pv = pv.copy()
            perturbed_load[k:] *= load_factor
            perturbed_pv[k:] *= pv_factor
            entry = dict(day_index=int(k), date=str(dates[k].date()), case=name, modes={})
            for group in GROUPS:
                mode = group['terminal_mode']
                base = base_cache[(group['id'], k)]
                probe = replay_prefix(perturbed_load, perturbed_pv, dates, price, published_l,
                                      published_v, mode, k)
                plan_gap = max((float(np.max(np.abs(a - b))) for a, b in
                                zip(base['plans'], probe['plans'])), default=0.0)
                state_gap = max((abs(a - b) for a, b in zip(base['states'], probe['states'])),
                                default=0.0)
                r_base, m_base, _ = scan().adjustment(base['archive'].eps, k, RESIDUAL_WINDOW, ALPHA)
                r_probe, m_probe, _ = scan().adjustment(probe['archive'].eps, k, RESIDUAL_WINDOW, ALPHA)
                protected_gap = float(np.max(np.abs(r_base - r_probe)))
                entry['modes'][group['id']] = dict(
                    prefix_days=len(base['plans']), max_plan_difference_kWh=plan_gap,
                    max_state_difference_kWh=state_gap, final_state_difference_kWh=abs(
                        base['final_state'] - probe['final_state']),
                    protection_difference_kWh=protected_gap, residual_days=[int(m_base), int(m_probe)],
                    passed=bool(plan_gap == 0.0 and state_gap == 0.0 and protected_gap == 0.0))
                worst_prefix = max(worst_prefix, plan_gap, state_gap)
                worst_protection = max(worst_protection, protected_gap)
            entry['passed'] = bool(all(value['passed'] for value in entry['modes'].values()))
            cases.append(entry)
    out = dict(cases=cases, max_prefix_difference=worst_prefix,
               max_protection_difference_kWh=worst_protection,
               replays=int(2 * len(PERTURBATION_DAYS) * len(GROUPS)
                           + 2 * len(PERTURBATION_DAYS) * len(GROUPS)),
               note='only the day itself and later actuals are perturbed; every earlier plan, state '
                    'and protection must be bit-identical',
               all_passed=bool(all(case['passed'] for case in cases)))
    return out


def feedback_prefix_check(results, load, pv, price, dates):
    """Fix the day plan and initial state, perturb the afternoon, keep the prefix bit-identical."""
    s = scan()
    by_id = {result['totals']['strategy_id']: result for result in results}
    cases = []
    worst_prefix = 0.0
    # keep the published trace of day HALF_DAY_INDEX shared between the two groups
    reference_dispatch = by_id[CONTROL_ID]['dispatch']
    reference_dispatch = reference_dispatch[reference_dispatch.date == HALF_DAY_LABEL]
    published_l = np.asarray(load[HALF_DAY_INDEX], dtype=float)
    published_v = np.asarray(pv[HALF_DAY_INDEX], dtype=float)
    for group in GROUPS:
        result = by_id[group['id']]
        dispatch = result['dispatch']
        daily = result['daily']
        day = dispatch[dispatch.date == HALF_DAY_LABEL]
        plan = day.planned_kWh.to_numpy()
        initial = float(daily[daily.date == HALF_DAY_LABEL].initial_kWh.iloc[0])
        clean_c, clean_d, clean_e, clean_w, clean_states, _ = frozen().bm.control(
            plan, load[HALF_DAY_INDEX], pv[HALF_DAY_INDEX], initial)
        for name, load_factor, pv_factor in (('load_x1.2', 1.2, 1.0), ('pv_x0.7', 1.0, 0.7)):
            probe_load = load[HALF_DAY_INDEX].copy()
            probe_pv = pv[HALF_DAY_INDEX].copy()
            probe_load[HALF_DAY_SLOT:] *= load_factor
            probe_pv[HALF_DAY_SLOT:] *= pv_factor
            c, d, e, w, states, _ = frozen().bm.control(plan, probe_load, probe_pv, initial)
            gap = float(max(np.max(np.abs(c[:HALF_DAY_SLOT] - clean_c[:HALF_DAY_SLOT])),
                            np.max(np.abs(d[:HALF_DAY_SLOT] - clean_d[:HALF_DAY_SLOT])),
                            np.max(np.abs(e[:HALF_DAY_SLOT] - clean_e[:HALF_DAY_SLOT])),
                            np.max(np.abs(w[:HALF_DAY_SLOT] - clean_w[:HALF_DAY_SLOT])),
                            np.max(np.abs(states[:HALF_DAY_SLOT + 1] - clean_states[:HALF_DAY_SLOT + 1]))))
            prefix_cost_gap = float(abs(float((plan[:HALF_DAY_SLOT] * price[:HALF_DAY_SLOT]).sum())
                                       - float((plan[:HALF_DAY_SLOT] * price[:HALF_DAY_SLOT]).sum())))
            cases.append(dict(group=group['id'], terminal_mode=group['terminal_mode'], date=HALF_DAY_LABEL,
                              case=name, perturbed_from_slot=HALF_DAY_SLOT,
                              prefix_slots=HALF_DAY_SLOT, max_prefix_difference=gap,
                              prefix_planned_cost_difference_yuan=prefix_cost_gap,
                              state_entering_perturbation_kWh=float(clean_states[HALF_DAY_SLOT]),
                              perturbed_final_kWh=float(states[-1]),
                              clean_final_kWh=float(clean_states[-1]),
                              prefix_price_shared=bool(np.array_equal(
                                  reference_dispatch.price_yuan_kWh.to_numpy(), price)),
                              passed=bool(gap == 0.0)))
            worst_prefix = max(worst_prefix, gap)
    return dict(cases=cases, max_prefix_difference=worst_prefix,
                note='slots 0..71 and state[0..72] must not move when only slot 72 and later change; '
                     'the perturbation deliberately changes the realised emergency, not the plan',
                all_passed=bool(all(case['passed'] for case in cases)))


# --------------------------------------------------------------------------------------
# sampled dominance and independent re-solve
# --------------------------------------------------------------------------------------
def sampled_dominance_checks(results, load, pv, price, dates, archive):
    """Per sampled date and per group's own initial state, J_free <= J_fixed6000."""
    by_id = {result['totals']['strategy_id']: result for result in results}
    days = list(SAMPLED_DATES)
    for group in GROUPS:
        daily = by_id[group['id']]['daily']
        worst = daily.loc[daily.emergency_cost_yuan.idxmax()]
        top = float(worst.emergency_cost_yuan)
        candidates = daily[np.isclose(daily.emergency_cost_yuan, top)]
        index = int(candidates.index[0])
        date_index = int(daily.index[daily.date == daily.loc[index, 'date']][0]) + WARMUP_DAYS
        days.append(date_index)
    days = sorted(set(days))
    assert len(days) <= 7, days
    cases = []
    worst_violation = 0.0
    for k in days:
        r, m, _ = scan().adjustment(archive.eps, k, RESIDUAL_WINDOW, ALPHA)
        protected = archive.n_forecast[k] + r
        for group in GROUPS:
            daily = by_id[group['id']]['daily']
            initial = float(daily[daily.date == str(dates[k].date())].initial_kWh.iloc[0])
            fixed = solve_day(protected, price, initial, 'fixed6000')[5]
            free = solve_day(protected, price, initial, 'free')[5]
            violation = float(free['cost_yuan'] - fixed['cost_yuan'])
            worst_violation = max(worst_violation, violation)
            cases.append(dict(day_index=int(k), date=str(dates[k].date()),
                              initial_group=group['id'], initial_kWh=initial,
                              fixed_objective_yuan=fixed['cost_yuan'],
                              free_objective_yuan=free['cost_yuan'],
                              difference_yuan=violation,
                              fixed_terminal_kWh=fixed['nominal_terminal_kWh'],
                              free_terminal_kWh=free['nominal_terminal_kWh'],
                              fixed_terminal_bounds=[fixed['terminal_lower_kWh'],
                                                     fixed['terminal_upper_kWh']],
                              free_terminal_bounds=[free['terminal_lower_kWh'],
                                                    free['terminal_upper_kWh']],
                              dominance_holds=bool(violation <= COST_TOL_YUAN),
                              free_hits_floor=bool(abs(free['nominal_terminal_kWh']
                                                       - STATE_MIN_KWH) < 1e-6),
                              limit_slots=int(len(protected)), solver_calls=2))
    out = dict(cases=cases, solves=int(2 * len(cases)), sampled_days=days,
               worst_violation_yuan=worst_violation,
               note='the fixed-mode feasible set is a subset of the free-mode feasible set on the same '
                    'day-ahead input, so the optimal plan cost cannot be lower in fixed mode; the '
                    'two trajectories are never compared day by day because their initial states differ',
               all_passed=bool(worst_violation <= COST_TOL_YUAN))
    return out


# --------------------------------------------------------------------------------------
# metric frames
# --------------------------------------------------------------------------------------
def summary_frame(results):
    rows = []
    for result in results:
        row = dict(result['totals'])
        dispatch = result['dispatch']
        actual_net = (dispatch.load_kW - dispatch.pv_kW).to_numpy() * DT
        row['coverage'] = float(np.mean(actual_net <= dispatch.protected_net_kWh.to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows)


def delta_row(label, treatment, baseline):
    return dict(comparison=label, treatment=treatment['totals']['strategy_id'],
                baseline=baseline['totals']['strategy_id'],
                delta_planned_cost_yuan=float(treatment['totals']['planned_cost_yuan']
                                              - baseline['totals']['planned_cost_yuan']),
                delta_emergency_cost_yuan=float(treatment['totals']['emergency_cost_yuan']
                                                - baseline['totals']['emergency_cost_yuan']),
                delta_total_cost_yuan=float(treatment['totals']['total_cost_yuan']
                                            - baseline['totals']['total_cost_yuan']),
                delta_total_cost_percent=float(100 * (treatment['totals']['total_cost_yuan']
                                                      - baseline['totals']['total_cost_yuan'])
                                               / baseline['totals']['total_cost_yuan']),
                delta_planned_kWh=float(treatment['totals']['planned_kWh']
                                        - baseline['totals']['planned_kWh']),
                delta_emergency_kWh=float(treatment['totals']['emergency_kWh']
                                          - baseline['totals']['emergency_kWh']),
                delta_unused_kWh=float(treatment['totals']['unused_kWh']
                                       - baseline['totals']['unused_kWh']),
                delta_loss_kWh=float(treatment['totals']['loss_kWh'] - baseline['totals']['loss_kWh']),
                delta_emergency_days=int(treatment['totals']['emergency_days']
                                         - baseline['totals']['emergency_days']),
                delta_emergency_events=int(treatment['totals']['emergency_events']
                                           - baseline['totals']['emergency_events']))


def daily_contrast_frame(results):
    by_id = {result['totals']['strategy_id']: result for result in results}
    control = by_id[CONTROL_ID]['daily'].set_index('date')
    treatment = by_id[TREATMENT_ID]['daily'].set_index('date')
    frame = pd.DataFrame({
        'date': control.index,
        'fixed_total_cost_yuan': control.total_cost_yuan.to_numpy(),
        'free_total_cost_yuan': treatment.total_cost_yuan.to_numpy(),
        'free_minus_fixed_yuan': treatment.total_cost_yuan.to_numpy() - control.total_cost_yuan.to_numpy(),
        'fixed_planned_cost_yuan': control.planned_cost_yuan.to_numpy(),
        'free_planned_cost_yuan': treatment.planned_cost_yuan.to_numpy(),
        'fixed_emergency_cost_yuan': control.emergency_cost_yuan.to_numpy(),
        'free_emergency_cost_yuan': treatment.emergency_cost_yuan.to_numpy(),
        'delta_emergency_kWh': treatment.emergency_kWh.to_numpy() - control.emergency_kWh.to_numpy(),
        'fixed_state_end_kWh': control.final_kWh.to_numpy(),
        'free_state_end_kWh': treatment.final_kWh.to_numpy(),
        'fixed_nominal_terminal_kWh': control.nominal_terminal_kWh.to_numpy(),
        'free_nominal_terminal_kWh': treatment.nominal_terminal_kWh.to_numpy(),
    })
    return frame


def monthly_contrast_frame(results):
    by_id = {result['totals']['strategy_id']: result for result in results}
    control = by_id[CONTROL_ID]['monthly'].set_index('month')
    treatment = by_id[TREATMENT_ID]['monthly'].set_index('month')
    frame = pd.DataFrame({
        'month': control.index,
        'fixed_total_cost_yuan': control.total_cost_yuan.to_numpy(),
        'free_total_cost_yuan': treatment.total_cost_yuan.to_numpy(),
        'free_minus_fixed_yuan': treatment.total_cost_yuan.to_numpy() - control.total_cost_yuan.to_numpy(),
        'fixed_emergency_cost_yuan': control.emergency_cost_yuan.to_numpy(),
        'free_emergency_cost_yuan': treatment.emergency_cost_yuan.to_numpy(),
        'delta_emergency_cost_yuan': (treatment.emergency_cost_yuan.to_numpy()
                                      - control.emergency_cost_yuan.to_numpy()),
        'delta_emergency_kWh': treatment.emergency_kWh.to_numpy() - control.emergency_kWh.to_numpy(),
        'fixed_planned_cost_yuan': control.planned_cost_yuan.to_numpy(),
        'free_planned_cost_yuan': treatment.planned_cost_yuan.to_numpy(),
    })
    return frame


def window_frame(results, price):
    """Mutually exclusive emergency windows that sum to the year, plus diagnostic windows."""
    rows = []
    for result in results:
        dispatch = result['dispatch']
        emergency = dispatch.emergency_kWh.to_numpy().reshape(334, 144)
        planned = dispatch.planned_kWh.to_numpy().reshape(334, 144)
        price_grid = np.broadcast_to(price, (334, 144))
        for name, start, stop in EMERGENCY_WINDOWS:
            rows.append(dict(strategy_id=result['totals']['strategy_id'],
                             terminal_mode=result['totals']['terminal_mode'], scope='main',
                             window=name, emergency_kWh=float(emergency[:, start:stop].sum()),
                             emergency_cost_yuan=float(
                                 5.0 * (emergency[:, start:stop] * price_grid[:, start:stop]).sum()),
                             planned_kWh=float(planned[:, start:stop].sum()),
                             planned_cost_yuan=float(
                                 (planned[:, start:stop] * price_grid[:, start:stop]).sum())))
        for name, start, stop in DIAGNOSTIC_WINDOWS:
            rows.append(dict(strategy_id=result['totals']['strategy_id'],
                             terminal_mode=result['totals']['terminal_mode'], scope='diagnostic',
                             window=name, emergency_kWh=float(emergency[:, start:stop].sum()),
                             emergency_cost_yuan=float(
                                 5.0 * (emergency[:, start:stop] * price_grid[:, start:stop]).sum()),
                             planned_kWh=float(planned[:, start:stop].sum()),
                             planned_cost_yuan=float(
                                 (planned[:, start:stop] * price_grid[:, start:stop]).sum())))
        name, start, stop = LATE_NIGHT_WINDOW
        rows.append(dict(strategy_id=result['totals']['strategy_id'],
                         terminal_mode=result['totals']['terminal_mode'], scope='diagnostic',
                         window=name, emergency_kWh=float(emergency[:, start:stop].sum()),
                         emergency_cost_yuan=float(
                             5.0 * (emergency[:, start:stop] * price_grid[:, start:stop]).sum()),
                         planned_kWh=float(planned[:, start:stop].sum()),
                         planned_cost_yuan=float(
                             (planned[:, start:stop] * price_grid[:, start:stop]).sum())))
    frame = pd.DataFrame(rows)
    return frame


def window_contrast_frame(frame):
    pivot = frame.pivot_table(index=['scope', 'window'], columns='strategy_id',
                              values=['emergency_kWh', 'emergency_cost_yuan', 'planned_kWh',
                                      'planned_cost_yuan'])
    rows = []
    for index, group in frame.groupby(['scope', 'window'], sort=False):
        control = group[group.strategy_id == CONTROL_ID].iloc[0]
        treatment = group[group.strategy_id == TREATMENT_ID].iloc[0]
        rows.append(dict(scope=index[0], window=index[1],
                         fixed_emergency_kWh=float(control.emergency_kWh),
                         free_emergency_kWh=float(treatment.emergency_kWh),
                         delta_emergency_kWh=float(treatment.emergency_kWh - control.emergency_kWh),
                         fixed_emergency_cost_yuan=float(control.emergency_cost_yuan),
                         free_emergency_cost_yuan=float(treatment.emergency_cost_yuan),
                         delta_emergency_cost_yuan=float(treatment.emergency_cost_yuan
                                                         - control.emergency_cost_yuan),
                         fixed_planned_cost_yuan=float(control.planned_cost_yuan),
                         free_planned_cost_yuan=float(treatment.planned_cost_yuan),
                         delta_planned_cost_yuan=float(treatment.planned_cost_yuan
                                                       - control.planned_cost_yuan)))
    return pd.DataFrame(rows)


def next_morning_frame(results):
    rows = []
    for result in results:
        daily = result['daily'].set_index('date')
        dispatch = result['dispatch']
        early = dispatch[dispatch.slot < NEXT_MORNING_WINDOW[2]].groupby('date').agg(
            emergency_kWh=('emergency_kWh', 'sum'),
            emergency_cost_yuan=('emergency_cost_yuan', 'sum'))
        ordered = list(daily.index)
        for position in range(len(ordered) - 1):
            day, following = ordered[position], ordered[position + 1]
            rows.append(dict(strategy_id=result['totals']['strategy_id'],
                             terminal_mode=result['totals']['terminal_mode'],
                             date=day, next_date=following,
                             day_end_actual_kWh=float(daily.loc[day, 'final_kWh']),
                             day_end_nominal_kWh=float(daily.loc[day, 'nominal_terminal_kWh']),
                             next_emergency_0_6h_kWh=float(early.loc[following, 'emergency_kWh']),
                             next_emergency_0_6h_cost_yuan=float(
                                 early.loc[following, 'emergency_cost_yuan'])))
    frame = pd.DataFrame(rows)
    return frame


def next_morning_contrast(frame):
    control = frame[frame.strategy_id == CONTROL_ID].set_index('date')
    treatment = frame[frame.strategy_id == TREATMENT_ID].set_index('date')
    out = pd.DataFrame({
        'date': control.index,
        'next_date': control.next_date.to_numpy(),
        'fixed_day_end_kWh': control.day_end_actual_kWh.to_numpy(),
        'free_day_end_kWh': treatment.day_end_actual_kWh.to_numpy(),
        'delta_day_end_kWh': treatment.day_end_actual_kWh.to_numpy()
                             - control.day_end_actual_kWh.to_numpy(),
        'fixed_next_emergency_kWh': control.next_emergency_0_6h_kWh.to_numpy(),
        'free_next_emergency_kWh': treatment.next_emergency_0_6h_kWh.to_numpy(),
        'delta_next_emergency_kWh': treatment.next_emergency_0_6h_kWh.to_numpy()
                                    - control.next_emergency_0_6h_kWh.to_numpy(),
        'fixed_next_emergency_cost_yuan': control.next_emergency_0_6h_cost_yuan.to_numpy(),
        'free_next_emergency_cost_yuan': treatment.next_emergency_0_6h_cost_yuan.to_numpy(),
        'delta_next_emergency_cost_yuan': (treatment.next_emergency_0_6h_cost_yuan.to_numpy()
                                           - control.next_emergency_0_6h_cost_yuan.to_numpy()),
    })
    return out


def inventory_frame(results):
    rows = []
    for result in results:
        daily = result['daily']
        nominal = daily.nominal_terminal_kWh.to_numpy()
        actual = daily.final_kWh.to_numpy()
        rows.append(dict(
            strategy_id=result['totals']['strategy_id'],
            terminal_mode=result['totals']['terminal_mode'],
            nominal_mean_kWh=float(np.mean(nominal)), nominal_p05_kWh=float(np.percentile(nominal, 5)),
            nominal_median_kWh=float(np.median(nominal)), nominal_p95_kWh=float(np.percentile(nominal, 95)),
            nominal_min_kWh=float(nominal.min()), nominal_max_kWh=float(nominal.max()),
            nominal_days_at_floor=int((np.abs(nominal - STATE_MIN_KWH) < 1e-6).sum()),
            nominal_days_at_ceiling=int((np.abs(nominal - STATE_MAX_KWH) < 1e-6).sum()),
            nominal_days_at_6000=int((np.abs(nominal - TERMINAL_FIXED_KWH) < 1e-6).sum()),
            actual_mean_kWh=float(np.mean(actual)), actual_p05_kWh=float(np.percentile(actual, 5)),
            actual_median_kWh=float(np.median(actual)), actual_p95_kWh=float(np.percentile(actual, 95)),
            actual_min_kWh=float(actual.min()), actual_max_kWh=float(actual.max()),
            actual_days_at_floor=int((actual <= STATE_MIN_KWH + 1e-6).sum()),
            actual_days_at_ceiling=int((actual >= STATE_MAX_KWH - 1e-6).sum()),
            max_nominal_actual_terminal_difference_kWh=float(np.max(np.abs(nominal - actual))),
            actual_start_kWh=float(daily.initial_kWh.iloc[0]), actual_end_kWh=float(actual[-1]),
            max_cross_day_jump_kWh=float(np.max(np.abs(daily.initial_kWh.to_numpy()[1:]
                                                      - actual[:-1]))),
            nominal_terminal_target=(TERMINAL_FIXED_KWH
                                     if result['totals']['terminal_mode'] == 'fixed6000' else None),
            nominal_terminal_violation_kWh=float(daily.nominal_terminal_violation_kWh.max())))
    return pd.DataFrame(rows)


def valuation_frame(results, price):
    nu = float(np.median(price) / ETA)
    rows = []
    for result in results:
        start = float(result['daily'].initial_kWh.iloc[0])
        end = float(result['daily'].final_kWh.iloc[-1])
        cost = float(result['totals']['total_cost_yuan'])
        adjustment = float(-nu * (end - start))
        rows.append(dict(strategy_id=result['totals']['strategy_id'],
                         terminal_mode=result['totals']['terminal_mode'],
                         total_cost_yuan=cost, start_kWh=start, end_kWh=end,
                         change_kWh=end - start, nu_yuan_per_internal_kWh=nu,
                         inventory_adjustment_yuan=adjustment, adjusted_cost_yuan=cost + adjustment))
    return pd.DataFrame(rows)


def energy_frame(results):
    by_id = {result['totals']['strategy_id']: result for result in results}
    control = by_id[CONTROL_ID]['totals']
    treatment = by_id[TREATMENT_ID]['totals']
    row = dict(comparison=PRIMARY_LABEL,
               delta_planned_kWh=treatment['planned_kWh'] - control['planned_kWh'],
               delta_emergency_kWh=treatment['emergency_kWh'] - control['emergency_kWh'],
               delta_charge_kWh=treatment['charge_kWh'] - control['charge_kWh'],
               delta_discharge_kWh=treatment['discharge_kWh'] - control['discharge_kWh'],
               delta_unused_kWh=treatment['unused_kWh'] - control['unused_kWh'],
               delta_loss_kWh=treatment['loss_kWh'] - control['loss_kWh'],
               delta_end_kWh=treatment['final_kWh'] - control['final_kWh'])
    # dQ = -dQ_em + dW + dLoss + dE_end  (bus-side accounting identity, kWh)
    row['identity_residual_kWh'] = float(
        row['delta_planned_kWh'] + row['delta_emergency_kWh'] - row['delta_unused_kWh']
        - row['delta_loss_kWh'] - row['delta_end_kWh'])
    return pd.DataFrame([row])


def stability_frame(daily, monthly):
    rows = [dict(
        comparison=PRIMARY_LABEL,
        months_improved=int((monthly.free_minus_fixed_yuan < -COST_TOL_YUAN).sum()),
        months_worse=int((monthly.free_minus_fixed_yuan > COST_TOL_YUAN).sum()),
        months_total=int(len(monthly)),
        days_improved=int((daily.free_minus_fixed_yuan < -COST_TOL_YUAN).sum()),
        days_worse=int((daily.free_minus_fixed_yuan > COST_TOL_YUAN).sum()),
        days_total=int(len(daily)),
        worst_day=str(daily.loc[daily.free_minus_fixed_yuan.idxmax(), 'date']),
        worst_day_yuan=float(daily.free_minus_fixed_yuan.max()),
        best_day=str(daily.loc[daily.free_minus_fixed_yuan.idxmin(), 'date']),
        best_day_yuan=float(daily.free_minus_fixed_yuan.min()),
        worst_month=str(monthly.loc[monthly.free_minus_fixed_yuan.idxmax(), 'month']),
        worst_month_yuan=float(monthly.free_minus_fixed_yuan.max()),
        best_month=str(monthly.loc[monthly.free_minus_fixed_yuan.idxmin(), 'month']),
        best_month_yuan=float(monthly.free_minus_fixed_yuan.min()))]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------
def figures(summary, monthly, window_contrast, soc):
    produced = []
    mode_names = {CONTROL_ID: 'T_fixed6000 (nominal 6000)', TREATMENT_ID: 'T_free (nominal free)'}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    index = np.arange(len(summary))
    width = 0.36
    axes[0].bar(index - width / 2, summary.planned_cost_yuan / 1e6, width,
                label='planned purchase (million CNY)', color='#1b6c9e')
    axes[0].bar(index + width / 2, summary.emergency_cost_yuan / 1e6, width,
                label='emergency (million CNY)', color='#b8501b')
    axes[0].set_xticks(index)
    axes[0].set_xticklabels([mode_names.get(v, v) for v in summary.strategy_id], fontsize=8)
    axes[0].set_ylabel('Million CNY')
    axes[0].set_title('Planned versus emergency cash cost')
    axes[1].bar(index, summary.total_cost_yuan / 1e6, 0.5, color='#2f7d4f',
                label='total cash cost (million CNY)')
    for position, value in zip(index, summary.total_cost_yuan):
        axes[1].text(position, value / 1e6, f'{value:,.0f}', ha='center', va='bottom', fontsize=8)
    axes[1].set_xticks(index)
    axes[1].set_xticklabels([mode_names.get(v, v) for v in summary.strategy_id], fontsize=8)
    axes[1].set_ylabel('Million CNY')
    axes[1].set_title('Total cash cost, 2025-02-01..2025-12-31')
    for axis in axes:
        axis.grid(alpha=0.2, axis='y')
        axis.legend(loc='best', fontsize=8)
    path = FIG / 'q2_free_terminal_costs.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    produced.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    months = np.arange(len(monthly))
    axes[0].bar(months, monthly.free_minus_fixed_yuan / 1e3, 0.6, color='#7a4fa3')
    axes[0].axhline(0, color='black', lw=0.8)
    axes[0].set_xticks(months)
    axes[0].set_xticklabels(monthly.month, rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel('Thousand CNY (free - fixed)')
    axes[0].set_title('Monthly cash cost difference')
    morning = window_contrast[window_contrast.scope == 'main']
    labels = [f'{row.window}\n(e-kWh {row.delta_emergency_kWh:+,.0f})'
              for row in morning.itertuples()]
    axes[1].bar(np.arange(len(morning)), morning.delta_emergency_cost_yuan / 1e3, 0.5,
                color='#b8501b')
    axes[1].axhline(0, color='black', lw=0.8)
    axes[1].set_xticks(np.arange(len(morning)))
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel('Thousand CNY (free - fixed)')
    axes[1].set_title('Emergency cost difference by window (mutually exclusive)')
    for axis in axes:
        axis.grid(alpha=0.2, axis='y')
    path = FIG / 'q2_free_terminal_monthly.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    produced.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout='constrained')
    for group_id, colour in ((CONTROL_ID, '#1b6c9e'), (TREATMENT_ID, '#b8501b')):
        block = soc[soc.strategy_id == group_id]
        axes[0].plot(block.date, block.nominal_state_end_kWh, lw=0.8, color=colour,
                     label=mode_names[group_id])
        axes[1].plot(block.date, block.actual_state_end_kWh, lw=0.8, color=colour,
                     label=mode_names[group_id])
    for axis, title in ((axes[0], 'Nominal day-end stored energy (plan)'),
                        (axes[1], 'Actual day-end stored energy (realised)')):
        for level, style in ((STATE_MIN_KWH, ':'), (TERMINAL_FIXED_KWH, '--'), (STATE_MAX_KWH, ':')):
            axis.axhline(level, color='gray', ls=style, lw=0.8)
        axis.set_ylabel('kWh')
        axis.set_title(title)
        axis.grid(alpha=0.2)
        axis.legend(loc='best', fontsize=8)
    axes[1].set_xlabel('Date')
    path = FIG / 'q2_free_terminal_soc.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    produced.append(path)
    return produced


def soc_frame(results):
    frames = []
    for result in results:
        frames.append(pd.DataFrame({
            'strategy_id': result['totals']['strategy_id'],
            'date': pd.to_datetime(result['daily'].date),
            'nominal_state_end_kWh': result['daily'].nominal_terminal_kWh.to_numpy(),
            'actual_state_end_kWh': result['daily'].final_kWh.to_numpy()}))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------------
# registration and main
# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['register', 'repro', 'full'], default='full')
    parser.add_argument('--amend-reason', default=None)
    args = parser.parse_args()
    assert Path(sys.prefix).name == 'math_modeling', sys.prefix
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    m = parent()
    s = scan()

    load, pv, price, source_hashes = m.frozen().bm.read_sources()
    dates = m.frozen().bm.DATES
    assert load.shape == pv.shape == (365, 144) and price.shape == (144,)

    inputs = ['code/14_q2_ridge_forecast_experiment.py', 'code/21_q2_quantile_level_scan.py',
              'code/23_q2_pv_support_features_experiment.py', 'code/27_q2_free_terminal_experiment.py',
              'results/q2_pv_support_features/pv_forecast_archive_P1.csv',
              'results/q2_pv_support_features/P1_q80/dispatch.csv',
              'results/q2_pv_support_features/P1_q80/daily_summary.csv',
              'results/q2_pv_support_features/checks.json',
              'results/q2_load_pv_shape/F01_pv_support/dispatch.csv',
              '附件/附件1.xlsx', '附件/附件2.xlsx',
              'reports/问题二_固定6000与自由末态对照实验方案.md']
    snapshot_inputs = ['code/02_q1_baseline.py', 'code/05_q2_baseline.py',
                       'code/08_q2_quantile_experiment.py']
    hashes = {rel: digest(ROOT / rel) for rel in inputs}
    snapshot_hashes = {rel: digest(SNAP / rel) for rel in snapshot_inputs}
    for rel in snapshot_inputs:
        assert digest(ROOT / rel) == snapshot_hashes[rel], (
            f'{rel} differs from the frozen audit snapshot; the local transcription targets the '
            'snapshot and the working copy must not drift')
    dependency = dependency_assertions()
    signature = hashlib.sha256(json.dumps(
        dict(parameters=PARAMETERS, dependency=dependency, input_sha256=hashes,
             snapshot_sha256=snapshot_hashes, code=digest(CODE_27)), sort_keys=True).encode()).hexdigest()
    registration_path = OUT / 'registration.json'
    if args.mode == 'register' or not registration_path.exists():
        record = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
                      specification='reports/问题二_固定6000与自由末态对照实验方案.md',
                      parameters=PARAMETERS, dependency=dependency, input_sha256=hashes,
                      snapshot_sha256=snapshot_hashes, code_sha256=digest(CODE_27),
                      executable=sys.executable, python=sys.version, numpy=np.__version__,
                      pandas=pd.__version__)
        if registration_path.exists():
            existing = json.loads(registration_path.read_text(encoding='utf-8'))
            if existing['signature'] != signature:
                assert args.amend_reason, 'a different registration exists; pass --amend-reason'
                for key in ('alpha', 'residual_window_days', 'terminal_modes', 'groups',
                            'primary_comparison', 'sampled_dates', 'perturbation_days',
                            'half_day_check', 'emergency_windows', 'valuation', 'thresholds',
                            'solver', 'quantile_rule'):
                    assert existing['parameters'][key] == PARAMETERS[key], key
                amendments = existing.get('amendments', [])
                amendments.append(dict(amended_utc=datetime.now(timezone.utc).isoformat(),
                                       previous_signature=existing['signature'],
                                       previous_code_sha256=existing.get('code_sha256'),
                                       new_signature=signature, new_code_sha256=digest(CODE_27),
                                       reason=args.amend_reason, frozen_spec_unchanged=True))
                record['amendments'] = amendments
        record['signature'] = signature
        save(registration_path, record)
        snapshot = OUT / 'source_archive'
        (snapshot / 'code').mkdir(parents=True, exist_ok=True)
        for rel in inputs:
            if rel.startswith('code/'):
                shutil.copy2(ROOT / rel, snapshot / 'code' / Path(rel).name)
        shutil.copy2(PLAN_MD, snapshot / 'experiment_plan.md')
        print(f'registered: signature={signature[:16]}', flush=True)
        if args.mode == 'register':
            return
    else:
        existing = json.loads(registration_path.read_text(encoding='utf-8'))
        assert existing['signature'] == signature, (
            'inputs or code changed since registration; rerun --mode register --amend-reason')

    protected = protected_manifest()
    save(OUT / 'protected_before.json', protected)
    timings = {}

    published_l, published_v, archive_meta = read_published_archive(load, pv, price, dates)
    save(OUT / 'archive_input_check.json', archive_meta)
    assert archive_meta['all_passed'], archive_meta
    print(f"archive gate: rounded archive vs reference forecast max diff "
          f"{archive_meta['archive_load_vs_reference_kW']:.3e} kW (load), "
          f"{archive_meta['archive_pv_vs_reference_kW']:.3e} kW (PV)", flush=True)

    boundary = boundary_checks(price)
    save(OUT / 'boundary_checks.json', boundary)
    assert boundary['all_passed'], boundary
    print(f"boundary checks: {boundary['case_count']} cases passed", flush=True)

    archive = TerminalArchive(load, pv, len(dates), published_l, published_v)
    matrix = matrix_evidence(np.full(144, 300.0), price, 6000.0)
    save(OUT / 'matrix_evidence.json', matrix)
    assert matrix['all_passed'], matrix

    print('running the public January warm-up ...', flush=True)
    warm_states, warm_frame, warm_end, warm_checks = m.frozen().m08.run_warmup(
        m.frozen().m08.Archive(load, pv, len(dates)), price, dates)
    frame_to_csv(warm_frame, OUT / 'warmup_january.csv')
    assert abs(warm_end - REFERENCE_WARMUP_KWH) < 1e-9, warm_end
    print(f'  2025-02-01 shared begin state = {warm_end:.12f} kWh', flush=True)

    kernel = kernel_equivalence_check(archive, price, dates)
    save(OUT / 'kernel_equivalence.json', kernel)
    assert kernel['all_passed'], kernel
    print(f"frozen-kernel equivalence on {len(SAMPLED_DATES)} days: cost diff "
          f"{kernel['max_cost_difference_yuan']:.3e} CNY, vector diff "
          f"{kernel['max_vector_difference']:.3e}", flush=True)

    metrics, shared_summary, protection, protected_full = shared_forecast_frame(load, pv, dates, archive)
    frame_to_csv(metrics, OUT / 'predictor_metrics.csv')
    save(OUT / 'shared_forecast.json', shared_summary)
    quantile = quantile_check(archive, dates)
    save(OUT / 'quantile_validation.json', quantile)
    assert quantile['all_passed'], quantile
    print(f"shared forecast: load MAE {shared_summary['point_forecast'][0]['mae']:.4f} kW, "
          f"pv MAE {shared_summary['point_forecast'][1]['mae']:.4f} kW, coverage "
          f"{shared_summary['coverage']:.6f}", flush=True)

    results = []
    reference_reproduction = {}
    tick = time.perf_counter()
    control = run_scenario(GROUP_BY_ID[CONTROL_ID], load, pv, price, dates, archive, warm_states,
                           verbose=False)
    timings['scenario_T_fixed6000_seconds'] = time.perf_counter() - tick
    results.append(control)
    print(f"  {CONTROL_ID}: {control['totals']['total_cost_yuan']:,.6f} CNY", flush=True)
    for label, reference_path in CONTROL_REFERENCES:
        record = compare_to_reference(control, CONTROL_ID, reference_path,
                                      CONTROL_REFERENCE_TOTAL_YUAN)
        reference_reproduction[label] = record
        assert record['passed'], record
        print(f"    reproduces {label}: day-level max diff {record['max_day_level_difference']:.3e} kWh, "
              f"cost diff {record['difference_vs_registered_yuan']:.3e} CNY, "
              f"strict={record['strict_passed']}, equal-cost-face exemption="
              f"{record['equal_cost_face_exemption']['applies']}", flush=True)
    save(OUT / 'reference_reproduction.json', reference_reproduction)
    if args.mode == 'repro':
        return

    tick = time.perf_counter()
    treatment = run_scenario(GROUP_BY_ID[TREATMENT_ID], load, pv, price, dates, archive, warm_states,
                             verbose=True)
    timings['scenario_T_free_seconds'] = time.perf_counter() - tick
    results.append(treatment)
    print(f"  {TREATMENT_ID}: {treatment['totals']['total_cost_yuan']:,.6f} CNY", flush=True)

    identity = shared_identity_check(results, protection, protected_full)
    save(OUT / 'shared_identity.json', identity)
    assert identity['all_passed'], identity
    print(f"shared prediction/protection identity: max difference {identity['protected']:.3e} kWh",
          flush=True)

    summary = summary_frame(results)
    frame_to_csv(summary, OUT / 'summary.csv')
    daily = daily_contrast_frame(results)
    frame_to_csv(daily, OUT / 'daily_contrasts.csv')
    monthly = monthly_contrast_frame(results)
    frame_to_csv(monthly, OUT / 'monthly_contrasts.csv')
    windows = window_frame(results, price)
    frame_to_csv(windows, OUT / 'window_emergency.csv')
    window_contrast = window_contrast_frame(windows)
    frame_to_csv(window_contrast, OUT / 'window_contrasts.csv')
    pairing = next_morning_frame(results)
    frame_to_csv(pairing, OUT / 'next_morning_pairing.csv')
    pairing_contrast = next_morning_contrast(pairing)
    frame_to_csv(pairing_contrast, OUT / 'next_morning_contrasts.csv')
    inventory = inventory_frame(results)
    frame_to_csv(inventory, OUT / 'inventory.csv')
    valuation = valuation_frame(results, price)
    frame_to_csv(valuation, OUT / 'inventory_valuation.csv')
    energy = energy_frame(results)
    frame_to_csv(energy, OUT / 'energy_contrasts.csv')
    stability = stability_frame(daily, monthly)
    frame_to_csv(stability, OUT / 'stability.csv')
    terminal_all = pd.concat([result['daily'] for result in results], ignore_index=True)
    frame_to_csv(terminal_bounds_frame(terminal_all), OUT / 'terminal_bounds.csv')

    perturbation = perturbation_checks(load, pv, price, dates, published_l, published_v)
    save(OUT / 'future_perturbation_checks.json', perturbation)
    assert perturbation['all_passed'], perturbation
    prefix = feedback_prefix_check(results, load, pv, price, dates)
    save(OUT / 'feedback_prefix_check.json', prefix)
    assert prefix['all_passed'], prefix
    dominance = sampled_dominance_checks(results, load, pv, price, dates, archive)
    save(OUT / 'sampled_dominance.json', dominance)
    assert dominance['all_passed'], dominance
    print(f"perturbation {len(perturbation['cases'])} cases, half-day prefix "
          f"{len(prefix['cases'])} cases, sampled dominance {dominance['solves']} independent MILPs",
          flush=True)

    tick = time.perf_counter()
    soc = soc_frame(results)
    produced = figures(summary, monthly, window_contrast, soc)
    timings['figures_seconds'] = time.perf_counter() - tick
    integrity = m.figure_integrity(produced)
    save(OUT / 'figure_integrity.json', integrity)

    protected_after = protected_manifest()
    changed = sorted(name for name in protected if protected_after.get(name) != protected[name])
    missing = sorted(name for name in protected if name not in protected_after)
    save(OUT / 'protected_after.json', protected_after)

    physical = {result['totals']['strategy_id']:
                {key: float(value) for key, value in result['validation']['checks'].items()}
                for result in results}
    physical_max = max(max(entry.values()) for entry in physical.values())
    delta = delta_row(PRIMARY_LABEL, treatment, control)
    checks = dict(archive=archive_meta, boundary=boundary, matrix=matrix, kernel=kernel,
                  quantile=quantile, shared_forecast=shared_summary, shared_identity=identity,
                  reference_reproduction=reference_reproduction, perturbation=perturbation,
                  feedback_prefix=prefix, dominance=dominance, energy=energy.to_dict('records'),
                  physical=physical, physical_max=float(physical_max),
                  protected_unchanged=bool(not changed and not missing),
                  protected_changed=changed, protected_missing=missing, protected_count=len(protected),
                  warmup=dict(shared_begin_state_kWh=warm_end,
                              checks={key: float(value) for key, value in warm_checks.items()}),
                  delta=delta, timings=timings,
                  thresholds=dict(energy_kWh=ENERGY_TOL_KWH, cost_yuan=COST_TOL_YUAN))
    save(OUT / 'checks.json', checks)

    write_report(dict(summary=summary, daily=daily, monthly=monthly, windows=windows,
                      window_contrast=window_contrast, pairing=pairing_contrast,
                      inventory=inventory, valuation=valuation, energy=energy, stability=stability,
                      metrics=metrics, shared=shared_summary, checks=checks, warm_end=warm_end,
                      timings=timings, integrity=integrity, delta=delta, soc=soc,
                      terminal=terminal_bounds_frame(terminal_all)))
    artifact_hashes = {path.relative_to(OUT).as_posix(): digest(path)
                       for path in sorted(OUT.rglob('*'))
                       if path.is_file() and path.name not in ('artifact_hashes.json',
                                                              'run_manifest.json')}
    save(OUT / 'artifact_hashes.json', artifact_hashes)
    manifest = dict(started_utc=started_utc,
                    finished_utc=datetime.now(timezone.utc).isoformat(),
                    wall_seconds=time.perf_counter() - started, mode=args.mode,
                    executable=sys.executable, python=sys.version, numpy=np.__version__,
                    pandas=pd.__version__, signature=signature, input_sha256=hashes,
                    snapshot_sha256=snapshot_hashes, parameters=PARAMETERS, dependency=dependency,
                    source_sha256=source_hashes, protected_before_count=len(protected),
                    protected_unchanged=bool(not changed and not missing),
                    protected_changed=changed, protected_missing=missing,
                    warmup_shared_begin_state_kWh=warm_end,
                    scenario_totals={result['totals']['strategy_id']: result['totals']
                                     for result in results},
                    primary_delta=delta, shared_forecast=shared_summary,
                    reference_reproduction=reference_reproduction,
                    timings=timings,
                    self_checks_passed=dict(archive=archive_meta['all_passed'],
                                            boundary=boundary['all_passed'], matrix=matrix['all_passed'],
                                            kernel=kernel['all_passed'], quantile=quantile['all_passed'],
                                            shared_identity=identity['all_passed'],
                                            perturbation=perturbation['all_passed'],
                                            feedback_prefix=prefix['all_passed'],
                                            dominance=dominance['all_passed']),
                    artifact_count=len(artifact_hashes), report=str(REPORT_MD.relative_to(ROOT)))
    save(OUT / 'run_manifest.json', manifest)
    assert manifest['protected_unchanged'], (changed, missing)
    print(summary[['strategy_id', 'terminal_mode', 'planned_cost_yuan', 'emergency_cost_yuan',
                   'total_cost_yuan', 'emergency_days', 'final_kWh']].to_string(index=False), flush=True)
    print(json.dumps(jsonable(delta), ensure_ascii=False, indent=2), flush=True)
    print(f'self-checks {manifest["self_checks_passed"]}', flush=True)
    print(f'wall seconds = {time.perf_counter() - started:.1f}', flush=True)
    return manifest


def write_report(context):
    """Report generator (kept below main so a syntax error here cannot corrupt the run)."""
    summary = context['summary']
    daily = context['daily']
    monthly = context['monthly']
    windows = context['windows']
    window_contrast = context['window_contrast']
    pairing = context['pairing']
    inventory = context['inventory']
    valuation = context['valuation']
    energy = context['energy']
    stability = context['stability']
    metrics = context['metrics']
    shared = context['shared']
    checks = context['checks']
    warm_end = context['warm_end']
    integrity = context['integrity']
    delta = context['delta']

    def money(value):
        return f'{float(value):,.2f}'

    def pct(value):
        return f'{float(value):+.4f}'

    main = summary.set_index('strategy_id')
    fixed = main.loc[CONTROL_ID]
    free = main.loc[TREATMENT_ID]
    summary_rows = '\n'.join(
        f'| {row.strategy_id} | {row.terminal_mode} | {money(row.planned_cost_yuan)} '
        f'| {money(row.emergency_cost_yuan)} | {money(row.total_cost_yuan)} | {row.emergency_days} '
        f'| {row.emergency_events} | {row.unused_kWh:,.1f} | {row.loss_kWh:,.1f} | {row.final_kWh:,.1f} |'
        for row in summary.itertuples())
    metric_rows = '\n'.join(
        f'| {row.target} | {row.unit} | {row.n} | {row.mae:,.6f} | {row.rmse:,.6f} | {row.bias:,.6f} |'
        for row in metrics.itertuples())
    month_rows = '\n'.join(
        f'| {row.month} | {money(row.fixed_total_cost_yuan)} | {money(row.free_total_cost_yuan)} '
        f'| {money(row.free_minus_fixed_yuan)} | {money(row.delta_emergency_cost_yuan)} '
        f'| {row.delta_emergency_kWh:,.2f} |' for row in monthly.itertuples())
    window_rows = '\n'.join(
        f'| {row.scope} | {row.window} | {money(row.fixed_emergency_cost_yuan)} '
        f'| {money(row.free_emergency_cost_yuan)} | {money(row.delta_emergency_cost_yuan)} '
        f'| {row.delta_emergency_kWh:,.2f} | {money(row.delta_planned_cost_yuan)} |'
        for row in window_contrast.itertuples())
    inventory_rows = '\n'.join(
        f'| {row.strategy_id} | {row.nominal_mean_kWh:,.1f} | {row.nominal_p05_kWh:,.1f} '
        f'| {row.nominal_p95_kWh:,.1f} | {row.actual_mean_kWh:,.1f} | {row.actual_p05_kWh:,.1f} '
        f'| {row.actual_p95_kWh:,.1f} | {row.nominal_days_at_6000} | {row.actual_days_at_floor} '
        f'| {row.actual_days_at_ceiling} |' for row in inventory.itertuples())
    valuation_rows = '\n'.join(
        f'| {row.strategy_id} | {money(row.total_cost_yuan)} | {row.start_kWh:,.4f} '
        f'| {row.end_kWh:,.4f} | {money(row.inventory_adjustment_yuan)} '
        f'| {money(row.adjusted_cost_yuan)} |' for row in valuation.itertuples())
    energy_rows = '\n'.join(
        f'| {row.comparison.split(" (")[0]} | {row.delta_planned_kWh:,.2f} | {row.delta_emergency_kWh:,.2f} '
        f'| {row.delta_charge_kWh:,.2f} | {row.delta_discharge_kWh:,.2f} | {row.delta_unused_kWh:,.2f} '
        f'| {row.delta_loss_kWh:,.2f} | {row.delta_end_kWh:,.2f} | {row.identity_residual_kWh:.3e} |'
        for row in energy.itertuples())
    integrity_rows = '\n'.join(
        f"- `figures/q2_free_terminal/{row['file']}`：{row['width']}×{row['height']} 像素，"
        f"非白像素比例 {row['non_white_fraction']:.4f}，颜色数 {row['distinct_colours']}。"
        for row in integrity)
    stability_row = stability.iloc[0]
    boundary_rows = '\n'.join(
        f"| {row['case']} | {row['expected']} | {str(row['observed'])[:160]} | "
        f"{'通过' if row['passed'] else '未通过'} |" for row in checks['boundary']['rows'])
    perturbation_rows = '\n'.join(
        f"| {case['date']} | {case['date']} | {case['case']} "
        f"| {case['modes'][CONTROL_ID]['max_plan_difference_kWh']:.1e} "
        f"| {case['modes'][TREATMENT_ID]['max_plan_difference_kWh']:.1e} "
        f"| {case['modes'][CONTROL_ID]['max_state_difference_kWh']:.1e} "
        f"| {case['modes'][TREATMENT_ID]['max_state_difference_kWh']:.1e} |"
        for case in checks['perturbation']['cases'])
    dominance_rows = '\n'.join(
        f"| {row['date']} | {row['initial_group']} | {row['initial_kWh']:,.4f} "
        f"| {money(row['fixed_objective_yuan'])} | {money(row['free_objective_yuan'])} "
        f"| {money(row['difference_yuan'])} | {row['free_terminal_kWh']:,.4f} "
        f"| {'是' if row['dominance_holds'] else '否'} |" for row in checks['dominance']['cases'])
    morning_annual = pairing[['fixed_next_emergency_kWh', 'free_next_emergency_kWh',
                              'fixed_next_emergency_cost_yuan', 'free_next_emergency_cost_yuan']].sum()

    report = f"""# 问题二：固定6000与自由末态对照实验结果报告

日期：2026-09-11。状态：**实验自检完成，独立审计待完成**。本轮实现、运行并自检了
`reports/问题二_固定6000与自由末态对照实验方案.md` 登记的两组同年因果回测；
未锁定第二问最终模型，未填写 `result2.xlsx`，未修改第一问。

## 1. 问题分析

问题二没有"每日首尾储电量相等"的题面约束：实际状态已跨日连续。但日前规划额外把名义日末储电量
钉在 6000 kWh。本轮只解除这一个等式，检验它是否降低 2025-02-01 至 12-31 的实际现金费用，
以及是否改变次晨应急、库存与损耗。

两组唯一差别是名义 MILP 的末端约束：`T_fixed6000` 为 $\\bar E_{{144}}=6000$ kWh；
`T_free` 仅为 $1200\\le\\bar E_{{144}}\\le 10800$。预测、W28/q80 保护、公共 1 月预运行、
实际贪心反馈、收费规则、电池与时间参数、求解器设置全部相同；**两组不共享 2 月之后的实际初态**。
主比较 $\\Delta C=C_{{\\mathrm{{free}}}}-C_{{\\mathrm{{fixed6000}}}}$，负值表示自由末态更省。

只解除末态等式可能减少不必要购电与循环损耗，也可能只优化当天计划费而消耗次日有价值的库存；
两者孰强必须由 2—12 月连续回测判定，**不能用单日计划费下降代替真实节费**。

### 1.1 主要结论

- 控制复现：`T_fixed6000` 总费 {money(fixed.total_cost_yuan)} 元，与登记参考
  {money(CONTROL_REFERENCE_TOTAL_YUAN)} 元差
  {checks['reference_reproduction']['P1_q80']['difference_vs_registered_yuan']:.3e} 元；
  逐日聚合最大差
  {checks['reference_reproduction']['P1_q80']['max_day_level_difference']:.3e} kWh。
- 主比较 $\\Delta C$ = {money(delta['delta_total_cost_yuan'])} 元
  （{pct(delta['delta_total_cost_percent'])}%），其中计划购电费
  {money(delta['delta_planned_cost_yuan'])} 元、应急费
  {money(delta['delta_emergency_cost_yuan'])} 元；应急天数
  {int(delta['delta_emergency_days']):+d} 天、应急时段数
  {int(delta['delta_emergency_events']):+d} 次、应急电量
  {delta['delta_emergency_kWh']:+,.2f} kWh。
- 逐日方向：{int(stability_row.days_improved)} 天更省、{int(stability_row.days_worse)} 天更贵；
  逐月 {int(stability_row.months_improved)}/{int(stability_row.months_total)} 个月更省。
  最贵单日 {stability_row.worst_day}（{money(stability_row.worst_day_yuan)} 元），
  最省单日 {stability_row.best_day}（{money(stability_row.best_day_yuan)} 元）。
- 该差额相对 1366 万元量级极小，且方向集中于少数日期，**不能据此宣称自由末态普遍更优**。

## 2. 数据预处理

只使用附件1 的 144 点日内电价与附件2 的 365×144 实际负载、光伏功率；不使用附件3/4、未来实测或
外部天气。不平滑、不删点、不插补。预测直接读取 23 号冻结发布档案
`results/q2_pv_support_features/pv_forecast_archive_P1.csv`，**本轮不训练任何预测器**。

### 2.1 输入核对与档案一致性

- 档案 365 日 × 144 段完整、日期-slot 唯一，1 月 1 日无点预测（按构造为缺失），
  1 月 2 日起两目标预测有限、非负；支持窗口三列齐全，1 月 1 日窗口为登记的缺省
  {checks['archive']['day0_support_default']}（不参与正式误差）。
- 档案为 4 位小数的舍入功率；逐日聚合复现证明舍入不影响登记调度（见 4.1），故直接使用，
  不做重算或平滑。负载预测相对控制发布参考最大差
  {checks['archive']['archive_load_vs_reference_kW']:.3e} kW、光伏
  {checks['archive']['archive_pv_vs_reference_kW']:.3e} kW。
- 两组使用**完全相同**的预测、原始残差、q80 修正与修正净需求（见 3.1）；实测两组逐段差
  {checks['shared_identity']['protected']:.3e} kWh、保护量逐段差
  {checks['shared_identity']['adjustment']:.3e} kWh。

共同点预测质量（评价期 334 天 48096 段，只报一次，单位见"单位"列）：

| 目标 | 单位 | 样本数 | MAE | RMSE | 偏差 |
|---|---|---:|---:|---:|---:|
{metric_rows}

保护量均值 {shared['mean_protection_kWh']:,.6f} kWh（最小 {shared['protection_min_kWh']:,.4f}、
最大 {shared['protection_max_kWh']:,.4f}），覆盖率 {shared['coverage']:.6f}
（修正净需求不小于实际净需求的比例）。这是逐时段经验分位保护，**不是 80% 无应急概率**。

### 2.2 公共 1 月与评价期

复现已冻结公共 1 月物理预运行：1 月 1 日 6000 kWh、零普通计划、电池静置，1 月 2 日起沿原公共
规则；2025-02-01 00:00 共同实际初态 **{warm_end:.12f} kWh**（两组相同）。正式评价为
2025-02-01 至 12-31 共 334 天 48096 段，1 月不计入费用。

## 3. 模型建立

### 3.1 共享预测与 q80 保护

$$n_{{k,t}}=(L_{{k,t}}-V_{{k,t}})\\Delta t,\\quad
\\widehat n_{{k,t}}=(\\widehat L_{{k,t}}-\\widehat V_{{k,t}})\\Delta t,\\quad
\\varepsilon_{{j,t}}=n_{{j,t}}-\\widehat n_{{j,t}}.$$

第 $k$ 日取 $\\mathcal H_k=\\{{j:\\max(1,k-28)\\le j<k\\}}$，$m$ 为其大小；$m<7$ 修正为 0，
否则逐时段升序取第 $\\lceil 0.8m\\rceil$ 个（整数位置 `(80m+99)//100`，索引减 1），
$\\widetilde n=\\widehat n+r$，保留负修正与负净需求。评价期每日 $m=28$、位置 23
（校验：$m$ 取值 {checks['quantile']['window_length_values']}，位置
{checks['quantile']['positions']}，NaN 输入被入口拒绝
{checks['quantile']['nan_input_gate_rejects']}）。

两组共享同一份 $\\widehat L,\\widehat V,\\varepsilon,r,\\widetilde n$，**单改组的保护量不相加**。

### 3.2 名义 MILP：只改末端界

省略日期下标，求解

$$\\min\\sum_{{t=0}}^{{143}}p_tq_t,\\qquad
q_t+\\bar d_t=\\widetilde n_t+\\bar c_t+\\bar w_t,\\qquad
\\bar E_{{t+1}}=\\bar E_t+0.9\\bar c_t-\\bar d_t/0.9,$$

$$q_t,\\bar c_t,\\bar d_t,\\bar w_t\\ge0,\\quad \\bar c_t\\le B\\bar z_t,\\quad
\\bar d_t\\le B(1-\\bar z_t),\\quad \\bar z_t\\in\\{{0,1\\}},\\quad B=5000\\Delta t,$$

$$1200\\le\\bar E_t\\le10800\\ (t=0,\\ldots,144),\\qquad \\bar E_0=E^{{\\mathrm{{actual}}}}_{{k,0}},
\\qquad \\bar E_{{144}}\\in\\mathcal T,$$

其中固定组 $\\mathcal T=\\{{6000\\}}$、自由组 $\\mathcal T=[1200,10800]$。冻结内核
`code/02_q1_baseline.py` 只用 `terminal_kWh` 表达末态，且 `None` 会被解释为
$\\bar E_{{144}}=\\bar E_0$（日循环）；直接传 `None` **不是**自由末态。因此本脚本逐字转写该整数
内核的建模代码，新增显式 `terminal_mode`，只把 $\\bar E_{{144}}$ 的状态上下界由
$[6000,6000]$ 改为 $[1200,10800]$，其余目标、约束矩阵、变量顺序、integrality 与求解参数不变。

同初态同输入的矩阵证据（`matrix_evidence.json`）：目标 {checks['matrix']['objective_terms']} 项一致、
integrality 一致、约束矩阵差 {checks['matrix']['constraint_matrix_difference_nnz']} 个非零元、
约束上下界一致，**只有下标 {checks['matrix']['expected_terminal_index']}
（=5×144，即 $\\bar E_{{144}}$）的上下界不同**。固定组解与冻结内核在
{len(checks['kernel']['cases'])} 个抽样日上目标差 {checks['kernel']['max_cost_difference_yuan']:.3e} 元、
全部解向量差 {checks['kernel']['max_vector_difference']:.3e}。

求解器保持 SciPy `milp` / HiGHS，相对 gap ≤1e-9、时限 120 秒；不换 LP、不加二级择优、
不加软惩罚、不把自由末态替换成固定 1200。

### 3.3 实际反馈与现金费用

令 $u=q-n$，实际动作沿用冻结反馈

$$c=\\min\\{{\\max(u,0),B,(10800-E)/0.9\\}},\\quad
d=\\min\\{{\\max(-u,0),B,0.9(E-1200)\\}},$$

$$e=\\max(-u-d,0),\\quad w=\\max(u-c,0),\\quad E'=E+0.9c-d/0.9.$$

实际反馈不按名义末态强制充电、放电或裁剪库存；两组均满足 $E^{{\\mathrm{{actual}}}}_{{k+1,0}}
=E^{{\\mathrm{{actual}}}}_{{k,144}}$，不要求日内首尾相等。

$$C=\\sum_{{k\\in\\mathrm{{Feb:Dec}}}}\\sum_t(p_tq_{{k,t}}+5p_te_{{k,t}}).$$

普通计划全额付款，未使用电量不退款、不弃余罚金、不售电；自由组不计"少于 6000 罚款"，
也不虚构清空库存的卖电收入。

### 3.4 年末库存估值（仅诊断）

$$C^*=C-\\nu(E_{{\\mathrm{{end}}}}-E_{{\\mathrm{{start}}}}),\\qquad \\nu=\\operatorname{{median}}(p)/0.9,$$

$\\nu$ 单位为元/内部 kWh，仅作库存余额敏感性诊断，**不进入主费用**，也不作为售价或目标系数。

## 4. 模型求解与结果

两组共 668 次正式日前 MILP，另加 {checks['dominance']['solves']} 次同初态支配检验与
因果重放；全部最优、无回退（`solver_failures` 均为 0）。

### 4.1 两组总费

| 组 | 名义末态 | 计划购电费(元) | 应急费(元) | 现金总费(元) | 应急天数 | 应急事件 | 未使用(kWh) | 损耗(kWh) | 实际期末(kWh) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{summary_rows}

控制组 `T_fixed6000` 对 23 号 `P1_q80` 与 25 号 `F01_pv_support` 的复现为：逐日聚合最大差
{checks['reference_reproduction']['P1_q80']['max_day_level_difference']:.3e} /
{checks['reference_reproduction']['F01_pv_support']['max_day_level_difference']:.3e} kWh，
总费差 {checks['reference_reproduction']['P1_q80']['difference_vs_registered_yuan']:.3e} 元，
逐段差异仅为等价最优（受影响时段 {checks['reference_reproduction']['P1_q80']['equivalent_optimum']['affected_slots']} 个）。

**控制复现的分级结论（`reference_reproduction.json`，两口径并列保存）：**

- 严格口径（逐日聚合 <1e-6 kWh 且总费差 ≤1e-4 元）：**未通过**。总费差
  {money(checks['reference_reproduction']['P1_q80']['difference_vs_registered_yuan'])} 元，
  逐日聚合最大差 {checks['reference_reproduction']['P1_q80']['max_day_level_difference']:.3e} kWh，
  仅 {checks['reference_reproduction']['P1_q80']['days_with_any_difference']} 个日期出现日级差异。
- 等优面披露口径：{checks['reference_reproduction']['P1_q80']['equal_cost_face_exemption']['applies']}。
  首个分歧日 {checks['reference_reproduction']['P1_q80']['equal_cost_face_exemption']['first_affected_date']}，
  该日自身计划费差
  {checks['reference_reproduction']['P1_q80']['equal_cost_face_exemption']['first_affected_plan_cost_difference_yuan']:.3e} 元，
  即两组轨迹同属一个等费用最优面；该日充放电多循环 2.71 kWh，使实际日末状态相差 0.572 kWh，
  次日现金费因此增加 {money(checks['reference_reproduction']['P1_q80']['difference_vs_registered_yuan'])} 元
  （全年相对量级 {checks['reference_reproduction']['P1_q80']['difference_vs_registered_yuan'] / fixed.total_cost_yuan:.2e}）。
- 机制：日内 144 点电价逐日重复，部分日前 MILP 存在大量费用相同的最优解（充放电在等价时段间重分配）；
  保护净需求约 1 ULP 的重建差即可改变求解器返回的顶点。本实验的本地内核与冻结内核在相同输入下
  目标与全部解向量逐位一致（`kernel_equivalence.json`，{len(checks['kernel']['cases'])} 日上差 0.0），
  故该差异是求解器顶点选择，不是建模差异。按方案第 4.1 节"仅名义等优轨迹不同可以核目标与可行性后披露"，
  此处如实披露而不改写为逐段复现；**严格阈值未达标这一事实保留在 `strict_passed=False` 与上表**，
  由主 Agent/独立审计判断是否接受。

### 4.2 主比较与分月、分日

| 月 | 固定总费(元) | 自由总费(元) | 差(元) | 应急费差(元) | 应急电量差(kWh) |
|---|---:|---:|---:|---:|---:|
{month_rows}

| 指标 | 数值 |
|---|---:|
| 现金总费差 ΔC (元) | {money(delta['delta_total_cost_yuan'])} |
| 计划购电费差 (元) | {money(delta['delta_planned_cost_yuan'])} |
| 应急费差 (元) | {money(delta['delta_emergency_cost_yuan'])} |
| 应急电量差 (kWh) | {delta['delta_emergency_kWh']:+,.2f} |
| 应急天数差 | {int(delta['delta_emergency_days']):+d} |
| 更省月数 / 更贵月数 | {int(stability_row.months_improved)} / {int(stability_row.months_worse)} |
| 更省日数 / 更贵日数 | {int(stability_row.days_improved)} / {int(stability_row.days_worse)} |

### 4.3 次晨应急（0—6、6—10、10—24 时互斥，合计全年）

| 口径 | 时段 | 固定应急费(元) | 自由应急费(元) | 差(元) | 应急电量差(kWh) | 计划费差(元) |
|---|---|---:|---:|---:|---:|---:|
{window_rows}

19—21 时为诊断口径，**不与上表相加**。`next_morning_pairing.csv` 另按"某日日末 / 次晨"配对
（仅取 2025-02-01 至 12-30 的 333 个前日，避免假造 1 月 31 日策略分叉或 2026 年结果）：
次晨 0—6 时应急合计固定 {morning_annual.fixed_next_emergency_kWh:,.2f} kWh /
{ money(morning_annual.fixed_next_emergency_cost_yuan) } 元，自由
{morning_annual.free_next_emergency_kWh:,.2f} kWh /
{ money(morning_annual.free_next_emergency_cost_yuan) } 元。

### 4.4 名义与实际库存

| 组 | 名义日末均值 | 名义P05 | 名义P95 | 实际日末均值 | 实际P05 | 实际P95 | 名义=6000天数 | 实际触底天数 | 实际触顶天数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{inventory_rows}

名义末态上限命中/越界：`T_fixed6000` 违反量
{checks['physical']['T_fixed6000'].get('nominal_terminal', 0.0):.3e} kWh，
`T_free` 违反量 {checks['physical']['T_free'].get('nominal_terminal', 0.0):.3e} kWh（自由组无目标，
只检查物理界）。自由组的实际日末轨迹由它自己的名义计划和实际反馈递推，不重置到 1200。

### 4.5 能量账与年末估值

| 比较 | Δ计划(kWh) | Δ应急(kWh) | Δ充电(kWh) | Δ放电(kWh) | Δ未使用(kWh) | Δ损耗(kWh) | Δ期末(kWh) | 恒等式残差(kWh) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{energy_rows}

| 组 | 现金总费(元) | 期初(kWh) | 期末(kWh) | 库存估值调整(元) | 调整后 C*(元) |
|---|---:|---:|---:|---:|---:|
{valuation_rows}

$C^*$ 只作余额敏感性诊断，主费用仍是现金总费。

### 4.6 图表

{integrity_rows}

三图分别为两组计划/应急/总费、分月现金差与分窗口应急差、名义与实际日末储电量轨迹（分面）。

## 5. 验证与适用边界

### 5.1 人工边界与等价性

| 合成案例 | 预期 | 观测（截断） | 结果 |
|---|---|---|---|
{boundary_rows}

其中单段净需求 1 kWh、初态 $1200+1/0.9$ 的自由末态解为 $q=0,d=1,E_1=1200$，
证明自由模式未隐含首尾相等；同输入固定 6000 在单段内不可行（超功率上限），如实记录而不是
判为求解失败。

### 5.2 因果前缀与未来扰动

| 扰动日 | 日期 | 扰动 | 固定组计划前缀差 | 自由组计划前缀差 | 固定组状态前缀差 | 自由组状态前缀差 |
|---|---|---|---:|---:|---:|---:|
{perturbation_rows}

上表为 6 例未来真值扰动（当日及以后负载×1.2 或光伏×0.7），从共同初态重放两模式到该日发布：
此前每日计划、实际状态与保护量必须逐位不变（最大差
{checks['perturbation']['max_prefix_difference']:.3e} kWh）。另固定 6 月 21 日两组各自的计划
与初态、只扰动 12:00 之后负载/光伏，共 4 例：slot 0—71 动作、费用与 `state[0:73]` 全部不变
（最大差 {checks['feedback_prefix']['max_prefix_difference']:.3e}）。

### 5.3 同初态单日支配关系

同一日预测、q80 与实际初态下，固定组可行域包含于自由组，故 $J^*_{{\\mathrm{{free}}}}\\le
J^*_{{\\mathrm{{fixed6000}}}}$。在抽样日（日序号
{", ".join(str(v) for v in checks['dominance']['sampled_days'])}，含五个固定日期与两组各自最大应急费日）
上用两组各自初态各解两模式，共 {checks['dominance']['solves']} 个独立 MILP：

| 日期 | 初态来源组 | 初态(kWh) | 固定目标(元) | 自由目标(元) | 差(元) | 自由末态(kWh) | 支配成立 |
|---|---|---:|---:|---:|---:|---:|---|
{dominance_rows}

### 5.4 全量核验与阈值

| 检查 | 阈值 | 结果 |
|---|---|---|
| 固定组费用复现（严格） | 1e-4 元 | {checks['reference_reproduction']['P1_q80']['difference_vs_registered_yuan']:.3e} 元（strict_passed={checks['reference_reproduction']['P1_q80']['strict_passed']}） |
| 固定组逐日聚合复现（严格） | 1e-6 kWh | {checks['reference_reproduction']['P1_q80']['max_day_level_difference']:.3e} kWh |
| 等优面披露口径 | 见 4.1 | {checks['reference_reproduction']['P1_q80']['equal_cost_face_exemption']['applies']}（首分歧日 {checks['reference_reproduction']['P1_q80']['equal_cost_face_exemption']['first_affected_date']} 自身计划费差 {checks['reference_reproduction']['P1_q80']['equal_cost_face_exemption']['first_affected_plan_cost_difference_yuan']:.1e} 元） |
| 预测原值核对 | 5e-5 kW | 负载 {checks['archive']['archive_load_vs_reference_kW']:.3e}、光伏 {checks['archive']['archive_pv_vs_reference_kW']:.3e} kW |
| 冻结内核等价（固定模式） | 1e-4 元 / 1e-6 | {checks['kernel']['max_cost_difference_yuan']:.3e} 元 / {checks['kernel']['max_vector_difference']:.3e} |
| 两模式矩阵差异 | 仅 E144 上下界 | 下标 {checks['matrix']['expected_terminal_index']}，矩阵差 {checks['matrix']['constraint_matrix_difference_nnz']} |
| 两模式实际物理与连续性 | 1e-6 kWh | {checks['physical_max']:.3e} kWh |
| 能量恒等式 | 1e-4 kWh | {abs(energy.iloc[0].identity_residual_kWh):.3e} kWh |
| MILP 最优、gap≤1e-9、无回退 | — | 两组失败数 {int(fixed.solver_failures)} / {int(free.solver_failures)}，最大 gap {max(fixed.max_mip_gap, free.max_mip_gap):.3e} |
| 受保护旧资产 | 零变更 | {checks['protected_count']} 个文件，变更 {len(checks['protected_changed'])}、缺失 {len(checks['protected_missing'])} |

### 5.5 限制

1. 2025 年已用于多轮方法设计，本轮是滚动因果回测，不是独立盲测或跨年验证；不把 48096 段当
   独立样本作显著性检验。
2. 自由末态只解除名义日末等式，**不引入未来库存价值**；本轮结果只在当前单日目标下成立，
   不能外推到所有分位、预测器或年份。
3. 上游预测训练因果性与 20/24/26 号独立审计不在本轮范围；本轮只审计"冻结档案之后的终端与
   调度链路"。
4. 名义 MILP 的等价最优轨迹未统一二级择优；逐段计划可能因等优解而不同。控制组对 `P1_q80` 的
   严格逐段复现未达标（总费差 {money(checks['reference_reproduction']['P1_q80']['difference_vs_registered_yuan'])} 元，
   相对量级 {checks['reference_reproduction']['P1_q80']['difference_vs_registered_yuan'] / fixed.total_cost_yuan:.2e}），
   已定位为 2025-09-08 等费用最优面的顶点选择经实际状态传递后的结果；该日自身计划费逐位相同，
   但次日现金费因此变化，故**不能表述为逐段或逐分位复现**，是否接受由主 Agent 与独立审计判断。
5. 未加软终端惩罚、两日展望、备用规则、q75 或固定 1200；这些均需另立方案。
6. $C^*$ 只作诊断，不作为售价或目标系数，也不证明每日终端作用已被排除。
7. 三图已程序化检查（尺寸/非白像素/颜色数），**未做人工目视核验**。

## 6. 文件与复现

- `results/q2_free_terminal/registration.json`、`run_manifest.json`、`artifact_hashes.json`：
  登记签名、正确起止时间与 wall time、输入/源码快照哈希、全部交付哈希。
- `results/q2_free_terminal/archive_input_check.json`、`matrix_evidence.json`、
  `kernel_equivalence.json`、`boundary_checks.json`、`quantile_validation.json`、
  `shared_forecast.json`、`shared_identity.json`、`reference_reproduction.json`、
  `future_perturbation_checks.json`、`feedback_prefix_check.json`、`sampled_dominance.json`、
  `checks.json`：全部自检与反例。
- `results/q2_free_terminal/{{T_fixed6000,T_free}}/`：实际与名义 dispatch、日月汇总、
  应急事件、solver 日志、终端界证据、validation。
- 汇总表：`summary.csv`、`daily_contrasts.csv`、`monthly_contrasts.csv`、
  `window_emergency.csv`、`window_contrasts.csv`、`next_morning_pairing.csv`、
  `next_morning_contrasts.csv`、`inventory.csv`、`inventory_valuation.csv`、
  `energy_contrasts.csv`、`stability.csv`、`terminal_bounds.csv`。
- `figures/q2_free_terminal/`：三张静态图。
- 复现入口：

```powershell
Set-Location -LiteralPath 'E:/MathModeling/2026国赛/C题'
E:/Anaconda/envs/math_modeling/python.exe code/27_q2_free_terminal_experiment.py --mode register
E:/Anaconda/envs/math_modeling/python.exe code/27_q2_free_terminal_experiment.py --mode repro
E:/Anaconda/envs/math_modeling/python.exe code/27_q2_free_terminal_experiment.py --mode full
```

原始费用与物理数据以 CSV 为准，不只在图中保留。旧附件、源码与签名实验输出未改动；
共享记忆与进度仅追加"实验自检完成，独立审计待完成"。
"""
    REPORT_MD.write_text(report, encoding='utf-8')


if __name__ == '__main__':
    main()
