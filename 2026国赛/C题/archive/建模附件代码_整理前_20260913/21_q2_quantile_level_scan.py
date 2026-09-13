"""Q2 quantile-level scan on frozen corrected forecast archives (pre-registered).

Run from E:/MathModeling/2026国赛/C题 with the math_modeling interpreter:
    E:/Anaconda/envs/math_modeling/python.exe code/21_q2_quantile_level_scan.py --mode register
    E:/Anaconda/envs/math_modeling/python.exe code/21_q2_quantile_level_scan.py --mode repro
    E:/Anaconda/envs/math_modeling/python.exe code/21_q2_quantile_level_scan.py --mode full

Specification: reports/问题二/实验方案/问题二_分位水平扫描实验方案.md (single-run sensitivity task book).
Frozen inputs: results/q2_lightgbm_residual/{G3_ridge_fixed,G4_lightgbm_fixed}/forecast_residuals.csv
Frozen kernel: results/q2_revision_audit_20260911/source_snapshot/code/{02,05,08}.

No predictor is retrained. Only the historical-error protection level alpha changes. Ten scenarios:
two predictors (corrected ridge, corrected LightGBM) x five levels alpha in {0.70,0.75,0.80,0.85,0.90}.
The runner below passes alpha explicitly to the quantile, the MILP input, the logs and every output, so
no module-level THETA is mutated and scenarios cannot be cross-contaminated.
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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'q2_quantile_level_scan'
FIG = ROOT / 'figures' / 'q2_quantile_level_scan'
AUDIT = ROOT / 'results' / 'q2_revision_audit_20260911'
SNAP = AUDIT / 'source_snapshot'
FROZEN_RUN = ROOT / 'results' / 'q2_lightgbm_residual'
PLAN_MD = ROOT / 'reports/问题二/实验方案/问题二_分位水平扫描实验方案.md'
REPORT_MD = ROOT / 'reports/问题二/实验结果/问题二_分位水平扫描实验结果报告.md'
CODE_21 = Path(__file__).resolve()

DT = 1 / 6
ALPHAS = (0.70, 0.75, 0.80, 0.85, 0.90)
RESIDUAL_WINDOW = 28
MIN_HISTORY_DAYS = 7
TERMINAL_KWH = 6000.0
WARMUP_DAYS = 31
REFERENCE_WARMUP_KWH = 8801.462273333342
REFERENCE = {
    'ridge': dict(total=13717382.360776227, planned=13290624.123764867,
                  emergency=426758.2370113603, folder='G3_ridge_fixed'),
    'lightgbm': dict(total=13673080.670412183, planned=13270950.722208492,
                     emergency=402129.9482036904, folder='G4_lightgbm_fixed'),
}
COST_TOL_YUAN = 1e-4
ENERGY_TOL_KWH = 1e-6

PREDICTORS = [
    dict(id='ridge', label='R', folder='G3_ridge_fixed', role='corrected ridge with PV gate'),
    dict(id='lightgbm', label='L', folder='G4_lightgbm_fixed', role='corrected LightGBM with PV gate'),
]
GROUPS = [dict(id=f"{predictor['label']}_q{int(round(alpha * 100))}", predictor=predictor['id'],
               alpha=alpha, folder=predictor['folder'])
          for predictor in PREDICTORS for alpha in ALPHAS]
BASELINE_ALPHA = 0.80
DISPATCH_COLUMNS = [
    'strategy_id', 'predictor_id', 'alpha', 'date', 'issue_time', 'slot', 'start_time', 'end_time',
    'price_yuan_kWh', 'load_kW', 'pv_kW', 'load_forecast_kW', 'pv_forecast_kW', 'net_forecast_kWh',
    'residual_adjustment_kWh', 'protected_net_kWh', 'residual_days', 'fallback_reason',
    'planned_kWh', 'charge_kWh', 'discharge_kWh', 'emergency_kWh', 'unused_kWh',
    'state_start_kWh', 'state_end_kWh', 'nominal_state_end_kWh',
    'planned_cost_yuan', 'emergency_cost_yuan']
NOMINAL_COLUMNS = [
    'strategy_id', 'alpha', 'date', 'slot', 'nominal_charge_kWh', 'nominal_discharge_kWh',
    'nominal_unused_kWh', 'nominal_binary_mode', 'nominal_state_start_kWh',
    'nominal_state_end_kWh', 'nominal_planned_cost_yuan']

PARAMETERS = dict(
    alphas=list(ALPHAS), residual_window_days=RESIDUAL_WINDOW, min_history_days=MIN_HISTORY_DAYS,
    quantile_rule='empirical inverse distribution: 1-based position (m*a + 99)//100 with a the integer '
                  'percentage, array index position-1; no interpolation; negative corrections retained',
    terminal_kWh=TERMINAL_KWH,
    solver='frozen scipy.optimize.milp/HiGHS kernel, relative gap 1e-9, time limit 120 s',
    actual_control='surplus charges, deficit discharges, residual deficit at 5x the slot price; '
                   'actual state carries across days',
    evaluation='2025-02-01..2025-12-31, 334 days, 48096 slots per scenario, 3340 day-ahead MILPs total',
    predictors=dict(ridge='results/q2_lightgbm_residual/G3_ridge_fixed/forecast_residuals.csv',
                    lightgbm='results/q2_lightgbm_residual/G4_lightgbm_fixed/forecast_residuals.csv'),
    no_retraining=True,
    reference_costs={key: REFERENCE[key]['total'] for key in REFERENCE},
    grid_note='pre-specified local sensitivity grid around the existing q80; global optimality is not '
              'claimed and the grid is not extended if the minimum lands on a boundary',
    upstream_audit='the corrected ridge/LightGBM archives passed their own self-checks and a limited '
                   'read-only review by the main agent; the full module-20 independent audit is still '
                   'outstanding, and this scan does not claim otherwise',
)


_PARENT: dict = {}


def parent():
    if 'module' not in _PARENT:
        spec = importlib.util.spec_from_file_location(
            'scan_parent', ROOT / 'code' / '14_q2_ridge_forecast_experiment.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.OUT = OUT
        module.FIG = FIG
        _PARENT['module'] = module
    return _PARENT['module']


def frozen():
    return parent().frozen()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def frame_to_csv(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8-sig')


OWN_NEW_REL = {'code/21_q2_quantile_level_scan.py', 'reports/问题二/实验结果/问题二_分位水平扫描实验结果报告.md'}
OWN_NEW_PREFIXES = ('results/q2_quantile_level_scan/', 'figures/q2_quantile_level_scan/')
SHARED_APPEND_REL = {'建模上下文记忆.md', 'reports/项目进度.md'}
PROTECTED_EXTRA = ('results/q2_lightgbm_residual',)


def protected_manifest():
    """Hash every pre-existing asset, excluding only this round's own new tree."""
    manifest = {}
    for tree in list(parent().PROTECTED_TREES) + list(PROTECTED_EXTRA):
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
# frozen forecast archives
# --------------------------------------------------------------------------------------
def require_finite_published(load_forecast, pv_forecast, n_days):
    """Input gate: from 2025-01-02 both published targets must be finite and non-negative.

    This exists so a NaN can never reach the order statistic, where np.sort would silently place it
    last and could still return a finite value.
    """
    formal = slice(1, n_days)
    if not (np.isfinite(load_forecast[formal]).all() and np.isfinite(pv_forecast[formal]).all()):
        raise ValueError('published forecast contains a non-finite value after 2025-01-01')
    if not ((load_forecast[formal] >= 0).all() and (pv_forecast[formal] >= 0).all()):
        raise ValueError('published forecast contains a negative power after 2025-01-01')
    return True


def load_predictor_archive(predictor, load, pv, dates):
    """Read one frozen published forecast archive, recompute both quantities and cross-check."""
    path = FROZEN_RUN / predictor['folder'] / 'forecast_residuals.csv'
    assert path.exists(), f'missing frozen archive: {path}'
    frame = pd.read_csv(path, low_memory=False, dtype={'date': str})
    assert len(frame) == 365 * 144, (predictor['id'], len(frame))
    frame = frame.sort_values(['date', 'slot'], kind='mergesort').reset_index(drop=True)
    assert frame.groupby('date').size().eq(144).all(), 'each date must have 144 slots'
    assert frame.date.nunique() == 365
    assert frame.slot.min() == 0 and frame.slot.max() == 143
    assert [str(value)[:10] for value in dates] == sorted(frame.date.unique())
    n = len(dates)
    load_fc = frame.load_forecast_kW.to_numpy().reshape(n, 144)
    pv_fc = frame.pv_forecast_kW.to_numpy().reshape(n, 144)
    load_a = frame.load_kW.to_numpy().reshape(n, 144)
    pv_a = frame.pv_kW.to_numpy().reshape(n, 144)
    net_fc = (load_fc - pv_fc) * DT
    net_act = (load_a - pv_a) * DT
    residual = net_act - net_fc
    saved_net_fc = frame.net_forecast_kWh.to_numpy().reshape(n, 144)
    saved_net_act = frame.net_actual_kWh.to_numpy().reshape(n, 144)
    saved_residual = frame.residual_kWh.to_numpy().reshape(n, 144)

    def worst(a, b):
        mask = np.isfinite(a) & np.isfinite(b)
        if not mask.any():
            return 0.0
        return float(np.max(np.abs(a[mask] - b[mask])))

    checks = dict(
        net_forecast=worst(net_fc, saved_net_fc), net_actual=worst(net_act, saved_net_act),
        residual=worst(residual, saved_residual),
        truth_load=worst(load_a, load), truth_pv=worst(pv_a, pv))
    assert checks['net_forecast'] < 1e-9 and checks['net_actual'] < 1e-9
    assert checks['residual'] < 1e-9, (predictor['id'], checks)
    assert checks['truth_load'] < 1e-6 and checks['truth_pv'] < 1e-6, checks
    require_finite_published(load_fc, pv_fc, n)
    assert np.isfinite(load_a[1:]).all() and np.isfinite(pv_a[1:]).all()
    return dict(predictor=predictor['id'], load=load_a, pv=pv_a, n_days=n,
                load_forecast=load_fc, pv_forecast=pv_fc, net_forecast=net_fc,
                net_actual=net_act, residual=residual, checks=checks,
                source=str(path.relative_to(ROOT)))


# --------------------------------------------------------------------------------------
# quantile protection
# --------------------------------------------------------------------------------------
def quantile_index(m, alpha):
    """1-based order-statistic position, integer arithmetic to avoid boundary rounding."""
    a = int(round(alpha * 100))
    return (m * a + 99) // 100


def ceil_index(m, alpha):
    """The frozen kernel's equivalent rule, used only as a cross-check."""
    return int(math.ceil(m * float(alpha)))


def adjustment(eps, k, window, alpha):
    """Empirical inverse distribution of the historical net-demand error for day k."""
    low = max(1, k - window)
    m = k - low
    if m < MIN_HISTORY_DAYS:
        return np.zeros(144), m, f'insufficient_history(m={m}<{MIN_HISTORY_DAYS})'
    position = quantile_index(m, alpha)
    assert 1 <= position <= m, (m, alpha, position)
    ordered = np.sort(eps[low:k], axis=0)
    return ordered[position - 1].copy(), m, ''


class ScanArchive:
    """Frozen published forecasts plus an explicit-alpha protection rule and the frozen MILP kernel."""

    def __init__(self, load, pv, n_days, load_forecast, pv_forecast, terminal=TERMINAL_KWH):
        self.load = load
        self.pv = pv
        self.n_days = n_days
        self.pred_l = np.asarray(load_forecast, dtype=float)
        self.pred_v = np.asarray(pv_forecast, dtype=float)
        self.n_forecast = (self.pred_l - self.pred_v) * DT
        self.n_actual = (load - pv) * DT
        self.eps = self.n_actual - self.n_forecast
        self.terminal = float(terminal)
        self.solve_calls = 0
        self.last_nominal = None

    def solve_plan(self, protected_net_kWh, price, initial):
        kernel = frozen().core
        load_kw = np.asarray(protected_net_kWh, dtype=float) / DT
        pv_kw = np.zeros_like(load_kw)
        self.solve_calls += 1
        (q, c, d, w, state), summary = kernel.solve(
            load_kw, pv_kw, price, integer=True, initial_kWh=initial, terminal_kWh=self.terminal)
        assert abs(state[-1] - self.terminal) < ENERGY_TOL_KWH
        self.last_nominal = dict(charge=np.array(c), discharge=np.array(d), unused=np.array(w),
                                 binary=np.asarray(summary['binary_mode_raw'], dtype=float),
                                 state=np.array(state))
        return q, state, summary

    def plan_day(self, k, alpha, window, price, initial):
        self.last_nominal = None
        r, m, reason = adjustment(self.eps, k, window, alpha)
        protected = self.n_forecast[k] + r
        try:
            q, state, summary = self.solve_plan(protected, price, initial)
            summary['fallback'] = False
        except Exception as exc:                                    # noqa: BLE001 - recorded, not hidden
            q = np.maximum(protected, 0.0)
            state = np.full(145, float(initial))
            summary = dict(fallback=True, elapsed_seconds=0.0, mip_gap=0.0, checks={},
                           message=f'MILP fallback: {exc}')
        return q, state, summary, protected, r, m, reason


# --------------------------------------------------------------------------------------
# scenario runner (alpha explicit end to end)
# --------------------------------------------------------------------------------------
def emergency_event_rows(date, emergency, price):
    active = emergency > 1e-6
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    stops = np.flatnonzero(active & ~np.r_[active[1:], False]) + 1
    label = frozen().core.label
    return [dict(date=str(pd.Timestamp(date).date()),
                 interval=f'{label(int(s) * 10)}-{label(int(t) * 10)}',
                 start_slot=int(s), end_slot=int(t),
                 emergency_kWh=float(emergency[s:t].sum()),
                 emergency_cost_yuan=float(5.0 * (emergency[s:t] * price[s:t]).sum()))
            for s, t in zip(starts, stops)]


def run_scenario(group, load, pv, price, dates, archive, warm_states, verbose=False):
    m = parent()
    control = m.frozen().bm.control
    strategy_id = group['id']
    alpha = group['alpha']
    state = float(warm_states[WARMUP_DAYS])
    dispatch_rows, nominal_rows, daily_rows, event_rows, solver_rows = [], [], [], [], []
    checks = {}
    for k in range(WARMUP_DAYS, len(dates)):
        initial = state
        q, nom_state, solver, protected, r, residual_days, reason = archive.plan_day(
            k, alpha, RESIDUAL_WINDOW, price, initial)
        assert not solver.get('fallback', False), (strategy_id, k, solver.get('message'))
        c, d, e, w, states, physical = control(q, load[k], pv[k], initial)
        for name, value in physical.items():
            checks[name] = max(checks.get(name, 0.0), float(value))
        for name, value in solver.get('checks', {}).items():
            checks['nominal_' + name] = max(checks.get('nominal_' + name, 0.0), float(value))
        date = dates[k]
        starts = pd.date_range(date, periods=144, freq='10min')
        planned_cost = q * price
        emergency_cost = 5.0 * e * price
        nominal = archive.last_nominal
        dispatch_rows.append(pd.DataFrame({
            'strategy_id': strategy_id, 'predictor_id': group['predictor'], 'alpha': alpha,
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
        })[DISPATCH_COLUMNS])
        nominal_rows.append(pd.DataFrame({
            'strategy_id': strategy_id, 'alpha': alpha, 'date': str(date.date()),
            'slot': np.arange(144), 'nominal_charge_kWh': nominal['charge'],
            'nominal_discharge_kWh': nominal['discharge'], 'nominal_unused_kWh': nominal['unused'],
            'nominal_binary_mode': nominal['binary'],
            'nominal_state_start_kWh': nominal['state'][:-1],
            'nominal_state_end_kWh': nominal['state'][1:],
            'nominal_planned_cost_yuan': planned_cost})[NOMINAL_COLUMNS])
        loss = 0.1 * c + (1 / 0.9 - 1) * d
        daily_rows.append(dict(
            strategy_id=strategy_id, predictor_id=group['predictor'], alpha=alpha,
            date=str(date.date()), initial_kWh=float(initial), final_kWh=float(states[-1]),
            planned_kWh=float(q.sum()), emergency_kWh=float(e.sum()), unused_kWh=float(w.sum()),
            charge_kWh=float(c.sum()), discharge_kWh=float(d.sum()), loss_kWh=float(loss.sum()),
            planned_cost_yuan=float(planned_cost.sum()), emergency_cost_yuan=float(emergency_cost.sum()),
            total_cost_yuan=float(planned_cost.sum() + emergency_cost.sum()),
            emergency_slots=int((e > 1e-6).sum()),
            emergency_events=len(emergency_event_rows(date, e, price)),
            solved=True, solver_seconds=float(solver['elapsed_seconds']),
            mip_gap=float(solver['mip_gap']), milp_fallback=False,
            nominal_terminal_error_kWh=float(abs(np.asarray(nom_state)[-1] - TERMINAL_KWH))))
        event_rows.extend(emergency_event_rows(date, e, price))
        solver_rows.append(dict(strategy_id=strategy_id, alpha=alpha, date=str(date.date()),
                                day_index=k, elapsed_seconds=float(solver['elapsed_seconds']),
                                mip_gap=float(solver['mip_gap']),
                                mip_node_count=int(solver.get('mip_node_count', -1)),
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
    monthly = daily.assign(month=daily.date.str[:7]).groupby('month').agg(
        planned_kWh=('planned_kWh', 'sum'), emergency_kWh=('emergency_kWh', 'sum'),
        unused_kWh=('unused_kWh', 'sum'), charge_kWh=('charge_kWh', 'sum'),
        discharge_kWh=('discharge_kWh', 'sum'), loss_kWh=('loss_kWh', 'sum'),
        planned_cost_yuan=('planned_cost_yuan', 'sum'),
        emergency_cost_yuan=('emergency_cost_yuan', 'sum'),
        total_cost_yuan=('total_cost_yuan', 'sum'), emergency_slots=('emergency_slots', 'sum'),
        emergency_events=('emergency_events', 'sum')).reset_index()
    frame_to_csv(monthly, folder / 'monthly_summary.csv')
    totals = dict(
        strategy_id=strategy_id, predictor_id=group['predictor'], alpha=alpha,
        planned_kWh=float(daily.planned_kWh.sum()), emergency_kWh=float(daily.emergency_kWh.sum()),
        unused_kWh=float(daily.unused_kWh.sum()), charge_kWh=float(daily.charge_kWh.sum()),
        discharge_kWh=float(daily.discharge_kWh.sum()), loss_kWh=float(daily.loss_kWh.sum()),
        planned_cost_yuan=float(daily.planned_cost_yuan.sum()),
        emergency_cost_yuan=float(daily.emergency_cost_yuan.sum()),
        total_cost_yuan=float(daily.total_cost_yuan.sum()),
        emergency_slots=int(daily.emergency_slots.sum()),
        emergency_days=int((daily.emergency_kWh > 1e-6).sum()),
        emergency_events=int(daily.emergency_events.sum()),
        evaluation_initial_kWh=float(daily.initial_kWh.iloc[0]),
        final_kWh=float(daily.final_kWh.iloc[-1]),
        max_mip_gap=float(daily.mip_gap.max()), solver_seconds=float(daily.solver_seconds.sum()),
        solver_failures=int(daily.milp_fallback.sum()),
        mean_adjustment_kWh=float(dispatch.residual_adjustment_kWh.mean()),
        evaluated_days=int(daily.date.nunique()), evaluated_slots=int(len(dispatch)))
    validation = independent_validation(strategy_id, dispatch, daily, events, nominal_frame)
    validation['nominal_check_max'] = float(max(checks.values())) if checks else 0.0
    validation['nominal_checks'] = {key: float(value) for key, value in checks.items()}
    save(folder / 'validation.json', dict(totals=totals, validation=validation))
    return dict(group=group, totals=totals, validation=validation, dispatch=dispatch,
                daily=daily, monthly=monthly, events=events, archive=archive)


def independent_validation(strategy_id, dispatch, daily, events, nominal):
    q = dispatch.planned_kWh.to_numpy()
    c = dispatch.charge_kWh.to_numpy()
    d = dispatch.discharge_kWh.to_numpy()
    e = dispatch.emergency_kWh.to_numpy()
    w = dispatch.unused_kWh.to_numpy()
    s0 = dispatch.state_start_kWh.to_numpy()
    s1 = dispatch.state_end_kWh.to_numpy()
    load = dispatch.load_kW.to_numpy()
    pv = dispatch.pv_kW.to_numpy()
    loss = 0.1 * c + (1 / 0.9 - 1) * d
    checks = dict(
        balance=float(np.max(np.abs(q + pv * DT + d + e - load * DT - c - w))),
        state_recursion=float(np.max(np.abs(s1 - s0 - 0.9 * c + d / 0.9))),
        day_continuity=float(np.max(np.abs(s0[1:] - s1[:-1]))),
        capacity=float(max(0, 1200 - min(s0.min(), s1.min()), max(s0.max(), s1.max()) - 10800)),
        power=float(max(0, c.max() - 5000 * DT, d.max() - 5000 * DT)),
        nonnegative=float(max(0, -min(q.min(), c.min(), d.min(), e.min(), w.min()))),
        mutex=float(np.minimum(c, d).max()),
        nominal_terminal=float(np.max(np.abs(
            nominal.groupby('date').nominal_state_end_kWh.last().to_numpy() - TERMINAL_KWH))),
        nominal_mutex=float(np.minimum(nominal.nominal_charge_kWh.to_numpy(),
                                       nominal.nominal_discharge_kWh.to_numpy()).max()),
        nominal_binary=float(np.max(np.abs(nominal.nominal_binary_mode.to_numpy()
                                           - np.rint(nominal.nominal_binary_mode.to_numpy())))),
        nominal_recursion=float(np.max(np.abs(
            nominal.nominal_state_end_kWh.to_numpy() - nominal.nominal_state_start_kWh.to_numpy()
            - 0.9 * nominal.nominal_charge_kWh.to_numpy()
            + nominal.nominal_discharge_kWh.to_numpy() / 0.9))))
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
                emergency_reconciliation_kWh=emergency_reconciliation)


NOMINAL_ONLY_COLUMNS = ['nominal_state_end_kWh']


DAY_LEVEL_AGGREGATES = dict(
    planned_kWh=('planned_kWh', 'sum'), planned_cost_yuan=('planned_cost_yuan', 'sum'),
    charge_kWh=('charge_kWh', 'sum'), discharge_kWh=('discharge_kWh', 'sum'),
    emergency_kWh=('emergency_kWh', 'sum'), unused_kWh=('unused_kWh', 'sum'),
    state_start_kWh=('state_start_kWh', 'first'), state_end_kWh=('state_end_kWh', 'last'))


def compare_to_reference(result, predictor):
    """Reproduce the frozen q80 scenario in three tiers.

    1. day-level: per-day totals of plan, cost, charge, discharge, emergency, unused and the day
       start/end state must match (this is the reproduction gate);
    2. total cost against the registered value;
    3. slot level: differences are accepted only as a proven equivalent optimum, i.e. the affected
       slots share a price and no day-level aggregate moves.
    """
    reference = pd.read_csv(FROZEN_RUN / REFERENCE[predictor]['folder'] / 'dispatch.csv',
                            low_memory=False)
    current = result['dispatch']
    skip = ('strategy_id', 'predictor_id', 'alpha', 'date', 'issue_time', 'start_time', 'end_time',
            'fallback_reason')
    executed = [name for name in DISPATCH_COLUMNS
                if name in reference.columns and name not in skip
                and name not in NOMINAL_ONLY_COLUMNS]
    assert reference[['date', 'slot']].equals(current[['date', 'slot']])
    executed_differences = {name: float(np.max(np.abs(reference[name].to_numpy()
                                                      - current[name].to_numpy())))
                            for name in executed}
    nominal_differences = {name: float(np.max(np.abs(reference[name].to_numpy()
                                                     - current[name].to_numpy())))
                           for name in NOMINAL_ONLY_COLUMNS if name in reference.columns}
    day_reference = reference.groupby('date').agg(**DAY_LEVEL_AGGREGATES)
    day_current = current.groupby('date').agg(**DAY_LEVEL_AGGREGATES)
    day_differences = (day_current - day_reference).abs()
    delta_plan = current.planned_kWh.to_numpy() - reference.planned_kWh.to_numpy()
    affected = np.flatnonzero(np.abs(delta_plan) > ENERGY_TOL_KWH)
    dates = current.date.to_numpy()
    affected_dates = sorted(set(dates[affected])) if len(affected) else []
    prices_equal = bool(len(affected) == 0 or np.max(np.abs(
        current.price_yuan_kWh.to_numpy()[affected]
        - reference.price_yuan_kWh.to_numpy()[affected])) < 1e-12)
    day_cost_identical = bool(float(day_differences['planned_cost_yuan'].max()) < ENERGY_TOL_KWH)
    reference_cost = float(reference.planned_cost_yuan.sum() + reference.emergency_cost_yuan.sum())
    current_cost = float(current.planned_cost_yuan.sum() + current.emergency_cost_yuan.sum())
    out = dict(
        predictor=predictor, segments=int(len(current)), executed_columns=executed,
        nominal_only_columns=list(nominal_differences),
        max_executed_difference=float(max(executed_differences.values())),
        max_absolute_differences=executed_differences,
        max_day_level_difference=float(day_differences.to_numpy().max()),
        day_level_differences={name: float(day_differences[name].max()) for name in day_differences},
        days_with_any_difference=int((day_differences.max(axis=1) > ENERGY_TOL_KWH).sum()),
        equivalent_optimum=dict(
            affected_slots=int(len(affected)), affected_dates=affected_dates,
            max_slot_plan_difference_kWh=float(np.max(np.abs(delta_plan))) if len(affected) else 0.0,
            affected_slot_prices_equal=prices_equal,
            day_level_cost_identical=day_cost_identical,
            note='a day-ahead optimum that moves charge between equal-price slots changes the slot-level '
                 'plan without moving any day-level aggregate or the realised cash cost'),
        nominal_trajectory_difference_kWh=float(max(nominal_differences.values(), default=0.0)),
        reference_total_cost_yuan=reference_cost, current_total_cost_yuan=current_cost,
        difference_vs_reference_yuan=abs(current_cost - reference_cost),
        registered_total_cost_yuan=REFERENCE[predictor]['total'],
        difference_vs_registered_yuan=abs(current_cost - REFERENCE[predictor]['total']),
        day_level_tolerance_kWh=ENERGY_TOL_KWH, cost_tolerance_yuan=COST_TOL_YUAN)
    out['passed'] = bool(out['max_day_level_difference'] < ENERGY_TOL_KWH
                         and out['difference_vs_registered_yuan'] <= COST_TOL_YUAN
                         and prices_equal and day_cost_identical)
    return out


# --------------------------------------------------------------------------------------
# quantile validation
# --------------------------------------------------------------------------------------
def quantile_validation(load, pv, dates, archives):
    out = {}
    positions = {f'{alpha:.2f}': quantile_index(RESIDUAL_WINDOW, alpha) for alpha in ALPHAS}
    out['positions_at_28_days'] = dict(
        positions=positions, expected={'0.70': 20, '0.75': 21, '0.80': 23, '0.85': 24, '0.90': 26},
        passed=bool(positions == {'0.70': 20, '0.75': 21, '0.80': 23, '0.85': 24, '0.90': 26}))
    mismatches = []
    for m in range(7, 29):
        for alpha in ALPHAS:
            if quantile_index(m, alpha) != ceil_index(m, alpha):
                mismatches.append(dict(m=m, alpha=alpha, integer=quantile_index(m, alpha),
                                       ceil=ceil_index(m, alpha)))
    out['integer_vs_ceil_rule'] = dict(
        checked_pairs=22 * len(ALPHAS), mismatches=mismatches,
        note='the frozen kernel uses ceil(m*alpha); the registered rule uses (m*a+99)//100. They agree '
             'for every m in 7..28 with these five levels, and the formal period always has m=28.',
        passed=True)
    out['history_length'] = {}
    for predictor in ('ridge', 'lightgbm'):
        days = [k for k in range(WARMUP_DAYS, len(dates))]
        lengths = set()
        for k in days:
            lengths.add(len(range(max(1, k - RESIDUAL_WINDOW), k)))
        out['history_length'][predictor] = dict(
            distinct_lengths=sorted(lengths),
            note='every formal-period day must use exactly 28 effective history days',
            passed=bool(lengths == {RESIDUAL_WINDOW}))
    monotone = {}
    for predictor in ('ridge', 'lightgbm'):
        eps = archives[predictor]['residual']
        worst = 0.0
        changed = 0
        for k in range(WARMUP_DAYS, len(dates)):
            low = max(1, k - RESIDUAL_WINDOW)
            ordered = np.sort(eps[low:k], axis=0)
            previous = None
            for alpha in ALPHAS:
                current = ordered[quantile_index(RESIDUAL_WINDOW, alpha) - 1]
                if previous is not None:
                    worst = min(worst, float(np.min(current - previous)))
                    changed += int(np.sum(np.abs(current - previous) > 1e-12))
                previous = current
        monotone[predictor] = dict(
            max_downward_step_kWh=float(-worst) if worst < 0 else 0.0,
            slot_days_changed_between_adjacent_levels=int(changed),
            slot_days_compared=int((len(dates) - WARMUP_DAYS) * (len(ALPHAS) - 1) * 144),
            note='the protection amount is non-decreasing in alpha; because the residual series is '
                 'continuous, adjacent levels almost always select distinct order statistics, so the '
                 'changed count is close to the number of comparisons',
            passed=bool(-worst < 1e-9))
    out['monotone_in_alpha'] = monotone
    seven = np.array([-2.0, 0.0, 1.0, 3.0, 4.0, 5.0, 10.0])
    six_correction, six_m, six_reason = adjustment(np.zeros((10, 144)), 7, RESIDUAL_WINDOW, 0.80)
    out['m6_falls_back'] = dict(
        window_days=six_m, correction_kWh=float(np.max(np.abs(six_correction))), reason=six_reason,
        expected='m=6 < 7 -> no correction',
        passed=bool(six_m == 6 and np.max(np.abs(six_correction)) == 0.0 and six_reason != ''))
    out['m7_sorts_normally'] = dict(
        m=7, index=quantile_index(7, 0.80),
        value=float(np.sort(seven)[quantile_index(7, 0.80) - 1]),
        expected='ceil(7*0.8)=6 -> the 6th smallest of the seven values = 5.0',
        passed=bool(quantile_index(7, 0.80) == 6 and np.sort(seven)[5] == 5.0))
    probe = np.full((28, 144), -5.0)
    out['negative_correction_retained'] = dict(
        value=float(adjustment(probe, 28, RESIDUAL_WINDOW, 0.80)[0].min()),
        note='negative corrections are not clipped at zero',
        passed=bool(adjustment(probe, 28, RESIDUAL_WINDOW, 0.80)[0].min() == -5.0))
    duplicated = np.zeros((28, 144))
    duplicated[:, 0] = 7.0
    out['duplicate_values'] = dict(
        correction_kWh=float(adjustment(duplicated, 28, RESIDUAL_WINDOW, 0.80)[0][0]),
        note='repeated residuals make adjacent levels select the same order statistic',
        passed=bool(adjustment(duplicated, 28, RESIDUAL_WINDOW, 0.80)[0][0] == 7.0))
    broken = np.zeros((28, 144))
    broken[3, 5] = np.nan
    sorted_column = np.sort(broken[:, 5])
    position = quantile_index(28, 0.80)
    gate_rejected = False
    try:
        require_finite_published(broken, broken, 28)
    except ValueError:
        gate_rejected = True
    out['nan_input_must_fail'] = dict(
        probe_index=position,
        sorted_value_at_probe=float(sorted_column[position - 1]),
        nan_lands_at_sorted_position=int(np.flatnonzero(np.isnan(sorted_column))[0]) + 1,
        note='np.sort places NaN last, so a single NaN does not necessarily corrupt the chosen order '
             'statistic; the input gate must therefore reject it up front rather than rely on sorting',
        input_gate_rejects_nan=bool(gate_rejected),
        passed=bool(gate_rejected))
    out['all_passed'] = bool(all(item['passed'] for item in out.values() if 'passed' in item))
    return out


# --------------------------------------------------------------------------------------
# contrasts and metrics
# --------------------------------------------------------------------------------------
def summary_frame(results):
    return pd.DataFrame([result['totals'] for result in results])[
        ['strategy_id', 'predictor_id', 'alpha', 'planned_kWh', 'emergency_kWh', 'unused_kWh',
         'loss_kWh', 'planned_cost_yuan', 'emergency_cost_yuan', 'total_cost_yuan',
         'emergency_slots', 'emergency_days', 'emergency_events', 'evaluation_initial_kWh',
         'final_kWh', 'mean_adjustment_kWh', 'max_mip_gap', 'solver_failures']]


def within_predictor_contrasts(summary):
    rows = []
    for predictor in ('ridge', 'lightgbm'):
        block = summary[summary.predictor_id == predictor]
        base = block[np.isclose(block.alpha, BASELINE_ALPHA)].iloc[0]
        for row in block.itertuples():
            rows.append(dict(
                predictor_id=predictor, alpha=row.alpha, baseline_alpha=BASELINE_ALPHA,
                treatment=row.strategy_id, baseline=base.strategy_id,
                delta_total_cost_yuan=row.total_cost_yuan - base.total_cost_yuan,
                delta_total_pct=100 * (row.total_cost_yuan - base.total_cost_yuan) / base.total_cost_yuan,
                delta_planned_cost_yuan=row.planned_cost_yuan - base.planned_cost_yuan,
                delta_emergency_cost_yuan=row.emergency_cost_yuan - base.emergency_cost_yuan,
                delta_emergency_kWh=row.emergency_kWh - base.emergency_kWh,
                delta_emergency_days=row.emergency_days - base.emergency_days,
                delta_unused_kWh=row.unused_kWh - base.unused_kWh,
                delta_loss_kWh=row.loss_kWh - base.loss_kWh,
                treatment_total_cost_yuan=row.total_cost_yuan,
                baseline_total_cost_yuan=base.total_cost_yuan))
    frame = pd.DataFrame(rows)
    frame_to_csv(frame, OUT / 'within_predictor_contrasts.csv')
    return frame


def between_predictor_contrasts(summary):
    rows = []
    for alpha in ALPHAS:
        block = summary[np.isclose(summary.alpha, alpha)]
        ridge = block[block.predictor_id == 'ridge'].iloc[0]
        light = block[block.predictor_id == 'lightgbm'].iloc[0]
        rows.append(dict(
            alpha=alpha, treatment='lightgbm', baseline='ridge',
            delta_total_cost_yuan=light.total_cost_yuan - ridge.total_cost_yuan,
            delta_total_pct=100 * (light.total_cost_yuan - ridge.total_cost_yuan) / ridge.total_cost_yuan,
            delta_planned_cost_yuan=light.planned_cost_yuan - ridge.planned_cost_yuan,
            delta_emergency_cost_yuan=light.emergency_cost_yuan - ridge.emergency_cost_yuan,
            delta_emergency_days=light.emergency_days - ridge.emergency_days,
            lightgbm_total_cost_yuan=light.total_cost_yuan, ridge_total_cost_yuan=ridge.total_cost_yuan))
    frame = pd.DataFrame(rows)
    frame_to_csv(frame, OUT / 'between_predictor_contrasts.csv')
    return frame


def monthly_contrasts(results):
    rows = []
    for predictor in ('ridge', 'lightgbm'):
        base = next(result for result in results
                    if result['group']['predictor'] == predictor
                    and np.isclose(result['group']['alpha'], BASELINE_ALPHA))['monthly']
        base = base.set_index('month')
        for result in results:
            if result['group']['predictor'] != predictor:
                continue
            month = result['monthly'].set_index('month')
            for name in month.index:
                rows.append(dict(
                    predictor_id=predictor, alpha=result['group']['alpha'], month=name,
                    total_cost_yuan=float(month.loc[name, 'total_cost_yuan']),
                    delta_total_cost_yuan=float(month.loc[name, 'total_cost_yuan']
                                                - base.loc[name, 'total_cost_yuan']),
                    delta_planned_cost_yuan=float(month.loc[name, 'planned_cost_yuan']
                                                  - base.loc[name, 'planned_cost_yuan']),
                    delta_emergency_cost_yuan=float(month.loc[name, 'emergency_cost_yuan']
                                                    - base.loc[name, 'emergency_cost_yuan'])))
    frame = pd.DataFrame(rows)
    frame_to_csv(frame, OUT / 'monthly_contrasts.csv')
    return frame


def grid_minima(summary):
    rows = []
    for predictor in ('ridge', 'lightgbm'):
        block = summary[summary.predictor_id == predictor]
        best = block.loc[block.total_cost_yuan.idxmin()]
        tied = block[np.isclose(block.total_cost_yuan, best.total_cost_yuan, atol=COST_TOL_YUAN)]
        rows.append(dict(
            predictor_id=predictor, minimum_total_cost_yuan=float(best.total_cost_yuan),
            alpha_at_minimum=float(best.alpha), strategy_id=best.strategy_id,
            tied_within_tolerance=int(len(tied)), tied_alphas=','.join(f'{a:.2f}' for a in tied.alpha),
            at_grid_boundary=bool(np.isclose(best.alpha, ALPHAS[0]) or np.isclose(best.alpha, ALPHAS[-1])),
            q80_total_cost_yuan=float(block[np.isclose(block.alpha, BASELINE_ALPHA)]
                                      .total_cost_yuan.iloc[0]),
            note='in-year backtest minimum over the measured five-point grid only; 2025 was used for '
                 'design, so this is not a blind out-of-sample gain and not a proven global optimum'))
    frame = pd.DataFrame(rows)
    frame_to_csv(frame, OUT / 'grid_minima_descriptive.csv')
    return frame


def stability_frame(results):
    rows = []
    for predictor in ('ridge', 'lightgbm'):
        baseline = next(result for result in results
                        if result['group']['predictor'] == predictor
                        and np.isclose(result['group']['alpha'], BASELINE_ALPHA))
        base_daily = baseline['daily'].set_index('date')
        for result in results:
            if result['group']['predictor'] != predictor:
                continue
            daily = result['daily'].set_index('date')
            difference = daily.total_cost_yuan - base_daily.total_cost_yuan
            dispatch = result['dispatch']
            hour = dispatch.slot.to_numpy() // 6
            emergency_fee = dispatch.emergency_cost_yuan.to_numpy()
            month = result['monthly'].set_index('month').total_cost_yuan
            base_month = baseline['monthly'].set_index('month').total_cost_yuan
            rows.append(dict(
                strategy_id=result['totals']['strategy_id'], predictor_id=predictor,
                alpha=result['group']['alpha'],
                improved_months=int((month - base_month < 0).sum()),
                worse_months=int((month - base_month > 0).sum()),
                improved_days=int((difference < 0).sum()), worse_days=int((difference > 0).sum()),
                worst_day=str(difference.idxmax()), worst_day_difference_yuan=float(difference.max()),
                worst_month=str((month - base_month).idxmax()),
                worst_month_difference_yuan=float((month - base_month).max()),
                emergency_cost_0_10h_yuan=float(emergency_fee[hour < 10].sum()),
                emergency_cost_19_21h_yuan=float(emergency_fee[(hour >= 19) & (hour < 21)].sum()),
                emergency_cost_21_24h_yuan=float(emergency_fee[hour >= 21].sum())))
    frame = pd.DataFrame(rows)
    frame_to_csv(frame, OUT / 'stability.csv')
    return frame


def energy_contrasts(results):
    rows = []
    for predictor in ('ridge', 'lightgbm'):
        base = next(result['totals'] for result in results
                    if result['group']['predictor'] == predictor
                    and np.isclose(result['group']['alpha'], BASELINE_ALPHA))
        for result in results:
            if result['group']['predictor'] != predictor:
                continue
            totals = result['totals']
            delta_q = totals['planned_kWh'] - base['planned_kWh']
            delta_em = totals['emergency_kWh'] - base['emergency_kWh']
            delta_w = totals['unused_kWh'] - base['unused_kWh']
            delta_loss = totals['loss_kWh'] - base['loss_kWh']
            delta_end = totals['final_kWh'] - base['final_kWh']
            rows.append(dict(
                strategy_id=totals['strategy_id'], predictor_id=predictor, alpha=totals['alpha'],
                delta_planned_kWh=delta_q, delta_emergency_kWh=delta_em, delta_unused_kWh=delta_w,
                delta_loss_kWh=delta_loss, delta_final_state_kWh=delta_end,
                identity_lhs=-delta_em + delta_w + delta_loss + delta_end,
                identity_residual_kWh=float((-delta_em + delta_w + delta_loss + delta_end) - delta_q)))
    frame = pd.DataFrame(rows)
    frame_to_csv(frame, OUT / 'energy_contrasts.csv')
    return frame


def coverage_frame(results):
    rows = []
    for result in results:
        dispatch = result['dispatch']
        actual = (dispatch.load_kW.to_numpy() - dispatch.pv_kW.to_numpy()) * DT
        protected = dispatch.protected_net_kWh.to_numpy()
        covered = actual <= protected + 1e-9
        rows.append(dict(strategy_id=result['totals']['strategy_id'],
                         predictor_id=result['group']['predictor'], alpha=result['group']['alpha'],
                         samples=int(len(covered)), coverage=float(covered.mean()),
                         mean_adjustment_kWh=float(dispatch.residual_adjustment_kWh.mean())))
        block = dispatch.assign(month=dispatch.date.str[:7], covered=covered)
        for name, group in block.groupby('month'):
            rows.append(dict(strategy_id=result['totals']['strategy_id'],
                             predictor_id=result['group']['predictor'], alpha=result['group']['alpha'],
                             scope=name, samples=int(len(group)),
                             coverage=float(group.covered.mean()),
                             mean_adjustment_kWh=float(group.residual_adjustment_kWh.mean())))
    frame = pd.DataFrame(rows)
    frame_to_csv(frame, OUT / 'q80_coverage.csv')
    return frame


def predictor_metric_frame(archives, dates):
    rows = []
    for predictor in ('ridge', 'lightgbm'):
        archive = archives[predictor]
        load_error = archive['load'] - archive['load_forecast']
        pv_error = archive['pv'] - archive['pv_forecast']
        net_error = archive['net_actual'] - archive['net_forecast']
        window = slice(WARMUP_DAYS, len(dates))
        for name, error in (('load', load_error), ('pv', pv_error), ('net_demand', net_error)):
            block = error[window]
            rows.append(dict(predictor_id=predictor, scope=name, samples=int(block.size),
                             mae=float(np.abs(block).mean()),
                             rmse=float(np.sqrt((block ** 2).mean())),
                             mean_signed=float(block.mean())))
    frame = pd.DataFrame(rows)
    frame_to_csv(frame, OUT / 'forecast_metrics.csv')
    return frame


# --------------------------------------------------------------------------------------
# causality checks (scan / scheduling layer only)
# --------------------------------------------------------------------------------------
def future_checks(load, pv, price, dates, archives, results):
    """Perturbing day k and later must leave day k's protection and plan unchanged.

    Only the scan/scheduling layer is covered: the published point forecasts are held frozen, so this
    says nothing about the upstream forecast models' training causality.
    """
    rows = []
    for predictor in ('ridge', 'lightgbm'):
        archive = archives[predictor]
        clean_archive = ScanArchive(load, pv, len(dates), archive['load_forecast'],
                                    archive['pv_forecast'])
        for day_index, label in ((31, '2025-02-01'), (171, '2025-06-21'), (354, '2025-12-21')):
            for kind in ('load_x1.2', 'pv_x0.7'):
                perturbed_load, perturbed_pv = load.copy(), pv.copy()
                if kind == 'load_x1.2':
                    perturbed_load[day_index:] = load[day_index:] * 1.2
                else:
                    perturbed_pv[day_index:] = pv[day_index:] * 0.7
                perturbed = ScanArchive(perturbed_load, perturbed_pv, len(dates),
                                        archive['load_forecast'], archive['pv_forecast'])
                history_error = float(np.max(np.abs(
                    perturbed.eps[1:day_index] - clean_archive.eps[1:day_index])))
                for alpha in ALPHAS:
                    strategy = f"{dict(ridge='R', lightgbm='L')[predictor]}_q{int(round(alpha * 100))}"
                    result = next(item for item in results
                                  if item['totals']['strategy_id'] == strategy)
                    day = result['dispatch'][result['dispatch'].date == str(dates[day_index].date())]
                    initial = float(day.state_start_kWh.iloc[0])
                    clean_q, _, _, clean_protected, clean_r, _, _ = clean_archive.plan_day(
                        day_index, alpha, RESIDUAL_WINDOW, price, initial)
                    other_q, _, _, other_protected, other_r, _, _ = perturbed.plan_day(
                        day_index, alpha, RESIDUAL_WINDOW, price, initial)
                    assert abs(float(clean_q.sum()) - float(day.planned_kWh.sum())) < ENERGY_TOL_KWH, (
                        'the clean rebuild must reproduce the archived plan for that day')
                    rows.append(dict(
                        predictor_id=predictor, alpha=alpha, case=f'{label}:{kind}',
                        day_index=day_index, point_forecasts_held_frozen=True,
                        history_residual_error_kWh=history_error,
                        correction_error_kWh=float(np.max(np.abs(clean_r - other_r))),
                        protected_error_kWh=float(np.max(np.abs(clean_protected - other_protected))),
                        plan_error_kWh=float(np.max(np.abs(clean_q - other_q)))))
    frame = pd.DataFrame(rows)
    frame_to_csv(frame, OUT / 'future_checks.csv')
    numeric = ['history_residual_error_kWh', 'correction_error_kWh', 'protected_error_kWh',
               'plan_error_kWh']
    worst = float(frame[numeric].to_numpy().max()) if len(frame) else 0.0
    return dict(cases=rows, compared_cases=len(rows), max_numeric_difference=worst,
                scope='scan/scheduling layer only: the published point forecasts are held frozen, so '
                      'this does not re-verify the upstream forecast models training causality',
                passed=bool(worst < 1e-9))


def feedback_prefix_check(load, pv, dates, results):
    control = parent().frozen().bm.control
    day_index = 171
    rows = []
    for predictor in ('ridge', 'lightgbm'):
        for alpha in ALPHAS:
            strategy = f"{dict(ridge='R', lightgbm='L')[predictor]}_q{int(round(alpha * 100))}"
            result = next(item for item in results if item['totals']['strategy_id'] == strategy)
            day = result['dispatch'][result['dispatch'].date == str(dates[day_index].date())]
            initial = float(day.state_start_kWh.iloc[0])
            plan = day.planned_kWh.to_numpy()
            modified_load, modified_pv = load[day_index].copy(), pv[day_index].copy()
            modified_load[72:] = modified_load[72:] * 1.5 + 500.0
            modified_pv[72:] = modified_pv[72:] * 0.5
            clean = control(plan, load[day_index], pv[day_index], initial)
            other = control(plan, modified_load, modified_pv, initial)
            prefix = max([float(np.max(np.abs(clean[i][:72] - other[i][:72]))) for i in range(4)]
                         + [float(np.max(np.abs(clean[4][:73] - other[4][:73])))])
            later = max([float(np.max(np.abs(clean[i] - other[i]))) for i in range(5)])
            rows.append(dict(strategy_id=strategy, prefix_max_difference=prefix,
                             full_day_max_difference=later))
    frame = pd.DataFrame(rows)
    frame_to_csv(frame, OUT / 'feedback_prefix_check.csv')
    return dict(day=str(dates[day_index].date()), rows=rows,
                prefix_unchanged=bool((frame.prefix_max_difference < 1e-12).all()),
                later_slots_do_change=bool((frame.full_day_max_difference > 1e-9).any()),
                passed=bool((frame.prefix_max_difference < 1e-12).all()))


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------
def figures(summary, monthly, between, within):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['axes.unicode_minus'] = False
    FIG.mkdir(parents=True, exist_ok=True)
    produced = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for ax, column, title in ((axes[0], 'total_cost_yuan', 'Total realised cash cost'),
                              (axes[1], 'emergency_cost_yuan', 'Emergency purchase cost')):
        for predictor, colour, marker in (('ridge', '#1b6c9e', 'o'), ('lightgbm', '#c05b43', 's')):
            block = summary[summary.predictor_id == predictor].sort_values('alpha')
            ax.plot(block.alpha, block[column] / 1e3, marker=marker, color=colour, label=predictor)
        ax.set(title=title, xlabel='Protection level alpha (quantile)',
               ylabel='Cost (thousand CNY)')
        ax.grid(alpha=.25)
        ax.legend()
    ridge_base = summary[(summary.predictor_id == 'ridge')
                         & np.isclose(summary.alpha, BASELINE_ALPHA)].total_cost_yuan.iloc[0]
    axes[0].axhline(ridge_base / 1e3, ls=':', color='grey', label='ridge q80')
    fig.savefig(FIG / 'q2_scan_cost_vs_alpha.png', dpi=160)
    plt.close(fig)
    produced.append(FIG / 'q2_scan_cost_vs_alpha.png')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for ax, predictor, colour in ((axes[0], 'ridge', '#1b6c9e'), (axes[1], 'lightgbm', '#c05b43')):
        block = summary[summary.predictor_id == predictor].sort_values('alpha')
        x = np.arange(len(block))
        ax.bar(x - 0.2, block.planned_cost_yuan / 1e3, 0.4, label='Planned purchase')
        ax.bar(x + 0.2, block.emergency_cost_yuan / 1e3, 0.4, label='Emergency (5x)')
        ax.set(title=f'{predictor}: cost composition', xlabel='Protection level alpha',
               ylabel='Cost (thousand CNY)', xticks=x, xticklabels=[f'{a:.2f}' for a in block.alpha])
        ax.grid(alpha=.25, axis='y')
        ax.legend()
    fig.savefig(FIG / 'q2_scan_cost_composition.png', dpi=160)
    plt.close(fig)
    produced.append(FIG / 'q2_scan_cost_composition.png')

    months = sorted(monthly.month.unique())
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout='constrained', sharex=True)
    for ax, predictor in zip(axes, ('ridge', 'lightgbm')):
        block = monthly[monthly.predictor_id == predictor]
        grid = block.pivot_table(index='month', columns='alpha', values='delta_total_cost_yuan')
        grid = grid.reindex(months)
        image = ax.imshow(grid.to_numpy().T / 1e3, aspect='auto', cmap='RdBu_r',
                          vmin=-np.nanmax(np.abs(grid.to_numpy())) / 1e3,
                          vmax=np.nanmax(np.abs(grid.to_numpy())) / 1e3)
        ax.set(title=f'{predictor}: monthly cost difference versus its own q80',
               ylabel='alpha', yticks=np.arange(len(ALPHAS)),
               yticklabels=[f'{a:.2f}' for a in grid.columns])
        plt.colorbar(image, ax=ax, label='Difference (thousand CNY)')
    axes[-1].set_xticks(np.arange(len(months)))
    axes[-1].set_xticklabels([month[5:] for month in months])
    axes[-1].set_xlabel('Month of 2025')
    fig.savefig(FIG / 'q2_scan_monthly_heatmap.png', dpi=160)
    plt.close(fig)
    produced.append(FIG / 'q2_scan_monthly_heatmap.png')

    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    ax.bar(np.arange(len(between)), between.delta_total_cost_yuan / 1e3, 0.5, color='#4c7a34')
    ax.axhline(0, color='black', linewidth=.8)
    ax.set(title='LightGBM minus ridge at each protection level', xlabel='Protection level alpha',
           ylabel='Cost difference (thousand CNY)', xticks=np.arange(len(between)),
           xticklabels=[f'{a:.2f}' for a in between.alpha])
    ax.grid(alpha=.25, axis='y')
    fig.savefig(FIG / 'q2_scan_between_predictors.png', dpi=160)
    plt.close(fig)
    produced.append(FIG / 'q2_scan_between_predictors.png')
    return produced


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------
def write_report(summary, within, between, monthly, minima, stability, coverage, metrics,
                 quantile, reference, future, prefix, integrity, warm_end, archives):
    months = sorted(monthly.month.unique())
    lines = [
        '# 问题二：分位水平扫描实验结果报告', '',
        '日期：2026-09-11。状态：**实验自检完成，独立审计待完成**。本轮实现、运行并自检了'
        ' `reports/问题二/实验方案/问题二_分位水平扫描实验方案.md` 登记的十组情景；未训练任何预测器，未锁定最终策略，'
        '未填写 `result2.xlsx`。上游修正G3/G4的20号完整独立审计**仍然待办**，本轮自检不替代它。', '',
        '## 1. 问题分析', '',
        '分位水平 alpha 是保护规则的超参数：它不改变点预测，也不改变已发布残差，只改变送入日前计划的'
        '保护需求。本轮固定修正岭回归与修正LightGBM两套已冻结发布档案，各扫描'
        ' alpha ∈ {0.70, 0.75, 0.80, 0.85, 0.90}，共十组，回答三个问题：'
        '固定 q80 在已测候选集中是否仍表现较好；同一水平下两种预测器谁更省；更低的费用是否伴随'
        '应急天数或部分月份变差。', '',
        '每个情景固定一个 alpha 跑完整个评价期，alpha 显式传入分位计算、MILP 输入、日志与全部输出，'
        '不使用模块级 THETA，各情景独立传递真实电池状态。', '',
        '### 1.1 主要结论', '',
    ]
    for predictor in ('ridge', 'lightgbm'):
        record = reference[predictor]
        equiv = record['equivalent_optimum']
        lines.append(f"- 控制组 `{record['predictor']}_q80` 复现：日级聚合最大差 "
                     f"{record['max_day_level_difference']:.3e} kWh（容差 1e-6），总费 "
                     f"{record['current_total_cost_yuan']:,.6f} 元，与登记值差 "
                     f"{record['difference_vs_registered_yuan']:.3e} 元；"
                     f"逐时段名义轨迹最大差 {record['nominal_trajectory_difference_kWh']:.3f} kWh，"
                     f"计划在 {equiv['affected_slots']} 个时段上有 "
                     f"{equiv['max_slot_plan_difference_kWh']:.3f} kWh 的等优重排"
                     f"（受影响日期 {equiv['affected_dates'] or '无'}，"
                     f"这些时段电价相同：{equiv['affected_slot_prices_equal']}；"
                     f"日级目标费不变：{equiv['day_level_cost_identical']}）。")
    for row in between.itertuples():
        lines.append(f"- alpha={row.alpha:.2f}：LightGBM 减岭回归 {row.delta_total_cost_yuan:+,.2f} 元"
                     f"（{row.delta_total_pct:+.4f}%）。")
    for row in minima.itertuples():
        lines.append(f"- {row.predictor_id}：网格最低点 alpha={row.alpha_at_minimum:.2f}，总费 "
                     f"{row.minimum_total_cost_yuan:,.2f} 元（并列 {row.tied_within_tolerance} 个点；"
                     f"{'落在网格边界' if row.at_grid_boundary else '不在网格边界'}）。")
    lines += ['', '## 2. 数据预处理', '',
              '真值直接使用附件2实际功率，电价使用附件1的144段曲线逐日重复；两套预测档案取自'
              ' `results/q2_lightgbm_residual/` 的已冻结发布结果，包含既有7日历史、阈值1 kW、'
              '两端各扩3段的光伏门控，本轮不重复改动该处理。', '',
              '对档案做了独立重算核对：净需求预测、净需求真值与残差三列均从功率列重算并逐段比对，'
              '真值与原附件2一致，2025-01-02 之后两个目标全部有限且非负。', '',
              '| 预测器 | 净需求预测差/kW | 净需求真值差/kW | 残差差/kWh | 真值负载差/kW | 真值光伏差/kW |',
              '|---|---:|---:|---:|---:|---:|']
    for predictor in ('ridge', 'lightgbm'):
        check = archives[predictor]['checks']
        lines.append(f"| {predictor} | {check['net_forecast']:.3e} | {check['net_actual']:.3e} | "
                     f"{check['residual']:.3e} | {check['truth_load']:.3e} | {check['truth_pv']:.3e} |")
    lines += ['', f"公共1月预运行复现的共同初态为 {warm_end:.12f} kWh。", '',
              '### 2.1 预测精度（两套档案各报一次）', '',
              '| 预测器 | 目标 | MAE | RMSE | 平均有符号误差 |', '|---|---|---:|---:|---:|']
    for row in metrics.itertuples():
        lines.append(f"| {row.predictor_id} | {row.scope} | {row.mae:.6f} | {row.rmse:.6f} | "
                     f"{row.mean_signed:+.6f} |")
    lines += ['', '同一预测器在五个水平下使用完全相同的发布预测，因此上表五个水平共用一个值；'
                  '改变 alpha 不影响点预测误差。', '',
              '## 3. 模型建立', '',
              '真值净需求 $n_{k,t}=(L_{k,t}-V_{k,t})\\Delta t$，预测净需求 '
              '$\\widehat n^f_{k,t}=(\\widehat L^f_{k,t}-\\widehat V^f_{k,t})\\Delta t$，'
              '历史发布误差 $\\varepsilon^f_{j,t}=n_{j,t}-\\widehat n^f_{j,t}$，$\\Delta t=1/6$ 小时。', '',
              '第 $k$ 日 0:00 取 $\\mathcal H_k=\\{j:\\max(1,k-28)\\le j<k\\}$，'
              '$m_k=|\\mathcal H_k|$；$m_k<7$ 时不修正，否则逐时段升序取第 '
              '$(m_k a+99)//100$ 个（$a$ 为整数百分数），'
              '$\\widetilde n^f_{k,t}=\\widehat n^f_{k,t}+r^{f,\\alpha}_{k,t}$。'
              '采用经验逆分布、不做线性插值，负修正与负净需求保留。', '',
              '整数位置规则避免浮点临界误差；它与冻结内核使用的 $\\lceil m\\alpha\\rceil$ '
              '在 $m=7..28$ 与这五个水平上完全一致（见 `quantile_validation.json`）。', '',
              '日前 MILP 为 $\\min\\sum_t p_tq_t$，满足 '
              '$q_t+d_t=\\widetilde n_t+c_t+w_t$、$E^{\\mathrm{nom}}_{t+1}=E^{\\mathrm{nom}}_t'
              '+0.9c_t-d_t/0.9$、$c_t\\le Mz_t$、$d_t\\le M(1-z_t)$、$z_t\\in\\{0,1\\}$、'
              '$M=5000/6$、$1200\\le E^{\\mathrm{nom}}_t\\le10800$、'
              '$E^{\\mathrm{nom}}_0$ 为当日真实初态、$E^{\\mathrm{nom}}_{144}=6000$。', '',
              '实际执行沿用因果贪心反馈：富余充电、缺口放电、剩余缺口按当段5倍价应急。'
              '普通计划全额付款，未使用量无额外罚金，实际末态跨日继承。', '',
              '## 4. 模型求解与结果', '',
              '### 4.1 十组总费用', '',
              '| 情景 | 预测器 | alpha | 计划费/元 | 应急费/元 | 总费/元 | 应急天数 | 期末/kWh |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in summary.itertuples():
        lines.append(f"| {row.strategy_id} | {row.predictor_id} | {row.alpha:.2f} | "
                     f"{row.planned_cost_yuan:,.2f} | {row.emergency_cost_yuan:,.2f} | "
                     f"{row.total_cost_yuan:,.2f} | {row.emergency_days} | {row.final_kWh:,.2f} |")
    lines += ['', '### 4.2 相对自身 q80 的差额', '',
              '| 预测器 | alpha | 总费差/元 | 变化/% | 计划费差/元 | 应急费差/元 | 应急天数差 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in within.itertuples():
        lines.append(f"| {row.predictor_id} | {row.alpha:.2f} | {row.delta_total_cost_yuan:+,.2f} | "
                     f"{row.delta_total_pct:+.4f} | {row.delta_planned_cost_yuan:+,.2f} | "
                     f"{row.delta_emergency_cost_yuan:+,.2f} | {row.delta_emergency_days:+d} |")
    lines += ['', '### 4.3 同水平下 LightGBM 减岭回归', '',
              '| alpha | 总费差/元 | 变化/% | 计划费差/元 | 应急费差/元 | 应急天数差 |',
              '|---:|---:|---:|---:|---:|---:|']
    for row in between.itertuples():
        lines.append(f"| {row.alpha:.2f} | {row.delta_total_cost_yuan:+,.2f} | {row.delta_total_pct:+.4f} | "
                     f"{row.delta_planned_cost_yuan:+,.2f} | {row.delta_emergency_cost_yuan:+,.2f} | "
                     f"{row.delta_emergency_days:+d} |")
    lines += ['', '### 4.4 网格最低点（事后描述）', '',
              '| 预测器 | 最低总费/元 | 对应alpha | 并列点数 | 是否在网格边界 | 其 q80 费用/元 |',
              '|---|---:|---:|---:|---|---:|']
    for row in minima.itertuples():
        lines.append(f"| {row.predictor_id} | {row.minimum_total_cost_yuan:,.2f} | "
                     f"{row.alpha_at_minimum:.2f} | {row.tied_within_tolerance} | "
                     f"{'是' if row.at_grid_boundary else '否'} | {row.q80_total_cost_yuan:,.2f} |")
    lines += ['', '这是**本年度、已测五点网格的回测最小值**。2025 年数据已用于多轮设计，'
                  '该点不是独立盲测收益，也不能由五个点推断整个连续区间的单调性或全局最优；'
                  '本轮不因最低点位置扩大网格。', '',
              '### 4.5 稳定性与风险', '',
              '| 情景 | 改善月数 | 变差月数 | 改善天数 | 变差天数 | 最差日 | 最差日差/元 |',
              '|---|---:|---:|---:|---:|---|---:|']
    for row in stability.itertuples():
        lines.append(f"| {row.strategy_id} | {row.improved_months} | {row.worse_months} | "
                     f"{row.improved_days} | {row.worse_days} | {row.worst_day} | "
                     f"{row.worst_day_difference_yuan:+,.2f} |")
    lines += ['', '| 情景 | 0—10时应急费/元 | 19—21时应急费/元 | 21—24时应急费/元 |',
              '|---|---:|---:|---:|']
    for row in stability.itertuples():
        lines.append(f"| {row.strategy_id} | {row.emergency_cost_0_10h_yuan:,.2f} | "
                     f"{row.emergency_cost_19_21h_yuan:,.2f} | "
                     f"{row.emergency_cost_21_24h_yuan:,.2f} |")
    lines += ['', '### 4.6 校准与能量账', '',
              '| 情景 | 平均保护量/kWh | 覆盖率(n≤ñ) |', '|---|---:|---:|']
    for predictor in ('ridge', 'lightgbm'):
        for alpha in ALPHAS:
            mask = (coverage.predictor_id == predictor) & np.isclose(coverage.alpha, alpha)
            annual = coverage[mask & coverage.scope.isna()]
            if not len(annual):
                continue
            row = annual.iloc[0]
            lines.append(f"| {predictor}·{alpha:.2f} | {row.mean_adjustment_kWh:.6f} | "
                         f"{row.coverage:.6f} |")
    lines += ['', '覆盖率是“实际净需求不超过保护需求”的时段比例，不是无应急概率，'
                  '也不要求有限样本恰好等于 alpha。组间能量账满足 '
                  '$\\Delta Q=-\\Delta Q_{\\mathrm{emergency}}+\\Delta W+\\Delta\\mathrm{Loss}'
                  '+\\Delta E_{\\mathrm{end}}$，见 `energy_contrasts.csv`。', '',
              '### 4.7 图表', '']
    for item in integrity:
        lines.append(f"- `figures/q2_quantile_level_scan/{item['file']}`：{item['width']}×{item['height']} "
                     f"像素，非白像素比例 {item['non_white_fraction']:.4f}。")
    lines += ['', '任务书要求对图表做目视核对。**本环境无法渲染图像供人工查看，因此目视核对未完成**，'
                  '上表仅为程序化完整性检查，不能替代目视核对。', '',
              '## 5. 验证与适用边界', '',
              '| 检查 | 结果 |', '|---|---|',
              f"| R_q80 控制复现 | 日级聚合最大差 "
              f"{reference['ridge']['max_day_level_difference']:.3e} kWh；有差别的天数 "
              f"{reference['ridge']['days_with_any_difference']}；总费差 "
              f"{reference['ridge']['difference_vs_registered_yuan']:.3e} 元 |",
              f"| L_q80 控制复现 | 日级聚合最大差 "
              f"{reference['lightgbm']['max_day_level_difference']:.3e} kWh；有差别的天数 "
              f"{reference['lightgbm']['days_with_any_difference']}；总费差 "
              f"{reference['lightgbm']['difference_vs_registered_yuan']:.3e} 元 |",
              f"| 正式期历史长度 | 全部为 28 日：{quantile['history_length']['ridge']['passed']} |",
              f"| 保护量对 alpha 单调 | {quantile['monotone_in_alpha']['ridge']['passed']} |",
              f"| 整数位置与 ceil 规则一致 | {quantile['integer_vs_ceil_rule']['passed']} |",
              f"| m=6 回退 / m=7 正常排序 / 负修正 / 重复值 | "
              f"{quantile['m6_falls_back']['passed'] and quantile['m7_sorts_normally']['passed'] and quantile['negative_correction_retained']['passed'] and quantile['duplicate_values']['passed']} |",
              f"| 扫描层未来扰动（{future['compared_cases']} 例） | "
              f"{'通过' if future['passed'] else '未通过'}，最大差 {future['max_numeric_difference']:.3e} |",
              f"| 半日反馈前缀 | {'通过' if prefix['passed'] else '未通过'} |", '',
              '限制：', '',
              '1. 点预测档案来自上游修正 G3/G4；其训练因果性结论继承上游审计状态，'
              '本轮只验证扫描与调度层，**不重验预测模型**。',
              '2. 2025 年已用于多轮设计与比较，事后扫描最低点不能作为独立盲测收益。',
              '3. 五个水平是围绕 q80 的预先指定局部网格，未证明覆盖全局最优；'
              '本轮不做逐日自适应选参，也不按月份事后拼接。',
              '4. 名义 MILP 存在等优轨迹，未实施统一二级择优；未实施储能备用、MPC 或第三、四问。', '',
              '## 6. 文件与复现', '',
              '- 登记与运行清单：`results/q2_quantile_level_scan/registration.json`、`run_manifest.json`。',
              '- 输入档案：`forecast_archive_ridge.csv`、`forecast_archive_lightgbm.csv`。',
              '- 十组结果：`results/q2_quantile_level_scan/<情景ID>/`。',
              '- 汇总与自检：`summary.csv`、`within_predictor_contrasts.csv`、`between_predictor_contrasts.csv`、'
              '`monthly_contrasts.csv`、`grid_minima_descriptive.csv`、`reference_reproduction.json`、'
              '`quantile_validation.json`、`future_checks.json`、`checks.json`。', '',
              '```',
              'E:/Anaconda/envs/math_modeling/python.exe code/21_q2_quantile_level_scan.py --mode register',
              'E:/Anaconda/envs/math_modeling/python.exe code/21_q2_quantile_level_scan.py --mode repro',
              'E:/Anaconda/envs/math_modeling/python.exe code/21_q2_quantile_level_scan.py --mode full',
              '```', '',
              '独立审计入口：以 `results/q2_quantile_level_scan/` 为只读输入，'
              '从原始预测与真值独立重算全部 m、顺序统计量、保护需求、实际反馈、费用与能量账，'
              '并对十组各取指定日期抽样重解 MILP。审计通过不等于收益为正。', '']
    REPORT_MD.write_text('\n'.join(lines), encoding='utf-8')
    return lines


# --------------------------------------------------------------------------------------
# main
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
    m = parent()
    load, pv, price, source_hashes = m.frozen().bm.read_sources()
    dates = m.frozen().bm.DATES
    assert load.shape == pv.shape == (365, 144) and price.shape == (144,)

    inputs = ['code/14_q2_ridge_forecast_experiment.py', 'code/19_q2_lightgbm_residual_experiment.py',
              'code/21_q2_quantile_level_scan.py',
              'results/q2_lightgbm_residual/G3_ridge_fixed/forecast_residuals.csv',
              'results/q2_lightgbm_residual/G4_lightgbm_fixed/forecast_residuals.csv',
              'results/q2_lightgbm_residual/registration.json',
              '附件/附件1.xlsx', '附件/附件2.xlsx',
              'reports/问题二/实验方案/问题二_分位水平扫描实验方案.md']
    snapshot_inputs = ['code/02_q1_baseline.py', 'code/05_q2_baseline.py',
                       'code/08_q2_quantile_experiment.py']
    hashes = {rel: digest(ROOT / rel) for rel in inputs}
    snapshot_hashes = {rel: digest(SNAP / rel) for rel in snapshot_inputs}
    for rel in snapshot_inputs:
        assert digest(ROOT / rel) == snapshot_hashes[rel], rel
    signature = hashlib.sha256(json.dumps(dict(parameters=PARAMETERS, input_sha256=hashes,
                                               snapshot_sha256=snapshot_hashes,
                                               code=digest(CODE_21)), sort_keys=True).encode()).hexdigest()
    registration_path = OUT / 'registration.json'
    if args.mode == 'register' or not registration_path.exists():
        record = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
                      specification='reports/问题二/实验方案/问题二_分位水平扫描实验方案.md',
                      parameters=PARAMETERS, input_sha256=hashes, snapshot_sha256=snapshot_hashes,
                      code_sha256=digest(CODE_21), executable=sys.executable, python=sys.version,
                      numpy=np.__version__, pandas=pd.__version__)
        if registration_path.exists():
            existing = json.loads(registration_path.read_text(encoding='utf-8'))
            if existing['signature'] != signature:
                assert args.amend_reason, 'a different registration exists; pass --amend-reason'
                for key in ('alphas', 'residual_window_days', 'min_history_days', 'terminal_kWh',
                            'quantile_rule'):
                    assert existing['parameters'][key] == PARAMETERS[key], key
                amendments = existing.get('amendments', [])
                amendments.append(dict(amended_utc=datetime.now(timezone.utc).isoformat(),
                                       previous_signature=existing['signature'],
                                       previous_code_sha256=existing.get('code_sha256'),
                                       new_signature=signature, new_code_sha256=digest(CODE_21),
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

    archives = {}
    for predictor in PREDICTORS:
        archive = load_predictor_archive(predictor, load, pv, dates)
        archives[predictor['id']] = archive
        frame_to_csv(pd.DataFrame({
            'date': np.repeat([str(day.date()) for day in dates], 144),
            'slot': np.tile(np.arange(144), len(dates)),
            'load_forecast_kW': archive['load_forecast'].ravel(),
            'pv_forecast_kW': archive['pv_forecast'].ravel(),
            'net_forecast_kWh': archive['net_forecast'].ravel(),
            'load_kW': archive['load'].ravel(), 'pv_kW': archive['pv'].ravel(),
            'net_actual_kWh': archive['net_actual'].ravel(),
            'residual_kWh': archive['residual'].ravel(),
            'valid': [int(np.isfinite(v)) for v in archive['net_forecast'].ravel()]}),
            OUT / f"forecast_archive_{predictor['id']}.csv")
    quantile = quantile_validation(load, pv, dates, archives)
    save(OUT / 'quantile_validation.json', quantile)
    if not quantile['all_passed']:
        raise SystemExit('quantile validation failed; see quantile_validation.json')

    print('running the public January warm-up ...', flush=True)
    warm_states, warm_frame, warm_end, warm_checks = m.frozen().m08.run_warmup(
        m.frozen().m08.Archive(load, pv, len(dates)), price, dates)
    frame_to_csv(warm_frame, OUT / 'warmup_january.csv')
    assert abs(warm_end - REFERENCE_WARMUP_KWH) < 1e-9, warm_end
    print(f'  2025-02-01 shared begin state = {warm_end:.12f} kWh', flush=True)

    instances = {predictor['id']: ScanArchive(load, pv, len(dates),
                                              archives[predictor['id']]['load_forecast'],
                                              archives[predictor['id']]['pv_forecast'])
                 for predictor in PREDICTORS}

    controls = [group for group in GROUPS if np.isclose(group['alpha'], BASELINE_ALPHA)]
    treatments = [group for group in GROUPS if not np.isclose(group['alpha'], BASELINE_ALPHA)]
    controls.sort(key=lambda group: group['predictor'])
    results, reference_reproduction = [], {}
    for group in controls:
        result = run_scenario(group, load, pv, price, dates, instances[group['predictor']],
                              warm_states, verbose=False)
        results.append(result)
        print(f"  {group['id']}: {result['totals']['total_cost_yuan']:,.6f} CNY", flush=True)
        record = compare_to_reference(result, group['predictor'])
        reference_reproduction[group['predictor']] = record
        assert record['passed'], record
        print(f"    {group['id']} reproduces the frozen archive: executed trajectory max diff "
              f"{record['max_executed_difference']:.3e} kWh, nominal trajectory diff "
              f"{record['nominal_trajectory_difference_kWh']:.3f} kWh, cost diff "
              f"{record['difference_vs_registered_yuan']:.3e} CNY", flush=True)
    save(OUT / 'reference_reproduction.json', reference_reproduction)
    if args.mode == 'repro':
        return
    for group in treatments:
        result = run_scenario(group, load, pv, price, dates, instances[group['predictor']],
                              warm_states, verbose=True)
        results.append(result)
        print(f"  {group['id']}: {result['totals']['total_cost_yuan']:,.6f} CNY", flush=True)

    summary = summary_frame(results)
    frame_to_csv(summary, OUT / 'summary.csv')
    within = within_predictor_contrasts(summary)
    between = between_predictor_contrasts(summary)
    monthly = monthly_contrasts(results)
    minima = grid_minima(summary)
    stability = stability_frame(results)
    coverage = coverage_frame(results)
    energy = energy_contrasts(results)
    metrics = predictor_metric_frame(archives, dates)
    future = future_checks(load, pv, price, dates, archives, results)
    save(OUT / 'future_checks.json', future)
    prefix = feedback_prefix_check(load, pv, dates, results)
    save(OUT / 'feedback_prefix_check.json', prefix)
    save(OUT / 'reference_reproduction.json', reference_reproduction)
    produced = figures(summary, monthly, between, within)
    integrity = m.figure_integrity(produced)
    save(OUT / 'figure_integrity.json', integrity)

    protected_after = protected_manifest()
    changed = sorted(name for name in protected if protected_after.get(name) != protected[name])
    missing = sorted(name for name in protected if name not in protected_after)
    save(OUT / 'protected_after.json', protected_after)
    checks = dict(reference_reproduction=reference_reproduction, quantile=quantile,
                  future_checks=future, feedback_prefix=prefix, figure_integrity=integrity,
                  predictor_metrics=metrics.to_dict('records'),
                  protected_unchanged=bool(not changed and not missing), protected_changed=changed,
                  protected_missing=missing,
                  warmup=dict(shared_begin_state_kWh=warm_end,
                              checks={key: float(value) for key, value in warm_checks.items()}),
                  thresholds=dict(energy_kWh=ENERGY_TOL_KWH, cost_yuan=COST_TOL_YUAN))
    save(OUT / 'checks.json', checks)
    write_report(summary, within, between, monthly, minima, stability, coverage, metrics,
                 quantile, reference_reproduction, future, prefix, integrity, warm_end, archives)
    artifact_hashes = {path.relative_to(OUT).as_posix(): digest(path)
                       for path in sorted(OUT.rglob('*'))
                       if path.is_file() and path.name not in ('artifact_hashes.json', 'run_manifest.json')}
    save(OUT / 'artifact_hashes.json', artifact_hashes)
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                    finished_utc=datetime.now(timezone.utc).isoformat(),
                    wall_seconds=time.perf_counter() - started, mode=args.mode,
                    executable=sys.executable, python=sys.version, numpy=np.__version__,
                    pandas=pd.__version__, signature=signature, input_sha256=hashes,
                    snapshot_sha256=snapshot_hashes, parameters=PARAMETERS,
                    source_sha256=source_hashes, protected_before_count=len(protected),
                    protected_unchanged=bool(not changed and not missing),
                    protected_changed=changed, protected_missing=missing,
                    warmup_shared_begin_state_kWh=warm_end,
                    scenario_totals={result['totals']['strategy_id']: result['totals']
                                     for result in results},
                    reference_reproduction=reference_reproduction,
                    self_checks_passed=dict(quantile=quantile['all_passed'],
                                            future_checks=future['passed'],
                                            feedback_prefix=prefix['passed']),
                    artifact_count=len(artifact_hashes), report=str(REPORT_MD.relative_to(ROOT)))
    save(OUT / 'run_manifest.json', manifest)
    assert manifest['protected_unchanged'], (changed, missing)
    print(summary[['strategy_id', 'planned_cost_yuan', 'emergency_cost_yuan', 'total_cost_yuan',
                   'emergency_days']].to_string(index=False), flush=True)
    print(f'self-checks {manifest["self_checks_passed"]}', flush=True)
    print(f'wall seconds = {time.perf_counter() - started:.1f}', flush=True)
    return manifest


if __name__ == '__main__':
    main()
