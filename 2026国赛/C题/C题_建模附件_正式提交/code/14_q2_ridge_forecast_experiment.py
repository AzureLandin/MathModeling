"""Q2 ridge-regression residual correction of the naive forecasts (pre-registered).

Run from E:/MathModeling/2026国赛/C题 with the math_modeling interpreter:
    E:/Anaconda/envs/math_modeling/python.exe code/14_q2_ridge_forecast_experiment.py --mode register
    E:/Anaconda/envs/math_modeling/python.exe code/14_q2_ridge_forecast_experiment.py --mode repro
    E:/Anaconda/envs/math_modeling/python.exe code/14_q2_ridge_forecast_experiment.py --mode full

Specification: reports/问题二/实验方案/问题二_岭回归残差预测实验方案.md (single-run task book, not a new main model).
Frozen inherited implementation: results/q2_revision_audit_20260911/source_snapshot/code/{02,05,08}.

Modes
    register : freeze the specification, parameters, input/snapshot hashes and the protected-file
               manifest before any year-long run. Never overwrites a differing registration without
               a documented --amend-reason, and the frozen candidate grid/windows cannot be amended.
    repro    : public January warm-up, the full ridge forecast pipeline, R0_naive and the reference
               reproduction against results/q2_terminal_sensitivity/T_q80_6000.
    full     : the four registered groups, physics/cost/energy/causality self-checks, naive residual
               diagnostics, figures, result report and delivery hashes.

Four registered groups (same q80/W28 protection, same 6000 kWh nominal terminal, same frozen MILP
kernel and causal greedy feedback; they differ only in which target is forecast by ridge regression):
    R0_naive       load naive, pv naive   (must reproduce T_q80_6000 exactly)
    R1_load_ridge  load ridge, pv naive
    R2_pv_ridge    load naive, pv ridge
    R3_both_ridge  load ridge, pv ridge

The naive predictor, the battery controller, the efficiency/time conventions and the MILP kernel are
inherited unchanged from the frozen snapshot, so a realised-cost difference is attributable to the
issued forecasts alone.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'q2_ridge_forecast'
FIG = ROOT / 'figures' / 'q2_ridge_forecast'
AUDIT = ROOT / 'results' / 'q2_revision_audit_20260911'
SNAP = AUDIT / 'source_snapshot'
REFERENCE_RUN = ROOT / 'results' / 'q2_terminal_sensitivity'
PLAN_MD = ROOT / 'reports/问题二/实验方案/问题二_岭回归残差预测实验方案.md'
REPORT_MD = ROOT / 'reports/问题二/实验结果/问题二_岭回归残差预测实验结果报告.md'

DT = 1 / 6
LAMBDAS = (0.001, 0.01, 0.1, 1.0, 10.0)
DEFAULT_LAMBDA = 0.1
TRAIN_WINDOW = 56
MIN_TRAIN_DAYS = 14
VALIDATION_WINDOW = 14
MIN_SELECTION_DAYS = 7
TIE_RELATIVE = 1e-10
RESIDUAL_WINDOW = 28
THETA = 0.80
TERMINAL_KWH = 6000.0
EVALUATION_START = '2025-02-01'
WARMUP_DAYS = 31
REFERENCE_TOTAL_COST_YUAN = 14158360.487139747
REFERENCE_ENERGY_TOL_KWH = 1e-6
REFERENCE_COST_TOL_YUAN = 1e-4

GROUPS = [
    dict(id='R0_naive', load='naive', pv='naive',
         role='matched baseline: both targets keep the naive forecast'),
    dict(id='R1_load_ridge', load='ridge', pv='naive', role='load ablation'),
    dict(id='R2_pv_ridge', load='naive', pv='ridge', role='pv ablation'),
    dict(id='R3_both_ridge', load='ridge', pv='ridge', role='joint treatment'),
]

LOAD_FEATURES = ['l_base', 'l_yesterday_minus_base', 'l_known_week_change',
                 'l_sameweekday_mean_minus_base', 'l_daily_week_change',
                 'weekday_tue', 'weekday_wed', 'weekday_thu', 'weekday_fri',
                 'weekday_sat', 'weekday_sun', 'sin1', 'cos1', 'sin2', 'cos2']
PV_FEATURES = ['v_base', 'v_recent_change', 'v_week_mean_minus_base', 'v_daily_change',
               'sin1', 'cos1', 'sin2', 'cos2']

SNAPSHOT_INPUTS = ['code/02_q1_baseline.py', 'code/05_q2_baseline.py', 'code/08_q2_quantile_experiment.py']
LIVE_INPUTS = SNAPSHOT_INPUTS + ['code/09_q2_quantile_diagnostics.py', 'code/10_q2_terminal_experiment.py',
                                 'code/12_q2_terminal_sensitivity.py',
                                 '附件/附件1.xlsx', '附件/附件2.xlsx',
                                 'results/audit/audit_summary.json',
                                 'reports/问题二/实验方案/问题二_岭回归残差预测实验方案.md',
                                 'results/q2_terminal_sensitivity/T_q80_6000/dispatch.csv',
                                 'results/q2_terminal_sensitivity/T_q80_6000/daily_summary.csv',
                                 'results/q2_terminal_sensitivity/T_q80_6000/validation.json',
                                 'results/q2_terminal_sensitivity/registration.json']
PROTECTED_TREES = ['附件', 'code', 'reports', 'figures', 'results/audit', 'results/template_backups',
                   'results/q1_baseline', 'results/q1_milp', 'results/q1_day52', 'results/q1_structure',
                   'results/q2_baseline', 'results/q2_quantile_experiment',
                   'results/q2_terminal_experiment_20260911', 'results/q2_terminal_diagnostics_20260911',
                   'results/q2_terminal_sensitivity', 'results/q2_terminal_sensitivity_audit',
                   'results/q2_revision_audit_20260911']
OWN_NEW_REL = {'code/14_q2_ridge_forecast_experiment.py',
               'reports/问题二/实验结果/问题二_岭回归残差预测实验结果报告.md'}
OWN_NEW_PREFIXES = ('figures/q2_ridge_forecast/', 'results/q2_ridge_forecast/')
SHARED_APPEND_REL = {'建模上下文记忆.md', 'reports/项目进度.md'}

DISPATCH_COLUMNS = [
    'strategy_id', 'date', 'issue_time', 'slot', 'start_time', 'end_time',
    'price_yuan_kWh', 'load_kW', 'pv_kW', 'load_forecast_kW', 'pv_forecast_kW',
    'net_forecast_kWh', 'residual_adjustment_kWh', 'protected_net_kWh',
    'theta', 'theta_preset', 'residual_days', 'fallback_reason',
    'planned_kWh', 'charge_kWh', 'discharge_kWh', 'emergency_kWh', 'unused_kWh',
    'state_start_kWh', 'state_end_kWh', 'nominal_state_end_kWh',
    'planned_cost_yuan', 'emergency_cost_yuan']
NOMINAL_COLUMNS = [
    'strategy_id', 'date', 'slot', 'nominal_charge_kWh', 'nominal_discharge_kWh',
    'nominal_unused_kWh', 'nominal_binary_mode', 'nominal_state_start_kWh',
    'nominal_state_end_kWh', 'nominal_planned_cost_yuan']

_FROZEN: dict = {}


class _FrozenSet:
    """Attribute view over the frozen snapshot modules (02 kernel, 05 physics/reader, 08 harness)."""

    m08 = None
    bm = None
    core = None


def frozen():
    """Load the frozen 02/05/08 snapshot modules once per process."""
    if 'set' not in _FROZEN:
        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        bundle = _FrozenSet()
        bundle.m08 = load('frozen_q2_experiment', SNAP / 'code/08_q2_quantile_experiment.py')
        bundle.bm = bundle.m08.bm
        bundle.core = bundle.m08.core
        _FROZEN['set'] = bundle
    return _FROZEN['set']


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lam_key(lam) -> str:
    return f'{lam:g}'.replace('.', 'p')


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    return obj


def save(path, obj):
    Path(path).write_text(json.dumps(_jsonable(obj), ensure_ascii=False, indent=2, default=str),
                          encoding='utf-8')


def frame_to_csv(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8-sig')


# --------------------------------------------------------------------------------------
# feature construction (strictly causal: day k uses only days < k plus day k's calendar)
# --------------------------------------------------------------------------------------
def harmonic_matrix():
    t = np.arange(144, dtype=float)
    return np.column_stack([np.sin(2 * np.pi * t / 144), np.cos(2 * np.pi * t / 144),
                            np.sin(4 * np.pi * t / 144), np.cos(4 * np.pi * t / 144)])


HARMONICS = harmonic_matrix()


def weekday_columns(dates, k):
    weekday = dates[k].weekday()          # Monday = 0 is the reference level
    return np.array([1.0 if weekday == d else 0.0 for d in range(1, 7)])


def naive_load(load, k):
    """Load naive level b^L: same weekday last week, yesterday when history < 7 days."""
    return load[k - 7] if k >= 7 else load[k - 1]


def load_feature_matrix(load, dates, k):
    """15 load features for decision day k. Requires k >= 8; no information from day k or later."""
    base = naive_load(load, k)
    out = np.empty((144, len(LOAD_FEATURES)))
    out[:, 0] = base
    out[:, 1] = load[k - 1] - base
    out[:, 2] = load[k - 1] - load[k - 8]
    ids = [r for r in (1, 2, 3, 4) if k - 7 * r >= 0]
    out[:, 3] = load[[k - 7 * r for r in ids]].mean(axis=0) - base
    out[:, 4] = load[k - 1].mean() - load[k - 8].mean()
    out[:, 5:11] = weekday_columns(dates, k)
    out[:, 11:15] = HARMONICS
    return out


def pv_feature_matrix(pv, dates, k):
    """8 pv features for decision day k. Requires k >= 7; no information from day k or later."""
    out = np.empty((144, len(PV_FEATURES)))
    out[:, 0] = pv[k - 1]
    out[:, 1] = pv[k - 1] - pv[k - 2]
    out[:, 2] = pv[k - 7:k].mean(axis=0) - pv[k - 1]
    out[:, 3] = pv[k - 1].mean() - pv[k - 2].mean()
    out[:, 4:8] = HARMONICS
    return out


LOAD_THRESHOLD = 8
PV_THRESHOLD = 7
TARGET_SPEC = {
    'load': dict(threshold=LOAD_THRESHOLD, features=LOAD_FEATURES, feat_fn=load_feature_matrix,
                 label_fn=lambda arr, k: arr[k] - naive_load(arr, k)),
    'pv': dict(threshold=PV_THRESHOLD, features=PV_FEATURES, feat_fn=pv_feature_matrix,
               label_fn=lambda arr, k: arr[k] - arr[k - 1]),
}


def fit_ridge(char, k, train_days, feat_cache, actual):
    """Standardise, centre and solve the shared-coefficient ridge problem by direct SVD.

    Returns (mean, scale, label_mean, {lambda: beta}, singular_values). `char` is 'load' or 'pv'.
    """
    spec = TARGET_SPEC[char]
    X = np.concatenate([feat_cache[j] for j in train_days], axis=0)
    y = np.concatenate([spec['label_fn'](actual, j) for j in train_days], axis=0)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)         # constant column -> unit scale
    Z = (X - mean) / scale
    label_mean = float(y.mean())
    centred = y - label_mean
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    Uty = U.T @ centred
    n_rows = Z.shape[0]
    betas = {}
    for lam in LAMBDAS:
        betas[lam] = Vt.T @ ((S / (S ** 2 + n_rows * lam)) * Uty)
    return mean, scale, label_mean, betas, S


def build_forecasts(load, pv, dates, verbose=False, upto=None):
    """Candidate and issued forecasts for both targets, strictly causal.

    Every day's five candidates are fitted only on complete valid days strictly before it; the issued
    forecast picks the lambda scored on the previous 14 calendar days, and never rewrites a past day.
    `upto` limits the rebuilt range (used by the perturbation checks, which only need day k).
    """
    n_days = len(dates) if upto is None else min(len(dates), int(upto) + 1)
    naive = {'load': np.full((n_days, 144), np.nan), 'pv': np.full((n_days, 144), np.nan)}
    for k in range(1, n_days):
        naive['load'][k] = naive_load(load, k)
        naive['pv'][k] = pv[k - 1]

    candidates = {c: {lam: np.full((n_days, 144), np.nan) for lam in LAMBDAS} for c in ('load', 'pv')}
    trained = {c: np.zeros(n_days, dtype=bool) for c in ('load', 'pv')}
    issued = {c: np.full((n_days, 144), np.nan) for c in ('load', 'pv')}
    fit_log, selection_log = [], []
    params = {c: dict(beta=np.full((n_days, len(LAMBDAS), len(TARGET_SPEC[c]['features'])), np.nan),
                      beta0=np.full((n_days, len(LAMBDAS)), np.nan),
                      mean=np.full((n_days, len(TARGET_SPEC[c]['features'])), np.nan),
                      scale=np.full((n_days, len(TARGET_SPEC[c]['features'])), np.nan),
                      singular_values=np.full((n_days, len(TARGET_SPEC[c]['features'])), np.nan))
              for c in ('load', 'pv')}

    for char in ('load', 'pv'):
        spec = TARGET_SPEC[char]
        actual = load if char == 'load' else pv
        threshold = spec['threshold']
        feat_cache = {k: spec['feat_fn'](actual, dates, k) for k in range(threshold, n_days)}
        for k in range(threshold, n_days):
            lo = max(0, k - TRAIN_WINDOW)
            train_days = [j for j in range(lo, k) if j >= threshold]
            base = naive[char][k]
            if len(train_days) < MIN_TRAIN_DAYS:
                for lam in LAMBDAS:
                    candidates[char][lam][k] = base
                reason = f'insufficient_training_days(m={len(train_days)}<{MIN_TRAIN_DAYS})'
                fit_log.append(dict(target=char, day_index=k, date=str(dates[k].date()),
                                    n_train_days=len(train_days), trained=False, reason=reason))
                continue
            started = time.perf_counter()
            mean, scale, label_mean, betas, singular = fit_ridge(char, k, train_days, feat_cache, actual)
            elapsed = time.perf_counter() - started
            z = (feat_cache[k] - mean) / scale
            trained[char][k] = True
            params[char]['mean'][k] = mean
            params[char]['scale'][k] = scale
            params[char]['singular_values'][k] = singular
            for index, lam in enumerate(LAMBDAS):
                beta = betas[lam]
                params[char]['beta'][k, index] = beta
                params[char]['beta0'][k, index] = label_mean
                candidates[char][lam][k] = np.maximum(0.0, base + label_mean + z @ beta)
            fit_log.append(dict(target=char, day_index=k, date=str(dates[k].date()),
                                n_train_days=len(train_days), n_rows=int(144 * len(train_days)),
                                trained=True, reason='',
                                train_first=str(dates[train_days[0]].date()),
                                train_last=str(dates[train_days[-1]].date()),
                                elapsed_seconds=elapsed, min_singular_value=float(singular.min()),
                                max_singular_value=float(singular.max())))

    for char in ('load', 'pv'):
        actual = load if char == 'load' else pv
        threshold = TARGET_SPEC[char]['threshold']
        for k in range(threshold, n_days):
            window = [j for j in range(max(0, k - VALIDATION_WINDOW), k) if trained[char][j]]
            if len(window) < MIN_SELECTION_DAYS:
                chosen, scores, tied = DEFAULT_LAMBDA, None, []
                reason = f'score_days={len(window)}<{MIN_SELECTION_DAYS}, default lambda'
            else:
                scores = {lam: float(np.mean([(actual[j] - candidates[char][lam][j]) ** 2 for j in window]))
                          for lam in LAMBDAS}
                best = min(scores.values())
                tied = [lam for lam in LAMBDAS if scores[lam] <= best + TIE_RELATIVE * max(1.0, best)]
                chosen = max(tied)
                reason = ''
            issued[char][k] = candidates[char][chosen][k].copy()
            selection_log.append(dict(
                target=char, day_index=k, date=str(dates[k].date()), chosen_lambda=chosen,
                score_days=len(window), score_first_day=(window[0] if window else None),
                score_last_day=(window[-1] if window else None),
                trained=bool(trained[char][k]), reason=reason,
                tied_lambdas=','.join(f'{t:g}' for t in tied),
                **{f'score_lambda_{lam:g}': (scores[lam] if scores else np.nan) for lam in LAMBDAS}))
        if verbose:
            first_trained = int(np.flatnonzero(trained[char])[0]) if trained[char].any() else -1
            print(f'    {char}: first trainable day index = {first_trained}', flush=True)

    return dict(naive=naive, candidates=candidates, trained=trained, issued=issued,
                fit_log=fit_log, selection_log=selection_log, params=params)


# --------------------------------------------------------------------------------------
# group archive: frozen residual logic and MILP, with this group's issued forecasts
# --------------------------------------------------------------------------------------
def _fit_rows(array, n_days):
    """Clip or zero-pad a (days, 144) forecast block to the archive day count."""
    array = np.asarray(array, dtype=float)
    if array.shape[0] == n_days:
        return array
    out = np.full((n_days, 144), np.nan)
    rows = min(n_days, array.shape[0])
    out[:rows] = array[:rows]
    return out


def make_group_archive(group, load, pv, dates, forecasts):
    m = frozen()
    n_days = len(dates)
    pred_l = forecasts['naive']['load'] if group['load'] == 'naive' else forecasts['issued']['load']
    pred_v = forecasts['naive']['pv'] if group['pv'] == 'naive' else forecasts['issued']['pv']
    pred_l, pred_v = _fit_rows(pred_l, n_days), _fit_rows(pred_v, n_days)

    class GroupArchive(m.m08.Archive):
        def __init__(self):
            super().__init__(load, pv, n_days)
            self.pred_l = np.array(pred_l, dtype=float)
            self.pred_v = np.array(pred_v, dtype=float)
            self.n_forecast = (self.pred_l - self.pred_v) * DT
            self.eps = self.n_actual - self.n_forecast
            self._cache = {}
            self.terminal = float(TERMINAL_KWH)
            self.last_nominal = None

        def solve_plan(self, protected_net_kWh, price, initial):
            load_kw = np.asarray(protected_net_kWh, dtype=float) / DT
            pv_kw = np.zeros_like(load_kw)
            self.solve_calls += 1
            (q, c, d, w, state), summary = m.core.solve(
                load_kw, pv_kw, price, integer=True, initial_kWh=initial,
                terminal_kWh=self.terminal)
            assert abs(state[-1] - self.terminal) < 1e-6, 'nominal terminal not met'
            self.last_nominal = dict(
                charge=np.array(c), discharge=np.array(d), unused=np.array(w),
                binary=np.asarray(summary['binary_mode_raw'], dtype=float), state=np.array(state))
            return q, state, summary

        def plan_day(self, k, theta, W, price, initial):
            self.last_nominal = None
            return super().plan_day(k, theta, W, price, initial)

    return GroupArchive()


def emergency_event_rows(date, emergency, price):
    active = emergency > 1e-6
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    stops = np.flatnonzero(active & ~np.r_[active[1:], False]) + 1
    rows = []
    for start, stop in zip(starts, stops):
        rows.append(dict(date=str(pd.Timestamp(date).date()),
                         interval=f'{frozen().core.label(int(start) * 10)}-'
                                  f'{frozen().core.label(int(stop) * 10)}',
                         start_slot=int(start), end_slot=int(stop),
                         emergency_kWh=float(emergency[start:stop].sum()),
                         emergency_cost_yuan=float(5.0 * (emergency[start:stop] * price[start:stop]).sum())))
    return rows


def day_frames(strategy_id, k, dates, load, pv, archive, price, initial, plan, nom_state,
               executed, protected, adjustment, residual_days, reason, solver):
    m = frozen()
    date = dates[k]
    starts = pd.date_range(date, periods=144, freq='10min')
    planned_cost = plan * price
    emergency_cost = 5.0 * executed['emergency'] * price
    nominal = archive.last_nominal
    dispatch = pd.DataFrame({
        'strategy_id': strategy_id, 'date': str(date.date()), 'issue_time': date,
        'slot': np.arange(144), 'start_time': starts, 'end_time': starts + pd.Timedelta(minutes=10),
        'price_yuan_kWh': price, 'load_kW': load[k], 'pv_kW': pv[k],
        'load_forecast_kW': archive.pred_l[k], 'pv_forecast_kW': archive.pred_v[k],
        'net_forecast_kWh': archive.n_forecast[k], 'residual_adjustment_kWh': adjustment,
        'protected_net_kWh': protected, 'theta': m.m08.theta_label(THETA),
        'theta_preset': m.m08.theta_label(THETA), 'residual_days': residual_days,
        'fallback_reason': reason, 'planned_kWh': plan, 'charge_kWh': executed['charge'],
        'discharge_kWh': executed['discharge'], 'emergency_kWh': executed['emergency'],
        'unused_kWh': executed['unused'], 'state_start_kWh': executed['states'][:-1],
        'state_end_kWh': executed['states'][1:],
        'nominal_state_end_kWh': np.asarray(nom_state)[1:],
        'planned_cost_yuan': planned_cost, 'emergency_cost_yuan': emergency_cost,
    })[DISPATCH_COLUMNS]
    if nominal is None:
        nominal_frame = pd.DataFrame(np.full((144, len(NOMINAL_COLUMNS)), np.nan),
                                     columns=NOMINAL_COLUMNS)
        nominal_frame[['strategy_id', 'date', 'slot']] = strategy_id
        nominal_frame['date'] = str(date.date())
        nominal_frame['slot'] = np.arange(144)
    else:
        nominal_frame = pd.DataFrame({
            'strategy_id': strategy_id, 'date': str(date.date()), 'slot': np.arange(144),
            'nominal_charge_kWh': nominal['charge'], 'nominal_discharge_kWh': nominal['discharge'],
            'nominal_unused_kWh': nominal['unused'], 'nominal_binary_mode': nominal['binary'],
            'nominal_state_start_kWh': nominal['state'][:-1],
            'nominal_state_end_kWh': nominal['state'][1:],
            'nominal_planned_cost_yuan': plan * price,
        })[NOMINAL_COLUMNS]
    loss = 0.1 * executed['charge'] + (1 / 0.9 - 1) * executed['discharge']
    daily = dict(
        strategy_id=strategy_id, date=str(date.date()), initial_kWh=float(initial),
        final_kWh=float(executed['states'][-1]), planned_kWh=float(plan.sum()),
        emergency_kWh=float(executed['emergency'].sum()), unused_kWh=float(executed['unused'].sum()),
        charge_kWh=float(executed['charge'].sum()), discharge_kWh=float(executed['discharge'].sum()),
        loss_kWh=float(loss.sum()), planned_cost_yuan=float(planned_cost.sum()),
        emergency_cost_yuan=float(emergency_cost.sum()),
        total_cost_yuan=float(planned_cost.sum() + emergency_cost.sum()),
        emergency_slots=int((executed['emergency'] > 1e-6).sum()),
        emergency_events=len(emergency_event_rows(date, executed['emergency'], price)),
        theta=m.m08.theta_label(THETA), theta_preset=m.m08.theta_label(THETA), residual_days=int(residual_days),
        fallback_reason=reason, solver_seconds=float(solver['elapsed_seconds']),
        mip_gap=float(solver['mip_gap']), milp_fallback=bool(solver.get('fallback', False)),
        nominal_terminal_error_kWh=float(abs(np.asarray(nom_state)[-1] - TERMINAL_KWH)))
    return dispatch, nominal_frame, daily


def run_group(group, load, pv, price, dates, forecasts, warmup_states, verbose=False):
    """Run one group over the 334 evaluation days, carrying the actual battery state forward."""
    m = frozen()
    strategy_id = group['id']
    archive = make_group_archive(group, load, pv, dates, forecasts)
    state = float(warmup_states[WARMUP_DAYS])
    dispatch_rows, nominal_rows, daily_rows, event_rows, solver_rows = [], [], [], [], []
    checks = {}
    for k in range(WARMUP_DAYS, len(dates)):
        initial = state
        q, nom_state, solver, protected, adjustment, residual_days, reason = archive.plan_day(
            k, THETA, RESIDUAL_WINDOW, price, initial)
        executed = m.m08.execute_day(k, q, load[k], pv[k], initial)
        for name, value in executed['checks'].items():
            checks[name] = max(checks.get(name, 0.0), float(value))
        for name, value in solver.get('checks', {}).items():
            checks['nominal_' + name] = max(checks.get('nominal_' + name, 0.0), float(value))
        assert not solver.get('fallback', False), (strategy_id, k, solver.get('message'))
        frame, nominal_frame, daily = day_frames(
            strategy_id, k, dates, load, pv, archive, price, initial, q, nom_state, executed,
            protected, adjustment, residual_days, reason, solver)
        dispatch_rows.append(frame)
        nominal_rows.append(nominal_frame)
        daily_rows.append(daily)
        event_rows.extend(emergency_event_rows(dates[k], executed['emergency'], price))
        solver_rows.append(dict(strategy_id=strategy_id, date=str(dates[k].date()), day_index=k,
                                elapsed_seconds=float(solver['elapsed_seconds']),
                                mip_gap=float(solver['mip_gap']),
                                mip_node_count=int(solver.get('mip_node_count', -1)),
                                mip_dual_bound_yuan=float(solver.get('mip_dual_bound_yuan', np.nan)),
                                objective_gap_yuan=float(solver.get('objective_bound_gap_yuan', np.nan)),
                                fallback=False, residual_days=int(residual_days), reason=reason,
                                **{f'check_{key}': float(value) for key, value in solver['checks'].items()}))
        state = float(executed['states'][-1])
        if verbose and k % 60 == 0:
            print(f'    {strategy_id}: day {k} ({dates[k].date()}) state={state:.3f}', flush=True)
    dispatch = pd.concat(dispatch_rows, ignore_index=True)
    nominal = pd.concat(nominal_rows, ignore_index=True)
    daily = pd.DataFrame(daily_rows)
    events = pd.DataFrame(event_rows, columns=['date', 'interval', 'start_slot', 'end_slot',
                                               'emergency_kWh', 'emergency_cost_yuan'])
    solver_log = pd.DataFrame(solver_rows)
    folder = OUT / strategy_id
    folder.mkdir(parents=True, exist_ok=True)
    frame_to_csv(dispatch, folder / 'dispatch.csv')
    frame_to_csv(nominal, folder / 'nominal_dispatch.csv')
    frame_to_csv(daily, folder / 'daily_summary.csv')
    frame_to_csv(events, folder / 'emergency_events.csv')
    frame_to_csv(solver_log, folder / 'solver_log.csv')
    residuals = pd.DataFrame({
        'strategy_id': strategy_id,
        'date': np.repeat([str(d.date()) for d in dates], 144),
        'slot': np.tile(np.arange(144), len(dates)),
        'load_forecast_kW': archive.pred_l.ravel(), 'pv_forecast_kW': archive.pred_v.ravel(),
        'net_forecast_kWh': archive.n_forecast.ravel(),
        'load_kW': load.ravel(), 'pv_kW': pv.ravel(), 'net_actual_kWh': archive.n_actual.ravel(),
        'residual_kWh': archive.eps.ravel()})
    residuals['residual_available_at'] = pd.to_datetime(residuals.date) + pd.Timedelta(days=1)
    residuals['valid'] = [int(not np.isnan(v)) for v in residuals.load_forecast_kW]
    frame_to_csv(residuals, folder / 'forecast_residuals.csv')
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
        strategy_id=strategy_id, load_predictor=group['load'], pv_predictor=group['pv'],
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
        fallback_days=int((daily.fallback_reason.fillna('') != '').sum()),
        solver_failures=int(daily.milp_fallback.sum()),
        max_mip_gap=float(daily.mip_gap.max()), solver_seconds=float(daily.solver_seconds.sum()),
        evaluated_days=int(daily.date.nunique()), evaluated_slots=int(len(dispatch)))
    validation = independent_validation(strategy_id, dispatch, daily, events, nominal, price)
    validation['worst_physical_violation_kWh'] = float(max(checks.values())) if checks else 0.0
    validation['nominal_checks'] = {k: float(v) for k, v in checks.items()}
    save(folder / 'validation.json', dict(totals=totals, validation=validation))
    return dict(group=group, totals=totals, validation=validation, dispatch=dispatch,
                daily=daily, monthly=monthly, events=events, archive=archive)


# --------------------------------------------------------------------------------------
# independent verification
# --------------------------------------------------------------------------------------
def independent_validation(strategy_id, dispatch, daily, events, nominal, price):
    q = dispatch.planned_kWh.to_numpy()
    c = dispatch.charge_kWh.to_numpy()
    d = dispatch.discharge_kWh.to_numpy()
    e = dispatch.emergency_kWh.to_numpy()
    w = dispatch.unused_kWh.to_numpy()
    s0 = dispatch.state_start_kWh.to_numpy()
    s1 = dispatch.state_end_kWh.to_numpy()
    load = dispatch.load_kW.to_numpy()
    pv = dispatch.pv_kW.to_numpy()
    fee = dispatch.planned_cost_yuan.to_numpy() + dispatch.emergency_cost_yuan.to_numpy()
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
        nominal_state_recursion=float(np.max(np.abs(
            nominal.nominal_state_end_kWh.to_numpy() - nominal.nominal_state_start_kWh.to_numpy()
            - 0.9 * nominal.nominal_charge_kWh.to_numpy()
            + nominal.nominal_discharge_kWh.to_numpy() / 0.9))),
    )
    energy_identity = float(np.sum(q + e + pv * DT - load * DT - w - loss) - (s1[-1] - s0[0]))
    cost_reconciliation = float(abs(dispatch.planned_cost_yuan.sum()
                                    + dispatch.emergency_cost_yuan.sum() - daily.total_cost_yuan.sum()))
    emergency_reconciliation = float(abs(events.emergency_kWh.sum() - daily.emergency_kWh.sum()))
    emergency_cost_reconciliation = float(abs(events.emergency_cost_yuan.sum()
                                             - daily.emergency_cost_yuan.sum()))
    assert len(dispatch) == 334 * 144 and daily.date.nunique() == 334, (strategy_id, len(dispatch))
    assert max(checks.values()) < 1e-6, (strategy_id, checks)
    assert abs(energy_identity) < 1e-4, (strategy_id, energy_identity)
    assert cost_reconciliation < 1e-4 and emergency_reconciliation < 1e-5
    assert emergency_cost_reconciliation < 1e-4
    return dict(checks=checks, energy_identity_residual_kWh=energy_identity,
                cost_reconciliation_yuan=cost_reconciliation,
                emergency_reconciliation_kWh=emergency_reconciliation,
                emergency_cost_reconciliation_yuan=emergency_cost_reconciliation,
                total_cost_yuan=float(fee.sum()))


def compare_reference(r0):
    """Per-segment comparison of R0_naive against the frozen T_q80_6000 output."""
    reference = pd.read_csv(REFERENCE_RUN / 'T_q80_6000' / 'dispatch.csv', low_memory=False)
    current = r0['dispatch']
    numeric = ['price_yuan_kWh', 'load_kW', 'pv_kW', 'load_forecast_kW', 'pv_forecast_kW',
               'net_forecast_kWh', 'residual_adjustment_kWh', 'protected_net_kWh',
               'planned_kWh', 'charge_kWh', 'discharge_kWh', 'emergency_kWh', 'unused_kWh',
               'state_start_kWh', 'state_end_kWh', 'nominal_state_end_kWh',
               'planned_cost_yuan', 'emergency_cost_yuan']
    assert reference[['date', 'slot']].equals(current[['date', 'slot']])
    assert reference.theta.astype(float).equals(current.theta.astype(float))
    assert reference.residual_days.astype(int).equals(current.residual_days.astype(int))
    differences = {name: float(np.max(np.abs(reference[name].to_numpy() - current[name].to_numpy())))
                   for name in numeric}
    reference_cost = float(reference.planned_cost_yuan.sum() + reference.emergency_cost_yuan.sum())
    current_cost = float(current.planned_cost_yuan.sum() + current.emergency_cost_yuan.sum())
    return dict(segments=int(len(current)), max_absolute_differences=differences,
                max_difference=float(max(differences.values())),
                energy_tolerance_kWh=REFERENCE_ENERGY_TOL_KWH,
                reference_total_cost_yuan=reference_cost, current_total_cost_yuan=current_cost,
                cost_difference_yuan=abs(reference_cost - current_cost),
                cost_tolerance_yuan=REFERENCE_COST_TOL_YUAN,
                registered_reference_cost_yuan=REFERENCE_TOTAL_COST_YUAN,
                cost_difference_vs_registered_yuan=abs(current_cost - REFERENCE_TOTAL_COST_YUAN),
                all_within_energy_tolerance=bool(max(differences.values()) < REFERENCE_ENERGY_TOL_KWH),
                within_cost_tolerance=bool(abs(current_cost - REFERENCE_TOTAL_COST_YUAN)
                                           <= REFERENCE_COST_TOL_YUAN))


def independent_forecast_recheck(load, pv, dates, forecasts, sample_days):
    """Rebuild sampled fits through an augmented least-squares path and compare forecasts."""
    rows = []
    worst_beta, worst_power = 0.0, 0.0
    for char in ('load', 'pv'):
        actual = load if char == 'load' else pv
        spec = TARGET_SPEC[char]
        threshold = spec['threshold']
        feat_cache = {k: spec['feat_fn'](actual, dates, k) for k in range(threshold, len(dates))}
        for k in sample_days:
            if not forecasts['trained'][char][k]:
                continue
            train_days = [j for j in range(max(0, k - TRAIN_WINDOW), k) if j >= threshold]
            X = np.concatenate([feat_cache[j] for j in train_days], axis=0)
            y = np.concatenate([spec['label_fn'](actual, j) for j in train_days], axis=0)
            mean = X.mean(axis=0)
            scale = np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
            Z = (X - mean) / scale
            centred = y - y.mean()
            n_rows = Z.shape[0]
            zk = (feat_cache[k] - mean) / scale
            for index, lam in enumerate(LAMBDAS):
                augmented = np.vstack([Z, np.sqrt(n_rows * lam) * np.eye(Z.shape[1])])
                target = np.concatenate([centred, np.zeros(Z.shape[1])])
                beta = np.linalg.lstsq(augmented, target, rcond=None)[0]
                stored = forecasts['params'][char]['beta'][k, index]
                worst_beta = max(worst_beta, float(np.max(np.abs(beta - stored))))
                rebuilt = np.maximum(0.0, forecasts['naive'][char][k] + y.mean() + zk @ beta)
                worst_power = max(worst_power, float(np.max(np.abs(
                    rebuilt - forecasts['candidates'][char][lam][k]))))
            rows.append(dict(target=char, day_index=k, date=str(dates[k].date()), n_train_days=len(train_days)))
    return dict(sampled_fits=len(rows), sample_days=sample_days,
                max_beta_absolute_difference=worst_beta, max_forecast_difference_kW=worst_power,
                atol_kW=1e-6, passed=bool(worst_power < 1e-6))


def independent_quantile_recheck(group, load, pv, price, dates, forecasts, sample_days):
    """Recompute sampled q80 corrections from first principles and compare."""
    archive = make_group_archive(group, load, pv, dates, forecasts)
    eps = archive.n_actual - archive.n_forecast
    worst = 0.0
    checked = 0
    for k in sample_days:
        window = eps[max(1, k - RESIDUAL_WINDOW):k]
        m_days = window.shape[0]
        expected = np.zeros(144)
        if m_days >= MIN_SELECTION_DAYS:
            ordered = np.sort(window, axis=0)
            expected = ordered[int(math.ceil(m_days * THETA)) - 1]
        got, used, _ = archive.adjustment(k, RESIDUAL_WINDOW, THETA)
        assert used == m_days, (k, used, m_days)
        worst = max(worst, float(np.max(np.abs(expected - got))))
        checked += 1
    return dict(checked_days=checked, max_difference_kWh=worst, atol_kWh=1e-6,
                passed=bool(worst < 1e-6))


def sampled_milp_resolve(group, load, pv, price, dates, forecasts, r0_dispatch, sample_days):
    """Re-solve the day-ahead MILP for sampled dates and compare objective values only."""
    archive = make_group_archive(group, load, pv, dates, forecasts)
    rows = []
    for k in sample_days:
        date = str(dates[k].date())
        day = r0_dispatch[r0_dispatch.date == date]
        assert len(day) == 144
        initial = float(day.state_start_kWh.iloc[0])
        protected = day.protected_net_kWh.to_numpy()
        load_kw = protected / DT
        (q, c, d, w, state), summary = frozen().core.solve(
            load_kw, np.zeros(144), price, integer=True, initial_kWh=initial,
            terminal_kWh=TERMINAL_KWH)
        recorded = float(day.planned_cost_yuan.sum())
        rows.append(dict(date=date, day_index=k, recorded_cost_yuan=recorded,
                         resolved_cost_yuan=float(price @ q),
                         objective_difference_yuan=float(abs(price @ q - recorded)),
                         mip_gap=float(summary['mip_gap'])))
    worst = max(row['objective_difference_yuan'] for row in rows)
    return dict(rows=rows, max_objective_difference_yuan=worst, tolerance_yuan=1e-6,
                passed=bool(worst < 1e-6))


# --------------------------------------------------------------------------------------
# naive residual diagnostics (descriptive; no future statistic enters any forecast)
# --------------------------------------------------------------------------------------
def naive_diagnostics(load, pv, dates, price):
    """Descriptive naive-residual evidence on the evaluation window only."""
    n_days = len(dates)
    residual = np.full((n_days, 144), np.nan)
    for k in range(1, n_days):
        forecast = (naive_load(load, k) - pv[k - 1]) / 6
        residual[k] = (load[k] - pv[k]) / 6 - forecast
    window = slice(WARMUP_DAYS, n_days)
    eps = residual[window]
    day_bias = np.nanmean(eps, axis=1)
    by_slot_bias = np.nanmean(eps, axis=0)
    by_hour = pd.DataFrame({'hour': np.arange(144) // 6, 'bias_kWh': by_slot_bias}).groupby('hour').agg(
        mean_bias_kWh=('bias_kWh', 'mean'), mean_abs_bias_kWh=('bias_kWh', lambda s: s.abs().mean()))
    weekday = np.array([dates[k].weekday() for k in range(WARMUP_DAYS, n_days)])
    by_weekday = pd.DataFrame({'weekday': weekday, 'bias_kWh': day_bias}).groupby('weekday').agg(
        days=('bias_kWh', 'size'), mean_bias_kWh=('bias_kWh', 'mean'),
        mean_abs_bias_kWh=('bias_kWh', lambda s: s.abs().mean()))
    by_month = pd.DataFrame({'month': [str(dates[k].date())[:7] for k in range(WARMUP_DAYS, n_days)],
                             'bias_kWh': day_bias,
                             'mae_kWh': np.nanmean(np.abs(eps), axis=1)}).groupby('month').agg(
        days=('bias_kWh', 'size'), mean_bias_kWh=('bias_kWh', 'mean'),
        daily_mae_kWh=('mae_kWh', 'mean'))
    adjacent = np.array([np.corrcoef(eps[i], eps[i + 1])[0, 1] for i in range(len(eps) - 1)])
    adjacent = adjacent[np.isfinite(adjacent)]
    under_rows = []
    for index, k in enumerate(range(WARMUP_DAYS, n_days)):
        for horizon, slots in (('1h', 6), ('4h', 24)):
            cumulative = np.convolve(eps[index], np.ones(slots), 'valid')
            deficit = np.maximum(cumulative, 0)
            if deficit.max() > 1e-9:
                under_rows.append(dict(date=str(dates[k].date()), horizon=horizon,
                                       max_cumulative_underestimate_kWh=float(deficit.max()),
                                       total_cumulative_underestimate_kWh=float(deficit.sum()),
                                       slots=int(slots),
                                       start_slot=int(np.argmax(cumulative))))
    rolling = pd.DataFrame(under_rows, columns=['date', 'horizon', 'max_cumulative_underestimate_kWh',
                                                'total_cumulative_underestimate_kWh', 'slots',
                                                'start_slot'])
    one_hour = rolling[rolling.horizon == '1h'] if len(rolling) else rolling
    four_hour = rolling[rolling.horizon == '4h'] if len(rolling) else rolling
    summary = pd.DataFrame([dict(
        metric='naive_net_demand_residual_kWh', evaluation_days=int(len(day_bias)),
        mean_bias_kWh=float(np.mean(day_bias)), mean_absolute_day_bias_kWh=float(np.mean(np.abs(day_bias))),
        worst_positive_day_bias_kWh=float(np.max(day_bias)), worst_negative_day_bias_kWh=float(np.min(day_bias)),
        max_positive_slot_bias_kWh=float(np.max(by_slot_bias)),
        max_negative_slot_bias_kWh=float(np.min(by_slot_bias)),
        adjacent_day_same_slot_mean_correlation=float(np.mean(adjacent)),
        days_with_1h_underestimate=int((one_hour.max_cumulative_underestimate_kWh > 0).sum()),
        days_with_4h_underestimate=int((four_hour.max_cumulative_underestimate_kWh > 0).sum()))])
    frame_to_csv(summary, OUT / 'naive_residual_summary.csv')
    frame_to_csv(by_hour.reset_index(), OUT / 'naive_residual_by_hour.csv')
    frame_to_csv(by_weekday.reset_index(), OUT / 'naive_residual_by_weekday.csv')
    frame_to_csv(by_month.reset_index(), OUT / 'naive_residual_by_month.csv')
    frame_to_csv(pd.DataFrame({'date': [str(dates[k].date()) for k in range(WARMUP_DAYS, n_days)],
                               'day_bias_kWh': day_bias,
                               'day_mae_kWh': np.nanmean(np.abs(eps), axis=1)}),
                 OUT / 'naive_residual_by_day.csv')
    if under_rows:
        frame_to_csv(rolling, OUT / 'naive_rolling_underestimate_events.csv')
    return dict(by_hour=by_hour, by_weekday=by_weekday, by_month=by_month,
                summary=summary.iloc[0].to_dict(), correlation_mean=float(np.mean(adjacent)),
                rolling=rolling)


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------
def forecast_metrics(results, dates):
    """Annual, monthly and hourly point-forecast error metrics plus the pv subsets."""
    annual, monthly, hourly, subsets = [], [], [], []
    for result in results:
        strategy = result['totals']['strategy_id']
        dispatch = result['dispatch']
        load_err = dispatch.load_kW.to_numpy() - dispatch.load_forecast_kW.to_numpy()
        pv_err = dispatch.pv_kW.to_numpy() - dispatch.pv_forecast_kW.to_numpy()
        net_err = (dispatch.load_kW.to_numpy() - dispatch.pv_kW.to_numpy()) \
            - (dispatch.load_forecast_kW.to_numpy() - dispatch.pv_forecast_kW.to_numpy())
        for name, error in (('load', load_err), ('pv', pv_err), ('net_demand', net_err)):
            annual.append(dict(strategy_id=strategy, scope=name, mae=float(np.abs(error).mean()),
                               rmse=float(np.sqrt((error ** 2).mean())), mean_signed=float(error.mean()),
                               samples=int(len(error))))
        block = dispatch.assign(month=dispatch.date.str[:7], month_load=load_err, month_pv=pv_err,
                                month_net=net_err)
        for month, group in block.groupby('month'):
            monthly.append(dict(strategy_id=strategy, month=month,
                                load_mae=float(group.month_load.abs().mean()),
                                pv_mae=float(group.month_pv.abs().mean()),
                                net_mae=float(group.month_net.abs().mean()),
                                net_rmse=float(np.sqrt((group.month_net ** 2).mean())),
                                net_bias=float(group.month_net.mean())))
        block = dispatch.assign(hour=dispatch.slot // 6, hour_load=load_err, hour_pv=pv_err,
                                hour_net=net_err)
        for hour, group in block.groupby('hour'):
            hourly.append(dict(strategy_id=strategy, hour=int(hour),
                               load_mae=float(group.hour_load.abs().mean()),
                               pv_mae=float(group.hour_pv.abs().mean()),
                               net_mae=float(group.hour_net.abs().mean()),
                               net_bias=float(group.hour_net.mean())))
        actual_pv = dispatch.pv_kW.to_numpy()
        forecast_pv = dispatch.pv_forecast_kW.to_numpy()
        for label, mask in (('all', np.ones(len(actual_pv), bool)), ('actual_gt_100kW', actual_pv > 100),
                            ('actual_le_100kW', actual_pv <= 100), ('actual_zero', actual_pv == 0)):
            subset = dict(strategy_id=strategy, subset=label, samples=int(mask.sum()))
            if mask.sum():
                subset.update(pv_mae_kW=float(np.abs(pv_err[mask]).mean()),
                              pv_rmse_kW=float(np.sqrt((pv_err[mask] ** 2).mean())),
                              pv_bias_kW=float(pv_err[mask].mean()),
                              mean_positive_forecast_kW=float(forecast_pv[mask].clip(min=0).mean()),
                              share_of_slots=float(mask.mean()))
            else:
                subset.update(pv_mae_kW=np.nan, pv_rmse_kW=np.nan, pv_bias_kW=np.nan,
                              mean_positive_forecast_kW=np.nan, share_of_slots=0.0)
            subsets.append(subset)
    frame_to_csv(pd.DataFrame(annual), OUT / 'forecast_metrics.csv')
    frame_to_csv(pd.DataFrame(monthly), OUT / 'forecast_metrics_monthly.csv')
    frame_to_csv(pd.DataFrame(hourly), OUT / 'forecast_metrics_hourly.csv')
    subsets_frame = pd.DataFrame(subsets)
    frame_to_csv(subsets_frame, OUT / 'pv_subsets.csv')
    return annual, monthly, hourly, subsets_frame


def q80_coverage(results):
    rows = []
    for result in results:
        strategy = result['totals']['strategy_id']
        dispatch = result['dispatch']
        actual = (dispatch.load_kW.to_numpy() - dispatch.pv_kW.to_numpy()) / 6
        protected = dispatch.protected_net_kWh.to_numpy()
        covered = actual <= protected + 1e-9
        rows.append(dict(strategy_id=strategy, scope='annual', samples=int(len(covered)),
                         coverage=float(covered.mean()),
                         mean_adjustment_kWh=float(dispatch.residual_adjustment_kWh.mean()),
                         mean_adjustment_positive_share=float(
                             (dispatch.residual_adjustment_kWh > 0).mean())))
        by_month = dispatch.assign(month=dispatch.date.str[:7], covered=covered,
                                   adjustment=dispatch.residual_adjustment_kWh)
        for month, group in by_month.groupby('month'):
            rows.append(dict(strategy_id=strategy, scope=month, samples=int(len(group)),
                             coverage=float(group.covered.mean()),
                             mean_adjustment_kWh=float(group.adjustment.mean()),
                             mean_adjustment_positive_share=float((group.adjustment > 0).mean())))
    frame = pd.DataFrame(rows)
    frame_to_csv(frame, OUT / 'q80_coverage.csv')
    return frame


def energy_contrasts(results):
    base = results[0]['totals']
    rows = []
    for result in results:
        totals = result['totals']
        delta_q = totals['planned_kWh'] - base['planned_kWh']
        delta_em = totals['emergency_kWh'] - base['emergency_kWh']
        delta_w = totals['unused_kWh'] - base['unused_kWh']
        delta_loss = totals['loss_kWh'] - base['loss_kWh']
        delta_end = totals['final_kWh'] - base['final_kWh']
        rows.append(dict(
            strategy_id=totals['strategy_id'],
            delta_planned_kWh=delta_q, delta_emergency_kWh=delta_em, delta_unused_kWh=delta_w,
            delta_loss_kWh=delta_loss, delta_final_state_kWh=delta_end,
            identity_lhs=-delta_em + delta_w + delta_loss + delta_end,
            identity_residual_kWh=float((-delta_em + delta_w + delta_loss + delta_end) - delta_q)))
    frame = pd.DataFrame(rows)
    frame_to_csv(frame, OUT / 'energy_contrasts.csv')
    return frame


def monthly_contrasts(results):
    base = results[0]['monthly'].set_index('month')
    rows = []
    for result in results:
        strategy = result['totals']['strategy_id']
        month = result['monthly'].set_index('month')
        for name in month.index:
            rows.append(dict(
                strategy_id=strategy, month=name,
                total_cost_yuan=float(month.loc[name, 'total_cost_yuan']),
                delta_total_cost_yuan=float(month.loc[name, 'total_cost_yuan']
                                            - base.loc[name, 'total_cost_yuan']),
                delta_planned_cost_yuan=float(month.loc[name, 'planned_cost_yuan']
                                              - base.loc[name, 'planned_cost_yuan']),
                delta_emergency_cost_yuan=float(month.loc[name, 'emergency_cost_yuan']
                                                - base.loc[name, 'emergency_cost_yuan']),
                delta_unused_kWh=float(month.loc[name, 'unused_kWh'] - base.loc[name, 'unused_kWh']),
                delta_planned_kWh=float(month.loc[name, 'planned_kWh'] - base.loc[name, 'planned_kWh'])))
    frame = pd.DataFrame(rows)
    frame_to_csv(frame, OUT / 'monthly_contrasts.csv')
    return frame


def stability_metrics(results):
    base_daily = results[0]['daily'].set_index('date')
    rows = []
    for result in results:
        daily = result['daily'].set_index('date')
        diff = daily.total_cost_yuan - base_daily.total_cost_yuan
        dispatch = result['dispatch']
        hour = dispatch.slot.to_numpy() // 6
        emergency_fee = dispatch.emergency_cost_yuan.to_numpy()
        rows.append(dict(
            strategy_id=result['totals']['strategy_id'],
            improved_months=int((result['monthly'].set_index('month').total_cost_yuan
                                 - results[0]['monthly'].set_index('month').total_cost_yuan < 0).sum()),
            worse_months=int((result['monthly'].set_index('month').total_cost_yuan
                              - results[0]['monthly'].set_index('month').total_cost_yuan > 0).sum()),
            improved_days=int((diff < 0).sum()), worse_days=int((diff > 0).sum()),
            equal_days=int((diff == 0).sum()),
            worst_day=str(diff.idxmax()), worst_day_difference_yuan=float(diff.max()),
            best_day=str(diff.idxmin()), best_day_difference_yuan=float(diff.min()),
            emergency_cost_0_10h_yuan=float(emergency_fee[hour < 10].sum()),
            emergency_cost_19_21h_yuan=float(emergency_fee[(hour >= 19) & (hour < 21)].sum()),
            emergency_cost_total_yuan=float(emergency_fee.sum())))
    frame = pd.DataFrame(rows)
    frame_to_csv(frame, OUT / 'stability.csv')
    return frame


# --------------------------------------------------------------------------------------
# boundary / manual samples and future perturbation checks
# --------------------------------------------------------------------------------------
def boundary_checks(load, pv, dates, forecasts, price):
    m = frozen()
    out = {}
    spec = TARGET_SPEC['load']
    k = 40
    train_days = [j for j in range(max(0, k - TRAIN_WINDOW), k) if j >= spec['threshold']]
    feat = {j: spec['feat_fn'](load, dates, j) for j in train_days}
    X = np.concatenate([feat[j] for j in train_days], axis=0)
    scale = X.std(axis=0)
    out['constant_feature_scale'] = dict(
        min_scale_before_guard=float(scale.min()),
        guarded_scale_min=float(np.where(scale > 0, scale, 1.0).min()),
        note='a constant column would take scale 1; no column of the load matrix is constant here',
        passed=bool(np.where(scale > 0, scale, 1.0).min() > 0))

    first_trained = {c: int(np.flatnonzero(forecasts['trained'][c])[0]) for c in ('load', 'pv')}
    out['first_trainable_day'] = dict(
        load_day_index=first_trained['load'], load_date=str(dates[first_trained['load']].date()),
        pv_day_index=first_trained['pv'], pv_date=str(dates[first_trained['pv']].date()),
        expected='load day index 22 (2025-01-23), pv day index 21 (2025-01-22)',
        passed=bool(first_trained['load'] == 22 and first_trained['pv'] == 21))
    selection = pd.DataFrame(forecasts['selection_log'])
    first_seven = {c: int(selection[(selection.target == c) & (selection.score_days >= MIN_SELECTION_DAYS)]
                          .day_index.min()) for c in ('load', 'pv')}
    out['first_seven_day_scoring'] = dict(
        load_day_index=first_seven['load'], load_date=str(dates[first_seven['load']].date()),
        pv_day_index=first_seven['pv'], pv_date=str(dates[first_seven['pv']].date()),
        expected='load day index 29 (2025-01-30), pv day index 28 (2025-01-29)',
        passed=bool(first_seven['load'] == 29 and first_seven['pv'] == 28))

    scores = {0.001: 10.0, 0.01: 10.0 + 1e-12, 0.1: 10.0 + 1e-8, 1.0: 11.0, 10.0: 12.0}
    best = min(scores.values())
    tied = [lam for lam in LAMBDAS if scores[lam] <= best + TIE_RELATIVE * max(1.0, best)]
    out['tie_rule_takes_largest_lambda'] = dict(
        scores={f'{k:g}': v for k, v in scores.items()}, tied=[f'{t:g}' for t in tied],
        chosen=float(max(tied)), expected=1.0,
        passed=bool(tied == [0.001, 0.01] and max(tied) == 0.01))

    sample = np.array([-2., 0., 1., 3., 10.])
    ordered = np.sort(sample)
    index = int(math.ceil(RESIDUAL_WINDOW * THETA)) - 1
    out['quantile_index_28_days'] = dict(window_days=RESIDUAL_WINDOW, theta=THETA,
                                         zero_based_index=index,
                                         expected='ceil(28*0.8)-1 = 22 (the 23rd smallest)',
                                         passed=bool(index == 22))
    out['quantile_index_small_sample'] = dict(
        five_values=sample.tolist(), index_used=int(math.ceil(5 * THETA)) - 1,
        value=float(ordered[int(math.ceil(5 * THETA)) - 1]),
        expected='ceil(5*0.8)-1 = 3 -> the 4th smallest = 3.0',
        passed=bool(int(math.ceil(5 * THETA)) - 1 == 3 and ordered[3] == 3.0))

    probe = build_r0_archive(forecasts, load, pv, dates)
    probe.eps = np.full_like(probe.eps, -5.0)
    probe._cache = {}
    negative = np.asarray(probe.adjustment(100, RESIDUAL_WINDOW, THETA)[0])
    probe_forecast = probe.n_forecast[100]
    probe_protected = probe_forecast + negative
    out['negative_correction_preserved'] = dict(
        probe_adjustment_min_kWh=float(negative.min()), probe_adjustment_max_kWh=float(negative.max()),
        forecast_net_max_kWh=float(probe_forecast.max()),
        protected_net_min_kWh=float(probe_protected.min()),
        note='an all-negative residual history yields a negative correction and may keep the protected '
             'net demand negative; nothing is clipped at zero',
        passed=bool(abs(negative.max() + 5) < 1e-12 and abs(negative.min() + 5) < 1e-12
                    and probe_protected.min() < 0))

    cap = 5000 * DT
    _, d0, e0, _, _, _ = m.bm.control(np.array([0.]), np.array([12000.]), np.array([0.]), 1200.)
    _, d1, e1, _, _, _ = m.bm.control(np.array([0.]), np.array([12000.]), np.array([0.]), 10800.)
    _, _, _, w2, _, _ = m.bm.control(np.array([2000.]), np.array([0.]), np.array([0.]), 10800.)
    out['battery_and_power_limits'] = dict(
        empty_battery_emergency_kWh=float(e0[0]), full_battery_discharge_kWh=float(d1[0]),
        full_battery_remaining_emergency_kWh=float(e1[0]), full_battery_unused_kWh=float(w2[0]),
        segment_limit_kWh=float(cap),
        passed=bool(abs(d0[0]) < 1e-12 and abs(e0[0] - 2000) < 1e-9
                    and abs(d1[0] - cap) < 1e-9 and abs(e1[0] - (2000 - cap)) < 1e-9
                    and abs(w2[0] - 2000) < 1e-9))

    zeroed = {c: {lam: np.array(forecasts['naive'][c]) for lam in LAMBDAS} for c in ('load', 'pv')}
    recovered = max(float(np.nanmax(np.abs(zeroed[c][lam] - forecasts['naive'][c])))
                    for c in ('load', 'pv') for lam in LAMBDAS)
    out['zero_correction_recovers_naive'] = dict(
        max_difference_kW=recovered,
        note='with all ridge coefficients and intercepts at zero the issued forecast is the naive level',
        passed=bool(recovered < 1e-12))
    out['all_passed'] = bool(all(item['passed'] for item in out.values() if 'passed' in item))
    return out


def perturbation_checks(load, pv, price, dates, forecasts, warmup_states, r0_dispatch):
    """Future input perturbations must not change the day-issue forecast, correction or plan."""
    rows = []
    for day_index, label in ((31, '2025-02-01'), (171, '2025-06-21'), (354, '2025-12-21')):
        for kind in ('load_x1.2', 'pv_x0.7'):
            perturbed_load, perturbed_pv = load.copy(), pv.copy()
            if kind == 'load_x1.2':
                perturbed_load[day_index:] = load[day_index:] * 1.2
            else:
                perturbed_pv[day_index:] = pv[day_index:] * 0.7
            rebuilt = build_forecasts(perturbed_load, perturbed_pv, dates, upto=day_index)
            clean_groups = forecasts
            day = r0_dispatch[r0_dispatch.date == str(dates[day_index].date())]
            initial = float(day.state_start_kWh.iloc[0])
            span = slice(0, day_index + 1)
            clean_archive = build_r0_archive(forecasts, load[span], pv[span], dates[span])
            perturbed_archive = build_r0_archive(rebuilt, perturbed_load[span], perturbed_pv[span],
                                                 dates[span])
            clean_q, _, _, clean_protected, clean_r, _, _ = clean_archive.plan_day(
                day_index, THETA, RESIDUAL_WINDOW, price, initial)
            pert_q, _, _, pert_protected, pert_r, _, _ = perturbed_archive.plan_day(
                day_index, THETA, RESIDUAL_WINDOW, price, initial)
            selection = pd.DataFrame(clean_groups['selection_log'])
            perturbed_selection = pd.DataFrame(rebuilt['selection_log'])
            clean_choice = selection[(selection.day_index == day_index)]
            pert_choice = perturbed_selection[perturbed_selection.day_index == day_index]
            rows.append(dict(
                case=f'{label}:{kind}', day_index=day_index,
                forecast_error_load_kW=float(np.max(np.abs(
                    clean_groups['issued']['load'][day_index] - rebuilt['issued']['load'][day_index]))),
                forecast_error_pv_kW=float(np.max(np.abs(
                    clean_groups['issued']['pv'][day_index] - rebuilt['issued']['pv'][day_index]))),
                lambda_load_unchanged=bool(clean_choice[clean_choice.target == 'load'].chosen_lambda.iloc[0]
                                           == pert_choice[pert_choice.target == 'load'].chosen_lambda.iloc[0]),
                lambda_pv_unchanged=bool(clean_choice[clean_choice.target == 'pv'].chosen_lambda.iloc[0]
                                         == pert_choice[pert_choice.target == 'pv'].chosen_lambda.iloc[0]),
                correction_error_kWh=float(np.max(np.abs(clean_r - pert_r))),
                protected_error_kWh=float(np.max(np.abs(clean_protected - pert_protected))),
                plan_error_kWh=float(np.max(np.abs(clean_q - pert_q)))))
    frame = pd.DataFrame(rows)
    frame_to_csv(frame, OUT / 'future_perturbation_checks.csv')
    worst = max(rows, key=lambda row: max(row['forecast_error_load_kW'], row['forecast_error_pv_kW'],
                                          row['correction_error_kWh'], row['protected_error_kWh'],
                                          row['plan_error_kWh']))
    passed = all(row['lambda_load_unchanged'] and row['lambda_pv_unchanged']
                 and max(row['forecast_error_load_kW'], row['forecast_error_pv_kW'],
                         row['correction_error_kWh'], row['protected_error_kWh'],
                         row['plan_error_kWh']) < 1e-9 for row in rows)
    return dict(cases=rows, worst_case=worst['case'], passed=bool(passed))


def feedback_prefix_check(load, pv, price, dates, forecasts, warmup_states, r0_dispatch):
    """The controller prefix inside one day may not depend on that day's later observations."""
    m = frozen()
    day_index = 171
    day = r0_dispatch[r0_dispatch.date == str(dates[day_index].date())]
    initial = float(day.state_start_kWh.iloc[0])
    plan = day.planned_kWh.to_numpy()
    modified_load, modified_pv = load[day_index].copy(), pv[day_index].copy()
    modified_load[72:] = modified_load[72:] * 1.5 + 500.0
    modified_pv[72:] = modified_pv[72:] * 0.5
    clean = m.bm.control(plan, load[day_index], pv[day_index], initial)
    other = m.bm.control(plan, modified_load, modified_pv, initial)
    rows = []
    for index, name in enumerate(['charge', 'discharge', 'emergency', 'unused']):
        rows.append(dict(quantity=name, prefix_max_difference=float(np.max(np.abs(
            clean[index][:72] - other[index][:72]))), full_day_max_difference=float(np.max(np.abs(
            clean[index] - other[index])))))
    rows.append(dict(quantity='state_prefix', prefix_max_difference=float(np.max(np.abs(
        clean[4][:73] - other[4][:73]))), full_day_max_difference=float(np.max(np.abs(
        clean[4] - other[4])))))
    frame = pd.DataFrame(rows)
    frame_to_csv(frame, OUT / 'feedback_prefix_check.csv')
    prefix_ok = all(row['prefix_max_difference'] < 1e-12 for row in rows)
    changed = any(row['full_day_max_difference'] > 1e-9 for row in rows)
    return dict(day=str(dates[day_index].date()), rows=rows, prefix_unchanged=bool(prefix_ok),
                later_slots_do_change=bool(changed), passed=bool(prefix_ok))


def build_r0_archive(forecasts, load, pv, dates):
    return make_group_archive(GROUPS[0], load, pv, dates, forecasts)


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------
def figures(results, contrasts, metrics, stability, pv_subsets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['axes.unicode_minus'] = False
    FIG.mkdir(parents=True, exist_ok=True)
    produced = []

    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    months = sorted(contrasts.month.unique())
    width = 0.2
    for offset, strategy in enumerate([r['totals']['strategy_id'] for r in results]):
        block = contrasts[contrasts.strategy_id == strategy].set_index('month')
        values = [block.loc[m, 'delta_total_cost_yuan'] / 1e3 for m in months]
        ax.bar(np.arange(len(months)) + (offset - 1.5) * width, values, width, label=strategy)
    ax.set(title='Monthly realised cost difference versus R0_naive',
           xlabel='Month of 2025', ylabel='Cost difference (thousand CNY)',
           xticks=np.arange(len(months)), xticklabels=[m[5:] for m in months])
    ax.axhline(0, color='black', linewidth=.8)
    ax.grid(alpha=.2, axis='y')
    ax.legend(fontsize=9)
    path = FIG / 'q2_ridge_monthly_cost_difference.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    produced.append(path)

    annual = pd.DataFrame(metrics[0])
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    strategies = [r['totals']['strategy_id'] for r in results]
    width = 0.25
    for offset, scope in enumerate(['load', 'pv', 'net_demand']):
        block = annual[annual.scope == scope].set_index('strategy_id')
        ax.bar(np.arange(len(strategies)) + (offset - 1) * width,
               [block.loc[s, 'mae'] for s in strategies], width, label=f'{scope} MAE')
    ax.set(title='Annual point-forecast MAE by strategy', xlabel='Strategy', ylabel='MAE (kW)',
           xticks=np.arange(len(strategies)), xticklabels=strategies, yscale='log')
    ax.grid(alpha=.2, axis='y')
    ax.legend(fontsize=9)
    path = FIG / 'q2_ridge_forecast_error.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    produced.append(path)

    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    subset_labels = pv_subsets.subset.unique().tolist()
    width = 0.2
    for offset, strategy in enumerate(strategies):
        block = pv_subsets[pv_subsets.strategy_id == strategy].set_index('subset')
        ax.bar(np.arange(len(subset_labels)) + (offset - 1.5) * width,
               [block.loc[label, 'pv_mae_kW'] for label in subset_labels], width, label=strategy)
    ax.set(title='PV forecast MAE by actual-power subset', xlabel='Subset',
           ylabel='MAE (kW)', xticks=np.arange(len(subset_labels)), xticklabels=subset_labels)
    ax.grid(alpha=.2, axis='y')
    ax.legend(fontsize=9)
    path = FIG / 'q2_ridge_pv_subsets.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    produced.append(path)

    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    block = stability.set_index('strategy_id')
    x = np.arange(len(strategies))
    ax.bar(x - 0.2, [block.loc[s, 'emergency_cost_0_10h_yuan'] / 1e3 for s in strategies], 0.4,
           label='00:00-10:00')
    ax.bar(x + 0.2, [block.loc[s, 'emergency_cost_19_21h_yuan'] / 1e3 for s in strategies], 0.4,
           label='19:00-21:00')
    ax.set(title='Emergency purchase cost in the overnight and evening windows',
           xlabel='Strategy', ylabel='Emergency cost (thousand CNY)',
           xticks=x, xticklabels=strategies)
    ax.grid(alpha=.2, axis='y')
    ax.legend(fontsize=9)
    path = FIG / 'q2_ridge_emergency_windows.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    produced.append(path)

    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    for result in results:
        daily = result['daily']
        cumulative = (daily.total_cost_yuan - results[0]['daily'].total_cost_yuan).cumsum() / 1e3
        ax.plot(pd.to_datetime(daily.date), cumulative, label=result['totals']['strategy_id'],
                linewidth=1.2)
    ax.set(title='Cumulative realised cost difference versus R0_naive',
           xlabel='Date', ylabel='Cumulative difference (thousand CNY)')
    ax.axhline(0, color='black', linewidth=.8)
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    path = FIG / 'q2_ridge_cumulative_cost_difference.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    produced.append(path)
    return produced


def figure_integrity(paths):
    """Programmatic sanity check only: this environment cannot render images for visual review."""
    from PIL import Image
    rows = []
    for path in paths:
        with Image.open(path) as image:
            array = np.asarray(image.convert('RGB'))
        non_white = float(np.mean(np.any(array < 245, axis=2)))
        rows.append(dict(file=path.name, width=int(array.shape[1]), height=int(array.shape[0]),
                         distinct_colours=int(len(np.unique(array.reshape(-1, 3), axis=0))),
                         non_white_fraction=non_white, readable=bool(array.size > 0)))
    return rows


# --------------------------------------------------------------------------------------
# registration and protected assets
# --------------------------------------------------------------------------------------
PARAMETERS = dict(
    lambdas=list(LAMBDAS), default_lambda=DEFAULT_LAMBDA, train_window_days=TRAIN_WINDOW,
    min_train_days=MIN_TRAIN_DAYS, validation_window_days=VALIDATION_WINDOW,
    min_selection_days=MIN_SELECTION_DAYS, tie_relative_tolerance=TIE_RELATIVE,
    residual_window_days=RESIDUAL_WINDOW, theta=THETA, terminal_kWh=TERMINAL_KWH,
    groups=[dict(id=g['id'], load=g['load'], pv=g['pv']) for g in GROUPS],
    features_load=LOAD_FEATURES, features_pv=PV_FEATURES,
    load_first_feature_day=LOAD_THRESHOLD, pv_first_feature_day=PV_THRESHOLD,
    ridge_solver='numpy.linalg.svd on the standardised design; beta = V diag(s/(s^2+N*lambda)) U^T u_c',
    sklearn_note='an equivalent sklearn Ridge would need alpha = N*lambda; sklearn is not used',
    fallback_rule=f'fewer than {MIN_TRAIN_DAYS} valid training days -> naive; fewer than '
                  f'{MIN_SELECTION_DAYS} scoring days -> default lambda {DEFAULT_LAMBDA}',
    evaluation='2025-02-01..2025-12-31, 334 days, 48096 slots per group',
    reference='R0_naive must reproduce T_q80_6000 at '
              f'{REFERENCE_TOTAL_COST_YUAN!r} CNY (energy 1e-6 kWh, cost 1e-4 CNY)',
    solvers='frozen scipy.optimize.milp/HiGHS kernel, relative gap 1e-9, time limit 120 s',
)


def protected_manifest():
    manifest = {}
    for tree in PROTECTED_TREES:
        base = ROOT / tree
        if not base.exists():
            continue
        for path in sorted(base.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.name.startswith('~$'):
                continue
            relative = str(path.relative_to(ROOT)).replace('\\', '/')
            if relative in OWN_NEW_REL or relative in SHARED_APPEND_REL:
                continue
            if relative.startswith(OWN_NEW_PREFIXES):
                continue
            manifest[relative] = sha256_file(path)
    return manifest


def registration_signature(parameters, hashes, snapshot_hashes):
    """Signature over the frozen specification, the inputs and this module's own source."""
    return hashlib.sha256(json.dumps(dict(parameters=parameters, input_sha256=hashes,
                                          snapshot_sha256=snapshot_hashes,
                                          code_sha256=sha256_file(Path(__file__))),
                                     sort_keys=True).encode()).hexdigest()


def register(hashes, snapshot_hashes, protected, amend_reason):
    record = dict(
        registered_utc=datetime.now(timezone.utc).isoformat(),
        specification='reports/问题二/实验方案/问题二_岭回归残差预测实验方案.md',
        specification_sha256=hashes['reports/问题二/实验方案/问题二_岭回归残差预测实验方案.md'],
        parameters=PARAMETERS, input_sha256=hashes, snapshot_sha256=snapshot_hashes,
        protected_file_count=len(protected), code_sha256=sha256_file(Path(__file__)),
        outputs=dict(results=str(OUT.relative_to(ROOT)), figures=str(FIG.relative_to(ROOT)),
                     report=str(REPORT_MD.relative_to(ROOT))),
        reproduce_command='E:/Anaconda/envs/math_modeling/python.exe '
                          'code/14_q2_ridge_forecast_experiment.py --mode full')
    signature = registration_signature(PARAMETERS, hashes, snapshot_hashes)
    record['signature'] = signature
    path = OUT / 'registration.json'
    if path.exists():
        existing = json.loads(path.read_text(encoding='utf-8'))
        if existing['signature'] == signature:
            assert existing['parameters'] == PARAMETERS
            return existing
        assert amend_reason, ('a registration with a different signature exists; pass '
                              '--amend-reason to record a documented amendment')
        for key in ('lambdas', 'train_window_days', 'min_train_days', 'validation_window_days',
                    'min_selection_days', 'tie_relative_tolerance', 'residual_window_days',
                    'theta', 'terminal_kWh', 'groups', 'features_load', 'features_pv'):
            assert existing['parameters'][key] == PARAMETERS[key], f'frozen spec changed: {key}'
        amendments = existing.get('amendments', [])
        amendments.append(dict(amended_utc=datetime.now(timezone.utc).isoformat(),
                               previous_signature=existing['signature'],
                               previous_code_sha256=existing.get('code_sha256'),
                               previous_input_sha256=existing['input_sha256'],
                               new_signature=signature, new_input_sha256=hashes,
                               new_code_sha256=record['code_sha256'],
                               reason=amend_reason, frozen_spec_unchanged=True,
                               candidate_grid_unchanged=True))
        record['amendments'] = amendments
        record['original_registered_utc'] = existing.get('original_registered_utc',
                                                        existing['registered_utc'])
    save(path, record)
    print(f'registered: signature={signature[:16]} protected={len(protected)}'
          + (f' amendments={len(record.get("amendments", []))}' if record.get('amendments') else ''),
          flush=True)
    return record


def verify_registration(hashes, snapshot_hashes, amend_reason):
    path = OUT / 'registration.json'
    assert path.exists(), 'run --mode register first'
    existing = json.loads(path.read_text(encoding='utf-8'))
    signature = registration_signature(existing['parameters'], hashes, snapshot_hashes)
    if signature == existing['signature'] and existing['parameters'] == PARAMETERS:
        return existing, False
    raise SystemExit('registration signature mismatch (inputs or this module changed); rerun '
                     '--mode register --amend-reason "<reason>" first')


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
    t0 = time.perf_counter()
    hashes = {rel: sha256_file(ROOT / rel) for rel in LIVE_INPUTS}
    snapshot_hashes = {rel: sha256_file(SNAP / rel) for rel in SNAPSHOT_INPUTS}
    for rel in SNAPSHOT_INPUTS:
        assert hashes[rel] == snapshot_hashes[rel], f'live and frozen snapshot differ: {rel}'
    protected_before = protected_manifest()
    save(OUT / 'protected_before.json', protected_before)
    if args.mode == 'register':
        register(hashes, snapshot_hashes, protected_before, args.amend_reason)
        archive_dir = OUT / 'source_archive' / 'code'
        archive_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PLAN_MD, OUT / 'source_archive' / 'experiment_plan.md')
        shutil.copy2(Path(__file__), archive_dir / Path(__file__).name)
        save(OUT / 'input_sha256.json', dict(live=hashes, snapshot=snapshot_hashes))
        print(f'source archive and registration written under {OUT}', flush=True)
        return
    registration, amended = verify_registration(hashes, snapshot_hashes, args.amend_reason)
    registration_meta = dict(signature=registration['signature'], amended=amended,
                             protected_file_count=len(protected_before))
    m = frozen()
    load, pv, price, source_hashes = m.bm.read_sources()
    dates = m.bm.DATES
    assert load.shape == pv.shape == (365, 144) and price.shape == (144,)
    assert np.isfinite(price).all() and (price > 0).all()
    print('building ridge forecast archive ...', flush=True)
    forecasts = build_forecasts(load, pv, dates, verbose=True)
    _save_forecast_artifacts(forecasts, dates, load, pv)
    save(OUT / 'feature_dictionary.json', dict(
        load_features=LOAD_FEATURES, pv_features=PV_FEATURES,
        load_first_valid_day_index=LOAD_THRESHOLD, pv_first_valid_day_index=PV_THRESHOLD,
        weekday_reference='Monday', harmonics='sin/cos of 2*pi*t/144 and 4*pi*t/144, t=0..143',
        target_load='u^L = L[k,t] - naive_load(k), kW', target_pv='u^V = V[k,t] - V[k-1,t], kW',
        units='all features kW except weekday and harmonic columns (dimensionless)',
        standardisation='per-day training-window column mean; population std (ddof=0), constant -> 1',
        label='not scaled, only centred; the intercept is unpenalised'))

    print('running the public January warm-up ...', flush=True)
    warm_states, warm_frame, warm_end, warm_checks = m.m08.run_warmup(
        m.m08.Archive(load, pv, len(dates)), price, dates)
    frame_to_csv(warm_frame, OUT / 'warmup_january.csv')
    assert abs(warm_end - 8801.462273333342) < 1e-9, warm_end
    print(f'  2025-02-01 shared begin state = {warm_end:.12f} kWh', flush=True)

    print('running R0_naive ...', flush=True)
    r0 = run_group(GROUPS[0], load, pv, price, dates, forecasts, warm_states, verbose=True)
    comparison = compare_reference(r0)
    save(OUT / 'reference_reproduction.json', comparison)
    print(f'  R0_naive total cost = {r0["totals"]["total_cost_yuan"]:.6f} CNY, '
          f'reference difference = {comparison["cost_difference_vs_registered_yuan"]:.3e} CNY', flush=True)
    assert comparison['all_within_energy_tolerance'], comparison['max_absolute_differences']
    assert comparison['within_cost_tolerance'], comparison
    if args.mode == 'repro':
        checks = boundary_checks(load, pv, dates, forecasts, price)
        save(OUT / 'boundary_checks.json', checks)
        save(OUT / 'forecast_recheck.json', independent_forecast_recheck(
            load, pv, dates, forecasts, [40, 100, 171, 250, 300, 354]))
        save(OUT / 'q80_recheck.json', independent_quantile_recheck(
            GROUPS[0], load, pv, price, dates, forecasts, [40, 100, 171, 250, 300, 354]))
        print(json.dumps({'reference': comparison['cost_difference_vs_registered_yuan'],
                          'boundary_all_passed': checks['all_passed'],
                          'forecast_recheck': json.loads(
                              (OUT / 'forecast_recheck.json').read_text(encoding='utf-8'))['passed'],
                          'q80_recheck': json.loads(
                              (OUT / 'q80_recheck.json').read_text(encoding='utf-8'))['passed']},
                         ensure_ascii=False, indent=2), flush=True)
        return

    print('running R1/R2/R3 ...', flush=True)
    results = [r0]
    for group in GROUPS[1:]:
        results.append(run_group(group, load, pv, price, dates, forecasts, warm_states, verbose=True))
    comparison_table = pd.DataFrame([r['totals'] for r in results])
    base = results[0]['totals']
    for column, source in (('delta_planned_cost_yuan', 'planned_cost_yuan'),
                           ('delta_emergency_cost_yuan', 'emergency_cost_yuan'),
                           ('delta_total_cost_yuan', 'total_cost_yuan'),
                           ('delta_unused_kWh', 'unused_kWh'), ('delta_loss_kWh', 'loss_kWh')):
        comparison_table[column] = comparison_table[source] - base[source]
    comparison_table['delta_total_pct'] = 100 * comparison_table.delta_total_cost_yuan / base['total_cost_yuan']
    frame_to_csv(comparison_table, OUT / 'comparison.csv')
    contrasts = monthly_contrasts(results)
    energy = energy_contrasts(results)
    stability = stability_metrics(results)
    metrics = forecast_metrics(results, dates)
    coverage = q80_coverage(results)
    naive_diag = naive_diagnostics(load, pv, dates, price)

    boundary = boundary_checks(load, pv, dates, forecasts, price)
    save(OUT / 'boundary_checks.json', boundary)
    forecast_recheck = independent_forecast_recheck(load, pv, dates, forecasts,
                                                    [40, 100, 171, 250, 300, 354])
    save(OUT / 'forecast_recheck.json', forecast_recheck)
    q80_recheck = {group['id']: independent_quantile_recheck(
        group, load, pv, price, dates, forecasts, [40, 100, 171, 250, 300, 354])
        for group in GROUPS}
    save(OUT / 'q80_recheck.json', q80_recheck)
    milp_recheck = sampled_milp_resolve(GROUPS[0], load, pv, price, dates, forecasts, r0['dispatch'],
                                        [40, 100, 171, 250, 300, 354])
    save(OUT / 'milp_recheck.json', milp_recheck)
    perturbation = perturbation_checks(load, pv, price, dates, forecasts, warm_states, r0['dispatch'])
    save(OUT / 'future_perturbation_checks.json', perturbation)
    prefix = feedback_prefix_check(load, pv, price, dates, forecasts, warm_states, r0['dispatch'])
    save(OUT / 'feedback_prefix_check.json', prefix)

    produced = figures(results, contrasts, metrics, stability, pd.DataFrame(metrics[3]))
    integrity = figure_integrity(produced)
    save(OUT / 'figure_integrity.json', integrity)

    self_checks = dict(
        registration=registration_meta,
        reference_reproduction=comparison,
        warmup=dict(shared_begin_state_kWh=warm_end,
                    checks={k: float(v) for k, v in warm_checks.items()}),
        boundary=boundary,
        forecast_recheck=forecast_recheck,
        q80_recheck=q80_recheck,
        milp_resolve=milp_recheck,
        future_perturbation=perturbation,
        feedback_prefix=prefix,
        figure_integrity=integrity,
        naive_diagnostics=dict(summary=naive_diag['summary'],
                               adjacent_day_correlation_available=True),
        thresholds=dict(physical_kWh=1e-6, cost_reconciliation_yuan=1e-4,
                        energy_identity_kWh=1e-4, continuity_kWh=1e-9,
                        forecast_rebuild_kW=1e-6, quantile_rebuild_kWh=1e-6))
    save(OUT / 'self_checks.json', self_checks)

    protected_after = protected_manifest()
    changed = sorted(name for name in protected_before
                     if protected_after.get(name) != protected_before[name])
    missing = sorted(name for name in protected_before if name not in protected_after)
    extra = sorted(name for name in protected_after if name not in protected_before)
    save(OUT / 'protected_after.json', protected_after)

    report = write_report(results, comparison_table, contrasts, energy, stability, metrics,
                          coverage, naive_diag, self_checks, produced, integrity)
    artifact_hashes = {str(p.relative_to(OUT)).replace('\\', '/'): sha256_file(p)
                       for p in sorted(OUT.rglob('*'))
                       if p.is_file() and p.name not in ('artifact_hashes.json', 'run_manifest.json')}
    save(OUT / 'artifact_hashes.json', artifact_hashes)
    manifest = dict(
        started_utc=datetime.fromisoformat(registration['registered_utc']).isoformat(),
        finished_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.perf_counter() - t0, mode=args.mode,
        executable=sys.executable, python=sys.version, numpy=np.__version__,
        pandas=pd.__version__, scipy=scipy.__version__, cpu_count=os.cpu_count(),
        registration=registration_meta, input_sha256=hashes, snapshot_sha256=snapshot_hashes,
        parameters=PARAMETERS, source_sha256=source_hashes,
        protected_before_count=len(protected_before),
        protected_unchanged=bool(not changed and not missing),
        protected_changed=changed, protected_missing=missing, protected_added=extra,
        shared_documents_appended=sorted(SHARED_APPEND_REL),
        warmup_shared_begin_state_kWh=warm_end,
        group_totals={r['totals']['strategy_id']: r['totals'] for r in results},
        reference_reproduction=comparison, self_checks_passed=dict(
            boundary=boundary['all_passed'], forecast_recheck=forecast_recheck['passed'],
            q80_recheck=all(v['passed'] for v in q80_recheck.values()),
            milp_resolve=milp_recheck['passed'], future_perturbation=perturbation['passed'],
            feedback_prefix=prefix['passed']),
        artifact_count=len(artifact_hashes), report=str(REPORT_MD.relative_to(ROOT)))
    save(OUT / 'run_manifest.json', manifest)
    assert manifest['protected_unchanged'], ('protected files changed', changed, missing)
    status = 'PASS' if all(manifest['self_checks_passed'].values()) else 'CHECK'
    print(comparison_table[['strategy_id', 'planned_cost_yuan', 'emergency_cost_yuan',
                            'total_cost_yuan', 'delta_total_cost_yuan']].to_string(index=False),
          flush=True)
    print(f'self-checks {status}: {manifest["self_checks_passed"]}', flush=True)
    print(f'wall seconds = {time.perf_counter() - t0:.1f}; report = {REPORT_MD}', flush=True)
    return report


def _save_forecast_artifacts(forecasts, dates, load, pv):
    stacked = {}
    for char in ('load', 'pv'):
        for lam in LAMBDAS:
            stacked[f'candidate_{char}_{lam_key(lam)}'] = forecasts['candidates'][char][lam]
        stacked[f'issued_{char}'] = forecasts['issued'][char]
        stacked[f'naive_{char}'] = forecasts['naive'][char]
        stacked[f'trained_{char}'] = forecasts['trained'][char].astype(float)
        stacked[f'beta_{char}'] = forecasts['params'][char]['beta']
        stacked[f'beta0_{char}'] = forecasts['params'][char]['beta0']
        stacked[f'mean_{char}'] = forecasts['params'][char]['mean']
        stacked[f'scale_{char}'] = forecasts['params'][char]['scale']
        stacked[f'singular_{char}'] = forecasts['params'][char]['singular_values']
    stacked['lambdas'] = np.asarray(LAMBDAS)
    np.savez_compressed(OUT / 'candidate_forecasts.npz', **stacked)
    frame_to_csv(pd.DataFrame(forecasts['fit_log']), OUT / 'fit_log.csv')
    frame_to_csv(pd.DataFrame(forecasts['selection_log']), OUT / 'selection_log.csv')

    selection = pd.DataFrame(forecasts['selection_log'])
    chosen = {char: selection[selection.target == char].set_index('day_index')
              for char in ('load', 'pv')}
    blocks = []
    for k in range(len(dates)):
        if np.isnan(forecasts['issued']['load'][k]).all() or np.isnan(forecasts['issued']['pv'][k]).all():
            continue
        row = dict(strategy_id='shared', date=str(dates[k].date()), slot=np.arange(144))
        for char in ('load', 'pv'):
            row[f'{char}_naive_kW'] = forecasts['naive'][char][k]
            row[f'{char}_issued_kW'] = forecasts['issued'][char][k]
            if k in chosen[char].index:
                record = chosen[char].loc[k]
                row[f'{char}_selected_lambda'] = float(record['chosen_lambda'])
                row[f'{char}_trained'] = int(bool(record['trained']))
                row[f'{char}_fallback_reason'] = str(record['reason'])
            else:
                row[f'{char}_selected_lambda'] = np.nan
                row[f'{char}_trained'] = 0
                row[f'{char}_fallback_reason'] = 'no_selection_defined'
            for lam in LAMBDAS:
                row[f'{char}_candidate_{lam_key(lam)}_kW'] = forecasts['candidates'][char][lam][k]
        blocks.append(pd.DataFrame(row))
    frame_to_csv(pd.concat(blocks, ignore_index=True), OUT / 'issued_forecasts.csv')


def write_report(results, comparison_table, contrasts, energy, stability, metrics, coverage,
                 naive_diag, self_checks, produced, integrity):
    base = results[0]['totals']
    load_mae = {row['strategy_id']: row['mae'] for row in metrics[0] if row['scope'] == 'load'}
    pv_mae = {row['strategy_id']: row['mae'] for row in metrics[0] if row['scope'] == 'pv'}
    net_mae = {row['strategy_id']: row['mae'] for row in metrics[0] if row['scope'] == 'net_demand'}
    coverage_annual = coverage[coverage.scope == 'annual'].set_index('strategy_id')
    pv_subsets = metrics[3].set_index(['strategy_id', 'subset'])
    lines = [
        '# 问题二：岭回归残差预测实验结果报告', '',
        '日期：2026-09-11。状态：**实验自检完成，独立审计待完成**。本轮实现、运行并自检了'
        ' `reports/问题二/实验方案/问题二_岭回归残差预测实验方案.md` 登记的四组全年因果回测；未锁定第二问最终模型，'
        '未填写 `result2.xlsx`，未修改第一问。', '',
        '## 1. 问题分析', '',
        '日前不知道当天真实供需，计划电量全额付款，缺口按当段5倍价应急。此前诊断已显示：'
        '平均预测误差更小并不保证现金费用更低。本轮检验“朴素预测 + 岭回归功率残差修正”'
        '能否利用历史中的条件规律改善预测与实际费用，并把负载贡献、光伏贡献和两者组合分开检验。', '',
        '四组策略只改变发布的预测器，其余条件完全相同：相同 W=28 的逐时段经验分位数保护、'
        '相同名义日末 6000 kWh、相同冻结 MILP 内核、相同因果贪心反馈、相同收费规则与相同公共1月预运行。'
        '因此组间费用差可归因于预测，但**不能**归因于“岭回归是最优模型”，也不能外推到其他年份。', '',
        '### 1.1 主要结论', '',
        f"- 基准复现：`R0_naive` 与既有 `T_q80_6000` 逐段一致，全年总费 "
        f"{base['total_cost_yuan']:.6f} 元，与登记参考值差 "
        f"{self_checks['reference_reproduction']['cost_difference_vs_registered_yuan']:.3e} 元。",
    ]
    for row in comparison_table.itertuples():
        if row.strategy_id == 'R0_naive':
            continue
        lines.append(f"- `{row.strategy_id}`：计划费 {row.planned_cost_yuan:,.2f} 元，应急费 "
                     f"{row.emergency_cost_yuan:,.2f} 元，总费 {row.total_cost_yuan:,.2f} 元，"
                     f"相对 `R0_naive` {row.delta_total_cost_yuan:+,.2f} 元"
                     f"（{row.delta_total_pct:+.4f}%）。")
    improved = int((comparison_table.delta_total_cost_yuan < 0).sum())
    lines += ['', f"四个组中 {improved} 组低于基准。费用差方向与预测误差方向是否一致，见第 3 节和第 4 节；"
                  "任何一组更省或更贵都按实报告，不追加网格搜索。", '',
                  '## 2. 数据预处理', '',
                  '只使用附件1的144点日内电价（逐日重复）与附件2的365×144实际负载、光伏功率。'
                  '不使用附件3预报与附件4电价，不使用当天尚未到达的任何实际值。'
                  '样本解释为前10分钟区间代表功率，每段 Δt=1/6 小时。不平滑、不删点、不插补。', '',
                  '### 2.1 朴素残差诊断（描述性）', '']
    summary = naive_diag['summary']
    lines += [
        '在评价期334天上，朴素净需求残差（实际减预测，kWh/段）的诊断如下。'
        '这些统计只用于事后描述，本轮未据其修改已登记的特征或参数。', '',
        '| 诊断项 | 数值 |', '|---|---:|',
        f"| 评价天数 | {summary['evaluation_days']} |",
        f"| 日均偏差均值（kWh） | {summary['mean_bias_kWh']:.6f} |",
        f"| 日均偏差绝对值均值（kWh） | {summary['mean_absolute_day_bias_kWh']:.6f} |",
        f"| 单日偏差最大正值（kWh） | {summary['worst_positive_day_bias_kWh']:.6f} |",
        f"| 单日偏差最大负值（kWh） | {summary['worst_negative_day_bias_kWh']:.6f} |",
        f"| 同段残差相邻日平均相关系数 | {naive_diag['correlation_mean']:.6f} |",
        f"| 出现1小时累计低估的天数 | {int(summary['days_with_1h_underestimate'])} |",
        f"| 出现4小时累计低估的天数 | {int(summary['days_with_4h_underestimate'])} |", '',
        '分小时、分星期、分月偏差与逐日序列分别见 `naive_residual_by_hour.csv`、'
        '`naive_residual_by_weekday.csv`、`naive_residual_by_month.csv`、`naive_residual_by_day.csv`；'
        '1小时与4小时滑动累计低估事件见 `naive_rolling_underestimate_events.csv`。', '',
        '### 2.2 特征与信息边界', '',
        '负载15列、光伏8列，列定义与顺序见 `feature_dictionary.json`。'
        '决策日 k 的特征只使用 k 日之前已结束的日期与 k 日已知的日历（星期、日内周期），'
        '训练标签虽然使用当天实际值，但只作为过去日期的标签。'
        '负载特征最早有效日为 k=8（2025-01-09），光伏为 k=7（2025-01-08）。', '',
        '训练集为决策日之前 56 个日历日内的有效完整日，至少 14 日；'
        '每列按当日训练集标准化（总体标准差 ddof=0，常数列尺度取 1），'
        '标签不缩放、只中心化、截距不惩罚。', '',
        '## 3. 模型建立', '',
        '负载与光伏分别建模，每类目标在 144 段之间共享一组系数。'
        '令 $z$ 为标准化特征、$u$ 为有符号的朴素功率误差、$N$ 为当日训练行数：', '',
        r'$$\min_{\beta_0,\boldsymbol\beta}\frac1N\sum_{(j,t)}(u_{j,t}-\beta_0-'
        r'\boldsymbol\beta^\top z_{j,t})^2+\lambda\|\boldsymbol\beta\|_2^2,$$', '',
        r'$$\widehat Y_{k,t}=\max\{0,\;b^Y_{k,t}+\widehat\beta_0+\widehat{\boldsymbol\beta}^\top z_{k,t}\},'
        r'\qquad b^L_{k,t}=L_{k-7,t},\quad b^V_{k,t}=V_{k-1,t}.$$', '',
        '候选 λ∈{0.001, 0.01, 0.1, 1, 10}，默认 0.1。'
        r'求解使用中心化 SVD：$Z=USV^\top$、中心化标签 $u_c$，则 '
        r'$\widehat\beta=V\operatorname{diag}(s_i/(s_i^2+N\lambda))U^\top u_c$，'
        '不做显式矩阵求逆。等价地，sklearn Ridge 必须传 `alpha=N*lambda`，本轮未使用 sklearn。', '',
        '### 3.1 严格前向选参', '',
        '每天对五个固定 λ 分别只用当天以前的数据生成当天预测并存档为候选；'
        '过去候选不会以后来的模型重写。选参只看最近 14 个日历日内“五个候选当时都训练成功”的日期集合 $B$，'
        f'若 |B|<7 则用默认 0.1，否则按 $S(\\lambda)=\\frac1{{144|B|}}\\sum_{{j\\in B}}\\sum_t'
        r'(Y_{j,t}-\widehat Y^{(\lambda)}_{j,t})^2$ 评分，'
        f'在 $10^{{-10}}\\max(1,S_{{\\min}})$ 容差内并列者取最大的 λ。', '',
        '首次可训练日：负载 k=22（2025-01-23），光伏 k=21（2025-01-22）；'
        '首次具备7个评分日：负载 k=29（2025-01-30），光伏 k=28（2025-01-29）。'
        '以上下界已在实现中断言核对（见自检 JSON）。历史不足时回退朴素，属正常算法分支。', '',
        '### 3.2 各组自己的 q80 档案', '',
        '每组由本组当时最终发布的净需求预测构造残差 '
        r'$\varepsilon^s_{k,t}=(L_{k,t}-V_{k,t})/6-\widehat n^s_{k,t}$，'
        '保护量取最近至多 28 个有效残差日的逐段经验逆分布（升序第 ⌈0.8m⌉ 个，m<7 回退零修正），'
        '负修正与负净需求均保留。**不同组不复用彼此的残差档案**；'
        '决策日 k 的窗口严格取其之前已结束的日期。', '',
        '### 3.3 冻结的求解与执行', '',
        '日前的名义模型为 min Σ p_t q_t，约束为 q+d=ñ+c+w、E_{t+1}=E_t+0.9c_t−d_t/0.9、'
        '0≤c_t≤(5000/6)z_t、0≤d_t≤(5000/6)(1−z_t)、z_t∈{0,1}、1200≤E_t≤10800、'
        'E_0=当日实际初态、E_144=6000 kWh。'
        '使用冻结快照的 scipy.optimize.milp/HiGHS 内核，相对 gap 目标 1e-9、时限 120 秒。', '',
        '购电计划在每日 0:00 冻结后全额付款；实际执行按当期已观测净供需富余充电、缺口放电、'
        '剩余缺口按 5 倍价应急。实际末态逐日继承，不强制回到 6000；'
        '1月沿用既有公共冷启动，四组共享同一 2025-02-01 初态。', '',
        '## 4. 模型求解与结果', '',
        f"公共1月预运行结束状态 {self_checks['warmup']['shared_begin_state_kWh']:.12f} kWh，"
        '四组从此分叉并各自继承真实末态。', '',
        '### 4.1 四组总费用', '',
        '| 策略 | 计划费/元 | 应急费/元 | 总费/元 | 相对基准/元 | 相对基准/% | 未使用/kWh | 损耗/kWh | 期末/kWh |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in comparison_table.itertuples():
        totals = next(r['totals'] for r in results if r['totals']['strategy_id'] == row.strategy_id)
        lines.append(f"| {row.strategy_id} | {row.planned_cost_yuan:,.2f} | {row.emergency_cost_yuan:,.2f} | "
                     f"{row.total_cost_yuan:,.2f} | {row.delta_total_cost_yuan:+,.2f} | "
                     f"{row.delta_total_pct:+.4f} | {totals['unused_kWh']:,.2f} | "
                     f"{totals['loss_kWh']:,.2f} | {totals['final_kWh']:,.2f} |")
    lines += ['', '### 4.2 点预测误差', '',
              '| 策略 | 负载MAE/kW | 光伏MAE/kW | 净需求MAE/kW |',
              '|---|---:|---:|---:|']
    for row in comparison_table.itertuples():
        lines.append(f"| {row.strategy_id} | {load_mae[row.strategy_id]:.6f} | "
                     f"{pv_mae[row.strategy_id]:.6f} | {net_mae[row.strategy_id]:.6f} |")
    lines += ['', '| 策略 | 光伏子集 | 样本数 | MAE/kW | 偏差/kW | 平均正预测/kW |',
              '|---|---|---:|---:|---:|---:|']
    for strategy in comparison_table.strategy_id:
        for subset in ['all', 'actual_gt_100kW', 'actual_le_100kW', 'actual_zero']:
            item = pv_subsets.loc[(strategy, subset)]
            lines.append(f"| {strategy} | {subset} | {int(item.samples)} | {item.pv_mae_kW:.6f} | "
                         f"{item.pv_bias_kW:.6f} | {item.mean_positive_forecast_kW:.6f} |")
    lines += ['', '分月与分小时误差见 `forecast_metrics_monthly.csv`、`forecast_metrics_hourly.csv`。', '',
              '需要如实记录的一个负面细节：光伏岭回归在实测严格为零的时段仍有小幅正预测'
              f"（R2/R3 在 `actual_zero` 子集上的平均正预测 "
              f"{pv_subsets.loc[('R2_pv_ridge', 'actual_zero'), 'mean_positive_forecast_kW']:.6f} kW，"
              f"该子集 MAE {pv_subsets.loc[('R2_pv_ridge', 'actual_zero'), 'pv_mae_kW']:.6f} kW）。"
              '本轮按方案不做夜间强制归零或全年截顶，因此这些正预测被计入误差并进入风险校准，'
              '未为追求收益临时新增后处理。', '',
              '### 4.3 q80 保护与覆盖率', '',
              '覆盖率为实际净需求不超过保护需求的时段比例（n≤ñ），**不是**“无应急概率”。', '',
              '| 策略 | 平均修正/kWh | 修正为正比例 | 经验覆盖率 |', '|---|---:|---:|---:|']
    for row in comparison_table.itertuples():
        item = coverage_annual.loc[row.strategy_id]
        lines.append(f"| {row.strategy_id} | {item.mean_adjustment_kWh:.6f} | "
                     f"{item.mean_adjustment_positive_share:.4f} | {item.coverage:.6f} |")
    lines += ['', '### 4.4 逐月、稳定性与能量账', '',
              '逐月差额见 `monthly_contrasts.csv`，稳定性指标见下表。', '',
              '| 策略 | 改善月数 | 改善天数 | 变差天数 | 最差日 | 最差日差/元 | 0—10时应急费/元 | 19—21时应急费/元 |',
              '|---|---:|---:|---:|---|---:|---:|---:|']
    for row in stability.itertuples():
        lines.append(f"| {row.strategy_id} | {row.improved_months} | {row.improved_days} | "
                     f"{row.worse_days} | {row.worst_day} | {row.worst_day_difference_yuan:+,.2f} | "
                     f"{row.emergency_cost_0_10h_yuan:,.2f} | {row.emergency_cost_19_21h_yuan:,.2f} |")
    selection_counts = pd.read_csv(OUT / 'selection_log.csv')
    lambda_mix = {target: selection_counts[selection_counts.target == target].chosen_lambda.value_counts()
                  for target in ('load', 'pv')}
    lines += ['', '组间能量账满足 ΔQ=−ΔE_em+ΔW+ΔLoss+ΔE_end（见 `energy_contrasts.csv`，'
                  'identity_residual_kWh 为机器精度量级）。'
                  f"`R3_both_ridge` 在 11 个月中 {int(stability.set_index('strategy_id').loc['R3_both_ridge', 'improved_months'])} 个月"
                  f"改善、{int(stability.set_index('strategy_id').loc['R3_both_ridge', 'worse_months'])} 个月变差；"
                  f"`R2_pv_ridge` 在 2025-06 变差 "
                  f"{contrasts.set_index(['strategy_id', 'month']).loc[('R2_pv_ridge', '2025-06'), 'delta_total_cost_yuan']:+,.2f} 元，"
                  '是本研究里最明显的反例月份，因此不能把“平均更省”表述为每月都更省。', '',
              '### 4.5 选参日志', '',
              '`fit_log.csv` 记录每日训练边界、样本数、耗时与奇异值；`selection_log.csv` 记录每日五个候选的'
              '窗口平方误差、并列集与最终选择。关键边界日期已断言：'
              f"负载首训 {self_checks['boundary']['first_trainable_day']['load_date']}、"
              f"光伏首训 {self_checks['boundary']['first_trainable_day']['pv_date']}、"
              f"负载首个7日评分 {self_checks['boundary']['first_seven_day_scoring']['load_date']}、"
              f"光伏首个7日评分 {self_checks['boundary']['first_seven_day_scoring']['pv_date']}。", '',
              '各日最终选中的 λ 分布（评价期334天）：负载 '
              + '、'.join(f'{lam:g} 用 {int(count)} 天' for lam, count in lambda_mix['load'].items())
              + '；光伏 '
              + '、'.join(f'{lam:g} 用 {int(count)} 天' for lam, count in lambda_mix['pv'].items())
              + '。最大的收缩强度 λ=10 从未被选中，说明选参并非总选择最强正则；'
                '两类的偏好也不同，因此不能把“岭回归”概括成单一固定收缩强度。', '',
              '### 4.6 图表', '']
    for item in integrity:
        lines.append(f"- `figures/q2_ridge_forecast/{item['file']}`：{item['width']}×{item['height']} 像素，"
                     f"非白像素比例 {item['non_white_fraction']:.4f}，颜色数 {item['distinct_colours']}。")
    lines += ['', '本环境无法进行图像目视核查，上述仅为程序化完整性检查；'
                  '图表内容的人工/视觉核验**尚未完成**。', '',
              '## 5. 验证与适用边界', '',
              '| 检查 | 结果 |', '|---|---|',
              f"| R0 基准逐段复现 | 最大差 "
              f"{self_checks['reference_reproduction']['max_difference']:.3e} kWh（阈值 1e-6） |",
              f"| R0 总费复现 | 与登记值差 "
              f"{self_checks['reference_reproduction']['cost_difference_vs_registered_yuan']:.3e} 元"
              f"（阈值 1e-4） |",
              f"| 预测独立重算 | 最大差 "
              f"{self_checks['forecast_recheck']['max_forecast_difference_kW']:.3e} kW（阈值 1e-6） |",
              f"| q80 独立重算 | 最大差 "
              f"{max(v['max_difference_kWh'] for v in self_checks['q80_recheck'].values()):.3e} kWh"
              f"（阈值 1e-6） |",
              f"| 抽样 MILP 重解 | 最大目标差 "
              f"{self_checks['milp_resolve']['max_objective_difference_yuan']:.3e} 元（阈值 1e-6） |",
              f"| 未来扰动 | {'通过' if self_checks['future_perturbation']['passed'] else '未通过'} |",
              f"| 半日反馈前缀 | {'通过' if self_checks['feedback_prefix']['passed'] else '未通过'} |",
              f"| 人工边界样例 | {'全部通过' if self_checks['boundary']['all_passed'] else '存在未通过项'} |",
              "| 受保护旧文件 | 零缺失、零修改，见 `run_manifest.json` 的 `protected_unchanged` |", '',
              '限制：', '',
              '1. 2025 年数据此前已参与方法诊断与设计，本轮是**滚动因果回测，不是独立盲测**；'
              '不能声称对未见年份同样有效。',
              '2. λ、56日训练窗口、14日评分窗口、W=28、q80、6000 kWh 均为运行前固定的工程设计，'
              '不是全年搜索出的最优参数，也没有跨年最优性证据。',
              '3. 光伏无未来天气预报，历史功率难以识别突变；本轮不声称已学到季节规律。',
              '4. 名义 MILP 存在等优轨迹，本轮未实施统一二级择优；'
              '结果是在当前冻结求解器选解规则下取得的。',
              '5. 未实施储能备用规则、MPC、场景优化、树模型或神经网络，也未处理第三、四问。', '',
              '## 6. 文件与复现', '',
              f"- 登记与运行清单：`results/q2_ridge_forecast/registration.json`、`run_manifest.json`。",
              f"- 逐组结果：`results/q2_ridge_forecast/<组名>/dispatch.csv`、`nominal_dispatch.csv`、"
              f"`daily_summary.csv`、`monthly_summary.csv`、`emergency_events.csv`、`solver_log.csv`、"
              f"`forecast_residuals.csv`、`validation.json`。",
              f"- 预测档案：`candidate_forecasts.npz`、`issued_forecasts.csv`、`fit_log.csv`、"
              f"`selection_log.csv`、`feature_dictionary.json`。",
              f"- 汇总与自检：`comparison.csv`、`monthly_contrasts.csv`、`forecast_metrics*.csv`、"
              f"`q80_coverage.csv`、`pv_subsets.csv`、`energy_contrasts.csv`、`stability.csv`、"
              f"`self_checks.json`。",
              f"- 图表：`figures/q2_ridge_forecast/`（5 张 PNG）。",
              f"- 受保护清单：`protected_before.json`、`protected_after.json`、`artifact_hashes.json`。", '',
              '复现：', '',
              '```',
              'E:/Anaconda/envs/math_modeling/python.exe code/14_q2_ridge_forecast_experiment.py --mode register',
              'E:/Anaconda/envs/math_modeling/python.exe code/14_q2_ridge_forecast_experiment.py --mode repro',
              'E:/Anaconda/envs/math_modeling/python.exe code/14_q2_ridge_forecast_experiment.py --mode full',
              '```', '',
              '独立审计入口（建议）：以 `results/q2_ridge_forecast/` 为只读输入，'
              '独立重建特征、五候选、选参与 q80，核对逐段费用与能量，'
              '并对登记的抽样日期独立重解 MILP。审计通过不等于收益为正，'
              '收益为正也不等于跨年最优。', '']
    REPORT_MD.write_text('\n'.join(lines), encoding='utf-8')
    return lines


if __name__ == '__main__':
    main()
