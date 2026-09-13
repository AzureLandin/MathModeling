"""Q2 first-round improvement experiment: historical residual-quantile demand protection.

Run (math_modeling env):
    E:/Anaconda/envs/math_modeling/python.exe code/08_q2_quantile_experiment.py --mode repro
    E:/Anaconda/envs/math_modeling/python.exe code/08_q2_quantile_experiment.py --mode smoke
    E:/Anaconda/envs/math_modeling/python.exe code/08_q2_quantile_experiment.py --mode full --jobs 8

Modes
    repro : public January warm-up + A_base only, checked against the locked reference values
            recorded in the task book table 2.
    smoke : warm-up + an adaptive-flow smoke test over the first evaluation days with strict
            assertions on shared window start, residual cut-off and state continuity.
    full  : the six main experiments (A_base, four fixed quantiles, B_adaptive) and the two
            residual-window sensitivities (W=14, W=56).

Fixed conditions (task book table 1), all policies share them:
    price            attachment 1, 144 points, repeated every day
    load / PV        attachment 2 actual tables only
    forbidden        attachment 3 forecasts, attachment 4 prices, any not-yet-realised data
    time             sample labelled 00:10 represents 00:00-00:10; 144 slots cover 0:00-24:00
    dt               1/6 h
    efficiency       eta_c = eta_d = 0.9 (one-way 90%, round trip 81%)
    state bounds     1200 - 10800 kWh (battery internal)
    power limit      5000 kW bus side -> M = 5000/6 kWh per slot
    normal purchase  the 144-slot plan is fixed at 00:00 and frozen; unused energy is still paid
    emergency price  5x the normal price of the corresponding slot
    actual control   surplus charges, deficit discharges, residual deficit goes to emergency
    nominal terminal nominal end-of-day state = the day's actual beginning state
    actual terminal  no forced daily cycle; the actual end state carries to the next day
    evaluation       2025-02-01 .. 2025-12-31, 334 days, 48,096 slots per policy
    solver           original MILP kernel, relative gap target 1e-9, time limit 120 s

The point predictor, the battery controller, the efficiency convention, the time convention and
the nominal terminal condition are all inherited unchanged from code/05_q2_baseline.py, so that a
difference in realised cost can be attributed to the protection amount alone.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'q2_quantile_experiment'
FIG = ROOT / 'figures' / 'q2_quantile_experiment'
CODE_08 = Path(__file__).resolve()


def _load_module(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = _load_module('q1_core', 'code/02_q1_baseline.py')
bm = _load_module('q2_baseline_mod', 'code/05_q2_baseline.py')

DT = 1 / 6
EVALUATION_START = '2025-02-01'
EVALUATION_END = '2025-12-31'
CANDIDATE_ORDER = ('base', 0.50, 0.65, 0.80, 0.90)   # canonical tie-break order
TIE_TOLERANCE_YUAN = 1e-6
MIN_RESIDUAL_DAYS = 7
VALID_FIRST_DAY = 1          # day 0 has no point forecast, its residual is unusable

CONFIG = dict(
    dt_hours=DT, eta_c=0.9, eta_d=0.9,
    state_min_kWh=1200.0, state_max_kWh=10800.0,
    initial_kWh=6000.0, power_max_kW=5000.0, emergency_multiplier=5.0,
    evaluation_start=EVALUATION_START, evaluation_end=EVALUATION_END,
    evaluation_days=334, evaluation_slots=334 * 144,
    predictor=dict(load='L[k-7,t] seasonal naive, fallback L[k-1,t] when history < 7 days',
                   pv='V[k-1,t] daily-profile persistence',
                   unit='kW', issue_time='00:00 of the target day'),
    residual_unit='kWh', residual_definition='eps = (L-V)/6 - (Lhat-Vhat)/6',
    residual_valid_days='all days with a generated point forecast except day 0, i.e. j >= 1',
    residual_window_rule='the most recent W valid residual dates strictly before the decision day',
    quantile_definition='empirical inverse distribution: sort the m window residuals ascending, '
                        'take the value at 0-based index ceil(m*alpha)-1',
    min_residual_days=MIN_RESIDUAL_DAYS,
    fallback_rule='fewer than 7 valid residual dates -> no correction (theta=base), reason recorded',
    residual_window_default=28, validation_window_default=14,
    quantile_candidates=[0.50, 0.65, 0.80, 0.90],
    base_candidate_label='base',
    tie_rule='candidates within 1e-6 CNY of the window minimum are tied; keep the parameter used '
             'the previous day if it is among them, otherwise take the canonical order '
             'base, q50, q65, q80, q90',
    tie_tolerance_yuan=TIE_TOLERANCE_YUAN,
    observation_assumption='right-endpoint sample = representative power of the preceding 10-minute '
                           'interval; ideal in-interval fast feedback; the interval-end sample is not '
                           'treated as known at the interval start',
    no_sale_revenue=True, no_curtailment_penalty=True, no_battery_depreciation=True,
    solver=dict(kernel='code/02_q1_baseline.py solve() unchanged',
                milp_relative_gap=core.CONFIG['milp_relative_gap'],
                milp_time_limit_seconds=core.CONFIG['milp_time_limit_seconds'],
                nominal_terminal='E_bar[144] = E[0] = the day actual beginning state'),
    fallback_plan='if the MILP returns no usable solution: q = max(protected net demand, 0) with the '
                  'nominal battery held idle; the event is recorded, never reported as optimal',
    seed=20260910, tolerance=1e-6,
)

POLICIES = [
    dict(id='A_base', kind='fixed', theta='base', W=None, J=None,
         role='matched-condition main baseline, no correction'),
    dict(id='B_fixed_q50', kind='fixed', theta=0.50, W=28, J=None,
         role='median bias correction'),
    dict(id='B_fixed_q65', kind='fixed', theta=0.65, W=28, J=None, role='moderate protection'),
    dict(id='B_fixed_q80', kind='fixed', theta=0.80, W=28, J=None, role='higher protection'),
    dict(id='B_fixed_q90', kind='fixed', theta=0.90, W=28, J=None, role='strong protection'),
    dict(id='B_adaptive', kind='adaptive', theta=None, W=28, J=14,
         role='pre-specified main improvement: daily choice among the five rules above'),
    dict(id='S_adaptive_W14', kind='adaptive', theta=None, W=14, J=14, sensitivity=True,
         role='residual window sensitivity W=14'),
    dict(id='S_adaptive_W56', kind='adaptive', theta=None, W=56, J=14, sensitivity=True,
         role='residual window sensitivity W=56'),
]

REFERENCE = dict(   # task book table 2, locked lag results, 2025-02..2025-12
    planned_kWh=20248566.691808097,
    emergency_kWh=519323.55605119443,
    planned_cost_yuan=12265436.51341214,
    emergency_cost_yuan=3112931.8972103912,
    total_cost_yuan=15378368.410622533,
    evaluation_initial_kWh=8801.462273333342,
    final_kWh=6414.659659256388,
)
REFERENCE_ABS_TOL = 1e-4
SELECTED_DATES = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']
DISPATCH_COLUMNS = [
    'strategy_id', 'date', 'issue_time', 'slot', 'start_time', 'end_time',
    'price_yuan_kWh', 'load_kW', 'pv_kW', 'load_forecast_kW', 'pv_forecast_kW',
    'net_forecast_kWh', 'residual_adjustment_kWh', 'protected_net_kWh',
    'theta', 'theta_preset', 'residual_days', 'fallback_reason',
    'planned_kWh', 'charge_kWh', 'discharge_kWh', 'emergency_kWh', 'unused_kWh',
    'state_start_kWh', 'state_end_kWh', 'nominal_state_end_kWh',
    'planned_cost_yuan', 'emergency_cost_yuan',
]


def theta_label(theta):
    if isinstance(theta, str):
        return theta
    return f'{float(theta):.2f}'


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------
# forecast / residual archive (strategy independent: same predictor for every policy)
# --------------------------------------------------------------------------------------
class Archive:
    """Point forecasts and realized residuals, indexed by day. Built from the lag predictor."""

    def __init__(self, load, pv, n_days):
        dt = DT
        self.load = load
        self.pv = pv
        self.n_days = n_days
        self.pred_l = np.full((n_days, 144), np.nan)
        self.pred_v = np.full((n_days, 144), np.nan)
        for k in range(1, n_days):
            pred_l, pred_v, _ = bm.predict(load[:k], pv[:k], 'lag')
            self.pred_l[k] = pred_l
            self.pred_v[k] = pred_v
        self.n_forecast = (self.pred_l - self.pred_v) * dt
        self.n_actual = (load - pv) * dt
        self.eps = self.n_actual - self.n_forecast
        self._cache = {}
        self.solve_calls = 0

    def residual_dates(self, k, W):
        """Valid residual dates strictly before day k, most recent W of them."""
        lo = max(VALID_FIRST_DAY, k - W)
        return int(lo), int(k)

    def adjustment(self, k, W, theta):
        """Return (r[144], m, reason). theta is the string 'base' or a float quantile level."""
        key = (W, theta_label(theta), k)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        lo, hi = self.residual_dates(k, W)
        m = hi - lo
        if theta == 'base':
            out = (np.zeros(144), m, '')
        elif m < MIN_RESIDUAL_DAYS:
            out = (np.zeros(144), m, f'insufficient_residual_dates(m={m}<{MIN_RESIDUAL_DAYS})')
        else:
            ordered = np.sort(self.eps[lo:hi], axis=0)
            idx = int(math.ceil(m * float(theta))) - 1
            assert 0 <= idx < m, (m, theta, idx)
            out = (ordered[idx].copy(), m, '')
        self._cache[key] = out
        return out

    def solve_plan(self, protected_net_kWh, price, initial):
        """Thin wrapper: same MILP kernel, matrix structure, variable order and solver settings.

        The kernel builds the bus balance as (load - pv) * dt, so passing load = net/dt with
        pv = 0 feeds the protected net demand expressed in kWh without touching the kernel.
        """
        load_kw = np.asarray(protected_net_kWh, dtype=float) / DT
        pv_kw = np.zeros_like(load_kw)
        self.solve_calls += 1
        (q, c, d, w, state), summary = core.solve(
            load_kw, pv_kw, price, integer=True,
            initial_kWh=initial, terminal_kWh=initial)
        return q, state, summary

    def plan_day(self, k, theta, W, price, initial):
        """Return the frozen plan and the nominal trajectory for one day, with fallback handling."""
        r, m, reason = self.adjustment(k, W, theta)
        protected = self.n_forecast[k] + r
        try:
            q, nom_state, solver = self.solve_plan(protected, price, initial)
            solver['fallback'] = False
        except Exception as exc:                                    # noqa: BLE001 - recorded, not hidden
            q = np.maximum(protected, 0.0)
            nom_state = np.full(145, float(initial))
            solver = dict(fallback=True, elapsed_seconds=0.0, mip_gap=0.0, checks={},
                          message=f'MILP fallback: {exc}')
        return q, nom_state, solver, protected, r, m, reason


# --------------------------------------------------------------------------------------
# one simulated day: plan, then causal feedback on the realized net demand
# --------------------------------------------------------------------------------------
def execute_day(k, plan, load_day, pv_day, initial, battery=True):
    c, d, e, w, states, checks = bm.control(plan, load_day, pv_day, initial, battery=battery)
    return dict(charge=c, discharge=d, emergency=e, unused=w, states=states, checks=checks)


def emergency_event_rows(date, emergency, price):
    active = emergency > 1e-6
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    stops = np.flatnonzero(active & ~np.r_[active[1:], False]) + 1
    rows = []
    for s, t in zip(starts, stops):
        rows.append(dict(
            date=str(pd.Timestamp(date).date()),
            interval=f'{core.label(int(s) * 10)}-{core.label(int(t) * 10)}',
            start_slot=int(s), end_slot=int(t),
            emergency_kWh=float(emergency[s:t].sum()),
            emergency_cost_yuan=float(5.0 * (emergency[s:t] * price[s:t]).sum())))
    return rows


def day_records(strategy_id, k, theta, theta_preset, dates, load, pv, archive, price,
                initial, plan, nom_state, executed, protected, r, m, reason, solver):
    date = dates[k]
    starts = pd.date_range(date, periods=144, freq='10min')
    planned_cost = plan * price
    emergency_cost = 5.0 * executed['emergency'] * price
    frame = pd.DataFrame({
        'strategy_id': strategy_id,
        'date': str(date.date()),
        'issue_time': date,
        'slot': np.arange(144),
        'start_time': starts,
        'end_time': starts + pd.Timedelta(minutes=10),
        'price_yuan_kWh': price,
        'load_kW': load[k],
        'pv_kW': pv[k],
        'load_forecast_kW': archive.pred_l[k],
        'pv_forecast_kW': archive.pred_v[k],
        'net_forecast_kWh': archive.n_forecast[k],
        'residual_adjustment_kWh': r,
        'protected_net_kWh': protected,
        'theta': theta_label(theta),
        'theta_preset': theta_label(theta_preset),
        'residual_days': m,
        'fallback_reason': reason,
        'planned_kWh': plan,
        'charge_kWh': executed['charge'],
        'discharge_kWh': executed['discharge'],
        'emergency_kWh': executed['emergency'],
        'unused_kWh': executed['unused'],
        'state_start_kWh': executed['states'][:-1],
        'state_end_kWh': executed['states'][1:],
        'nominal_state_end_kWh': np.asarray(nom_state)[1:],
        'planned_cost_yuan': planned_cost,
        'emergency_cost_yuan': emergency_cost,
    })[DISPATCH_COLUMNS]
    daily = dict(
        strategy_id=strategy_id, date=str(date.date()),
        initial_kWh=float(initial), final_kWh=float(executed['states'][-1]),
        planned_kWh=float(plan.sum()), emergency_kWh=float(executed['emergency'].sum()),
        unused_kWh=float(executed['unused'].sum()),
        charge_kWh=float(executed['charge'].sum()), discharge_kWh=float(executed['discharge'].sum()),
        planned_cost_yuan=float(planned_cost.sum()), emergency_cost_yuan=float(emergency_cost.sum()),
        total_cost_yuan=float(planned_cost.sum() + emergency_cost.sum()),
        emergency_slots=int((executed['emergency'] > 1e-6).sum()),
        emergency_events=len(emergency_event_rows(date, executed['emergency'], price)),
        theta=theta_label(theta), theta_preset=theta_label(theta_preset),
        residual_days=int(m), fallback_reason=reason,
        solver_seconds=float(solver['elapsed_seconds']), mip_gap=float(solver['mip_gap']),
        milp_fallback=bool(solver.get('fallback', False)))
    return frame, daily


# --------------------------------------------------------------------------------------
# public January warm-up (built once, every policy forks from its 2025-02-01 state)
# --------------------------------------------------------------------------------------
def run_warmup(archive, price, dates):
    """Replicate the locked lag run for 2025-01-01..2025-01-31 and keep the daily begin states."""
    load, pv = archive.load, archive.pv
    state = float(CONFIG['initial_kWh'])
    states = {}
    rows = []
    checks = {}
    for k in range(0, 31):
        initial = state
        states[k] = initial
        if k == 0:
            plan = np.zeros(144)
            nom_state = np.full(145, initial)
            solver = dict(elapsed_seconds=0.0, mip_gap=0.0, checks={}, fallback=False)
        else:
            plan, nom_state, solver, _, _, _, _ = archive.plan_day(k, 'base', 28, price, initial)
        executed = execute_day(k, plan, load[k], pv[k], initial, battery=(k > 0))
        for name, value in executed['checks'].items():
            checks[name] = max(checks.get(name, 0.0), value)
        if k == 0:
            assert np.abs(executed['emergency'] - np.maximum(load[0] - pv[0], 0) / 6).max() < 1e-9
            assert np.allclose(executed['states'], initial, atol=0, rtol=0)
        state = float(executed['states'][-1])
        rows.append(dict(
            day_index=k, date=str(dates[k].date()), initial_kWh=initial, final_kWh=state,
            planned_kWh=float(plan.sum()), emergency_kWh=float(executed['emergency'].sum()),
            charge_kWh=float(executed['charge'].sum()), discharge_kWh=float(executed['discharge'].sum()),
            unused_kWh=float(executed['unused'].sum()),
            planned_cost_yuan=float((plan * price).sum()),
            emergency_cost_yuan=float(5.0 * (executed['emergency'] * price).sum()),
            note='cold start: zero plan, battery idle' if k == 0 else 'lag policy, shared warm-up'))
    states[31] = state       # 2025-02-01 00:00 begin state for every policy
    return states, pd.DataFrame(rows), state, checks


# --------------------------------------------------------------------------------------
# adaptive parameter selection
# --------------------------------------------------------------------------------------
def choose_candidate(scores, previous_theta, tolerance=TIE_TOLERANCE_YUAN):
    """Tie rule: within tolerance of the best window cost, keep yesterday's parameter if tied."""
    best = min(scores.values())
    tied = [th for th in CANDIDATE_ORDER if scores[th] <= best + tolerance]
    chosen = previous_theta if previous_theta in tied else tied[0]
    return chosen, tied


def select_theta(k, W, J, history, archive, price, dates, nu, previous_theta):
    """Simulate the five fixed candidates over the past J ended days from one shared start state."""
    a = max(VALID_FIRST_DAY, k - J)
    assert 1 <= a < k
    start = float(history[a])
    per_candidate = {}
    for theta in CANDIDATE_ORDER:
        state = start
        score = 0.0
        fallback_days = 0
        residual_days = []
        solver_failures = 0
        for j in range(a, k):
            assert max(archive.residual_dates(j, W)[0], 1) <= j - 1, 'history leak in residual window'
            q, nom_state, solver, protected, r, m, reason = archive.plan_day(j, theta, W, price, state)
            executed = execute_day(j, q, archive.load[j], archive.pv[j], state)
            score += float((q * price).sum()) + float(5.0 * (executed['emergency'] * price).sum())
            fallback_days += int(bool(reason))
            solver_failures += int(bool(solver.get('fallback', False)))
            residual_days.append(m)
            state = float(executed['states'][-1])
        per_candidate[theta] = dict(
            score_yuan=score, end_state=state, start_state=start,
            window_start=a, window_end=k - 1,
            residual_days_min=int(min(residual_days)), residual_days_max=int(max(residual_days)),
            fallback_days=int(fallback_days), solver_failures=int(solver_failures),
            valuation_score_yuan=score - nu * (state - start))
    chosen, tied = choose_candidate({th: per_candidate[th]['score_yuan'] for th in CANDIDATE_ORDER},
                                    previous_theta)
    ranking = sorted(CANDIDATE_ORDER,
                     key=lambda th: (per_candidate[th]['score_yuan'], CANDIDATE_ORDER.index(th)))
    for rank, theta in enumerate(ranking, start=1):
        per_candidate[theta]['rank'] = rank
        per_candidate[theta]['selected'] = int(theta == chosen)
        per_candidate[theta]['theta'] = theta_label(theta)
    return chosen, per_candidate, tied


# --------------------------------------------------------------------------------------
# targeted boundary and information checks (task book section 11.3)
# --------------------------------------------------------------------------------------
def run_boundary_checks(archive, price):
    out = {}
    sample = np.array([-2., 0., 1., 3., 10.])

    def empirical(a):
        m = len(sample)
        return float(np.sort(sample)[int(math.ceil(m * a)) - 1])

    out['manual_quantile_index'] = dict(
        sample=sample.tolist(), q50=empirical(.5), q65=empirical(.65),
        q80=empirical(.8), q90=empirical(.9),
        expected='q50=1, q80=3, q90=10 (five ordered values)',
        passed=bool(empirical(.5) == 1. and empirical(.8) == 3. and empirical(.9) == 10.))

    probe = copy.copy(archive)
    probe.eps = np.zeros_like(archive.eps)
    probe._cache = {}
    zero_ok = True
    for theta in (0.50, 0.65, 0.80, 0.90):
        r, m, reason = probe.adjustment(100, 28, theta)
        zero_ok &= bool(reason == '' and np.abs(r).max() == 0.0 and m == 28)
    out['zero_residual_gives_no_correction'] = dict(
        tested_quantiles=[0.50, 0.65, 0.80, 0.90], max_abs_adjustment_kWh=0.0, passed=zero_ok)

    r, m, reason = archive.adjustment(5, 28, 0.80)
    out['insufficient_history_falls_back'] = dict(
        day_index=5, valid_residual_dates=int(m), reason=reason,
        max_abs_adjustment_kWh=float(np.abs(r).max()),
        passed=bool(m == 4 and reason != '' and np.abs(r).max() == 0.0))
    r, m, reason = archive.adjustment(31, 56, 0.80)
    out['long_window_uses_only_existing_days'] = dict(
        day_index=31, requested_W=56, used_valid_dates=int(m), reason=reason,
        passed=bool(m == 30 and reason == ''))

    probe2 = copy.copy(archive)
    probe2.eps = np.full_like(archive.eps, -5.0)
    probe2._cache = {}
    r, _, _ = probe2.adjustment(100, 28, 0.50)
    protected = archive.n_forecast[100] + r
    out['negative_correction_and_negative_net_kept'] = dict(
        adjustment_kWh=float(r[0]), protected_net_min_kWh=float(protected.min()),
        forecast_net_min_kWh=float(archive.n_forecast[100].min()),
        protected_net_max_kWh=float(protected.max()),
        passed=bool(abs(r.max() + 5) < 1e-12 and abs(r.min() + 5) < 1e-12 and protected.min() < 0))

    valid_forecast = slice(VALID_FIRST_DAY, None)
    forecast_net = archive.n_forecast[valid_forecast]
    roundtrip = float(np.abs((forecast_net / DT) * DT - forecast_net).max())
    out['unit_conversion_applied_once'] = dict(
        max_roundtrip_error_kWh=roundtrip, threshold_kWh=1e-9,
        compared_days='day index 1..364 (day 0 has no point forecast by design)',
        forecast_matches_power_times_dt=bool(np.array_equal(
            forecast_net, (archive.pred_l[valid_forecast] - archive.pred_v[valid_forecast]) * DT)),
        passed=bool(roundtrip < 1e-9))

    cap = 5000 * DT
    _, d0, e0, _, _, _ = bm.control(np.array([0.]), np.array([12000.]), np.array([0.]), 1200.)
    _, d1, e1, _, _, _ = bm.control(np.array([0.]), np.array([12000.]), np.array([0.]), 10800.)
    _, _, _, w2, _, _ = bm.control(np.array([2000.]), np.array([0.]), np.array([0.]), 10800.)
    out['battery_and_power_boundaries'] = dict(
        empty_battery_discharge_kWh=float(d0[0]), empty_battery_emergency_kWh=float(e0[0]),
        full_battery_power_limited_discharge_kWh=float(d1[0]),
        full_battery_remaining_emergency_kWh=float(e1[0]),
        full_battery_surplus_unused_kWh=float(w2[0]), segment_limit_kWh=float(cap),
        passed=bool(abs(d0[0]) < 1e-12 and abs(e0[0] - 2000) < 1e-9
                    and abs(d1[0] - cap) < 1e-9 and abs(e1[0] - (2000 - cap)) < 1e-9
                    and abs(w2[0] - 2000) < 1e-9))

    scores = {'base': 100.0, 0.50: 100.0 + 1e-9, 0.65: 100.0 + 1e-3, 0.80: 120.0, 0.90: 130.0}
    picked_none, tied_none = choose_candidate(scores, None)
    picked_prev, _ = choose_candidate(scores, 0.50)
    picked_outside, _ = choose_candidate(scores, 0.65)
    out['tie_break_rule'] = dict(
        scores={theta_label(t): v for t, v in scores.items()},
        tied=[theta_label(t) for t in tied_none], chosen_without_previous=theta_label(picked_none),
        chosen_keeping_previous=theta_label(picked_prev),
        chosen_when_previous_outside_tolerance=theta_label(picked_outside),
        passed=bool(tied_none == ['base', 0.50] and picked_none == 'base'
                    and picked_prev == 0.50 and picked_outside == 'base'))

    price_positive = bool((price > 0).all())
    out['price_input'] = dict(all_positive=price_positive, n_slots=int(len(price)),
                              median_yuan_per_kWh=float(np.median(price)),
                              nu_yuan_per_kWh=float(np.median(price) / 0.9), passed=price_positive)
    return out


# --------------------------------------------------------------------------------------
# policy runner
# --------------------------------------------------------------------------------------
def run_policy(policy, bundle):
    strategy_id = policy['id']
    archive = bundle['archive']
    price = bundle['price']
    dates = bundle['dates']
    nu = bundle['nu']
    warm = bundle['warmup_states']
    a_monthly = bundle.get('a_monthly')
    out_dir = OUT / strategy_id
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluation_start = pd.Timestamp(EVALUATION_START)

    preset_label = theta_label(policy['theta']) if policy['theta'] is not None else 'adaptive'
    stop_day = int(bundle.get('stop_day') or len(dates))
    history = {k: float(v) for k, v in warm.items() if k <= 31}
    state = float(warm[31])
    previous_theta = None
    dispatch_rows, daily_rows, event_rows, selection_rows = [], [], [], []
    checks = {}
    solver_seconds = 0.0
    max_gap = 0.0
    fallback_days = 0
    solver_failures = 0

    for k in range(31, min(stop_day, len(dates))):
        initial = state
        history[k] = initial
        assert archive.residual_dates(k, policy['W'] or CONFIG['residual_window_default'])[1] == k, \
            'the residual window for the day being planned must end strictly before it'
        if policy['kind'] == 'adaptive':
            chosen, per_candidate, tied = select_theta(
                k, policy['W'], policy['J'], history, archive, price, dates, nu, previous_theta)
            theta = chosen
            for cand in CANDIDATE_ORDER:
                item = per_candidate[cand]
                selection_rows.append(dict(
                    strategy_id=strategy_id, date=str(dates[k].date()), day_index=k,
                    residual_window_days=policy['W'], validation_window_days=policy['J'],
                    window_start=item['window_start'], window_end=item['window_end'],
                    common_start_state_kWh=item['start_state'], candidate_theta=item['theta'],
                    window_cost_yuan=item['score_yuan'],
                    window_end_state_kWh=item['end_state'],
                    valuation_score_yuan=item['valuation_score_yuan'],
                    rank=item['rank'], selected=item['selected'],
                    residual_days_min=item['residual_days_min'],
                    residual_days_max=item['residual_days_max'],
                    fallback_days=item['fallback_days'],
                    solver_failures=item['solver_failures'],
                    tied_within_tolerance=int(item['theta'] in {theta_label(t) for t in tied})))
            previous_theta = chosen
        else:
            theta = policy['theta']

        q, nom_state, solver, protected, r, m, reason = archive.plan_day(
            k, theta, policy['W'] or CONFIG['residual_window_default'], price, initial)
        executed = execute_day(k, q, archive.load[k], archive.pv[k], initial)
        for name, value in executed['checks'].items():
            checks[name] = max(checks.get(name, 0.0), value)
        for name, value in solver.get('checks', {}).items():
            checks['nominal_' + name] = max(checks.get('nominal_' + name, 0.0), value)
        solver_seconds += float(solver['elapsed_seconds'])
        max_gap = max(max_gap, float(solver['mip_gap']))
        fallback_days += int(bool(reason))
        solver_failures += int(bool(solver.get('fallback', False)))

        frame, daily = day_records(strategy_id, k, theta, preset_label, dates, archive.load,
                                   archive.pv, archive, price, initial, q, nom_state, executed,
                                   protected, r, m, reason, solver)
        dispatch_rows.append(frame)
        daily_rows.append(daily)
        event_rows.extend(emergency_event_rows(dates[k], executed['emergency'], price))
        state = float(executed['states'][-1])
        history[k + 1] = state
        if k % 40 == 0:
            print(f'    {strategy_id}: day {k} ({dates[k].date()}) state={state:.3f}', flush=True)

    dispatch = pd.concat(dispatch_rows, ignore_index=True)
    daily = pd.DataFrame(daily_rows)
    events = pd.DataFrame(event_rows, columns=['date', 'interval', 'start_slot', 'end_slot',
                                               'emergency_kWh', 'emergency_cost_yuan'])
    selected = dispatch[dispatch.date.isin(SELECTED_DATES)]
    month = daily.assign(month=daily.date.str[:7]).groupby('month').agg(
        planned_kWh=('planned_kWh', 'sum'), emergency_kWh=('emergency_kWh', 'sum'),
        unused_kWh=('unused_kWh', 'sum'), charge_kWh=('charge_kWh', 'sum'),
        discharge_kWh=('discharge_kWh', 'sum'), planned_cost_yuan=('planned_cost_yuan', 'sum'),
        emergency_cost_yuan=('emergency_cost_yuan', 'sum'), total_cost_yuan=('total_cost_yuan', 'sum'),
        emergency_slots=('emergency_slots', 'sum'), emergency_events=('emergency_events', 'sum'),
    ).reset_index()
    if a_monthly is not None:
        month['delta_total_cost_yuan_vs_A'] = (
            month.total_cost_yuan.to_numpy()
            - a_monthly.set_index('month').total_cost_yuan.reindex(month.month).to_numpy())
    else:
        month['delta_total_cost_yuan_vs_A'] = 0.0

    dispatch.to_csv(out_dir / 'dispatch.csv', index=False, encoding='utf-8-sig')
    daily.to_csv(out_dir / 'daily_summary.csv', index=False, encoding='utf-8-sig')
    month.to_csv(out_dir / 'monthly_summary.csv', index=False, encoding='utf-8-sig')
    events.to_csv(out_dir / 'emergency_events.csv', index=False, encoding='utf-8-sig')
    if selected.size:
        selected.to_csv(out_dir / 'selected_dates_dispatch.csv', index=False, encoding='utf-8-sig')

    forecast_error = ((dispatch.load_kW - dispatch.pv_kW)
                      - (dispatch.load_forecast_kW - dispatch.pv_forecast_kW))
    totals = dict(
        strategy_id=strategy_id, kind=policy['kind'],
        residual_window=policy['W'], validation_window=policy['J'],
        theta_preset=theta_label(policy['theta']) if policy['theta'] is not None else 'adaptive',
        planned_kWh=float(daily.planned_kWh.sum()), emergency_kWh=float(daily.emergency_kWh.sum()),
        unused_kWh=float(daily.unused_kWh.sum()), charge_kWh=float(daily.charge_kWh.sum()),
        discharge_kWh=float(daily.discharge_kWh.sum()),
        planned_cost_yuan=float(daily.planned_cost_yuan.sum()),
        emergency_cost_yuan=float(daily.emergency_cost_yuan.sum()),
        total_cost_yuan=float(daily.total_cost_yuan.sum()),
        emergency_slots=int(daily.emergency_slots.sum()),
        emergency_days=int((daily.emergency_kWh > 1e-6).sum()),
        emergency_events=int(daily.emergency_events.sum()),
        evaluation_initial_kWh=float(daily.initial_kWh.iloc[0]),
        final_kWh=float(daily.final_kWh.iloc[-1]),
        nu_yuan_per_kWh=float(nu),
        adjusted_cost_yuan=float(daily.total_cost_yuan.sum()
                                 - nu * (daily.final_kWh.iloc[-1] - daily.initial_kWh.iloc[0])),
        solver_seconds=float(solver_seconds), max_mip_gap=float(max_gap),
        fallback_days=int(fallback_days), solver_failures=int(solver_failures),
        load_mae_kW=float((dispatch.load_kW - dispatch.load_forecast_kW).abs().mean()),
        pv_mae_kW=float((dispatch.pv_kW - dispatch.pv_forecast_kW).abs().mean()),
        net_mae_kW=float(forecast_error.abs().mean()),
        net_rmse_kW=float(np.sqrt((forecast_error ** 2).mean())),
        net_bias_kW=float(forecast_error.mean()),
        evaluated_slots=int(len(dispatch)), evaluated_days=int(daily.date.nunique()),
    )
    totals['emergency_cost_share'] = totals['emergency_cost_yuan'] / totals['total_cost_yuan']
    validation = dict(
        checks={k: float(v) for k, v in checks.items()},
        segments=int(len(dispatch)),
        day_boundary_max_error_kWh=float(np.abs(
            daily.initial_kWh.to_numpy()[1:] - daily.final_kWh.to_numpy()[:-1]).max()),
        cost_reconciliation_yuan=float(abs(
            dispatch.planned_cost_yuan.sum() + dispatch.emergency_cost_yuan.sum()
            - daily.total_cost_yuan.sum())),
        emergency_reconciliation_kWh=float(abs(
            events.emergency_kWh.sum() - daily.emergency_kWh.sum())),
        emergency_cost_reconciliation_yuan=float(abs(
            events.emergency_cost_yuan.sum() - daily.emergency_cost_yuan.sum())),
        energy_identity_residual_kWh=float(
            dispatch.planned_kWh.sum() + dispatch.emergency_kWh.sum()
            + dispatch.pv_kW.sum() * DT - dispatch.load_kW.sum() * DT - dispatch.unused_kWh.sum()
            - (0.1 * dispatch.charge_kWh.sum() + (1 / 0.9 - 1) * dispatch.discharge_kWh.sum())
            - (daily.final_kWh.iloc[-1] - daily.initial_kWh.iloc[0])),
        residual_cutoff_max_day_index_used=int(max(
            archive.residual_dates(k, policy['W'] or 28)[1] - 1 for k in range(31, len(dates)))),
        monthly_reconciliation_yuan=float(abs(
            month.total_cost_yuan.sum() - daily.total_cost_yuan.sum())),
    )
    (out_dir / 'validation.json').write_text(
        json.dumps(dict(totals=totals, validation=validation), ensure_ascii=False, indent=2),
        encoding='utf-8')
    return dict(totals=totals, validation=validation, month=month.to_dict(orient='records'),
                selection=selection_rows)


def make_bundle(load, pv, price, dates, warmup_states, nu, a_monthly=None, stop_day=None):
    return dict(archive=Archive(load, pv, len(dates)), price=price, dates=dates,
                warmup_states=warmup_states, nu=nu, a_monthly=a_monthly,
                load=load, pv=pv, stop_day=stop_day)


def _worker(payload):
    policy, bundle = payload
    return run_policy(policy, bundle)


# --------------------------------------------------------------------------------------
# figures data + main
# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['repro', 'smoke', 'full'], default='full')
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()

    assert Path(sys.prefix).name == 'math_modeling', sys.prefix
    np.random.seed(CONFIG['seed'])
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    load, pv, price, source_hashes = bm.read_sources()
    dates = bm.DATES
    assert load.shape == pv.shape == (365, 144) and price.shape == (144,)
    assert np.isfinite(price).all() and (price > 0).all()
    nu = float(np.median(price) / 0.9)

    archive = Archive(load, pv, len(dates))
    warmup_states, warmup_rows, warmup_end, warmup_checks = run_warmup(archive, price, dates)
    warmup_rows.to_csv(OUT / 'warmup_january.csv', index=False, encoding='utf-8-sig')
    boundary = run_boundary_checks(archive, price)
    boundary_passed = all(v['passed'] for v in boundary.values())
    print(f'boundary checks passed = {boundary_passed}', flush=True)

    # point-forecast / residual archive, one row per day and slot
    residual_frame = pd.DataFrame({
        'date': np.repeat([str(d.date()) for d in dates], 144),
        'slot': np.tile(np.arange(144), len(dates)),
        'issue_time': np.repeat(dates, 144),
        'load_forecast_kW': archive.pred_l.ravel(),
        'pv_forecast_kW': archive.pred_v.ravel(),
        'net_forecast_kWh': archive.n_forecast.ravel(),
        'load_kW': load.ravel(), 'pv_kW': pv.ravel(),
        'net_actual_kWh': archive.n_actual.ravel(),
        'residual_kWh': archive.eps.ravel(),
    })
    residual_frame['residual_available_at'] = pd.to_datetime(residual_frame.date) + pd.Timedelta(days=1)
    residual_frame['valid'] = [int(not np.isnan(v)) for v in residual_frame.load_forecast_kW]
    residual_frame.to_csv(OUT / 'forecast_residuals.csv', index=False, encoding='utf-8-sig')

    reference_check = None
    a_monthly = None
    results = {}
    selection_log = []

    def absorb(result):
        results[result['totals']['strategy_id']] = result
        selection_log.extend(result['selection'])

    print(f'warm-up done: 2025-02-01 begin state = {warmup_end:.12f} kWh', flush=True)
    print(f'reference value = {REFERENCE["evaluation_initial_kWh"]:.12f} kWh, '
          f'abs diff = {abs(warmup_end - REFERENCE["evaluation_initial_kWh"]):.3e}', flush=True)

    if args.mode == 'repro':
        absorb(run_policy(POLICIES[0], make_bundle(load, pv, price, dates, warmup_states, nu)))
    elif args.mode == 'smoke':
        absorb(run_policy(POLICIES[5], make_bundle(load, pv, price, dates, warmup_states, nu,
                                                  stop_day=45)))
    else:
        a_result = run_policy(POLICIES[0], make_bundle(load, pv, price, dates, warmup_states, nu))
        absorb(a_result)
        a_monthly = pd.DataFrame(a_result['month'])
        remaining = POLICIES[1:]
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(remaining))) as pool:
            futures = {pool.submit(_worker, (pol, make_bundle(load, pv, price, dates,
                                                              warmup_states, nu, a_monthly))): pol
                       for pol in remaining}
            for future in as_completed(futures):
                absorb(future.result())

    table = pd.DataFrame([r['totals'] for r in results.values()])
    if 'A_base' in results:
        base_total = results['A_base']['totals']
        for column, source in [('delta_planned_cost_yuan', 'planned_cost_yuan'),
                               ('delta_emergency_cost_yuan', 'emergency_cost_yuan'),
                               ('delta_total_cost_yuan', 'total_cost_yuan'),
                               ('delta_adjusted_cost_yuan', 'adjusted_cost_yuan')]:
            table[column] = table[source] - base_total[source]
        table['delta_total_pct_vs_A'] = 100 * table.delta_total_cost_yuan / base_total['total_cost_yuan']
        table['delta_adjusted_pct_vs_A'] = 100 * table.delta_adjusted_cost_yuan / base_total['total_cost_yuan']
    table.to_csv(OUT / 'comparison.csv', index=False, encoding='utf-8-sig')
    if selection_log:
        pd.DataFrame(selection_log).to_csv(OUT / 'selection_log.csv', index=False, encoding='utf-8-sig')

    if args.mode != 'full':
        comparison = compare_to_reference(results.get('A_base'))
        payload = dict(baseline_reference_check=comparison, boundary_checks=boundary,
                       boundary_checks_all_passed=boundary_passed)
        (OUT / 'repro_check.json').write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        if args.mode == 'smoke':
            smoke_assertions(results, warmup_states)
        return

    write_selected_dates(results)
    comparison = compare_to_reference(results.get('A_base'))
    worst = {}
    for result in results.values():
        for name, value in result['validation']['checks'].items():
            worst[name] = max(worst.get(name, 0.0), float(value))
    validation_summary = dict(
        thresholds=dict(physical_violation_kWh=CONFIG['tolerance'],
                        cost_reconciliation_yuan=1e-4,
                        energy_identity_residual_kWh=1e-4,
                        day_boundary_state_kWh=1e-9,
                        baseline_reference_abs_tolerance=REFERENCE_ABS_TOL),
        baseline_reference_check=comparison,
        baseline_reproduced_exactly=bool(comparison.get('all_within_tolerance', False)),
        boundary_checks=boundary, boundary_checks_all_passed=boundary_passed,
        worst_physical_violation_kWh=worst,
        warmup=dict(feb1_state_kWh=warmup_end,
                    checks={k: float(v) for k, v in warmup_checks.items()}),
        per_policy={sid: r['validation'] for sid, r in results.items()},
    )
    (OUT / 'validation_summary.json').write_text(
        json.dumps(validation_summary, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest = dict(
        run_time_utc=started.isoformat(),
        finished_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.perf_counter() - t0,
        executable=sys.executable, python=sys.version, numpy=np.__version__,
        pandas=pd.__version__, scipy=scipy.__version__, cpu_count=os.cpu_count(),
        jobs=args.jobs,
        source_sha256=source_hashes,
        code_sha256={'code/08_q2_quantile_experiment.py': sha256_file(CODE_08)},
        warmup_2025_02_01_state_kWh=warmup_end,
        warmup_checks={k: float(v) for k, v in warmup_checks.items()},
        baseline_reference_check=comparison,
        policy_status={sid: dict(completed=True, slots=r['totals']['evaluated_slots'],
                                 fallback_days=r['totals']['fallback_days'],
                                 solver_failures=r['totals']['solver_failures'])
                       for sid, r in results.items()},
        nu_yuan_per_kWh=nu, config=CONFIG, policies=POLICIES)
    (OUT / 'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                           encoding='utf-8')
    (OUT / 'config.json').write_text(json.dumps(
        dict(fixed_conditions=CONFIG, policies=POLICIES, reference=REFERENCE,
             tie_tolerance_yuan=TIE_TOLERANCE_YUAN, candidate_order=[theta_label(t) for t in CANDIDATE_ORDER]),
        ensure_ascii=False, indent=2), encoding='utf-8')
    print(table[['strategy_id', 'planned_cost_yuan', 'emergency_cost_yuan', 'total_cost_yuan',
                 'delta_total_cost_yuan']].to_string(index=False), flush=True)
    print(f'wall seconds = {time.perf_counter() - t0:.1f}', flush=True)


def compare_to_reference(a_result):
    if a_result is None:
        return {}
    totals = a_result['totals']
    out = {}
    for key, reference in REFERENCE.items():
        actual = totals[key]
        out[key] = dict(actual=float(actual), reference=float(reference),
                        abs_diff=float(abs(actual - reference)),
                        within_tolerance=bool(abs(actual - reference) <= REFERENCE_ABS_TOL))
    out['all_within_tolerance'] = bool(all(v['within_tolerance'] for v in out.values()
                                           if isinstance(v, dict)))
    return out


def smoke_assertions(results, warmup_states):
    """Strict checks on the adaptive flow: shared start, residual cut-off, state continuity."""
    key = 'B_adaptive'
    assert key in results, results.keys()
    daily = pd.read_csv(OUT / key / 'daily_summary.csv')
    selection = pd.read_csv(OUT / 'selection_log.csv')
    first = selection[selection.date == daily.date.iloc[0]]
    assert len(first) == len(CANDIDATE_ORDER)
    assert first.common_start_state_kWh.nunique() == 1, 'candidates must share the window start'
    assert (first.window_end == first.window_start + first.validation_window_days - 1).all()
    assert first.window_end.max() < 31, 'the first verification window must be strictly in January'
    assert np.allclose(daily.initial_kWh.to_numpy()[1:], daily.final_kWh.to_numpy()[:-1],
                       atol=1e-9, rtol=0), 'state must carry across evaluation days'
    assert abs(daily.initial_kWh.iloc[0] - warmup_states[31]) < 1e-9
    dispatch = pd.read_csv(OUT / key / 'dispatch.csv')
    assert len(dispatch) == 14 * 144, len(dispatch)
    residual_days = dispatch.residual_days.to_numpy()
    assert residual_days.min() >= MIN_RESIDUAL_DAYS
    print('smoke assertions passed: shared window start, January-only window, '
          'residual cut-off before each simulated day, continuous state', flush=True)


def write_selected_dates(results):
    folder = OUT / 'selected_dates'
    folder.mkdir(parents=True, exist_ok=True)
    ten, four, events, bounds = [], [], [], []
    for strategy_id in results:
        dispatch = pd.read_csv(OUT / strategy_id / 'dispatch.csv',
                               dtype={'theta': str, 'theta_preset': str, 'fallback_reason': str})
        daily = pd.read_csv(OUT / strategy_id / 'daily_summary.csv',
                            dtype={'theta': str, 'theta_preset': str, 'fallback_reason': str})
        picked = dispatch[dispatch.date.isin(SELECTED_DATES)].copy()
        ten.append(picked[picked.slot.isin([60, 72, 84, 96, 108, 120])][
            ['strategy_id', 'date', 'start_time', 'end_time', 'slot',
             'planned_kWh', 'emergency_kWh', 'planned_cost_yuan', 'emergency_cost_yuan']])
        picked['block'] = picked.slot // 24
        grouped = picked.groupby(['strategy_id', 'date', 'block'])[
            ['charge_kWh', 'discharge_kWh']].sum().reset_index()
        grouped['interval'] = [f'{b * 4:02d}:00-{(b + 1) * 4:02d}:00' for b in grouped.block]
        four.append(grouped[['strategy_id', 'date', 'interval', 'charge_kWh', 'discharge_kWh']])
        ev = pd.read_csv(OUT / strategy_id / 'emergency_events.csv')
        events.append(ev[ev.date.isin(SELECTED_DATES)].assign(strategy_id=strategy_id))
        sel = daily[daily.date.isin(SELECTED_DATES)][
            ['strategy_id', 'date', 'initial_kWh', 'final_kWh', 'planned_kWh', 'emergency_kWh',
             'planned_cost_yuan', 'emergency_cost_yuan', 'total_cost_yuan', 'theta']]
        bounds.append(sel)
    pd.concat(ten, ignore_index=True).to_csv(folder / 'selected_10min.csv', index=False, encoding='utf-8-sig')
    pd.concat(four, ignore_index=True).to_csv(folder / 'selected_4hour.csv', index=False, encoding='utf-8-sig')
    pd.concat(events, ignore_index=True).to_csv(folder / 'selected_emergency_events.csv', index=False, encoding='utf-8-sig')
    pd.concat(bounds, ignore_index=True).to_csv(folder / 'selected_boundary_states.csv', index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
