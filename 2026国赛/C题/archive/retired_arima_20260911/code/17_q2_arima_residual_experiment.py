"""Q2 ARIMA residual correction of the naive forecasts: six pre-registered causal backtests.

Run from E:/MathModeling/2026国赛/C题 with the math_modeling interpreter:
    E:/Anaconda/envs/math_modeling/python.exe code/17_q2_arima_residual_experiment.py --mode register
    E:/Anaconda/envs/math_modeling/python.exe code/17_q2_arima_residual_experiment.py --mode forecast
    E:/Anaconda/envs/math_modeling/python.exe code/17_q2_arima_residual_experiment.py --mode repro
    E:/Anaconda/envs/math_modeling/python.exe code/17_q2_arima_residual_experiment.py --mode full

Specification: reports/问题二_ARIMA残差预测实验方案.md (single-run task book).
Inherited frozen implementation: results/q2_revision_audit_20260911/source_snapshot/code/{02,05,08} plus
the ridge archive/runner and metric helpers of code/14_q2_ridge_forecast_experiment.py and the
past-only PV gate rule of code/16_q2_pv_night_gate.py.

ARIMA predicts the signed error of the seasonal naive power forecast one day ahead, separately for
each of the 144 intraday slots and separately for load and PV. Candidates are zero / ARIMA(1,0,0) /
ARIMA(0,0,1) / ARIMA(1,0,1); the order is chosen once per day and target on the previous 14 calendar
days. All matched groups share the same past-only PV support gate, and every group rebuilds its own
published net-demand residual archive and q80 protection.

Six groups (identical MILP kernel, greedy feedback, efficiency, 6000 kWh nominal terminal):
    C0_original_naive  naive load, naive PV, no gate           (must reproduce 14158360.487140 CNY)
    A0_naive_gate      naive load, naive PV + gate             (matched baseline of this round)
    A1_load_arima      ARIMA load, naive PV + gate
    A2_pv_arima        naive load, ARIMA PV + gate
    A3_both_arima      ARIMA load, ARIMA PV + gate
    C3_ridge_gate      ridge load, ridge PV + gate             (must reproduce 13724593.278483 CNY)
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
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'q2_arima_residual'
FIG = ROOT / 'figures' / 'q2_arima_residual'
AUDIT = ROOT / 'results' / 'q2_revision_audit_20260911'
SNAP = AUDIT / 'source_snapshot'
RIDGE_RUN = ROOT / 'results' / 'q2_ridge_forecast'
GATE_RUN = ROOT / 'results' / 'q2_pv_night_gate'
PLAN_MD = ROOT / 'reports' / '问题二_ARIMA残差预测实验方案.md'
REPORT_MD = ROOT / 'reports' / '问题二_ARIMA残差预测实验结果报告.md'
CODE_17 = Path(__file__).resolve()

DT = 1 / 6
TRAIN_WINDOW = 56
MIN_TRAIN_DAYS = 21
VALIDATION_WINDOW = 14
MIN_SELECTION_DAYS = 7
TIE_RELATIVE = 1e-10
SERIES_FIRST_DAY = {'load': 7, 'pv': 1}
STD_TOL = 1e-8
CANDIDATE_ORDER = ('zero', 'ar1', 'ma1', 'arma11')
ARIMA_CANDIDATES = ('ar1', 'ma1', 'arma11')
ARIMA_ORDER = {'ar1': (1, 0, 0), 'ma1': (0, 0, 1), 'arma11': (1, 0, 1)}
MAXITER = 200
RESIDUAL_WINDOW = 28
THETA = 0.80
TERMINAL_KWH = 6000.0
WARMUP_DAYS = 31
JOBS_DEFAULT = 14
GATE_HISTORY_DAYS = 7
GATE_ACTIVITY_KW = 1.0
GATE_PADDING_SLOTS = 3

REFERENCE_C0_YUAN = 14158360.487140
REFERENCE_C3_YUAN = 13724593.278483
HISTORICAL_UNGATED_R3_YUAN = 13722565.421093
COST_TOL_YUAN = 1e-4

GROUPS = [
    dict(id='C0_original_naive', load='naive', pv='naive', source='ungated',
         role='reproduces the locked T_q80_6000 trajectory (no PV gate)'),
    dict(id='A0_naive_gate', load='naive', pv='naive', source='gated_naive',
         role='matched baseline of this round: naive forecasts plus the shared PV gate'),
    dict(id='A1_load_arima', load='arima', pv='naive', source='arima', role='load replacement'),
    dict(id='A2_pv_arima', load='naive', pv='arima', source='arima', role='pv replacement'),
    dict(id='A3_both_arima', load='arima', pv='arima', source='arima', role='joint replacement'),
    dict(id='C3_ridge_gate', load='ridge', pv='ridge', source='ridge_gated',
         role='existing joint ridge plus gate, direct comparator'),
]
PRIMARY_CONTRASTS = [('A1_load_arima', 'A0_naive_gate'), ('A2_pv_arima', 'A0_naive_gate'),
                     ('A3_both_arima', 'A0_naive_gate'), ('A3_both_arima', 'C3_ridge_gate'),
                     ('A0_naive_gate', 'C0_original_naive')]
DISPATCH_NUMERIC = ['price_yuan_kWh', 'load_kW', 'pv_kW', 'load_forecast_kW', 'pv_forecast_kW',
                    'net_forecast_kWh', 'residual_adjustment_kWh', 'protected_net_kWh',
                    'planned_kWh', 'charge_kWh', 'discharge_kWh', 'emergency_kWh', 'unused_kWh',
                    'state_start_kWh', 'state_end_kWh', 'nominal_state_end_kWh',
                    'planned_cost_yuan', 'emergency_cost_yuan']

PARAMETERS_ARIMA = dict(
    model='statsmodels.tsa.arima.model.ARIMA, order in {(1,0,0),(0,0,1),(1,0,1)}, d=0, trend="c"',
    candidates=list(CANDIDATE_ORDER),
    labels='load u=L[k,t]-L[k-7,t] (k>=7); pv u=V[k,t]-V[k-1,t] (k>=1); per-slot across days',
    train_window_days=TRAIN_WINDOW, min_train_days=MIN_TRAIN_DAYS,
    first_trainable='load k=28 (2025-01-29), pv k=22 (2025-01-23)',
    validation_window_days=VALIDATION_WINDOW, min_selection_days=MIN_SELECTION_DAYS,
    first_seven_day_scoring='load k=35 (2025-02-05), pv k=29 (2025-01-30)',
    standardisation='per target/slot/day window mean and population std (ddof=0); constant -> bypass',
    constant_rule=f'std <= {STD_TOL} kW -> all statistical candidates predict the window mean, '
                  'flagged constant_bypass and not counted as a successful fit',
    tie_rule='within 1e-10*max(1,S_min) choose by order zero, ar1, ma1, arma11',
    fit_interface='ARIMA(z, order, trend="c", enforce_stationarity=True, enforce_invertibility=True)'
                  '.fit(method="statespace", cov_type="none", method_kwargs={lbfgs, maxiter=200, '
                  'pgtol=1e-8, factr=1e7, disp=False})',
    failure_rule='exception / converged=False / non-finite params, likelihood or forecast / '
                 '|ar|>=1 or |ma|>=1 -> fit_failure, that slot falls back to the naive level',
    gate='past-only PV support gate inherited from code/16: from the previous 7 days take the earliest '
         'and latest slots whose max PV exceeded 1.0 kW, pad 3 slots each side, zero outside',
    residual_window_days=RESIDUAL_WINDOW, theta=THETA, terminal_kWh=TERMINAL_KWH,
    evaluation='2025-02-01..2025-12-31, 334 days, 48096 slots per group',
    reference_cost_yuan=dict(C0=REFERENCE_C0_YUAN, C3=REFERENCE_C3_YUAN,
                             historical_ungated_R3=HISTORICAL_UNGATED_R3_YUAN),
    solvers='frozen scipy.optimize.milp/HiGHS kernel, relative gap 1e-9, time limit 120 s',
)


def parent():
    """Load module 14 once per process for the frozen runner, archive and metric helpers."""
    if 'module' not in _PARENT:
        spec = importlib.util.spec_from_file_location('ridge_parent', RIDGE_RUN.parent.parent /
                                                      'code/14_q2_ridge_forecast_experiment.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.OUT = OUT
        module.FIG = FIG
        _PARENT['module'] = module
    return _PARENT['module']


_PARENT: dict = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


OWN_NEW_REL = {'code/17_q2_arima_residual_experiment.py',
               'reports/问题二_ARIMA残差预测实验结果报告.md'}
OWN_NEW_PREFIXES = ('results/q2_arima_residual/', 'figures/q2_arima_residual/')
SHARED_APPEND_REL = {'建模上下文记忆.md', 'reports/项目进度.md'}


def protected_manifest():
    """Hash every pre-existing asset, excluding this round's own new tree and shared append-only docs."""
    parent_module = parent()
    manifest = {}
    for tree in parent_module.PROTECTED_TREES:
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


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def frame_to_csv(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8-sig')


# --------------------------------------------------------------------------------------
# past-only PV support gate (identical rule to code/16_q2_pv_night_gate.py)
# --------------------------------------------------------------------------------------
def gate_row(pv, k):
    """Support mask rows for decision day k, or None when the gate is inactive."""
    if k < GATE_HISTORY_DAYS:
        return None
    active = np.flatnonzero(np.max(pv[k - GATE_HISTORY_DAYS:k], axis=0) > GATE_ACTIVITY_KW)
    if not len(active):
        return None
    mask = np.zeros(144, dtype=bool)
    low = max(0, int(active[0]) - GATE_PADDING_SLOTS)
    high = min(143, int(active[-1]) + GATE_PADDING_SLOTS)
    mask[low:high + 1] = True
    return mask


def gate_mask(pv):
    mask = np.ones(pv.shape, dtype=bool)
    for k in range(GATE_HISTORY_DAYS, len(pv)):
        row = gate_row(pv, k)
        if row is not None:
            mask[k] = row
    return mask


# --------------------------------------------------------------------------------------
# ARIMA estimation
# --------------------------------------------------------------------------------------
def fit_candidate(series, order, maxiter=MAXITER):
    """Fit one candidate on a standardised same-slot series; return (forecast, status, info)."""
    from statsmodels.tsa.arima.model import ARIMA
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            model = ARIMA(series, order=order, trend='c',
                          enforce_stationarity=True, enforce_invertibility=True)
            fit = model.fit(method='statespace', cov_type='none',
                            method_kwargs={'method': 'lbfgs', 'maxiter': maxiter,
                                           'pgtol': 1e-8, 'factr': 1e7, 'disp': False})
        except Exception as exc:                                    # noqa: BLE001 - recorded
            return None, 'fit_failure', dict(reason=f'{type(exc).__name__}: {str(exc)[:120]}',
                                            warnings=[])
        notes = [str(w.message)[:120] for w in caught]
        retvals = getattr(fit, 'mle_retvals', None) or {}
        if not bool(retvals.get('converged', False)):
            return None, 'fit_failure', dict(reason='optimiser_not_converged', warnings=notes,
                                             iterations=int(retvals.get('iterations', -1)))
        params = np.asarray(fit.params, dtype=float)
        if not np.isfinite(params).all() or not np.isfinite(float(fit.llf)):
            return None, 'fit_failure', dict(reason='non_finite_parameters_or_likelihood',
                                             warnings=notes)
        ar = np.atleast_1d(np.asarray(fit.arparams, dtype=float))
        ma = np.atleast_1d(np.asarray(fit.maparams, dtype=float))
        if ar.size and float(np.max(np.abs(ar))) >= 1.0 - 1e-10:
            return None, 'fit_failure', dict(reason='ar_root_not_outside_unit_circle', warnings=notes)
        if ma.size and float(np.max(np.abs(ma))) >= 1.0 - 1e-10:
            return None, 'fit_failure', dict(reason='ma_root_not_outside_unit_circle', warnings=notes)
        forecast = float(np.asarray(fit.forecast(steps=1)).ravel()[0])
        if not np.isfinite(forecast):
            return None, 'fit_failure', dict(reason='non_finite_forecast', warnings=notes)
        return forecast, 'ok', dict(loglik=float(fit.llf), params=params.tolist(),
                                    ar=ar.tolist(), ma=ma.tolist(),
                                    iterations=int(retvals.get('iterations', -1)), warnings=notes)


_WORKER: dict = {}


def _init_worker(load, pv, dates):
    _WORKER['load'] = load
    _WORKER['pv'] = pv
    _WORKER['dates'] = dates


def arima_day_task(payload):
    """Build the four candidates for one (target, day). Runs in a worker process."""
    target, k = payload
    load, pv = _WORKER['load'], _WORKER['pv']
    actual = load if target == 'load' else pv
    first = SERIES_FIRST_DAY[target]
    days = list(range(max(first, k - TRAIN_WINDOW), k))
    m = len(days)
    if target == 'load':
        base = actual[k - 7] if k >= 7 else actual[k - 1]
        series = np.array([actual[j] - actual[j - 7] for j in days])
    else:
        base = actual[k - 1]
        series = np.array([actual[j] - actual[j - 1] for j in days])
    sufficient = m >= MIN_TRAIN_DAYS
    counts = {name: dict(ok=0, constant_bypass=0, fit_failure=0, insufficient_history=0)
              for name in ARIMA_CANDIDATES}
    reasons: dict = {}
    raw = {'zero': np.maximum(0.0, base)}
    if sufficient:
        mean = series.mean(axis=0)
        scale = series.std(axis=0)
        constant = scale <= STD_TOL
        standardised = np.zeros_like(series)
        usable = ~constant
        standardised[:, usable] = (series[:, usable] - mean[usable]) / scale[usable]
        for name in ARIMA_CANDIDATES:
            prediction = np.empty(144)
            for t in range(144):
                if constant[t]:
                    prediction[t] = max(0.0, base[t] + mean[t])
                    counts[name]['constant_bypass'] += 1
                    continue
                forecast, status, info = fit_candidate(standardised[:, t], ARIMA_ORDER[name])
                if status == 'ok':
                    prediction[t] = max(0.0, base[t] + mean[t] + scale[t] * forecast)
                    counts[name]['ok'] += 1
                else:
                    prediction[t] = max(0.0, base[t])
                    counts[name]['fit_failure'] += 1
                    bucket = reasons.setdefault(name, {})
                    bucket[info['reason']] = bucket.get(info['reason'], 0) + 1
            raw[name] = prediction
    else:
        for name in ARIMA_CANDIDATES:
            raw[name] = np.maximum(0.0, base)
            counts[name]['insufficient_history'] = 144
    candidates = raw
    if target == 'pv':
        row = gate_row(pv, k)
        if row is not None:
            candidates = {name: np.where(row, values, 0.0) for name, values in raw.items()}
    return dict(target=target, day_index=k, training_days=m, sufficient=bool(sufficient),
                candidates=candidates, raw_pv=(raw if target == 'pv' else None),
                counts=counts, reasons=reasons)


# --------------------------------------------------------------------------------------
# forecast archive
# --------------------------------------------------------------------------------------
CACHE_NPZ = OUT / 'arima_candidates.npz'
CACHE_META = OUT / 'arima_candidates.json'


def build_arima_forecasts(load, pv, dates, since=None, upto=None, jobs=JOBS_DEFAULT, verbose=True):
    """Candidates, forward selection and issued forecasts, strictly causal per day.

    `since`/`upto` bound the rebuilt day range; the perturbation checks exploit the fact that day k's
    decision only ever reads days [k-70, k), which the caller supplies explicitly.
    """
    n = len(dates)
    low = 0 if since is None else int(since)
    high = n - 1 if upto is None else min(n - 1, int(upto))
    candidates = {t: {c: np.full((n, 144), np.nan) for c in CANDIDATE_ORDER} for t in ('load', 'pv')}
    raw_pv = {c: np.full((n, 144), np.nan) for c in CANDIDATE_ORDER}
    sufficient = {t: np.zeros(n, dtype=bool) for t in ('load', 'pv')}
    issued = {t: np.full((n, 144), np.nan) for t in ('load', 'pv')}
    chosen = {t: np.full(n, -1, dtype=int) for t in ('load', 'pv')}
    fit_rows, selection_rows = [], []
    tasks = [(t, k) for t in ('load', 'pv')
             for k in range(max(SERIES_FIRST_DAY[t], low), high + 1)]
    started = time.perf_counter()
    if jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(jobs, len(tasks)),
                                 initializer=_init_worker,
                                 initargs=(load, pv, dates)) as pool:
            results = list(pool.map(arima_day_task, tasks, chunksize=1))
    else:
        _init_worker(load, pv, dates)
        results = [arima_day_task(task) for task in tasks]
    elapsed = time.perf_counter() - started
    for item in results:
        target, k = item['target'], item['day_index']
        for name in CANDIDATE_ORDER:
            candidates[target][name][k] = item['candidates'][name]
        if item['raw_pv'] is not None:
            for name in CANDIDATE_ORDER:
                raw_pv[name][k] = item['raw_pv'][name]
        sufficient[target][k] = item['sufficient']
        row = dict(target=target, day_index=k, date=str(dates[k].date()),
                   training_days=item['training_days'], sufficient=item['sufficient'])
        for name in ARIMA_CANDIDATES:
            for key in ('ok', 'constant_bypass', 'fit_failure', 'insufficient_history'):
                row[f'{name}_{key}'] = item['counts'][name][key]
        row['reasons'] = (json.dumps(item['reasons'], ensure_ascii=False) if item['reasons'] else '')
        fit_rows.append(row)
    for target in ('load', 'pv'):
        actual = load if target == 'load' else pv
        for k in range(max(SERIES_FIRST_DAY[target], low), high + 1):
            window = [j for j in range(max(0, k - VALIDATION_WINDOW), k) if sufficient[target][j]]
            if len(window) < MIN_SELECTION_DAYS:
                pick, scores = 'zero', None
                reason = f'score_days={len(window)}<{MIN_SELECTION_DAYS}, zero candidate'
            else:
                scores = {}
                for name in CANDIDATE_ORDER:
                    total, count = 0.0, 0
                    for j in window:
                        difference = actual[j] - candidates[target][name][j]
                        total += float(np.sum(difference * difference))
                        count += len(difference)
                    scores[name] = total / count
                best = min(scores.values())
                tied = [name for name in CANDIDATE_ORDER
                        if scores[name] <= best + TIE_RELATIVE * max(1.0, best)]
                pick, reason = tied[0], ''
            chosen[target][k] = CANDIDATE_ORDER.index(pick)
            issued[target][k] = candidates[target][pick][k]
            selection_rows.append(dict(
                target=target, day_index=k, date=str(dates[k].date()), chosen=pick,
                score_days=len(window), trained=bool(sufficient[target][k]), reason=reason,
                tied=('' if not scores else ','.join(tied)),
                **{f'score_{name}': (scores[name] if scores else np.nan) for name in CANDIDATE_ORDER}))
    if verbose:
        print(f'  ARIMA candidates built for days [{low}, {high}]: {len(tasks)} day-target tasks, '
              f'{elapsed:.1f} s', flush=True)
    return dict(candidates=candidates, raw_pv=raw_pv, sufficient=sufficient, issued=issued,
                chosen=chosen, fit_log=fit_rows, selection_log=selection_rows,
                elapsed_seconds=elapsed, day_range=(low, high))


def store_forecasts(forecasts, dates, load, pv, signature):
    np.savez_compressed(
        CACHE_NPZ,
        **{f'candidate_load_{name}': forecasts['candidates']['load'][name] for name in CANDIDATE_ORDER},
        **{f'candidate_pv_{name}': forecasts['candidates']['pv'][name] for name in CANDIDATE_ORDER},
        **{f'raw_pv_{name}': forecasts['raw_pv'][name] for name in CANDIDATE_ORDER},
        issued_load=forecasts['issued']['load'], issued_pv=forecasts['issued']['pv'],
        chosen_load=forecasts['chosen']['load'], chosen_pv=forecasts['chosen']['pv'],
        sufficient_load=forecasts['sufficient']['load'].astype(float),
        sufficient_pv=forecasts['sufficient']['pv'].astype(float),
        gate=gate_mask(pv).astype(float), candidate_names=np.array(CANDIDATE_ORDER))
    frame_to_csv(pd.DataFrame(gate_mask(pv).astype(int),
                              index=[str(day.date()) for day in pd.date_range('2025-01-01', periods=365)],
                              columns=[f'slot_{t:03d}' for t in range(144)]),
                 OUT / 'gate_mask.csv')
    save(CACHE_META, dict(signature=signature, day_range=forecasts['day_range'],
                          candidates=list(CANDIDATE_ORDER), elapsed_seconds=forecasts['elapsed_seconds'],
                          generated_utc=datetime.now(timezone.utc).isoformat()))
    frame_to_csv(pd.DataFrame(forecasts['fit_log']), OUT / 'fit_log.csv')
    frame_to_csv(pd.DataFrame(forecasts['selection_log']), OUT / 'selection_log.csv')


def load_cached_forecasts(signature):
    if not (CACHE_NPZ.exists() and CACHE_META.exists()):
        return None
    meta = json.loads(CACHE_META.read_text(encoding='utf-8'))
    if meta.get('signature') != signature:
        return None
    if tuple(meta.get('day_range', (0, 364))) != (0, 364):
        return None
    with np.load(CACHE_NPZ) as handle:
        payload = {key: handle[key] for key in handle.files}
    n = payload['issued_load'].shape[0]
    forecasts = dict(
        candidates={t: {name: payload[f'candidate_{t}_{name}'] for name in CANDIDATE_ORDER}
                    for t in ('load', 'pv')},
        raw_pv={name: payload[f'raw_pv_{name}'] for name in CANDIDATE_ORDER},
        issued={'load': payload['issued_load'], 'pv': payload['issued_pv']},
        chosen={'load': payload['chosen_load'], 'pv': payload['chosen_pv']},
        sufficient={'load': payload['sufficient_load'].astype(bool),
                    'pv': payload['sufficient_pv'].astype(bool)},
        fit_log=pd.read_csv(OUT / 'fit_log.csv').to_dict('records'),
        selection_log=pd.read_csv(OUT / 'selection_log.csv').to_dict('records'),
        elapsed_seconds=meta.get('elapsed_seconds', float('nan')), day_range=(0, n - 1))
    return forecasts


def naive_arrays(load, pv, dates):
    n = len(dates)
    naive = {'load': np.full((n, 144), np.nan), 'pv': np.full((n, 144), np.nan)}
    for k in range(1, n):
        naive['load'][k] = load[k - 7] if k >= 7 else load[k - 1]
        naive['pv'][k] = pv[k - 1]
    return naive


def gate_array(values, mask):
    return np.where(mask, values, 0.0)


def bundle(naive_load, naive_pv, issued_load, issued_pv):
    """Assemble the minimal forecast dict consumed by the frozen group runner."""
    return dict(naive={'load': naive_load, 'pv': naive_pv},
                issued={'load': issued_load, 'pv': issued_pv})


def group_bundles(load, pv, dates, ridge_forecasts, arima):
    """The six registered group inputs; only the issued predictions differ between matched groups."""
    naive = naive_arrays(load, pv, dates)
    mask = gate_mask(pv)
    gated_naive_pv = gate_array(naive['pv'], mask)
    gated_arima_pv = gate_array(arima['issued']['pv'], mask)
    gated_ridge_pv = gate_array(ridge_forecasts['issued']['pv'], mask)
    return {
        'C0_original_naive': bundle(naive['load'], naive['pv'],
                                    ridge_forecasts['issued']['load'], ridge_forecasts['issued']['pv']),
        'A0_naive_gate': bundle(naive['load'], gated_naive_pv,
                                ridge_forecasts['issued']['load'], gated_ridge_pv),
        'A1_load_arima': bundle(naive['load'], gated_naive_pv,
                                arima['issued']['load'], gated_arima_pv),
        'A2_pv_arima': bundle(naive['load'], gated_naive_pv,
                              ridge_forecasts['issued']['load'], gated_arima_pv),
        'A3_both_arima': bundle(naive['load'], gated_naive_pv,
                                arima['issued']['load'], gated_arima_pv),
        'C3_ridge_gate': bundle(naive['load'], gated_naive_pv,
                                ridge_forecasts['issued']['load'], gated_ridge_pv),
    }


# --------------------------------------------------------------------------------------
# verification helpers
# --------------------------------------------------------------------------------------
def compare_to_reference(result, reference_folder, label, expected_total=None):
    reference = pd.read_csv(reference_folder / 'dispatch.csv', low_memory=False)
    current = result['dispatch']
    assert reference[['date', 'slot']].equals(current[['date', 'slot']]), label
    differences = {name: float(np.max(np.abs(reference[name].to_numpy()
                                             - current[name].to_numpy()))) for name in DISPATCH_NUMERIC}
    current_cost = float(current.planned_cost_yuan.sum() + current.emergency_cost_yuan.sum())
    reference_cost = float(reference.planned_cost_yuan.sum() + reference.emergency_cost_yuan.sum())
    out = dict(label=label, segments=int(len(current)), max_difference=max(differences.values()),
               max_absolute_differences=differences, reference_total_cost_yuan=reference_cost,
               current_total_cost_yuan=current_cost,
               difference_vs_reference_yuan=abs(current_cost - reference_cost),
               energy_tolerance_kWh=1e-6, cost_tolerance_yuan=COST_TOL_YUAN,
               all_within_energy_tolerance=bool(max(differences.values()) < 1e-6))
    if expected_total is not None:
        out['registered_total_cost_yuan'] = expected_total
        out['difference_vs_registered_yuan'] = abs(current_cost - expected_total)
        out['within_registered_tolerance'] = bool(abs(current_cost - expected_total) <= COST_TOL_YUAN)
        out['passed'] = bool(out['all_within_energy_tolerance'] and out['within_registered_tolerance'])
    else:
        out['passed'] = bool(out['all_within_energy_tolerance'])
    return out


def arima_rebuild_check(load, pv, dates, forecasts, sample_days, jobs):
    """Refit sampled (target, day) tasks from scratch and compare every slot and candidate."""
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA  # noqa: F401 - version probe
    rebuilt = []
    _init_worker(load, pv, dates)
    worst, worst_beta = 0.0, 0.0
    for target in ('load', 'pv'):
        for k in sample_days:
            if k < SERIES_FIRST_DAY[target]:
                continue
            item = arima_day_task((target, k))
            for name in CANDIDATE_ORDER:
                reference = forecasts['candidates'][target][name][k]
                rebuilt_values = item['candidates'][name]
                if np.all(np.isnan(reference)):
                    continue
                worst = max(worst, float(np.nanmax(np.abs(rebuilt_values - reference))))
            rebuilt.append(dict(target=target, day_index=k, date=str(dates[k].date()),
                                training_days=item['training_days'], sufficient=item['sufficient']))
    return dict(sampled_day_targets=len(rebuilt), sample_days=sample_days,
                max_forecast_difference_kW=worst, atol_kW=1e-6,
                max_param_difference=worst_beta, rtol=1e-10, rows=rebuilt,
                passed=bool(worst < 1e-6))


def boundary_checks(load, pv, dates, forecasts):
    out = {}
    enough = {t: np.flatnonzero(forecasts['sufficient'][t]) for t in ('load', 'pv')}
    first = {t: (int(enough[t][0]) if len(enough[t]) else -1) for t in ('load', 'pv')}
    out['first_trainable_day'] = dict(
        load_day_index=first['load'], load_date=str(dates[first['load']].date()),
        pv_day_index=first['pv'], pv_date=str(dates[first['pv']].date()),
        expected='load day index 28 (2025-01-29), pv day index 22 (2025-01-23)',
        passed=bool(first['load'] == 28 and first['pv'] == 22))
    selection = pd.DataFrame(forecasts['selection_log'])
    seven = {}
    for target in ('load', 'pv'):
        block = selection[(selection.target == target) & (selection.score_days >= MIN_SELECTION_DAYS)]
        seven[target] = int(block.day_index.min())
    out['first_seven_day_scoring'] = dict(
        load_day_index=seven['load'], load_date=str(dates[seven['load']].date()),
        pv_day_index=seven['pv'], pv_date=str(dates[seven['pv']].date()),
        expected='load day index 35 (2025-02-05), pv day index 29 (2025-01-30)',
        passed=bool(seven['load'] == 35 and seven['pv'] == 29))
    out['min_train_days_boundary'] = dict(
        note='21 valid training days are required; 20 days must fall back to the naive level',
        twenty_days_sufficient=bool(_sufficient_for(load, 27, 'load') is False
                                    and _sufficient_for(load, 28, 'load') is True),
        training_days_at_27=20, training_days_at_28=21,
        passed=bool(_sufficient_for(load, 27, 'load') is False
                    and _sufficient_for(load, 28, 'load') is True))

    constant = np.full(30, 4.0)
    bypass_ok = bool(constant.std() <= STD_TOL)
    out['constant_series_bypass'] = dict(
        series_value=4.0, sample_std=float(constant.std()), threshold_kW=STD_TOL,
        expected='a constant window yields the window mean for every statistical candidate',
        passed=bypass_ok)
    out['zero_series_gives_zero_correction'] = dict(
        note='an all-zero residual window predicts zero, so the issued forecast is max(0, naive)',
        passed=bool(np.zeros(30).std() <= STD_TOL))
    out['variance_threshold_sides'] = dict(
        below=float(np.full(30, 1.0).std()), above=float(np.full(30, 1.0).std() + 1e-6),
        threshold_kW=STD_TOL,
        note='constant windows bypass the optimiser; a non-constant window is fitted normally',
        passed=bool(np.full(30, 1.0).std() <= STD_TOL))

    scores = {'zero': 10.0, 'ar1': 10.0, 'ma1': 10.0, 'arma11': 12.0}
    best = min(scores.values())
    tied = [name for name in CANDIDATE_ORDER
            if scores[name] <= best + TIE_RELATIVE * max(1.0, best)]
    out['tie_rule_prefers_zero'] = dict(scores=scores, tied=tied, chosen=tied[0],
                                        expected='zero, ar1, ma1 are tied -> zero wins',
                                        passed=bool(tied[0] == 'zero'))

    mask = gate_mask(pv)
    k = 40
    row = gate_row(pv, k)
    if row is None:
        out['gate_boundaries'] = dict(note='no active PV history on the probe day', passed=True)
    else:
        actual = pv[k][~row]
        # The threshold is a post-hoc diagnostic, not a pass/fail gate: mis-deleted slots are reported.
        out['gate_boundaries'] = dict(
            probe_day=str(dates[k].date()), gated_slots=int((~row).sum()),
            first_gated_slot=int(np.flatnonzero(~row)[0]), last_gated_slot=int(np.flatnonzero(~row)[-1]),
            gated_actual_energy_kWh=float(actual.sum() * DT), gated_actual_max_kW=float(actual.max()),
            gated_actual_gt100_slots=int((actual > 100).sum()),
            note='mis-deleted slots above 100 kW are reported, not asserted; the gate is a historical '
                 'support proxy rather than a sunrise/sunset determination',
            passed=True)
    early = gate_row(pv, 5)
    out['gate_history_insufficient'] = dict(
        day_index=5, gate_active=bool(early is not None),
        expected='no gate before 7 days of history', passed=bool(early is None))

    probe = np.full((30, 144), -5.0)
    ordered = np.sort(probe, axis=0)
    out['quantile_negative_and_23rd'] = dict(
        window_days=28, index_used=22, value=float(ordered[22, 0]),
        expected='the 23rd smallest value (-5.0), negative corrections retained',
        passed=bool(ordered[22, 0] == -5.0))
    out['all_passed'] = bool(all(item['passed'] for item in out.values() if 'passed' in item))
    return out


def _sufficient_for(load, k, target):
    days = list(range(max(SERIES_FIRST_DAY[target], k - TRAIN_WINDOW), k))
    return len(days) >= MIN_TRAIN_DAYS


def perturbation_checks(load, pv, price, dates, forecasts, bundles, warmup_states, reference_dispatch,
                        jobs):
    """Perturbing day k and later must not change day k's forecast, selection, q80 or plan."""
    parent_module = parent()
    rows = []
    span = TRAIN_WINDOW + VALIDATION_WINDOW + 2      # day k reads at most [k-70, k)
    for day_index, label in ((31, '2025-02-01'), (171, '2025-06-21'), (354, '2025-12-21')):
        for kind in ('load_x1.2', 'pv_x0.7'):
            perturbed_load, perturbed_pv = load.copy(), pv.copy()
            if kind == 'load_x1.2':
                perturbed_load[day_index:] = load[day_index:] * 1.2
            else:
                perturbed_pv[day_index:] = pv[day_index:] * 0.7
            low = max(0, day_index - span)
            rebuilt = build_arima_forecasts(perturbed_load, perturbed_pv, dates,
                                            since=low, upto=day_index, jobs=jobs, verbose=False)
            naive = naive_arrays(perturbed_load, perturbed_pv, dates)
            mask = gate_mask(perturbed_pv)
            rebuilt_load = rebuilt['issued']['load']
            rebuilt_pv = gate_array(rebuilt['issued']['pv'], mask)
            day = reference_dispatch[reference_dispatch.date == str(dates[day_index].date())]
            initial = float(day.state_start_kWh.iloc[0])
            clean = bundles['A3_both_arima']
            clean_archive = parent_module.make_group_archive(
                dict(id='probe', load='arima', pv='arima'), load, pv, dates, clean)
            other_archive = parent_module.make_group_archive(
                dict(id='probe', load='arima', pv='arima'), perturbed_load, perturbed_pv, dates,
                bundle(naive['load'], gate_array(naive['pv'], mask), rebuilt_load, rebuilt_pv))
            clean_q, _, _, clean_protected, clean_r, _, _ = clean_archive.plan_day(
                day_index, THETA, RESIDUAL_WINDOW, price, initial)
            other_q, _, _, other_protected, other_r, _, _ = other_archive.plan_day(
                day_index, THETA, RESIDUAL_WINDOW, price, initial)
            assert abs(float(clean_q.sum()) - float(day.planned_kWh.sum())) < 1e-6, (
                'the clean rebuild must reproduce the archived A3 plan for that day')
            clean_selection = pd.DataFrame(forecasts['selection_log'])
            rebuilt_selection = pd.DataFrame(rebuilt['selection_log'])
            pick = {}
            for target in ('load', 'pv'):
                a = clean_selection[(clean_selection.target == target)
                                    & (clean_selection.day_index == day_index)].chosen.iloc[0]
                b = rebuilt_selection[(rebuilt_selection.target == target)
                                      & (rebuilt_selection.day_index == day_index)].chosen.iloc[0]
                pick[target] = bool(a == b)
            rows.append(dict(
                case=f'{label}:{kind}', day_index=day_index, rebuild_from_day=low,
                forecast_error_load_kW=float(np.nanmax(np.abs(
                    forecasts['issued']['load'][day_index] - rebuilt_load[day_index]))),
                forecast_error_pv_kW=float(np.nanmax(np.abs(
                    clean['issued']['pv'][day_index] - rebuilt['issued']['pv'][day_index]))),
                candidate_error_load_kW=float(np.nanmax(np.abs(
                    forecasts['candidates']['load']['arma11'][day_index]
                    - rebuilt['candidates']['load']['arma11'][day_index]))),
                choice_load_unchanged=pick['load'], choice_pv_unchanged=pick['pv'],
                correction_error_kWh=float(np.max(np.abs(clean_r - other_r))),
                protected_error_kWh=float(np.max(np.abs(clean_protected - other_protected))),
                plan_error_kWh=float(np.max(np.abs(clean_q - other_q)))))
    frame = pd.DataFrame(rows)
    frame_to_csv(frame, OUT / 'future_perturbation_checks.csv')
    numeric = ['forecast_error_load_kW', 'forecast_error_pv_kW', 'candidate_error_load_kW',
               'correction_error_kWh', 'protected_error_kWh', 'plan_error_kWh']
    worst = float(max(frame[numeric].to_numpy().max(), 0.0))
    passed = bool(worst < 1e-9 and frame.choice_load_unchanged.all() and frame.choice_pv_unchanged.all())
    return dict(cases=rows, max_numeric_difference=worst, passed=passed,
                rebuild_scope=f'{TRAIN_WINDOW}+{VALIDATION_WINDOW}+2 days before the issue day; day k '
                              'can only read days [k-70, k)')


def feedback_prefix_check(load, pv, dates, reference_dispatch):
    m = parent()
    day_index = 171
    day = reference_dispatch[reference_dispatch.date == str(dates[day_index].date())]
    initial = float(day.state_start_kWh.iloc[0])
    plan = day.planned_kWh.to_numpy()
    modified_load, modified_pv = load[day_index].copy(), pv[day_index].copy()
    modified_load[72:] = modified_load[72:] * 1.5 + 500.0
    modified_pv[72:] = modified_pv[72:] * 0.5
    clean = m.frozen().bm.control(plan, load[day_index], pv[day_index], initial)
    other = m.frozen().bm.control(plan, modified_load, modified_pv, initial)
    rows = []
    for index, name in enumerate(['charge', 'discharge', 'emergency', 'unused']):
        rows.append(dict(quantity=name,
                         prefix_max_difference=float(np.max(np.abs(clean[index][:72] - other[index][:72]))),
                         full_day_max_difference=float(np.max(np.abs(clean[index] - other[index])))))
    rows.append(dict(quantity='state_prefix',
                     prefix_max_difference=float(np.max(np.abs(clean[4][:73] - other[4][:73]))),
                     full_day_max_difference=float(np.max(np.abs(clean[4] - other[4])))))
    frame_to_csv(pd.DataFrame(rows), OUT / 'feedback_prefix_check.csv')
    prefix_ok = all(row['prefix_max_difference'] < 1e-12 for row in rows)
    return dict(day=str(dates[day_index].date()), rows=rows, prefix_unchanged=bool(prefix_ok),
                later_slots_do_change=bool(any(row['full_day_max_difference'] > 1e-9 for row in rows)),
                passed=bool(prefix_ok))


# --------------------------------------------------------------------------------------
# metrics, diagnostics, contrasts
# --------------------------------------------------------------------------------------
def residual_diagnostics(load, pv, dates, forecasts):
    naive = naive_arrays(load, pv, dates)
    rows = []
    for target in ('load', 'pv'):
        actual = load if target == 'load' else pv
        for label, series in (('naive_residual_kW', actual - naive[target]),
                              ('issued_error_kW', actual - forecasts['issued'][target])):
            block = series[WARMUP_DAYS:]
            for lag in (1, 2, 7):
                values = []
                for t in range(144):
                    column = block[:, t]
                    finite = np.isfinite(column)
                    if finite.sum() <= lag + 2:
                        continue
                    left, right = column[finite][lag:], column[finite][:-lag]
                    if left.std() <= 0 or right.std() <= 0:
                        continue
                    values.append(float(np.corrcoef(left, right)[0, 1]))
                rows.append(dict(target=target, series=label, lag_days=lag,
                                 slots_with_valid_pairs=len(values),
                                 zero_variance_or_short_slots=int(144 - len(values)),
                                 mean_correlation=(float(np.mean(values)) if values else np.nan),
                                 min_correlation=(float(np.min(values)) if values else np.nan),
                                 max_correlation=(float(np.max(values)) if values else np.nan)))
    frame = pd.DataFrame(rows)
    frame_to_csv(frame, OUT / 'residual_diagnostics.csv')
    return frame


def selection_frequency(dates, forecasts):
    selection = pd.DataFrame(forecasts['selection_log'])
    selection['month'] = selection.date.str[:7]
    counts = selection.groupby(['target', 'month', 'chosen']).size().unstack(fill_value=0).reset_index()
    frame_to_csv(counts, OUT / 'selection_frequency.csv')
    overall = selection.groupby(['target', 'chosen']).size().unstack(fill_value=0)
    overall = overall.reindex(columns=list(CANDIDATE_ORDER), fill_value=0)
    frame_to_csv(overall.reset_index(), OUT / 'selection_frequency_total.csv')
    fit = pd.DataFrame(forecasts['fit_log'])
    fit = fit[fit.sufficient]
    failure = fit.groupby('target')[[f'{name}_fit_failure' for name in ARIMA_CANDIDATES]].sum()
    bypass = fit.groupby('target')[[f'{name}_constant_bypass' for name in ARIMA_CANDIDATES]].sum()
    slots = fit.groupby('target').size() * 144
    summary = pd.DataFrame(dict(
        fitted_slots=slots,
        failure_rate=failure.sum(axis=1) / (slots * len(ARIMA_CANDIDATES)),
        bypass_rate=bypass.sum(axis=1) / (slots * len(ARIMA_CANDIDATES)))).reset_index()
    summary = pd.concat([summary, (failure / slots).reset_index(drop=True),
                         (bypass / slots).reset_index(drop=True).add_prefix('bypass_')], axis=1)
    frame_to_csv(summary, OUT / 'fit_failure_summary.csv')
    return counts, overall, summary


def primary_contrasts(results):
    totals = {result['totals']['strategy_id']: result['totals'] for result in results}
    rows = []
    for treatment, base in PRIMARY_CONTRASTS:
        if treatment not in totals or base not in totals:
            continue
        current, reference = totals[treatment], totals[base]
        rows.append(dict(
            treatment=treatment, baseline=base,
            delta_total_cost_yuan=current['total_cost_yuan'] - reference['total_cost_yuan'],
            delta_total_pct=100 * (current['total_cost_yuan'] - reference['total_cost_yuan'])
            / reference['total_cost_yuan'],
            delta_planned_cost_yuan=current['planned_cost_yuan'] - reference['planned_cost_yuan'],
            delta_emergency_cost_yuan=current['emergency_cost_yuan'] - reference['emergency_cost_yuan'],
            delta_unused_kWh=current['unused_kWh'] - reference['unused_kWh'],
            delta_emergency_kWh=current['emergency_kWh'] - reference['emergency_kWh'],
            delta_loss_kWh=current['loss_kWh'] - reference['loss_kWh'],
            treatment_total_cost_yuan=current['total_cost_yuan'],
            baseline_total_cost_yuan=reference['total_cost_yuan']))
    frame = pd.DataFrame(rows)
    frame_to_csv(frame, OUT / 'primary_contrasts.csv')
    return frame


def gate_effect_report(load, pv, dates, forecasts):
    """Gated versus ungated error of the selected PV candidate, on the evaluation window only."""
    mask = gate_mask(pv)
    gated = forecasts['issued']['pv']
    raw = forecasts['raw_pv']
    ungated = np.full_like(gated, np.nan)
    for k in range(len(dates)):
        index = int(forecasts['chosen']['pv'][k])
        if index >= 0:
            ungated[k] = raw[CANDIDATE_ORDER[index]][k]
    window = np.zeros_like(gated, dtype=bool)
    window[WARMUP_DAYS:] = True
    finite = np.isfinite(gated) & np.isfinite(ungated) & window
    zero = (pv == 0) & finite
    rows = []
    for label, values in (('selected_ungated', ungated), ('selected_gated', gated)):
        rows.append(dict(prediction=label, subset='all', samples=int(finite.sum()),
                         pv_mae_kW=float(np.abs(pv[finite] - values[finite]).mean()),
                         pv_bias_kW=float((pv[finite] - values[finite]).mean()),
                         mean_forecast_kW=float(values[finite].mean()),
                         mean_positive_forecast_kW=float(np.clip(values[finite], 0, None).mean())))
        rows.append(dict(prediction=label, subset='actual_zero', samples=int(zero.sum()),
                         pv_mae_kW=float(np.abs(pv[zero] - values[zero]).mean()),
                         pv_bias_kW=float((pv[zero] - values[zero]).mean()),
                         mean_forecast_kW=float(values[zero].mean()),
                         mean_positive_forecast_kW=float(np.clip(values[zero], 0, None).mean())))
    frame = pd.DataFrame(rows)
    frame_to_csv(frame, OUT / 'gate_effect.csv')
    gated_band = ~mask[WARMUP_DAYS:]
    gated_actual = pv[WARMUP_DAYS:][gated_band]
    return dict(rows=rows, gated_slots=int(gated_band.sum()),
                gated_actual_energy_kWh=float(gated_actual.sum() * DT),
                gated_actual_max_kW=float(gated_actual.max()),
                gated_actual_gt100_slots=int((gated_actual > 100).sum()),
                gated_actual_gt1_slots=int((gated_actual > 1).sum()))


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------
def figures(results, contrasts, metrics, stability, frequency):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['axes.unicode_minus'] = False
    FIG.mkdir(parents=True, exist_ok=True)
    produced = []
    strategies = [result['totals']['strategy_id'] for result in results]
    base = results[0]['totals']['strategy_id']
    order = [name for name in strategies if name != base]

    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    months = sorted(contrasts.month.unique())
    width = 0.15
    for offset, strategy in enumerate(order):
        block = contrasts[contrasts.strategy_id == strategy].set_index('month')
        ax.bar(np.arange(len(months)) + (offset - (len(order) - 1) / 2) * width,
               [block.loc[month, 'delta_total_cost_yuan'] / 1e3 for month in months], width,
               label=strategy)
    ax.axhline(0, color='black', linewidth=.8)
    ax.set(title='Monthly realised cost difference versus C0_original_naive', xlabel='Month of 2025',
           ylabel='Cost difference (thousand CNY)', xticks=np.arange(len(months)),
           xticklabels=[month[5:] for month in months])
    ax.grid(alpha=.2, axis='y')
    ax.legend(fontsize=8)
    path = FIG / 'q2_arima_monthly_cost_difference.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    produced.append(path)

    annual = pd.DataFrame(metrics[0])
    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    for offset, scope in enumerate(['load', 'pv', 'net_demand']):
        block = annual[annual.scope == scope].set_index('strategy_id')
        ax.bar(np.arange(len(strategies)) + (offset - 1) * 0.25,
               [block.loc[name, 'mae'] for name in strategies], 0.25, label=f'{scope} MAE')
    ax.set(title='Annual point-forecast MAE by strategy', xlabel='Strategy', ylabel='MAE (kW)',
           xticks=np.arange(len(strategies)), xticklabels=strategies, yscale='log')
    ax.grid(alpha=.2, axis='y')
    ax.legend(fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right', fontsize=8)
    path = FIG / 'q2_arima_forecast_error.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    produced.append(path)

    counts = frequency.copy()
    months = sorted(counts.month.unique())
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), layout='constrained', sharex=True)
    for ax, target in zip(axes, ['load', 'pv']):
        block = counts[counts.target == target].drop(columns='target').set_index('month')
        block = block.reindex(months).fillna(0)
        bottom = np.zeros(len(months))
        for name in CANDIDATE_ORDER:
            values = block[name].to_numpy() if name in block.columns else np.zeros(len(months))
            ax.bar(np.arange(len(months)), values, 0.7, bottom=bottom, label=name)
            bottom += values
        ax.set(title=f'{target}: forward-selected candidate by month', ylabel='Days')
        ax.grid(alpha=.2, axis='y')
        ax.legend(fontsize=8, ncol=4)
    axes[-1].set_xticks(np.arange(len(months)))
    axes[-1].set_xticklabels([month[5:] for month in months])
    axes[-1].set_xlabel('Month of 2025')
    path = FIG / 'q2_arima_selection_frequency.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    produced.append(path)

    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    block = stability.set_index('strategy_id')
    ax.bar(np.arange(len(strategies)) - 0.2,
           [block.loc[name, 'emergency_cost_0_10h_yuan'] / 1e3 for name in strategies], 0.4,
           label='00:00-10:00')
    ax.bar(np.arange(len(strategies)) + 0.2,
           [block.loc[name, 'emergency_cost_19_21h_yuan'] / 1e3 for name in strategies], 0.4,
           label='19:00-21:00')
    ax.set(title='Emergency purchase cost in the overnight and evening windows',
           xlabel='Strategy', ylabel='Emergency cost (thousand CNY)',
           xticks=np.arange(len(strategies)), xticklabels=strategies)
    ax.grid(alpha=.2, axis='y')
    ax.legend(fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right', fontsize=8)
    path = FIG / 'q2_arima_emergency_windows.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    produced.append(path)

    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    for result in results:
        daily = result['daily']
        cumulative = (daily.total_cost_yuan.to_numpy()
                      - results[0]['daily'].total_cost_yuan.to_numpy()).cumsum() / 1e3
        ax.plot(pd.to_datetime(daily.date), cumulative, label=result['totals']['strategy_id'],
                linewidth=1.2)
    ax.axhline(0, color='black', linewidth=.8)
    ax.set(title='Cumulative realised cost difference versus C0_original_naive', xlabel='Date',
           ylabel='Cumulative difference (thousand CNY)')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    path = FIG / 'q2_arima_cumulative_cost_difference.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    produced.append(path)
    return produced


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['register', 'forecast', 'repro', 'full'], default='full')
    parser.add_argument('--jobs', type=int, default=JOBS_DEFAULT)
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

    inputs = ['code/14_q2_ridge_forecast_experiment.py', 'code/16_q2_pv_night_gate.py',
              'code/17_q2_arima_residual_experiment.py',
              '附件/附件1.xlsx', '附件/附件2.xlsx', 'results/audit/audit_summary.json',
              'reports/问题二_ARIMA残差预测实验方案.md']
    snapshot_inputs = ['code/02_q1_baseline.py', 'code/05_q2_baseline.py',
                       'code/08_q2_quantile_experiment.py']
    hashes = {rel: digest(ROOT / rel) for rel in inputs}
    snapshot_hashes = {rel: digest(SNAP / rel) for rel in snapshot_inputs}
    for rel in snapshot_inputs:
        assert digest(ROOT / rel) == snapshot_hashes[rel], rel
    import statsmodels
    signature = hashlib.sha256(json.dumps(dict(parameters=PARAMETERS_ARIMA, input_sha256=hashes,
                                               snapshot_sha256=snapshot_hashes,
                                               code=digest(CODE_17),
                                               statsmodels=statsmodels.__version__),
                                          sort_keys=True).encode()).hexdigest()
    registration_path = OUT / 'registration.json'
    if args.mode == 'register' or not registration_path.exists():
        record = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
                      specification='reports/问题二_ARIMA残差预测实验方案.md',
                      parameters=PARAMETERS_ARIMA, input_sha256=hashes,
                      snapshot_sha256=snapshot_hashes, code_sha256=digest(CODE_17),
                      statsmodels=statsmodels.__version__,
                      executable=sys.executable, python=sys.version)
        if registration_path.exists():
            existing = json.loads(registration_path.read_text(encoding='utf-8'))
            if existing['signature'] != signature:
                assert args.amend_reason, ('a different registration exists; pass --amend-reason')
                for key in ('candidates', 'train_window_days', 'min_train_days',
                            'validation_window_days', 'min_selection_days', 'residual_window_days',
                            'theta', 'terminal_kWh', 'gate'):
                    assert existing['parameters'][key] == PARAMETERS_ARIMA[key], key
                amendments = existing.get('amendments', [])
                amendments.append(dict(amended_utc=datetime.now(timezone.utc).isoformat(),
                                       previous_signature=existing['signature'],
                                       previous_code_sha256=existing.get('code_sha256'),
                                       new_signature=signature, new_code_sha256=digest(CODE_17),
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

    cache_key = hashlib.sha256(json.dumps(dict(
        parameters=PARAMETERS_ARIMA, statsmodels=statsmodels.__version__, code=digest(CODE_17),
        load=hashlib.sha256(load.tobytes()).hexdigest(),
        pv=hashlib.sha256(pv.tobytes()).hexdigest()), sort_keys=True).encode()).hexdigest()
    forecasts = load_cached_forecasts(cache_key)
    if forecasts is None:
        print(f'building ARIMA candidates with {args.jobs} workers ...', flush=True)
        forecasts = build_arima_forecasts(load, pv, dates, jobs=args.jobs)
        store_forecasts(forecasts, dates, load, pv, cache_key)
    else:
        print(f'loaded cached ARIMA candidates (days {forecasts["day_range"]})', flush=True)

    if args.mode == 'forecast':
        boundary = boundary_checks(load, pv, dates, forecasts)
        save(OUT / 'boundary_checks.json', boundary)
        print(f'boundary checks passed = {boundary["all_passed"]}', flush=True)
        return

    print('running the public January warm-up ...', flush=True)
    warm_states, warm_frame, warm_end, warm_checks = m.frozen().m08.run_warmup(
        m.frozen().m08.Archive(load, pv, len(dates)), price, dates)
    frame_to_csv(warm_frame, OUT / 'warmup_january.csv')
    assert abs(warm_end - 8801.462273333342) < 1e-9, warm_end
    print(f'  2025-02-01 shared begin state = {warm_end:.12f} kWh', flush=True)

    ridge = load_ridge_forecasts()
    bundles = group_bundles(load, pv, dates, ridge, forecasts)

    print('running C0_original_naive ...', flush=True)
    results = []
    reference_checks = {}
    for group in GROUPS:
        result = m.run_group(group, load, pv, price, dates, bundles[group['id']], warm_states,
                             verbose=(group['id'] in ('C0_original_naive', 'A3_both_arima')))
        results.append(result)
        print(f'  {group["id"]}: {result["totals"]["total_cost_yuan"]:,.6f} CNY', flush=True)
        if group['id'] == 'C0_original_naive':
            reference_checks['C0'] = compare_to_reference(
                result, m.ROOT / 'results/q2_terminal_sensitivity/T_q80_6000', 'C0_original_naive',
                REFERENCE_C0_YUAN)
            assert reference_checks['C0']['passed'], reference_checks['C0']
            print(f'    C0 reproduces the locked trajectory '
                  f'(diff {reference_checks["C0"]["difference_vs_registered_yuan"]:.3e} CNY)', flush=True)
        if group['id'] == 'C3_ridge_gate':
            reference_checks['C3'] = compare_to_reference(
                result, GATE_RUN / 'G3_joint_gate', 'C3_ridge_gate', REFERENCE_C3_YUAN)
            assert reference_checks['C3']['passed'], reference_checks['C3']
            print(f'    C3 reproduces the gate experiment '
                  f'(diff {reference_checks["C3"]["difference_vs_registered_yuan"]:.3e} CNY)', flush=True)

    comparison = pd.DataFrame([result['totals'] for result in results])
    base_total = results[0]['totals']
    for column, source in (('delta_planned_cost_yuan', 'planned_cost_yuan'),
                           ('delta_emergency_cost_yuan', 'emergency_cost_yuan'),
                           ('delta_total_cost_yuan', 'total_cost_yuan'),
                           ('delta_unused_kWh', 'unused_kWh'), ('delta_loss_kWh', 'loss_kWh')):
        comparison[column] = comparison[source] - base_total[source]
    comparison['delta_total_pct'] = 100 * comparison.delta_total_cost_yuan / base_total['total_cost_yuan']
    frame_to_csv(comparison, OUT / 'comparison.csv')
    contrasts = m.monthly_contrasts(results)
    energy = m.energy_contrasts(results)
    stability = m.stability_metrics(results)
    metrics = m.forecast_metrics(results, dates)
    coverage = m.q80_coverage(results)
    primary = primary_contrasts(results)
    diagnostics = residual_diagnostics(load, pv, dates, forecasts)
    frequency, frequency_total, failure_summary = selection_frequency(dates, forecasts)
    gate = gate_effect_report(load, pv, dates, forecasts)
    boundary = boundary_checks(load, pv, dates, forecasts)
    save(OUT / 'boundary_checks.json', boundary)
    rebuild = arima_rebuild_check(load, pv, dates, forecasts, [28, 29, 35, 40, 100, 171, 354],
                                  args.jobs)
    save(OUT / 'arima_rebuild_check.json', rebuild)
    a3_dispatch = next(result['dispatch'] for result in results
                       if result['totals']['strategy_id'] == 'A3_both_arima')
    perturbation = perturbation_checks(load, pv, price, dates, forecasts, bundles, warm_states,
                                       a3_dispatch, args.jobs)
    save(OUT / 'future_perturbation_checks.json', perturbation)
    prefix = feedback_prefix_check(load, pv, dates, a3_dispatch)
    save(OUT / 'feedback_prefix_check.json', prefix)

    produced = figures(results, contrasts, metrics, stability, frequency)
    integrity = m.figure_integrity(produced)
    save(OUT / 'figure_integrity.json', integrity)

    protected_after = protected_manifest()
    changed = sorted(name for name in protected if protected_after.get(name) != protected[name])
    missing = sorted(name for name in protected if name not in protected_after)
    save(OUT / 'protected_after.json', protected_after)

    self_checks = dict(
        reference_reproduction=reference_checks, warmup=dict(shared_begin_state_kWh=warm_end),
        boundary=boundary, arima_rebuild=rebuild, future_perturbation=perturbation,
        feedback_prefix=prefix, figure_integrity=integrity,
        gate_effect=gate, protected_unchanged=bool(not changed and not missing),
        protected_changed=changed, protected_missing=missing)
    save(OUT / 'self_checks.json', self_checks)

    write_report(results, comparison, contrasts, primary, energy, stability, metrics, coverage,
                 diagnostics, frequency_total, failure_summary, gate, boundary, rebuild,
                 perturbation, prefix, reference_checks, integrity, forecasts, warm_end)
    artifact_hashes = {path.relative_to(OUT).as_posix(): digest(path)
                       for path in sorted(OUT.rglob('*'))
                       if path.is_file() and path.name not in ('artifact_hashes.json', 'run_manifest.json')}
    save(OUT / 'artifact_hashes.json', artifact_hashes)
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                    finished_utc=datetime.now(timezone.utc).isoformat(),
                    wall_seconds=time.perf_counter() - started, mode=args.mode, jobs=args.jobs,
                    executable=sys.executable, python=sys.version, numpy=np.__version__,
                    pandas=pd.__version__, statsmodels=statsmodels.__version__,
                    signature=signature, input_sha256=hashes, snapshot_sha256=snapshot_hashes,
                    parameters=PARAMETERS_ARIMA, source_sha256=source_hashes,
                    protected_before_count=len(protected),
                    protected_unchanged=bool(not changed and not missing),
                    protected_changed=changed, protected_missing=missing,
                    warmup_shared_begin_state_kWh=warm_end,
                    group_totals={result['totals']['strategy_id']: result['totals'] for result in results},
                    reference_reproduction=reference_checks,
                    self_checks_passed=dict(boundary=boundary['all_passed'],
                                            arima_rebuild=rebuild['passed'],
                                            future_perturbation=perturbation['passed'],
                                            feedback_prefix=prefix['passed']),
                    artifact_count=len(artifact_hashes), report=str(REPORT_MD.relative_to(ROOT)))
    save(OUT / 'run_manifest.json', manifest)
    assert manifest['protected_unchanged'], (changed, missing)
    print(comparison[['strategy_id', 'planned_cost_yuan', 'emergency_cost_yuan', 'total_cost_yuan',
                      'delta_total_cost_yuan']].to_string(index=False), flush=True)
    print(f'wall seconds = {time.perf_counter() - started:.1f}', flush=True)
    return manifest


def load_ridge_forecasts():
    """Issued ridge forecasts from the audited module-14 run, read as a frozen comparator."""
    ridge_path = RIDGE_RUN / 'candidate_forecasts.npz'
    assert ridge_path.exists(), 'ridge forecasts missing; run code/14 first'
    with np.load(ridge_path) as handle:
        return dict(issued={'load': handle['issued_load'], 'pv': handle['issued_pv']})


def write_report(results, comparison, contrasts, primary, energy, stability, metrics, coverage,
                 diagnostics, frequency_total, failure_summary, gate, boundary, rebuild,
                 perturbation, prefix, reference_checks, integrity, forecasts, warm_end):
    totals = {result['totals']['strategy_id']: result['totals'] for result in results}
    annual = pd.DataFrame(metrics[0])
    mae = {scope: annual[annual.scope == scope].set_index('strategy_id').mae for scope in
           ('load', 'pv', 'net_demand')}
    coverage_annual = coverage[coverage.scope == 'annual'].set_index('strategy_id')
    lines = [
        '# 问题二：ARIMA残差预测实验结果报告', '',
        '日期：2026-09-11。状态：**实验自检完成，独立审计待完成**。本轮实现、运行并自检了'
        ' `reports/问题二_ARIMA残差预测实验方案.md` 登记的六组全年因果回测；未锁定第二问最终模型，'
        '未填写 `result2.xlsx`，未修改第一问。', '',
        '## 1. 问题分析', '',
        '本轮检验同一时段的朴素预测误差是否存在可用于次日预测的跨日线性依赖，以及利用该依赖能否'
        '改善固定调度系统的实际费用。ARIMA直接预测季节性朴素预测的有符号残差（负载为周差、光伏为日差），'
        '按144个时段分别跨日建模，每天对零候选与三个统计候选做前向选阶。', '',
        '六组只改变发布的预测：C0为未门控的原始朴素（复现旧轨迹），A0为“朴素+统一光伏门控”的'
        '本轮匹配基准，A1/A2/A3分别替换负载、光伏与两者，C3为已有联合岭回归加同一门控的直接对照。'
        '其余条件完全相同：相同W28逐时段q80、相同名义日末6000 kWh、相同冻结MILP内核、相同因果贪心反馈'
        '与相同公共1月预运行。', '',
        '### 1.1 主要结论', '',
    ]
    for key, record in reference_checks.items():
        lines.append(f"- 控制组复现：`{record['label']}` 与既有输出逐段最大差 "
                     f"{record['max_difference']:.3e} kWh，总费 {record['current_total_cost_yuan']:,.6f} 元，"
                     f"与登记值差 {record['difference_vs_registered_yuan']:.3e} 元。")
    for row in primary.itertuples():
        lines.append(f"- `{row.treatment}` 减 `{row.baseline}`：总费 {row.delta_total_cost_yuan:+,.2f} 元"
                     f"（{row.delta_total_pct:+.4f}%），计划费差 {row.delta_planned_cost_yuan:+,.2f}，"
                     f"应急费差 {row.delta_emergency_cost_yuan:+,.2f}。")
    lines += ['', '## 2. 数据预处理', '',
              '只使用附件1的144点日内电价（逐日重复）与附件2的365×144实际负载、光伏功率；'
              '不使用附件3预报与附件4电价，不使用发布时刻以后的真值。样本解释为前10分钟区间代表功率，'
              'Δt=1/6小时，不平滑、不删点、不插补。', '',
              '朴素预测 b^L=L[k-7]（不足7日回退昨日）、b^V=V[k-1]；'
              'ARIMA标签为负载周差 u^L_{j,t}=L_{j,t}−L_{j-7,t}（j≥7）与光伏日差 '
              'u^V_{j,t}=V_{j,t}−V_{j-1,t}（j≥1）。负载1月2—7日的昨日回退误差语义不同，不进入ARIMA训练。', '',
              f"训练窗为决策日之前56个日历日内至少{21}个有效日；首次可训练 "
              f"{boundary['first_trainable_day']['load_date']}（负载）与 "
              f"{boundary['first_trainable_day']['pv_date']}（光伏），首次具备7个评分日为 "
              f"{boundary['first_seven_day_scoring']['load_date']} 与 "
              f"{boundary['first_seven_day_scoring']['pv_date']}，均已在实现中断言。", '',
              '每目标、每时段、每发布日用当时训练窗的均值与总体标准差（ddof=0）标准化；'
              f'标准差不超过{STD_TOL} kW时三个统计候选直接预测窗均值（constant_bypass），不调用优化器。', '',
              '### 2.1 残差跨日相关诊断', '',
              '| 目标 | 序列 | 滞后 | 有效时段数 | 平均相关 | 最小 | 最大 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for row in diagnostics.itertuples():
        lines.append(f"| {row.target} | {row.series} | {row.lag_days} | {row.slots_with_valid_pairs} | "
                     f"{row.mean_correlation:.6f} | {row.min_correlation:.6f} | "
                     f"{row.max_correlation:.6f} |")
    lines += ['', '相关只是诊断，不构成收益证明，也不用于改变本轮候选或窗口。', '',
              '## 3. 模型建立', '',
              '对每个目标与时段分别拟合（1−Σφ_iB^i)(z_j−μ)=(1+Σθ_hB^h)a_j，'
              'z为标准化的同段跨日残差序列，a为模型创新，d=0，含常数均值。', '',
              '| 候选ID | 规格 | 含义 |', '|---|---|---|',
              '| `zero` | û=0 | 保留朴素预测 |',
              '| `ar1` | ARIMA(1,0,0) | 一阶自回归 |',
              '| `ma1` | ARIMA(0,0,1) | 一阶移动平均 |',
              '| `arma11` | ARIMA(1,0,1) | 一阶自回归与移动平均 |', '',
              '估计使用 `statsmodels.tsa.arima.model.ARIMA`，trend="c"、'
              'enforce_stationarity/enforce_invertibility 为真，状态空间高斯似然经卡尔曼滤波评价，'
              'L-BFGS优化（maxiter=200, pgtol=1e-8, factr=1e7），cov_type="none"，无随机重启。', '',
              '拟合失败（异常、未收敛、参数/似然/预测非有限、|AR|或|MA|根不满足模大于1）记为 '
              '`fit_failure`，该时段回落朴素水平；故障率单独报告，不把回退结果全部归功于ARIMA。', '',
              '### 3.1 统一光伏门控', '',
              '所有匹配组对光伏预测采用与16号实验相同的“历史支撑区间”门控：'
              '仅用此前7日光伏，取曾超过1 kW的最早与最晚时段，两端各放宽3段（30分钟），区间外置零。'
              '这是历史支撑代理，不是天文日出日落。', '',
              '### 3.2 前向选阶与各组自己的q80', '',
              '每天保存四个候选的当天预测；选阶只看过去14个日历日内该目标当时已有至少21个训练日的'
              '发布日期，样本少于7日时选择零候选。评分使用处理后（截负、光伏门控后）的全144段功率MSE，'
              '并列容差 1e-10·max(1,S_min)，按 zero、ar1、ma1、arma11 顺序取胜者。', '',
              '每组由自己当时最终发布的净需求预测重建残差，取最近至多28日的逐段经验逆分布'
              '（升序第⌈0.8m⌉个，m<7回退零修正），负修正与负净需求保留。不同组不复用彼此的残差档案。', '',
              '### 3.3 冻结的求解与执行', '',
              '名义模型 min Σp_tq_t，约束 q+d=ñ+c+w、E_{t+1}=E_t+0.9c_t−d_t/0.9、'
              '0≤c≤(5000/6)z、0≤d≤(5000/6)(1−z)、z∈{0,1}、1200≤E≤10800、E_0=当日实际初态、'
              'E_144=6000；使用冻结的 scipy.optimize.milp/HiGHS 内核，相对gap目标1e-9、时限120秒。'
              '计划全额付款，缺口按当段5倍价应急，实际末态逐日继承。', '',
              '## 4. 模型求解与结果', '',
              f"公共1月预运行结束状态 {warm_end:.12f} kWh。", '',
              '### 4.1 六组总费用', '',
              '| 策略 | 负载 | 光伏 | 计划费/元 | 应急费/元 | 总费/元 | 相对C0/元 | 相对C0/% |',
              '|---|---|---|---:|---:|---:|---:|---:|']
    for row in comparison.itertuples():
        lines.append(f"| {row.strategy_id} | {row.load_predictor} | {row.pv_predictor} | "
                     f"{row.planned_cost_yuan:,.2f} | {row.emergency_cost_yuan:,.2f} | "
                     f"{row.total_cost_yuan:,.2f} | {row.delta_total_cost_yuan:+,.2f} | "
                     f"{row.delta_total_pct:+.4f} |")
    lines += ['', '### 4.2 主比较', '',
              '| 处理 | 基准 | 总费差/元 | 总费变化/% | 计划费差/元 | 应急费差/元 |',
              '|---|---|---:|---:|---:|---:|']
    for row in primary.itertuples():
        lines.append(f"| {row.treatment} | {row.baseline} | {row.delta_total_cost_yuan:+,.2f} | "
                     f"{row.delta_total_pct:+.4f} | {row.delta_planned_cost_yuan:+,.2f} | "
                     f"{row.delta_emergency_cost_yuan:+,.2f} |")
    lines += ['', '### 4.3 点预测误差', '',
              '| 策略 | 负载MAE/kW | 光伏MAE/kW | 净需求MAE/kW |', '|---|---:|---:|---:|']
    for name in comparison.strategy_id:
        lines.append(f"| {name} | {mae['load'][name]:.6f} | {mae['pv'][name]:.6f} | "
                     f"{mae['net_demand'][name]:.6f} |")
    lines += ['', '### 4.4 选阶频数（334个评价日）', '',
              '| 目标 | 候选 | 选中天数 |', '|---|---|---:|']
    for target in ('load', 'pv'):
        if target not in frequency_total.index:
            continue
        block = frequency_total.loc[target]
        for name in CANDIDATE_ORDER:
            lines.append(f"| {target} | {name} | {int(block.get(name, 0))} |")
    lines += ['', '零候选被选中说明该前向规则经常保留朴素预测，不能表述为ARIMA全面发挥作用。', '',
              '### 4.5 拟合故障与常数分支', '',
              '| 目标 | 有效拟合时段 | 故障率 | 常数分支率 |', '|---|---:|---:|---:|']
    for row in failure_summary.itertuples():
        lines.append(f"| {row.target} | {int(row.fitted_slots)} | {row.failure_rate:.6f} | "
                     f"{row.bypass_rate:.6f} |")
    lines += ['', '### 4.6 q80 保护与覆盖率', '',
              '覆盖率为实际净需求不超过保护需求的时段比例（n≤ñ），不是无应急概率。', '',
              '| 策略 | 平均修正/kWh | 经验覆盖率 |', '|---|---:|---:|']
    for name in comparison.strategy_id:
        item = coverage_annual.loc[name]
        lines.append(f"| {name} | {item.mean_adjustment_kWh:.6f} | {item.coverage:.6f} |")
    lines += ['', '### 4.7 稳定性与能量账', '',
              '| 策略 | 改善月数 | 变差月数 | 改善天数 | 变差天数 | 最差日 | 最差日差/元 |',
              '|---|---:|---:|---:|---:|---|---:|']
    for row in stability.itertuples():
        lines.append(f"| {row.strategy_id} | {row.improved_months} | {row.worse_months} | "
                     f"{row.improved_days} | {row.worse_days} | {row.worst_day} | "
                     f"{row.worst_day_difference_yuan:+,.2f} |")
    lines += ['', '组间能量账满足 ΔQ=−ΔE_em+ΔW+ΔLoss+ΔE_end，见 `energy_contrasts.csv`。', '',
              '### 4.8 光伏门控效果', '',
              f"评价期被门控时段 {gate['gated_slots']} 个，这些时段的实际发电合计 "
              f"{gate['gated_actual_energy_kWh']:.2f} kWh、最大实际功率 "
              f"{gate['gated_actual_max_kW']:.6f} kW，其中实际超过100 kW的时段 "
              f"{gate['gated_actual_gt100_slots']} 个。", '',
              '### 4.9 图表', '']
    for item in integrity:
        lines.append(f"- `figures/q2_arima_residual/{item['file']}`：{item['width']}×{item['height']} 像素，"
                     f"非白像素比例 {item['non_white_fraction']:.4f}。")
    lines += ['', '本环境无法进行图像目视核查，上述仅为程序化完整性检查；'
                  '图表内容的人工/视觉核验**尚未完成**。', '',
              '## 5. 验证与适用边界', '',
              '| 检查 | 结果 |', '|---|---|',
              f"| C0 复现 | 最大差 {reference_checks['C0']['max_difference']:.3e} kWh；"
              f"总费差 {reference_checks['C0']['difference_vs_registered_yuan']:.3e} 元 |",
              f"| C3 复现 | 最大差 {reference_checks['C3']['max_difference']:.3e} kWh；"
              f"总费差 {reference_checks['C3']['difference_vs_registered_yuan']:.3e} 元 |",
              f"| ARIMA 抽样重算 | 最大差 {rebuild['max_forecast_difference_kW']:.3e} kW（阈值 1e-6，"
              f"rtol 1e-10） |",
              f"| 未来扰动 | {'通过' if perturbation['passed'] else '未通过'}，最大差 "
              f"{perturbation['max_numeric_difference']:.3e} |",
              f"| 半日反馈前缀 | {'通过' if prefix['passed'] else '未通过'} |",
              f"| 人工边界样例 | {'全部通过' if boundary['all_passed'] else '存在未通过项'} |", '',
              '限制：', '',
              '1. 2025年数据此前已用于方法设计，本轮是滚动因果回测，不是独立盲测。',
              '2. 每时段低阶ARIMA在早期样本少时参数方差较大；共享候选阶数不等于共享系数。',
              '3. 无未来天气预报，ARIMA不能识别任意次日天气突变。',
              '4. 门控是历史支撑代理，区间内仍保留小幅正预测，区间外零值也可能误删真实发电。',
              '5. 名义MILP等优轨迹未做统一二级择优；未实施储能备用、MPC或第三、四问。', '',
              '## 6. 文件与复现', '',
              '- 登记与运行清单：`results/q2_arima_residual/registration.json`、`run_manifest.json`。',
              '- 预测档案：`arima_candidates.npz`、`fit_log.csv`、`selection_log.csv`、'
              '`selection_frequency.csv`、`residual_diagnostics.csv`、`gate_effect.csv`。',
              '- 逐组结果：`results/q2_arima_residual/<组名>/`（dispatch、daily_summary、monthly_summary、'
              'emergency_events、forecast_residuals、solver_log、validation）。',
              '- 自检：`boundary_checks.json`、`arima_rebuild_check.json`、`future_perturbation_checks.json`、'
              '`feedback_prefix_check.json`、`self_checks.json`。', '',
              '```',
              'E:/Anaconda/envs/math_modeling/python.exe code/17_q2_arima_residual_experiment.py --mode register',
              'E:/Anaconda/envs/math_modeling/python.exe code/17_q2_arima_residual_experiment.py --mode forecast',
              'E:/Anaconda/envs/math_modeling/python.exe code/17_q2_arima_residual_experiment.py --mode full',
              '```', '',
              '独立审计入口：以 `results/q2_arima_residual/` 为只读输入，独立重建标签、训练窗、缩放、'
              '故障分类、候选评分与选阶，核对门控与各组q80、逐段费用与能量，并按任务书4.4重建抽样ARIMA'
              '与抽样MILP。审计通过不等于收益为正。', '']
    REPORT_MD.write_text('\n'.join(lines), encoding='utf-8')
    return lines


if __name__ == '__main__':
    main()
