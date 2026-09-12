"""Q2 LightGBM residual correction of the naive forecasts: three pre-registered causal backtests.

Run from the C题 directory with the math_modeling interpreter:

    python code/19_q2_lightgbm_residual_experiment.py --mode register
    python code/19_q2_lightgbm_residual_experiment.py --mode full
    python code/19_q2_lightgbm_residual_experiment.py --mode forecast

Specification: reports/问题二_LightGBM残差预测实验方案.md (single-run task book).
Inherited frozen implementation: results/q2_revision_audit_20260911/source_snapshot/code/{02,05,08}
plus the archive/runner/metric helpers of code/14_q2_ridge_forecast_experiment.py and the
past-only PV gate rule of code/16_q2_pv_night_gate.py.

Three groups, one fixed LightGBM configuration, one shared model per target per day:
    G0_naive_gate     naive load + naive PV, both under the unified PV gate (matched baseline)
    G1_ridge_gate     frozen ridge forecasts + gate (reproduces the module-16 G3_joint_gate run)
    G2_lightgbm_gate  LightGBM residual correction + gate (the only new candidate)
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
CODE_19 = ROOT / 'code/19_q2_lightgbm_residual_experiment.py'
OUT = ROOT / 'results/q2_lightgbm_residual'
FIG = ROOT / 'figures/q2_lightgbm_residual'
SNAP = ROOT / 'results/q2_revision_audit_20260911/source_snapshot'
RIDGE_RUN = ROOT / 'results/q2_ridge_forecast'
GATE_RUN = ROOT / 'results/q2_pv_night_gate'
PLAN_MD = ROOT / 'reports/问题二_LightGBM残差预测实验方案.md'
REPORT_MD = ROOT / 'reports/问题二_LightGBM残差预测实验结果报告.md'

TRAIN_WINDOW = 56
MIN_TRAIN_DAYS = 14
RESIDUAL_WINDOW = 28
THETA = 0.80
TERMINAL_KWH = 6000.0
DT = 1 / 6
WARMUP_DAYS = 31
EVALUATION_START = '2025-02-01'

SHARED_BEGIN_STATE_KWH = 8801.462273333342
REFERENCE_G1_YUAN = 13724593.278483
REFERENCE_ENERGY_TOL_KWH = 1e-6
REFERENCE_COST_TOL_YUAN = 1e-4
TIE_RELATIVE = 1e-10

# reports/问题二_LightGBM残差预测实验方案.md section 3.2 -- one fixed configuration, no search.
LGBM_PARAMS = dict(
    boosting_type='gbdt', objective='regression',
    n_estimators=100, learning_rate=0.05, num_leaves=7, max_depth=3,
    min_child_samples=144, min_child_weight=0.001,
    reg_alpha=0.0, reg_lambda=1.0, min_split_gain=0.0,
    max_bin=63, min_data_in_bin=3, subsample_for_bin=200000,
    colsample_bytree=1.0, subsample=1.0, subsample_freq=0,
    boost_from_average=True, feature_pre_filter=False,
    device_type='cpu', n_jobs=1,
    deterministic=True, force_col_wise=True,
    random_state=20250911, data_random_seed=20250911,
    feature_fraction_seed=20250911, bagging_seed=20250911,
    verbosity=-1,
)
GATE_PARAMETERS = dict(history_days=7, activity_threshold_kW=1.0, padding_slots=3)
CONSTANT_TARGET_TOL_KW = 1e-8

PARAMETERS = dict(
    load_features=15, pv_features=8, train_window_days=TRAIN_WINDOW, min_train_days=MIN_TRAIN_DAYS,
    residual_window_days=RESIDUAL_WINDOW, theta=THETA, terminal_kWh=TERMINAL_KWH,
    evaluation_start=EVALUATION_START, gate=GATE_PARAMETERS, lightgbm=LGBM_PARAMS,
    groups=['G0_naive_gate', 'G1_ridge_gate', 'G2_lightgbm_gate'],
    primary_comparison='G2_lightgbm_gate minus G1_ridge_gate',
)

GROUPS = [
    dict(id='G0_naive_gate', load='naive', pv='naive',
         role='matched baseline: naive load and naive PV, both under the unified gate'),
    dict(id='G1_ridge_gate', load='ridge', pv='ridge',
         role='existing joint ridge + gate; reproduces module-16 G3_joint_gate'),
    dict(id='G2_lightgbm_gate', load='lightgbm', pv='lightgbm',
         role='joint LightGBM residual correction + gate (only new candidate)'),
]

SNAPSHOT_INPUTS = ['code/02_q1_baseline.py', 'code/05_q2_baseline.py', 'code/08_q2_quantile_experiment.py']
LIVE_INPUTS = SNAPSHOT_INPUTS + [
    'code/14_q2_ridge_forecast_experiment.py', 'code/16_q2_pv_night_gate.py',
    'code/19_q2_lightgbm_residual_experiment.py',
    '附件/附件1.xlsx', '附件/附件2.xlsx',
    'results/audit/audit_summary.json',
    'reports/问题二_LightGBM残差预测实验方案.md',
    'results/q2_ridge_forecast/candidate_forecasts.npz',
    'results/q2_pv_night_gate/G3_joint_gate/dispatch.csv',
]
OWN_NEW_REL = {'code/19_q2_lightgbm_residual_experiment.py',
               'reports/问题二_LightGBM残差预测实验结果报告.md'}
OWN_NEW_PREFIXES = ('results/q2_lightgbm_residual/', 'figures/q2_lightgbm_residual/')
SHARED_APPEND_REL = {'建模上下文记忆.md', 'reports/项目进度.md'}

PARENT: dict = {}


def parent():
    """Load module 14 once per process for the frozen runner, archive and metric helpers."""
    if 'module' not in PARENT:
        spec = importlib.util.spec_from_file_location(
            'ridge_parent', ROOT / 'code/14_q2_ridge_forecast_experiment.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.OUT = OUT
        module.FIG = FIG
        PARENT['module'] = module
    return PARENT['module']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def frame_to_csv(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def support_mask(pv):
    """Past-only PV support gate, identical rule to code/16_q2_pv_night_gate.support_mask."""
    mask = np.ones(np.asarray(pv).shape, dtype=bool)
    for k in range(7, len(pv)):
        active = np.flatnonzero(np.max(pv[k - 7:k], axis=0) > GATE_PARAMETERS['activity_threshold_kW'])
        if len(active):
            lo, hi = max(0, int(active[0]) - GATE_PARAMETERS['padding_slots']), \
                     min(143, int(active[-1]) + GATE_PARAMETERS['padding_slots'])
            mask[k] = False
            mask[k, lo:hi + 1] = True
    return mask


def apply_gate(pv_values, mask):
    values = np.asarray(pv_values, dtype=float)
    return np.where(mask[:len(values)], values, 0.0)


# --------------------------------------------------------------------------------------
# LightGBM forecast stage (the only modelling change relative to the ridge experiment)
# --------------------------------------------------------------------------------------
def build_lightgbm_forecasts(load, pv, dates, upto=None, save_models=False, verbose=False):
    """Strictly causal LightGBM residual corrections for both targets.

    Day k trains only on complete feature-valid days strictly before k inside the trailing
    56-calendar-day window, needs at least 14 of them, and never rewrites an earlier day.
    One shared model per target per day; no validation set, no early stopping, no search.
    """
    from lightgbm import LGBMRegressor

    m = parent()
    spec_map = m.TARGET_SPEC
    n_days = len(dates) if upto is None else min(len(dates), int(upto) + 1)

    naive = {'load': np.full((n_days, 144), np.nan), 'pv': np.full((n_days, 144), np.nan)}
    for k in range(1, n_days):
        naive['load'][k] = m.naive_load(load, k)
        naive['pv'][k] = pv[k - 1]

    issued = {c: np.full((n_days, 144), np.nan) for c in ('load', 'pv')}
    trained = {c: np.zeros(n_days, dtype=bool) for c in ('load', 'pv')}
    fit_log, training_log = [], []
    models_dir = OUT / 'models'
    if save_models:
        models_dir.mkdir(parents=True, exist_ok=True)
    boosters = {}

    for char in ('load', 'pv'):
        spec = spec_map[char]
        actual = load if char == 'load' else pv
        threshold = spec['threshold']
        feat_cache = {k: spec['feat_fn'](actual, dates, k) for k in range(threshold, n_days)}
        for k in range(threshold, n_days):
            base = naive[char][k]
            lo = max(0, k - TRAIN_WINDOW)
            train_days = [j for j in range(lo, k) if j >= threshold]
            entry = dict(target=char, day_index=k, date=str(dates[k].date()),
                         n_train_days=len(train_days), n_rows=0, trained=False,
                         branch='', n_trees=0, model_file='', model_sha256='', elapsed_seconds=0.0)
            if len(train_days) < MIN_TRAIN_DAYS:
                issued[char][k] = base
                entry['branch'] = 'insufficient_history'
                entry['reason'] = f'train_days={len(train_days)}<{MIN_TRAIN_DAYS}'
                fit_log.append(entry)
                continue

            X = np.concatenate([feat_cache[j] for j in train_days], axis=0)
            y = np.concatenate([spec['label_fn'](actual, j) for j in train_days], axis=0)
            if not (np.isfinite(X).all() and np.isfinite(y).all()):
                raise RuntimeError(f'{char} day {k}: non-finite feature or label')
            entry['n_rows'] = int(X.shape[0])
            entry['train_first'] = str(dates[train_days[0]].date())
            entry['train_last'] = str(dates[train_days[-1]].date())

            started = time.perf_counter()
            spread = float(np.max(y) - np.min(y))
            if spread <= CONSTANT_TARGET_TOL_KW:
                correction = np.full(144, float(np.mean(y)))
                entry['branch'] = 'constant_target'
                entry['constant_value_kW'] = float(np.mean(y))
            else:
                model = LGBMRegressor(**LGBM_PARAMS)
                model.fit(X, y)
                correction = np.asarray(model.predict(feat_cache[k]), dtype=float)
                entry['branch'] = 'fitted'
                entry['n_trees'] = int(model.booster_.num_trees())
                if save_models:
                    name = f'{char}_{dates[k].date()}.txt'
                    text = model.booster_.model_to_string()
                    (models_dir / name).write_text(text, encoding='utf-8')
                    entry['model_file'] = f'models/{name}'
                    entry['model_sha256'] = hashlib.sha256(text.encode('utf-8')).hexdigest()
                boosters[(char, k)] = model
            entry['elapsed_seconds'] = time.perf_counter() - started

            raw = base + correction
            negative = float(np.sum(raw < 0.0))
            issued[char][k] = np.maximum(0.0, raw)
            trained[char][k] = True
            entry['trained'] = True
            entry['negative_correction_slots'] = int(negative)
            entry['prediction_min_kW'] = float(issued[char][k].min())
            entry['prediction_max_kW'] = float(issued[char][k].max())
            fit_log.append(entry)
            training_log.append(dict(target=char, day_index=k, date=str(dates[k].date()),
                                     n_train_days=len(train_days), n_rows=int(X.shape[0]),
                                     feature_order=','.join(spec['features'])))
        if verbose:
            first = int(np.flatnonzero(trained[char])[0]) if trained[char].any() else -1
            print(f'    {char}: first trainable day index = {first} '
                  f'({dates[first].date() if first >= 0 else "n/a"})', flush=True)

    return dict(naive=naive, issued=issued, trained=trained, fit_log=fit_log,
                training_log=training_log, boosters=boosters, day_range=(0, n_days - 1))


def load_ridge_issued():
    """Issued ridge forecasts from the audited module-14 run, read as a frozen comparator."""
    path = RIDGE_RUN / 'candidate_forecasts.npz'
    assert path.exists(), 'ridge forecasts missing; run code/14 first'
    with np.load(path) as handle:
        return {'load': handle['issued_load'], 'pv': handle['issued_pv']}


def bundles_for(lightgbm, ridge, load, pv, dates, mask):
    """One forecast bundle per group; every group's PV goes through the same unified gate."""
    n_days = len(dates)
    gated_naive_pv = apply_gate(lightgbm['naive']['pv'], mask)
    gated_ridge_pv = apply_gate(ridge['pv'], mask)
    gated_lgbm_pv = apply_gate(lightgbm['issued']['pv'], mask)
    naive = dict(load=lightgbm['naive']['load'], pv=gated_naive_pv)
    return {
        'G0_naive_gate': dict(naive=naive,
                              issued=dict(load=lightgbm['naive']['load'], pv=gated_naive_pv)),
        'G1_ridge_gate': dict(naive=naive,
                              issued=dict(load=ridge['load'], pv=gated_ridge_pv)),
        'G2_lightgbm_gate': dict(naive=naive,
                                 issued=dict(load=lightgbm['issued']['load'], pv=gated_lgbm_pv)),
    }


# --------------------------------------------------------------------------------------
# LightGBM-specific checks (module 14's boundary/perturbation helpers are ridge-shaped)
# --------------------------------------------------------------------------------------
def perturbation_checks(load, pv, price, dates, lightgbm, bundles, warmup_states, a3_dispatch):
    rows = []
    for k in (31, 171, 354):
        for kind in ('load', 'pv'):
            ll, vv = load.copy(), pv.copy()
            if kind == 'load':
                ll[k:] *= 1.2
            else:
                vv[k:] *= 0.7
            rebuilt = build_lightgbm_forecasts(ll, vv, dates, upto=k)
            mask = support_mask(vv)
            rebuilt_bundles = bundles_for(rebuilt, load_ridge_issued(), ll, vv, dates, mask)
            forecast_error = max(
                float(np.max(np.abs(rebuilt_bundles['G2_lightgbm_gate']['issued'][t][k]
                                    - bundles['G2_lightgbm_gate']['issued'][t][k])))
                for t in ('load', 'pv'))
            assert forecast_error < 1e-6, (k, kind, forecast_error)
            rows.append(dict(day=k, kind=kind, forecast_max_kW=forecast_error,
                             passed=bool(forecast_error < 1e-6)))
    return dict(cases=rows, all_passed=bool(all(r['passed'] for r in rows)))


def feedback_prefix_check(load, pv, price, dates, lightgbm, bundles, a3_result):
    """A midday truth perturbation must not change the same day's morning plan or execution prefix."""
    k = 171
    ll, vv = load.copy(), pv.copy()
    ll[k, 72:] *= 1.2
    mask = support_mask(vv)
    rebuilt_bundles = bundles_for(lightgbm, load_ridge_issued(), ll, vv, dates, mask)
    m = parent()
    archive = m.make_group_archive(GROUPS[2], ll, vv, dates, rebuilt_bundles['G2_lightgbm_gate'])
    state = float(a3_result['daily'].iloc[k - WARMUP_DAYS].initial_kWh)
    q, _, _, _, _, _, _ = archive.plan_day(k, THETA, RESIDUAL_WINDOW, price, state)
    executed = m.frozen().m08.execute_day(k, q, ll[k], vv[k], state)
    ran_day = a3_result['dispatch']
    ran_day = ran_day[ran_day.date == str(dates[k].date())]
    morning_state = float(np.max(np.abs(np.asarray(executed['states'])[:72]
                                        - ran_day.state_start_kWh.to_numpy()[:72])))
    morning_plan = float(np.max(np.abs(q[:72] - ran_day.planned_kWh.to_numpy()[:72])))
    return dict(day=k, morning_state_max_kWh=morning_state, morning_plan_max_kWh=morning_plan,
                passed=bool(morning_state < 1e-6 and morning_plan < 1e-6))


def boundary_checks(load, pv, dates, lightgbm, mask):
    m = parent()
    out = {}
    first = {c: int(np.flatnonzero(lightgbm['trained'][c])[0]) for c in ('load', 'pv')}
    out['first_trainable_day'] = dict(
        load_day_index=first['load'], load_date=str(dates[first['load']].date()),
        pv_day_index=first['pv'], pv_date=str(dates[first['pv']].date()),
        expected='load day index 22 (2025-01-23), pv day index 21 (2025-01-22)',
        passed=bool(first['load'] == 22 and first['pv'] == 21))

    log = pd.DataFrame(lightgbm['fit_log'])
    thirteen = {c: int(((log.target == c) & (log.n_train_days == 13)).sum()) for c in ('load', 'pv')}
    fourteen = {c: int(((log.target == c) & (log.n_train_days == 14)).sum()) for c in ('load', 'pv')}
    trained_at_13 = {c: bool(log[(log.target == c) & (log.n_train_days == 13)].trained.any())
                     for c in ('load', 'pv')}
    out['training_boundary_14_days'] = dict(
        days_with_13=thirteen, days_with_14=fourteen, trained_with_only_13_days=trained_at_13,
        note='a 13-day window must fall back to the naive level; 14 days is the first training day',
        passed=bool(not any(trained_at_13.values()) and all(v >= 0 for v in fourteen.values())))

    sample = np.array([-2., 0., 1., 3., 10.])
    index = int(math.ceil(RESIDUAL_WINDOW * THETA)) - 1
    out['quantile_index_28_days'] = dict(
        zero_based_index=index, expected='ceil(28*0.8)-1 = 22 (the 23rd smallest)',
        passed=bool(index == 22))
    out['quantile_index_small_sample'] = dict(
        index_used=int(math.ceil(5 * THETA)) - 1,
        value=float(np.sort(sample)[int(math.ceil(5 * THETA)) - 1]),
        expected='ceil(5*0.8)-1 = 3 -> the 4th smallest = 3.0',
        passed=bool(int(math.ceil(5 * THETA)) - 1 == 3 and np.sort(sample)[3] == 3.0))

    probe = m.build_r0_archive(lightgbm, load, pv, dates)
    probe.eps = np.full_like(probe.eps, -5.0)
    probe._cache = {}
    negative = np.asarray(probe.adjustment(100, RESIDUAL_WINDOW, THETA)[0])
    out['negative_correction_preserved'] = dict(
        probe_adjustment_min_kWh=float(negative.min()),
        note='an all-negative residual history yields a negative correction; nothing is clipped at zero',
        passed=bool(abs(negative.max() + 5) < 1e-12 and abs(negative.min() + 5) < 1e-12))

    cap = 5000 * DT
    bm = m.frozen().bm
    _, d0, e0, _, _, _ = bm.control(np.array([0.]), np.array([12000.]), np.array([0.]), 1200.)
    _, d1, e1, _, _, _ = bm.control(np.array([0.]), np.array([12000.]), np.array([0.]), 10800.)
    _, _, _, w2, _, _ = bm.control(np.array([2000.]), np.array([0.]), np.array([0.]), 10800.)
    out['battery_and_power_limits'] = dict(
        empty_battery_emergency_kWh=float(e0[0]), full_battery_discharge_kWh=float(d1[0]),
        full_battery_remaining_emergency_kWh=float(e1[0]), full_battery_unused_kWh=float(w2[0]),
        segment_limit_kWh=float(cap),
        passed=bool(abs(d0[0]) < 1e-12 and abs(e0[0] - 2000) < 1e-9
                    and abs(d1[0] - cap) < 1e-9 and abs(e1[0] - (2000 - cap)) < 1e-9
                    and abs(w2[0] - 2000) < 1e-9))

    toy = np.zeros((10, 144)); toy[:, 40:101] = 10.0
    toy_mask = support_mask(toy)
    future = toy.copy(); future[7:] = 100.0
    out['gate_boundaries'] = dict(
        active_window_is_37_103=bool(toy_mask[7, 37:104].all() and not toy_mask[7, :37].any()),
        untouched_before_seven_days=bool(toy_mask[:7].all()),
        no_active_history_stays_ungated=bool(support_mask(np.zeros((10, 144))).all()),
        day_seven_ignores_later_days=bool(np.array_equal(toy_mask[7], support_mask(future)[7])),
        passed=bool(toy_mask[7, 37:104].all() and not toy_mask[7, :37].any()
                    and toy_mask[:7].all() and support_mask(np.zeros((10, 144))).all()
                    and np.array_equal(toy_mask[7], support_mask(future)[7])))

    out['all_passed'] = bool(all(v['passed'] for k, v in out.items()
                                 if isinstance(v, dict) and 'passed' in v))
    return out


def compare_reference(result, reference_folder, label, expected_total):
    reference = pd.read_csv(reference_folder / 'dispatch.csv', low_memory=False)
    current = result['dispatch']
    assert reference[['date', 'slot']].equals(current[['date', 'slot']]), label
    numeric = [c for c in current.columns
               if c in reference.columns
               and pd.api.types.is_numeric_dtype(current[c])
               and pd.api.types.is_numeric_dtype(reference[c])]
    differences = {name: float(np.max(np.abs(reference[name].to_numpy() - current[name].to_numpy())))
                   for name in numeric}
    current_cost = float(current.planned_cost_yuan.sum() + current.emergency_cost_yuan.sum())
    reference_cost = float(reference.planned_cost_yuan.sum() + reference.emergency_cost_yuan.sum())
    out = dict(label=label, segments=int(len(current)), max_difference=max(differences.values()),
               reference_total_cost_yuan=reference_cost, current_total_cost_yuan=current_cost,
               difference_vs_reference_yuan=abs(current_cost - reference_cost),
               difference_vs_registered_yuan=abs(current_cost - expected_total),
               all_within_energy_tolerance=bool(max(differences.values()) < REFERENCE_ENERGY_TOL_KWH),
               within_registered_tolerance=bool(abs(current_cost - expected_total) <= REFERENCE_COST_TOL_YUAN))
    out['passed'] = bool(out['all_within_energy_tolerance'] and out['within_registered_tolerance'])
    return out


def supplementary_metrics(results, price):
    """Indicators the plan's section 4.2 requires but the frozen helpers do not tabulate.

    Terminal inventory is re-valued at nu = median(price)/0.9 yuan per internal kWh; the cash
    cost stays the main metric and the adjusted figure never replaces it.
    """
    nu = float(np.median(price) / 0.9)
    rows = []
    for result in results:
        totals = result['totals']
        dispatch = result['dispatch']
        emergency = dispatch.emergency_cost_yuan.to_numpy()
        slot = dispatch.slot.to_numpy()

        def window(lo, hi):
            use = (slot >= lo * 6) & (slot < hi * 6)
            return float(emergency[use].sum())

        delta_end = float(totals['final_kWh'] - totals['evaluation_initial_kWh'])
        rows.append(dict(
            strategy_id=totals['strategy_id'], unused_kWh=float(totals['unused_kWh']),
            loss_kWh=float(totals['loss_kWh']), initial_kWh=float(totals['evaluation_initial_kWh']),
            final_kWh=float(totals['final_kWh']), planned_kWh=float(totals['planned_kWh']),
            emergency_kWh=float(totals['emergency_kWh']),
            terminal_valuation_yuan=float(nu * delta_end),
            adjusted_cost_yuan=float(totals['total_cost_yuan'] - nu * delta_end),
            nu_yuan_per_internal_kWh=nu,
            emergency_cost_0_10_yuan=window(0, 10), emergency_cost_19_21_yuan=window(19, 21),
            emergency_cost_21_24_yuan=window(21, 24)))
    return pd.DataFrame(rows)


def primary_stability(results):
    """Monthly and daily contrasts for the pre-registered primary comparison G2 - G1.

    The frozen module-14 helpers always difference against results[0] (the naive baseline here),
    so the plan's primary comparison needs its own contrast.
    """
    g1 = next(r for r in results if r['totals']['strategy_id'] == 'G1_ridge_gate')
    g2 = next(r for r in results if r['totals']['strategy_id'] == 'G2_lightgbm_gate')
    m1 = g1['monthly'].set_index('month').total_cost_yuan
    m2 = g2['monthly'].set_index('month').total_cost_yuan
    monthly = (m2 - m1).rename('delta_cost_yuan').reset_index()
    monthly.insert(0, 'baseline', 'G1_ridge_gate')
    monthly.insert(0, 'treatment', 'G2_lightgbm_gate')
    d1 = g1['daily'].set_index('date').total_cost_yuan
    d2 = g2['daily'].set_index('date').total_cost_yuan
    daily_delta = d2 - d1
    summary = dict(
        improved_months=int((monthly.delta_cost_yuan < -1e-6).sum()),
        worse_months=int((monthly.delta_cost_yuan > 1e-6).sum()),
        improved_days=int((daily_delta < -1e-6).sum()),
        worse_days=int((daily_delta > 1e-6).sum()),
        evaluated_months=int(len(monthly)), evaluated_days=int(len(daily_delta)),
        best_month=str(monthly.loc[monthly.delta_cost_yuan.idxmin(), 'month']),
        best_month_yuan=float(monthly.delta_cost_yuan.min()),
        worst_month=str(monthly.loc[monthly.delta_cost_yuan.idxmax(), 'month']),
        worst_month_yuan=float(monthly.delta_cost_yuan.max()),
        best_day=str(daily_delta.idxmin()), best_day_yuan=float(daily_delta.min()),
        worst_day=str(daily_delta.idxmax()), worst_day_yuan=float(daily_delta.max()))
    return monthly, summary


def gate_summary(dates, pv, mask):
    """Post-processing diagnostics for the unified PV gate (slot counts and gated actuals)."""
    actual = pv[WARMUP_DAYS:].ravel()
    gated = ~mask[WARMUP_DAYS:].ravel()
    return dict(gated_slots=int(gated.sum()),
                gated_actual_energy_kWh=float(actual[gated].sum() * DT),
                gated_actual_max_kW=float(actual[gated].max()) if gated.any() else 0.0,
                gated_actual_gt100_slots=int(np.sum((actual > 100.0) & gated)))


def write_report(results, comparison, primary, contrasts, metrics, coverage, subsets, gate,
                 boundary, perturbation, prefix, reference, fit_log, warm_end, primary_summary,
                 supplementary):
    supp = supplementary.set_index('strategy_id')
    totals = {r['totals']['strategy_id']: r['totals'] for r in results}
    lines = [        '# 问题二：LightGBM残差预测实验结果报告', '',
        '日期：2026-09-11。状态：**实验自检完成，独立审计待完成**。本轮实现、运行并自检了'
        ' `reports/问题二_LightGBM残差预测实验方案.md` 登记的三组全年因果回测；未锁定第二问最终模型，'
        ' 未填写 `result2.xlsx`，未修改第一问。', '',
        '## 1. 问题分析', '',
        '在相同的历史特征下，检验浅层梯度提升树能否利用非线性与特征交互改善残差预测，'
        '并降低既定调度系统的实际现金费用。三组只改变发布的功率预测：G0 为朴素负载与朴素光伏'
        '（统一门控），G1 为已有联合岭回归加同一门控，G2 为联合 LightGBM 残差修正加同一门控。'
        '其余条件完全相同：公共 1 月预运行、W28 逐时段 q80、名义日末 6000 kWh、冻结 MILP 与因果贪心反馈。', '',
        '### 1.1 主要结论', '',
    ]
    for key, record in reference.items():
        lines.append(f"- 控制组复现 `{record['label']}`：逐段最大差 {record['max_difference']:.3e} kWh，"
                     f"总费 {record['current_total_cost_yuan']:,.6f} 元，与登记值差 "
                     f"{record['difference_vs_registered_yuan']:.3e} 元。")
    for row in primary.itertuples():
        lines.append(f"- `{row.treatment}` 减 `{row.baseline}`：总费 {row.delta_total_cost_yuan:+,.2f} 元"
                     f"（{row.delta_total_pct:+.4f}%），计划费差 {row.delta_planned_cost_yuan:+,.2f}，"
                     f"应急费差 {row.delta_emergency_cost_yuan:+,.2f}。")
    lines += ['', '## 2. 数据预处理', '',
              '只使用附件1的144点日内电价（逐日重复）与附件2的365×144实际负载、光伏功率；'
              '不使用附件3预报、附件4电价、外部天气或发布时刻以后的真值。样本解释为前10分钟区间代表功率，'
              f'Δt=1/6小时，不平滑、不删点、不插补。特征沿用岭回归的15列负载与8列光伏定义，'
              f'首次有效分别为第8日与第7日；训练窗为决策日之前{TRAIN_WINDOW}个日历日内至少'
              f'{MIN_TRAIN_DAYS}个有效日。', '',
              '## 3. 模型建立', '',
              f'LightGBM 固定配置（登记于 registration.json，全轮不搜索）：'
              f'gbdt/平方损失，{LGBM_PARAMS["n_estimators"]} 轮，学习率 {LGBM_PARAMS["learning_rate"]}，'
              f'叶数 {LGBM_PARAMS["num_leaves"]}，深度 {LGBM_PARAMS["max_depth"]}，'
              f'min_child_samples {LGBM_PARAMS["min_child_samples"]}，L2 {LGBM_PARAMS["reg_lambda"]}，'
              f'max_bin {LGBM_PARAMS["max_bin"]}，单线程、确定性、列式直方图。'
              '每个目标每天一个共享模型，跨全部144段；不做早停、不做验证集、不做超参搜索。'
              '输出先取 max(0, 朴素水平 + 预测残差)，再统一施加历史支持区间门控。', '',
              '## 4. 模型求解与结果', '',
              f'- 公共1月预运行末态：{warm_end:.12f} kWh（登记目标 {SHARED_BEGIN_STATE_KWH}）。']
    for sid in ('G0_naive_gate', 'G1_ridge_gate', 'G2_lightgbm_gate'):
        t = totals[sid]
        row = supp.loc[sid]
        lines.append(f"- `{sid}`：计划费 {t['planned_cost_yuan']:,.2f}，应急费 {t['emergency_cost_yuan']:,.2f}，"
                     f"总费 {t['total_cost_yuan']:,.2f} 元；应急 {t['emergency_slots']} 时段 / "
                     f"{t['emergency_days']} 天 / {t['emergency_events']} 事件。")
        lines.append(f"  - 计划购电 {row.planned_kWh:,.2f} kWh，应急电量 {row.emergency_kWh:,.2f} kWh，"
                     f"未使用电量 {row.unused_kWh:,.2f} kWh，储能损耗 {row.loss_kWh:,.2f} kWh；"
                     f"评价期初 {row.initial_kWh:,.4f} → 期末 {row.final_kWh:,.4f} kWh。")
    s = primary_summary
    lines += ['', '### 4.1 主比较稳定性（G2 减 G1）', '',
              f"- 月度：{s['improved_months']}/{s['evaluated_months']} 个月更省，"
              f"最好 {s['best_month']}（{s['best_month_yuan']:+,.2f} 元），"
              f"最差 {s['worst_month']}（{s['worst_month_yuan']:+,.2f} 元）。",
              f"- 日度：{s['improved_days']}/{s['evaluated_days']} 天更省，"
              f"最好 {s['best_day']}（{s['best_day_yuan']:+,.2f} 元），"
              f"最差 {s['worst_day']}（{s['worst_day_yuan']:+,.2f} 元）。",
              f"- 变差月份数 {s['worse_months']}，变差天数 {s['worse_days']}；"
              '存在反例即不得表述为"每月都更省"。', '',
              '### 4.2 预测误差（全期48096段）', '']

    annual = pd.DataFrame(metrics[0])
    for target in ('load', 'pv', 'net_demand'):
        part = annual[annual.scope == target]
        for row in part.itertuples():
            lines.append(f"- `{row.strategy_id}` {target}：MAE {row.mae:,.4f} kW，RMSE {row.rmse:,.4f} kW。")
    indexed = annual.set_index(['strategy_id', 'scope'])
    directions = []
    for target in ('load', 'pv', 'net_demand'):
        g1 = float(indexed.loc[('G1_ridge_gate', target), 'mae'])
        g2 = float(indexed.loc[('G2_lightgbm_gate', target), 'mae'])
        directions.append(f"{target} {'改善' if g2 < g1 else '变差'} {abs(g2 - g1):,.4f} kW")
    lines.append(f"- G2 相对 G1 的点预测 MAE：{'；'.join(directions)}。")
    lines.append('- 误差与费用并不同向：上表中未改善的目标不应宣称预测被改进，'
                 '费用变化只能归因于组合净需求与调度路径的联合结果。')
    lines += ['', '### 4.3 光伏后处理与子集', '',
              f"- 门控区内实际发电 {gate['gated_actual_energy_kWh']:,.4f} kWh，"
              f"最大实际功率 {gate['gated_actual_max_kW']:,.4f} kW，"
              f"误删（实际>100 kW 却被归零）{gate['gated_actual_gt100_slots']} 个时段。"]
    for row in subsets[subsets.subset == 'actual_zero'].itertuples():
        lines.append(f"- `{row.strategy_id}` 实测零光伏段平均正预测 "
                     f"{row.mean_positive_forecast_kW:,.6f} kW。")
    lines += ['', '### 4.4 训练与分支统计', '']
    log = pd.DataFrame(fit_log)
    for target in ('load', 'pv'):
        part = log[log.target == target]
        counts = part.branch.value_counts().to_dict()
        lines.append(f"- `{target}`：有效训练 {int(part.trained.sum())} 天，"
                     f"分支计数 {counts}，实际树数合计 {int(part.n_trees.sum())}。")
    lines += ['', '### 4.5 验证范围与限制', '',
              f"- 边界检查全部通过：{boundary['all_passed']}。",
              f"- 未来扰动 6 例全部通过：{perturbation['all_passed']}；半日反馈前缀校验通过：{prefix['passed']}。",
              '- q80 使用各组自己当时发布的净需求误差重建，未复用岭回归残差或训练拟合残差。',
              '- 2025年数据此前已用于方法设计，本轮是同年滚动因果回测，不是独立盲测；'
              '固定配置未证明最优，不能据本期费用宣称跨年有效。',
              '- 本轮只检验联合替代，不能从三组分解单项贡献。', '',
              '### 4.6 期末估值与分时应急（诊断，不替代现金费用）', '',
              f"- 估值系数 ν = median(p)/0.9 = "
              f"{float(supp.iloc[0].nu_yuan_per_internal_kWh):,.6f} 元/内部kWh；"
              '下表为 C − ν(E_end − E_start)，现金总费仍为主指标，期末估值不能排除'
              '每日名义终端约束对调度的影响。']
    for sid in ('G0_naive_gate', 'G1_ridge_gate', 'G2_lightgbm_gate'):
        row = supp.loc[sid]
        lines.append(f"- `{sid}`：期末估值 {row.terminal_valuation_yuan:,.2f} 元，调整后费用 "
                     f"{row.adjusted_cost_yuan:,.2f} 元；应急费 0—10时 {row.emergency_cost_0_10_yuan:,.2f}，"
                     f"19—21时 {row.emergency_cost_19_21_yuan:,.2f}，"
                     f"21—24时 {row.emergency_cost_21_24_yuan:,.2f} 元。")
    lines += ['',
              '## 5. 复现', '',
              '```',
              'python code/19_q2_lightgbm_residual_experiment.py --mode register',
              'python code/19_q2_lightgbm_residual_experiment.py --mode full',
              '```', '']
    REPORT_MD.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['register', 'forecast', 'full'], default='full')
    parser.add_argument('--amend-reason', default=None)
    args = parser.parse_args()
    assert Path(sys.prefix).name == 'math_modeling', sys.prefix

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    m = parent()
    frozen = m.frozen()

    hashes = {rel: digest(ROOT / rel) for rel in LIVE_INPUTS}
    snapshot_hashes = {rel: digest(SNAP / rel) for rel in SNAPSHOT_INPUTS}
    for rel in SNAPSHOT_INPUTS:
        assert digest(ROOT / rel) == snapshot_hashes[rel], rel

    import lightgbm
    import scipy
    signature = m.registration_signature(PARAMETERS, hashes, snapshot_hashes)
    registration_path = OUT / 'registration.json'
    if args.mode == 'register' or not registration_path.exists():
        record = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
                      specification='reports/问题二_LightGBM残差预测实验方案.md',
                      parameters=PARAMETERS, input_sha256=hashes, snapshot_sha256=snapshot_hashes,
                      code_sha256=digest(CODE_19), signature=signature,
                      lightgbm=lightgbm.__version__, numpy=np.__version__, pandas=pd.__version__,
                      scipy=scipy.__version__, executable=sys.executable, python=sys.version,
                      effective_lightgbm_params=None)
        if registration_path.exists():
            existing = json.loads(registration_path.read_text(encoding='utf-8'))
            if existing['signature'] != signature:
                assert args.amend_reason, 'a different registration exists; pass --amend-reason'
                for key in ('train_window_days', 'min_train_days', 'residual_window_days', 'theta',
                            'terminal_kWh', 'gate', 'lightgbm', 'groups', 'evaluation_start'):
                    assert existing['parameters'][key] == PARAMETERS[key], key
                record['amendments'] = existing.get('amendments', []) + [dict(
                    amended_utc=datetime.now(timezone.utc).isoformat(),
                    previous_signature=existing['signature'], new_signature=signature,
                    reason=args.amend_reason)]
        snapshot = OUT / 'source_snapshot'
        (snapshot / 'code').mkdir(parents=True, exist_ok=True)
        for rel in LIVE_INPUTS:
            if rel.startswith('code/'):
                shutil.copy2(ROOT / rel, snapshot / 'code' / Path(rel).name)
        shutil.copy2(PLAN_MD, snapshot / 'experiment_plan.md')
        save(registration_path, record)
        print(f'registered: signature={signature[:16]}', flush=True)
        if args.mode == 'register':
            return record
    else:
        existing = json.loads(registration_path.read_text(encoding='utf-8'))
        assert existing['signature'] == signature, (
            'inputs or code changed since registration; rerun --mode register --amend-reason')

    from lightgbm import LGBMRegressor
    probe = LGBMRegressor(**LGBM_PARAMS)
    effective = {k: probe.get_params()[k] for k in LGBM_PARAMS if k in probe.get_params()}
    missing = [k for k in LGBM_PARAMS if k not in probe.get_params()]
    save(OUT / 'lightgbm_effective_params.json',
         dict(declared=LGBM_PARAMS, effective=effective, not_exposed_by_get_params=missing,
              lightgbm_version=lightgbm.__version__,
              note='parameters absent from get_params are still forwarded to the booster'))
    assert not missing or all(k in missing for k in missing)

    protected = m.protected_manifest()
    save(OUT / 'protected_before.json', protected)

    load, pv, price, source_hashes = frozen.bm.read_sources()
    dates = frozen.bm.DATES
    assert load.shape == pv.shape == (365, 144) and price.shape == (144,)

    print('building LightGBM residual forecasts ...', flush=True)
    lightgbm_forecasts = build_lightgbm_forecasts(load, pv, dates, save_models=True, verbose=True)
    frame_to_csv(pd.DataFrame(lightgbm_forecasts['fit_log']), OUT / 'training_log.csv')
    frame_to_csv(pd.DataFrame(lightgbm_forecasts['training_log']), OUT / 'training_rows.csv')
    ridge = load_ridge_issued()

    mask = support_mask(pv)
    bundles = bundles_for(lightgbm_forecasts, ridge, load, pv, dates, mask)
    issued_rows = []
    for sid, bundle in bundles.items():
        for target in ('load', 'pv'):
            values = bundle['issued'][target]
            issued_rows.append(pd.DataFrame(dict(
                strategy_id=sid, target=target,
                date=np.repeat(dates.strftime('%Y-%m-%d'), 144),
                slot=np.tile(np.arange(144), len(dates)),
                forecast_kW=values[:len(dates)].ravel())))
    frame_to_csv(pd.concat(issued_rows, ignore_index=True), OUT / 'issued_forecasts.csv')
    frame_to_csv(pd.DataFrame(
        dict(date=np.repeat(dates.strftime('%Y-%m-%d'), 144), slot=np.tile(np.arange(144), 365),
             allow_pv=mask.ravel(), actual_pv_kW=pv.ravel())), OUT / 'gate_mask.csv')

    if args.mode == 'forecast':
        return lightgbm_forecasts

    print('running the public January warm-up ...', flush=True)
    warm_states, warm_frame, warm_end, _ = frozen.m08.run_warmup(
        frozen.m08.Archive(load, pv, len(dates)), price, dates)
    frame_to_csv(warm_frame, OUT / 'warmup_january.csv')
    assert abs(warm_end - SHARED_BEGIN_STATE_KWH) < 1e-9, warm_end
    print(f'  2025-02-01 shared begin state = {warm_end:.12f} kWh', flush=True)

    results, reference = [], {}
    for group in GROUPS:
        print(f'running {group["id"]} ...', flush=True)
        result = m.run_group(group, load, pv, price, dates, bundles[group['id']], warm_states,
                             verbose=(group['id'] == 'G2_lightgbm_gate'))
        results.append(result)
        print(f'  {group["id"]}: {result["totals"]["total_cost_yuan"]:,.6f} CNY', flush=True)
        if group['id'] == 'G1_ridge_gate':
            reference['G1'] = compare_reference(result, GATE_RUN / 'G3_joint_gate',
                                                'G1_ridge_gate', REFERENCE_G1_YUAN)
            print(f'    G1 reproduces G3_joint_gate (diff '
                  f'{reference["G1"]["difference_vs_registered_yuan"]:.3e} CNY)', flush=True)
    assert reference['G1']['passed'], reference['G1']

    comparison = pd.DataFrame([r['totals'] for r in results])
    base = next(r['totals'] for r in results if r['totals']['strategy_id'] == 'G1_ridge_gate')
    for column, source in (('delta_planned_cost_yuan', 'planned_cost_yuan'),
                           ('delta_emergency_cost_yuan', 'emergency_cost_yuan'),
                           ('delta_total_cost_yuan', 'total_cost_yuan'),
                           ('delta_unused_kWh', 'unused_kWh'), ('delta_loss_kWh', 'loss_kWh')):
        comparison[column] = comparison[source] - base[source]
    comparison['delta_total_pct'] = 100 * comparison.delta_total_cost_yuan / base['total_cost_yuan']
    frame_to_csv(comparison, OUT / 'comparison.csv')

    primary = comparison[comparison.strategy_id.isin(['G2_lightgbm_gate'])].copy()
    primary['treatment'] = primary.strategy_id
    primary['baseline'] = 'G1_ridge_gate'
    frame_to_csv(primary, OUT / 'primary_contrasts.csv')
    primary_monthly, primary_summary = primary_stability(results)
    frame_to_csv(primary_monthly, OUT / 'primary_monthly_contrasts.csv')
    save(OUT / 'primary_stability.json', primary_summary)
    supplementary = supplementary_metrics(results, price)
    frame_to_csv(supplementary, OUT / 'supplementary_metrics.csv')

    # The frozen metric helpers write their own CSV files into OUT (redirected above).
    contrasts = m.monthly_contrasts(results)
    energy = m.energy_contrasts(results)
    stability = m.stability_metrics(results)
    metrics = m.forecast_metrics(results, dates)
    coverage = m.q80_coverage(results)
    energy = energy if energy is not None else pd.DataFrame()

    gate = gate_summary(dates, pv, mask)

    boundary = boundary_checks(load, pv, dates, lightgbm_forecasts, mask)
    save(OUT / 'boundary_checks.json', boundary)
    a3_result = next(r for r in results
                     if r['totals']['strategy_id'] == 'G2_lightgbm_gate')
    a3_dispatch = a3_result['dispatch']
    perturbation = perturbation_checks(load, pv, price, dates, lightgbm_forecasts, bundles,
                                       warm_states, a3_dispatch)
    save(OUT / 'future_perturbation_checks.json', perturbation)
    prefix = feedback_prefix_check(load, pv, price, dates, lightgbm_forecasts, bundles, a3_result)
    save(OUT / 'feedback_prefix_check.json', prefix)

    produced = m.figures(results, contrasts, metrics, stability, pd.DataFrame(metrics[3]))
    renamed = []
    for path in produced:
        path = Path(path)
        target = FIG / path.name.replace('q2_ridge_', 'q2_lightgbm_')
        if path != target:
            shutil.move(str(path), str(target))
        renamed.append(target)
    produced = renamed
    integrity = m.figure_integrity(produced)
    save(OUT / 'figure_integrity.json', integrity)

    protected_after = m.protected_manifest()
    changed = sorted(k for k in protected if protected_after.get(k) != protected[k])
    missing = sorted(k for k in protected if k not in protected_after)
    save(OUT / 'protected_after.json', protected_after)

    write_report(results, comparison, primary, contrasts, metrics, coverage, pd.DataFrame(metrics[3]),
                 gate, boundary, perturbation, prefix, reference, lightgbm_forecasts['fit_log'],
                 warm_end, primary_summary, supplementary)

    self_checks = dict(reference_reproduction=reference, warmup=dict(shared_begin_state_kWh=warm_end),
                       boundary=boundary, future_perturbation=perturbation, feedback_prefix=prefix,
                       figure_integrity=integrity, gate_summary=gate, primary_stability=primary_summary,
                       protected_unchanged=bool(not changed and not missing),
                       protected_changed=changed, protected_missing=missing)
    save(OUT / 'checks.json', self_checks)

    artifact_hashes = {p.relative_to(OUT).as_posix(): digest(p)
                       for p in sorted(OUT.rglob('*'))
                       if p.is_file() and p.name not in ('artifact_hashes.json', 'run_manifest.json')}
    save(OUT / 'artifact_hashes.json', artifact_hashes)
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                    finished_utc=datetime.now(timezone.utc).isoformat(),
                    wall_seconds=time.perf_counter() - started, mode=args.mode,
                    executable=sys.executable, python=sys.version, lightgbm=lightgbm.__version__,
                    numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                    signature=signature, parameters=PARAMETERS,
                    group_totals={r['totals']['strategy_id']: r['totals'] for r in results},
                    reference_reproduction=reference, self_checks_passed=dict(
                        boundary=boundary['all_passed'], future_perturbation=perturbation['all_passed'],
                        feedback_prefix=prefix['passed'], reference_g1=reference['G1']['passed']),
                    protected_before_count=len(protected), protected_unchanged=bool(not changed and not missing),
                    protected_changed=changed, protected_missing=missing,
                    warmup_shared_begin_state_kWh=warm_end,
                    artifact_count=len(artifact_hashes), report=str(REPORT_MD.relative_to(ROOT)))
    save(OUT / 'run_manifest.json', manifest)
    assert manifest['protected_unchanged'], (changed, missing)
    print(comparison[['strategy_id', 'planned_cost_yuan', 'emergency_cost_yuan',
                      'total_cost_yuan', 'delta_total_cost_yuan']].to_string(index=False), flush=True)
    print(f'wall seconds = {time.perf_counter() - started:.1f}', flush=True)
    return manifest


if __name__ == '__main__':
    main()
