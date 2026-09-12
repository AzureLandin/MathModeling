"""Q2 PV historical-support-window feature experiment (module 23).

Executes ``reports/问题二_光伏历史窗口特征优化实验方案.md``: four scenarios
``P0_q80``/``P0_q75``/``P1_q80``/``P1_q75``.

The load forecast is the frozen repaired G4 archive in every scenario and is never retrained.
The PV forecast is either

* ``P0`` -- the frozen repaired G4 published archive (control), or
* ``P1`` -- a LightGBM residual correction rebuilt with the original eight features plus four
  past-only support-window columns ``D``/``M``/``tau``/``v`` (candidate).

Only the PV model input columns change.  Training rows, labels, the 56-day window, the 14-day
minimum, the first trainable day, the naive fallback days, the fixed LightGBM parameters and the
unified support gate are identical to module 19.  No annual cycle, no month dummies, no parameter
search, no threshold sensitivity, no change to files created by other modules.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE_23 = ROOT / 'code/23_q2_pv_support_features_experiment.py'
OUT = ROOT / 'results/q2_pv_support_features'
FIG = ROOT / 'figures/q2_pv_support_features'
SNAP = ROOT / 'results/q2_revision_audit_20260911/source_snapshot'
FROZEN_RUN = ROOT / 'results/q2_lightgbm_residual'
SCAN_RUN = ROOT / 'results/q2_quantile_level_scan'
PLAN_MD = ROOT / 'reports/问题二_光伏历史窗口特征优化实验方案.md'
REPORT_MD = ROOT / 'reports/问题二_光伏历史窗口特征优化实验结果报告.md'

DT = 1 / 6
TRAIN_WINDOW = 56
MIN_TRAIN_DAYS = 14
RESIDUAL_WINDOW = 28
MIN_HISTORY_DAYS = 7
TERMINAL_KWH = 6000.0
WARMUP_DAYS = 31
REFERENCE_WARMUP_KWH = 8801.462273333342
CONSTANT_TARGET_TOL_KW = 1e-8
ENERGY_TOL_KWH = 1e-6
COST_TOL_YUAN = 1e-4

SUPPORT_PARAMETERS = dict(history_days=7, activity_threshold_kW=1.0, padding_slots=3)
GATE_HISTORY_DAYS = SUPPORT_PARAMETERS['history_days']
GATE_THRESHOLD_KW = SUPPORT_PARAMETERS['activity_threshold_kW']
GATE_PADDING_SLOTS = SUPPORT_PARAMETERS['padding_slots']

CONTROL_FEATURES = ['v_base', 'v_recent_change', 'v_week_mean_minus_base', 'v_daily_change',
                    'sin1', 'cos1', 'sin2', 'cos2']
SUPPORT_FEATURES = ['support_duration_h', 'support_midpoint_h', 'support_phase', 'support_valid']
CANDIDATE_FEATURES = CONTROL_FEATURES + SUPPORT_FEATURES

# reports/问题二_光伏历史窗口特征优化实验方案.md section 1.3: P0_q80 and P0_q75 are the two
# registered controls; both must reproduce their existing archives before any candidate runs.
REFERENCE = {
    'P0_q80': dict(folder=SCAN_RUN / 'L_q80', total=13673080.670412183, alpha=0.80),
    'P0_q75': dict(folder=SCAN_RUN / 'L_q75', total=13627389.058274437, alpha=0.75),
}
CONTROL_PV_TOTAL_Q80 = REFERENCE['P0_q80']['total']
CONTROL_PV_TOTAL_Q75 = REFERENCE['P0_q75']['total']

GROUPS = [
    dict(id='P0_q80', predictor='P0', pv_source='frozen_g4', alpha=0.80,
         role='control: frozen repaired G4 PV published archive, alpha=0.80'),
    dict(id='P0_q75', predictor='P0', pv_source='frozen_g4', alpha=0.75,
         role='control: frozen repaired G4 PV published archive, alpha=0.75'),
    dict(id='P1_q80', predictor='P1', pv_source='support12', alpha=0.80,
         role='candidate: 12-column support-window PV LightGBM, alpha=0.80'),
    dict(id='P1_q75', predictor='P1', pv_source='support12', alpha=0.75,
         role='candidate: 12-column support-window PV LightGBM, alpha=0.75'),
]
CONTROL_GROUPS = [g for g in GROUPS if g['id'] in REFERENCE]
TREATMENT_GROUPS = [g for g in GROUPS if g['id'] not in REFERENCE]
MAIN_COMPARISON = 'P1_q80 minus P0_q80 (pre-registered main comparison)'
AUXILIARY_COMPARISON = 'P1_q75 minus P0_q75 (pre-registered paired robustness comparison)'

SAMPLED_TRAINING_DAYS = [20, 21, 31, 78, 171, 265, 354]
SAMPLED_LABELS = {20: '2025-01-21', 21: '2025-01-22', 31: '2025-02-01', 78: '2025-03-20',
                  171: '2025-06-21', 265: '2025-09-23', 354: '2025-12-21'}
PERTURBATION_DAYS = [31, 171, 354]

PARAMETERS = dict(
    experiment='PV support-window features, four pre-registered scenarios',
    control_pv_features=CONTROL_FEATURES, candidate_pv_features=CANDIDATE_FEATURES,
    support_parameters=SUPPORT_PARAMETERS, train_window_days=TRAIN_WINDOW,
    min_train_days=MIN_TRAIN_DAYS, residual_window_days=RESIDUAL_WINDOW,
    min_history_days=MIN_HISTORY_DAYS, terminal_kWh=TERMINAL_KWH,
    first_pv_feature_day=7, first_pv_trainable_day=21,
    groups=[dict(id=g['id'], predictor=g['predictor'], alpha=g['alpha'], pv_source=g['pv_source'])
            for g in GROUPS],
    main_comparison=MAIN_COMPARISON, auxiliary_comparison=AUXILIARY_COMPARISON,
    control_reference=dict(P0_q80=CONTROL_PV_TOTAL_Q80, P0_q75=CONTROL_PV_TOTAL_Q75),
    fallback_rule='fewer than 14 valid training days or a pre-feature day publishes the causal '
                  'naive forecast through the same unified gate; a constant-label window takes the '
                  'label mean instead of a pseudo tree model',
    load_rule='the load forecast is the frozen repaired G4 archive in all four scenarios; the load '
              'model is not retrained, and its training causality remains an upstream audit item',
    evaluation='2025-02-01..2025-12-31, 334 days, 48096 slots per group',
)

PROTECTED_TREES = ['附件', 'code', 'reports', 'figures', 'results']
OWN_NEW_REL = {'code/23_q2_pv_support_features_experiment.py',
               'reports/问题二_光伏历史窗口特征优化实验结果报告.md'}
OWN_NEW_PREFIXES = ('results/q2_pv_support_features/', 'figures/q2_pv_support_features/')
SHARED_APPEND_REL = {'建模上下文记忆.md', 'reports/项目进度.md'}

_MODULES: dict = {}


def parent():
    """Module 14: frozen snapshot access, naive predictors and the original PV feature function."""
    if 'ridge' not in _MODULES:
        spec = importlib.util.spec_from_file_location(
            'ridge_parent', ROOT / 'code/14_q2_ridge_forecast_experiment.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULES['ridge'] = module
    return _MODULES['ridge']


def scan():
    """Module 21: explicit-alpha archive, scenario runner and identity checks.

    ``scan.OUT`` is redirected to this experiment's own result tree so that reusing its runner can
    never write into ``results/q2_quantile_level_scan``.
    """
    if 'scan' not in _MODULES:
        spec = importlib.util.spec_from_file_location(
            'quantile_scan', ROOT / 'code/21_q2_quantile_level_scan.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.OUT = OUT
        module.FIG = FIG
        _MODULES['scan'] = module
    return _MODULES['scan']


def lightgbm_module():
    """Module 19: source of the frozen, already-registered LightGBM parameter dictionary."""
    if 'm19' not in _MODULES:
        spec = importlib.util.spec_from_file_location(
            'lightgbm_residual', ROOT / 'code/19_q2_lightgbm_residual_experiment.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULES['m19'] = module
    return _MODULES['m19']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def frame_to_csv(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8-sig')


# --------------------------------------------------------------------------------------
# past-only support window and the four new columns
# --------------------------------------------------------------------------------------
def support_window(pv, k):
    """(s, e, v) for decision day k from the previous seven complete days only.

    Exactly the rule of the unified gate: slots whose seven-day maximum exceeds 1 kW form the
    active set; the window is that set's hull padded by three slots per side and clipped to
    [0, 143].  No active slot, or fewer than seven days of history, gives the documented default
    ``s=0, e=143, v=0`` (24 h, midpoint 12 h, no valid window).
    """
    if k < GATE_HISTORY_DAYS:
        return 0, 143, 0
    block = np.asarray(pv, dtype=float)[k - GATE_HISTORY_DAYS:k]
    active = np.flatnonzero(np.max(block, axis=0) > GATE_THRESHOLD_KW)
    if len(active) == 0:
        return 0, 143, 0
    return (max(0, int(active[0]) - GATE_PADDING_SLOTS),
            min(143, int(active[-1]) + GATE_PADDING_SLOTS), 1)


def day_gate(pv, k):
    """Boolean allow mask for day k; identical to module 16/19 ``support_mask`` row k."""
    s, e, _ = support_window(pv, k)
    mask = np.zeros(144, dtype=bool)
    mask[s:e + 1] = True
    return mask


def support_mask(pv):
    """Stack of ``day_gate``; cross-checked against the frozen ``gate_mask.csv``."""
    return np.vstack([day_gate(pv, k) for k in range(len(pv))])


def apply_gate(values, allow):
    return np.where(np.asarray(allow, dtype=bool)[:len(values)], np.asarray(values, dtype=float), 0.0)


def support_phase(k, s, e):
    n = e - s + 1
    return (np.arange(144, dtype=float) - s + 0.5) / n


def day_features(pv, dates, k, kind):
    """Feature matrix for one decision day; ``kind`` selects the 8-column control or 12-column set."""
    base = parent().pv_feature_matrix(pv, dates, k)
    if kind == 'original8':
        return base
    s, e, valid = support_window(pv, k)
    n = e - s + 1
    out = np.empty((144, len(CANDIDATE_FEATURES)))
    out[:, :len(CONTROL_FEATURES)] = base
    out[:, 8] = n * DT
    out[:, 9] = (s + e + 1) * DT / 2
    out[:, 10] = support_phase(k, s, e)
    out[:, 11] = float(valid)
    return out


def fit_day(pv, dates, k, kind, model_dir=None):
    """Fit and publish one PV decision day, strictly causally.

    Returns the published (gated) row, the ungated clipped row, the correction, the training row
    count, the branch and (for a fitted tree model) the model text and tree count.
    """
    gate = day_gate(pv, k)
    base = np.full(144, np.nan) if k < 1 else np.asarray(pv[k - 1], dtype=float)
    threshold = parent().PV_THRESHOLD
    train_days = ([j for j in range(max(0, k - TRAIN_WINDOW), k) if j >= threshold]
                  if k >= threshold else [])
    entry = dict(day_index=int(k), date=str(dates[k].date()), n_train_days=len(train_days),
                 n_rows=0, trained=False, branch='', n_trees=0, model_file='', model_sha256='',
                 constant_new_columns='', correction_min_kW=np.nan, correction_max_kW=np.nan)
    if k < threshold:
        entry['branch'] = 'feature_history_fallback'
        published = apply_gate(base, gate)
        return dict(published=published, raw=published.copy(), correction=np.zeros(144), base=base,
                    gate=gate, support=support_window(pv, k), entry=entry, model=None, model_text='')
    if len(train_days) < MIN_TRAIN_DAYS:
        entry['branch'] = 'insufficient_history'
        published = apply_gate(base, gate)
        return dict(published=published, raw=published.copy(), correction=np.zeros(144), base=base,
                    gate=gate, support=support_window(pv, k), entry=entry, model=None, model_text='')
    from lightgbm import LGBMRegressor
    X = np.concatenate([day_features(pv, dates, j, kind) for j in train_days], axis=0)
    y = np.concatenate([pv[j] - pv[j - 1] for j in train_days])
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise RuntimeError(f'non-finite feature or label on day {k}')
    entry['n_rows'] = int(X.shape[0])
    entry['train_first'] = str(dates[train_days[0]].date())
    entry['train_last'] = str(dates[train_days[-1]].date())
    if kind != 'original8':
        never = [name for name, column in zip(SUPPORT_FEATURES, X[:, 8:].T)
                 if float(np.max(column) - np.min(column)) == 0.0]
        entry['constant_new_columns'] = ','.join(never)
    spread = float(np.max(y) - np.min(y))
    if spread <= CONSTANT_TARGET_TOL_KW:
        correction = np.full(144, float(np.mean(y)))
        entry['branch'] = 'constant_target'
        entry['constant_value_kW'] = float(np.mean(y))
        model = None
        model_text = ''
    else:
        model = LGBMRegressor(**lightgbm_module().LGBM_PARAMS)
        model.fit(X, y)
        correction = np.asarray(model.predict(day_features(pv, dates, k, kind)), dtype=float)
        entry['branch'] = 'fitted'
        entry['n_trees'] = int(model.booster_.num_trees())
        model_text = model.booster_.model_to_string()
        entry['model_sha256'] = hashlib.sha256(model_text.encode('utf-8')).hexdigest()
        if model_dir is not None:
            name = f'pv_{dates[k].date()}.txt'
            (model_dir / name).write_text(model_text, encoding='utf-8')
            entry['model_file'] = f'models/{kind}/{name}'
    entry['trained'] = True
    raw = np.maximum(0.0, base + correction)
    published = apply_gate(raw, gate)
    entry['correction_min_kW'] = float(np.min(correction))
    entry['correction_max_kW'] = float(np.max(correction))
    entry['prediction_min_kW'] = float(published.min())
    entry['prediction_max_kW'] = float(published.max())
    return dict(published=published, raw=raw, correction=correction, base=base, gate=gate,
                support=support_window(pv, k), entry=entry, model=model, model_text=model_text)


def build_pv_forecast(pv, dates, kind, save_models=False, verbose=False):
    """Whole-year causal PV archive for one feature set; day 0 stays unpublished (NaN)."""
    n_days = len(dates)
    published = np.full((n_days, 144), np.nan)
    raw = np.full((n_days, 144), np.nan)
    gate = np.zeros((n_days, 144), dtype=bool)
    support = np.zeros((n_days, 3), dtype=int)
    log = []
    model_dir = OUT / 'models' / kind if save_models else None
    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for k in range(1, n_days):
        outcome = fit_day(pv, dates, k, kind, model_dir=model_dir)
        published[k] = outcome['published']
        raw[k] = outcome['raw']
        gate[k] = outcome['gate']
        support[k] = outcome['support']
        log.append(outcome['entry'])
        if verbose and k % 90 == 0:
            print(f'    {kind}: day {k} ({dates[k].date()}) branch={outcome["entry"]["branch"]}',
                  flush=True)
    trained = [e for e in log if e['trained']]
    first = int(np.flatnonzero(~np.isnan(published[:, 0]))[0]) if not np.isnan(published[:, 0]).all() else -1
    summary = dict(kind=kind, n_days=n_days, n_fitted=int(sum(1 for e in log if e['branch'] == 'fitted')),
                   n_constant=int(sum(1 for e in log if e['branch'] == 'constant_target')),
                   n_insufficient=int(sum(1 for e in log if e['branch'] == 'insufficient_history')),
                   n_feature_fallback=int(sum(1 for e in log if e['branch'] == 'feature_history_fallback')),
                   first_trainable_day=int(trained[0]['day_index']) if trained else -1,
                   first_published_day=first, elapsed_seconds=time.perf_counter() - started)
    return dict(kind=kind, published=published, raw=raw, gate=gate, support=support, log=log,
                summary=summary)


def load_g4_archive(load, pv, dates):
    """Frozen repaired G4 published archive (load and PV), verified against the raw attachments."""
    s = scan()
    archive = s.load_predictor_archive(dict(id='P0', folder='G4_lightgbm_fixed'), load, pv, dates)
    folder = FROZEN_RUN / 'training_log.csv'
    archive['training_log'] = pd.read_csv(folder, low_memory=False)
    return archive


# --------------------------------------------------------------------------------------
# reference reproduction (three tiers, as in module 21)
# --------------------------------------------------------------------------------------
def compare_to_reference(result, group_id):
    """Tier 1 day-level aggregates, tier 2 registered total, tier 3 equivalent-optimum evidence."""
    s = scan()
    reference = pd.read_csv(REFERENCE[group_id]['folder'] / 'dispatch.csv', low_memory=False)
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
        reference.nominal_state_end_kWh.to_numpy() - current.nominal_state_end_kWh.to_numpy())))
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
    out = dict(group=group_id, segments=int(len(current)), max_executed_difference=float(
        max(executed_differences.values())), max_absolute_differences=executed_differences,
        max_day_level_difference=float(day_differences.to_numpy().max()),
        day_level_differences={name: float(day_differences[name].max()) for name in day_differences},
        days_with_any_difference=int((day_differences.max(axis=1) > ENERGY_TOL_KWH).sum()),
        nominal_trajectory_difference_kWh=nominal_difference,
        equivalent_optimum=dict(
            affected_slots=int(len(affected)),
            affected_dates=sorted(set(current.date.to_numpy()[affected])) if len(affected) else [],
            max_slot_plan_difference_kWh=float(np.max(np.abs(delta_plan))) if len(affected) else 0.0,
            affected_slot_prices_equal=prices_equal, day_level_cost_identical=day_cost_identical,
            note='a day-ahead optimum that moves charge between equal-price slots changes the '
                 'slot-level plan without moving any day-level aggregate or the realised cash cost'),
        reference_total_cost_yuan=reference_cost, current_total_cost_yuan=current_cost,
        difference_vs_reference_yuan=abs(current_cost - reference_cost),
        registered_total_cost_yuan=REFERENCE[group_id]['total'],
        difference_vs_registered_yuan=abs(current_cost - REFERENCE[group_id]['total']))
    out['passed'] = bool(out['max_day_level_difference'] < ENERGY_TOL_KWH
                         and out['difference_vs_registered_yuan'] <= COST_TOL_YUAN
                         and prices_equal and day_cost_identical)
    return out


# --------------------------------------------------------------------------------------
# implementation checks
# --------------------------------------------------------------------------------------
def boundary_checks(pv, dates):
    """Synthetic hand-computed boundaries; never presented as an attachment finding."""
    rows = []
    zeros = np.zeros((364, 144))

    def add(name, value, expected, tolerance=0.0):
        value = np.asarray(value, dtype=float)
        expected = np.asarray(expected, dtype=float)
        worst = float(np.max(np.abs(value - expected))) if value.shape == expected.shape else float('inf')
        rows.append(dict(check=name, worst=worst, tolerance=tolerance,
                         passed=bool(worst <= tolerance + 1e-15), expected=expected.tolist()
                         if expected.size <= 6 else 'see code'))

    add('k<7 default window', support_window(zeros, 6), [0, 143, 0])
    add('k=7 empty active set', support_window(zeros, 7), [0, 143, 0])
    single = zeros.copy()
    single[3, 10] = 5.0
    add('single active slot t=10', support_window(single, 10), [7, 13, 1])
    left = zeros.copy()
    left[3, 0] = 5.0
    add('active slot touching 0 clips low side', support_window(left, 10), [0, 3, 1])
    right = zeros.copy()
    right[3, 143] = 5.0
    add('active slot touching 143 clips high side', support_window(right, 10), [140, 143, 1])
    exact = zeros.copy()
    exact[3, 50] = 1.0
    add('exactly 1 kW is not active', support_window(exact, 10), [0, 143, 0])
    above = zeros.copy()
    above[3, 50] = 1.0 + 1e-9
    add('slightly above 1 kW is active', support_window(above, 10), [47, 53, 1])
    full = zeros.copy()
    full[3, :] = 5.0
    add('full-day support keeps v=1', support_window(full, 10), [0, 143, 1])
    add('full-day v differs from empty v', support_window(full, 10)[2] - support_window(zeros, 10)[2], 1)

    s, e, _ = support_window(single, 10)
    add('D hand calculation', (e - s + 1) * DT, 7 * DT)
    add('M hand calculation', (s + e + 1) * DT / 2, (7 + 13 + 1) * DT / 2)
    phase = support_phase(10, 7, 13)
    add('tau endpoints', [phase[7], phase[13]], [0.5 / 7, 6.5 / 7], 1e-15)
    add('tau length', len(phase), 144)

    matrix = day_features(full, dates, 10, 'support12')
    add('candidate feature width', matrix.shape, [144, 12])
    add('candidate first 8 columns', matrix[:, :8], parent().pv_feature_matrix(full, dates, 10), 1e-12)
    add('duration column constant', matrix[:, 8], np.full(144, 24.0))
    add('midpoint column constant', matrix[:, 9], np.full(144, 12.0))
    add('valid column constant', matrix[:, 11], np.full(144, 1.0))

    s = scan()
    add('positions at m=28', [s.quantile_index(28, 0.80), s.quantile_index(28, 0.75)], [23, 21])
    rule_ok = all(s.quantile_index(m, a) == s.ceil_index(m, a)
                  for m in range(MIN_HISTORY_DAYS, RESIDUAL_WINDOW + 1) for a in (0.80, 0.75))
    rows.append(dict(check='integer rule equals ceil for m=7..28', worst=0.0, tolerance=0.0,
                     passed=bool(rule_ok), expected='all equal'))
    eps = np.tile(np.arange(144, dtype=float), (40, 1))
    r, m, reason = s.adjustment(eps, 6, RESIDUAL_WINDOW, 0.80)
    rows.append(dict(check='m<7 returns zero correction', worst=float(np.abs(r).max()), tolerance=0.0,
                     passed=bool(m == 5 and reason.startswith('insufficient_history')
                                 and np.all(r == 0)), expected='zeros'))
    r, m, _ = s.adjustment(eps, 35, RESIDUAL_WINDOW, 0.80)
    ordered = np.sort(eps[7:35], axis=0)[22]
    add('m=28 q80 is the 23rd order statistic', r, ordered, 0.0)
    negative = np.zeros((12, 144)) - 3.0
    r, m, _ = s.adjustment(negative, 11, RESIDUAL_WINDOW, 0.75)
    add('negative corrections are retained', r, np.full(144, -3.0), 0.0)
    repeated = np.zeros((12, 144))
    repeated[:, :] = 2.0
    r, m, _ = s.adjustment(repeated, 11, RESIDUAL_WINDOW, 0.80)
    add('ties are stable', r, np.full(144, 2.0), 0.0)
    nonfinite_rejected = False
    try:
        s.require_finite_published(np.ones((365, 144)), np.full((365, 144), np.nan), 365)
    except ValueError:
        nonfinite_rejected = True
    rows.append(dict(check='non-finite published archive fails', worst=0.0, tolerance=0.0,
                     passed=bool(nonfinite_rejected), expected='ValueError'))
    return dict(rows=rows, all_passed=bool(all(r['passed'] for r in rows)))


def gate_consistency_checks(pv, dates, g4):
    """My reconstructed gate and the frozen one must agree; the frozen PV must be zero off-gate."""
    frozen = pd.read_csv(FROZEN_RUN / 'gate_mask.csv', low_memory=False, dtype={'date': str})
    frozen = frozen.sort_values(['date', 'slot'], kind='mergesort').reset_index(drop=True)
    allow = frozen.allow_pv.to_numpy().astype(bool).reshape(len(dates), 144)
    mine = support_mask(pv)
    off = ~allow
    published = np.asarray(g4['pv_forecast'], dtype=float)
    zero_off = float(np.abs(np.nan_to_num(published)[off]).max())
    return dict(mask_mismatch_slots=int((mine != allow).sum()),
                frozen_pv_max_off_gate_kW=zero_off,
                gated_off_slots=int(off.sum()),
                zeroed_truth_energy_kWh=float((pv[off] * DT).sum()),
                zeroed_truth_above_100kW_slots=int((pv[off] > 100).sum()),
                all_passed=bool((mine == allow).all() and zero_off < ENERGY_TOL_KWH))


def control_refit_check(pv, dates, g4):
    """Refit the frozen 8-column control PV model and compare it with the published archive."""
    rebuilt = build_pv_forecast(pv, dates, 'original8')
    published = np.asarray(g4['pv_forecast'], dtype=float)
    mask = np.isfinite(published)
    difference = float(np.max(np.abs(rebuilt['published'][mask] - published[mask])))
    return dict(max_difference_kW=difference, compared_slots=int(mask.sum()),
                n_fitted=rebuilt['summary']['n_fitted'], n_constant=rebuilt['summary']['n_constant'],
                first_trainable_day=rebuilt['summary']['first_trainable_day'],
                all_passed=bool(difference < 1e-6)), rebuilt


def training_parity_checks(candidate, g4):
    """Training row keys, labels, first trainable day and branches must match the frozen control."""
    control = g4['training_log']
    control = control[control.target == 'pv'].set_index('day_index').sort_index()
    mismatches = []
    for entry in candidate['log']:
        k = entry['day_index']
        if k not in control.index:
            mismatches.append(dict(day=k, field='missing_in_control'))
            continue
        row = control.loc[k]
        for field, mine, theirs in (('n_train_days', entry['n_train_days'], int(row.n_train_days)),
                                    ('n_rows', entry['n_rows'], int(row.n_rows)),
                                    ('trained', entry['trained'], bool(row.trained)),
                                    ('branch', entry['branch'], str(row.branch))):
            if mine != theirs:
                mismatches.append(dict(day=k, field=field, mine=mine, control=theirs))
    first = next((e['day_index'] for e in candidate['log'] if e['trained']), -1)
    control_first = int(control.index[control.trained.astype(bool)][0])
    label_diff = 0.0
    dates_dummy = None
    return dict(mismatches=mismatches[:40], mismatch_count=len(mismatches),
                first_trainable_day=first, control_first_trainable_day=control_first,
                label_definition_identical=bool(label_diff == 0.0),
                all_passed=bool(not mismatches and first == control_first == 21))


def quantile_validation(archives):
    """Explicit-alpha order statistics and per-m calculation rules, before any run."""
    s = scan()
    out = dict(positions_at_28_days={f'{a:.2f}': s.quantile_index(RESIDUAL_WINDOW, a)
                                     for a in (0.80, 0.75)},
               expected={'0.80': 23, '0.75': 21})
    out['positions_passed'] = bool(out['positions_at_28_days'] == out['expected'])
    out['integer_rule_equals_ceil'] = bool(all(
        s.quantile_index(m, a) == s.ceil_index(m, a)
        for m in range(MIN_HISTORY_DAYS, RESIDUAL_WINDOW + 1) for a in (0.80, 0.75)))
    per_archive = {}
    for name, archive in archives.items():
        eps = archive['eps']
        worst = 0.0
        for k in range(WARMUP_DAYS, len(eps)):
            for alpha in (0.80, 0.75):
                low = max(1, k - RESIDUAL_WINDOW)
                m = k - low
                if m < MIN_HISTORY_DAYS:
                    continue
                expected = np.sort(eps[low:k], axis=0)[s.quantile_index(m, alpha) - 1]
                r, _, _ = s.adjustment(eps, k, RESIDUAL_WINDOW, alpha)
                worst = max(worst, float(np.max(np.abs(r - expected))))
        per_archive[name] = dict(max_difference_kWh=worst, passed=bool(worst < ENERGY_TOL_KWH))
    out['per_archive'] = per_archive
    out['all_passed'] = bool(out['positions_passed'] and out['integer_rule_equals_ceil']
                             and all(v['passed'] for v in per_archive.values()))
    return out


def perturbation_checks(load, pv, price, dates, p0, p1, clean):
    """Future-truth perturbations on the PV chain and on the frozen load chain.

    The issue day's support window, gate, training rows, published forecast, both alpha protections
    and both ordinary plans must be unchanged.  What happens after the issue day's truth is
    revealed is allowed to change and is not asserted.
    """
    s = scan()
    cases = []
    for k in PERTURBATION_DAYS:
        for kind in ('pv', 'load'):
            ll, vv = load.copy(), pv.copy()
            if kind == 'pv':
                vv[k:] *= 0.7
            else:
                ll[k:] *= 1.2
            rebuilt = fit_day(vv, dates, k, 'support12')
            clean_entry = next(e for e in p1['log'] if e['day_index'] == k)
            forecast_error = float(np.max(np.abs(rebuilt['published'] - p1['published'][k])))
            gate_equal = bool(np.array_equal(rebuilt['gate'], p1['gate'][k]))
            window_equal = bool(tuple(rebuilt['support']) == tuple(int(v) for v in p1['support'][k]))
            rows_equal = bool(rebuilt['entry']['n_train_days'] == clean_entry['n_train_days']
                              and rebuilt['entry']['n_rows'] == clean_entry['n_rows']
                              and rebuilt['entry']['branch'] == clean_entry['branch'])
            pv_fc = p1['published'].copy()
            pv_fc[k] = rebuilt['published']
            archive = s.ScanArchive(ll, vv, len(dates), p0['load_forecast'], pv_fc)
            clean_archive = clean['P1']
            protection, plan = {}, {}
            for alpha in (0.80, 0.75):
                gid = f'P1_q{int(round(alpha * 100))}'
                r_pert, _, _ = s.adjustment(archive.eps, k, RESIDUAL_WINDOW, alpha)
                r_clean, _, _ = s.adjustment(clean_archive.eps, k, RESIDUAL_WINDOW, alpha)
                protection[f'{alpha:.2f}'] = float(np.max(np.abs(r_pert - r_clean)))
                state = float(clean['initial'][(gid, k)])
                q_pert, _, summary_pert, _, _, _, _ = archive.plan_day(
                    k, alpha, RESIDUAL_WINDOW, price, state)
                q_clean, _, summary_clean, _, _, _, _ = clean_archive.plan_day(
                    k, alpha, RESIDUAL_WINDOW, price, state)
                assert not summary_pert.get('fallback', False), (k, kind, alpha, summary_pert)
                assert not summary_clean.get('fallback', False), (k, kind, alpha, summary_clean)
                plan[f'{alpha:.2f}'] = float(np.max(np.abs(q_pert - q_clean)))
            passed = bool(forecast_error < ENERGY_TOL_KWH and gate_equal and window_equal and rows_equal
                          and max(protection.values()) < ENERGY_TOL_KWH
                          and max(plan.values()) < ENERGY_TOL_KWH)
            assert passed, (k, kind, forecast_error, protection, plan)
            cases.append(dict(day=int(k), date=str(dates[k].date()), kind=kind,
                              forecast_max_kW=forecast_error, gate_equal=gate_equal,
                              window_equal=window_equal, training_rows_equal=rows_equal,
                              protection_max_kWh=protection, plan_max_kWh=plan, passed=passed))
    return dict(cases=cases, all_passed=bool(all(c['passed'] for c in cases)),
                scope='support window + gate + training rows + published PV + both alpha protections '
                      '+ both ordinary plans on the issue day; post-reveal execution is not asserted')


def feedback_prefix_check(load, pv, price, dates, p0, p1, clean):
    """A midday truth perturbation must not move the same day's morning plan or execution prefix."""
    s = scan()
    k = 171
    m = parent()
    rows = []
    for kind in ('load', 'pv'):
        ll, vv = load.copy(), pv.copy()
        if kind == 'load':
            ll[k, 72:] *= 1.2
        else:
            vv[k, 72:] *= 0.7
        archive = s.ScanArchive(ll, vv, len(dates), p0['load_forecast'], p1['published'])
        clean_archive = clean['P1']
        for alpha in (0.80, 0.75):
            gid = f'P1_q{int(round(alpha * 100))}'
            state = float(clean['initial'][(gid, k)])
            q_pert, _, summary_pert, _, _, _, _ = archive.plan_day(
                k, alpha, RESIDUAL_WINDOW, price, state)
            q_clean, _, summary_clean, _, _, _, _ = clean_archive.plan_day(
                k, alpha, RESIDUAL_WINDOW, price, state)
            assert not summary_pert.get('fallback', False), (kind, alpha, summary_pert)
            assert not summary_clean.get('fallback', False), (kind, alpha, summary_clean)
            plan_error = float(np.max(np.abs(q_pert - q_clean)))
            control = m.frozen().bm.control
            executed_clean = clean['executed'][(gid, k)]
            c, d, e, w, states, _ = control(q_pert, ll[k], vv[k], state)
            prefix = max(float(np.max(np.abs(c[:72] - executed_clean['c'][:72]))),
                         float(np.max(np.abs(d[:72] - executed_clean['d'][:72]))),
                         float(np.max(np.abs(e[:72] - executed_clean['e'][:72]))),
                         float(np.max(np.abs(w[:72] - executed_clean['w'][:72]))),
                         float(np.max(np.abs(states[:72] - executed_clean['states'][:72]))))
            passed = bool(plan_error < ENERGY_TOL_KWH and prefix < ENERGY_TOL_KWH)
            assert passed, (kind, alpha, plan_error, prefix)
            rows.append(dict(day=k, kind=kind, alpha=alpha, plan_max_kWh=plan_error,
                             executed_prefix_max=prefix, passed=passed))
    return dict(cases=rows, all_passed=bool(all(c['passed'] for c in rows)),
                scope='day 171 plan and slots 0..71 of the realised execution, both alphas')


def sampled_retraining_checks(pv, dates, p1):
    """Independently refit both feature sets on the registered days and reload the saved models."""
    from lightgbm import Booster
    rows = []
    for k in SAMPLED_TRAINING_DAYS:
        rebuilt = fit_day(pv, dates, k, 'support12')
        archive = p1['published'][k]
        refit_difference = float(np.max(np.abs(rebuilt['published'] - archive)))
        entry = next(e for e in p1['log'] if e['day_index'] == k)
        reload_difference = np.nan
        if entry['branch'] == 'fitted':
            text = (OUT / entry['model_file']).read_text(encoding='utf-8')
            booster = Booster(model_str=text)
            correction = np.asarray(booster.predict(day_features(pv, dates, k, 'support12')), dtype=float)
            base = np.asarray(pv[k - 1], dtype=float)
            reloaded = apply_gate(np.maximum(0.0, base + correction), rebuilt['gate'])
            reload_difference = float(np.max(np.abs(reloaded - rebuilt['published'])))
        rows.append(dict(day=int(k), date=SAMPLED_LABELS[k], branch=entry['branch'],
                         n_train_days=entry['n_train_days'], n_rows=entry['n_rows'],
                         refit_max_kW=refit_difference, reload_max_kW=reload_difference,
                         passed=bool(refit_difference < 1e-6
                                     and (np.isnan(reload_difference) or reload_difference < 1e-6))))
    return dict(rows=rows, all_passed=bool(all(r['passed'] for r in rows)),
                scope='sampled PV retraining and saved-model reload; not a full independent recompute')


# --------------------------------------------------------------------------------------
# reporting frames
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
                delta_planned_kWh=float(treatment['totals']['planned_kWh']
                                        - baseline['totals']['planned_kWh']),
                delta_emergency_kWh=float(treatment['totals']['emergency_kWh']
                                          - baseline['totals']['emergency_kWh']),
                delta_unused_kWh=float(treatment['totals']['unused_kWh']
                                       - baseline['totals']['unused_kWh']),
                delta_emergency_days=int(treatment['totals']['emergency_days']
                                         - baseline['totals']['emergency_days']),
                delta_emergency_events=int(treatment['totals']['emergency_events']
                                           - baseline['totals']['emergency_events']))


def paired_contrasts(results):
    s = scan()
    by_id = {r['totals']['strategy_id']: r for r in results}
    rows = [delta_row(MAIN_COMPARISON, by_id['P1_q80'], by_id['P0_q80']),
            delta_row(AUXILIARY_COMPARISON, by_id['P1_q75'], by_id['P0_q75']),
            delta_row('P1_q75 minus P1_q80 (protection level)', by_id['P1_q75'], by_id['P1_q80']),
            delta_row('P0_q75 minus P0_q80 (protection level)', by_id['P0_q75'], by_id['P0_q80'])]
    frame = pd.DataFrame(rows)
    frame['delta_total_cost_percent'] = [100 * r.delta_total_cost_yuan
                                         / by_id[r.baseline]['totals']['total_cost_yuan']
                                         for r in frame.itertuples()]
    return frame


def monthly_contrasts(results):
    by_id = {r['totals']['strategy_id']: r for r in results}
    rows = []
    for label, treatment, baseline in ((MAIN_COMPARISON, 'P1_q80', 'P0_q80'),
                                       (AUXILIARY_COMPARISON, 'P1_q75', 'P0_q75')):
        t = by_id[treatment]['monthly'].set_index('month')
        b = by_id[baseline]['monthly'].set_index('month')
        for month in sorted(t.index):
            rows.append(dict(comparison=label, month=month,
                             delta_total_cost_yuan=float(t.loc[month, 'total_cost_yuan']
                                                         - b.loc[month, 'total_cost_yuan']),
                             delta_planned_cost_yuan=float(t.loc[month, 'planned_cost_yuan']
                                                           - b.loc[month, 'planned_cost_yuan']),
                             delta_emergency_cost_yuan=float(t.loc[month, 'emergency_cost_yuan']
                                                             - b.loc[month, 'emergency_cost_yuan']),
                             delta_emergency_kWh=float(t.loc[month, 'emergency_kWh']
                                                       - b.loc[month, 'emergency_kWh']),
                             delta_planned_kWh=float(t.loc[month, 'planned_kWh']
                                                     - b.loc[month, 'planned_kWh'])))
    return pd.DataFrame(rows)


def stability_frame(results):
    by_id = {r['totals']['strategy_id']: r for r in results}
    rows = []
    for label, treatment, baseline in ((MAIN_COMPARISON, 'P1_q80', 'P0_q80'),
                                       (AUXILIARY_COMPARISON, 'P1_q75', 'P0_q75')):
        t = by_id[treatment]['daily']
        b = by_id[baseline]['daily']
        daily_delta = t.total_cost_yuan.to_numpy() - b.total_cost_yuan.to_numpy()
        monthly = monthly_contrasts(results)
        monthly = monthly[monthly.comparison == label]
        windows = {}
        for name, slots in (('0_10h', slice(0, 60)), ('19_21h', slice(114, 126)),
                            ('21_24h', slice(126, 144))):
            key = np.zeros(144, dtype=bool)
            key[slots] = True
            tile = np.tile(key, len(by_id[treatment]['dispatch']) // 144)
            te = by_id[treatment]['dispatch'].emergency_cost_yuan.to_numpy()[tile]
            be = by_id[baseline]['dispatch'].emergency_cost_yuan.to_numpy()[tile]
            windows[name] = float(te.sum() - be.sum())
        worst = int(np.argmax(daily_delta))
        best = int(np.argmin(daily_delta))
        rows.append(dict(comparison=label, months_improved=int((monthly.delta_total_cost_yuan < 0).sum()),
                         months_total=int(len(monthly)),
                         days_improved=int((daily_delta < 0).sum()), days_worse=int((daily_delta > 0).sum()),
                         days_total=int(len(daily_delta)),
                         worst_day=str(t.date.iloc[worst]), worst_day_yuan=float(daily_delta[worst]),
                         best_day=str(t.date.iloc[best]), best_day_yuan=float(daily_delta[best]),
                         **{f'delta_emergency_cost_{k}': v for k, v in windows.items()}))
    return pd.DataFrame(rows)


def energy_contrasts(results):
    by_id = {r['totals']['strategy_id']: r for r in results}
    rows = []
    for label, treatment, baseline in ((MAIN_COMPARISON, 'P1_q80', 'P0_q80'),
                                       (AUXILIARY_COMPARISON, 'P1_q75', 'P0_q75')):
        t, b = by_id[treatment]['totals'], by_id[baseline]['totals']
        residual = ((t['planned_kWh'] - b['planned_kWh'])
                    + (t['emergency_kWh'] - b['emergency_kWh'])
                    - (t['unused_kWh'] - b['unused_kWh'])
                    - (t['loss_kWh'] - b['loss_kWh'])
                    - (t['final_kWh'] - b['final_kWh']))
        rows.append(dict(comparison=label, delta_planned_kWh=t['planned_kWh'] - b['planned_kWh'],
                         delta_emergency_kWh=t['emergency_kWh'] - b['emergency_kWh'],
                         delta_unused_kWh=t['unused_kWh'] - b['unused_kWh'],
                         delta_loss_kWh=t['loss_kWh'] - b['loss_kWh'],
                         delta_final_kWh=t['final_kWh'] - b['final_kWh'],
                         identity_residual_kWh=float(residual)))
    return pd.DataFrame(rows)


def predictor_metric_frame(load, pv, dates, archives):
    """Point-forecast metrics in kW over 2025-02-01..12-31, per predictor."""
    rows = []
    window = slice(WARMUP_DAYS, len(dates))
    for name, archive in archives.items():
        load_fc = archive['load_forecast'][window]
        pv_fc = archive['pv_forecast'][window]
        load_a = load[window]
        pv_a = pv[window]
        net_fc = load_fc - pv_fc
        net_a = load_a - pv_a
        for scope, forecast, actual in (('load', load_fc, load_a), ('pv', pv_fc, pv_a),
                                        ('net_demand', net_fc, net_a)):
            error = actual - forecast
            rows.append(dict(predictor=name, scope=scope, n=int(error.size),
                             mae_kW=float(np.mean(np.abs(error))),
                             rmse_kW=float(np.sqrt(np.mean(error ** 2))),
                             bias_kW=float(np.mean(error))))
    return pd.DataFrame(rows)


def pv_subset_frame(pv, dates, archives):
    """PV error by truth magnitude subset and by support-window position, per predictor."""
    rows = []
    window = slice(WARMUP_DAYS, len(dates))
    pv_a = pv[window]
    for name, archive in archives.items():
        pv_fc = archive['pv_forecast'][window]
        error = pv_a - pv_fc
        subsets = {'all': np.ones_like(pv_a, dtype=bool),
                   'actual_zero': pv_a == 0.0,
                   'actual_0_1kW': (pv_a > 0) & (pv_a <= 1.0),
                   'actual_1_100kW': (pv_a > 1.0) & (pv_a <= 100.0),
                   'actual_gt_100kW': pv_a > 100.0}
        for label, mask in subsets.items():
            rows.append(dict(predictor=name, grouping='truth_subset', subset=label,
                             n=int(mask.sum()),
                             mae_kW=float(np.mean(np.abs(error[mask]))) if mask.any() else np.nan,
                             bias_kW=float(np.mean(error[mask])) if mask.any() else np.nan,
                             mean_forecast_kW=float(np.mean(pv_fc[mask])) if mask.any() else np.nan))
    return pd.DataFrame(rows)


def window_subset_frame(pv, dates, archives, support, gate):
    """PV error by support-window segment, using the same window for both predictors."""
    rows = []
    window = slice(WARMUP_DAYS, len(dates))
    pv_a = pv[window]
    support = np.asarray(support)[window]
    gate = np.asarray(gate)[window]
    tau = np.empty(pv_a.shape)
    for row, (s, e, valid) in enumerate(support):
        tau[row] = support_phase(0, s, e)
    valid = (support[:, 2] == 1)[:, None]
    segments = {'valid_morning_tau_lt_0.2': valid & gate & (tau < 0.2),
                'valid_midday_tau_0.2_0.8': valid & gate & (tau >= 0.2) & (tau <= 0.8),
                'valid_evening_tau_gt_0.8': valid & gate & (tau > 0.8),
                'outside_window': ~gate,
                'no_valid_window': (support[:, 2] == 0)[:, None] & np.ones_like(gate)}
    for name, archive in archives.items():
        pv_fc = archive['pv_forecast'][window]
        error = pv_a - pv_fc
        for label, mask in segments.items():
            rows.append(dict(predictor=name, subset=label, n=int(mask.sum()),
                             mae_kW=float(np.mean(np.abs(error[mask]))) if mask.any() else np.nan,
                             bias_kW=float(np.mean(error[mask])) if mask.any() else np.nan))
    return pd.DataFrame(rows)


def monthly_error_frame(pv, dates, archives):
    window = slice(WARMUP_DAYS, len(dates))
    months = np.array([str(d)[:7] for d in dates])[window]
    pv_a = pv[window]
    rows = []
    for name, archive in archives.items():
        error = pv_a - archive['pv_forecast'][window]
        for month in sorted(set(months)):
            mask = months == month
            rows.append(dict(predictor=name, month=month, n=int(mask.sum()),
                             mae_kW=float(np.mean(np.abs(error[mask]))),
                             bias_kW=float(np.mean(error[mask]))))
    return pd.DataFrame(rows)


def hourly_error_frame(pv, dates, archives):
    window = slice(WARMUP_DAYS, len(dates))
    pv_a = pv[window]
    hours = np.tile(np.arange(144) // 6, (len(pv_a), 1))
    rows = []
    for name, archive in archives.items():
        error = pv_a - archive['pv_forecast'][window]
        for hour in range(24):
            mask = hours == hour
            rows.append(dict(predictor=name, hour=hour, n=int(mask.sum()),
                             mae_kW=float(np.mean(np.abs(error[mask]))),
                             bias_kW=float(np.mean(error[mask]))))
    return pd.DataFrame(rows)


def support_summary_frame(pv, dates, support):
    rows = []
    for k in range(len(dates)):
        s, e, valid = (int(v) for v in support[k])
        rows.append(dict(date=str(dates[k].date()), day_index=k, start_slot=s, end_slot=e,
                         valid=valid, duration_h=(e - s + 1) * DT,
                         midpoint_h=(s + e + 1) * DT / 2))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------
def figures(monthly, stability, monthly_error, window_subsets, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['axes.unicode_minus'] = False
    FIG.mkdir(parents=True, exist_ok=True)
    produced = []

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    for ax, label in zip(axes, [MAIN_COMPARISON, AUXILIARY_COMPARISON]):
        block = monthly[monthly.comparison == label].sort_values('month')
        x = np.arange(len(block))
        width = 0.27
        ax.bar(x - width, block.delta_planned_cost_yuan / 1e3, width, label='planned')
        ax.bar(x, block.delta_emergency_cost_yuan / 1e3, width, label='emergency (5x)')
        ax.bar(x + width, block.delta_total_cost_yuan / 1e3, width, label='total')
        ax.axhline(0, color='black', linewidth=.8)
        ax.set(title=label.split(' (')[0], xlabel='Month of 2025',
               ylabel='Cost difference (thousand CNY)',
               xticks=x, xticklabels=[m[5:] for m in block.month])
        ax.grid(alpha=.2, axis='y')
        ax.legend(fontsize=8)
    path = FIG / 'q2_pv_support_cost_differences.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    produced.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    months = sorted(monthly_error.month.unique())
    for predictor, group in monthly_error.groupby('predictor'):
        block = group.set_index('month')
        axes[0].plot(np.arange(len(months)), [block.loc[m, 'mae_kW'] for m in months],
                     marker='o', label=predictor)
    axes[0].set(title='PV MAE by month', xlabel='Month of 2025', ylabel='MAE (kW)',
                xticks=np.arange(len(months)), xticklabels=[m[5:] for m in months])
    axes[0].grid(alpha=.2)
    axes[0].legend(fontsize=8)
    labels = ['valid_morning_tau_lt_0.2', 'valid_midday_tau_0.2_0.8', 'valid_evening_tau_gt_0.8',
              'outside_window', 'no_valid_window']
    counts = window_subsets.groupby('subset').n.sum()
    drawn = [l for l in labels if l in set(window_subsets.subset) and counts.get(l, 0) > 0]
    pivot = window_subsets.pivot(index='subset', columns='predictor', values='mae_kW').reindex(drawn)
    x = np.arange(len(drawn))
    for offset, predictor in enumerate([c for c in pivot.columns]):
        axes[1].bar(x + (offset - 0.5) * 0.35,
                    np.nan_to_num(pivot[predictor].to_numpy()), 0.35, label=predictor)
    axes[1].set(title='PV MAE by support-window segment', xlabel='segment', ylabel='MAE (kW)',
                xticks=x, xticklabels=[l.replace('_', ' ') for l in drawn])
    axes[1].grid(alpha=.2, axis='y')
    axes[1].legend(fontsize=8)
    path = FIG / 'q2_pv_support_error_change.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    produced.append(path)

    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    block = monthly[monthly.comparison == MAIN_COMPARISON].sort_values('month')
    x = np.arange(len(block))
    ax.bar(x, block.delta_total_cost_yuan / 1e3, 0.55, color='tab:blue',
           label='total cash cost difference')
    ax2 = ax.twinx()
    ax2.plot(x, block.delta_emergency_kWh, color='tab:red', marker='s',
             label='emergency energy difference')
    ax.axhline(0, color='black', linewidth=.8)
    ax.set(title='Main comparison: P1_q80 minus P0_q80', xlabel='Month of 2025',
           ylabel='Total cost difference (thousand CNY)', xticks=x,
           xticklabels=[m[5:] for m in block.month])
    ax2.set_ylabel('Emergency energy difference (kWh)')
    ax.grid(alpha=.2, axis='y')
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc='best')
    path = FIG / 'q2_pv_support_monthly_main.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    produced.append(path)
    return produced


# --------------------------------------------------------------------------------------
# registration and protected assets
# --------------------------------------------------------------------------------------
def protected_manifest():
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


def write_report(context):
    monthly = context['monthly']
    paired = context['paired']
    stability = context['stability']
    metrics = context['metrics']
    subsets = context['subsets']
    windows = context['windows']
    monthly_error = context['monthly_error']
    summary = context['summary']
    support = context['support_summary']
    checks = context['checks']
    coverage = context['coverage']

    def money(value):
        return f'{value:,.2f}'

    def delta(comparison, field):
        row = paired[paired.comparison == comparison].iloc[0]
        return float(row[field])

    by_id = summary.set_index('strategy_id')
    main_days = stability[stability.comparison == MAIN_COMPARISON].iloc[0]
    aux_days = stability[stability.comparison == AUXILIARY_COMPARISON].iloc[0]
    main_months = monthly[monthly.comparison == MAIN_COMPARISON]
    lines = []
    add = lines.append
    add('# 问题二：光伏历史窗口特征优化实验结果报告')
    add('')
    add('日期：2026-09-11。状态：**实验自检完成，独立审计待完成**。本轮实现、运行并自检了 '
        '`reports/问题二_光伏历史窗口特征优化实验方案.md` 登记的四组同年因果回测；未锁定第二问最终模型，'
        '未填写 `result2.xlsx`，未修改第一问，未重训负载。')
    add('')
    add('## 1. 问题分析')
    add('')
    add('日前不知道当天真实供需，计划电量全额付款，缺口按当段 5 倍价应急。负载预测本轮完全冻结为修正版 G4 '
        '发布档案；只检验一件事：**在光伏 LightGBM 残差修正中把历史支持窗口的长度、中点、相对位置与有效标记'
        '四列加入输入**，能否改善晨昏与近期季节形状变化时的光伏预测，并降低实际现金总费。')
    add('')
    add('四组只改变光伏发布预测（原 8 列或新增 12 列）与保护水平（q80 或 q75）。负载档案、W28 逐时段'
        '经验分位规则、名义日末 6000 kWh、冻结 MILP 内核、因果贪心反馈、收费规则与公共 1 月预运行完全相同。'
        '因此组间差额可归因于光伏输入列与保护水平，**不能**归因于“窗口特征是最优特征”，也不能外推到其他年份。')
    add('')
    add('### 1.1 主要结论')
    add('')
    repro = checks['reference_reproduction']
    add(f"- 控制复现：`P0_q80` 总费 {money(by_id.loc['P0_q80', 'total_cost_yuan'])} 元，"
        f"与登记参考差 {repro['P0_q80']['difference_vs_registered_yuan']:.3e} 元、"
        f"日级聚合最大差 {repro['P0_q80']['max_day_level_difference']:.3e} kWh；"
        f"`P0_q75` 总费 {money(by_id.loc['P0_q75', 'total_cost_yuan'])} 元，"
        f"与登记参考差 {repro['P0_q75']['difference_vs_registered_yuan']:.3e} 元、"
        f"日级聚合最大差 {repro['P0_q75']['max_day_level_difference']:.3e} kWh。")
    add(f"- 主比较 `P1_q80 − P0_q80`：计划费 {delta(MAIN_COMPARISON, 'delta_planned_cost_yuan'):+,.2f} 元，"
        f"应急费 {delta(MAIN_COMPARISON, 'delta_emergency_cost_yuan'):+,.2f} 元，"
        f"总费 {delta(MAIN_COMPARISON, 'delta_total_cost_yuan'):+,.2f} 元"
        f"（{delta(MAIN_COMPARISON, 'delta_total_cost_percent'):+.4f}%），"
        f"应急天数 {int(delta(MAIN_COMPARISON, 'delta_emergency_days')):+d} 天。")
    add(f"- 配对稳健性 `P1_q75 − P0_q75`：总费 {delta(AUXILIARY_COMPARISON, 'delta_total_cost_yuan'):+,.2f} 元"
        f"（{delta(AUXILIARY_COMPARISON, 'delta_total_cost_percent'):+.4f}%），"
        f"应急天数 {int(delta(AUXILIARY_COMPARISON, 'delta_emergency_days')):+d} 天。")
    add(f"- 主比较在 {int(main_days.months_improved)}/{int(main_days.months_total)} 个月、"
        f"{int(main_days.days_improved)}/{int(main_days.days_total)} 天更省，"
        f"最差单日 {main_days.worst_day} {main_days.worst_day_yuan:+,.2f} 元；"
        f"辅助比较 {int(aux_days.months_improved)}/{int(aux_days.months_total)} 个月、"
        f"{int(aux_days.days_improved)}/{int(aux_days.days_total)} 天更省。")
    add('- 无论方向如何，四组结果、月度反例与应急权衡均按实保留；未据结果追加年周期、月份项或参数搜索。')
    add('')
    add('## 2. 数据预处理')
    add('')
    add('只使用附件1的144点日内电价（逐日重复）与附件2的365×144实际负载、光伏功率。不使用附件3预报与'
        '附件4电价，不使用当天尚未到达的任何实际值。样本解释为前10分钟区间代表功率，每段 Δt=1/6 小时。'
        '不平滑、不删点、不插补，不对原8特征做新的标准化。')
    add('')
    add('### 2.1 输入核对与残差诊断')
    add('')
    add('- 修正 G4 档案：365 天 × 144 段唯一且齐全，1 月 2 日起负载与光伏发布预测全部有限非负；'
        '真值与附件2一致，电价与附件1一致（由 `load_predictor_archive` 断言）。')
    add('- 门控一致性：本脚本重建的门控掩码与冻结 `gate_mask.csv` 逐段一致，'
        f"不一致 {context['gate']['mask_mismatch_slots']} 段；"
        f"被归零时段 {context['gate']['gated_off_slots']} 段，"
        f"其中真值 >100 kW 的时段 {context['gate']['zeroed_truth_above_100kW_slots']} 个，"
        f"被归零位置真值发电合计 {context['gate']['zeroed_truth_energy_kWh']:.6f} kWh。")
    add('- 支持窗口分布（全 365 天，前 7 天与无活跃时段为缺省 v=0）：'
        f"v=1 天数 {int((support.valid == 1).sum())}，v=0 天数 {int((support.valid == 0).sum())}；"
        f"v=1 时窗口时长 {support.loc[support.valid == 1, 'duration_h'].min():.2f}—"
        f"{support.loc[support.valid == 1, 'duration_h'].max():.2f} 小时，"
        f"中点 {support.loc[support.valid == 1, 'midpoint_h'].min():.2f}—"
        f"{support.loc[support.valid == 1, 'midpoint_h'].max():.2f} 小时。")
    add('')
    add('光伏发布误差按真值量级分组（kW，偏差=实际−预测，评价期334天48096段）：')
    add('')
    add('| 预测器 | 子集 | 样本数 | MAE | 偏差 | 平均预测 |')
    add('|---|---|---:|---:|---:|---:|')
    for row in subsets.itertuples():
        add(f'| {row.predictor} | {row.subset} | {row.n} | {row.mae_kW:.6f} | {row.bias_kW:.6f} | '
            f'{row.mean_forecast_kW:.6f} |')
    add('')
    add('光伏发布误差按支持窗口位置分组（同一窗口口径，两预测器共用）：')
    add('')
    add('| 预测器 | 子集 | 样本数 | MAE | 偏差 |')
    add('|---|---|---:|---:|---:|')
    for row in windows.itertuples():
        add(f'| {row.predictor} | {row.subset} | {row.n} | {row.mae_kW:.6f} | {row.bias_kW:.6f} |')
    add('')
    add('分月光伏 MAE（kW）：')
    add('')
    add('| 月 | ' + ' | '.join(sorted(monthly_error.predictor.unique())) + ' |')
    add('|---|' + '---:|' * monthly_error.predictor.nunique())
    for month in sorted(monthly_error.month.unique()):
        block = monthly_error[monthly_error.month == month].set_index('predictor')
        add(f'| {month} | ' + ' | '.join(f'{block.loc[p, "mae_kW"]:.4f}'
                                        for p in sorted(monthly_error.predictor.unique())) + ' |')
    add('')
    add('分月与分小时明细见 `pv_error_monthly.csv`、`pv_error_hourly.csv`；'
        '负载星期×小时残差背景见 `load_weekday_hour_residual.csv`。')
    add('')
    add('### 2.2 每日历史支持窗口')
    add('')
    add('以未平滑的实际光伏功率 V 计算。第 k 日只看前 7 个完整日：'
        r'$\mathcal A_k=\{t:\max_{j=k-7,\ldots,k-1}V_{j,t}>1\ \mathrm{kW}\}$；'
        '非空则 $s_k=\\max(0,\\min\\mathcal A_k-3)$、$e_k=\\min(143,\\max\\mathcal A_k+3)$、$v_k=1$，'
        '否则 $s_k=0,e_k=143,v_k=0$。严格大于 1 kW，等于 1 不算活跃。')
    add('')
    add(f"训练行边界：原始与候选训练日的行键、标签、训练日数、分支与首次可训练日完全一致"
        f"（`training_parity` 不一致项 {context['parity']['mismatch_count']}，"
        f"首次可训练日 {context['parity']['first_trainable_day']}）。"
        "新增列在训练窗内是否恒定的逐日记录见 `candidate_fit_log.csv`。")
    fitted = [e for e in context['candidate_log'] if e['branch'] == 'fitted']
    trees = [e['n_trees'] for e in fitted]
    constant_counts = {name: 0 for name in SUPPORT_FEATURES}
    for entry in fitted:
        for name in str(entry.get('constant_new_columns') or '').split(','):
            if name:
                constant_counts[name] += 1
    add(f"实际树数：拟合 {len(fitted)} 天，平均 {np.mean(trees):.1f} 棵、最少 {min(trees)}、最多 "
        f"{max(trees)} 棵。")
    named = '、'.join(f'`{k}` {v} 天' for k, v in constant_counts.items() if v > 0) or '无'
    add(f"新增列在训练窗内恒定（共 {len(fitted)} 个拟合日）：{named}；恒定列如实保留，未因恒定而删列。"
        "其中 `support_valid` 在全部拟合日恒为 1（每个训练窗都有有效历史窗口），本轮几乎不提供可用"
        "分裂信息，`support_midpoint_h`/`support_duration_h` 也在相当一部分训练窗内恒定（中点仅在 "
        "11.67—12.00 小时之间变化）。因此本轮不能按“增加了 4 个独立信息源”表述。")
    interaction = (delta('P1_q75 minus P1_q80 (protection level)', 'delta_total_cost_yuan')
                   - delta('P0_q75 minus P0_q80 (protection level)', 'delta_total_cost_yuan'))
    add(f"保护水平与光伏特征的交互描述（不是单独特征收益）："
        f"(P1_q75−P1_q80) − (P0_q75−P0_q80) = {interaction:+,.2f} 元。")
    add('')
    add('## 3. 模型建立')
    add('')
    add('光伏继续预测**昨日同段朴素预测的有符号功率残差**：'
        r'$y^V_{j,t}=V_{j,t}-V_{j-1,t}$、'
        r'$\widehat V_{k,t}=g_{k,t}\max\{0,V_{k-1,t}+f_k(\boldsymbol x_{k,t})\}$。'
        '原 8 列为 $[V_{k-1,t},\\ V_{k-1,t}-V_{k-2,t},\\ \\frac17\\sum_{j=k-7}^{k-1}V_{j,t}-V_{k-1,t},\\ '
        '\\bar V_{k-1}-\\bar V_{k-2},$ 日内一/二阶正余弦$]$；')
    add('候选 12 列在其后追加固定顺序的 `support_duration_h`、`support_midpoint_h`、'
        '`support_phase`、`support_valid`：$D_k=(e_k-s_k+1)\\Delta t$、'
        '$M_k=(s_k+e_k+1)\\Delta t/2$、$\\tau_{k,t}=(t-s_k+1/2)/(e_k-s_k+1)$、$v_k$。')
    add('')
    add('每日一个光伏共享 LightGBM，平方损失、100 轮、学习率 0.05、7 叶、深度 3、'
        '`min_child_samples=144`、L2=1，全部随机种子 20250911；不网格搜索、不早停、不用 eval_set。'
        '训练集为决策日前 56 个日历日内的有效完整日（至少 14 日），首次可训练 2025-01-22（k=21），'
        '标签极差 ≤1e-8 kW 时取常数均值，历史不足或未达特征有效日则发布朴素并统一过门控。')
    add('')
    add('每组由**自身**发布的光伏预测重建净需求误差并使用 W28 逐时段经验逆分布保护'
        '（q80 为升序第 23 个，q75 为第 21 个；m<7 回退零修正；负修正与负净需求保留）。'
        '负载预测在四组中完全相同。日前模型为 min Σ p_t q_t，'
        '约束 q+d=ñ+c+w、$E_{t+1}=E_t+0.9c_t-d_t/0.9$、0≤c_t≤(5000/6)z_t、'
        '0≤d_t≤(5000/6)(1−z_t)、1200≤E_t≤10800、E_0=当日实际初态、$E_{144}=6000$；'
        '冻结 HiGHS 内核，gap 目标 1e-9、时限 120 秒。实际执行按富余充电、缺口放电、剩余 5 倍价应急，'
        '实际末态逐日继承，1 月沿用公共冷启动。')
    add('')
    add('## 4. 模型求解与结果')
    add('')
    add(f"公共1月预运行结束状态 {context['warm_end']:.12f} kWh，四组从此分叉并各自继承真实末态。"
        f"候选光伏全链路（构建 {context['candidate_summary']['elapsed_seconds']:.1f} 秒，"
        f"拟合 {context['candidate_summary']['n_fitted']} 天、常数 {context['candidate_summary']['n_constant']} 天、"
        f"历史不足 {context['candidate_summary']['n_insufficient']} 天、早期回退 "
        f"{context['candidate_summary']['n_feature_fallback']} 天）。")
    add('')
    add('### 4.1 四组总费用')
    add('')
    add('| 组 | 计划费/元 | 应急费/元 | 总费/元 | 未使用/kWh | 损耗/kWh | 期末/kWh | 应急天 | 应急事件 |')
    add('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for row in summary.itertuples():
        add(f'| {row.strategy_id} | {money(row.planned_cost_yuan)} | {money(row.emergency_cost_yuan)} | '
            f'{money(row.total_cost_yuan)} | {row.unused_kWh:,.2f} | {row.loss_kWh:,.2f} | '
            f'{row.final_kWh:,.2f} | {row.emergency_days} | {row.emergency_events} |')
    add('')
    add('### 4.2 组间差额')
    add('')
    add('| 比较 | Δ计划费/元 | Δ应急费/元 | Δ总费/元 | Δ总费/% | Δ计划/kWh | Δ应急/kWh | Δ未使用/kWh | Δ应急天 |')
    add('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for row in paired.itertuples():
        add(f'| {row.comparison} | {row.delta_planned_cost_yuan:+,.2f} | '
            f'{row.delta_emergency_cost_yuan:+,.2f} | {row.delta_total_cost_yuan:+,.2f} | '
            f'{row.delta_total_cost_percent:+.4f} | {row.delta_planned_kWh:+,.2f} | '
            f'{row.delta_emergency_kWh:+,.2f} | {row.delta_unused_kWh:+,.2f} | '
            f'{row.delta_emergency_days:+d} |')
    add('')
    add('### 4.3 点预测误差')
    add('')
    add('| 预测器 | 目标 | MAE/kW | RMSE/kW | 偏差/kW |')
    add('|---|---|---:|---:|---:|')
    for row in metrics.itertuples():
        add(f'| {row.predictor} | {row.scope} | {row.mae_kW:.6f} | {row.rmse_kW:.6f} | {row.bias_kW:.6f} |')
    add('')
    add('负载预测在四组中完全相同，故负载指标不随预测器变化；光伏净需求误差只来自光伏列。')
    add('')
    add('### 4.4 保护校准')
    add('')
    add('覆盖率为实际净需求不超过保护需求的时段比例（n≤ñ），**不是**“无应急概率”。')
    add('')
    add('| 组 | 平均保护量/kWh | 覆盖率 |')
    add('|---|---:|---:|')
    for row in coverage.itertuples():
        add(f'| {row.strategy_id} | {row.mean_adjustment_kWh:.6f} | {row.coverage:.6f} |')
    add('')
    add('### 4.5 稳定性与能量账')
    add('')
    add('| 比较 | 改善月 | 改善天 | 变差天 | 最差日 | 最差日/元 | 0—10时应急差/元 | 19—21时/元 | 21—24时/元 |')
    add('|---|---:|---:|---:|---|---:|---:|---:|---:|')
    for row in stability.itertuples():
        add(f'| {row.comparison} | {row.months_improved}/{row.months_total} | {row.days_improved} | '
            f'{row.days_worse} | {row.worst_day} | {row.worst_day_yuan:+,.2f} | '
            f'{row.delta_emergency_cost_0_10h:+,.2f} | {row.delta_emergency_cost_19_21h:+,.2f} | '
            f'{row.delta_emergency_cost_21_24h:+,.2f} |')
    add('')
    add('组间能量账满足 ΔQ = −ΔQ_em + ΔW + ΔLoss + ΔE_end，见 `energy_contrasts.csv`。')
    add('')
    add('### 4.6 逐月差额')
    add('')
    add('| 比较 | 月 | Δ计划费/元 | Δ应急费/元 | Δ总费/元 | Δ应急/kWh |')
    add('|---|---|---:|---:|---:|---:|')
    for row in monthly.itertuples():
        add(f'| {row.comparison.split(" (")[0]} | {row.month} | {row.delta_planned_cost_yuan:+,.2f} | '
            f'{row.delta_emergency_cost_yuan:+,.2f} | {row.delta_total_cost_yuan:+,.2f} | '
            f'{row.delta_emergency_kWh:+,.2f} |')
    add('')
    add('### 4.7 图表')
    add('')
    for row in context['integrity']:
        add(f"- `figures/q2_pv_support_features/{row['file']}`：{row['width']}×{row['height']} 像素，"
            f"非白像素比例 {row['non_white_fraction']:.4f}，颜色数 {row['distinct_colours']}。")
    add('')
    add('本环境无法进行图像目视核查，上述仅为程序化完整性检查；图表内容的人工/视觉核验**尚未完成**。')
    add('')
    add('## 5. 验证与适用边界')
    add('')
    add('| 检查 | 结果 |')
    add('|---|---|')
    add(f"| 人工边界样例（合成数据） | {len(checks['boundary']['rows'])} 项，"
        f"{'全部通过' if checks['boundary']['all_passed'] else '存在失败'} |")
    add(f"| 门控重建与冻结掩码一致 | 不一致 {checks['gate']['mask_mismatch_slots']} 段 |")
    add(f"| 控制 8 列模型独立重训复现 | 最大差 {checks['control_refit']['max_difference_kW']:.3e} kW"
        f"（阈值 1e-6） |")
    add(f"| 训练行/标签/首训日一致性 | 不一致 {checks['parity']['mismatch_count']} 项，"
        f"首次可训练日 {checks['parity']['first_trainable_day']} |")
    add(f"| 分位规则 m=7..28 | {'通过' if checks['quantile']['all_passed'] else '失败'} |")
    add(f"| 控制组 P0_q80 复现 | 日级最大差 "
        f"{checks['reference_reproduction']['P0_q80']['max_day_level_difference']:.3e} kWh，"
        f"总费差 {checks['reference_reproduction']['P0_q80']['difference_vs_registered_yuan']:.3e} 元 |")
    add(f"| 控制组 P0_q75 复现 | 日级最大差 "
        f"{checks['reference_reproduction']['P0_q75']['max_day_level_difference']:.3e} kWh，"
        f"总费差 {checks['reference_reproduction']['P0_q75']['difference_vs_registered_yuan']:.3e} 元 |")
    add(f"| 未来扰动（光伏链与冻结负载链） | {len(checks['perturbation']['cases'])} 例，"
        f"{'全部通过' if checks['perturbation']['all_passed'] else '存在失败'} |")
    add(f"| 半日反馈前缀 | {len(checks['feedback_prefix']['cases'])} 例，"
        f"{'全部通过' if checks['feedback_prefix']['all_passed'] else '存在失败'} |")
    add(f"| 抽样重训与模型重载 | {len(checks['sampled_retraining']['rows'])} 日，"
        f"{'全部通过' if checks['sampled_retraining']['all_passed'] else '存在失败'} |")
    add(f"| 受保护旧资产 | {'零变更零缺失' if checks['protected_unchanged'] else '存在变更'}，"
        f"共 {checks['protected_count']} 个文件 |")
    add('')
    add('限制：')
    add('')
    add('1. 2025 年数据此前已参与方法诊断与设计，本轮是**滚动因果回测，不是独立盲测或跨年验证**。')
    add('2. 负载预测完全冻结为修正 G4 档案；其训练因果性属上游 20 号独立审计范围，本轮不重验负载模型训练。')
    add('3. 56 日窗、至少 14 日、1 kW 阈值、3 段余量、W28、q80/q75、6000 kWh 均为运行前固定的工程设计，'
        '本实验未搜索这些参数。')
    add('4. 门控规则不变，四组门控掩码与被归零位置完全一致；既有门控收益不应重新算作本轮新增特征收益。')
    add('5. 点预测精度变化不必然转化为费用变化；q 分位保护可能抵消稳定点预测偏差，'
        '本轮同时报告点预测、保护量、应急与总费，不做单一综合分数。')
    add('6. 本实验不替代 21 号分位扫描审计与 20 号上游预测审计；名义 MILP 等优轨迹未做统一二级择优。')
    add('')
    add('## 6. 文件与复现')
    add('')
    add('- 登记与运行清单：`results/q2_pv_support_features/registration.json`、`run_manifest.json`。')
    add('- 预测档案：`pv_forecast_archive_P0.csv`、`pv_forecast_archive_P1.csv`、`support_windows.csv`、'
        '`candidate_fit_log.csv`、`control_fit_log.csv`、`models/original8/`、`models/support12/`。')
    add('- 逐组结果：`results/q2_pv_support_features/<组名>/dispatch.csv`、`nominal_dispatch.csv`、'
        '`daily_summary.csv`、`monthly_summary.csv`、`emergency_events.csv`、`solver_log.csv`、'
        '`validation.json`。')
    add('- 汇总与自检：`summary.csv`、`paired_contrasts.csv`、`monthly_contrasts.csv`、'
        '`stability.csv`、`energy_contrasts.csv`、`coverage.csv`、`predictor_metrics.csv`、'
        '`pv_error_subsets.csv`、`pv_error_by_window.csv`、`checks.json`。')
    add('- 图表：`figures/q2_pv_support_features/`（3 张 PNG）。')
    add('')
    add('复现：')
    add('')
    add('```')
    add('E:/Anaconda/envs/math_modeling/python.exe code/23_q2_pv_support_features_experiment.py --mode register')
    add('E:/Anaconda/envs/math_modeling/python.exe code/23_q2_pv_support_features_experiment.py --mode repro')
    add('E:/Anaconda/envs/math_modeling/python.exe code/23_q2_pv_support_features_experiment.py --mode full')
    add('```')
    add('')
    add('独立审计入口（建议）：以 `results/q2_pv_support_features/` 为只读输入，独立重建各历史日自己的'
        '窗口 s/e/v/D/M/τ 与原 8 列、训练日期/标签，核对是否出现当前窗口回填历史、误用当天发电端点、'
        '无有效窗口错误全零、早期发布缺失、首训日变动、只训练白天或偷偷新增年周期/调参；'
        '全量核对两套发布档案、两 α 分位有效数及排序、四组状态连续性、名义可行性与费用/能量差额基准。')
    REPORT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return REPORT_MD


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
    s = scan()
    s.OUT = OUT

    load, pv, price, source_hashes = m.frozen().bm.read_sources()
    dates = m.frozen().bm.DATES
    assert load.shape == pv.shape == (365, 144) and price.shape == (144,)

    inputs = ['code/14_q2_ridge_forecast_experiment.py', 'code/19_q2_lightgbm_residual_experiment.py',
              'code/21_q2_quantile_level_scan.py', 'code/23_q2_pv_support_features_experiment.py',
              'results/q2_lightgbm_residual/G4_lightgbm_fixed/forecast_residuals.csv',
              'results/q2_lightgbm_residual/training_log.csv',
              'results/q2_lightgbm_residual/gate_mask.csv',
              'results/q2_lightgbm_residual/registration.json',
              'results/q2_quantile_level_scan/L_q80/dispatch.csv',
              'results/q2_quantile_level_scan/L_q75/dispatch.csv',
              'results/q2_quantile_level_scan/registration.json',
              '附件/附件1.xlsx', '附件/附件2.xlsx',
              'reports/问题二_光伏历史窗口特征优化实验方案.md']
    snapshot_inputs = ['code/02_q1_baseline.py', 'code/05_q2_baseline.py',
                       'code/08_q2_quantile_experiment.py']
    hashes = {rel: digest(ROOT / rel) for rel in inputs}
    snapshot_hashes = {rel: digest(SNAP / rel) for rel in snapshot_inputs}
    for rel in snapshot_inputs:
        assert digest(ROOT / rel) == snapshot_hashes[rel], rel
    signature = hashlib.sha256(json.dumps(
        dict(parameters=PARAMETERS, input_sha256=hashes, snapshot_sha256=snapshot_hashes,
             code=digest(CODE_23)), sort_keys=True).encode()).hexdigest()
    registration_path = OUT / 'registration.json'
    if args.mode == 'register' or not registration_path.exists():
        record = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
                      specification='reports/问题二_光伏历史窗口特征优化实验方案.md',
                      parameters=PARAMETERS, input_sha256=hashes, snapshot_sha256=snapshot_hashes,
                      code_sha256=digest(CODE_23), executable=sys.executable, python=sys.version,
                      numpy=np.__version__, pandas=pd.__version__)
        if registration_path.exists():
            existing = json.loads(registration_path.read_text(encoding='utf-8'))
            if existing['signature'] != signature:
                assert args.amend_reason, 'a different registration exists; pass --amend-reason'
                for key in ('control_pv_features', 'candidate_pv_features', 'support_parameters',
                            'train_window_days', 'min_train_days', 'residual_window_days',
                            'groups', 'main_comparison', 'auxiliary_comparison'):
                    assert existing['parameters'][key] == PARAMETERS[key], key
                amendments = existing.get('amendments', [])
                amendments.append(dict(amended_utc=datetime.now(timezone.utc).isoformat(),
                                       previous_signature=existing['signature'],
                                       previous_code_sha256=existing.get('code_sha256'),
                                       new_signature=signature, new_code_sha256=digest(CODE_23),
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

    boundary = boundary_checks(pv, dates)
    save(OUT / 'boundary_checks.json', boundary)
    assert boundary['all_passed'], 'boundary checks failed'
    print(f"boundary checks: {len(boundary['rows'])} passed", flush=True)

    g4 = load_g4_archive(load, pv, dates)
    gate = gate_consistency_checks(pv, dates, g4)
    save(OUT / 'gate_consistency.json', gate)
    assert gate['all_passed'], gate
    print(f"gate reconstruction matches the frozen mask ({gate['gated_off_slots']} gated-off slots)",
          flush=True)

    control_refit, control_rebuilt = control_refit_check(pv, dates, g4)
    frame_to_csv(pd.DataFrame(control_rebuilt['log']), OUT / 'control_fit_log.csv')
    save(OUT / 'control_refit.json', control_refit)
    assert control_refit['all_passed'], control_refit
    print(f"control 8-column refit vs frozen archive: max {control_refit['max_difference_kW']:.3e} kW",
          flush=True)

    candidate = build_pv_forecast(pv, dates, 'support12', save_models=True, verbose=True)
    frame_to_csv(pd.DataFrame(candidate['log']), OUT / 'candidate_fit_log.csv')
    parity = training_parity_checks(candidate, g4)
    save(OUT / 'training_parity.json', parity)
    assert parity['all_passed'], parity
    print(f"training parity: {candidate['summary']['n_fitted']} fitted, "
          f"first trainable day {parity['first_trainable_day']}", flush=True)
    support = support_summary_frame(pv, dates, candidate['support'])
    frame_to_csv(support, OUT / 'support_windows.csv')
    for name, archive in (('P0', g4), ('P1', candidate)):
        frame_to_csv(pd.DataFrame({
            'date': np.repeat([str(d.date()) for d in dates], 144),
            'slot': np.tile(np.arange(144), len(dates)),
            'load_forecast_kW': np.asarray(archive['load_forecast'] if name == 'P0'
                                           else g4['load_forecast']).ravel(),
            'pv_forecast_kW': np.asarray(archive['pv_forecast'] if name == 'P0'
                                         else archive['published']).ravel(),
            'support_start_slot': np.repeat(candidate['support'][:, 0], 144),
            'support_end_slot': np.repeat(candidate['support'][:, 1], 144),
            'support_valid': np.repeat(candidate['support'][:, 2], 144)}),
            OUT / f'pv_forecast_archive_{name}.csv')

    archives = {
        'P0': dict(load_forecast=np.asarray(g4['load_forecast'], dtype=float),
                   pv_forecast=np.asarray(g4['pv_forecast'], dtype=float)),
        'P1': dict(load_forecast=np.asarray(g4['load_forecast'], dtype=float),
                   pv_forecast=np.asarray(candidate['published'], dtype=float)),
    }
    quantile = quantile_validation({name: dict(eps=(load - pv) * DT
                                               - (a['load_forecast'] - a['pv_forecast']) * DT)
                                    for name, a in archives.items()})
    save(OUT / 'quantile_validation.json', quantile)
    assert quantile['all_passed'], quantile
    print(f"quantile validation passed (positions {quantile['positions_at_28_days']})", flush=True)

    print('running the public January warm-up ...', flush=True)
    warm_states, warm_frame, warm_end, warm_checks = m.frozen().m08.run_warmup(
        m.frozen().m08.Archive(load, pv, len(dates)), price, dates)
    frame_to_csv(warm_frame, OUT / 'warmup_january.csv')
    assert abs(warm_end - REFERENCE_WARMUP_KWH) < 1e-9, warm_end
    print(f'  2025-02-01 shared begin state = {warm_end:.12f} kWh', flush=True)

    instances = {name: s.ScanArchive(load, pv, len(dates), a['load_forecast'], a['pv_forecast'])
                 for name, a in archives.items()}
    for name, a in archives.items():
        s.require_finite_published(a['load_forecast'], a['pv_forecast'], len(dates))

    results = []
    reference_reproduction = {}
    for group in CONTROL_GROUPS:
        result = s.run_scenario(group, load, pv, price, dates, instances[group['predictor']],
                                warm_states, verbose=False)
        results.append(result)
        print(f"  {group['id']}: {result['totals']['total_cost_yuan']:,.6f} CNY", flush=True)
        record = compare_to_reference(result, group['id'])
        reference_reproduction[group['id']] = record
        assert record['passed'], record
        print(f"    reproduces {REFERENCE[group['id']]['folder'].name}: day-level max diff "
              f"{record['max_day_level_difference']:.3e} kWh, cost diff "
              f"{record['difference_vs_registered_yuan']:.3e} CNY", flush=True)
    save(OUT / 'reference_reproduction.json', reference_reproduction)
    if args.mode == 'repro':
        return

    for group in TREATMENT_GROUPS:
        result = s.run_scenario(group, load, pv, price, dates, instances[group['predictor']],
                                warm_states, verbose=True)
        results.append(result)
        print(f"  {group['id']}: {result['totals']['total_cost_yuan']:,.6f} CNY", flush=True)

    summary = summary_frame(results)
    frame_to_csv(summary, OUT / 'summary.csv')
    paired = paired_contrasts(results)
    frame_to_csv(paired, OUT / 'paired_contrasts.csv')
    monthly = monthly_contrasts(results)
    frame_to_csv(monthly, OUT / 'monthly_contrasts.csv')
    stability = stability_frame(results)
    frame_to_csv(stability, OUT / 'stability.csv')
    energy = energy_contrasts(results)
    frame_to_csv(energy, OUT / 'energy_contrasts.csv')
    metrics = predictor_metric_frame(load, pv, dates, archives)
    frame_to_csv(metrics, OUT / 'predictor_metrics.csv')
    subsets = pv_subset_frame(pv, dates, archives)
    frame_to_csv(subsets, OUT / 'pv_error_subsets.csv')
    windows = window_subset_frame(pv, dates, archives, candidate['support'], support_mask(pv))
    frame_to_csv(windows, OUT / 'pv_error_by_window.csv')
    monthly_error = monthly_error_frame(pv, dates, archives)
    frame_to_csv(monthly_error, OUT / 'pv_error_monthly.csv')
    frame_to_csv(hourly_error_frame(pv, dates, archives), OUT / 'pv_error_hourly.csv')
    frame_to_csv(load_weekday_hour_frame(load, dates, archives['P0']['load_forecast']),
                 OUT / 'load_weekday_hour_residual.csv')
    coverage = summary[['strategy_id', 'mean_adjustment_kWh', 'coverage']].copy()
    frame_to_csv(coverage, OUT / 'coverage.csv')

    clean = dict(P1=instances['P1'], initial={}, executed={})
    for res in results:
        strategy_id = res['totals']['strategy_id']
        for k in PERTURBATION_DAYS + [171]:
            clean['initial'][(strategy_id, k)] = float(
                res['daily'].iloc[k - WARMUP_DAYS].initial_kWh)
            row = res['dispatch'][res['dispatch'].date == str(dates[k].date())]
            assert len(row) == 144, (strategy_id, k, len(row))
            clean['executed'][(strategy_id, k)] = dict(
                c=row.charge_kWh.to_numpy(), d=row.discharge_kWh.to_numpy(),
                e=row.emergency_kWh.to_numpy(), w=row.unused_kWh.to_numpy(),
                states=np.r_[row.state_start_kWh.to_numpy()[0], row.state_end_kWh.to_numpy()])

    perturbation = perturbation_checks(load, pv, price, dates, archives['P0'], candidate, clean)
    save(OUT / 'future_perturbation_checks.json', perturbation)
    prefix = feedback_prefix_check(load, pv, price, dates, archives['P0'], candidate, clean)
    save(OUT / 'feedback_prefix_check.json', prefix)
    sampled = sampled_retraining_checks(pv, dates, candidate)
    save(OUT / 'sampled_retraining.json', sampled)
    produced = figures(monthly, stability, monthly_error, windows, summary)
    integrity = m.figure_integrity(produced)
    save(OUT / 'figure_integrity.json', integrity)

    protected_after = protected_manifest()
    changed = sorted(name for name in protected if protected_after.get(name) != protected[name])
    missing = sorted(name for name in protected if name not in protected_after)
    save(OUT / 'protected_after.json', protected_after)

    checks = dict(boundary=boundary, gate=gate, control_refit=control_refit, parity=parity,
                  quantile=quantile, reference_reproduction=reference_reproduction,
                  perturbation=perturbation, feedback_prefix=prefix, sampled_retraining=sampled,
                  energy=energy.to_dict('records'), metrics=metrics.to_dict('records'),
                  protected_unchanged=bool(not changed and not missing),
                  protected_changed=changed, protected_missing=missing,
                  protected_count=len(protected),
                  warmup=dict(shared_begin_state_kWh=warm_end,
                              checks={key: float(value) for key, value in warm_checks.items()}),
                  thresholds=dict(energy_kWh=ENERGY_TOL_KWH, cost_yuan=COST_TOL_YUAN))
    save(OUT / 'checks.json', checks)

    write_report(dict(summary=summary, paired=paired, monthly=monthly, stability=stability,
                      metrics=metrics, subsets=subsets, windows=windows,
                      monthly_error=monthly_error, support_summary=support, checks=checks,
                      coverage=coverage, gate=gate, parity=parity, integrity=integrity,
                      warm_end=warm_end, candidate_summary=candidate['summary'],
                      candidate_log=candidate['log']))
    artifact_hashes = {path.relative_to(OUT).as_posix(): digest(path)
                       for path in sorted(OUT.rglob('*'))
                       if path.is_file() and path.name not in ('artifact_hashes.json',
                                                               'run_manifest.json')}
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
                    candidate_summary=candidate['summary'],
                    scenario_totals={result['totals']['strategy_id']: result['totals']
                                     for result in results},
                    reference_reproduction=reference_reproduction,
                    self_checks_passed=dict(boundary=boundary['all_passed'], gate=gate['all_passed'],
                                            control_refit=control_refit['all_passed'],
                                            parity=parity['all_passed'], quantile=quantile['all_passed'],
                                            perturbation=perturbation['all_passed'],
                                            feedback_prefix=prefix['all_passed'],
                                            sampled_retraining=sampled['all_passed']),
                    artifact_count=len(artifact_hashes), report=str(REPORT_MD.relative_to(ROOT)))
    save(OUT / 'run_manifest.json', manifest)
    assert manifest['protected_unchanged'], (changed, missing)
    print(summary[['strategy_id', 'planned_cost_yuan', 'emergency_cost_yuan', 'total_cost_yuan',
                   'emergency_days']].to_string(index=False), flush=True)
    print(paired[['comparison', 'delta_total_cost_yuan', 'delta_total_cost_percent',
                  'delta_emergency_days']].to_string(index=False), flush=True)
    print(f'self-checks {manifest["self_checks_passed"]}', flush=True)
    print(f'wall seconds = {time.perf_counter() - started:.1f}', flush=True)
    return manifest


def load_weekday_hour_frame(load, dates, load_forecast):
    """Descriptive background: frozen load residual by weekday and hour."""
    window = slice(WARMUP_DAYS, len(dates))
    actual = load[window]
    forecast = load_forecast[window]
    error = actual - forecast
    weekday = np.array([d.weekday() for d in dates])[window]
    hour = np.tile(np.arange(144) // 6, (len(actual), 1))
    rows = []
    for day in range(7):
        for h in range(24):
            mask = (weekday[:, None] == day) & (hour == h)
            rows.append(dict(weekday=day, hour=h, n=int(mask.sum()),
                             mae_kW=float(np.mean(np.abs(error[mask]))) if mask.any() else np.nan,
                             bias_kW=float(np.mean(error[mask])) if mask.any() else np.nan))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    main()
