#!/usr/bin/env python
"""问题二 25 号实验：负载周期形状特征与光伏历史窗口特征的联合消融。

对应任务书 reports/问题二_负载与光伏周期特征联合优化实验方案.md。

四组固定 q80 同年因果回测：

* ``F00_original``    原 15 列负载 LightGBM ＋ 原 8 列光伏 LightGBM（门控），修正 G4 档案；
* ``F10_load_shape``  负载增至 19 列，光伏与 F00 完全相同；
* ``F01_pv_support``  光伏增至 12 列，负载与 F00 完全相同；
* ``F11_joint``       两者同时改动。

主比较 ΔC_L=F10−F00、ΔC_V=F01−F00、ΔC_LV=F11−F00，条件差 F11−F01、F11−F10 与
交互项 I_C=C11−C10−C01+C00。只新增本实验的脚本、结果、图与报告；旧资产只读。
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
CODE_25 = ROOT / 'code/25_q2_load_pv_shape_experiment.py'
OUT = ROOT / 'results/q2_load_pv_shape'
FIG = ROOT / 'figures/q2_load_pv_shape'
SNAP = ROOT / 'results/q2_revision_audit_20260911/source_snapshot'
FROZEN_RUN = ROOT / 'results/q2_lightgbm_residual'
SCAN_RUN = ROOT / 'results/q2_quantile_level_scan'
PV23_RUN = ROOT / 'results/q2_pv_support_features'
PLAN_MD = ROOT / 'reports/问题二_负载与光伏周期特征联合优化实验方案.md'
REPORT_MD = ROOT / 'reports/问题二_负载与光伏周期特征联合优化实验结果报告.md'

DT = 1 / 6
ALPHA = 0.80
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
CONTROL_ID = 'F00_original'
CONTROL_TOTAL_YUAN = 13673080.670412183
PV23_P1_TOTAL_YUAN = 13658867.74953597

# ---- load shape features (section 3.1 of the plan) ----
LOAD_TEMPLATE_WEEKS = (1, 2, 3, 4)
LOAD_SHAPE_WINDOW_SLOTS = 6                 # the fixed one-hour local window
LOAD_NEAR_CONSTANT_TOL_KW = 1e-8            # only guards division by a near-constant template
LOAD_SLOPE_TOL_KW_PER_H = 1e-8              # descriptive only, never used for fitting
LOAD_RELATIVE_HIGH_LEVEL = 0.5              # descriptive only, never used for fitting

# ---- PV support window (identical to module 16/19/23) ----
PV_SUPPORT_PARAMETERS = dict(history_days=7, activity_threshold_kW=1.0, padding_slots=3)

LOAD_CONTROL_FEATURES = ['l_base', 'l_yesterday_minus_base', 'l_known_week_change',
                         'l_sameweekday_mean_minus_base', 'l_daily_week_change',
                         'weekday_tue', 'weekday_wed', 'weekday_thu', 'weekday_fri',
                         'weekday_sat', 'weekday_sun', 'sin1', 'cos1', 'sin2', 'cos2']
LOAD_SHAPE_FEATURES = ['load_template_mean_kW', 'load_template_relative_level',
                       'load_template_backward_slope_kW_per_h',
                       'load_template_forward_slope_kW_per_h']
LOAD_CANDIDATE_FEATURES = LOAD_CONTROL_FEATURES + LOAD_SHAPE_FEATURES

PV_CONTROL_FEATURES = ['v_base', 'v_recent_change', 'v_week_mean_minus_base', 'v_daily_change',
                       'sin1', 'cos1', 'sin2', 'cos2']
PV_SHAPE_FEATURES = ['support_duration_h', 'support_midpoint_h', 'support_phase', 'support_valid']
PV_CANDIDATE_FEATURES = PV_CONTROL_FEATURES + PV_SHAPE_FEATURES

GROUPS = [
    dict(id='F00_original', predictor='L0V0', load_source='frozen_g4', pv_source='frozen_g4',
         alpha=ALPHA, role='control: repaired G4 load and PV published archives'),
    dict(id='F10_load_shape', predictor='L1V0', load_source='load_shape19', pv_source='frozen_g4',
         alpha=ALPHA, role='load ablation: 19-column load LightGBM, PV frozen at G4'),
    dict(id='F01_pv_support', predictor='L0V1', load_source='frozen_g4', pv_source='support12',
         alpha=ALPHA, role='PV ablation: 12-column support-window PV LightGBM, load frozen at G4'),
    dict(id='F11_joint', predictor='L1V1', load_source='load_shape19', pv_source='support12',
         alpha=ALPHA, role='joint treatment: both candidate feature sets'),
]
GROUP_BY_ID = {g['id']: g for g in GROUPS}
CONTROL_GROUP = GROUP_BY_ID[CONTROL_ID]
TREATMENT_GROUPS = [g for g in GROUPS if g['id'] != CONTROL_ID]
CONTRASTS = [
    ('F10_load_shape minus F00_original (pre-registered load comparison)', 'F10_load_shape', CONTROL_ID),
    ('F01_pv_support minus F00_original (pre-registered PV comparison)', 'F01_pv_support', CONTROL_ID),
    ('F11_joint minus F00_original (pre-registered joint comparison)', 'F11_joint', CONTROL_ID),
    ('F11_joint minus F01_pv_support (conditional on load frozen at F01)', 'F11_joint', 'F01_pv_support'),
    ('F11_joint minus F10_load_shape (conditional on PV frozen at F10)', 'F11_joint', 'F10_load_shape'),
]
PRIMARY_LABELS = [c[0] for c in CONTRASTS[:3]]
INTERACTION_LABEL = 'interaction I_C = C11 - C10 - C01 + C00'

SAMPLED_TRAINING_DAYS = [20, 21, 22, 31, 78, 171, 265, 354]
SAMPLED_LABELS = {20: '2025-01-21', 21: '2025-01-22', 22: '2025-01-23', 31: '2025-02-01',
                  78: '2025-03-20', 171: '2025-06-21', 265: '2025-09-23', 354: '2025-12-21'}
PERTURBATION_DAYS = [31, 171, 354]
SAMPLED_MILP_DAYS = [31, 78, 171, 265, 354]

PARAMETERS = dict(
    experiment='joint load-shape and PV support-window feature ablation, four pre-registered '
               'groups at a fixed q80 protection level',
    alpha=ALPHA,
    load_control_features=LOAD_CONTROL_FEATURES,
    load_candidate_features=LOAD_CANDIDATE_FEATURES,
    pv_control_features=PV_CONTROL_FEATURES,
    pv_candidate_features=PV_CANDIDATE_FEATURES,
    support_parameters=PV_SUPPORT_PARAMETERS,
    load_template_weeks=list(LOAD_TEMPLATE_WEEKS),
    load_template_rule='H_k = equal-weight mean of the up to four same-weekday days k-7r, '
                       'k-7r>=0; each training day j uses its own H_j built from days < j',
    load_shape_window_slots=LOAD_SHAPE_WINDOW_SLOTS,
    train_window_days=TRAIN_WINDOW, min_train_days=MIN_TRAIN_DAYS,
    residual_window_days=RESIDUAL_WINDOW, min_history_days=MIN_HISTORY_DAYS,
    terminal_kWh=TERMINAL_KWH,
    first_load_feature_day=8, first_load_trainable_day=22,
    first_pv_feature_day=7, first_pv_trainable_day=21,
    evaluation='2025-02-01..2025-12-31, 334 days, 48096 slots per group',
    diagnostic_thresholds=dict(relative_high_level=LOAD_RELATIVE_HIGH_LEVEL,
                               slope_tolerance_kW_per_h=LOAD_SLOPE_TOL_KW_PER_H,
                               near_constant_template_kW=LOAD_NEAR_CONSTANT_TOL_KW),
    groups=[dict(id=g['id'], predictor=g['predictor'], load_source=g['load_source'],
                 pv_source=g['pv_source'], alpha=g['alpha']) for g in GROUPS],
    comparisons=[label for label, _, _ in CONTRASTS],
    interaction=INTERACTION_LABEL,
    control_reference=dict(F00_original=CONTROL_TOTAL_YUAN),
    cross_module_reference=dict(pv_forecast_archive_P1=PV23_P1_TOTAL_YUAN),
    fallback_rule='fewer than 14 eligible training days, or a pre-feature day, publishes the '
                  'causal naive forecast; a constant-label window takes the label mean instead of '
                  'a pseudo tree model',
    protection_rule='each group rebuilds its own net-demand error archive and q80 protection; '
                    'single-change protection amounts are never added together',
)

PROTECTED_TREES = ['附件', 'code', 'reports', 'figures', 'results']
OWN_NEW_REL = {'code/25_q2_load_pv_shape_experiment.py',
               'reports/问题二_负载与光伏周期特征联合优化实验结果报告.md'}
OWN_NEW_PREFIXES = ('results/q2_load_pv_shape/', 'figures/q2_load_pv_shape/')
SHARED_APPEND_REL = {'建模上下文记忆.md', 'reports/项目进度.md'}

_MODULES: dict = {}


def parent():
    """Module 14: frozen snapshot access, the original 15/8-column feature functions."""
    if 'ridge' not in _MODULES:
        spec = importlib.util.spec_from_file_location(
            'ridge_parent', ROOT / 'code/14_q2_ridge_forecast_experiment.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULES['ridge'] = module
    return _MODULES['ridge']


def scan():
    """Module 21: explicit-alpha archive, scenario runner and identity checks.

    ``scan.OUT`` is redirected to this experiment's own tree so reusing its runner can never write
    into ``results/q2_quantile_level_scan``.
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


def pv_module():
    """Module 23: PV support-window features, gate reconstruction and the four PV checks.

    Its ``OUT``/``FIG`` are redirected so the rebuilt 12-column PV models land in this
    experiment's own result tree.
    """
    if 'pv23' not in _MODULES:
        spec = importlib.util.spec_from_file_location(
            'pv_support_features', ROOT / 'code/23_q2_pv_support_features_experiment.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.OUT = OUT
        module.FIG = FIG
        _MODULES['pv23'] = module
    return _MODULES['pv23']


def lightgbm_module():
    """Module 19: source of the frozen, already-registered LightGBM parameter dictionary."""
    if 'm19' not in _MODULES:
        spec = importlib.util.spec_from_file_location(
            'lightgbm_residual', ROOT / 'code/19_q2_lightgbm_residual_experiment.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULES['m19'] = module
    return _MODULES['m19']


def dependency_assertions():
    """The reused feature definitions must still be the registered ones."""
    m = parent()
    p = pv_module()
    assert list(m.LOAD_FEATURES) == LOAD_CONTROL_FEATURES, m.LOAD_FEATURES
    assert list(m.PV_FEATURES) == PV_CONTROL_FEATURES, m.PV_FEATURES
    assert list(p.CONTROL_FEATURES) == PV_CONTROL_FEATURES, p.CONTROL_FEATURES
    assert list(p.SUPPORT_FEATURES) == PV_SHAPE_FEATURES, p.SUPPORT_FEATURES
    assert dict(p.SUPPORT_PARAMETERS) == PV_SUPPORT_PARAMETERS, p.SUPPORT_PARAMETERS
    assert m.LOAD_THRESHOLD == 8 and m.PV_THRESHOLD == 7
    assert p.TRAIN_WINDOW == TRAIN_WINDOW and p.MIN_TRAIN_DAYS == MIN_TRAIN_DAYS
    assert p.RESIDUAL_WINDOW == RESIDUAL_WINDOW and p.CONSTANT_TARGET_TOL_KW == CONSTANT_TARGET_TOL_KW
    return dict(load_features=len(LOAD_CONTROL_FEATURES),
                load_candidate_features=len(LOAD_CANDIDATE_FEATURES),
                pv_features=len(PV_CONTROL_FEATURES), pv_candidate_features=len(PV_CANDIDATE_FEATURES))


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


# --------------------------------------------------------------------------------------
# load: same-weekday historical template and the four new columns
# --------------------------------------------------------------------------------------
def load_template_days(k):
    """The same-weekday historical days available on decision day k (empty before k=7)."""
    return [r for r in LOAD_TEMPLATE_WEEKS if k - 7 * r >= 0]


def load_template(load, k):
    """Equal-weight mean curve H_k of the up to four same-weekday days; ``None`` when empty."""
    ids = load_template_days(k)
    if not ids:
        return None, ids
    return np.asarray(load, dtype=float)[[k - 7 * r for r in ids]].mean(axis=0), ids


def load_shape_core(load, k):
    """The template plus the four raw shape columns, before being tiled into a design matrix."""
    template, ids = load_template(load, k)
    assert template is not None, k
    template_mean = float(template.mean())
    lowest = float(template.min())
    amplitude = float(template.max() - lowest)
    if amplitude > LOAD_NEAR_CONSTANT_TOL_KW:
        relative = (template - lowest) / amplitude
    else:
        relative = np.zeros(144, dtype=float)
    t = np.arange(144)
    backward_index = np.maximum(0, t - LOAD_SHAPE_WINDOW_SLOTS)
    forward_index = np.minimum(143, t + LOAD_SHAPE_WINDOW_SLOTS)
    backward_span = (t - backward_index) * DT
    forward_span = (forward_index - t) * DT
    backward = np.zeros(144, dtype=float)
    mask = backward_span > 0
    backward[mask] = (template[mask] - template[backward_index[mask]]) / backward_span[mask]
    forward = np.zeros(144, dtype=float)
    mask = forward_span > 0
    forward[mask] = (template[forward_index[mask]] - template[mask]) / forward_span[mask]
    return dict(template=template, ids=ids, template_mean=template_mean, amplitude=amplitude,
                relative=relative, backward=backward, forward=forward)


def load_shape_features(load, k):
    """The four appended columns, in the registered order."""
    core = load_shape_core(load, k)
    return np.column_stack([np.full(144, core['template_mean']), core['relative'],
                            core['backward'], core['forward']])


def load_day_features(load, dates, k, kind):
    """Design matrix for decision day k; ``original15`` or ``shape19``."""
    base = parent().load_feature_matrix(load, dates, k)
    if kind == 'original15':
        return base
    out = np.empty((144, len(LOAD_CANDIDATE_FEATURES)))
    out[:, :len(LOAD_CONTROL_FEATURES)] = base
    out[:, len(LOAD_CONTROL_FEATURES):] = load_shape_features(load, k)
    return out


def fit_load_day(load, dates, k, kind, model_dir=None):
    """Fit and publish one load decision day, strictly causally.

    The base level is the naive same-weekday level ``L_{k-7}`` (yesterday before day 7) and the
    label is its signed power residual, exactly as in the frozen 15-column model.  Only the input
    columns change.
    """
    threshold = parent().LOAD_THRESHOLD
    base = np.full(144, np.nan) if k < 1 else np.asarray(parent().naive_load(load, k), dtype=float)
    train_days = ([j for j in range(max(0, k - TRAIN_WINDOW), k) if j >= threshold]
                  if k >= threshold else [])
    entry = dict(target='load', day_index=int(k), date=str(dates[k].date()),
                 n_train_days=len(train_days), n_rows=0, trained=False, branch='', n_trees=0,
                 model_file='', model_sha256='', constant_new_columns='', n_template_days=0,
                 near_constant_template_days=0, correction_min_kW=np.nan,
                 correction_max_kW=np.nan)
    if k < threshold:
        entry['branch'] = 'feature_history_fallback'
        entry['reason'] = f'day_index<{threshold}: published naive fallback'
        published = np.maximum(0.0, base)
        return dict(published=published, raw=published.copy(), correction=np.zeros(144), base=base,
                    entry=entry, model=None, model_text='')
    if len(train_days) < MIN_TRAIN_DAYS:
        entry['branch'] = 'insufficient_history'
        entry['reason'] = f'train_days={len(train_days)}<{MIN_TRAIN_DAYS}'
        published = np.maximum(0.0, base)
        return dict(published=published, raw=published.copy(), correction=np.zeros(144), base=base,
                    entry=entry, model=None, model_text='')
    from lightgbm import LGBMRegressor
    X = np.concatenate([load_day_features(load, dates, j, kind) for j in train_days], axis=0)
    y = np.concatenate([np.asarray(load[j], dtype=float) - parent().naive_load(load, j)
                        for j in train_days])
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise RuntimeError(f'non-finite load feature or label on day {k}')
    entry['n_rows'] = int(X.shape[0])
    entry['train_first'] = str(dates[train_days[0]].date())
    entry['train_last'] = str(dates[train_days[-1]].date())
    entry['n_template_days'] = len(load_template_days(k))
    entry['near_constant_template_days'] = int(sum(
        1 for j in train_days if load_shape_core(load, j)['amplitude'] <= LOAD_NEAR_CONSTANT_TOL_KW))
    if kind != 'original15':
        never = [name for name, column in zip(LOAD_SHAPE_FEATURES, X[:, 15:].T)
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
        correction = np.asarray(model.predict(load_day_features(load, dates, k, kind)), dtype=float)
        entry['branch'] = 'fitted'
        entry['n_trees'] = int(model.booster_.num_trees())
        model_text = model.booster_.model_to_string()
        entry['model_sha256'] = hashlib.sha256(model_text.encode('utf-8')).hexdigest()
        if model_dir is not None:
            name = f'load_{dates[k].date()}.txt'
            (model_dir / name).write_text(model_text, encoding='utf-8')
            entry['model_file'] = f'models/{kind}/{name}'
    entry['trained'] = True
    raw = np.maximum(0.0, base + correction)
    entry['correction_min_kW'] = float(np.min(correction))
    entry['correction_max_kW'] = float(np.max(correction))
    entry['prediction_min_kW'] = float(raw.min())
    entry['prediction_max_kW'] = float(raw.max())
    return dict(published=raw, raw=raw, correction=correction, base=base, entry=entry,
                model=model, model_text=model_text)


def build_load_forecast(load, dates, kind, save_models=False, verbose=False):
    """Whole-year causal load archive for one feature set; day 0 stays unpublished (NaN)."""
    n_days = len(dates)
    published = np.full((n_days, 144), np.nan)
    raw = np.full((n_days, 144), np.nan)
    log = []
    model_dir = OUT / 'models' / kind if save_models else None
    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for k in range(1, n_days):
        outcome = fit_load_day(load, dates, k, kind, model_dir=model_dir)
        published[k] = outcome['published']
        raw[k] = outcome['raw']
        log.append(outcome['entry'])
        if verbose and k % 90 == 0:
            print(f'    {kind}: day {k} ({dates[k].date()}) branch={outcome["entry"]["branch"]}',
                  flush=True)
    trained = [e for e in log if e['trained']]
    first = int(np.flatnonzero(~np.isnan(published[:, 0]))[0]) if not np.isnan(published[:, 0]).all() else -1
    summary = dict(kind=kind, target='load', n_days=n_days,
                   n_fitted=int(sum(1 for e in log if e['branch'] == 'fitted')),
                   n_constant=int(sum(1 for e in log if e['branch'] == 'constant_target')),
                   n_insufficient=int(sum(1 for e in log if e['branch'] == 'insufficient_history')),
                   n_feature_fallback=int(sum(1 for e in log if e['branch'] == 'feature_history_fallback')),
                   first_trainable_day=int(trained[0]['day_index']) if trained else -1,
                   first_published_day=first, elapsed_seconds=time.perf_counter() - started)
    return dict(kind=kind, published=published, raw=raw, log=log, summary=summary)


def load_g4_archive(load, pv, dates):
    """Frozen repaired G4 published archive (load and PV), verified against the raw attachments."""
    s = scan()
    archive = s.load_predictor_archive(dict(id='P0', folder='G4_lightgbm_fixed'), load, pv, dates)
    archive['training_log'] = pd.read_csv(FROZEN_RUN / 'training_log.csv', low_memory=False)
    return archive


# --------------------------------------------------------------------------------------
# reference reproduction (three tiers, as in module 21/23)
# --------------------------------------------------------------------------------------
def compare_to_reference(result, group_id, reference_path, registered_total):
    """Tier 1 day-level aggregates, tier 2 registered total, tier 3 equivalent-optimum evidence."""
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
               segments=int(len(current)), max_executed_difference=float(
                   max(executed_differences.values())),
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
    out['passed'] = bool(out['max_day_level_difference'] < ENERGY_TOL_KWH
                         and out['difference_vs_registered_yuan'] <= COST_TOL_YUAN
                         and prices_equal and day_cost_identical)
    return out


# --------------------------------------------------------------------------------------
# implementation checks
# --------------------------------------------------------------------------------------
def boundary_checks(load, dates, pv):
    """Synthetic hand-computed boundaries; never presented as an attachment finding."""
    rows = []

    def add(name, value, expected, tolerance=0.0):
        value = np.asarray(value, dtype=float)
        expected = np.asarray(expected, dtype=float)
        worst = float(np.max(np.abs(value - expected))) if value.shape == expected.shape else float('inf')
        rows.append(dict(check=name, worst=worst, tolerance=tolerance,
                         passed=bool(worst <= tolerance + 1e-15)))

    n_days = len(dates)
    flat = np.full((n_days, 144), 7.0)
    core = load_shape_core(flat, 100)
    add('flat template mean', np.full(144, core['template_mean']), np.full(144, 7.0))
    add('flat template relative level is zero', core['relative'], np.zeros(144))
    add('flat template backward slope is zero', core['backward'], np.zeros(144))
    add('flat template forward slope is zero', core['forward'], np.zeros(144))
    add('flat template is near-constant', float(core['amplitude'] <= LOAD_NEAR_CONSTANT_TOL_KW), 1.0)

    ramp = np.tile(np.arange(144, dtype=float), (n_days, 1))
    core = load_shape_core(ramp, 100)
    add('ramp relative level is monotone endpoints', [core['relative'][0], core['relative'][143]], [0.0, 1.0])
    add('ramp internal backward slope', core['backward'][7:138], np.full(131, 6.0), 1e-12)
    add('ramp internal forward slope', core['forward'][7:138], np.full(131, 6.0), 1e-12)
    add('ramp backward slope is zero at t=0', core['backward'][0], 0.0)
    add('ramp forward slope is zero at t=143', core['forward'][143], 0.0)
    add('ramp shortened right window renormalises', core['forward'][140], (ramp[0, 143] - ramp[0, 140]) / (3 * DT))
    add('ramp shortened left window renormalises', core['backward'][3], (ramp[0, 3] - ramp[0, 0]) / (3 * DT))
    ramp_matrix = load_day_features(ramp, dates, 100, 'shape19')
    add('shape19 width', ramp_matrix.shape, [144, 19])
    add('shape19 first 15 columns are the frozen 15',
        ramp_matrix[:, :15], parent().load_feature_matrix(ramp, dates, 100), 1e-12)
    add('shape19 column 15 is the template mean', ramp_matrix[:, 15], np.full(144, float(ramp.mean(axis=1).mean())))
    add('shape19 column 19 is the forward slope', ramp_matrix[:, 18], core['forward'])

    double = np.zeros((n_days, 144))
    t = np.arange(144, dtype=float)
    double[:, :] = 100 + 50 * np.exp(-((t - 30) / 10) ** 2) + 80 * np.exp(-((t - 110) / 12) ** 2)
    profile = load_shape_core(double, 100)['relative']
    maxima = int(np.sum((profile[1:-1] > profile[:-2]) & (profile[1:-1] >= profile[2:])))
    rows.append(dict(check='double-peak template keeps two humps', worst=float(abs(maxima - 2)),
                     tolerance=0.0, passed=bool(maxima == 2)))
    add('double-peak relative level stays in [0,1]',
        [float(profile.min()) >= 0.0, float(profile.max()) <= 1.0], [1.0, 1.0])

    for k, expected_days in ((7, [1]), (14, [1, 2]), (21, [1, 2, 3]), (28, [1, 2, 3, 4]),
                             (6, []), (300, [1, 2, 3, 4])):
        rows.append(dict(check=f'template day count at k={k}',
                         worst=float(load_template_days(k) != expected_days), tolerance=0.0,
                         passed=bool(load_template_days(k) == expected_days)))
    template, ids = load_template(load, 28)
    add('H equals the mean of the four same-weekday days', template,
        load[[28 - 7 * r for r in ids]].mean(axis=0))

    # causality: changing the decision day and everything after it must not move its own features
    future = load.copy()
    future[100:] *= 1.2
    add('shape19 day-100 features ignore day 100 onward',
        load_day_features(future, dates, 100, 'shape19'),
        load_day_features(load, dates, 100, 'shape19'))
    add('original15 day-100 features ignore day 100 onward',
        load_day_features(future, dates, 100, 'original15'),
        load_day_features(load, dates, 100, 'original15'))
    for j in (95, 96, 97):
        add(f'day-100 training features on day {j} ignore day 100 onward',
            load_day_features(future, dates, j, 'shape19'),
            load_day_features(load, dates, j, 'shape19'))

    entry_k21 = fit_load_day(load, dates, 21, 'original15')['entry']
    entry_k22 = fit_load_day(load, dates, 22, 'original15')['entry']
    rows.append(dict(check='load day 21 still short of 14 training days',
                     worst=float(entry_k21['n_train_days'] != 13 or entry_k21['trained']),
                     tolerance=0.0,
                     passed=bool(entry_k21['n_train_days'] == 13 and not entry_k21['trained']
                                 and entry_k21['branch'] == 'insufficient_history')))
    rows.append(dict(check='load day 22 is the first eligible training day',
                     worst=float(entry_k22['n_train_days'] != 14 or not entry_k22['trained']),
                     tolerance=0.0,
                     passed=bool(entry_k22['n_train_days'] == 14 and entry_k22['trained'])))

    pv_boundary = pv_module().boundary_checks(pv, dates)
    rows.extend(pv_boundary['rows'])
    return dict(rows=rows, all_passed=bool(all(r['passed'] for r in rows)))


def load_template_parity_check(load, dates):
    """Every day's H must equal the frozen 15-column model's same-weekday mean column."""
    worst = 0.0
    for k in range(8, len(dates)):
        base = parent().naive_load(load, k)
        template, _ = load_template(load, k)
        column = parent().load_feature_matrix(load, dates, k)[:, 3]
        worst = max(worst, float(np.max(np.abs((template - base) - column))))
    return dict(max_difference_kW=worst, days=int(len(dates) - 8),
                all_passed=bool(worst == 0.0))


def feature_extension_check(load, pv, dates):
    """The candidate matrices must extend the frozen ones by exactly the appended columns."""
    worst_load = 0.0
    worst_pv = 0.0
    for k in range(8, len(dates)):
        mine = load_day_features(load, dates, k, 'shape19')
        frozen = load_day_features(load, dates, k, 'original15')
        worst_load = max(worst_load, float(np.max(np.abs(mine[:, :15] - frozen))))
        assert mine.shape == (144, len(LOAD_CANDIDATE_FEATURES))
    for k in range(7, len(dates)):
        mine = pv_module().day_features(pv, dates, k, 'support12')
        frozen = pv_module().day_features(pv, dates, k, 'original8')
        worst_pv = max(worst_pv, float(np.max(np.abs(mine[:, :8] - frozen))))
        assert mine.shape == (144, len(PV_CANDIDATE_FEATURES))
    return dict(load_max_difference_kW=worst_load, pv_max_difference_kW=worst_pv,
                load_columns=len(LOAD_CANDIDATE_FEATURES), pv_columns=len(PV_CANDIDATE_FEATURES),
                all_passed=bool(worst_load == 0.0 and worst_pv == 0.0))


def refit_control_check(load, pv, dates, g4):
    """Independently refit both frozen control models (15 and 8 columns) against the archive."""
    load_rebuilt = build_load_forecast(load, dates, 'original15')
    load_published = np.asarray(g4['load_forecast'], dtype=float)
    load_mask = np.isfinite(load_published)
    load_difference = float(np.max(np.abs(load_rebuilt['published'][load_mask] - load_published[load_mask])))
    pv_record, pv_rebuilt = pv_module().control_refit_check(pv, dates, g4)
    return dict(load_max_difference_kW=load_difference, load_compared_slots=int(load_mask.sum()),
                load_n_fitted=load_rebuilt['summary']['n_fitted'],
                load_n_constant=load_rebuilt['summary']['n_constant'],
                load_first_trainable_day=load_rebuilt['summary']['first_trainable_day'],
                pv_max_difference_kW=pv_record['max_difference_kW'],
                pv_compared_slots=pv_record['compared_slots'],
                pv_n_fitted=pv_record['n_fitted'], pv_first_trainable_day=pv_record['first_trainable_day'],
                all_passed=bool(load_difference < 1e-6 and pv_record['all_passed'])), load_rebuilt


def training_parity_checks(candidate, g4, target):
    """Training row keys, labels, first trainable day and branches must match the frozen control."""
    control = g4['training_log']
    control = control[control.target == target].set_index('day_index').sort_index()
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
    expected_first = 22 if target == 'load' else 21
    return dict(target=target, mismatches=mismatches[:40], mismatch_count=len(mismatches),
                first_trainable_day=first, control_first_trainable_day=control_first,
                expected_first_trainable_day=expected_first,
                all_passed=bool(not mismatches and first == control_first == expected_first))


def quantile_validation(group_errors):
    """Explicit-alpha order statistics and the per-m calculation rules, before any scenario run."""
    s = scan()
    out = dict(positions_at_28_days={f'{ALPHA:.2f}': s.quantile_index(RESIDUAL_WINDOW, ALPHA)},
               expected={f'{ALPHA:.2f}': 23})
    out['positions_passed'] = bool(out['positions_at_28_days'] == out['expected'])
    out['integer_rule_equals_ceil'] = bool(all(
        s.quantile_index(m, ALPHA) == s.ceil_index(m, ALPHA)
        for m in range(MIN_HISTORY_DAYS, RESIDUAL_WINDOW + 1)))
    per_group = {}
    for name, eps in group_errors.items():
        worst = 0.0
        for k in range(WARMUP_DAYS, len(eps)):
            low = max(1, k - RESIDUAL_WINDOW)
            m = k - low
            if m < MIN_HISTORY_DAYS:
                continue
            expected = np.sort(eps[low:k], axis=0)[s.quantile_index(m, ALPHA) - 1]
            r, _, _ = s.adjustment(eps, k, RESIDUAL_WINDOW, ALPHA)
            worst = max(worst, float(np.max(np.abs(r - expected))))
        per_group[name] = dict(max_difference_kWh=worst, passed=bool(worst < ENERGY_TOL_KWH))
    out['per_group'] = per_group
    out['all_passed'] = bool(out['positions_passed'] and out['integer_rule_equals_ceil']
                             and all(v['passed'] for v in per_group.values()))
    return out


def support_window_check(pv, candidate_support, dates):
    """Rebuild every day's window with the registered rule.

    Day 0 has no decision and must carry the documented default ``(0, 143, 0)``; the candidate
    archive built by the shared PV builder leaves its unused day-0 row at zeros, so the archive
    columns for day 0 are rewritten from this rebuild rather than trusted.
    """
    rebuilt = np.vstack([pv_module().support_window(pv, k) for k in range(len(dates))])
    candidate = np.asarray(candidate_support, dtype=int)
    mismatch = int(np.max(np.abs(rebuilt[1:] - candidate[1:]))) if len(rebuilt) > 1 else 0
    return dict(day0_rebuilt=tuple(int(v) for v in rebuilt[0]),
                day0_candidate_row=tuple(int(v) for v in candidate[0]),
                day0_is_documented_default=bool(tuple(int(v) for v in rebuilt[0]) == (0, 143, 0)),
                mismatch_from_day1=mismatch,
                valid_days=int((rebuilt[:, 2] == 1).sum()),
                no_valid_window_days=int((rebuilt[:, 2] == 0).sum()),
                all_passed=bool(mismatch == 0 and tuple(int(v) for v in rebuilt[0]) == (0, 143, 0)))


def population_identity_check(archives, group_archives, pv):
    """The shared-forecast design must hold exactly, and both PV archives must share one gate."""
    def identical(left, right):
        """Day 0 is a documented NaN in both archives; compare as declared identities."""
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        if np.array_equal(left, right, equal_nan=True):
            return 0.0
        finite = np.isfinite(left) & np.isfinite(right)
        return float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else float('inf')

    rows = []
    for a, b in (('F00_original', 'F01_pv_support'), ('F10_load_shape', 'F11_joint')):
        left = archives[group_archives[a][0]]['load_forecast']
        right = archives[group_archives[b][0]]['load_forecast']
        rows.append(dict(check=f'{a} and {b} share the identical load archive',
                         worst=identical(left, right)))
    for a, b in (('F00_original', 'F10_load_shape'), ('F01_pv_support', 'F11_joint')):
        left = archives[group_archives[a][1]]['pv_forecast']
        right = archives[group_archives[b][1]]['pv_forecast']
        rows.append(dict(check=f'{a} and {b} share the identical PV archive',
                         worst=identical(left, right)))
    mask = pv_module().support_mask(pv)
    off = ~mask
    v0 = np.nan_to_num(np.asarray(archives['V0']['pv_forecast'], dtype=float))
    v1 = np.nan_to_num(np.asarray(archives['V1']['pv_forecast'], dtype=float))
    gate = dict(gated_off_slots=int(off.sum()),
                v0_max_off_gate_kW=float(np.abs(v0[off]).max()),
                v1_max_off_gate_kW=float(np.abs(v1[off]).max()),
                v0_above_100kW_off_gate_slots=int((v0[off] > 100.0).sum()),
                v1_above_100kW_off_gate_slots=int((v1[off] > 100.0).sum()),
                truth_off_gate_kWh=float((np.asarray(pv, dtype=float)[off] * DT).sum()))
    passed = bool(all(r['worst'] == 0.0 for r in rows)
                  and gate['v0_max_off_gate_kW'] < ENERGY_TOL_KWH
                  and gate['v1_max_off_gate_kW'] < ENERGY_TOL_KWH
                  and gate['v0_above_100kW_off_gate_slots'] == 0
                  and gate['v1_above_100kW_off_gate_slots'] == 0)
    return dict(rows=rows, gate=gate, all_passed=passed)


def cross_module_pv_check(pv_candidate, f01_result):
    """This experiment's PV rebuild must reproduce module 23's audited P1 archive and dispatch."""
    reference_frame = pd.read_csv(PV23_RUN / 'pv_forecast_archive_P1.csv', low_memory=False,
                                 dtype={'date': str})
    reference_frame = reference_frame.sort_values(['date', 'slot'], kind='mergesort').reset_index(drop=True)
    reference = reference_frame.pv_forecast_kW.to_numpy().reshape(len(pv_candidate['published']), 144)
    mine = np.asarray(pv_candidate['published'], dtype=float)
    mask = np.isfinite(reference)
    forecast_difference = float(np.max(np.abs(mine[mask] - reference[mask])))
    support_reference = reference_frame[['support_start_slot', 'support_end_slot', 'support_valid']]
    support_reference = support_reference.to_numpy().reshape(len(mine), 144, 3)[:, 0]
    support_difference = int(np.max(np.abs(support_reference
                                           - np.asarray(pv_candidate['support'], dtype=int))))
    dispatch = compare_to_reference(f01_result, 'F01_pv_support',
                                   PV23_RUN / 'P1_q80' / 'dispatch.csv', PV23_P1_TOTAL_YUAN)
    return dict(pv_forecast_max_difference_kW=forecast_difference, compared_slots=int(mask.sum()),
                support_window_max_difference=int(support_difference), dispatch=dispatch,
                all_passed=bool(forecast_difference < 1e-6 and support_difference == 0
                                and dispatch['passed']))


def perturbation_checks(load, pv, price, dates, clean, load_candidate, pv_candidate, guard):
    """Six future-truth perturbations across both candidate chains.

    For the issue day the load template (including the forward one-hour slope), the PV support
    window, both training-row sets, both published forecasts, all four groups' q80 protection and
    all four groups' ordinary plans must be unchanged.  Post-reveal execution is not asserted.
    """
    s = scan()
    cases = []
    for k in PERTURBATION_DAYS:
        for kind in ('load', 'pv'):
            ll, vv = load.copy(), pv.copy()
            if kind == 'load':
                ll[k:] *= 1.2
            else:
                vv[k:] *= 0.7
            load_rebuilt = fit_load_day(ll, dates, k, 'shape19')
            pv_rebuilt = pv_module().fit_day(vv, dates, k, 'support12')
            clean_load_entry = next(e for e in load_candidate['log'] if e['day_index'] == k)
            clean_pv_entry = next(e for e in pv_candidate['log'] if e['day_index'] == k)
            load_error = float(np.max(np.abs(load_rebuilt['published'] - load_candidate['published'][k])))
            pv_error = float(np.max(np.abs(pv_rebuilt['published'] - pv_candidate['published'][k])))
            template_equal = bool(np.max(np.abs(
                load_shape_features(ll, k) - load_shape_features(load, k))) == 0.0)
            window_equal = bool(tuple(pv_rebuilt['support']) == tuple(int(v) for v in pv_candidate['support'][k]))
            rows_equal = bool(
                load_rebuilt['entry']['n_train_days'] == clean_load_entry['n_train_days']
                and load_rebuilt['entry']['n_rows'] == clean_load_entry['n_rows']
                and load_rebuilt['entry']['branch'] == clean_load_entry['branch']
                and pv_rebuilt['entry']['n_train_days'] == clean_pv_entry['n_train_days']
                and pv_rebuilt['entry']['n_rows'] == clean_pv_entry['n_rows']
                and pv_rebuilt['entry']['branch'] == clean_pv_entry['branch'])
            load_fc = np.asarray(load_candidate['published'], dtype=float).copy()
            pv_fc = np.asarray(pv_candidate['published'], dtype=float).copy()
            load_fc[k] = load_rebuilt['published']
            pv_fc[k] = pv_rebuilt['published']
            perturbed = {
                'F00_original': s.ScanArchive(ll, vv, len(dates), guard['L0'], guard['V0']),
                'F10_load_shape': s.ScanArchive(ll, vv, len(dates), load_fc, guard['V0']),
                'F01_pv_support': s.ScanArchive(ll, vv, len(dates), guard['L0'], pv_fc),
                'F11_joint': s.ScanArchive(ll, vv, len(dates), load_fc, pv_fc),
            }
            protection, plan = {}, {}
            for group_id, archive in perturbed.items():
                state = float(clean['initial'][(group_id, k)])
                r_pert, _, _ = s.adjustment(archive.eps, k, RESIDUAL_WINDOW, ALPHA)
                r_clean, _, _ = s.adjustment(clean[group_id].eps, k, RESIDUAL_WINDOW, ALPHA)
                protection[group_id] = float(np.max(np.abs(r_pert - r_clean)))
                q_pert, _, summary_pert, _, _, _, _ = archive.plan_day(
                    k, ALPHA, RESIDUAL_WINDOW, price, state)
                q_clean, _, summary_clean, _, _, _, _ = clean[group_id].plan_day(
                    k, ALPHA, RESIDUAL_WINDOW, price, state)
                assert not summary_pert.get('fallback', False), (k, kind, group_id, summary_pert)
                assert not summary_clean.get('fallback', False), (k, kind, group_id, summary_clean)
                plan[group_id] = float(np.max(np.abs(q_pert - q_clean)))
            passed = bool(load_error < ENERGY_TOL_KWH and pv_error < ENERGY_TOL_KWH
                          and template_equal and window_equal and rows_equal
                          and max(protection.values()) < ENERGY_TOL_KWH
                          and max(plan.values()) < ENERGY_TOL_KWH)
            assert passed, (k, kind, load_error, pv_error, protection, plan)
            cases.append(dict(day=int(k), date=str(dates[k].date()), kind=kind,
                              load_forecast_max_kW=load_error, pv_forecast_max_kW=pv_error,
                              load_template_equal=template_equal, support_window_equal=window_equal,
                              training_rows_equal=rows_equal, protection_max_kWh=protection,
                              plan_max_kWh=plan, passed=passed))
    return dict(cases=cases, all_passed=bool(all(c['passed'] for c in cases)),
                scope='load template and forward one-hour slope + PV support window + both training '
                      'row sets + both published forecasts + all four groups q80 protection and '
                      'ordinary plans on the issue day; post-reveal execution is not asserted')


def feedback_prefix_check(load, pv, price, dates, clean, load_candidate, pv_candidate, guard):
    """A midday truth perturbation must not move the same day's morning plans or executions."""
    s = scan()
    k = 171
    m = parent()
    executed = {}
    for group in GROUPS:
        row = clean[group['id'] + '_dispatch'][clean[group['id'] + '_dispatch'].date == str(dates[k].date())]
        assert len(row) == 144, (group['id'], len(row))
        executed[group['id']] = dict(
            c=row.charge_kWh.to_numpy(), d=row.discharge_kWh.to_numpy(),
            e=row.emergency_kWh.to_numpy(), w=row.unused_kWh.to_numpy(),
            states=np.r_[row.state_start_kWh.to_numpy()[0], row.state_end_kWh.to_numpy()])
    rows = []
    for kind in ('load', 'pv'):
        ll, vv = load.copy(), pv.copy()
        if kind == 'load':
            ll[k, 72:] *= 1.2
        else:
            vv[k, 72:] *= 0.7
        load_fc = np.asarray(load_candidate['published'], dtype=float).copy()
        pv_fc = np.asarray(pv_candidate['published'], dtype=float).copy()
        if kind == 'load':
            load_fc[k] = fit_load_day(ll, dates, k, 'shape19')['published']
        else:
            pv_fc[k] = pv_module().fit_day(vv, dates, k, 'support12')['published']
        perturbed = {
            'F00_original': s.ScanArchive(ll, vv, len(dates), guard['L0'], guard['V0']),
            'F10_load_shape': s.ScanArchive(ll, vv, len(dates), load_fc, guard['V0']),
            'F01_pv_support': s.ScanArchive(ll, vv, len(dates), guard['L0'], pv_fc),
            'F11_joint': s.ScanArchive(ll, vv, len(dates), load_fc, pv_fc),
        }
        for group_id, archive in perturbed.items():
            state = float(clean['initial'][(group_id, k)])
            q_pert, _, summary_pert, _, _, _, _ = archive.plan_day(
                k, ALPHA, RESIDUAL_WINDOW, price, state)
            q_clean, _, summary_clean, _, _, _, _ = clean[group_id].plan_day(
                k, ALPHA, RESIDUAL_WINDOW, price, state)
            assert not summary_pert.get('fallback', False), (kind, group_id, summary_pert)
            assert not summary_clean.get('fallback', False), (kind, group_id, summary_clean)
            plan_error = float(np.max(np.abs(q_pert - q_clean)))
            control = m.frozen().bm.control
            c, d, e, w, states, _ = control(q_pert, ll[k], vv[k], state)
            prefix = max(float(np.max(np.abs(c[:72] - executed[group_id]['c'][:72]))),
                         float(np.max(np.abs(d[:72] - executed[group_id]['d'][:72]))),
                         float(np.max(np.abs(e[:72] - executed[group_id]['e'][:72]))),
                         float(np.max(np.abs(w[:72] - executed[group_id]['w'][:72]))),
                         float(np.max(np.abs(states[:73] - executed[group_id]['states'][:73]))))
            passed = bool(plan_error < ENERGY_TOL_KWH and prefix < ENERGY_TOL_KWH)
            assert passed, (kind, group_id, plan_error, prefix)
            rows.append(dict(day=k, kind=kind, group=group_id, plan_max_kWh=plan_error,
                             executed_prefix_max=prefix, passed=passed))
    return dict(cases=rows, all_passed=bool(all(c['passed'] for c in rows)),
                scope='day 171 plan and slots 0..71 of the realised execution plus the state entering '
                      'slot 72, all four groups')


def sampled_retraining_checks(load, pv, dates, g4, load_candidate, pv_candidate):
    """Repeat-fit all four target-versions on the registered days and reload the model text.

    ``original15``/``original8`` are compared against the frozen G4 archive, ``shape19``/
    ``support12`` against this run's own candidate archive.  Model text is written to a scratch
    directory, reloaded through ``Booster(model_str=...)`` and re-predicted.  This calls the same
    fitting implementation as the full build: it is a repeat-fit and reload consistency check, not
    an independent reimplementation.
    """
    from lightgbm import Booster
    rows = []
    for k in SAMPLED_TRAINING_DAYS:
        for target, kind, truth, reference, prefix in (
                ('load', 'original15', load, np.asarray(g4['load_forecast'], dtype=float), 'load'),
                ('load', 'shape19', load, load_candidate['published'], 'load'),
                ('pv', 'original8', pv, np.asarray(g4['pv_forecast'], dtype=float), 'pv'),
                ('pv', 'support12', pv, pv_candidate['published'], 'pv')):
            model_dir = OUT / 'sampled_models' / kind
            model_dir.mkdir(parents=True, exist_ok=True)
            if target == 'load':
                rebuilt = fit_load_day(truth, dates, k, kind, model_dir=model_dir)
            else:
                rebuilt = pv_module().fit_day(truth, dates, k, kind, model_dir=model_dir)
            refit_difference = float(np.max(np.abs(rebuilt['published'] - reference[k])))
            entry = rebuilt['entry']
            reload_difference = np.nan
            if entry['branch'] == 'fitted':
                text = (model_dir / f'{prefix}_{dates[k].date()}.txt').read_text(encoding='utf-8')
                booster = Booster(model_str=text)
                if target == 'load':
                    correction = np.asarray(
                        booster.predict(load_day_features(truth, dates, k, kind)), dtype=float)
                    base = np.asarray(parent().naive_load(truth, k), dtype=float)
                    reloaded = np.maximum(0.0, base + correction)
                else:
                    correction = np.asarray(
                        booster.predict(pv_module().day_features(truth, dates, k, kind)), dtype=float)
                    base = np.asarray(truth[k - 1], dtype=float)
                    reloaded = pv_module().apply_gate(np.maximum(0.0, base + correction),
                                                      rebuilt['gate'])
                reload_difference = float(np.max(np.abs(reloaded - rebuilt['published'])))
            rows.append(dict(day=int(k), date=SAMPLED_LABELS[k], target=target, kind=kind,
                             branch=entry['branch'], n_train_days=entry['n_train_days'],
                             n_rows=entry['n_rows'], refit_max_kW=refit_difference,
                             reload_max_kW=reload_difference,
                             passed=bool(refit_difference < 1e-6
                                         and (np.isnan(reload_difference) or reload_difference < 1e-6))))
    return dict(rows=rows, all_passed=bool(all(r['passed'] for r in rows)),
                checks=int(len(rows)), target_version_days=int(len(rows)),
                scope='8 registered days x 4 target-version combinations (original15/shape19/'
                      'original8/support12); same-implementation repeat fit plus saved-model text '
                      'reload consistency, not an independent reimplementation and not a full-year '
                      'independent recompute')


def sampled_milp_checks(results, price, dates):
    """Re-solve the registered sampled days plus each group's worst emergency day from scratch."""
    s = scan()
    kernel = parent().frozen().core
    date_strings = np.array([str(d.date()) for d in dates])
    rows = []
    for result in results:
        group_id = result['totals']['strategy_id']
        archive = result['archive']
        daily = result['daily'].set_index('date')
        days = sorted(set([str(dates[k].date()) for k in SAMPLED_MILP_DAYS]))
        worst_day = str(daily.emergency_cost_yuan.idxmax())
        if worst_day not in days:
            days.append(worst_day)
        for date in days:
            k = int(np.flatnonzero(date_strings == date)[0])
            state = float(daily.loc[date, 'initial_kWh'])
            r, _, _ = s.adjustment(archive.eps, k, RESIDUAL_WINDOW, ALPHA)
            protected = archive.n_forecast[k] + r
            (q, c, d, w, nominal_state), summary = kernel.solve(
                protected / DT, np.zeros(144), price, integer=True, initial_kWh=state,
                terminal_kWh=TERMINAL_KWH)
            objective = float((q * price).sum())
            recorded = float(daily.loc[date, 'planned_cost_yuan'])
            capacity_violation = float(max(0.0, 1200 - float(min(nominal_state.min(), 1200)),
                                           float(max(nominal_state.max(), 10800)) - 10800))
            rows.append(dict(group=group_id, date=date, day_index=int(k),
                             objective_yuan=objective, recorded_planned_cost_yuan=recorded,
                             objective_difference_yuan=abs(objective - recorded),
                             mip_gap=float(summary.get('mip_gap', 0.0)),
                             terminal_error_kWh=float(abs(nominal_state[-1] - TERMINAL_KWH)),
                             balance_residual_kWh=float(np.max(np.abs(q + d - protected - c - w))),
                             capacity_violation_kWh=capacity_violation,
                             passed=bool(abs(objective - recorded) < COST_TOL_YUAN
                                         and abs(nominal_state[-1] - TERMINAL_KWH) < ENERGY_TOL_KWH)))
    return dict(rows=rows, all_passed=bool(all(r['passed'] for r in rows)), solves=int(len(rows)),
                scope='registered sampled days plus each group worst emergency day, re-solved from '
                      'the recorded initial state; equal-optimum nominal trajectories may differ')


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
        row['mean_protection_kWh'] = float(dispatch.residual_adjustment_kWh.mean())
        row['initial_kWh'] = float(result['daily'].initial_kWh.iloc[0])
        row['maximum_mip_gap'] = float(result['daily'].mip_gap.max())
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


def contrast_frame(results):
    by_id = {r['totals']['strategy_id']: r for r in results}
    rows = [delta_row(label, by_id[treatment], by_id[baseline])
            for label, treatment, baseline in CONTRASTS]
    interaction = (by_id['F11_joint']['totals']['total_cost_yuan']
                   - by_id['F10_load_shape']['totals']['total_cost_yuan']
                   - by_id['F01_pv_support']['totals']['total_cost_yuan']
                   + by_id['F00_original']['totals']['total_cost_yuan'])
    rows.append(dict(comparison=INTERACTION_LABEL, treatment='F11_joint', baseline='additive model',
                     delta_planned_cost_yuan=np.nan, delta_emergency_cost_yuan=np.nan,
                     delta_total_cost_yuan=float(interaction), delta_planned_kWh=np.nan,
                     delta_emergency_kWh=np.nan, delta_unused_kWh=np.nan,
                     delta_emergency_days=np.nan, delta_emergency_events=np.nan))
    frame = pd.DataFrame(rows)
    frame['delta_total_cost_percent'] = [
        (100 * r.delta_total_cost_yuan / by_id[r.baseline]['totals']['total_cost_yuan'])
        if r.baseline in by_id else np.nan for r in frame.itertuples()]
    return frame


def monthly_contrast_frame(results):
    by_id = {r['totals']['strategy_id']: r for r in results}
    rows = []
    for label, treatment, baseline in CONTRASTS:
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


def stability_frame(results, monthly):
    by_id = {r['totals']['strategy_id']: r for r in results}
    rows = []
    for label, treatment, baseline in CONTRASTS:
        t, b = by_id[treatment]['daily'], by_id[baseline]['daily']
        daily_delta = t.total_cost_yuan.to_numpy() - b.total_cost_yuan.to_numpy()
        block = monthly[monthly.comparison == label]
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
        rows.append(dict(comparison=label,
                         months_improved=int((block.delta_total_cost_yuan < 0).sum()),
                         months_total=int(len(block)),
                         days_improved=int((daily_delta < 0).sum()),
                         days_worse=int((daily_delta > 0).sum()), days_total=int(len(daily_delta)),
                         worst_day=str(t.date.iloc[worst]), worst_day_yuan=float(daily_delta[worst]),
                         best_day=str(t.date.iloc[best]), best_day_yuan=float(daily_delta[best]),
                         **{f'delta_emergency_cost_{k}': v for k, v in windows.items()}))
    return pd.DataFrame(rows)


def energy_frame(results):
    by_id = {r['totals']['strategy_id']: r for r in results}
    rows = []
    for label, treatment, baseline in CONTRASTS:
        t, b = by_id[treatment]['totals'], by_id[baseline]['totals']
        delta = dict(planned=t['planned_kWh'] - b['planned_kWh'],
                     emergency=t['emergency_kWh'] - b['emergency_kWh'],
                     unused=t['unused_kWh'] - b['unused_kWh'],
                     loss=t['loss_kWh'] - b['loss_kWh'],
                     final=t['final_kWh'] - b['final_kWh'],
                     initial=t['evaluation_initial_kWh'] - b['evaluation_initial_kWh'])
        rows.append(dict(comparison=label, delta_planned_kWh=delta['planned'],
                         delta_emergency_kWh=delta['emergency'], delta_unused_kWh=delta['unused'],
                         delta_loss_kWh=delta['loss'], delta_end_kWh=delta['final'] - delta['initial'],
                         delta_initial_kWh=delta['initial'],
                         identity_residual_kWh=float(delta['planned'] + delta['emergency']
                                                     - delta['unused'] - delta['loss']
                                                     - (delta['final'] - delta['initial']))))
    return pd.DataFrame(rows)


def inventory_adjusted_frame(results, price):
    """Auxiliary inventory diagnostic C* = C - nu (E_end - E_start), nu = median(price)/0.9."""
    nu = float(np.median(price)) / 0.9
    rows = []
    for result in results:
        totals = result['totals']
        rows.append(dict(strategy_id=totals['strategy_id'], nu_yuan_per_internal_kWh=nu,
                         total_cost_yuan=totals['total_cost_yuan'],
                         inventory_adjustment_yuan=float(-nu * (totals['final_kWh']
                                                               - totals['evaluation_initial_kWh'])),
                         adjusted_cost_yuan=float(totals['total_cost_yuan']
                                                  - nu * (totals['final_kWh']
                                                          - totals['evaluation_initial_kWh']))))
    return pd.DataFrame(rows)


def predictor_metric_frame(load, pv, dates, archives):
    """Single-target point-forecast metrics in kW over 2025-02-01..12-31."""
    rows = []
    window = slice(WARMUP_DAYS, len(dates))
    specs = [('L0', 'load', archives['L0']['load_forecast'], load),
             ('L1', 'load', archives['L1']['load_forecast'], load),
             ('V0', 'pv', archives['V0']['pv_forecast'], pv),
             ('V1', 'pv', archives['V1']['pv_forecast'], pv)]
    for name, scope, forecast, actual in specs:
        error = actual[window] - np.asarray(forecast, dtype=float)[window]
        rows.append(dict(predictor=name, scope=scope, n=int(error.size),
                         mae_kW=float(np.mean(np.abs(error))),
                         rmse_kW=float(np.sqrt(np.mean(error ** 2))),
                         bias_kW=float(np.mean(error))))
    return pd.DataFrame(rows)


def net_demand_metric_frame(load, pv, dates, archives, group_archives):
    """The four net-demand combinations, one per group."""
    rows = []
    window = slice(WARMUP_DAYS, len(dates))
    actual = (load - pv)[window]
    for group_id, (load_name, pv_name) in group_archives.items():
        forecast = (np.asarray(archives[load_name]['load_forecast'], dtype=float)[window]
                    - np.asarray(archives[pv_name]['pv_forecast'], dtype=float)[window])
        error = actual - forecast
        rows.append(dict(group=group_id, load_source=load_name, pv_source=pv_name,
                         n=int(error.size), mae_kW=float(np.mean(np.abs(error))),
                         rmse_kW=float(np.sqrt(np.mean(error ** 2))),
                         bias_kW=float(np.mean(error))))
    return pd.DataFrame(rows)


def load_error_frame(load, dates, archives, shapes):
    """Load error by month, by weekday x hour, and by the descriptive shape groupings."""
    window = slice(WARMUP_DAYS, len(dates))
    actual = load[window]
    months = np.array([str(d)[:7] for d in dates])[window]
    weekdays = np.array([d.weekday() for d in dates])[window]
    hours = np.tile(np.arange(144) // 6, (len(actual), 1))
    high = shapes['high'][window]
    low = shapes['low'][window]
    rising = shapes['rising'][window]
    falling = shapes['falling'][window]
    level = shapes['level'][window]
    rows = []
    for name, archive in archives.items():
        if not name.startswith('L'):
            continue
        error = actual - archive['load_forecast'][window]
        for label, mask, grouping in (
                ('all', np.ones_like(actual, dtype=bool), 'scope'),
                ('relative_high_z_ge_0.5', high, 'shape_level'),
                ('relative_low_z_lt_0.5', low, 'shape_level'),
                ('near_constant_template', shapes['constant'][window], 'shape_level'),
                ('slope_rising', rising, 'shape_slope'),
                ('slope_falling', falling, 'shape_slope'),
                ('slope_level', level, 'shape_slope')):
            rows.append(dict(predictor=name, grouping=grouping, subset=label, n=int(mask.sum()),
                             mae_kW=float(np.mean(np.abs(error[mask]))) if mask.any() else np.nan,
                             bias_kW=float(np.mean(error[mask])) if mask.any() else np.nan))
        for weekday in range(7):
            for hour in range(24):
                mask = (weekdays[:, None] == weekday) & (hours == hour)
                rows.append(dict(predictor=name, grouping='weekday_hour',
                                 subset=f'wd{weekday}_h{hour:02d}', n=int(mask.sum()),
                                 mae_kW=float(np.mean(np.abs(error[mask]))) if mask.any() else np.nan,
                                 bias_kW=float(np.mean(error[mask])) if mask.any() else np.nan))
        for month in sorted(set(months)):
            mask = months == month
            rows.append(dict(predictor=name, grouping='month', subset=month, n=int(mask.sum()),
                             mae_kW=float(np.mean(np.abs(error[mask]))),
                             bias_kW=float(np.mean(error[mask]))))
    return pd.DataFrame(rows)


def load_shape_frame(load, dates):
    """Per-day template size, amplitude, constancy and the descriptive grouping counts."""
    rows = []
    for k in range(len(dates)):
        ids = load_template_days(k)
        if not ids:
            rows.append(dict(date=str(dates[k].date()), day_index=k, n_template_days=0,
                             template_mean_kW=np.nan, template_amplitude_kW=np.nan,
                             near_constant_template=np.nan, n_high_slots=0, n_low_slots=0,
                             n_rising_slots=0, n_falling_slots=0, n_level_slots=0))
            continue
        core = load_shape_core(load, k)
        slope = 0.5 * (core['backward'] + core['forward'])
        constant = bool(core['amplitude'] <= LOAD_NEAR_CONSTANT_TOL_KW)
        blank = np.zeros(144, dtype=bool)
        high = blank if constant else (core['relative'] >= LOAD_RELATIVE_HIGH_LEVEL)
        low = blank if constant else (core['relative'] < LOAD_RELATIVE_HIGH_LEVEL)
        rows.append(dict(date=str(dates[k].date()), day_index=k, n_template_days=len(ids),
                         template_mean_kW=core['template_mean'],
                         template_amplitude_kW=core['amplitude'], near_constant_template=constant,
                         n_high_slots=int(high.sum()), n_low_slots=int(low.sum()),
                         n_rising_slots=int((slope > LOAD_SLOPE_TOL_KW_PER_H).sum()),
                         n_falling_slots=int((slope < -LOAD_SLOPE_TOL_KW_PER_H).sum()),
                         n_level_slots=int((np.abs(slope) <= LOAD_SLOPE_TOL_KW_PER_H).sum())))
    return pd.DataFrame(rows)


def descriptive_masks(load, dates):
    """Slot-level descriptive masks used only for result diagnostics, never for fitting."""
    n_days = len(dates)
    shape = {key: np.zeros((n_days, 144), dtype=bool)
             for key in ('high', 'low', 'constant', 'rising', 'falling', 'level')}
    shape['constant'][:] = False
    for k in range(n_days):
        ids = load_template_days(k)
        if not ids:
            shape['constant'][k] = True
            continue
        core = load_shape_core(load, k)
        slope = 0.5 * (core['backward'] + core['forward'])
        if core['amplitude'] <= LOAD_NEAR_CONSTANT_TOL_KW:
            shape['constant'][k] = True
        else:
            shape['high'][k] = core['relative'] >= LOAD_RELATIVE_HIGH_LEVEL
            shape['low'][k] = core['relative'] < LOAD_RELATIVE_HIGH_LEVEL
        shape['rising'][k] = slope > LOAD_SLOPE_TOL_KW_PER_H
        shape['falling'][k] = slope < -LOAD_SLOPE_TOL_KW_PER_H
        shape['level'][k] = np.abs(slope) <= LOAD_SLOPE_TOL_KW_PER_H
    return shape


def save_load_shape_archive(load, dates):
    n_days = len(dates)
    template = np.full((n_days, 144), np.nan)
    relative = np.full((n_days, 144), np.nan)
    backward = np.full((n_days, 144), np.nan)
    forward = np.full((n_days, 144), np.nan)
    for k in range(n_days):
        if not load_template_days(k):
            continue
        core = load_shape_core(load, k)
        template[k] = core['template']
        relative[k] = core['relative']
        backward[k] = core['backward']
        forward[k] = core['forward']
    np.savez_compressed(OUT / 'load_shape_features.npz', template=template, relative=relative,
                        backward=backward, forward=forward,
                        column_order=np.array(LOAD_SHAPE_FEATURES, dtype='U64'))
    return dict(columns=LOAD_SHAPE_FEATURES, days_with_template=int(np.isfinite(template[:, 0]).sum()))


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------
def figures(summary, monthly, load_subsets, windows, subsets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['axes.unicode_minus'] = False
    FIG.mkdir(parents=True, exist_ok=True)
    produced = []

    weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    block = load_subsets[(load_subsets.grouping == 'weekday_hour')
                         & load_subsets.subset.str.startswith('wd')]
    for predictor, group in block.groupby('predictor'):
        values = []
        for weekday in range(7):
            rows = group[group.subset.str.startswith(f'wd{weekday}_')]
            weight = rows.n.to_numpy()
            values.append(float(np.average(rows.mae_kW.to_numpy(), weights=weight)) if weight.sum() else np.nan)
        axes[0].plot(np.arange(7), values, marker='o', label=predictor)
    axes[0].set(title='Load MAE by weekday', xlabel='Weekday of 2025', ylabel='MAE (kW)',
                xticks=np.arange(7), xticklabels=weekdays)
    axes[0].grid(alpha=.2)
    axes[0].legend(fontsize=8)

    shape = load_subsets[(load_subsets.grouping == 'shape_slope')
                         & load_subsets.subset.isin(['slope_rising', 'slope_falling', 'slope_level'])]
    pivot = shape.pivot(index='subset', columns='predictor', values='mae_kW')
    order = ['slope_rising', 'slope_falling', 'slope_level']
    x = np.arange(len(order))
    for offset, predictor in enumerate(pivot.columns):
        axes[1].bar(x + (offset - 0.5) * 0.35, pivot.loc[order, predictor].to_numpy(), 0.35,
                    label=predictor)
    axes[1].set(title='Load MAE by local template slope segment', xlabel='segment', ylabel='MAE (kW)',
                xticks=x, xticklabels=['rising', 'falling', 'near level'])
    axes[1].grid(alpha=.2, axis='y')
    axes[1].legend(fontsize=8)
    path = FIG / 'q2_load_pv_load_error.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    produced.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    labels = ['valid_morning_tau_lt_0.2', 'valid_midday_tau_0.2_0.8', 'valid_evening_tau_gt_0.8',
              'outside_window']
    pivot = windows.pivot(index='subset', columns='predictor', values='mae_kW').reindex(labels)
    x = np.arange(len(labels))
    for offset, predictor in enumerate(pivot.columns):
        axes[0].bar(x + (offset - 0.5) * 0.35, np.nan_to_num(pivot[predictor].to_numpy()), 0.35,
                    label=predictor)
    axes[0].set(title='PV MAE by support-window segment', xlabel='segment', ylabel='MAE (kW)',
                xticks=x, xticklabels=['morning', 'midday', 'evening', 'outside'])
    axes[0].grid(alpha=.2, axis='y')
    axes[0].legend(fontsize=8)
    truth_labels = ['actual_zero', 'actual_0_1kW', 'actual_1_100kW', 'actual_gt_100kW']
    pivot = subsets[subsets.subset.isin(truth_labels)].pivot(
        index='subset', columns='predictor', values='mae_kW').reindex(truth_labels)
    x = np.arange(len(truth_labels))
    for offset, predictor in enumerate(pivot.columns):
        axes[1].bar(x + (offset - 0.5) * 0.35, np.nan_to_num(pivot[predictor].to_numpy()), 0.35,
                    label=predictor)
    axes[1].set(title='PV MAE by truth magnitude', xlabel='subset', ylabel='MAE (kW)',
                xticks=x, xticklabels=['0', '0-1', '1-100', '>100 kW'])
    axes[1].grid(alpha=.2, axis='y')
    axes[1].legend(fontsize=8)
    path = FIG / 'q2_load_pv_pv_error.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    produced.append(path)

    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    order = [g['id'] for g in GROUPS]
    block = summary.set_index('strategy_id').loc[order]
    x = np.arange(len(order))
    width = 0.27
    ax.bar(x - width, block.planned_cost_yuan / 1e3, width, label='planned')
    ax.bar(x, block.emergency_cost_yuan / 1e3, width, label='emergency (5x)')
    ax.bar(x + width, (block.planned_cost_yuan + block.emergency_cost_yuan) / 1e3, width,
           label='total')
    for index, value in enumerate((block.planned_cost_yuan + block.emergency_cost_yuan) / 1e3):
        ax.annotate(f'{value:,.1f}', (x[index] + width, value), ha='center', va='bottom', fontsize=8)
    ax.set(title='Realised cash cost by group (2025-02-01..12-31, q80)',
           xlabel='group', ylabel='Cost (thousand CNY)', xticks=x, xticklabels=order)
    ax.grid(alpha=.2, axis='y')
    ax.legend(fontsize=8)
    path = FIG / 'q2_load_pv_costs.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    produced.append(path)

    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    months = sorted(monthly.month.unique())
    x = np.arange(len(months))
    width = 0.27
    for offset, label in enumerate(PRIMARY_LABELS):
        block = monthly[monthly.comparison == label].set_index('month')
        ax.bar(x + (offset - 1) * width,
               [block.loc[m, 'delta_total_cost_yuan'] / 1e3 for m in months], width,
               label=label.split(' (')[0])
    ax.axhline(0, color='black', linewidth=.8)
    ax.set(title='Monthly realised cost difference versus F00_original', xlabel='Month of 2025',
           ylabel='Cost difference (thousand CNY)', xticks=x, xticklabels=[m[5:] for m in months])
    ax.grid(alpha=.2, axis='y')
    ax.legend(fontsize=8)
    path = FIG / 'q2_load_pv_monthly_delta.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    produced.append(path)
    return produced


# --------------------------------------------------------------------------------------
# registration and protected assets
# --------------------------------------------------------------------------------------
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
    p = pv_module()
    s.OUT = OUT

    load, pv, price, source_hashes = m.frozen().bm.read_sources()
    dates = m.frozen().bm.DATES
    assert load.shape == pv.shape == (365, 144) and price.shape == (144,)

    inputs = ['code/14_q2_ridge_forecast_experiment.py', 'code/19_q2_lightgbm_residual_experiment.py',
              'code/21_q2_quantile_level_scan.py', 'code/23_q2_pv_support_features_experiment.py',
              'code/25_q2_load_pv_shape_experiment.py',
              'results/q2_lightgbm_residual/G4_lightgbm_fixed/dispatch.csv',
              'results/q2_lightgbm_residual/G4_lightgbm_fixed/forecast_residuals.csv',
              'results/q2_lightgbm_residual/training_log.csv',
              'results/q2_lightgbm_residual/gate_mask.csv',
              'results/q2_quantile_level_scan/L_q80/dispatch.csv',
              'results/q2_pv_support_features/pv_forecast_archive_P1.csv',
              'results/q2_pv_support_features/support_windows.csv',
              'results/q2_pv_support_features/P1_q80/dispatch.csv',
              'results/q2_pv_support_features/checks.json',
              '附件/附件1.xlsx', '附件/附件2.xlsx',
              'reports/问题二_负载与光伏周期特征联合优化实验方案.md']
    snapshot_inputs = ['code/02_q1_baseline.py', 'code/05_q2_baseline.py',
                       'code/08_q2_quantile_experiment.py']
    hashes = {rel: digest(ROOT / rel) for rel in inputs}
    snapshot_hashes = {rel: digest(SNAP / rel) for rel in snapshot_inputs}
    for rel in snapshot_inputs:
        assert digest(ROOT / rel) == snapshot_hashes[rel], rel
    dependency = dependency_assertions()
    signature = hashlib.sha256(json.dumps(
        dict(parameters=PARAMETERS, dependency=dependency, input_sha256=hashes,
             snapshot_sha256=snapshot_hashes, code=digest(CODE_25)), sort_keys=True).encode()).hexdigest()
    registration_path = OUT / 'registration.json'
    if args.mode == 'register' or not registration_path.exists():
        record = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
                      specification='reports/问题二_负载与光伏周期特征联合优化实验方案.md',
                      parameters=PARAMETERS, dependency=dependency, input_sha256=hashes,
                      snapshot_sha256=snapshot_hashes, code_sha256=digest(CODE_25),
                      executable=sys.executable, python=sys.version, numpy=np.__version__,
                      pandas=pd.__version__)
        if registration_path.exists():
            existing = json.loads(registration_path.read_text(encoding='utf-8'))
            if existing['signature'] != signature:
                assert args.amend_reason, 'a different registration exists; pass --amend-reason'
                for key in ('load_control_features', 'load_candidate_features',
                            'pv_control_features', 'pv_candidate_features', 'alpha',
                            'support_parameters', 'load_template_weeks', 'load_shape_window_slots',
                            'train_window_days', 'min_train_days', 'residual_window_days',
                            'groups', 'comparisons', 'diagnostic_thresholds'):
                    assert existing['parameters'][key] == PARAMETERS[key], key
                amendments = existing.get('amendments', [])
                amendments.append(dict(amended_utc=datetime.now(timezone.utc).isoformat(),
                                       previous_signature=existing['signature'],
                                       previous_code_sha256=existing.get('code_sha256'),
                                       new_signature=signature, new_code_sha256=digest(CODE_25),
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

    boundary = boundary_checks(load, dates, pv)
    save(OUT / 'boundary_checks.json', boundary)
    assert boundary['all_passed'], 'boundary checks failed'
    print(f"boundary checks: {len(boundary['rows'])} passed", flush=True)

    template_parity = load_template_parity_check(load, dates)
    save(OUT / 'load_template_parity.json', template_parity)
    assert template_parity['all_passed'], template_parity
    extension = feature_extension_check(load, pv, dates)
    save(OUT / 'feature_extension.json', extension)
    assert extension['all_passed'], extension

    g4 = load_g4_archive(load, pv, dates)
    gate = p.gate_consistency_checks(pv, dates, g4)
    save(OUT / 'gate_consistency.json', gate)
    assert gate['all_passed'], gate
    print(f"gate reconstruction matches the frozen mask ({gate['gated_off_slots']} gated-off slots)",
          flush=True)

    tick = time.perf_counter()
    control_refit, load_control = refit_control_check(load, pv, dates, g4)
    timings['control_refit_seconds'] = time.perf_counter() - tick
    frame_to_csv(pd.DataFrame(load_control['log']), OUT / 'control_fit_log.csv')
    save(OUT / 'control_refit.json', control_refit)
    assert control_refit['all_passed'], control_refit
    print(f"control refit: load {control_refit['load_max_difference_kW']:.3e} kW, "
          f"pv {control_refit['pv_max_difference_kW']:.3e} kW", flush=True)

    tick = time.perf_counter()
    load_candidate = build_load_forecast(load, dates, 'shape19', save_models=True, verbose=True)
    timings['load_candidate_seconds'] = time.perf_counter() - tick
    frame_to_csv(pd.DataFrame(load_candidate['log']), OUT / 'candidate_load_fit_log.csv')
    tick = time.perf_counter()
    pv_candidate = p.build_pv_forecast(pv, dates, 'support12', save_models=True, verbose=True)
    timings['pv_candidate_seconds'] = time.perf_counter() - tick
    frame_to_csv(pd.DataFrame(pv_candidate['log']), OUT / 'candidate_pv_fit_log.csv')

    load_parity = training_parity_checks(load_candidate, g4, 'load')
    pv_parity = training_parity_checks(pv_candidate, g4, 'pv')
    parity = dict(load=load_parity, pv=pv_parity,
                  all_passed=bool(load_parity['all_passed'] and pv_parity['all_passed']))
    save(OUT / 'training_parity.json', parity)
    assert parity['all_passed'], parity
    print(f"training parity: load first day {load_parity['first_trainable_day']}, "
          f"pv first day {pv_parity['first_trainable_day']}", flush=True)

    templates = load_shape_frame(load, dates)
    frame_to_csv(templates, OUT / 'load_templates.csv')
    support_check = support_window_check(pv, pv_candidate['support'], dates)
    save(OUT / 'support_window_check.json', support_check)
    assert support_check['all_passed'], support_check
    support_windows = np.vstack([p.support_window(pv, k) for k in range(len(dates))])
    support = p.support_summary_frame(pv, dates, support_windows)
    frame_to_csv(support, OUT / 'support_windows.csv')
    save(OUT / 'load_shape_archive.json', save_load_shape_archive(load, dates))
    print(f"support windows: day 0 = {support_check['day0_rebuilt']} (documented default), "
          f"{support_check['mismatch_from_day1']} mismatches from day 1", flush=True)

    archives = {
        'L0': dict(load_forecast=np.asarray(g4['load_forecast'], dtype=float), kind='frozen_g4'),
        'L1': dict(load_forecast=np.asarray(load_candidate['published'], dtype=float), kind='shape19'),
        'V0': dict(pv_forecast=np.asarray(g4['pv_forecast'], dtype=float), kind='frozen_g4'),
        'V1': dict(pv_forecast=np.asarray(pv_candidate['published'], dtype=float), kind='support12'),
    }
    group_archives = {'F00_original': ('L0', 'V0'), 'F10_load_shape': ('L1', 'V0'),
                      'F01_pv_support': ('L0', 'V1'), 'F11_joint': ('L1', 'V1')}

    population = population_identity_check(archives, group_archives, pv)
    save(OUT / 'population_identity.json', population)
    assert population['all_passed'], population
    print(f"population identity: shared load/PV archives exact, "
          f"{population['gate']['gated_off_slots']} gated-off slots with zero published PV", flush=True)

    for name in ('L0', 'L1', 'V0', 'V1'):
        frame_to_csv(pd.DataFrame({
            'date': np.repeat([str(d.date()) for d in dates], 144),
            'slot': np.tile(np.arange(144), len(dates)),
            'load_forecast_kW': (np.asarray(archives[name]['load_forecast']).ravel()
                                 if name.startswith('L')
                                 else np.asarray(archives['L0']['load_forecast']).ravel()),
            'pv_forecast_kW': (np.asarray(archives[name]['pv_forecast']).ravel()
                               if name.startswith('V')
                               else np.asarray(archives['V0']['pv_forecast']).ravel()),
            'support_start_slot': np.repeat(support_windows[:, 0], 144),
            'support_end_slot': np.repeat(support_windows[:, 1], 144),
            'support_valid': np.repeat(support_windows[:, 2], 144),
            'support_defined': np.repeat(np.arange(len(dates)) > 0, 144)}),
            OUT / f'forecast_archive_{name}.csv')

    group_errors = {}
    for group_id, (load_name, pv_name) in group_archives.items():
        group_errors[group_id] = ((load - pv) * DT
                                 - (archives[load_name]['load_forecast']
                                    - archives[pv_name]['pv_forecast']) * DT)
    quantile = quantile_validation(group_errors)
    save(OUT / 'quantile_validation.json', quantile)
    assert quantile['all_passed'], quantile
    print(f"quantile validation passed (m=28 position {quantile['positions_at_28_days']})", flush=True)

    print('running the public January warm-up ...', flush=True)
    warm_states, warm_frame, warm_end, warm_checks = m.frozen().m08.run_warmup(
        m.frozen().m08.Archive(load, pv, len(dates)), price, dates)
    frame_to_csv(warm_frame, OUT / 'warmup_january.csv')
    assert abs(warm_end - REFERENCE_WARMUP_KWH) < 1e-9, warm_end
    print(f'  2025-02-01 shared begin state = {warm_end:.12f} kWh', flush=True)

    instances = {}
    for group_id, (load_name, pv_name) in group_archives.items():
        s.require_finite_published(archives[load_name]['load_forecast'],
                                   archives[pv_name]['pv_forecast'], len(dates))
        instances[group_id] = s.ScanArchive(load, pv, len(dates),
                                            archives[load_name]['load_forecast'],
                                            archives[pv_name]['pv_forecast'])

    results = []
    reference_reproduction = {}
    control = s.run_scenario(CONTROL_GROUP, load, pv, price, dates, instances[CONTROL_ID],
                            warm_states, verbose=False)
    results.append(control)
    print(f"  {CONTROL_ID}: {control['totals']['total_cost_yuan']:,.6f} CNY", flush=True)
    for label, reference_path, registered in (
            ('G4_lightgbm_fixed', FROZEN_RUN / 'G4_lightgbm_fixed' / 'dispatch.csv', CONTROL_TOTAL_YUAN),
            ('L_q80', SCAN_RUN / 'L_q80' / 'dispatch.csv', CONTROL_TOTAL_YUAN)):
        record = compare_to_reference(control, CONTROL_ID, reference_path, registered)
        reference_reproduction[label] = record
        assert record['passed'], record
        print(f"    reproduces {label}: day-level max diff {record['max_day_level_difference']:.3e} kWh, "
              f"cost diff {record['difference_vs_registered_yuan']:.3e} CNY", flush=True)
    save(OUT / 'reference_reproduction.json', reference_reproduction)
    if args.mode == 'repro':
        return

    for group in TREATMENT_GROUPS:
        tick = time.perf_counter()
        result = s.run_scenario(group, load, pv, price, dates, instances[group['id']],
                                warm_states, verbose=True)
        timings[f"scenario_{group['id']}_seconds"] = time.perf_counter() - tick
        results.append(result)
        print(f"  {group['id']}: {result['totals']['total_cost_yuan']:,.6f} CNY", flush=True)

    cross = cross_module_pv_check(
        pv_candidate, next(r for r in results if r['totals']['strategy_id'] == 'F01_pv_support'))
    save(OUT / 'cross_module_pv.json', cross)
    assert cross['all_passed'], cross
    print(f"F01 reproduces module 23 P1_q80: PV max diff {cross['pv_forecast_max_difference_kW']:.3e} kW, "
          f"cost diff {cross['dispatch']['difference_vs_registered_yuan']:.3e} CNY", flush=True)

    summary = summary_frame(results)
    frame_to_csv(summary, OUT / 'summary.csv')
    contrasts = contrast_frame(results)
    frame_to_csv(contrasts, OUT / 'contrasts.csv')
    monthly = monthly_contrast_frame(results)
    frame_to_csv(monthly, OUT / 'monthly_contrasts.csv')
    stability = stability_frame(results, monthly)
    frame_to_csv(stability, OUT / 'stability.csv')
    energy = energy_frame(results)
    frame_to_csv(energy, OUT / 'energy_contrasts.csv')
    frame_to_csv(inventory_adjusted_frame(results, price), OUT / 'inventory_adjusted.csv')
    metrics = predictor_metric_frame(load, pv, dates, archives)
    frame_to_csv(metrics, OUT / 'predictor_metrics.csv')
    net_metrics = net_demand_metric_frame(load, pv, dates, archives, group_archives)
    frame_to_csv(net_metrics, OUT / 'net_demand_metrics.csv')
    shapes = descriptive_masks(load, dates)
    load_subsets = load_error_frame(load, dates, archives, shapes)
    frame_to_csv(load_subsets, OUT / 'load_error_subsets.csv')
    subsets = p.pv_subset_frame(pv, dates, {'V0': archives['V0'], 'V1': archives['V1']})
    frame_to_csv(subsets, OUT / 'pv_error_subsets.csv')
    windows = p.window_subset_frame(pv, dates, {'V0': archives['V0'], 'V1': archives['V1']},
                                    support_windows, p.support_mask(pv))
    frame_to_csv(windows, OUT / 'pv_error_by_window.csv')
    frame_to_csv(p.monthly_error_frame(pv, dates, {'V0': archives['V0'], 'V1': archives['V1']}),
                 OUT / 'pv_error_monthly.csv')
    frame_to_csv(p.hourly_error_frame(pv, dates, {'V0': archives['V0'], 'V1': archives['V1']}),
                 OUT / 'pv_error_hourly.csv')
    frame_to_csv(p.load_weekday_hour_frame(load, dates, archives['L0']['load_forecast']),
                 OUT / 'load_weekday_hour_residual.csv')
    coverage = summary[['strategy_id', 'mean_protection_kWh', 'coverage']].copy()
    frame_to_csv(coverage, OUT / 'coverage.csv')

    by_group = {result['totals']['strategy_id']: result for result in results}
    clean = {'initial': {}, 'executed': {}}
    for group in GROUPS:
        clean[group['id']] = instances[group['id']]
        clean[group['id'] + '_dispatch'] = by_group[group['id']]['dispatch']
        for k in PERTURBATION_DAYS:
            clean['initial'][(group['id'], k)] = float(by_group[group['id']]['daily']
                                                       .iloc[k - WARMUP_DAYS].initial_kWh)
    guard = dict(L0=archives['L0']['load_forecast'], V0=archives['V0']['pv_forecast'])

    perturbation = perturbation_checks(load, pv, price, dates, clean, load_candidate, pv_candidate, guard)
    save(OUT / 'future_perturbation_checks.json', perturbation)
    assert perturbation['all_passed'], perturbation
    prefix = feedback_prefix_check(load, pv, price, dates, clean, load_candidate, pv_candidate, guard)
    save(OUT / 'feedback_prefix_check.json', prefix)
    assert prefix['all_passed'], prefix
    sampled = sampled_retraining_checks(load, pv, dates, g4, load_candidate, pv_candidate)
    save(OUT / 'sampled_retraining.json', sampled)
    assert sampled['all_passed'], sampled
    sampled_milp = sampled_milp_checks(results, price, dates)
    save(OUT / 'sampled_milp.json', sampled_milp)
    assert sampled_milp['all_passed'], sampled_milp
    print(f"perturbation {len(perturbation['cases'])} cases, prefix {len(prefix['cases'])} cases, "
          f"retraining {sampled['checks']} checks, re-solved MILP {sampled_milp['solves']}", flush=True)

    tick = time.perf_counter()
    produced = figures(summary, monthly, load_subsets, windows, subsets)
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
    checks = dict(boundary=boundary, template_parity=template_parity, extension=extension, gate=gate,
                  population=population, support_windows=support_check,
                  control_refit=control_refit, load_parity=load_parity, pv_parity=pv_parity,
                  quantile=quantile, reference_reproduction=reference_reproduction, cross_module=cross,
                  perturbation=perturbation, feedback_prefix=prefix, sampled_retraining=sampled,
                  sampled_milp=sampled_milp, energy=energy.to_dict('records'), physical=physical,
                  physical_max=float(physical_max),
                  protected_unchanged=bool(not changed and not missing),
                  protected_changed=changed, protected_missing=missing, protected_count=len(protected),
                  warmup=dict(shared_begin_state_kWh=warm_end,
                              checks={key: float(value) for key, value in warm_checks.items()}),
                  timings=timings,
                  thresholds=dict(energy_kWh=ENERGY_TOL_KWH, cost_yuan=COST_TOL_YUAN))
    save(OUT / 'checks.json', checks)

    write_report(dict(summary=summary, contrasts=contrasts, monthly=monthly, stability=stability,
                      energy=energy, metrics=metrics, net_metrics=net_metrics,
                      load_subsets=load_subsets, subsets=subsets, windows=windows,
                      templates=templates, support=support, checks=checks, gate=gate,
                      parity=parity, integrity=integrity, warm_end=warm_end, timings=timings,
                      load_summary=load_candidate['summary'], pv_summary=pv_candidate['summary'],
                      load_log=load_candidate['log'], pv_log=pv_candidate['log'],
                      inventory=inventory_adjusted_frame(results, price),
                      coverage=coverage))
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
                    load_summary=load_candidate['summary'], pv_summary=pv_candidate['summary'],
                    scenario_totals={result['totals']['strategy_id']: result['totals']
                                     for result in results},
                    significance=dict(
                        load_comparison=float(contrasts.iloc[0].delta_total_cost_yuan),
                        pv_comparison=float(contrasts.iloc[1].delta_total_cost_yuan),
                        joint_comparison=float(contrasts.iloc[2].delta_total_cost_yuan),
                        interaction=float(contrasts.iloc[5].delta_total_cost_yuan)),
                    reference_reproduction=reference_reproduction, cross_module=cross,
                    timings=timings,
                    self_checks_passed=dict(boundary=boundary['all_passed'],
                                            template_parity=template_parity['all_passed'],
                                            extension=extension['all_passed'], gate=gate['all_passed'],
                                            population=population['all_passed'],
                                            support_windows=support_check['all_passed'],
                                            control_refit=control_refit['all_passed'],
                                            parity=parity['all_passed'], quantile=quantile['all_passed'],
                                            cross_module=cross['all_passed'],
                                            perturbation=perturbation['all_passed'],
                                            feedback_prefix=prefix['all_passed'],
                                            sampled_retraining=sampled['all_passed'],
                                            sampled_milp=sampled_milp['all_passed']),
                    artifact_count=len(artifact_hashes), report=str(REPORT_MD.relative_to(ROOT)))
    save(OUT / 'run_manifest.json', manifest)
    assert manifest['protected_unchanged'], (changed, missing)
    print(summary[['strategy_id', 'planned_cost_yuan', 'emergency_cost_yuan', 'total_cost_yuan',
                   'emergency_days']].to_string(index=False), flush=True)
    print(contrasts[['comparison', 'delta_total_cost_yuan', 'delta_total_cost_percent',
                     'delta_emergency_days']].to_string(index=False), flush=True)
    print(f'self-checks {manifest["self_checks_passed"]}', flush=True)
    print(f'wall seconds = {time.perf_counter() - started:.1f}', flush=True)
    return manifest


def write_report(context):
    """Report generator (kept below main so a syntax error here cannot corrupt the run)."""
    summary = context['summary']
    contrasts = context['contrasts']
    monthly = context['monthly']
    stability = context['stability']
    energy = context['energy']
    metrics = context['metrics']
    net_metrics = context['net_metrics']
    load_subsets = context['load_subsets']
    subsets = context['subsets']
    windows = context['windows']
    templates = context['templates']
    support = context['support']
    checks = context['checks']
    gate = context['gate']
    parity = context['parity']
    integrity = context['integrity']
    warm_end = context['warm_end']
    timings = context['timings']
    load_summary = context['load_summary']
    pv_summary = context['pv_summary']
    inventory = context['inventory']
    coverage = context['coverage']

    def money(value):
        return f'{value:,.2f}'

    def condition_word(value):
        """'贵' when the treatment costs more, '省' when it costs less."""
        return '贵' if value > 0 else '省'

    def delta(label, field):
        row = contrasts[contrasts.comparison == label]
        return float(row.iloc[0][field])

    main = summary.set_index('strategy_id')
    primary_rows = '\n'.join(
        f'| {row.comparison.split(" (")[0]} | {row.treatment} − {row.baseline} '
        f'| {money(row.delta_planned_cost_yuan)} | {money(row.delta_emergency_cost_yuan)} '
        f'| {money(row.delta_total_cost_yuan)} | {row.delta_total_cost_percent:.4f} '
        f'| {int(row.delta_emergency_days):+d} |'
        for row in contrasts.iloc[:5].itertuples())
    summary_rows = '\n'.join(
        f'| {row.strategy_id} | {money(row.planned_cost_yuan)} | {money(row.emergency_cost_yuan)} '
        f'| {money(row.total_cost_yuan)} | {row.unused_kWh:,.1f} | {row.loss_kWh:,.1f} '
        f'| {row.final_kWh:,.1f} | {row.emergency_days} | {row.emergency_events} |'
        for row in summary.itertuples())
    metric_rows = '\n'.join(
        f'| {row.predictor} | {row.scope} | {row.mae_kW:,.6f} | {row.rmse_kW:,.6f} | {row.bias_kW:,.6f} |'
        for row in metrics.itertuples())
    net_rows = '\n'.join(
        f'| {row.group} | {row.load_source} | {row.pv_source} | {row.mae_kW:,.6f} '
        f'| {row.rmse_kW:,.6f} | {row.bias_kW:,.6f} |' for row in net_metrics.itertuples())
    monthly_rows = '\n'.join(
        f'| {row.comparison.split(" (")[0]} | {row.month} | {money(row.delta_planned_cost_yuan)} '
        f'| {money(row.delta_emergency_cost_yuan)} | {money(row.delta_total_cost_yuan)} '
        f'| {row.delta_emergency_kWh:,.2f} |' for row in monthly.itertuples())
    stability_rows = '\n'.join(
        f'| {row.comparison.split(" (")[0]} | {row.months_improved}/{row.months_total} '
        f'| {row.days_improved} | {row.days_worse} | {row.worst_day} | {money(row.worst_day_yuan)} '
        f'| {money(row.delta_emergency_cost_0_10h)} | {money(row.delta_emergency_cost_19_21h)} '
        f'| {money(row.delta_emergency_cost_21_24h)} |' for row in stability.itertuples())
    energy_rows = '\n'.join(
        f'| {row.comparison.split(" (")[0]} | {row.delta_planned_kWh:,.2f} '
        f'| {row.delta_emergency_kWh:,.2f} | {row.delta_unused_kWh:,.2f} | {row.delta_loss_kWh:,.2f} '
        f'| {row.delta_end_kWh:,.2f} | {row.identity_residual_kWh:.3e} |' for row in energy.itertuples())
    load_subset_rows = '\n'.join(
        f'| {row.predictor} | {row.grouping} | {row.subset} | {row.n} | {row.mae_kW:,.6f} '
        f'| {row.bias_kW:,.6f} |'
        for row in load_subsets[load_subsets.grouping != 'weekday_hour'].itertuples())
    window_rows = '\n'.join(
        f'| {row.predictor} | {row.subset} | {row.n} | {row.mae_kW:,.6f} | {row.bias_kW:,.6f} |'
        for row in windows.itertuples())
    subset_rows = '\n'.join(
        f'| {row.predictor} | {row.subset} | {row.n} | {row.mae_kW:,.6f} | {row.bias_kW:,.6f} |'
        for row in subsets.itertuples())
    integrity_rows = '\n'.join(
        f"- `figures/q2_load_pv_shape/{row['file']}`：{row['width']}×{row['height']} 像素，"
        f"非白像素比例 {row['non_white_fraction']:.4f}，颜色数 {row['distinct_colours']}。"
        for row in integrity)
    inventory_rows = '\n'.join(
        f'| {row.strategy_id} | {money(row.total_cost_yuan)} | {money(row.inventory_adjustment_yuan)} '
        f'| {money(row.adjusted_cost_yuan)} |' for row in inventory.itertuples())
    coverage_rows = '\n'.join(
        f'| {row.strategy_id} | {row.mean_protection_kWh:,.6f} | {row.coverage:.6f} |'
        for row in coverage.itertuples())
    constant_load = int((load_summary['n_fitted'] + load_summary['n_constant']
                         + load_summary['n_insufficient'] + load_summary['n_feature_fallback']))
    template_days = templates[templates.n_template_days > 0]
    constant_template_days = int(template_days.near_constant_template.sum())
    mean_template_days = float(template_days.n_template_days.mean())
    interaction = delta(INTERACTION_LABEL, 'delta_total_cost_yuan')

    load0 = metrics[(metrics.predictor == 'L0')].iloc[0]
    load1 = metrics[(metrics.predictor == 'L1')].iloc[0]
    net00 = net_metrics[net_metrics.group == 'F00_original'].iloc[0]
    net10 = net_metrics[net_metrics.group == 'F10_load_shape'].iloc[0]
    stability9 = stability[stability.comparison == CONTRASTS[0][0]].iloc[0]
    stability10 = stability[stability.comparison == CONTRASTS[1][0]].iloc[0]
    stability11 = stability[stability.comparison == CONTRASTS[2][0]].iloc[0]
    month10_jun = monthly[(monthly.comparison == CONTRASTS[0][0]) & (monthly.month == '2025-06')].iloc[0]
    month11_jun = monthly[(monthly.comparison == CONTRASTS[2][0]) & (monthly.month == '2025-06')].iloc[0]
    month11_jul = monthly[(monthly.comparison == CONTRASTS[2][0]) & (monthly.month == '2025-07')].iloc[0]

    def tree_stats(log):
        fitted = [e['n_trees'] for e in log if e['branch'] == 'fitted']
        if not fitted:
            return '无'
        return (f'{len(fitted)} 天拟合，平均 {np.mean(fitted):.1f} 棵、'
                f'最少 {min(fitted)}、最多 {max(fitted)} 棵')

    def constant_new_column_days(log):
        keys = [name for name in LOAD_SHAPE_FEATURES + PV_SHAPE_FEATURES]
        counts = {name: 0 for name in keys}
        for entry in log:
            value = entry.get('constant_new_columns')
            if not isinstance(value, str):
                continue                       # empty in memory, NaN after a CSV round-trip
            for name in value.split(','):
                if name in counts:
                    counts[name] += 1
        return '、'.join(f'{name} {counts[name]} 天' for name in keys if counts[name]) or '无'

    report = f"""# 问题二：负载与光伏周期特征联合优化实验结果报告

日期：2026-09-11。状态：**实验自检完成，独立审计待完成**。本轮实现、运行并自检了
`reports/问题二_负载与光伏周期特征联合优化实验方案.md` 登记的四组同年因果回测（固定 q80）；
未锁定第二问最终模型，未填写 `result2.xlsx`，未修改第一问。

## 1. 问题分析

日前不知道当天真实供需，计划电量全额付款，缺口按当段 5 倍价应急。本轮把负载预测也纳入消融：
在负载 LightGBM 残差修正的 15 列输入后追加 **同星期历史模板的日均水平、日内相对高低、向前与向后
1 小时斜率**四列（共 19 列），在光伏 LightGBM 残差修正的 8 列输入后追加 **前 7 日发电支持窗口的
长度、中点、相对位置与有效标记**四列（共 12 列），观察四项对照差。

四组只改变负载与光伏发布预测：
`F00_original`（两者都是修正 G4 档案）、`F10_load_shape`（只改负载）、`F01_pv_support`（只改光伏）、
`F11_joint`（两者同时改）。保护水平固定 q80，W28 逐时段经验分位、名义日末 6000 kWh、冻结 MILP
内核、因果贪心反馈、收费规则与公共 1 月预运行完全相同；**每组分别重建自己的净需求误差与保护量，
单改组的保护量不相加**。因此组间差额可归因于输入列，不能归因于"新增特征是最优特征"，也不能外推。

### 1.1 主要结论

- 控制复现：`F00_original` 总费 {money(float(main.loc['F00_original', 'total_cost_yuan']))} 元，
  与登记参考差 {checks['reference_reproduction']['G4_lightgbm_fixed']['difference_vs_registered_yuan']:.3e} 元
  （与 21 号 `L_q80` 差 {checks['reference_reproduction']['L_q80']['difference_vs_registered_yuan']:.3e} 元）。
- 只改负载 ΔC_L = F10 − F00：{money(delta(CONTRASTS[0][0], 'delta_total_cost_yuan'))} 元
  （{delta(CONTRASTS[0][0], 'delta_total_cost_percent'):.4f}%），应急天数
  {int(delta(CONTRASTS[0][0], 'delta_emergency_days')):+d} 天。
- 只改光伏 ΔC_V = F01 − F00：{money(delta(CONTRASTS[1][0], 'delta_total_cost_yuan'))} 元
  （{delta(CONTRASTS[1][0], 'delta_total_cost_percent'):.4f}%），应急天数
  {int(delta(CONTRASTS[1][0], 'delta_emergency_days')):+d} 天。
- 联合 ΔC_LV = F11 − F00：{money(delta(CONTRASTS[2][0], 'delta_total_cost_yuan'))} 元
  （{delta(CONTRASTS[2][0], 'delta_total_cost_percent'):.4f}%），应急天数
  {int(delta(CONTRASTS[2][0], 'delta_emergency_days')):+d} 天。
- 交互项 I_C = C11 − C10 − C01 + C00 = {money(interaction)} 元；
  {"组合低于两项单独下降之和" if interaction < 0 else ("组合高于两项单独下降之和" if interaction > 0 else "组合与两项单独效果之和数值并列")}，
  但分位数与储能非线性使费用本不必可加，这不是统计显著性证据。
- 条件差：F11 − F01（负载在 F01 之上）= {money(delta(CONTRASTS[3][0], 'delta_total_cost_yuan'))} 元，
  F11 − F10（光伏在 F10 之上）= {money(delta(CONTRASTS[4][0], 'delta_total_cost_yuan'))} 元。

## 2. 数据预处理

只使用附件1 的 144 点日内电价与附件2 的 365×144 实际负载、光伏功率；不使用附件3/4、未来实测或
外部天气。不平滑、不删点、不插补。`F00` 复现修正 G4 发布档案，其训练因果性仍属上游 20 号审计范围。

### 2.1 输入核对与残差诊断

- 门控一致性（全年 365 天口径）：重建掩码与冻结 `gate_mask.csv` 不一致 **{gate['mask_mismatch_slots']}** 段；
  被归零 {gate['gated_off_slots']} 段，其中真值 >100 kW 的时段 {gate['zeroed_truth_above_100kW_slots']} 个，
  被归零位置真值发电合计 {gate['zeroed_truth_energy_kWh']:.6f} kWh。四组门控完全相同。
- 负载模板覆盖：365 天中有效模板天数 {len(template_days)}，平均可用同星期日 {mean_template_days:.2f} 个；
  其中近常数模板（极差 ≤ {LOAD_NEAR_CONSTANT_TOL_KW:g} kW）{constant_template_days} 天，
  相对高低列固定为 0，仅记日志，不额外增加列；无模板的早期天数同样按缺省处理。
- 光伏支持窗口：v=1 共 {int((support.valid == 1).sum())} 天，v=0 共 {int((support.valid == 0).sum())} 天；
  v=1 时窗口时长 {support[support.valid == 1].duration_h.min():.2f}—{support[support.valid == 1].duration_h.max():.2f} 小时，
  中点 {support[support.valid == 1].midpoint_h.min():.2f}—{support[support.valid == 1].midpoint_h.max():.2f} 小时。

负载发布误差按描述性分组（kW，偏差=实际−预测，评价期 334 天 48096 段）：

| 预测器 | 分组 | 子集 | 样本数 | MAE | 偏差 |
|---|---|---|---|---:|---:|---:|
{load_subset_rows}

光伏发布误差按支持窗口位置分组（同一窗口口径，两预测器共用）：

| 预测器 | 子集 | 样本数 | MAE | 偏差 |
|---|---|---:|---:|---:|
{window_rows}

光伏误差按真值量级分组：

| 预测器 | 子集 | 样本数 | MAE | 偏差 |
|---|---|---:|---:|---:|
{subset_rows}

分月与分小时明细见 `load_error_subsets.csv`、`pv_error_monthly.csv`、`pv_error_hourly.csv`、
`pv_error_by_window.csv`。

### 2.2 每日历史模板与支持窗口

负载模板 $H_k$ 取最近最多 4 个同星期日的等权平均（$k-7r\\ge 0$），训练日 $j$ 只用它自己的
$H_j$；"向前 1 小时"是 $H$ 在时段轴上的局部窗口，第 $k$ 日 0:00 已拥有全部 144 段，不读取预测日
实测，边缘窗口缩短并按实际时长归一化，不跨午夜循环。

光伏窗口沿用：$\\mathcal A_k=\\{{t:\\max_{{j=k-7,\\ldots,k-1}}V_{{j,t}}>1\\ \\mathrm{{kW}}\\}}$，
非空则 $s_k=\\max(0,\\min\\mathcal A_k-3)$、$e_k=\\min(143,\\max\\mathcal A_k+3)$、$v_k=1$，否则
$s_k=0,e_k=143,v_k=0$。严格大于 1 kW；训练日也只用自己当日的窗口。

训练行边界：原始与候选训练日的行键、标签、训练日数、分支与首次可训练日完全一致
（负载首训日 {parity['load']['first_trainable_day']}，光伏首训日 {parity['pv']['first_trainable_day']}，
不一致项 {parity['load']['mismatch_count'] + parity['pv']['mismatch_count']}）。
实际树数：负载 {tree_stats(context['load_log'])}；光伏 {tree_stats(context['pv_log'])}。
新增列在训练窗内恒定（被如实保留，未删列）的天数：负载 {constant_new_column_days(context['load_log'])}；
光伏 {constant_new_column_days(context['pv_log'])}；逐日记录见 `candidate_load_fit_log.csv`、
`candidate_pv_fit_log.csv`。

## 3. 模型建立

两个目标都继续预测**朴素功率残差**：$y^L_{{j,t}}=L_{{j,t}}-L_{{j-7,t}}$、
$y^V_{{j,t}}=V_{{j,t}}-V_{{j-1,t}}$；发布
$\\widehat L^{{(a)}}_{{k,t}}=\\max\\{{0,b^L_{{k,t}}+f^L_k(\\boldsymbol x^{{L,19}}_{{k,t}})\\}}$、
$\\widehat V^{{(b)}}_{{k,t}}=g_{{k,t}}\\max\\{{0,V_{{k-1,t}}+f^V_k(\\boldsymbol x^{{V,12}}_{{k,t}})\\}}$。
负载不门控、不削峰；光伏统一门控。

负载新增四列（单位 kW、无量纲、kW/h、kW/h）：
$\\mu_k=\\frac1{{144}}\\sum_t H_{{k,t}}$、
$z_{{k,t}}=(H_{{k,t}}-h_k^{{\\min}})/A_k$（$A_k\\le 10^{{-8}}$ kW 时固定 0）、
$u^-_{{k,t}}=(H_{{k,t}}-H_{{k,a_t}})/[(t-a_t)\\Delta t]$、
$u^+_{{k,t}}=(H_{{k,b_t}}-H_{{k,t}})/[(b_t-t)\\Delta t]$，
$a_t=\\max(0,t-6)$、$b_t=\\min(143,t+6)$，端点取 0。
光伏新增四列：$D_k=(e_k-s_k+1)\\Delta t$、$M_k=(s_k+e_k+1)\\Delta t/2$、
$\\tau_{{k,t}}=(t-s_k+1/2)/(e_k-s_k+1)$、$v_k$。

每日每个目标一个跨 144 时段共享 LightGBM，配置与 19 号逐项相同（gbdt/平方损失、100 轮、
学习率 0.05、7 叶、深度 3、`min_child_samples=144`、L2=1、全部随机种子 20250911）；
不网格搜索、不早停、不用 eval_set。训练集为决策日前 56 个日历日内的资格日（至少 14 日）：
负载首训 2025-01-23（k=22），光伏首训 2025-01-22（k=21）；标签极差 ≤1e-8 kW 取常数均值；
早期按公共朴素回退发布。负载 **没有**窗口外归零或强制削峰。

每组由**自身**发布的两个预测重建净需求误差并使用 W28 逐时段经验逆分布保护（q80 为升序第 23 个，
m<7 回退零修正）；联合组的保护量由联合净需求误差独立计算，**不是**两单改组保护量之和。
日前模型为 $\\min\\sum p_tq_t$，约束与冻结内核一致（$q+d=\\tilde n+c+w$、
$E_{{t+1}}=E_t+0.9c_t-d_t/0.9$、$0\\le c_t\\le(5000/6)z_t$、$0\\le d_t\\le(5000/6)(1-z_t)$、
$1200\\le E_t\\le10800$、$E_0=$ 当日实际初态、$E_{{144}}=6000$），gap 目标 1e-9、时限 120 秒。
实际执行按富余充电、缺口放电、剩余 5 倍价应急，实际末态逐日继承；1 月沿用公共冷启动。

## 4. 模型求解与结果

公共 1 月预运行结束状态 {warm_end:.12f} kWh，四组从此分叉并各自继承真实末态。
新建两个目标档案耗时：负载 {timings.get('load_candidate_seconds', float('nan')):.1f} 秒
（拟合 {load_summary['n_fitted']} 天、常数 {load_summary['n_constant']} 天、
历史不足 {load_summary['n_insufficient']} 天、早期回退 {load_summary['n_feature_fallback']} 天，共 {constant_load} 天），
光伏 {timings.get('pv_candidate_seconds', float('nan')):.1f} 秒
（拟合 {pv_summary['n_fitted']} 天、常数 {pv_summary['n_constant']} 天、
历史不足 {pv_summary['n_insufficient']} 天、早期回退 {pv_summary['n_feature_fallback']} 天）。

### 4.1 四组总费用

| 组 | 计划费/元 | 应急费/元 | 总费/元 | 未使用/kWh | 损耗/kWh | 期末/kWh | 应急天 | 应急事件 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{summary_rows}

### 4.2 组间差额

| 比较 | | Δ计划费/元 | Δ应急费/元 | Δ总费/元 | Δ总费/% | Δ应急天 |
|---|---:|---:|---:|---:|---:|
{primary_rows}

| 比较 | Δ总费/元 |
|---|---:|
| {INTERACTION_LABEL} | {money(interaction)} |

结果解读（不预设联合更好，也不只挑最省的一组）：

1. **F10 只改负载**：负载点预测 MAE 从 {load0.mae_kW:,.6f} 变为 {load1.mae_kW:,.6f} kW、
   RMSE 从 {load0.rmse_kW:,.6f} 变为 {load1.rmse_kW:,.6f} kW，但偏差从 {load0.bias_kW:+,.6f} 变为
   {load1.bias_kW:+,.6f} kW；净需求 MAE 从 {net00.mae_kW:,.6f} 变为 {net10.mae_kW:,.6f} kW，
   而净需求 RMSE 从 {net00.rmse_kW:,.6f} 变为 {net10.rmse_kW:,.6f} kW、偏差从 {net00.bias_kW:+,.6f}
   变为 {net10.bias_kW:+,.6f} kW。现金总费反而增加 {money(abs(delta(CONTRASTS[0][0], 'delta_total_cost_yuan')))} 元
   （计划费 {money(delta(CONTRASTS[0][0], 'delta_planned_cost_yuan'))} 元、
   应急费 {money(delta(CONTRASTS[0][0], 'delta_emergency_cost_yuan'))} 元），
   且**应急天数由 {int(main.loc['F00_original', 'emergency_days'])} 降到
   {int(main.loc['F10_load_shape', 'emergency_days'])}，应急费却上升**，
   即本轮回测里应急变成更少但更深或落在更贵时段的事件。按方案第 4.6 节，精度（部分）改善但不节费
   不能称负载优化已经改善最终目标：**本组不满足节费判据**。
2. **F01 只改光伏**：总费变化 {money(delta(CONTRASTS[1][0], 'delta_total_cost_yuan'))} 元，主要来自应急费
   {money(delta(CONTRASTS[1][0], 'delta_emergency_cost_yuan'))} 元，应急天数
   {int(delta(CONTRASTS[1][0], 'delta_emergency_days')):+d} 天；
   {int(stability10.months_improved)}/{int(stability10.months_total)} 个月更省，
   最差单日 {stability10.worst_day} 变差 {money(abs(float(stability10.worst_day_yuan)))} 元。
3. **F11 联合**：总费变化 {money(delta(CONTRASTS[2][0], 'delta_total_cost_yuan'))} 元，
   比只改光伏的 F01 {condition_word(delta(CONTRASTS[3][0], 'delta_total_cost_yuan'))}
   {money(abs(delta(CONTRASTS[3][0], 'delta_total_cost_yuan')))} 元，
   比只改负载的 F10 {condition_word(delta(CONTRASTS[4][0], 'delta_total_cost_yuan'))}
   {money(abs(delta(CONTRASTS[4][0], 'delta_total_cost_yuan')))} 元。
   交互项 I_C = {money(interaction)} 元，而两项单独差额之和为
   {money(delta(CONTRASTS[1][0], 'delta_total_cost_yuan') + delta(CONTRASTS[0][0], 'delta_total_cost_yuan'))} 元：
   本次连续回测中把负载四列与光伏四列叠加**并不比只改光伏更好**。
   这不解释为两类特征在物理上互相抵消，也不构成总体显著性或因果机制证明。
4. 反例与权衡按实保留：F10 有 {int(stability9.months_improved)}/{int(stability9.months_total)} 个月更省，
   但被 {month10_jun.month} 的 {money(month10_jun.delta_total_cost_yuan)} 元主导；
   F11 有 {int(stability11.months_improved)}/{int(stability11.months_total)} 个月更省，
   但被 {month11_jun.month} 的 {money(month11_jun.delta_total_cost_yuan)} 元与
   {month11_jul.month} 的 {money(month11_jul.delta_total_cost_yuan)} 元主导，不称任一方案全面稳健。

### 4.3 点预测误差

| 预测器 | 目标 | MAE/kW | RMSE/kW | 偏差/kW |
|---|---|---:|---:|---:|
{metric_rows}

四种净需求组合（实际−预测，kW）：

| 组 | 负载来源 | 光伏来源 | MAE/kW | RMSE/kW | 偏差/kW |
|---|---|---|---:|---:|---:|
{net_rows}

### 4.4 保护校准

覆盖率为实际净需求不超过保护需求的时段比例（n≤ñ），**不是**"无应急概率"。

| 组 | 平均保护量/kWh | 覆盖率 |
|---|---:|---:|
{coverage_rows}

### 4.5 稳定性与能量账

| 比较 | 改善月 | 改善天 | 变差天 | 最差日 | 最差日/元 | 0—10时/元 | 19—21时/元 | 21—24时/元 |
|---|---:|---:|---:|---|---:|---:|---:|---:|
{stability_rows}

组间能量账 $\\Delta Q=-\\Delta Q_{{\\mathrm{{em}}}}+\\Delta W+\\Delta\\mathrm{{Loss}}+\\Delta E_{{end}}$：

| 比较 | Δ计划/kWh | Δ应急/kWh | Δ未使用/kWh | Δ损耗/kWh | Δ期末/kWh | 恒等式残差/kWh |
|---|---:|---:|---:|---:|---:|---:|
{energy_rows}

库存辅助诊断 $C^*=C-\\nu(E_{{end}}-E_{{start}})$，$\\nu=\\operatorname{{median}}(p)/0.9$ 元/内部 kWh
（不替代主指标）：

| 组 | 总费/元 | 库存调整/元 | 调整后/元 |
|---|---:|---:|---:|
{inventory_rows}

### 4.6 逐月差额

| 比较 | 月 | Δ计划费/元 | Δ应急费/元 | Δ总费/元 | Δ应急/kWh |
|---|---|---:|---:|---:|---:|
{monthly_rows}

### 4.7 图表

{integrity_rows}

本环境无法进行图像目视核查，上述仅为程序化完整性检查；图表内容的人工/视觉核验**尚未完成**。

## 5. 验证与适用边界

| 检查 | 结果 |
|---|---|
| 人工边界样例（合成数据） | {len(checks['boundary']['rows'])} 项，全部通过 |
| 负载模板与冻结 15 列的同星期均值一致 | 最大差 {checks['template_parity']['max_difference_kW']:.3e} kW（{checks['template_parity']['days']} 天） |
| 候选矩阵仅追加列 | 负载 {checks['extension']['load_max_difference_kW']:.3e} kW、光伏 {checks['extension']['pv_max_difference_kW']:.3e} kW |
| 门控重建与冻结掩码一致（全年） | 不一致 {gate['mask_mismatch_slots']} 段 |
| 支持窗口逐日重建 | 第 1 日起不一致 {checks['support_windows']['mismatch_from_day1']} 段；第 0 日（无决策、不参与正式误差）记为文档缺省 {checks['support_windows']['day0_rebuilt']} |
| 四组共享预测身份与统一门控 | 共享档案最大差 {max(r['worst'] for r in checks['population']['rows']):.3e} kW；两条光伏档案门控外最大 {max(checks['population']['gate']['v0_max_off_gate_kW'], checks['population']['gate']['v1_max_off_gate_kW']):.3e} kW |
| 控制 15/8 列模型独立重训复现 | 负载 {checks['control_refit']['load_max_difference_kW']:.3e} kW、光伏 {checks['control_refit']['pv_max_difference_kW']:.3e} kW |
| 训练行/标签/首训日一致性 | 不一致 {checks['load_parity']['mismatch_count'] + checks['pv_parity']['mismatch_count']} 项 |
| 分位规则 m=7..28 | 通过 |
| 控制组 F00 复现（G4 与 L_q80） | 日级最大差 {max(checks['reference_reproduction']['G4_lightgbm_fixed']['max_day_level_difference'], checks['reference_reproduction']['L_q80']['max_day_level_difference']):.3e} kWh |
| F01 与 23 号 P1_q80 逐段一致 | 预测最大差 {checks['cross_module']['pv_forecast_max_difference_kW']:.3e} kW，费用差 {checks['cross_module']['dispatch']['difference_vs_registered_yuan']:.3e} 元 |
| 未来扰动（两条新增链） | {len(checks['perturbation']['cases'])} 例，全部通过 |
| 半日反馈前缀（四组，含进入第 72 段的状态） | {len(checks['feedback_prefix']['cases'])} 例，全部通过 |
| 抽样重复拟合与模型文本重载（同一实现，非独立重实现） | {checks['sampled_retraining']['checks']} 项（8 日 × 4 目标-版本），全部通过 |
| 抽样 MILP 重解 | {checks['sampled_milp']['solves']} 次，全部通过 |
| 物理最大误差（平衡/递归/容量/互斥/名义终点） | {checks['physical_max']:.3e}（阈值 {ENERGY_TOL_KWH:g}） |
| 受保护旧资产 | 零变更零缺失，共 {checks['protected_count']} 个文件 |

限制：

1. 2025 年数据此前已参与方法诊断与设计，本轮是**滚动因果回测，不是独立盲测或跨年验证**；
   单年费用差不证明跨年泛化，也不把 48096 段当独立样本做显著性检验。
2. 原 15/8 列对照发布档案本轮冻结复用，其训练因果性属上游 20 号独立审计范围；
   本轮只做规定抽样复核，不称全年重验了旧训练。抽样重训调用的是与全年构建相同的实现，
   它是重复拟合与模型文本重载的一致性检查，**不是**独立重实现。
3. 56 日窗、至少 14 日、6 段（1 小时）局部窗口、1 kW 阈值、3 段余量、W28、q80、6000 kWh
   均为运行前固定的工程设计，本实验未搜索这些参数；未尝试 30/90/120 分钟窗口。
4. 光伏门控规则不变，四组门控掩码与被归零位置完全一致；既有门控收益不重新算作本轮新增特征收益。
   门控与支持窗口统计为全年 365 天口径；第 0 日无决策、按文档缺省 s=0/e=143/v=0 记录且不参与正式误差。
5. 负载模板的"同星期"不引入节假日分类；1 小时局部斜率与相对高低是待检验的特征表示，
   不声称引入独立新信息，部分列在多数训练窗内恒定（见 `candidate_load_fit_log.csv`）。
6. 点预测精度变化不必然转化为费用变化；q 分位保护可能抵消稳定点预测偏差，本轮同时报告
   点预测、保护量、应急与总费，不做单一综合分数。
7. 本实验不替代 20 号上游预测审计、21 号分位扫描审计与 23 号光伏特征实验的独立审计；
   23 号的接收核查意见（起止时间、窗口第 0 日、前缀状态、重复拟合口径等）已在本实验逐条避免。

## 6. 文件与复现

- 登记与运行清单：`results/q2_load_pv_shape/registration.json`、`run_manifest.json`。
- 预测档案：`forecast_archive_L0.csv`、`forecast_archive_L1.csv`、`forecast_archive_V0.csv`、
  `forecast_archive_V1.csv`；`load_templates.csv`、`load_shape_features.npz`、`support_windows.csv`；
  `models/original15/`、`models/shape19/`、`models/original8/`、`models/support12/`。
- 逐组结果：`results/q2_load_pv_shape/<组名>/dispatch.csv`、`nominal_dispatch.csv`、
  `daily_summary.csv`、`monthly_summary.csv`、`emergency_events.csv`、`solver_log.csv`、
  `validation.json`。
- 汇总与自检：`summary.csv`、`contrasts.csv`、`monthly_contrasts.csv`、`stability.csv`、
  `energy_contrasts.csv`、`inventory_adjusted.csv`、`coverage.csv`、`predictor_metrics.csv`、
  `net_demand_metrics.csv`、`load_error_subsets.csv`、`pv_error_subsets.csv`、
  `pv_error_by_window.csv`、`pv_error_monthly.csv`、`pv_error_hourly.csv`、`checks.json`。
- 图表：`figures/q2_load_pv_shape/`（4 张 PNG）。

复现：

```
E:/Anaconda/envs/math_modeling/python.exe code/25_q2_load_pv_shape_experiment.py --mode register
E:/Anaconda/envs/math_modeling/python.exe code/25_q2_load_pv_shape_experiment.py --mode repro
E:/Anaconda/envs/math_modeling/python.exe code/25_q2_load_pv_shape_experiment.py --mode full
```

独立审计入口（建议）：以 `results/q2_load_pv_shape/` 为只读输入，独立重建各历史日自己的 H、μ/z/前后
1 小时斜率与光伏 s/e/v/D/M/τ，核对是否出现"向后一小时"误用当天真值、跨午夜循环、拿当前模板回填历史、
负载窗口外归零、未来日出日落、联合组重复训练或暗调参、旧缺口档案、相加单改组分位量；全量核对四组共享
预测身份、独立净需求误差与 SOC、控制复现、名义可行性与费用/能量差额基准。
"""
    REPORT_MD.write_text(report, encoding='utf-8')


if __name__ == '__main__':
    main()
