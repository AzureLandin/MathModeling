from pathlib import Path
import hashlib, json, math
import numpy as np
import pandas as pd

ROOT = Path.cwd()
BASE = ROOT / 'results/q3_intraday_load_cost'
S2DIR = BASE / 'S2_load_q75'
OUT = ROOT / 'results/q3_intraday_load_cost_audit'
OUT.mkdir(parents=True, exist_ok=True)

# Read-only audit inputs.
dispatch = pd.read_csv(S2DIR / 'dispatch.csv')
summary = pd.read_csv(BASE / 'summary.csv')
decisions = pd.read_csv(S2DIR / 'revision_decisions.csv')
plans = pd.read_csv(S2DIR / 'plan_versions.csv')
with np.load(ROOT / 'results/q3_intraday_load_forward/forecast_archive.npz') as forward_archive:
    f2_power = forward_archive['forecast'][2].astype(float)
    load_truth = forward_archive['truth_load'].astype(float)
with np.load(ROOT / 'results/q3_bias_correction_diagnostic/bias_forecast_archive.npz') as bias_archive:
    pv_truth = bias_archive['truth_pv'].astype(float)
with np.load(ROOT / 'results/q3_rolling_baseline/prediction_protection.npz') as rb_archive:
    pv_forecasts = rb_archive['pv_forecasts'].astype(float)
with np.load(BASE / 'protection.npz') as protection_archive:
    saved_protected = protection_archive['protected_f2'].astype(float)
    saved_counts = protection_archive['counts_f2'].astype(np.int64)
    saved_availability = protection_archive['availability'].astype(bool)

DT = 1 / 6
DAYS, VERSIONS, TARGETS = f2_power.shape
UPDATE_HOURS = [0, 6, 12, 18]
FIRST_TARGET = [0, 36, 72, 108]
WINDOW = 28
MIN_SAMPLES = 7
TOL = 1e-8

# Independent F2 q75 reconstruction from the saved forecast and truth archives.
net_forecast = (f2_power - pv_forecasts) * DT
truth_net = (load_truth - pv_truth) * DT
errors = truth_net[:, None, :] - net_forecast
counts = np.zeros((DAYS, VERSIONS, TARGETS), dtype=np.int64)
rho = np.zeros((DAYS, VERSIONS, TARGETS), dtype=float)
for day_idx in range(DAYS):
    for version_idx, pub_hour in enumerate(UPDATE_HOURS):
        first_target = FIRST_TARGET[version_idx]
        for target_idx in range(first_target, TARGETS):
            history = []
            for hist_day in range(max(0, day_idx - WINDOW), day_idx):
                available = (target_idx + 1 <= 144 * (day_idx - hist_day) + 6 * pub_hour)
                if available and np.isfinite(errors[hist_day, version_idx, target_idx]):
                    history.append(float(errors[hist_day, version_idx, target_idx]))
            m = len(history)
            counts[day_idx, version_idx, target_idx] = m
            if m >= MIN_SAMPLES:
                position = math.ceil(0.75 * m) - 1
                rho[day_idx, version_idx, target_idx] = sorted(history)[position]
protected = net_forecast + rho

# Protection identity and sample-boundary checks.
prot_max = float(np.nanmax(np.abs(protected - saved_protected)))
count_mismatch = int(np.sum(counts != saved_counts))
expected_availability = np.zeros((WINDOW + 1, VERSIONS, TARGETS), dtype=bool)
for lag_idx, lag_days in enumerate(range(1, WINDOW + 1), start=1):
    for version_idx, pub_hour in enumerate(UPDATE_HOURS):
        expected_availability[lag_idx, version_idx, :] = [
            target_idx + 1 <= 144 * lag_days + 6 * pub_hour
            for target_idx in range(TARGETS)
        ]
availability_mismatch = int(np.sum(saved_availability != expected_availability))
# The saved availability archive is [lag, version, target], while the expression above is the same shape.

# Ledger checks on the 48,096 natural intervals; the trailing template row is excluded from cash sums.
ledger = dispatch.copy()
ledger['interval_start'] = pd.to_datetime(ledger['interval_start'])
ledger = ledger.sort_values('interval_start').reset_index(drop=True)
natural = ledger[ledger['interval_start'] < pd.Timestamp('2026-01-01 00:00:00')]
expected_rows = 334 * 144
balance = (ledger['q_eff_kWh'] + ledger['emergency_kWh'] - ledger['charge_kWh']
           + ledger['discharge_kWh'] - ledger['unused_kWh'] - ledger['net_kWh']).abs()
recursion = (ledger['state_end_kWh'] - (ledger['state_start_kWh']
             + 0.9 * ledger['charge_kWh'] - ledger['discharge_kWh'] / 0.9)).abs()
continuity = (ledger['state_start_kWh'].iloc[1:].to_numpy()
              - ledger['state_end_kWh'].iloc[:-1].to_numpy())

# Rebuild each cash component directly from settlement fields.
ordinary = ledger['price_yuan_kWh'] * ledger['q_eff_kWh']
adjustment = 0.5 * ledger['price_yuan_kWh'] * (ledger['q_eff_kWh'] - ledger['q0_kWh']).abs()
emergency = 5.0 * ledger['price_yuan_kWh'] * ledger['emergency_kWh']
total = ordinary + adjustment + emergency
saved_total = ledger['total_cost_yuan']
row_cash_error = (total - saved_total).abs()
calc_totals = {
    'ordinary': float(ordinary.loc[natural.index].sum()),
    'adjustment': float(adjustment.loc[natural.index].sum()),
    'emergency': float(emergency.loc[natural.index].sum()),
    'total': float(total.loc[natural.index].sum()),
}
summary_row = summary.set_index('strategy_id').loc['S2_load_q75']
summary_errors = {
    'ordinary': abs(calc_totals['ordinary'] - float(summary_row['ordinary_cost_yuan'])),
    'adjustment': abs(calc_totals['adjustment'] - float(summary_row['adjustment_cost_yuan'])),
    'emergency': abs(calc_totals['emergency'] - float(summary_row['emergency_cost_yuan'])),
    'total': abs(calc_totals['total'] - float(summary_row['natural_total_yuan'])),
}

# Revision and state handoff checks at the 0/6/12/18 update instants.
state_by_start = ledger.set_index('interval_start')['state_start_kWh']
state_update_errors = []
for _, rec in decisions.iterrows():
    ts = pd.Timestamp(rec['published_at'])
    if ts in state_by_start:
        state_update_errors.append(abs(float(rec['state_at_update_kWh']) - float(state_by_start.loc[ts])))
    else:
        state_update_errors.append(np.nan)
state_update_errors = np.asarray(state_update_errors, dtype=float)
state_update_max = float(np.nanmax(state_update_errors))
accepted = int(decisions['accepted'].astype(bool).sum())
rejected = int((~decisions['accepted'].astype(bool)).sum())
# Every accepted revision must have a strictly negative predicted cost delta under the saved rule.
accept_rule_violations = int(np.sum(decisions.loc[decisions['accepted'].astype(bool), 'delta_predicted_yuan'] >= -1e-4))

# Basic plan-version structural checks: all revision rows have a previous plan and candidate.
revision_rows = plans[plans['decision'].eq('revision')]
plan_structural = {
    'revision_rows': int(len(revision_rows)),
    'revision_rows_with_previous': int(revision_rows['previous_kWh'].notna().sum()),
    'revision_rows_with_candidate': int(revision_rows['candidate_kWh'].notna().sum()),
    'all_revision_plan_rows_accepted': bool(revision_rows['accepted'].astype(bool).all()),
    'decision_count_matches_update_records': bool(accepted == len(decisions)),
}

result = {
    'status': 'complete',
    'scope': 'independent read-only audit of saved S2 ledger; no model or dispatch rerun',
    'configuration': {'alpha': 0.75, 'window_days': WINDOW, 'min_samples': MIN_SAMPLES,
                     'update_hours': UPDATE_HOURS, 'first_target': FIRST_TARGET, 'dt_hours': DT},
    'q75_reconstruction': {
        'shape': [DAYS, VERSIONS, TARGETS],
        'max_abs_difference_to_saved_protected_kWh': prot_max,
        'count_mismatch_cells': count_mismatch,
        'availability_mismatch_cells': availability_mismatch,
        'minimum_count_after_warmup': int(counts[WINDOW:, :, :].min()),
    },
    'ledger': {
        'saved_rows': int(len(ledger)), 'natural_rows': int(len(natural)), 'expected_natural_rows': expected_rows,
        'max_bus_balance_error_kWh': float(balance.max()),
        'max_soc_recursion_error_kWh': float(recursion.max()),
        'max_state_continuity_error_kWh': float(np.max(np.abs(continuity))),
        'max_row_cash_error_yuan': float(row_cash_error.max()),
        'cash_totals_recomputed_yuan': calc_totals,
        'summary_component_errors_yuan': summary_errors,
    },
    'revision_and_handoff': {
        'decision_rows': int(len(decisions)), 'accepted': accepted, 'rejected': rejected,
        'accepted_rule_violations': accept_rule_violations,
        'max_state_at_update_error_kWh': state_update_max,
        'plan_structure': plan_structural,
    },
    'source_hashes': {
        'dispatch': hashlib.sha256((S2DIR / 'dispatch.csv').read_bytes()).hexdigest(),
        'protection': hashlib.sha256((BASE / 'protection.npz').read_bytes()).hexdigest(),
        'decisions': hashlib.sha256((S2DIR / 'revision_decisions.csv').read_bytes()).hexdigest(),
    },
}

assert len(ledger) == 48097 and len(natural) == expected_rows
assert prot_max < TOL and count_mismatch == 0 and availability_mismatch == 0
assert balance.max() < TOL and recursion.max() < TOL and np.max(np.abs(continuity)) < TOL
assert row_cash_error.max() < 1e-6 and max(summary_errors.values()) < 1e-6
assert len(decisions) == 1002 and accepted == 1002 and rejected == 0 and accept_rule_violations == 0
assert state_update_max < TOL
assert plan_structural['revision_rows'] > 0 and plan_structural['revision_rows_with_previous'] == plan_structural['revision_rows'] and plan_structural['revision_rows_with_candidate'] == plan_structural['revision_rows']
(OUT / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
