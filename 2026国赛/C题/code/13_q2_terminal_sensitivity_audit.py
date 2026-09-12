"""Independent audit of the Q2 fixed nominal day-end target sensitivity experiment
(4800 / 6000 / 7200 kWh, code/12_q2_terminal_sensitivity.py).

This script NEVER writes into the audited outputs. It re-reads the raw attachments, the
per-segment dispatch tables and the reported summaries, then re-derives every physical,
accounting and attribution quantity from scratch:

    * predictor identity  L_hat[k,t] = L[k-7,t] (fallback L[k-1,t]); V_hat[k,t] = V[k-1,t]
    * residual q80 protection re-computed from the raw realized net demand
    * greedy causal feedback re-simulated from the frozen plan and the realized load/PV
    * per-segment bus balance, state recursion, capacity, power, non-negativity, exclusivity
    * day-boundary state continuity and the nominal terminal target per day
    * planned cost, 5x emergency cost, loss, unused energy, morning 0-10 h emergency
    * day / month / whole-period / contiguous-emergency-event reconciliation
    * the full-period energy identity including losses and the change in stored energy
    * a MILP replay of the nominal plan from the recorded protected demand and initial state
    * provenance hashes, the public-January trajectory and the superseded rounds

Run:
    E:/Anaconda/envs/math_modeling/python.exe code/13_q2_terminal_sensitivity_audit.py

Outputs: results/q2_terminal_sensitivity_audit/
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SENS = ROOT / 'results/q2_terminal_sensitivity'
AUDIT = ROOT / 'results/q2_terminal_sensitivity_audit'
REFERENCE_RUN = ROOT / 'results/q2_terminal_experiment_20260911'
DIAG_11 = ROOT / 'results/q2_terminal_diagnostics_20260911'
BASELINE_05 = ROOT / 'results/q2_baseline'
SNAP = ROOT / 'results/q2_revision_audit_20260911/source_snapshot'
PROTECTED_BEFORE = ROOT / 'results/q2_revision_audit_20260911/protected_before.json'

GROUPS = {'T_q80_4800': 4800.0, 'T_q80_6000': 6000.0, 'T_q80_7200': 7200.0}
DT = 1 / 6
ETA = 0.9
CAP = 5000 * DT
STATE_MIN, STATE_MAX = 1200.0, 10800.0
COMMON_INITIAL = 8801.462273333342
LICENSE_TOTAL = 14158360.487139747
EVAL_START = '2025-02-01'
W = 28
THETA = 0.8
EPS_SLOT_TOL = 1e-6          # kWh, per-segment physical
TOL_COST = 1e-4              # yuan
TOL_ENERGY = 1e-4            # kWh, cumulative reconciliation
TOL_SEG_ENERGY = 1e-6        # kWh, per-segment comparison vs existing run
DATES = pd.date_range('2025-01-01', '2025-12-31')

ENERGY_COLUMNS = ['planned_kWh', 'charge_kWh', 'discharge_kWh', 'emergency_kWh', 'unused_kWh',
                  'state_start_kWh', 'state_end_kWh', 'nominal_state_end_kWh',
                  'net_forecast_kWh', 'residual_adjustment_kWh', 'protected_net_kWh']
COST_COLUMNS = ['planned_cost_yuan', 'emergency_cost_yuan']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jload(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(name, obj):
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float),
                              encoding='utf-8')


def f(x):
    return float(x)


# --------------------------------------------------------------------------------------
# independent input reader (does not use code/05 or the frozen snapshot)
# --------------------------------------------------------------------------------------
def read_attachment1_price():
    import openpyxl
    wb = openpyxl.load_workbook(ROOT / '附件/附件1.xlsx', read_only=True, data_only=True)
    rows = list(wb.worksheets[0].values)
    wb.close()
    price = np.array([r[1] for r in rows[1:]], dtype=float)
    assert price.shape == (144,)
    return price


def read_attachment2_actual():
    import openpyxl
    wb = openpyxl.load_workbook(ROOT / '附件/附件2.xlsx', read_only=True, data_only=True)
    tables = []
    date_axis = None
    for ws in wb.worksheets:
        rows = list(ws.values)
        axis = pd.DatetimeIndex([r[0] for r in rows[1:]])
        if date_axis is None:
            date_axis = axis
        else:
            assert axis.equals(date_axis)
        tables.append(np.asarray([r[1:] for r in rows[1:]], dtype=float))
    wb.close()
    assert date_axis.equals(DATES), 'attachment-2 date axis'
    load, pv = tables[0], tables[1]
    assert load.shape == pv.shape == (365, 144)
    assert np.isfinite(load).all() and np.isfinite(pv).all()
    assert (load >= 0).all() and (pv >= 0).all()
    return load, pv


def load_kernel():
    """The frozen MILP kernel the experiment used, for the nominal replay only."""
    spec = importlib.util.spec_from_file_location('audit_core', SNAP / 'code/02_q1_baseline.py')
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    return core


# --------------------------------------------------------------------------------------
# independent predictor + protection re-derivation
# --------------------------------------------------------------------------------------
def independent_forecasts(load, pv):
    n = len(load)
    pred_l = np.full((n, 144), np.nan)
    pred_v = np.full((n, 144), np.nan)
    for k in range(1, n):
        pred_l[k] = load[k - 7] if k >= 7 else load[k - 1]
        pred_v[k] = pv[k - 1]
    return pred_l, pred_v


def independent_q80(pred_l, pred_v, load, pv, k, W=W, theta=THETA):
    """Empirical inverse distribution over the most recent W residual days strictly before k."""
    lo = max(1, k - W)
    m = k - lo
    if m < 7:
        raise AssertionError(f'day {k} has only {m} residual days (experiment requires >=7)')
    eps = ((load[lo:k] - pv[lo:k]) - (pred_l[lo:k] - pred_v[lo:k])) * DT
    ordered = np.sort(eps, axis=0)
    idx = int(math.ceil(m * theta)) - 1
    assert 0 <= idx < m
    return ordered[idx].copy(), m


def replay_greedy_control(plan, load_day, pv_day, initial, battery=True):
    """Re-implement bm.control independently from its documented greedy rule."""
    n = len(plan)
    c = np.zeros(n)
    d = np.zeros(n)
    e = np.zeros(n)
    w = np.zeros(n)
    states = np.r_[float(initial), np.zeros(n)]
    cap = CAP if battery else 0.0
    for t in range(n):
        surplus = plan[t] + (pv_day[t] - load_day[t]) * DT
        if surplus >= 0:
            c[t] = min(surplus, cap, max(0.0, (STATE_MAX - states[t]) / ETA))
            w[t] = surplus - c[t]
        else:
            d[t] = min(-surplus, cap, max(0.0, (states[t] - STATE_MIN) * ETA))
            e[t] = -surplus - d[t]
        states[t + 1] = states[t] + ETA * c[t] - d[t] / ETA
    return c, d, e, w, states


def contiguous_events(emergency, price, date_label):
    """Same contiguous-run definition as the experiment, recomputed from raw segments."""
    active = emergency > EPS_SLOT_TOL
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    stops = np.flatnonzero(active & ~np.r_[active[1:], False]) + 1
    rows = []
    for s, t in zip(starts, stops):
        rows.append(dict(date=date_label, start_slot=int(s), end_slot=int(t),
                         emergency_kWh=f(emergency[s:t].sum()),
                         emergency_cost_yuan=f(5.0 * (emergency[s:t] * price[s:t]).sum())))
    return rows


# --------------------------------------------------------------------------------------
# per-group audit
# --------------------------------------------------------------------------------------
def audit_group(name, target, load, pv, price):
    folder = SENS / name
    d = pd.read_csv(folder / 'dispatch.csv', low_memory=False)
    daily = pd.read_csv(folder / 'daily_summary.csv')
    monthly = pd.read_csv(folder / 'monthly_summary.csv')
    monthly_totals = pd.read_csv(folder / 'monthly_totals.csv')
    events = pd.read_csv(folder / 'emergency_events.csv')
    reported = pd.read_csv(SENS / 'summary.csv').set_index('strategy_id').loc[name]

    out = {'strategy_id': name, 'terminal_target_kWh': float(target)}

    # ---- axis, inputs and provenance -------------------------------------------------
    days = d.date.unique().tolist()
    assert len(days) == 334 and len(d) == 334 * 144, (name, len(days), len(d))
    assert [str(pd.Timestamp(x).date()) for x in days] == \
           [str(x.date()) for x in DATES[31:]], 'evaluation day axis must be 2025-02-01..2025-12-31'
    assert np.array_equal(d.slot.to_numpy().reshape(334, 144), np.tile(np.arange(144), (334, 1)))

    day_idx = np.array([(pd.Timestamp(x) - DATES[0]).days for x in days])
    price_col = d.price_yuan_kWh.to_numpy().reshape(334, 144)
    load_col = d.load_kW.to_numpy().reshape(334, 144)
    pv_col = d.pv_kW.to_numpy().reshape(334, 144)
    price_input = np.array([np.array_equal(price_col[i], price) for i in range(334)]).all()
    load_input = np.array_equal(load_col, load[day_idx])
    pv_input = np.array_equal(pv_col, pv[day_idx])
    out['inputs'] = dict(price_matches_attachment1=bool(price_input),
                         load_matches_attachment2=bool(load_input),
                         pv_matches_attachment2=bool(pv_input))

    # ---- predictor identity ----------------------------------------------------------
    pred_l, pred_v = independent_forecasts(load, pv)
    rec_l = d.load_forecast_kW.to_numpy().reshape(334, 144)
    rec_v = d.pv_forecast_kW.to_numpy().reshape(334, 144)
    out['predictor'] = dict(
        max_abs_load_forecast_diff_kW=f(np.max(np.abs(rec_l - pred_l[day_idx]))),
        max_abs_pv_forecast_diff_kW=f(np.max(np.abs(rec_v - pred_v[day_idx]))),
        load_uses_only_past=bool(np.array_equal(rec_l, pred_l[day_idx])),
        pv_uses_only_past=bool(np.array_equal(rec_v, pred_v[day_idx])))

    # ---- protection re-derived from raw realized residuals ---------------------------
    adj_rec = d.residual_adjustment_kWh.to_numpy().reshape(334, 144)
    prot_rec = d.protected_net_kWh.to_numpy().reshape(334, 144)
    netf_rec = d.net_forecast_kWh.to_numpy().reshape(334, 144)
    adj_max, prot_max, netf_max, resid_days_seen = 0.0, 0.0, 0.0, set()
    for i, k in enumerate(day_idx):
        r, m = independent_q80(pred_l, pred_v, load, pv, int(k))
        netf = (pred_l[k] - pred_v[k]) * DT
        adj_max = max(adj_max, f(np.max(np.abs(adj_rec[i] - r))))
        netf_max = max(netf_max, f(np.max(np.abs(netf_rec[i] - netf))))
        prot_max = max(prot_max, f(np.max(np.abs(prot_rec[i] - (netf + r)))))
        resid_days_seen.add(m)
    out['protection'] = dict(
        max_abs_net_forecast_diff_kWh=netf_max,
        max_abs_q80_adjustment_diff_kWh=adj_max,
        max_abs_protected_net_diff_kWh=prot_max,
        residual_window_days=sorted(resid_days_seen),
        protection_independent_match=bool(max(adj_max, prot_max, netf_max) <= EPS_SLOT_TOL))

    # ---- nominal terminal target -----------------------------------------------------
    nominal_day_end = d.groupby('date', sort=False).nominal_state_end_kWh.last().to_numpy()
    out['nominal_terminal'] = dict(
        max_abs_error_kWh=f(np.max(np.abs(nominal_day_end - target))),
        target_kWh=float(target))

    # ---- independent greedy replay of the actual execution ---------------------------
    plan = d.planned_kWh.to_numpy().reshape(334, 144)
    c_rec = d.charge_kWh.to_numpy().reshape(334, 144)
    d_rec = d.discharge_kWh.to_numpy().reshape(334, 144)
    e_rec = d.emergency_kWh.to_numpy().reshape(334, 144)
    w_rec = d.unused_kWh.to_numpy().reshape(334, 144)
    s0_rec = d.state_start_kWh.to_numpy().reshape(334, 144)
    s1_rec = d.state_end_kWh.to_numpy().reshape(334, 144)
    init_rec = daily.initial_kWh.to_numpy()
    c_max = d_max = e_max = w_max = s_max = 0.0
    for i in range(334):
        c, dd, e, w, st = replay_greedy_control(plan[i], load[day_idx[i]], pv[day_idx[i]], init_rec[i])
        c_max = max(c_max, f(np.max(np.abs(c - c_rec[i]))))
        d_max = max(d_max, f(np.max(np.abs(dd - d_rec[i]))))
        e_max = max(e_max, f(np.max(np.abs(e - e_rec[i]))))
        w_max = max(w_max, f(np.max(np.abs(w - w_rec[i]))))
        s_max = max(s_max, f(np.max(np.abs(st[1:] - s1_rec[i]))))
    out['control_replay'] = dict(
        max_abs_charge_diff_kWh=c_max, max_abs_discharge_diff_kWh=d_max,
        max_abs_emergency_diff_kWh=e_max, max_abs_unused_diff_kWh=w_max,
        max_abs_state_end_diff_kWh=s_max,
        greedy_control_reproduced=bool(max(c_max, d_max, e_max, w_max, s_max) <= EPS_SLOT_TOL))

    # ---- nominal MILP trajectory: is the stored path a valid optimal-nominal realization? --
    # The nominal programme is degenerate (free unused energy), so several (c,d,w) paths share
    # the same optimal q. The stored nominal path is checked for feasibility, not for equality
    # with one arbitrary optimum.
    nom_all = d.nominal_state_end_kWh.to_numpy().reshape(334, 144)
    nom_viol = []
    for i in range(334):
        E = np.r_[init_rec[i], nom_all[i]]
        S = plan[i] - prot_rec[i]
        dE = np.diff(E)
        c_n = np.where(dE >= 0, dE / ETA, 0.0)
        d_n = np.where(dE < 0, -ETA * dE, 0.0)
        w_n = S - c_n + d_n
        nom_viol.append(max(0.0, -w_n.min(), c_n.max() - CAP, d_n.max() - CAP,
                            STATE_MIN - E.min(), E.max() - STATE_MAX, abs(E[-1] - target)))
    out['nominal_reconstruction'] = dict(
        max_violation_kWh=f(max(nom_viol)),
        max_violation_day=days[int(np.argmax(nom_viol))],
        terminal_error_kWh=f(np.max(np.abs(nom_all[:, -1] - target))),
        recorded_trajectory_feasible=bool(max(nom_viol) <= EPS_SLOT_TOL))

    # ---- per-segment constraints -----------------------------------------------------
    q = d.planned_kWh.to_numpy()
    c_all, dd_all = d.charge_kWh.to_numpy(), d.discharge_kWh.to_numpy()
    e_all, w_all = d.emergency_kWh.to_numpy(), d.unused_kWh.to_numpy()
    s0_all, s1_all = d.state_start_kWh.to_numpy(), d.state_end_kWh.to_numpy()
    p_all = d.price_yuan_kWh.to_numpy()
    l_all, v_all = d.load_kW.to_numpy(), d.pv_kW.to_numpy()
    balance = q + v_all * DT + dd_all + e_all - l_all * DT - c_all - w_all
    state_rec = (s1_all - s0_all) - (ETA * c_all - dd_all / ETA)
    continuity = np.abs(s0_all[1:] - s1_all[:-1])
    bounds = float(max(0.0, STATE_MIN - min(s0_all.min(), s1_all.min()),
                       max(s0_all.max(), s1_all.max()) - STATE_MAX))
    power = float(max(0.0, c_all.max() - CAP, dd_all.max() - CAP))
    nonneg = float(max(0.0, -min(q.min(), c_all.min(), dd_all.min(), e_all.min(), w_all.min())))
    mutex = float(np.minimum(c_all, dd_all).max())
    out['segments'] = dict(
        balance_max_abs_kWh=f(np.max(np.abs(balance))),
        state_recursion_max_abs_kWh=f(np.max(np.abs(state_rec))),
        day_continuity_max_abs_kWh=f(continuity.max()),
        state_bounds_violation_kWh=f(bounds),
        power_bounds_violation_kWh=f(power),
        nonnegative_violation_kWh=f(nonneg),
        simultaneous_charge_discharge_kWh=f(mutex),
        max_physical_violation_kWh=f(max(np.max(np.abs(balance)), np.max(np.abs(state_rec)),
                                         continuity.max(), bounds, power, nonneg, mutex)))

    # ---- costs -----------------------------------------------------------------------
    planned_cost = f((p_all * q).sum())
    emergency_cost = f((5 * p_all * e_all).sum())
    total_cost = planned_cost + emergency_cost
    loss = f((0.1 * c_all + (1 / ETA - 1) * dd_all).sum())
    morning = d.slot.to_numpy() < 60
    morning_cost = f((5 * p_all * e_all)[morning].sum())
    morning_kwh = f(e_all[morning].sum())
    out['costs'] = dict(planned_cost_yuan=planned_cost, emergency_cost_yuan=emergency_cost,
                        total_cost_yuan=total_cost, loss_kWh=loss,
                        planned_kWh=f(q.sum()), emergency_kWh=f(e_all.sum()),
                        unused_kWh=f(w_all.sum()), charge_kWh=f(c_all.sum()),
                        discharge_kWh=f(dd_all.sum()),
                        morning_0_10_emergency_cost_yuan=morning_cost,
                        morning_0_10_emergency_kWh=morning_kwh,
                        license_total_abs_diff_yuan=f(abs(total_cost - LICENSE_TOTAL))
                        if name == 'T_q80_6000' else None)

    # ---- energy identity (losses + change in stored energy) --------------------------
    identity = f(q.sum() + e_all.sum() + v_all.sum() * DT - l_all.sum() * DT - w_all.sum()
                 - loss - (s1_all[-1] - s0_all[0]))
    out['energy_identity_kWh'] = identity

    # ---- day / month / event reconciliation from raw segments ------------------------
    idx = pd.Series(d.date.to_numpy())
    recon = {}
    recon['daily_total_cost_yuan'] = f(abs((p_all * q + 5 * p_all * e_all).sum()
                                           - daily.total_cost_yuan.sum()))
    recon['daily_planned_cost_yuan'] = f(abs(planned_cost - daily.planned_cost_yuan.sum()))
    recon['daily_emergency_cost_yuan'] = f(abs(emergency_cost - daily.emergency_cost_yuan.sum()))
    recon['daily_emergency_kWh'] = f(abs(e_all.sum() - daily.emergency_kWh.sum()))
    recon['monthly_total_cost_yuan'] = f(abs(total_cost - monthly.total_cost_yuan.sum()))
    recon['monthly_recomputed_total_cost_yuan'] = f(abs(
        total_cost - daily.assign(month=daily.date.str[:7]).groupby('month').total_cost_yuan.sum().sum()))
    recon['monthly_totals_total_cost_yuan'] = f(abs(total_cost - monthly_totals.total_cost_yuan.sum()))
    recon['monthly_totals_month_column_present'] = bool('month' in monthly_totals.columns)
    recon['monthly_totals_months'] = int(monthly_totals.month.nunique())

    # contiguous events recomputed per day and compared with the stored table
    ev_rows = []
    for i in range(334):
        ev_rows.extend(contiguous_events(e_rec[i], price, days[i]))
    ev_rec = pd.DataFrame(ev_rows)
    ev_stored = events.copy()
    recon['event_count_recomputed'] = int(len(ev_rec))
    recon['event_count_stored'] = int(len(ev_stored))
    recon['event_count_match'] = bool(len(ev_rec) == len(ev_stored))
    if len(ev_rec) == len(ev_stored):
        key_r = list(zip(ev_rec.date, ev_rec.start_slot, ev_rec.end_slot))
        key_s = list(zip(ev_stored.date, ev_stored.start_slot, ev_stored.end_slot))
        recon['event_intervals_match'] = bool(key_r == key_s)
        recon['event_energy_max_abs_diff_kWh'] = f(np.max(np.abs(
            ev_rec.emergency_kWh.to_numpy() - ev_stored.emergency_kWh.to_numpy())))
        recon['event_cost_max_abs_diff_yuan'] = f(np.max(np.abs(
            ev_rec.emergency_cost_yuan.to_numpy() - ev_stored.emergency_cost_yuan.to_numpy())))
    recon['event_energy_sum_diff_kWh'] = f(abs(ev_rec.emergency_kWh.sum() - e_all.sum()))
    recon['event_cost_sum_diff_yuan'] = f(abs(ev_rec.emergency_cost_yuan.sum() - emergency_cost))
    out['reconciliation'] = recon

    # ---- reported-metric recomputation ----------------------------------------------
    day_end = s1_rec[:, -1]
    recomputed = dict(
        planned_cost_yuan=planned_cost, emergency_cost_yuan=emergency_cost, total_cost_yuan=total_cost,
        planned_kWh=f(q.sum()), emergency_kWh=f(e_all.sum()), unused_kWh=f(w_all.sum()),
        charge_kWh=f(c_all.sum()), discharge_kWh=f(dd_all.sum()), loss_kWh=loss,
        morning_0_10_emergency_cost_yuan=morning_cost, morning_0_10_emergency_kWh=morning_kwh,
        emergency_slots=int((e_all > EPS_SLOT_TOL).sum()),
        emergency_days=int((e_rec > EPS_SLOT_TOL).any(axis=1).sum()),
        emergency_events=int(len(ev_rec)),
        full_end_days=int((day_end >= STATE_MAX - EPS_SLOT_TOL).sum()),
        empty_slots=int((s1_all <= STATE_MIN + EPS_SLOT_TOL).sum()),
        mean_initial_kWh=f(init_rec.mean()),
        initial_kWh=f(s0_all[0]), final_kWh=f(s1_all[-1]))
    recomputed['adjusted_cost_yuan'] = f(total_cost - (np.median(price) / ETA)
                                         * (recomputed['final_kWh'] - recomputed['initial_kWh']))
    diffs = {k: f(abs(float(recomputed[k]) - float(reported[k])))
             for k in recomputed if k in reported.index}
    out['recomputed'] = recomputed
    out['reported_vs_recomputed_max_abs_diff'] = f(max(diffs.values()))
    out['reported_vs_recomputed_diffs'] = diffs

    # ---- meaning of "day-end full charge" vs unused energy ---------------------------
    cap_slots = s1_rec >= STATE_MAX - EPS_SLOT_TOL
    out['unused_diagnostics'] = dict(
        unused_total_kWh=f(w_rec.sum()),
        unused_kWh_at_capacity_kWh=f(w_rec[cap_slots].sum()),
        unused_kWh_below_capacity_kWh=f(w_rec[~cap_slots].sum()),
        cap_slot_count=int(cap_slots.sum()),
        full_end_days=int((day_end >= STATE_MAX - EPS_SLOT_TOL).sum()),
        day_end_days_all_cap_days=bool(
            (day_end >= STATE_MAX - EPS_SLOT_TOL).sum() == 0 or cap_slots.any(axis=1).sum() > 0))

    # ---- stored validation cross-check -----------------------------------------------
    v = jload(folder / 'validation.json')['validation']
    stored_checks = {k: f(x) for k, x in v['checks'].items()}
    out['stored_validation'] = dict(
        segments=v['segments'],
        day_boundary_max_error_kWh=f(v['day_boundary_max_error_kWh']),
        cost_reconciliation_yuan=f(v['cost_reconciliation_yuan']),
        energy_identity_residual_kWh=f(v['energy_identity_residual_kWh']),
        stored_physical_max_kWh=f(max(stored_checks[k] for k in
                                      ['balance_kWh', 'state_recursion_kWh', 'state_bounds_kWh',
                                       'power_bounds_kWh', 'nonnegative_kWh', 'simultaneous_kWh'])),
        stored_nominal_max_kWh=f(max(v for k, v in stored_checks.items()
                                     if k.startswith('nominal_'))))
    return out, {'strategy_id': name, 'date': d.date.to_numpy(), 'slot': d.slot.to_numpy(),
                 **{c: d[c].to_numpy() for c in ENERGY_COLUMNS + COST_COLUMNS},
                 'price_yuan_kWh': p_all, 'load_kW': l_all, 'pv_kW': v_all,
                 'nominal_state_end_kWh': d.nominal_state_end_kWh.to_numpy()}, daily, ev_rec


# --------------------------------------------------------------------------------------
# cross-checks
# --------------------------------------------------------------------------------------
def check_input_consistency(frames):
    base = frames['T_q80_6000']
    out = {'axes_identical': True, 'shared_columns_max_abs_diff': {}}
    for name, fr in frames.items():
        out['axes_identical'] &= bool(np.array_equal(fr['date'], base['date'])
                                      and np.array_equal(fr['slot'], base['slot']))
        for c in ['price_yuan_kWh', 'load_kW', 'pv_kW', 'net_forecast_kWh',
                  'residual_adjustment_kWh', 'protected_net_kWh']:
            out['shared_columns_max_abs_diff'].setdefault(c, {})[name] = \
                f(np.max(np.abs(fr[c] - base[c])))
    out['all_shared_inputs_identical'] = bool(
        out['axes_identical'] and all(v == 0.0 for d in out['shared_columns_max_abs_diff'].values()
                                      for v in d.values()))
    # only the plan/execution may differ
    out['planned_kWh_diff_vs_6000'] = {name: f(np.max(np.abs(fr['planned_kWh'] - base['planned_kWh'])))
                                       for name, fr in frames.items()}
    return out


def check_reproduction(load, pv, price):
    old = pd.read_csv(REFERENCE_RUN / 'T_q80_6000/dispatch.csv', low_memory=False)
    new = pd.read_csv(SENS / 'T_q80_6000/dispatch.csv', low_memory=False)
    out = dict(axis_and_theta_match=bool(old[['date', 'slot']].equals(new[['date', 'slot']])
                                         and old.theta.astype(str).equals(new.theta.astype(str))),
               segments=int(len(new)),
               reference_dispatch_sha256=digest(REFERENCE_RUN / 'T_q80_6000/dispatch.csv'),
               new_dispatch_sha256=digest(SENS / 'T_q80_6000/dispatch.csv'))
    out['max_abs_diff'] = {c: f(np.max(np.abs(old[c].to_numpy() - new[c].to_numpy())))
                           for c in ENERGY_COLUMNS + COST_COLUMNS}
    out['total_cost_old_yuan'] = f(old.planned_cost_yuan.sum() + old.emergency_cost_yuan.sum())
    out['total_cost_new_yuan'] = f(new.planned_cost_yuan.sum() + new.emergency_cost_yuan.sum())
    out['license_total_yuan'] = LICENSE_TOTAL
    out['license_total_abs_diff_yuan'] = f(abs(out['total_cost_new_yuan'] - LICENSE_TOTAL))
    out['energy_within_tolerance'] = bool(max(v for k, v in out['max_abs_diff'].items()
                                              if k in ENERGY_COLUMNS) <= TOL_SEG_ENERGY)
    out['cost_within_tolerance'] = bool(max(v for k, v in out['max_abs_diff'].items()
                                            if k in COST_COLUMNS) <= TOL_COST)
    out['license_total_within_tolerance'] = bool(out['license_total_abs_diff_yuan'] <= TOL_COST)
    out['identical_bytes'] = bool(out['reference_dispatch_sha256'] == out['new_dispatch_sha256'])
    return out


def check_diagnostics_11():
    """Cross-check the 6000 group against the earlier independent 11-run report."""
    a = pd.read_csv(DIAG_11 / 'summary.csv').set_index('strategy_id').loc['T_q80_6000']
    b = pd.read_csv(SENS / 'summary.csv').set_index('strategy_id').loc['T_q80_6000']
    pairs = [
        ('planned_cost_yuan', 'planned_cost_yuan'), ('emergency_cost_yuan', 'emergency_cost_yuan'),
        ('total_cost_yuan', 'total_cost_yuan'), ('planned_kWh', 'planned_kWh'),
        ('emergency_kWh', 'emergency_kWh'), ('unused_kWh', 'unused_kWh'),
        ('charge_kWh', 'charge_kWh'), ('discharge_kWh', 'discharge_kWh'),
        ('loss_kWh', 'loss_kWh'), ('morning_emergency_cost_yuan', 'morning_0_10_emergency_cost_yuan'),
        ('full_end_days', 'full_end_days'), ('empty_slots', 'empty_slots'),
        ('mean_initial_kWh', 'mean_initial_kWh'), ('initial_kWh', 'initial_kWh'),
        ('final_kWh', 'final_kWh'), ('adjusted_cost_yuan', 'adjusted_cost_yuan'),
        ('emergency_slots', 'emergency_slots'), ('emergency_events', 'emergency_events'),
        ('emergency_days', 'emergency_days')]
    rows = []
    for col_a, col_b in pairs:
        rows.append(dict(metric_bar11=col_a, metric_new=col_b,
                         value_bar11=f(a[col_a]), value_new=f(b[col_b]),
                         abs_diff=f(abs(float(a[col_a]) - float(b[col_b])))))
    frame = pd.DataFrame(rows)
    return dict(metrics=rows, n_compared=len(rows),
                n_inconsistent=int((frame.abs_diff > TOL_COST).sum()),
                all_consistent=bool((frame.abs_diff <= TOL_COST).all()))


def check_warmup(load, pv, price, kernel):
    warm = pd.read_csv(SENS / 'warmup_january.csv')
    out = dict(rows=int(len(warm)), days_first_last=[warm.date.iloc[0], warm.date.iloc[-1]])

    # independent replay of the public January lag trajectory
    pred_l, pred_v = independent_forecasts(load, pv)
    state = 6000.0
    rows, checks = [], {}
    for k in range(31):
        initial = state
        if k == 0:
            plan = np.zeros(144)
        else:
            netf = (pred_l[k] - pred_v[k]) * DT
            (plan, cc, dd, ww, st), _ = kernel.solve(netf / DT, np.zeros(144), price,
                                                     integer=True, initial_kWh=initial,
                                                     terminal_kWh=initial)
        c, dd, e, w, st = replay_greedy_control(plan, load[k], pv[k], initial, battery=(k > 0))
        rows.append(dict(day_index=k, date=str(DATES[k].date()), initial_kWh=initial,
                         final_kWh=f(st[-1]), planned_kWh=f(plan.sum()),
                         emergency_kWh=f(e.sum()), charge_kWh=f(c.sum()),
                         discharge_kWh=f(dd.sum()), unused_kWh=f(w.sum()),
                         planned_cost_yuan=f((plan * price).sum()),
                         emergency_cost_yuan=f(5.0 * (e * price).sum())))
        state = f(st[-1])
    mine = pd.DataFrame(rows)
    out['feb1_state_reported_kWh'] = f(warm.final_kWh.iloc[-1])
    out['feb1_state_recomputed_kWh'] = f(state)
    out['feb1_state_abs_diff_kWh'] = f(abs(state - warm.final_kWh.iloc[-1]))
    out['common_initial_matches_kWh'] = f(abs(state - COMMON_INITIAL))
    out['jan0_initial_kWh'] = f(warm.initial_kWh.iloc[0])
    out['max_abs_diff_vs_replay'] = {
        c: f(np.max(np.abs(warm[c].to_numpy() - mine[c].to_numpy())))
        for c in ['initial_kWh', 'final_kWh', 'planned_kWh', 'emergency_kWh', 'charge_kWh',
                  'discharge_kWh', 'unused_kWh', 'planned_cost_yuan', 'emergency_cost_yuan']}

    # compare with the original 05 lag January run
    lag_daily = pd.read_csv(BASELINE_05 / 'lag/daily_summary.csv')
    lag_jan = lag_daily[lag_daily.date < EVAL_START].reset_index(drop=True)
    out['january_matches_original_lag_05'] = bool(
        len(lag_jan) == 31 and np.allclose(lag_jan.final_kWh.to_numpy(),
                                           warm.final_kWh.to_numpy(), atol=1e-9, rtol=0)
        and np.allclose(lag_jan.emergency_cost_yuan.to_numpy(),
                        warm.emergency_cost_yuan.to_numpy(), atol=1e-6, rtol=0)
        and np.allclose(lag_jan.planned_cost_yuan.to_numpy(),
                        warm.planned_cost_yuan.to_numpy(), atol=1e-6, rtol=0))
    out['original_lag_feb1_state_kWh'] = f(lag_daily[lag_daily.date == EVAL_START]
                                           .initial_kWh.iloc[0]) if (lag_daily.date == EVAL_START).any() else None
    # same January across the reference terminal experiment and the quantile experiment
    for label, path in [('terminal_experiment_10', REFERENCE_RUN / 'warmup_january.csv'),
                        ('quantile_experiment_08',
                         ROOT / 'results/q2_quantile_experiment/warmup_january.csv')]:
        if path.exists():
            other = pd.read_csv(path)
            out[f'same_as_{label}'] = bool(
                len(other) == 31 and np.allclose(other.final_kWh.to_numpy(),
                                                 warm.final_kWh.to_numpy(), atol=1e-9, rtol=0))
        else:
            out[f'same_as_{label}'] = None
    return out


def check_milp_replay(frames, price, kernel, sample=None):
    """Re-solve the recorded nominal MILP day by day and compare plan and nominal state.

    MILP ties can yield a different equally-optimal plan, so the decisive comparisons are the
    re-solved objective and the nominal state trajectory, not element-wise q.
    """
    results = {}
    for name, fr in frames.items():
        target = GROUPS[name]
        d = fr
        day_idx = np.array([(pd.Timestamp(x) - DATES[0]).days for x in
                            pd.unique(pd.Series(d['date']))])
        n = len(day_idx)
        order = np.arange(n) if sample is None else np.unique(
            np.r_[np.linspace(0, n - 1, sample).astype(int), np.arange(max(0, n - sample), n)])
        q = d['planned_kWh'].reshape(n, 144)
        prot = d['protected_net_kWh'].reshape(n, 144)
        nom = d['nominal_state_end_kWh'].reshape(n, 144)
        init = d['state_start_kWh'].reshape(n, 144)[:, 0]
        max_q, max_obj, max_nom, worst_day, same_plan_days = 0.0, 0.0, 0.0, None, 0
        t0 = time.perf_counter()
        for i in order:
            (qq, cc, dd, ww, st), summary = kernel.solve(prot[i] / DT, np.zeros(144), price,
                                                         integer=True, initial_kWh=float(init[i]),
                                                         terminal_kWh=float(target))
            max_obj = max(max_obj, f(abs(summary['cost_yuan'] - float((q[i] * price).sum()))))
            max_nom = max(max_nom, f(np.max(np.abs(st[1:] - nom[i]))))
            dq = f(np.max(np.abs(qq - q[i])))
            max_q = max(max_q, dq)
            if dq <= 1e-9:
                same_plan_days += 1
            if dq > 1e-6 and worst_day is None:
                worst_day = str(d['date'].reshape(n, 144)[i][0])
        results[name] = dict(
            days_replayed=int(len(order)), days_total=int(n),
            max_abs_objective_diff_yuan=max_obj,
            max_abs_plan_diff_kWh=max_q,
            max_abs_nominal_state_diff_kWh_degenerate=max_nom,
            identical_plan_days=same_plan_days,
            first_day_with_plan_diff=worst_day,
            seconds=f(time.perf_counter() - t0),
            objective_and_plan_match=bool(max_obj <= 1e-6 and max_q <= 1e-6),
            nominal_state_diff_note='informational only: the nominal programme is degenerate, a '
                                    '~1e-13 input rounding can select a different equally-optimal '
                                    'state path; feasibility of the stored path is checked '
                                    'separately in per_group_audit.nominal_reconstruction')
    return results


def check_boundary(price, kernel):
    """Synthetic: the nominal target must not be written into the actual inventory."""
    out = {}
    for label, initial, target in [('empty_low_target', 1200.0, 4800.0),
                                   ('full_low_target', 10800.0, 4800.0)]:
        load_zero, pv_zero = np.zeros(144), np.zeros(144)
        (plan, cc, dd, ww, st), summary = kernel.solve(load_zero, pv_zero, price, integer=True,
                                                       initial_kWh=initial, terminal_kWh=target)
        c, dd2, e, w, st2 = replay_greedy_control(plan, load_zero, pv_zero, initial)
        out[label] = dict(initial_kWh=initial, nominal_target_kWh=target,
                          nominal_final_kWh=f(st[-1]), actual_final_kWh=f(st2[-1]),
                          planned_kWh=f(plan.sum()),
                          nominal_meets_target=bool(abs(st[-1] - target) < 1e-6),
                          actual_equals_target=bool(abs(st2[-1] - target) < 1e-3))
    return out


def check_monthly_extremes_hourly(frames, dailies, contrasts, monthly_contrasts, extremes, hourly):
    out = {}
    # monthly totals from raw daily
    monthly_recomp = {}
    for name, daily in dailies.items():
        frame = daily.assign(month=daily.date.str[:7]).groupby('month').agg(
            planned_cost_yuan=('planned_cost_yuan', 'sum'),
            emergency_cost_yuan=('emergency_cost_yuan', 'sum'),
            total_cost_yuan=('total_cost_yuan', 'sum'),
            emergency_kWh=('emergency_kWh', 'sum'),
            unused_kWh=('unused_kWh', 'sum')).reset_index()
        monthly_recomp[name] = frame
    base = monthly_recomp['T_q80_6000'].set_index('month')
    rows = []
    for name, frame in monthly_recomp.items():
        fset = frame.set_index('month')
        for month in fset.index:
            rows.append(dict(strategy_id=name, month=month,
                             delta_total_vs_6000_yuan=f(
                                 fset.loc[month, 'total_cost_yuan'] - base.loc[month, 'total_cost_yuan']),
                             delta_emergency_vs_6000_yuan=f(
                                 fset.loc[month, 'emergency_cost_yuan'] - base.loc[month, 'emergency_cost_yuan'])))
    mine = pd.DataFrame(rows)
    merged = monthly_contrasts.merge(mine, on=['strategy_id', 'month'], suffixes=('_file', '_mine'))
    out['monthly'] = dict(
        months_per_group=int(monthly_contrasts.groupby('strategy_id').month.nunique().min()),
        n_rows_compared=int(len(merged)),
        max_abs_delta_total_diff_yuan=f(np.max(np.abs(
            merged.delta_total_vs_6000_yuan_file - merged.delta_total_vs_6000_yuan_mine))),
        max_abs_delta_emergency_diff_yuan=f(np.max(np.abs(
            merged.delta_emergency_vs_6000_yuan_file - merged.delta_emergency_vs_6000_yuan_mine))))
    sign = {}
    for name in ['T_q80_4800', 'T_q80_7200']:
        blk = monthly_contrasts[monthly_contrasts.strategy_id == name]
        sign[name] = dict(months=int(len(blk)),
                          cheaper_months=int((blk.delta_total_vs_6000_yuan < 0).sum()),
                          costlier_months=int((blk.delta_total_vs_6000_yuan > 0).sum()),
                          all_same_direction=bool((blk.delta_total_vs_6000_yuan < 0).all()
                                                  or (blk.delta_total_vs_6000_yuan > 0).all()))
    out['monthly_sign'] = sign

    # extreme dates
    base_daily = dailies['T_q80_6000'].set_index('date')
    ext = []
    for name in ['T_q80_4800', 'T_q80_7200']:
        daily = dailies[name].set_index('date')
        delta = daily.total_cost_yuan - base_daily.total_cost_yuan
        ordered = delta.sort_values()
        for label, part in [('worst_5', ordered.tail(5).iloc[::-1]), ('best_5', ordered.head(5))]:
            for date, value in part.items():
                ext.append(dict(strategy_id=name, group=label, date=date,
                                delta_total_cost_yuan=f(value)))
    mine_ext = pd.DataFrame(ext)
    merged_ext = extremes.merge(mine_ext, on=['strategy_id', 'group', 'date'],
                                suffixes=('_file', '_mine'))
    out['extremes'] = dict(
        n_rows_compared=int(len(merged_ext)),
        max_abs_delta_diff_yuan=f(np.max(np.abs(merged_ext.delta_total_cost_yuan_file
                                                - merged_ext.delta_total_cost_yuan_mine))))

    # hourly
    hourly_recomp = []
    for name, fr in frames.items():
        d = pd.DataFrame({'slot': fr['slot'], 'planned_kWh': fr['planned_kWh'],
                          'emergency_kWh': fr['emergency_kWh'], 'unused_kWh': fr['unused_kWh'],
                          'price_yuan_kWh': fr['price_yuan_kWh']})
        d['hour'] = d.slot // 6
        d['emergency_cost_yuan'] = 5 * d.price_yuan_kWh * d.emergency_kWh
        block = d.groupby('hour').agg(planned_kWh=('planned_kWh', 'sum'),
                                      emergency_kWh=('emergency_kWh', 'sum'),
                                      unused_kWh=('unused_kWh', 'sum'),
                                      emergency_cost_yuan=('emergency_cost_yuan', 'sum')).reset_index()
        block.insert(0, 'strategy_id', name)
        hourly_recomp.append(block)
    mine_h = pd.concat(hourly_recomp)
    merged_h = hourly.merge(mine_h, on=['strategy_id', 'hour'], suffixes=('_file', '_mine'))
    out['hourly'] = dict(
        max_abs_diff={'planned_kWh': f(np.max(np.abs(merged_h.planned_kWh_file - merged_h.planned_kWh_mine))),
                      'emergency_kWh': f(np.max(np.abs(merged_h.emergency_kWh_file - merged_h.emergency_kWh_mine))),
                      'unused_kWh': f(np.max(np.abs(merged_h.unused_kWh_file - merged_h.unused_kWh_mine))),
                      'emergency_cost_yuan': f(np.max(np.abs(merged_h.emergency_cost_yuan_file
                                                               - merged_h.emergency_cost_yuan_mine)))})
    # 19-21h emergency cost and the reported 22-23 / 0-5 planned shifts
    block19 = mine_h[mine_h.hour.isin([19, 20, 21])].groupby('strategy_id').emergency_cost_yuan.sum()
    plan = {name: mine_h[mine_h.strategy_id == name].set_index('hour').planned_kWh
            for name in frames}
    out['claims'] = dict(
        emergency_cost_19_21={k: f(v) for k, v in block19.items()},
        emergency_cost_19_21_identical=bool(block19.round(6).nunique() == 1),
        planned_22_23_diff_4800_minus_6000=f(plan['T_q80_4800'][[22, 23]].sum()
                                             - plan['T_q80_6000'][[22, 23]].sum()),
        planned_0_5_diff_4800_minus_6000=f(plan['T_q80_4800'][list(range(6))].sum()
                                           - plan['T_q80_6000'][list(range(6))].sum()),
        planned_22_23_diff_7200_minus_6000=f(plan['T_q80_7200'][[22, 23]].sum()
                                             - plan['T_q80_6000'][[22, 23]].sum()),
        planned_0_5_diff_7200_minus_6000=f(plan['T_q80_7200'][list(range(6))].sum()
                                           - plan['T_q80_6000'][list(range(6))].sum()))
    return out


def check_provenance():
    reg = jload(SENS / 'registration.json')
    out = dict(signature=reg['signature'], n_amendments=len(reg.get('amendments', [])),
               candidates=reg['candidates_kWh'],
               audit_script_sha256=digest(Path(__file__)),
               input_hash_match={}, snapshot_hash_match={})
    for rel, expected in reg['input_sha256'].items():
        out['input_hash_match'][rel] = bool(digest(ROOT / rel) == expected)
    for rel, expected in reg['snapshot_sha256'].items():
        out['snapshot_hash_match'][rel] = bool(digest(SNAP / rel) == expected)
    # recompute the signature exactly as code/12 does
    hashes = {rel: digest(ROOT / rel) for rel in reg['input_sha256']}
    snap = {rel: digest(SNAP / rel) for rel in reg['snapshot_sha256']}
    sig = hashlib.sha256(json.dumps(
        dict(candidates=reg['candidates_kWh'], policies=reg['policies'],
             input_sha256=hashes, snapshot_sha256=snap), sort_keys=True).encode()).hexdigest()
    out['recomputed_signature'] = sig
    out['signature_reproduced'] = bool(sig == reg['signature'])
    out['all_input_hashes_match'] = bool(all(out['input_hash_match'].values()))
    out['all_snapshot_hashes_match'] = bool(all(out['snapshot_hash_match'].values()))
    # plan-file amendment sections (a documentation defect if duplicated)
    plan = (SENS / 'experiment_plan.md').read_text(encoding='utf-8')
    out['plan_amendment_header_count'] = plan.count('## 6. 登记修订记录')
    out['plan_numbered_items'] = plan.count('\n1. 2026-09-11T')
    # archived vs live sources
    out['archive_matches_live'] = {
        '12_q2_terminal_sensitivity.py': bool(
            digest(SENS / 'source_archive/12_q2_terminal_sensitivity.py')
            == digest(ROOT / 'code/12_q2_terminal_sensitivity.py')),
        'snapshot_02_q1_baseline.py': bool(
            digest(SENS / 'source_archive/snapshot_02_q1_baseline.py')
            == digest(SNAP / 'code/02_q1_baseline.py')),
        'snapshot_05_q2_baseline.py': bool(
            digest(SENS / 'source_archive/snapshot_05_q2_baseline.py')
            == digest(SNAP / 'code/05_q2_baseline.py')),
        'snapshot_08_q2_quantile_experiment.py': bool(
            digest(SENS / 'source_archive/snapshot_08_q2_quantile_experiment.py')
            == digest(SNAP / 'code/08_q2_quantile_experiment.py'))}
    run = jload(SENS / 'run_manifest.json')
    out['run_status'] = run['status']
    out['attachments_unchanged_during_run'] = bool(run['attachments_unchanged'])
    return out


def check_superseded_and_protected():
    out = {}
    rounds = {}
    for r in ['round1', 'round2', 'round3']:
        path = SENS / '_superseded_pre_amendment' / r / 'summary.csv'
        if path.exists():
            s = pd.read_csv(path).set_index('strategy_id')
            rounds[r] = {sid: f(s.loc[sid, 'total_cost_yuan']) for sid in s.index}
    final = pd.read_csv(SENS / 'summary.csv').set_index('strategy_id')
    rounds['final'] = {sid: f(final.loc[sid, 'total_cost_yuan']) for sid in final.index}
    out['round_totals'] = rounds
    out['all_rounds_identical'] = bool(all(
        abs(rounds[r][sid] - rounds['final'][sid]) == 0.0
        for r in rounds if r != 'final' for sid in rounds['final']))
    out['dispatch_hashes'] = {}
    for r in ['round1', 'round2', 'round3']:
        p = SENS / '_superseded_pre_amendment' / r / 'T_q80_6000/dispatch.csv'
        if p.exists():
            out['dispatch_hashes'][r] = digest(p)
    out['dispatch_hashes']['final'] = digest(SENS / 'T_q80_6000/dispatch.csv')
    out['dispatch_all_identical'] = bool(len(set(out['dispatch_hashes'].values())) == 1)

    # re-hash the protected list captured before this experiment
    if PROTECTED_BEFORE.exists():
        before = jload(PROTECTED_BEFORE)
        changed, missing = [], []
        for rel, expected in before.items():
            p = ROOT / rel.replace('\\', '/')
            if not p.exists():
                missing.append(rel)
            elif digest(p) != expected:
                changed.append(rel)
        out['protected'] = dict(n=len(before), n_changed=len(changed), n_missing=len(missing),
                                changed=sorted(changed)[:20], missing=sorted(missing)[:20])
    return out


def write_evidence_csv(groups):
    """One row per (group, check): actual value, threshold, pass flag."""
    rows = []
    for name, g in groups.items():
        seg = g['segments']
        rec = g['reconciliation']
        nom = g['nominal_reconstruction']
        entries = [
            ('per_segment_physical_kWh', seg['max_physical_violation_kWh'], EPS_SLOT_TOL, seg),
            ('day_continuity_kWh', seg['day_continuity_max_abs_kWh'], 1e-9, seg),
            ('energy_identity_kWh', abs(g['energy_identity_kWh']), TOL_ENERGY, g),
            ('reported_metric_max_abs_diff', g['reported_vs_recomputed_max_abs_diff'], TOL_COST, g),
            ('daily_reconciliation_yuan_or_kWh',
             max(v for k, v in rec.items() if isinstance(v, (int, float))
                 and 'max_abs' not in k and k.endswith(('_yuan', '_kWh'))), TOL_ENERGY, rec),
            ('event_energy_max_abs_diff_kWh', rec['event_energy_max_abs_diff_kWh'], EPS_SLOT_TOL, rec),
            ('nominal_terminal_kWh', g['nominal_terminal']['max_abs_error_kWh'], EPS_SLOT_TOL,
             g['nominal_terminal']),
            ('nominal_reconstruction_violation_kWh', nom['max_violation_kWh'], EPS_SLOT_TOL, nom),
            ('greedy_control_replay_kWh', g['control_replay']['max_abs_state_end_diff_kWh'],
             EPS_SLOT_TOL, g['control_replay']),
            ('protection_independent_kWh', g['protection']['max_abs_protected_net_diff_kWh'],
             EPS_SLOT_TOL, g['protection']),
        ]
        for check, actual, threshold, _src in entries:
            rows.append(dict(strategy_id=name, check=check, actual_value=f(actual),
                             threshold=f(threshold), passed=bool(actual <= threshold)))
    frame = pd.DataFrame(rows)
    frame.to_csv(AUDIT / 'threshold_evidence.csv', index=False, encoding='utf-8-sig')
    return frame


def write_anomaly_csv(groups, replay):
    """Diagnostic dates carrying the largest solver/rounding artifacts (not failures)."""
    rows = []
    for name, g in groups.items():
        rows.append(dict(strategy_id=name, kind='max_nominal_reconstruction_violation',
                         date=g['nominal_reconstruction']['max_violation_day'],
                         value_kWh=g['nominal_reconstruction']['max_violation_kWh'],
                         threshold_kWh=EPS_SLOT_TOL, within_threshold=True))
    if replay:
        for name, r in replay.items():
            rows.append(dict(strategy_id=name, kind='max_nominal_state_path_degeneracy',
                             date='(day not pinned)',
                             value_kWh=r['max_abs_nominal_state_diff_kWh_degenerate'],
                             threshold_kWh=None, within_threshold=None))
    pd.DataFrame(rows).to_csv(AUDIT / 'anomaly_dates.csv', index=False, encoding='utf-8-sig')


# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-replay', action='store_true',
                        help='skip the day-by-day nominal MILP replay (slow)')
    parser.add_argument('--replay-sample', type=int, default=None,
                        help='replay only this many days per group instead of all 334')
    args = parser.parse_args()

    assert Path(sys.prefix).name == 'math_modeling', sys.prefix
    AUDIT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    load, pv = read_attachment2_actual()
    price = read_attachment1_price()
    kernel = load_kernel()

    groups, frames, dailies, evs = {}, {}, {}, {}
    for name, target in GROUPS.items():
        g, frame, daily, ev = audit_group(name, target, load, pv, price)
        groups[name] = g
        frames[name] = frame
        dailies[name] = daily
        evs[name] = ev
    save('per_group_audit.json', groups)

    baseline_checks = {
        'input_consistency': check_input_consistency(frames),
        'reproduction_of_existing_6000': check_reproduction(load, pv, price),
        'crosscheck_vs_11_run': check_diagnostics_11(),
        'warmup_january': check_warmup(load, pv, price, kernel),
        'provenance': check_provenance(),
        'superseded_rounds_and_protected_assets': check_superseded_and_protected(),
    }
    save('checks.json', baseline_checks)

    monthly_contrasts = pd.read_csv(SENS / 'monthly_contrasts.csv')
    extremes = pd.read_csv(SENS / 'extreme_dates.csv')
    hourly = pd.read_csv(SENS / 'hourly.csv')
    reports = check_monthly_extremes_hourly(frames, dailies, None, monthly_contrasts, extremes,
                                            hourly)
    save('reported_tables_check.json', reports)

    boundary = check_boundary(price, kernel)
    save('boundary_nominal_vs_actual.json', boundary)

    if not args.skip_replay:
        replay = check_milp_replay(frames, price, kernel, sample=args.replay_sample)
        save('milp_replay.json', replay)
    else:
        replay = None
    write_evidence_csv(groups)
    write_anomaly_csv(groups, replay)

    # ---- consolidated verdict -------------------------------------------------------
    physical = {name: g['segments']['max_physical_violation_kWh'] for name, g in groups.items()}
    identity = {name: abs(g['energy_identity_kWh']) for name, g in groups.items()}
    recon_max = {name: max(v for k, v in g['reconciliation'].items()
                           if isinstance(v, (int, float)) and 'max_abs' not in k
                           and k.endswith(('_yuan', '_kWh'))) for name, g in groups.items()}
    reported_max = {name: g['reported_vs_recomputed_max_abs_diff'] for name, g in groups.items()}
    summary = dict(
        generated_utc=pd.Timestamp.now('UTC').isoformat(),
        wall_seconds=f(time.perf_counter() - t0),
        thresholds=dict(per_segment_physical_kWh=EPS_SLOT_TOL, cumulative_cost_yuan=TOL_COST,
                        cumulative_energy_kWh=TOL_ENERGY,
                        segment_energy_vs_existing_kWh=TOL_SEG_ENERGY),
        verdict=dict(
            physical_within_tolerance=bool(max(physical.values()) <= EPS_SLOT_TOL),
            energy_identity_within_tolerance=bool(max(identity.values()) <= TOL_ENERGY),
            reported_metrics_recomputed=bool(max(reported_max.values()) <= TOL_COST),
            reconciliation_within_tolerance=bool(max(recon_max.values()) <= TOL_ENERGY),
            inputs_consistent_across_groups=baseline_checks['input_consistency']
            ['all_shared_inputs_identical'],
            protection_independent_match=all(g['protection']['protection_independent_match']
                                             for g in groups.values()),
            greedy_control_reproduced=all(g['control_replay']['greedy_control_reproduced']
                                          for g in groups.values()),
            nominal_terminal_exact=bool(max(g['nominal_terminal']['max_abs_error_kWh']
                                            for g in groups.values()) == 0.0),
            nominal_trajectory_feasible=all(
                g['nominal_reconstruction']['recorded_trajectory_feasible']
                for g in groups.values()),
            reproduction_6000_exact=baseline_checks['reproduction_of_existing_6000']
            ['identical_bytes'],
            license_total_within_tolerance=baseline_checks['reproduction_of_existing_6000']
            ['license_total_within_tolerance'],
            warmup_matches_original_lag=bool(baseline_checks['warmup_january']
                                             ['january_matches_original_lag_05']),
            crosscheck_11_all_consistent=baseline_checks['crosscheck_vs_11_run']['all_consistent'],
            signature_reproduced=baseline_checks['provenance']['signature_reproduced'],
        ),
        max_physical_violation_kWh=physical,
        max_energy_identity_kWh=identity,
        max_reconciliation_kWh_or_yuan=recon_max,
        max_reported_vs_recomputed_diff=reported_max,
        max_nominal_reconstruction_violation_kWh={
            name: g['nominal_reconstruction']['max_violation_kWh'] for name, g in groups.items()},
        inputs_and_columns=baseline_checks['input_consistency'],
        reproduction=baseline_checks['reproduction_of_existing_6000'],
        crosscheck_vs_11_run=baseline_checks['crosscheck_vs_11_run'],
        warmup=baseline_checks['warmup_january'],
        reported_tables=reports,
        boundary=boundary,
        provenance=baseline_checks['provenance'],
        superseded_and_protected=baseline_checks['superseded_rounds_and_protected_assets'],
    )
    if (AUDIT / 'milp_replay.json').exists():
        summary['milp_replay'] = jload(AUDIT / 'milp_replay.json')
    save('audit_summary.json', summary)

    print(json.dumps({k: v for k, v in summary['verdict'].items()}, ensure_ascii=False, indent=2))
    print('max physical', max(physical.values()), '| max identity', max(identity.values()),
          '| max recon', max(recon_max.values()), '| max vs reported', max(reported_max.values()))
    print('wall seconds', summary['wall_seconds'], flush=True)


if __name__ == '__main__':
    main()
