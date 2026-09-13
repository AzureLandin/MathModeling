"""Substitute attachment-2 day-52 load and PV into the locked problem-1 model.

Run (math_modeling env):
    E:/Anaconda/envs/math_modeling/python.exe code/13_q1_day52_recompute.py
    E:/Anaconda/envs/math_modeling/python.exe code/13_q1_day52_recompute.py --day 52

What is substituted and what is held fixed
    substituted   the 小区负载 and 光伏发电实际功率 rows of 附件2 for the selected day
                  (default day 52 of 2025, counting 2025-01-01 as day 1 -> 2025-02-21)
    held fixed    everything else of the locked problem-1 model: the 144-point price curve of
                  附件1, dt = 1/6 h, eta_c = eta_d = 0.9, state bounds 1200-10800 kWh, bus-side
                  power limit 5000 kW, charge/discharge mutual exclusion, E_0 = E_144 = 6000 kWh,
                  the original MILP kernel of code/02_q1_baseline.py (same matrix build, variable
                  order, mip_rel_gap 1e-9, time limit 120 s), and the right-endpoint time
                  convention (sample labelled 00:10 represents 00:00-00:10).

附件2 carries no price column, and problem 1 is defined on the 附件1 price curve; 附件4 (the
fluctuating price) belongs to problem 4 and is deliberately not used here.

Outputs
    results/q1_day52/          tables, configuration, validation
    figures/q1_day52.png       run curve
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import openpyxl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/q1_day52'
CORE = ROOT / 'code/02_q1_baseline.py'
LOCKED = ROOT / 'results/q1_milp/validation_summary.json'
SELECTED_SLOTS = [60, 72, 84, 96, 108, 120]      # table 1 intervals, slots 10:00 .. 20:00
DEFAULT_DAY = 52
DEFAULT_TERMINAL = 6000.0


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_core():
    spec = importlib.util.spec_from_file_location('q1_core_locked', CORE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_attachment1(core):
    wb = openpyxl.load_workbook(ROOT / '附件/附件1.xlsx', read_only=True, data_only=True)
    rows = list(wb.worksheets[0].values)
    wb.close()
    price = np.array([r[1] for r in rows[1:]], dtype=float)
    load = np.array([r[2] for r in rows[1:]], dtype=float)
    pv = np.array([r[3] for r in rows[1:]], dtype=float)
    assert price.shape == load.shape == pv.shape == (144,)
    assert (price > 0).all() and (load >= 0).all() and (pv >= 0).all()
    return load, pv, price


def read_attachment2_day(day, core):
    """Return (date, load_kW, pv_kW) for the given 1-based day of 2025."""
    wb = openpyxl.load_workbook(ROOT / '附件/附件2.xlsx', read_only=True, data_only=True)
    sheets = {ws.title: list(ws.values) for ws in wb.worksheets}
    wb.close()
    assert set(sheets) == {'小区负载', '光伏发电实际功率'}, sheets.keys()
    dates = [r[0] for r in sheets['小区负载'][1:]]
    assert len(dates) == 365
    expected = pd.Timestamp('2025-01-01') + pd.Timedelta(days=day - 1)
    assert pd.Timestamp(dates[0]) == pd.Timestamp('2025-01-01')
    assert pd.Timestamp(dates[-1]) == pd.Timestamp('2025-12-31')
    position = day - 1                       # header occupies row 0
    assert pd.Timestamp(dates[position]) == expected, (dates[position], expected)
    load = np.array(sheets['小区负载'][position + 1][1:], dtype=float)
    pv = np.array(sheets['光伏发电实际功率'][position + 1][1:], dtype=float)
    for name, other in [('小区负载', sheets['小区负载']), ('光伏发电实际功率', sheets['光伏发电实际功率'])]:
        assert pd.Timestamp(other[position + 1][0]) == expected, f'{name} row date mismatch'
    assert load.shape == pv.shape == (144,)
    assert np.isfinite(load).all() and np.isfinite(pv).all()
    assert (load >= 0).all() and (pv >= 0).all()
    return expected, load, pv


def solve_z_relaxation(load, pv, price, initial, terminal, core):
    """Same matrices as the locked kernel, but the charge-mode variable z is continuous in [0,1].

    The project's designated LP (core.solve(integer=False)) relaxes the upper limits of c and d
    independently and therefore admits simultaneous charging and discharging. This variant keeps
    the mutual-exclusion rows and only drops integrality, so its feasible set is a strict subset
    of the independent-bound LP and a superset of the MILP.
    """
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import lil_matrix, csr_matrix, hstack, vstack
    n = len(load)
    m = 5 * n + 1
    cap = core.CONFIG['power_max_kW'] * core.CONFIG['dt_hours']
    eta = core.CONFIG['charge_efficiency']
    obj = np.zeros(m + n)
    obj[:n] = price
    eq = lil_matrix((2 * n, m))
    rhs = np.r_[(load - pv) * core.CONFIG['dt_hours'], np.zeros(n)]
    for t in range(n):
        eq[t, t], eq[t, n + t], eq[t, 2 * n + t], eq[t, 3 * n + t] = 1, -1, 1, -1
        eq[n + t, 4 * n + t + 1], eq[n + t, 4 * n + t] = 1, -1
        eq[n + t, n + t], eq[n + t, 2 * n + t] = -eta, 1 / eta
    constraints = hstack([csr_matrix(eq), csr_matrix((2 * n, n))], format='csr')
    mutex = lil_matrix((2 * n, m + n))
    for t in range(n):
        mutex[t, n + t], mutex[t, m + t] = 1, -cap
        mutex[n + t, 2 * n + t], mutex[n + t, m + t] = 1, cap
    constraints = vstack([constraints, csr_matrix(mutex)], format='csr')
    lower, upper = np.zeros(m + n), np.full(m + n, np.inf)
    upper[n:3 * n] = cap
    lower[4 * n:5 * n + 1], upper[4 * n:5 * n + 1] = (core.CONFIG['state_min_kWh'],
                                                      core.CONFIG['state_max_kWh'])
    lower[4 * n] = upper[4 * n] = float(initial)
    lower[5 * n] = upper[5 * n] = float(terminal)
    lower[m:], upper[m:] = 0.0, 1.0
    result = milp(obj, integrality=np.zeros(m + n), bounds=Bounds(lower, upper),
                  constraints=LinearConstraint(constraints, np.r_[rhs, np.full(2 * n, -np.inf)],
                                               np.r_[rhs, np.zeros(n), np.full(n, cap)]),
                  options=dict(mip_rel_gap=core.CONFIG['milp_relative_gap'],
                               time_limit=core.CONFIG['milp_time_limit_seconds']))
    if not result.success:
        raise RuntimeError(result.message)
    x = result.x
    return x[:n], x[n:2 * n], x[2 * n:3 * n], x[3 * n:4 * n], x[4 * n:5 * n + 1]


def decycle(c, d, w, eta_c=0.9, eta_d=0.9):
    """Remove charge/discharge cycles from an LP plan, preserving q, the state path and the cost.

    With r = eta_c*eta_d and delta = min(c, d/r): c' = c - delta, d' = d - r*delta,
    w' = w + (1-r)*delta. At least one of c', d' is zero, the state recursion and the bus balance
    are unchanged, and only q enters the objective.
    """
    r = eta_c * eta_d
    delta = np.minimum(c, d / r)
    return c - delta, d - r * delta, w + (1 - r) * delta, delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--day', type=int, default=DEFAULT_DAY,
                        help='1-based day of 2025 (2025-01-01 is day 1)')
    parser.add_argument('--terminal-kWh', type=float, default=DEFAULT_TERMINAL)
    args = parser.parse_args()

    assert Path(sys.prefix).name == 'math_modeling', sys.prefix
    np.random.seed(20260910)
    OUT.mkdir(parents=True, exist_ok=True)

    core = load_core()
    load1, pv1, price = read_attachment1(core)
    date, load2, pv2 = read_attachment2_day(args.day, core)
    terminal = float(args.terminal_kWh)

    # Same locked model and kernel; only the load/PV day changes.
    (q, c, d, w, state), summary = core.solve(load2, pv2, price, integer=True,
                                              initial_kWh=terminal, terminal_kWh=terminal)
    (ql, cl, dl, wl, sl), lp_summary = core.solve(load2, pv2, price, integer=False,
                                                 initial_kWh=terminal, terminal_kWh=terminal)
    lz_q, lz_c, lz_d, lz_w, lz_s = solve_z_relaxation(load2, pv2, price, terminal, terminal, core)
    cl2, dl2, wl2, delta = decycle(cl, dl, wl,
                                   core.CONFIG['charge_efficiency'],
                                   core.CONFIG['discharge_efficiency'])
    lp_block = dict(
        designated_lp=dict(
            formulation='core.solve(integer=False): linprog/HiGHS, c_t and d_t bounded '
                        'independently by M, mutual exclusion relaxed',
            cost_yuan=float(price @ ql), grid_kWh=float(ql.sum()),
            charge_kWh=float(cl.sum()), discharge_kWh=float(dl.sum()),
            unused_kWh=float(wl.sum()), state_min_kWh=float(sl.min()),
            state_max_kWh=float(sl.max()), solver=lp_summary['solver'],
            cost_gap_vs_milp_yuan=float(abs(price @ ql - price @ q))),
        z_relaxation_lp=dict(
            formulation='same matrices as the MILP kernel with the charge-mode variable z in [0,1]; '
                        'mutual-exclusion rows kept',
            cost_yuan=float(price @ lz_q), grid_kWh=float(lz_q.sum()),
            charge_kWh=float(lz_c.sum()), discharge_kWh=float(lz_d.sum()),
            unused_kWh=float(lz_w.sum()), state_min_kWh=float(lz_s.min()),
            state_max_kWh=float(lz_s.max()),
            cost_gap_vs_milp_yuan=float(abs(price @ lz_q - price @ q)),
            cost_gap_vs_designated_lp_yuan=float(abs(price @ lz_q - price @ ql))),
        simultaneous_charge_discharge_kWh=dict(
            milp=float(np.minimum(c, d).max()), designated_lp=float(np.minimum(cl, dl).max()),
            z_relaxation_lp=float(np.minimum(lz_c, lz_d).max()),
            threshold_kWh=1e-6),
        decycled_designated_lp=dict(
            formulation="c'=c-delta, d'=d-0.81*delta, w'=w+0.19*delta with "
                        'delta=min(c, d/0.81); preserves q, the state path and the cost',
            removed_cycle_energy_kWh=float(delta.sum()),
            max_simultaneous_kWh=float(np.minimum(cl2, dl2).max()),
            cost_gap_vs_milp_yuan=float(abs(price @ ql - price @ q)),
            charge_kWh=float(cl2.sum()), discharge_kWh=float(dl2.sum()),
            unused_kWh=float(wl2.sum()),
            state_recursion_gap_kWh=float(np.max(np.abs(
                (sl[1:] - sl[:-1]) - (core.CONFIG['charge_efficiency'] * cl2
                                      - dl2 / core.CONFIG['discharge_efficiency'])))),
            balance_gap_kWh=float(np.max(np.abs(
                ql + dl2 - (load2 - pv2) / 6 - cl2 - wl2)))),
        lp_vs_milp_slots=dict(
            grid_kWh=float(np.max(np.abs(ql - q))), charge_kWh=float(np.max(np.abs(cl - c))),
            discharge_kWh=float(np.max(np.abs(dl - d))), unused_kWh=float(np.max(np.abs(wl - w))),
            state_kWh=float(np.max(np.abs(sl - state))),
            slots_with_grid_difference=int((np.abs(ql - q) > 1e-6).sum()),
            slots_with_state_difference=int((np.abs(sl - state) > 1e-6).sum()),
            grid_total_difference_kWh=float(abs(ql.sum() - q.sum()))),
        ordering_note='LP with independent bounds has the widest feasible set, the z-relaxation LP '
                      'a subset of it, and the MILP the smallest; hence '
                      'cost(LP) <= cost(z-relax) <= cost(MILP) must hold.')

    no_grid = np.maximum(load2 - pv2, 0.0) / 6
    no_storage = dict(grid_kWh=float(no_grid.sum()), cost_yuan=float(price @ no_grid),
                      unused_kWh=float(np.maximum(pv2 - load2, 0.0).sum() / 6))

    locked = json.loads(LOCKED.read_text(encoding='utf-8'))['summary']

    cost_lp, cost_lz, cost_milp = float(price @ ql), float(price @ lz_q), float(price @ q)
    lp_block['feasible_set_ordering_holds'] = bool(
        cost_lp <= cost_lz + 1e-9 and cost_lz <= cost_milp + 1e-9)
    lp_block['all_three_costs_equal'] = bool(
        max(cost_lp, cost_lz, cost_milp) - min(cost_lp, cost_lz, cost_milp) <= 1e-9)

    # Diagnostic only: the day curtails PV surplus even while the battery sits at its floor, so the
    # binding resource is not obvious. Re-solving with other forced day-end states shows whether the
    # curtailment is tied to the model's E_0 = E_144 = 6000 rule. These variants are NOT the
    # problem-1 result and are not reported as such.
    diagnostic = []
    for terminal_probe in (1200.0, 6000.0, 10800.0):
        (qp, cp, dp, wp, sp), summaryp = core.solve(load2, pv2, price, integer=True,
                                                   initial_kWh=terminal, terminal_kWh=terminal_probe)
        diagnostic.append(dict(terminal_kWh=terminal_probe, grid_kWh=float(qp.sum()),
                               cost_yuan=float(price @ qp), charge_kWh=float(cp.sum()),
                               discharge_kWh=float(dp.sum()), unused_kWh=float(wp.sum()),
                               state_min_kWh=float(sp.min()), state_max_kWh=float(sp.max()),
                               note='diagnostic only; not the problem-1 model result'))

    table = pd.DataFrame(dict(
        slot=np.arange(144),
        interval=[f'{core.label(t)}-{core.label(t + 10)}' for t in range(0, 1440, 10)],
        load_kW=load2, pv_kW=pv2, price_yuan_kWh=price,
        grid_kWh=q, charge_bus_kWh=c, discharge_bus_kWh=d, unused_kWh=w,
        state_start_kWh=state[:-1], state_end_kWh=state[1:], cost_yuan=q * price))
    table.to_csv(OUT / 'dispatch_10min.csv', index=False, encoding='utf-8-sig')

    groups = pd.DataFrame([dict(interval=f'{core.label(t * 10)}-{core.label((t + 24) * 10)}',
                                charge_kWh=float(c[t:t + 24].sum()),
                                discharge_kWh=float(d[t:t + 24].sum()))
                           for t in range(0, 144, 24)])
    groups.to_csv(OUT / 'charge_discharge_4hour.csv', index=False, encoding='utf-8-sig')

    selected = table[table.slot.isin(SELECTED_SLOTS)][['interval', 'grid_kWh', 'cost_yuan']]
    selected.to_csv(OUT / 'selected_intervals.csv', index=False, encoding='utf-8-sig')

    lp_table = pd.DataFrame(dict(
        slot=np.arange(144), interval=table.interval,
        load_kW=load2, pv_kW=pv2, price_yuan_kWh=price,
        grid_lp_kWh=ql, charge_lp_kWh=cl, discharge_lp_kWh=dl, unused_lp_kWh=wl,
        state_start_lp_kWh=sl[:-1], state_end_lp_kWh=sl[1:],
        charge_decycled_kWh=cl2, discharge_decycled_kWh=dl2, unused_decycled_kWh=wl2,
        grid_milp_kWh=q, charge_milp_kWh=c, discharge_milp_kWh=d, unused_milp_kWh=w,
        state_end_milp_kWh=state[1:]))
    lp_table.to_csv(OUT / 'lp_dispatch_10min.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([dict(interval=f'{core.label(t * 10)}-{core.label((t + 24) * 10)}',
                       charge_lp_kWh=float(cl[t:t + 24].sum()),
                       discharge_lp_kWh=float(dl[t:t + 24].sum()),
                       charge_milp_kWh=float(c[t:t + 24].sum()),
                       discharge_milp_kWh=float(d[t:t + 24].sum()))
                  for t in range(0, 144, 24)]).to_csv(
        OUT / 'lp_charge_discharge_4hour.csv', index=False, encoding='utf-8-sig')
    lp_table[lp_table.slot.isin(SELECTED_SLOTS)][
        ['interval', 'grid_lp_kWh', 'grid_milp_kWh']].to_csv(
        OUT / 'lp_selected_intervals.csv', index=False, encoding='utf-8-sig')

    comparison = pd.DataFrame([
        dict(case='附件1（问题一已锁定）', load_kW_mean=float(load1.mean()),
             pv_kW_sum=float(pv1.sum()), net_demand_kWh=float(((load1 - pv1) / 6).sum()),
             grid_kWh=float(locked['grid_kWh']), cost_yuan=float(locked['cost_yuan']),
             charge_kWh=float(locked['charge_kWh']), discharge_kWh=float(locked['discharge_kWh']),
             unused_kWh=float(locked['unused_kWh'])),
        dict(case=f'附件2第{args.day}天（{date.date()}）', load_kW_mean=float(load2.mean()),
             pv_kW_sum=float(pv2.sum()), net_demand_kWh=float(((load2 - pv2) / 6).sum()),
             grid_kWh=float(q.sum()), cost_yuan=float(price @ q),
             charge_kWh=float(c.sum()), discharge_kWh=float(d.sum()), unused_kWh=float(w.sum())),
        dict(case=f'第{args.day}天·无储能（解析）', load_kW_mean=float(load2.mean()),
             pv_kW_sum=float(pv2.sum()), net_demand_kWh=float(((load2 - pv2) / 6).sum()),
             grid_kWh=no_storage['grid_kWh'], cost_yuan=no_storage['cost_yuan'],
             charge_kWh=0.0, discharge_kWh=0.0, unused_kWh=no_storage['unused_kWh']),
    ])
    comparison.to_csv(OUT / 'comparison.csv', index=False, encoding='utf-8-sig')

    energy_identity = float(q.sum() + pv2.sum() / 6 - load2.sum() / 6 - w.sum()
                            - ((1 - 0.9) * c.sum() + (1 / 0.9 - 1) * d.sum())
                            - (state[-1] - state[0]))
    validation = dict(
        model='locked problem-1 MILP from code/02_q1_baseline.py, unchanged',
        substituted=f'附件2 row for day {args.day} ({date.date()}): load and PV only',
        terminal_kWh=terminal,
        solver=summary['solver'], solver_message=summary['solver_message'],
        elapsed_seconds=float(summary['elapsed_seconds']), mip_gap=float(summary['mip_gap']),
        nominal_checks={k: float(v) for k, v in summary['checks'].items()},
        max_nominal_violation_kWh=float(max(summary['checks'].values())),
        lp_milp_cost_gap_yuan=float(abs(lp_summary['cost_yuan'] - summary['cost_yuan'])),
        energy_identity_residual_kWh=energy_identity,
        no_storage_analytic_gap_yuan=float(abs(no_storage['cost_yuan'] - (price @ no_grid))),
        terminal_state_kWh=float(state[0]), final_state_kWh=float(state[-1]),
        state_min_kWh=float(state.min()), state_max_kWh=float(state.max()),
        thresholds=dict(physical_kWh=1e-6, energy_identity_kWh=1e-6),
    )
    validation['physical_within_tolerance'] = bool(validation['max_nominal_violation_kWh'] <= 1e-6)
    validation['energy_identity_within_tolerance'] = bool(abs(energy_identity) <= 1e-6)
    validation['terminal_constraint_satisfied'] = bool(
        abs(state[0] - terminal) < 1e-9 and abs(state[-1] - terminal) < 1e-9)

    save = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        day_index=args.day, date=str(date.date()),
        substituted_inputs='附件2 小区负载 and 光伏发电实际功率 for the selected day',
        price_source='附件1 (144 points), unchanged from the locked problem-1 model',
        not_used=['附件3 光伏预报', '附件4 波动电价'],
        day_resolution='2025-01-01 counts as day 1; the attachment row is day - 1 with the header at row 0',
        parameters=dict(dt_hours=core.CONFIG['dt_hours'], eta_c=core.CONFIG['charge_efficiency'],
                        eta_d=core.CONFIG['discharge_efficiency'],
                        state_min_kWh=core.CONFIG['state_min_kWh'],
                        state_max_kWh=core.CONFIG['state_max_kWh'],
                        power_max_kW=core.CONFIG['power_max_kW'],
                        initial_kWh=terminal,
                        milp_relative_gap=core.CONFIG['milp_relative_gap'],
                        milp_time_limit_seconds=core.CONFIG['milp_time_limit_seconds']),
        source_sha256={'附件/附件1.xlsx': digest(ROOT / '附件/附件1.xlsx'),
                       '附件/附件2.xlsx': digest(ROOT / '附件/附件2.xlsx'),
                       'code/02_q1_baseline.py': digest(CORE),
                       'code/13_q1_day52_recompute.py': digest(Path(__file__).resolve())},
        results=dict(grid_kWh=float(q.sum()), cost_yuan=float(price @ q),
                     charge_kWh=float(c.sum()), discharge_kWh=float(d.sum()),
                     unused_kWh=float(w.sum()), loss_kWh=float(summary['loss_kWh']),
                     no_storage_cost_yuan=no_storage['cost_yuan'],
                     saving_vs_no_storage_yuan=float(no_storage['cost_yuan'] - summary['cost_yuan'])),
        selected_intervals=selected.to_dict(orient='records'),
        four_hour_blocks=groups.to_dict(orient='records'),
        lp_comparison=lp_block,
        diagnostic_terminal_scan=diagnostic,
        comparison_vs_attachment1=dict(
            cost_difference_yuan=float((price @ q) - float(locked['cost_yuan'])),
            grid_difference_kWh=float(q.sum() - float(locked['grid_kWh'])),
            net_demand_difference_kWh=float(((load2 - pv2) / 6).sum() - ((load1 - pv1) / 6).sum()),
            same_price_curve=True, same_solver=True, same_terminal_rule=True),
        validation=validation)
    (OUT / 'validation_summary.json').write_text(
        json.dumps(save, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'config.json').write_text(
        json.dumps(dict(day_index=args.day, date=str(date.date()), terminal_kWh=terminal,
                        parameters=save['parameters'], source_sha256=save['source_sha256']),
                   ensure_ascii=False, indent=2), encoding='utf-8')

    hours = np.arange(144) / 6
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, layout='constrained')
    axes[0].step(hours, load2, where='post', label='Load (attachment 2, day %d)' % args.day)
    axes[0].step(hours, pv2, where='post', label='PV actual power (attachment 2)')
    axes[0].set(ylabel='Power (kW)',
                title=f'Problem-1 model on attachment-2 day {args.day} ({date.date()}), '
                      f'price curve from attachment 1')
    axes[1].step(hours, q * 6, where='post', label='Grid purchase')
    axes[1].step(hours, c * 6, where='post', label='Charge (bus side)')
    axes[1].step(hours, -d * 6, where='post', label='Discharge (bus side, negative)')
    axes[1].set(ylabel='Bus power (kW)')
    axes[2].plot(np.arange(145) / 6, state, label='Stored energy', color='#1b6c9e')
    axes[2].axhline(core.CONFIG['state_min_kWh'], ls=':', color='gray')
    axes[2].axhline(core.CONFIG['state_max_kWh'], ls=':', color='gray')
    axes[2].set(ylabel='Stored energy (kWh)', xlabel='Time of day (h)',
                xlim=(0, 24), xticks=np.arange(0, 25, 2))
    price_ax = axes[2].twinx()
    price_ax.step(hours, price, where='post', color='#b85c00', alpha=.65, label='Price')
    price_ax.set_ylabel('Price (CNY/kWh)')
    price_ax.legend(loc='upper right')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(loc='upper left', ncol=3)
    fig.savefig(ROOT / 'figures/q1_day52.png', dpi=180)
    plt.close(fig)

    print(json.dumps(dict(date=str(date.date()), grid_kWh=float(q.sum()),
                          cost_yuan=float(price @ q), charge_kWh=float(c.sum()),
                          discharge_kWh=float(d.sum()), unused_kWh=float(w.sum()),
                          loss_kWh=float(summary['loss_kWh']),
                          no_storage_cost_yuan=no_storage['cost_yuan'],
                          state_min=float(state.min()), state_max=float(state.max()),
                          lp_cost_yuan=cost_lp, z_relaxation_cost_yuan=cost_lz,
                          lp_cost_gap_yuan=cost_lp - cost_milp,
                          lp_max_simultaneous_kWh=float(np.minimum(cl, dl).max()),
                          decycled_max_simultaneous_kWh=float(np.minimum(cl2, dl2).max()),
                          lp_vs_milp_max_grid_diff_kWh=lp_block['lp_vs_milp_slots']['grid_kWh'],
                          lp_vs_milp_slots_differing=lp_block['lp_vs_milp_slots']['slots_with_grid_difference'],
                          ordering_holds=lp_block['feasible_set_ordering_holds'],
                          costs_all_equal=lp_block['all_three_costs_equal']),
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
