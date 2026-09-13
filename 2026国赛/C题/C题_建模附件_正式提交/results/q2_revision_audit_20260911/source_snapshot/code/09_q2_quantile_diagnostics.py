"""Independent verification and diagnostics for the Q2 quantile-protection experiment.

Run (math_modeling env), after code/08_q2_quantile_experiment.py --mode full:
    E:/Anaconda/envs/math_modeling/python.exe code/09_q2_quantile_diagnostics.py

This script never re-solves a plan and never rewrites a policy result. It re-reads the saved
per-policy dispatch tables and re-derives every cost, energy balance and physical check from
those raw columns, so the numbers here are independent of the accumulators used while running.

Outputs
    results/q2_quantile_experiment/independent_accounting.csv
    results/q2_quantile_experiment/endpoint_valuation.csv
    results/q2_quantile_experiment/coverage_by_price.csv
    results/q2_quantile_experiment/diagnostic_summary.json
    figures/q2_quantile_experiment/*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'q2_quantile_experiment'
FIG = ROOT / 'figures' / 'q2_quantile_experiment'
DT = 1 / 6
ETA_C, ETA_D = 0.9, 0.9
STATE_MIN, STATE_MAX = 1200.0, 10800.0
SEGMENT_LIMIT = 5000 * DT
TOL = 1e-6
MAIN = ['A_base', 'B_fixed_q50', 'B_fixed_q65', 'B_fixed_q80', 'B_fixed_q90', 'B_adaptive']
SENS = ['S_adaptive_W14', 'S_adaptive_W56']
ALL = MAIN + SENS
COLORS = dict(A_base='#444444', B_fixed_q50='#7fb3d5', B_fixed_q65='#5b9bd5',
              B_fixed_q80='#2e75b6', B_fixed_q90='#1f4e79', B_adaptive='#c00000',
              S_adaptive_W14='#e8a33d', S_adaptive_W56='#7f6000')


def read_all():
    data = {}
    for sid in ALL:
        folder = OUT / sid
        if not (folder / 'dispatch.csv').exists():
            continue
        data[sid] = dict(
            dispatch=pd.read_csv(folder / 'dispatch.csv',
                                 dtype={'theta': str, 'theta_preset': str, 'fallback_reason': str}),
            daily=pd.read_csv(folder / 'daily_summary.csv',
                              dtype={'theta': str, 'theta_preset': str, 'fallback_reason': str}),
            monthly=pd.read_csv(folder / 'monthly_summary.csv'),
            events=pd.read_csv(folder / 'emergency_events.csv') if (folder / 'emergency_events.csv').stat().st_size > 1 else pd.DataFrame())
    return data


def independent_accounting(data):
    """Re-derive all costs, balances and physical checks from the saved raw columns."""
    rows = []
    detail = {}
    for sid, block in data.items():
        d = block['dispatch']
        daily = block['daily']
        price = d.price_yuan_kWh.to_numpy()
        planned = d.planned_kWh.to_numpy()
        charge = d.charge_kWh.to_numpy()
        discharge = d.discharge_kWh.to_numpy()
        emergency = d.emergency_kWh.to_numpy()
        unused = d.unused_kWh.to_numpy()
        s_start = d.state_start_kWh.to_numpy()
        s_end = d.state_end_kWh.to_numpy()
        load = d.load_kW.to_numpy()
        pv = d.pv_kW.to_numpy()

        planned_cost = float((planned * price).sum())
        emergency_cost = float((5.0 * emergency * price).sum())
        total_cost = planned_cost + emergency_cost
        balance = planned + pv * DT + discharge + emergency - load * DT - charge - unused
        recursion = (s_end - s_start) - (ETA_C * charge - discharge / ETA_D)
        losses = float((0.1 * charge + (1 / ETA_D - 1) * discharge).sum())
        energy_identity = float(planned.sum() + emergency.sum() + pv.sum() * DT - load.sum() * DT
                                - unused.sum() - losses - (s_end[-1] - s_start[0]))
        starts = pd.to_datetime(d.start_time)
        ends = pd.to_datetime(d.end_time)
        gaps = float(np.abs(((ends - starts).dt.total_seconds() - 600.0)).max())
        rows.append(dict(
            strategy_id=sid, slots=len(d), days=int(d.date.nunique()),
            planned_kWh=float(planned.sum()), emergency_kWh=float(emergency.sum()),
            unused_kWh=float(unused.sum()), charge_kWh=float(charge.sum()),
            discharge_kWh=float(discharge.sum()),
            planned_cost_yuan=planned_cost, emergency_cost_yuan=emergency_cost,
            total_cost_yuan=total_cost,
            emergency_slots=int((emergency > TOL).sum()),
            emergency_days=int(d.assign(active=emergency > TOL).groupby('date').active.any().sum()),
            emergency_events=int(len(block['events'])),
            initial_kWh=float(s_start[0]), final_kWh=float(s_end[-1]),
            max_bus_balance_kWh=float(np.abs(balance).max()),
            max_state_recursion_kWh=float(np.abs(recursion).max()),
            max_state_bound_violation_kWh=float(max(0.0, STATE_MIN - min(s_start.min(), s_end.min()),
                                                    max(s_start.max(), s_end.max()) - STATE_MAX)),
            max_power_violation_kWh=float(max(0.0, charge.max() - SEGMENT_LIMIT,
                                              discharge.max() - SEGMENT_LIMIT)),
            min_nonnegative_kWh=float(min(0.0, planned.min(), charge.min(), discharge.min(),
                                          emergency.min(), unused.min())),
            max_simultaneous_kWh=float(np.minimum(charge, discharge).max()),
            day_boundary_max_kWh=float(np.abs(daily.initial_kWh.to_numpy()[1:]
                                             - daily.final_kWh.to_numpy()[:-1]).max()),
            slot_gap_seconds=gaps,
            energy_identity_residual_kWh=energy_identity,
            cost_vs_daily_yuan=float(abs(total_cost - daily.total_cost_yuan.sum())),
            monthly_reconciliation_yuan=float(abs(total_cost - block['monthly'].total_cost_yuan.sum())),
            events_cost_reconciliation_yuan=float(abs(
                block['events'].emergency_cost_yuan.sum() - emergency_cost) if len(block['events'])
                else abs(emergency_cost)),
            net_mae_kW=float((((load - pv) - (d.load_forecast_kW - d.pv_forecast_kW)).abs()).mean()),
        ))
        detail[sid] = dict(forecast_columns_hash=hash(d.net_forecast_kWh.to_numpy().tobytes()))
    table = pd.DataFrame(rows).set_index('strategy_id')
    return table, detail


def endpoint_valuation(data, nu):
    """Compare the J^0 and J^nu candidate rankings inside every adaptive selection day."""
    log = pd.read_csv(OUT / 'selection_log.csv')
    log['valuation_rank'] = log.groupby(['strategy_id', 'date']).valuation_score_yuan.rank(method='first')
    changed = []
    for (sid, date), group in log.groupby(['strategy_id', 'date']):
        best_cash = group.loc[group.window_cost_yuan.idxmin(), 'candidate_theta']
        best_val = group.loc[group.valuation_score_yuan.idxmin(), 'candidate_theta']
        changed.append(dict(strategy_id=sid, date=date, cash_choice=best_cash,
                            valuation_choice=best_val, changed=int(best_cash != best_val)))
    frame = pd.DataFrame(changed)
    summary = frame.groupby('strategy_id').agg(selection_days=('changed', 'size'),
                                               changed_days=('changed', 'sum')).reset_index()
    summary['changed_pct'] = 100 * summary.changed_days / summary.selection_days
    return frame, summary


def coverage_by_price(data):
    """Does the protection amount cut high-price underestimation, and what does it cost?"""
    rows = []
    for sid, block in data.items():
        d = block['dispatch']
        price = d.price_yuan_kWh.to_numpy()
        actual = (d.load_kW - d.pv_kW).to_numpy() * DT
        origin = d.net_forecast_kWh.to_numpy()
        protected = d.protected_net_kWh.to_numpy()
        cutoff = np.quantile(price, 0.75)
        top = price >= cutoff
        rows.append(dict(
            strategy_id=sid,
            slots=len(d),
            origin_shortfall_rate=float((actual > origin + TOL).mean()),
            protected_shortfall_rate=float((actual > protected + TOL).mean()),
            origin_shortfall_rate_top_price=float((actual[top] > origin[top] + TOL).mean()),
            protected_shortfall_rate_top_price=float((actual[top] > protected[top] + TOL).mean()),
            mean_adjustment_kWh=float((protected - origin).mean()),
            max_adjustment_kWh=float((protected - origin).max()),
            min_adjustment_kWh=float((protected - origin).min()),
            planned_kWh=float(d.planned_kWh.sum()),
            unused_kWh=float(d.unused_kWh.sum()),
        ))
    return pd.DataFrame(rows).set_index('strategy_id')


def deltas_vs_a(accounting):
    """Level and percentage change of every headline metric against A_base."""
    a = accounting.loc['A_base']
    columns = ['planned_kWh', 'emergency_kWh', 'unused_kWh', 'charge_kWh', 'discharge_kWh',
               'planned_cost_yuan', 'emergency_cost_yuan', 'total_cost_yuan',
               'emergency_slots', 'emergency_days', 'emergency_events', 'final_kWh']
    out = accounting.copy()
    for column in columns:
        out['d_' + column] = accounting[column] - a[column]
    out['d_total_pct_of_A_total'] = 100 * out.d_total_cost_yuan / a.total_cost_yuan
    out['d_planned_cost_pct_of_A_planned'] = 100 * out.d_planned_cost_yuan / a.planned_cost_yuan
    out['d_emergency_cost_pct_of_A_emergency'] = 100 * out.d_emergency_cost_yuan / a.emergency_cost_yuan
    out['d_emergency_kWh_pct_of_A'] = 100 * out.d_emergency_kWh / a.emergency_kWh
    out['d_emergency_slots_pct_of_A'] = 100 * out.d_emergency_slots / a.emergency_slots
    return out


def daily_cost_extremes(data, reference='A_base'):
    """Most improved and most worsened dates, so the report can show both sides."""
    a = data[reference]['daily'].set_index('date')
    rows = []
    for sid, block in data.items():
        if sid == reference:
            continue
        d = block['daily'].set_index('date')
        delta = (d.total_cost_yuan - a.total_cost_yuan).reindex(d.index).dropna()
        ordered = delta.sort_values()
        rows.append(dict(
            strategy_id=sid,
            improved_days=int((delta < 0).sum()), worsened_days=int((delta > 0).sum()),
            net_delta_yuan=float(delta.sum()),
            top5_improved_dates='; '.join(f'{k} {v:,.0f}' for k, v in ordered.head(5).items()),
            top5_worsened_dates='; '.join(f'{k} {v:,.0f}' for k, v in ordered.tail(5).items()),
            max_single_day_improvement_yuan=float(-ordered.iloc[0]),
            max_single_day_worsening_yuan=float(ordered.iloc[-1]),
            best_day=ordered.index[0], worst_day=ordered.index[-1]))
    return pd.DataFrame(rows)


def figures(data, accounting, valuation_summary):
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'figure.dpi': 150, 'axes.grid': True, 'grid.alpha': .25,
                         'axes.axisbelow': True, 'font.size': 9})

    # 1 cost decomposition of all eight policies
    order = [s for s in ALL if s in accounting.index]
    fig, ax = plt.subplots(figsize=(11, 4.6), layout='constrained')
    x = np.arange(len(order))
    planned = accounting.loc[order, 'planned_cost_yuan'].to_numpy() / 1e6
    emergency = accounting.loc[order, 'emergency_cost_yuan'].to_numpy() / 1e6
    ax.bar(x, planned, .62, label='Planned purchase cost', color='#2e75b6')
    ax.bar(x, emergency, .62, bottom=planned, label='Emergency cost (5x)', color='#c00000')
    for i, sid in enumerate(order):
        ax.text(i, planned[i] + emergency[i] + .12, f'{planned[i] + emergency[i]:.2f}',
                ha='center', fontsize=8)
    ax.set(xticks=x, ylabel='Cost (million CNY)', title='Q2 realized cost decomposition, 2025-02..2025-12')
    ax.set_xticklabels([s.replace('_', '\n') for s in order], fontsize=8)
    ax.axvline(5.5, color='gray', ls='--', lw=.8)
    ax.text(2.5, ax.get_ylim()[1] * .92, 'main experiments', ha='center', fontsize=8, color='gray')
    ax.text(6.5, ax.get_ylim()[1] * .92, 'W sensitivity', ha='center', fontsize=8, color='gray')
    ax.legend(loc='lower left')
    fig.savefig(FIG / 'fig1_cost_decomposition.png'); plt.close(fig)

    # 2 monthly cost difference of B_adaptive versus A
    a_month = data['A_base']['monthly'].set_index('month')
    fig, ax = plt.subplots(figsize=(10, 4.2), layout='constrained')
    for sid in MAIN[1:] + SENS:
        if sid not in data:
            continue
        month = data[sid]['monthly'].set_index('month')
        delta = (month.total_cost_yuan - a_month.total_cost_yuan).reindex(month.index) / 1e3
        ax.plot(month.index, delta.to_numpy(), marker='o', ms=3.5, lw=1.4,
                color=COLORS[sid], label=sid)
    ax.axhline(0, color='black', lw=.8)
    ax.set(title='Monthly realized cost difference versus A_base (not a cumulative total)',
           xlabel='Month of 2025', ylabel='Difference (thousand CNY)')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(FIG / 'fig2_adaptive_monthly_delta.png'); plt.close(fig)

    # 3 selected quantile over time and its distribution
    adaptive = [s for s in ['B_adaptive', 'S_adaptive_W14', 'S_adaptive_W56'] if s in data]
    labels = ['base', '0.50', '0.65', '0.80', '0.90']
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), layout='constrained',
                             gridspec_kw=dict(height_ratios=[2, 1]))
    for sid in adaptive:
        daily = data[sid]['daily']
        codes = [labels.index(str(v)) for v in daily.theta]
        axes[0].plot(pd.to_datetime(daily.date), codes, marker='.', ms=3, lw=1,
                     color=COLORS[sid], label=sid)
    axes[0].set(yticks=range(len(labels)), yticklabels=labels,
                title='Daily selected protection rule (adaptive policies)',
                ylabel='selected theta')
    axes[0].legend(fontsize=8, ncol=3)
    width = .26
    for j, sid in enumerate(adaptive):
        counts = data[sid]['daily'].theta.astype(str).value_counts().reindex(labels).fillna(0)
        axes[1].bar(np.arange(len(labels)) + (j - 1) * width, counts.to_numpy(), width,
                    color=COLORS[sid], label=sid)
    axes[1].set(xticks=range(len(labels)), xticklabels=labels,
                ylabel='Days selected', xlabel='protection rule (base = no correction)')
    fig.savefig(FIG / 'fig3_theta_timeline.png'); plt.close(fig)

    # 4 four specified dates: net demand, plan, state and emergency
    for date in ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']:
        fig, axes = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True, layout='constrained')
        hours = np.arange(144) / 6
        for sid in ['A_base', 'B_adaptive']:
            if sid not in data:
                continue
            d = data[sid]['dispatch']
            d = d[d.date == date]
            if not len(d):
                continue
            actual = (d.load_kW - d.pv_kW).to_numpy() * DT
            axes[0].step(hours, actual, where='post', color=COLORS[sid], lw=1.6,
                         label=f'{sid}: actual net demand')
            axes[0].step(hours, d.net_forecast_kWh.to_numpy(), where='post', color=COLORS[sid],
                         ls=':', lw=1.0, label=f'{sid}: origin point forecast')
            axes[0].step(hours, d.protected_net_kWh.to_numpy(), where='post', color=COLORS[sid],
                         ls='--', lw=1.0, label=f'{sid}: protected demand')
            axes[1].step(hours, d.planned_kWh.to_numpy(), where='post', color=COLORS[sid], lw=1.4,
                         label=f'{sid}: frozen planned purchase')
            axes[1].step(hours, d.emergency_kWh.to_numpy(), where='post', color=COLORS[sid],
                         ls='--', lw=1.2, label=f'{sid}: emergency purchase')
            axes[2].plot(np.arange(145) / 6,
                         np.r_[d.state_start_kWh.iloc[0], d.state_end_kWh.to_numpy()],
                         color=COLORS[sid], lw=1.4, label=f'{sid}: stored energy')
        axes[0].set(ylabel='Energy per slot (kWh)',
                    title=f'Specified date {date}: net demand, plan, storage and emergency')
        axes[1].set(ylabel='Energy per slot (kWh)')
        axes[2].axhline(STATE_MIN, color='gray', ls=':', lw=.8)
        axes[2].axhline(STATE_MAX, color='gray', ls=':', lw=.8)
        axes[2].set(ylabel='Stored energy (kWh)', xlabel='Time of day (h)',
                    xticks=np.arange(0, 25, 2), xlim=(0, 24))
        for ax in axes:
            ax.legend(fontsize=7, ncol=2, loc='upper left')
        fig.savefig(FIG / f'fig4_selected_{date.replace("-", "")}.png'); plt.close(fig)

    # 5 emergency cost by hour, to see whether the evening peak improvement moves elsewhere
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), layout='constrained')
    for sid in order:
        d = data[sid]['dispatch']
        cost = d.assign(hour=d.slot // 6).groupby('hour').emergency_cost_yuan.sum()
        cost = cost.reindex(range(24), fill_value=0.0)
        axes[0].plot(cost.index, cost.to_numpy() / 1e3, marker='o', ms=3, lw=1.3,
                     color=COLORS[sid], label=sid)
        energy = d.assign(hour=d.slot // 6).groupby('hour').emergency_kWh.sum()
        axes[1].plot(energy.reindex(range(24), fill_value=0.0).index,
                     energy.reindex(range(24), fill_value=0.0).to_numpy(), marker='o', ms=3,
                     lw=1.3, color=COLORS[sid], label=sid)
    axes[0].set(ylabel='Emergency cost (thousand CNY)', title='Emergency cost and energy by hour of day')
    axes[1].set(ylabel='Emergency energy (kWh)', xlabel='Hour of day', xticks=range(24))
    for ax in axes:
        ax.legend(fontsize=7, ncol=4)
    fig.savefig(FIG / 'fig5_emergency_by_hour.png'); plt.close(fig)


def main():
    assert Path(sys.prefix).name == 'math_modeling', sys.prefix
    FIG.mkdir(parents=True, exist_ok=True)
    nu = float(np.median(pd.read_csv(OUT / 'A_base' / 'dispatch.csv').price_yuan_kWh.unique()) / 0.9)
    data = read_all()
    accounting, detail = independent_accounting(data)
    accounting['nu_yuan_per_kWh'] = nu
    accounting['adjusted_cost_yuan'] = accounting.total_cost_yuan - nu * (
        accounting.final_kWh - accounting.initial_kWh)
    accounting.to_csv(OUT / 'independent_accounting.csv', encoding='utf-8-sig')

    change_frame, valuation_summary = endpoint_valuation(data, nu)
    change_frame.to_csv(OUT / 'endpoint_valuation_days.csv', index=False, encoding='utf-8-sig')
    valuation_summary.to_csv(OUT / 'endpoint_valuation.csv', index=False, encoding='utf-8-sig')
    coverage = coverage_by_price(data)
    coverage.to_csv(OUT / 'coverage_by_price.csv', encoding='utf-8-sig')
    deltas = deltas_vs_a(accounting)
    deltas.to_csv(OUT / 'deltas_vs_A.csv', encoding='utf-8-sig')
    extremes = daily_cost_extremes(data)
    extremes.to_csv(OUT / 'daily_cost_extremes.csv', index=False, encoding='utf-8-sig')

    comparison = pd.read_csv(OUT / 'comparison.csv').set_index('strategy_id')
    merged = comparison.join(accounting[['total_cost_yuan', 'planned_cost_yuan', 'emergency_cost_yuan',
                                        'energy_identity_residual_kWh', 'max_bus_balance_kWh',
                                        'max_state_recursion_kWh', 'max_simultaneous_kWh']],
                            rsuffix='_independent')
    merged['cost_reproduction_abs_yuan'] = (
        merged.total_cost_yuan - merged.total_cost_yuan_independent).abs()

    forecast_identical = accounting.net_mae_kW.round(9).nunique() == 1
    checks = dict(
        physical_max_bus_balance_kWh=float(accounting.max_bus_balance_kWh.max()),
        physical_max_state_recursion_kWh=float(accounting.max_state_recursion_kWh.max()),
        physical_max_state_bound_violation_kWh=float(accounting.max_state_bound_violation_kWh.max()),
        physical_max_power_violation_kWh=float(accounting.max_power_violation_kWh.max()),
        physical_min_nonnegative_kWh=float(accounting.min_nonnegative_kWh.min()),
        physical_max_simultaneous_kWh=float(accounting.max_simultaneous_kWh.max()),
        physical_threshold_kWh=TOL,
        day_boundary_max_kWh=float(accounting.day_boundary_max_kWh.max()),
        slot_gap_max_seconds=float(accounting.slot_gap_seconds.max()),
        slots_all_48096=bool((accounting.slots == 48096).all()),
        days_all_334=bool((accounting.days == 334).all()),
        cost_reproduction_max_abs_yuan=float(merged.cost_reproduction_abs_yuan.max()),
        cost_reconciliation_max_yuan=float(accounting.cost_vs_daily_yuan.max()),
        monthly_reconciliation_max_yuan=float(accounting.monthly_reconciliation_yuan.max()),
        events_cost_reconciliation_max_yuan=float(accounting.events_cost_reconciliation_yuan.max()),
        energy_identity_max_residual_kWh=float(accounting.energy_identity_residual_kWh.abs().max()),
        origin_point_forecast_identical_across_policies=bool(forecast_identical),
        origin_net_mae_kW=float(accounting.net_mae_kW.iloc[0]),
    )
    checks['physical_all_within_tolerance'] = bool(max(
        checks['physical_max_bus_balance_kWh'], checks['physical_max_state_recursion_kWh'],
        checks['physical_max_state_bound_violation_kWh'], checks['physical_max_power_violation_kWh'],
        abs(checks['physical_min_nonnegative_kWh']), checks['physical_max_simultaneous_kWh']) <= TOL)
    checks['cost_all_within_tolerance'] = bool(max(
        checks['cost_reproduction_max_abs_yuan'], checks['cost_reconciliation_max_yuan'],
        checks['monthly_reconciliation_max_yuan'], checks['events_cost_reconciliation_max_yuan'],
        checks['energy_identity_max_residual_kWh']) <= 1e-4)

    figures(data, accounting, valuation_summary)

    diag = dict(
        nu_yuan_per_kWh=nu,
        evaluation='2025-02-01..2025-12-31, 334 days, 48,096 slots per policy',
        independent_checks=checks,
        independent_accounting=accounting.reset_index().to_dict(orient='records'),
        endpoint_valuation_summary=valuation_summary.to_dict(orient='records'),
        coverage_by_price=coverage.reset_index().to_dict(orient='records'),
        daily_cost_extremes=extremes.to_dict(orient='records'),
        figures=sorted(p.name for p in FIG.glob('*.png')),
    )
    (OUT / 'diagnostic_summary.json').write_text(
        json.dumps(diag, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    print(json.dumps(dict(independent_checks=checks,
                          endpoint_valuation=valuation_summary.to_dict(orient='records')),
                     ensure_ascii=False, indent=2), flush=True)
    print(accounting[['planned_cost_yuan', 'emergency_cost_yuan', 'total_cost_yuan',
                      'adjusted_cost_yuan', 'final_kWh']].to_string(), flush=True)


if __name__ == '__main__':
    main()
